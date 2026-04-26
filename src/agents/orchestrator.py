from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import re
from .base import BaseAgent, AsyncBaseAgent
from .registry import registry
from ..core.state import MoEState
from ..llm.prompts import OrchestratorPrompts
from ..utils.code_analyzer import analyze_code
from ..utils.metrics import get_token_tracker
from ..core.registry import OrchestrationRegistry
from ..core.scoring import ScriptScorer
from ..utils.tracing import get_tracer, TraceEvent, TraceKind
from ..core.sandbox import CodeSandbox, SandboxPolicy


@dataclass
class _CandidateScript:
    code: str
    score: float
    details: Dict[str, Any]


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that generates an async Python orchestration script"""
    
    def __init__(
        self,
        llm_provider,
        available_experts: Optional[List[str]] = None,
        candidate_count: int = 1,
        script_bank: Optional[Any] = None, # kept for backward compat in init args
    ):
        super().__init__("Orchestrator", llm_provider)
        self.available_experts = available_experts or list(registry.types)
        self.candidate_count = max(1, int(candidate_count))
        self.prompts = OrchestratorPrompts()
        self.orchestration_registry = OrchestrationRegistry()
    
    async def execute(self, state: MoEState) -> Dict[str, Any]:
        query = state['query']
        code_failure = state.get('code_execution_error')
        previous_code = state.get('generated_code', '')
        descriptions = registry.descriptions()
        conversation_context = state.get('conversation_context', '')
        similar_rows: List[Dict[str, Any]] = []

        # Gather few-shot examples from registry
        few_shot: List[tuple] = []
        if not code_failure:
            similar_rows = self.orchestration_registry.search(query, top_k=2)
            few_shot = [(r["task_description"], r["script_content"]) for r in similar_rows]

        # Determine prompt based on whether it is a retry
        if code_failure and previous_code:
            prompt = self.prompts.create_retry_prompt(
                query=query,
                failed_code=previous_code,
                error=code_failure,
                available_experts=self.available_experts,
                expert_descriptions=descriptions,
            )
        else:
            prompt = self.prompts.create_orchestration_prompt(
                query=query,
                available_experts=self.available_experts,
                expert_descriptions=descriptions,
                few_shot_examples=few_shot or None,
                conversation_context=conversation_context,
            )

        # Trace: orchestrator start / retry
        _kind = TraceKind.ORCHESTRATOR_RETRY if code_failure else TraceKind.ORCHESTRATOR_START
        await get_tracer().emit(TraceEvent(
            kind=_kind.value, agent=self.name,
            data={"query_len": len(query), "is_retry": bool(code_failure), "few_shot": len(few_shot)},
        ))

        selection_details: Dict[str, Any]
        if not code_failure and self.candidate_count > 1:
            generated_code, selection_details = await self._generate_with_search(
                prompt=prompt,
                similar_rows=similar_rows,
            )
            await get_tracer().emit(TraceEvent(
                kind=TraceKind.CUSTOM.value,
                agent=self.name,
                data={
                    "phase": "candidate_selection",
                    "selection_mode": selection_details.get("selection_mode", "heuristic"),
                    "candidate_count": selection_details.get("candidate_count", 1),
                    "selected_score": selection_details.get("selected_score", 0.0),
                },
            ))
        else:
            response = await self.ainvoke_with_retry(prompt)
            generated_code = self._extract_code(response.content)
            selection_details = {
                "selection_mode": "single",
                "candidate_count": 1,
                "selected_score": 0.0,
            }

        await get_tracer().emit(TraceEvent(
            kind=TraceKind.ORCHESTRATOR_CODE_GENERATED.value, agent=self.name,
            data={"code_length": len(generated_code)},
        ))

        return {
            "generated_code": generated_code,
            "code_execution_error": "",
            "reasoning_steps": [self._log_step(
                action="Generated orchestration code",
                details={
                    "query": query,
                    "code_length": len(generated_code),
                    "is_retry": bool(code_failure),
                    "few_shot_count": len(few_shot),
                    "selection": selection_details,
                }
            )]
        }

    async def _generate_with_search(
        self,
        prompt: str,
        similar_rows: List[Dict[str, Any]],
    ) -> tuple[str, Dict[str, Any]]:
        """Generate multiple candidate scripts and select the best one heuristically."""
        raw_candidates: List[str] = []
        for _ in range(self.candidate_count):
            response = await self.ainvoke_with_retry(prompt)
            raw_candidates.append(self._extract_code(response.content))

        unique_candidates = list(dict.fromkeys(c for c in raw_candidates if c.strip()))
        if not unique_candidates:
            return "", {
                "selection_mode": "heuristic",
                "candidate_count": 0,
                "selected_score": 0.0,
                "top_scores": [],
            }

        scored: List[_CandidateScript] = []
        for code in unique_candidates:
            score, details = self._score_candidate(code, similar_rows)
            scored.append(_CandidateScript(code=code, score=score, details=details))

        scored.sort(key=lambda c: c.score, reverse=True)
        selected = scored[0]

        return selected.code, {
            "selection_mode": "heuristic",
            "candidate_count": len(unique_candidates),
            "selected_score": round(selected.score, 4),
            "top_scores": [round(c.score, 4) for c in scored[:3]],
            "selected_features": selected.details,
        }

    def _score_candidate(
        self,
        code: str,
        similar_rows: List[Dict[str, Any]],
    ) -> tuple[float, Dict[str, Any]]:
        """Score a candidate orchestration script using lightweight static heuristics."""
        plan = analyze_code(code)
        call_count = len(plan.calls)
        expert_count = len(plan.experts_used)
        parallel_groups = plan.gather_groups
        code_len = len(code)

        score = 0.0

        if "async def orchestrate" in code:
            score += 1.0
        else:
            score -= 3.0

        if call_count == 0:
            score -= 1.2
        else:
            score += min(call_count, 4) * 0.30
            if call_count > 8:
                score -= (call_count - 8) * 0.25

        if parallel_groups > 0:
            score += min(parallel_groups, 2) * 0.25
            if parallel_groups > 3:
                score -= (parallel_groups - 3) * 0.20

        if plan.has_parallel and plan.has_sequential:
            score += 0.15

        score += min(expert_count, 3) * 0.20

        if code_len > 12_000:
            score -= (code_len - 12_000) / 4_000

        prior_boost = self._registry_prior(plan.experts_used, similar_rows)
        score += prior_boost

        return score, {
            "call_count": call_count,
            "expert_count": expert_count,
            "parallel_groups": parallel_groups,
            "code_length": code_len,
            "registry_prior": round(prior_boost, 4),
        }

    def _registry_prior(
        self,
        experts_used: List[str],
        similar_rows: List[Dict[str, Any]],
    ) -> float:
        if not similar_rows:
            return 0.0

        best_similarity = max(float(r.get("similarity", 0.0)) for r in similar_rows)
        avg_quality = sum(
            max(float(r.get("score", 0.0)), 0.0) for r in similar_rows
        ) / len(similar_rows)

        overlap_scores: List[float] = []
        target = set(experts_used)
        if target:
            for row in similar_rows:
                src = row.get("script_content", "")
                if not src:
                    continue
                past_experts = set(analyze_code(src).experts_used)
                if not past_experts:
                    continue
                union = target | past_experts
                overlap_scores.append(len(target & past_experts) / len(union))

        expert_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0

        return (0.35 * best_similarity) + (0.35 * avg_quality) + (0.30 * expert_overlap)
    
    def _extract_code(self, response: str) -> str:
        match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback if no markdown blocks
        return response.strip()


class CodeExecutionAgent(AsyncBaseAgent):
    """Agent that executes the generated orchestrator script.

    This is an async-only agent — it inherits from ``AsyncBaseAgent``
    and has no synchronous ``execute()`` method.
    """

    def __init__(
        self,
        timeout_seconds: int = 60,
        isolate_process: bool = True,
        sandbox_policy: Optional[SandboxPolicy] = None,
        script_bank: Optional[Any] = None,
    ):
        super().__init__("CodeExecutor")
        self.sandbox = CodeSandbox(
            timeout_seconds=timeout_seconds,
            isolate_process=isolate_process,
            policy=sandbox_policy,
        )
        self.orchestration_registry = OrchestrationRegistry()
        self.scorer = ScriptScorer()

    def _summarize_trace(self, trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        agent_spans = [item for item in trace if item.get("type") == "agent"]
        atom_counts = [
            int((item.get("outputs") or {}).get("atom_count", 0))
            for item in agent_spans
        ]
        response_formats = sorted({
            str((item.get("outputs") or {}).get("response_format", "plain_text"))
            for item in agent_spans
        })
        return {
            "agent_span_count": len(agent_spans),
            "atom_count_total": sum(atom_counts),
            "max_atom_count": max(atom_counts) if atom_counts else 0,
            "response_formats": response_formats,
        }

    def _build_registry_metadata(
        self,
        *,
        plan: Any,
        trace: Optional[List[Dict[str, Any]]] = None,
        selected_experts: Optional[List[str]] = None,
        error: str = "",
    ) -> Dict[str, Any]:
        return {
            "selected_experts": list(selected_experts or []),
            "execution_plan": plan.to_dict(),
            "trace_summary": self._summarize_trace(trace or []),
            "outcome": "error" if error else "success",
            "error": error,
        }

    async def aexecute(self, state: MoEState) -> Dict[str, Any]:
        """Execute the generated script in the sandbox asynchronously"""
        code = state.get('generated_code', '')
        query = state.get('query', '')
        iterations = state.get('code_execution_iterations', 0)

        # Analyse the generated code's execution plan (best-effort)
        plan = analyze_code(code)

        await get_tracer().emit(TraceEvent(
            kind=TraceKind.SANDBOX_START.value, agent=self.name,
            data={"code_length": len(code), "iteration": iterations},
        ))

        try:
            execution_result = await self.sandbox.execute(code)

            # Start returning state early so scorer can use it
            temp_state = {
                "final_answer": execution_result["result"],
                "trace_dna": execution_result.get("trace", [])
            }
            
            # Score it async
            score = await self.scorer.score_execution(query, temp_state)

            # Record success in registry 
            self.orchestration_registry.store_script(
                task_description=query,
                script_content=code,
                score=score,
                metadata=self._build_registry_metadata(
                    plan=plan,
                    trace=execution_result.get("trace", []),
                    selected_experts=execution_result["selected_experts"],
                ),
            )

            await get_tracer().emit(TraceEvent(
                kind=TraceKind.SANDBOX_SUCCESS.value, agent=self.name,
                data={"experts": execution_result["selected_experts"]},
            ))

            return {
                "final_answer": execution_result["result"],
                "selected_experts": execution_result["selected_experts"],
                "expert_responses": execution_result["expert_responses"],
                "trace_dna": execution_result.get("trace", []),
                "sandbox_output": execution_result.get("sandbox_output", ""),
                "metadata": {
                    **state.get("metadata", {}),
                    "sandbox_security": execution_result.get("security", {}),
                },
                "code_execution_error": "",
                "code_execution_iterations": iterations + 1,
                "execution_plan": plan.to_dict(),
                "token_usage": get_token_tracker().summary(),
                "reasoning_steps": [self._log_step(
                    action="Executed Code Successfully",
                    details={
                        "result_length": len(execution_result["result"]),
                        "experts_called": execution_result["selected_experts"],
                        "parallel_groups": plan.gather_groups,
                    }
                )]
            }
        except Exception as e:
            await get_tracer().emit(TraceEvent(
                kind=TraceKind.SANDBOX_ERROR.value, agent=self.name,
                data={"error": str(e)},
            ))

            # Record failure in registry (optional, maybe score=0.0)
            self.orchestration_registry.store_script(
                task_description=query,
                script_content=code,
                score=0.0,
                metadata=self._build_registry_metadata(
                    plan=plan,
                    selected_experts=[],
                    error=str(e),
                ),
            )

            return {
                "code_execution_error": str(e),
                "code_execution_iterations": iterations + 1,
                "execution_plan": plan.to_dict(),
                "metadata": {
                    **state.get("metadata", {}),
                    "sandbox_security": {
                        "error": str(e),
                    },
                },
                "token_usage": get_token_tracker().summary(),
                "reasoning_steps": [self._log_step(
                    action="Code Execution Failed",
                    details={"error": str(e)}
                )]
            }
