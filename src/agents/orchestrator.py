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
from ..core.sandbox import CodeSandbox


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that generates an async Python orchestration script"""
    
    def __init__(
        self,
        llm_provider,
        available_experts: Optional[List[str]] = None,
        script_bank: Optional[Any] = None, # kept for backward compat in init args
    ):
        super().__init__("Orchestrator", llm_provider)
        self.available_experts = available_experts or list(registry.types)
        self.prompts = OrchestratorPrompts()
        self.orchestration_registry = OrchestrationRegistry()
    
    def execute(self, state: MoEState) -> Dict[str, Any]:
        query = state['query']
        code_failure = state.get('code_execution_error')
        previous_code = state.get('generated_code', '')
        descriptions = registry.descriptions()
        conversation_context = state.get('conversation_context', '')

        # Gather few-shot examples from registry
        few_shot: List[tuple] = []
        if not code_failure:
            similar = self.orchestration_registry.search(query, top_k=2)
            few_shot = [(r["task_description"], r["script_content"]) for r in similar]

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
        get_tracer().emit_sync(TraceEvent(
            kind=_kind.value, agent=self.name,
            data={"query_len": len(query), "is_retry": bool(code_failure), "few_shot": len(few_shot)},
        ))

        response = self.invoke_with_retry(prompt)
        generated_code = self._extract_code(response.content)

        get_tracer().emit_sync(TraceEvent(
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
                }
            )]
        }
    
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
        script_bank: Optional[Any] = None,
    ):
        super().__init__("CodeExecutor")
        self.sandbox = CodeSandbox(timeout_seconds=timeout_seconds)
        self.orchestration_registry = OrchestrationRegistry()
        self.scorer = ScriptScorer()

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
                score=score
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
                score=0.0
            )

            return {
                "code_execution_error": str(e),
                "code_execution_iterations": iterations + 1,
                "execution_plan": plan.to_dict(),
                "token_usage": get_token_tracker().summary(),
                "reasoning_steps": [self._log_step(
                    action="Code Execution Failed",
                    details={"error": str(e)}
                )]
            }
