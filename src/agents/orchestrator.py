from typing import Dict, Any, List, Optional
import re
from .base import BaseAgent, AsyncBaseAgent
from .registry import registry
from ..core.state import MoEState
from ..llm.prompts import OrchestratorPrompts
from ..core.sandbox import CodeSandbox
from ..utils.code_analyzer import analyze_code
from ..utils.metrics import get_token_tracker
from ..utils.script_bank import ScriptBank


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that generates an async Python orchestration script"""
    
    def __init__(
        self,
        llm_provider,
        available_experts: Optional[List[str]] = None,
        script_bank: Optional[ScriptBank] = None,
    ):
        super().__init__("Orchestrator", llm_provider)
        self.available_experts = available_experts or list(registry.types)
        self.prompts = OrchestratorPrompts()
        self.script_bank = script_bank
    
    def execute(self, state: MoEState) -> Dict[str, Any]:
        query = state['query']
        code_failure = state.get('code_execution_error')
        previous_code = state.get('generated_code', '')
        descriptions = registry.descriptions()

        # Gather few-shot examples from script bank
        few_shot: List[tuple] = []
        if self.script_bank and not code_failure:
            similar = self.script_bank.find_similar(query, top_k=2)
            few_shot = [(r.query, r.code) for r in similar]

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
            )
        
        response = self.invoke_with_retry(prompt)
        generated_code = self._extract_code(response.content)
        
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
        script_bank: Optional[ScriptBank] = None,
    ):
        super().__init__("CodeExecutor")
        self.sandbox = CodeSandbox(timeout_seconds=timeout_seconds)
        self.script_bank = script_bank

    async def aexecute(self, state: MoEState) -> Dict[str, Any]:
        """Execute the generated script in the sandbox asynchronously"""
        code = state.get('generated_code', '')
        query = state.get('query', '')
        iterations = state.get('code_execution_iterations', 0)

        # Analyse the generated code's execution plan (best-effort)
        plan = analyze_code(code)
        
        try:
            execution_result = await self.sandbox.execute(code)

            # Record success in script bank
            if self.script_bank:
                self.script_bank.record(
                    query=query,
                    code=code,
                    experts_used=execution_result["selected_experts"],
                    success=True,
                )

            return {
                "final_answer": execution_result["result"],
                "selected_experts": execution_result["selected_experts"],
                "expert_responses": execution_result["expert_responses"],
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
            # Record failure in script bank
            if self.script_bank:
                self.script_bank.record(
                    query=query, code=code, experts_used=[], success=False,
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
