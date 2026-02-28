from typing import Dict, Any, List
import re
from .base import BaseAgent
from ..core.state import MoEState
from ..llm.prompts import OrchestratorPrompts
from ..core.sandbox import CodeSandbox


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that generates an async Python orchestration script"""
    
    def __init__(self, llm_provider, available_experts: List[str]):
        super().__init__("Orchestrator", llm_provider)
        self.available_experts = available_experts
        self.prompts = OrchestratorPrompts()
    
    def execute(self, state: MoEState) -> Dict[str, Any]:
        query = state['query']
        code_failure = state.get('code_execution_error')
        previous_code = state.get('generated_code', '')
        
        # Determine prompt based on whether it is a retry
        if code_failure and previous_code:
            prompt = self.prompts.create_retry_prompt(
                query=query,
                failed_code=previous_code,
                error=code_failure,
                available_experts=self.available_experts,
            )
        else:
            prompt = self.prompts.create_orchestration_prompt(
                query=query,
                available_experts=self.available_experts
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
                    "is_retry": bool(code_failure)
                }
            )]
        }
    
    def _extract_code(self, response: str) -> str:
        match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback if no markdown blocks
        return response.strip()


class CodeExecutionAgent(BaseAgent):
    """Agent that executes the generated orchestrator script"""

    def __init__(self, llm_provider, timeout_seconds: int = 60):
        super().__init__("CodeExecutor", llm_provider)
        self.sandbox = CodeSandbox(timeout_seconds=timeout_seconds)

    def execute(self, state: MoEState) -> Dict[str, Any]:
        raise NotImplementedError("CodeExecutor must be run async via aexecute()")

    async def aexecute(self, state: MoEState) -> Dict[str, Any]:
        """Execute the generated script in the sandbox asynchronously"""
        code = state.get('generated_code', '')
        iterations = state.get('code_execution_iterations', 0)
        
        try:
            execution_result = await self.sandbox.execute(code)
            return {
                "final_answer": execution_result["result"],
                "selected_experts": execution_result["selected_experts"],
                "expert_responses": execution_result["expert_responses"],
                "code_execution_error": "",
                "code_execution_iterations": iterations + 1,
                "reasoning_steps": [self._log_step(
                    action="Executed Code Successfully",
                    details={
                        "result_length": len(execution_result["result"]),
                        "experts_called": execution_result["selected_experts"],
                    }
                )]
            }
        except Exception as e:
            return {
                "code_execution_error": str(e),
                "code_execution_iterations": iterations + 1,
                "reasoning_steps": [self._log_step(
                    action="Code Execution Failed",
                    details={"error": str(e)}
                )]
            }
