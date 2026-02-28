"""LEGACY: Static graph-based synthesizer, kept for benchmarking only."""

from typing import Dict, Any
from ..base import BaseAgent
from ...core.state import MoEState


class SynthesizerAgent(BaseAgent):
    """Synthesizer agent that combines multiple expert responses.

    .. deprecated::
        Replaced by the sandbox's code-driven synthesis in the
        programmatic orchestration pipeline.
    """

    def __init__(self, llm_provider):
        super().__init__("Synthesizer", llm_provider)

    @staticmethod
    def _create_synthesis_prompt(query: str, expert_responses: Dict[str, str]) -> str:
        responses_text = "\n\n".join(
            f"=== {expert.upper()} EXPERT ===\n{response}"
            for expert, response in expert_responses.items()
        )
        return (
            f'You are an expert synthesizer.\n\nOriginal Query: "{query}"\n\n'
            f'Expert Responses:\n{responses_text}\n\n'
            f'Synthesize a coherent final response:'
        )
    
    def execute(self, state: MoEState) -> Dict[str, Any]:
        """Synthesize expert responses into final answer"""
        expert_responses = state['expert_responses']
        query = state['query']
        
        if len(expert_responses) == 1:
            final_answer = list(expert_responses.values())[0]
        else:
            prompt = self._create_synthesis_prompt(
                query=query,
                expert_responses=expert_responses
            )
            
            response = self.invoke_with_retry(prompt)
            final_answer = response.content
        
        return {
            "final_answer": final_answer,
            "reasoning_steps": [self._log_step(
                action=f"Synthesized {len(expert_responses)} expert responses",
                details={
                    "num_experts": len(expert_responses),
                    "experts": list(expert_responses.keys())
                }
            )]
        }