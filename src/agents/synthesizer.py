from typing import Dict, Any
from .base import BaseAgent
from ..core.state import MoEState
from ..llm.prompts import SynthesizerPrompts


class SynthesizerAgent(BaseAgent):
    """Synthesizer agent that combines multiple expert responses"""
    
    def __init__(self, llm_provider):
        super().__init__("Synthesizer", llm_provider)
        self.prompts = SynthesizerPrompts()
    
    def execute(self, state: MoEState) -> Dict[str, Any]:
        """Synthesize expert responses into final answer"""
        expert_responses = state['expert_responses']
        query = state['query']
        
        if len(expert_responses) == 1:
            final_answer = list(expert_responses.values())[0]
        else:
            prompt = self.prompts.create_synthesis_prompt(
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