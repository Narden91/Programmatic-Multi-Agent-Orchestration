"""LEGACY: Static graph-based router, kept for benchmarking only."""

from typing import Dict, Any, List
from ..base import BaseAgent
from ...core.state import MoEState


class RouterAgent(BaseAgent):
    """Router agent that classifies queries and selects appropriate experts.

    .. deprecated::
        Replaced by OrchestratorAgent in the programmatic pipeline.
    """

    def __init__(self, llm_provider, available_experts: List[str]):
        """Initialize router agent"""
        super().__init__("Router", llm_provider)
        self.available_experts = available_experts

    @staticmethod
    def _create_routing_prompt(query: str, available_experts: List[str]) -> str:
        experts_desc = {
            "technical": "programming, technology, mathematics, sciences",
            "creative": "brainstorming, storytelling, creative content",
            "analytical": "data analysis, comparisons, logical decisions",
            "general": "general conversation, facts, basic information",
        }
        experts_list = "\n".join(
            f"- {e}: for {experts_desc.get(e, 'general queries')}" for e in available_experts
        )
        return (
            f'You are an intelligent router. Analyze this query and select '
            f'the most appropriate expert(s).\n\nQuery: "{query}"\n\n'
            f'Available experts:\n{experts_list}\n\nInstructions:\n'
            f'- Select ONE or MORE experts (up to 3)\n'
            f'- Respond ONLY with expert names separated by commas\n\nSelected experts:'
        )
    
    def execute(self, state: MoEState) -> Dict[str, Any]:
        """Analyze query and select appropriate experts"""
        query = state['query']
        
        prompt = self._create_routing_prompt(
            query=query,
            available_experts=self.available_experts
        )
        
        response = self.invoke_with_retry(prompt)
        selected_experts = self._parse_experts(response.content)
        
        valid_experts = [
            e for e in selected_experts 
            if e in self.available_experts
        ]
        
        if not valid_experts:
            valid_experts = ["general"]
        
        query_type = self._classify_query_type(valid_experts)
        
        return {
            "selected_experts": valid_experts,
            "query_type": query_type,
            "reasoning_steps": [self._log_step(
                action=f"Selected experts: {', '.join(valid_experts)}",
                details={
                    "query": query,
                    "available_experts": self.available_experts,
                    "selection_reasoning": response.content
                }
            )]
        }
    
    def _parse_experts(self, response: str) -> List[str]:
        """Parse expert names from LLM response using robust regex matching"""
        import re
        
        # Extract all occurrences of expert names from the response
        valid_expert_names = '|'.join(self.available_experts)
        pattern = rf'\b({valid_expert_names})\b'
        
        matches = re.findall(pattern, response.lower(), re.IGNORECASE)
        
        # Remove duplicates while preserving order
        seen = set()
        experts = []
        for match in matches:
            match_lower = match.lower()
            if match_lower not in seen:
                seen.add(match_lower)
                experts.append(match_lower)
        
        # Fallback: try simple comma split if regex found nothing
        if not experts:
            experts = [
                e.strip().lower() 
                for e in response.strip().split(',')
                if e.strip().lower() in self.available_experts
            ]
        
        return experts
    
    def _classify_query_type(self, experts: List[str]) -> str:
        """Classify query type based on selected experts"""
        if len(experts) > 1:
            return "multi-expert"
        return experts[0]