from typing import Dict, Any, Optional
from ..base import BaseAgent
from ...core.state import MoEState
from ...llm.prompts import ExpertPrompts
from ...utils.cache import ResponseCache


class GenericExpert(BaseAgent):
    """Generic expert that can handle any expert type"""
    
    EXPERT_PROMPTS = {
        'technical': ExpertPrompts.create_technical_prompt,
        'creative': ExpertPrompts.create_creative_prompt,
        'analytical': ExpertPrompts.create_analytical_prompt,
        'general': ExpertPrompts.create_general_prompt
    }
    
    EXPERT_NAMES = {
        'technical': 'Technical Expert',
        'creative': 'Creative Expert',
        'analytical': 'Analytical Expert',
        'general': 'General Expert'
    }
    
    def __init__(
        self, 
        expert_type: str, 
        llm_provider, 
        confidence_threshold: float = 0.75,
        cache: Optional[ResponseCache] = None
    ):
        """
        Initialize generic expert.
        
        Args:
            expert_type: Type of expert ('technical', 'creative', 'analytical', 'general')
            llm_provider: LLM provider instance
            confidence_threshold: Confidence score for this expert
            cache: Optional ResponseCache instance for caching responses
        """
        if expert_type not in self.EXPERT_PROMPTS:
            raise ValueError(f"Invalid expert type: {expert_type}. Must be one of {list(self.EXPERT_PROMPTS.keys())}")
        
        self.expert_type = expert_type
        expert_name = self.EXPERT_NAMES[expert_type]
        super().__init__(expert_name, llm_provider)
        self.confidence_threshold = confidence_threshold
        self.cache = cache
    
    def execute(self, state: MoEState) -> Dict[str, Any]:
        """Generate response to query"""
        if self._should_skip(state):
            return {}
        
        query = state['query']
        
        # Check cache first if available
        if self.cache:
            cached_response = self.cache.get(query, self.expert_type)
            if cached_response:
                return {
                    "expert_responses": {self.expert_type: cached_response},
                    "confidence_scores": {self.expert_type: self.confidence_threshold},
                    "reasoning_steps": [self._log_step(
                        action=f"Retrieved cached {self.expert_type} response",
                        details={
                            "query": query,
                            "cache_hit": True
                        }
                    )]
                }
        
        # Get the appropriate prompt creator for this expert type
        prompt_creator = self.EXPERT_PROMPTS[self.expert_type]
        prompt = prompt_creator(query)
        
        # Invoke LLM with retry logic
        response = self.invoke_with_retry(prompt)
        response_content = response.content
        
        # Cache the response if cache is available
        if self.cache:
            self.cache.set(query, self.expert_type, response_content)
        
        return {
            "expert_responses": {self.expert_type: response_content},
            "confidence_scores": {self.expert_type: self.confidence_threshold},
            "reasoning_steps": [self._log_step(
                action=f"Generated {self.expert_type} response",
                details={
                    "query": query,
                    "response_length": len(response_content),
                    "cache_hit": False
                }
            )]
        }
    
    def _should_skip(self, state: MoEState) -> bool:
        """Check if this expert should skip execution"""
        return self.expert_type not in state.get('selected_experts', [])
