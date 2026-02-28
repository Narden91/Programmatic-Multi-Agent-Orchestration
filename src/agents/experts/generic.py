from typing import Dict, Any, Optional
from ..base import BaseAgent
from ...core.state import MoEState
from ..registry import registry
from ...utils.cache import ResponseCache


class GenericExpert(BaseAgent):
    """Generic expert that can handle any registered expert type"""

    def __init__(
        self,
        expert_type: str,
        llm_provider,
        confidence_threshold: float = 0.75,
        cache: Optional[ResponseCache] = None,
    ):
        if expert_type not in registry:
            raise ValueError(
                f"Invalid expert type: {expert_type}. "
                f"Must be one of {registry.types}"
            )

        self.expert_type = expert_type
        expert_name = f"{expert_type.capitalize()} Expert"
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
        
        # Get the appropriate prompt from the registry
        prompt = registry.create_prompt(self.expert_type, query)
        
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
    
    async def aexecute(self, query: str) -> str:
        """Asynchronously generate response to isolated query for tool execution"""
        # Check cache first if available
        if self.cache:
            cached_response = self.cache.get(query, self.expert_type)
            if cached_response:
                return cached_response
        
        # Get the appropriate prompt from the registry
        prompt = registry.create_prompt(self.expert_type, query)

        # Invoke LLM with retry logic
        response = await self.ainvoke_with_retry(prompt)
        response_content = response.content
        
        # Cache the response if cache is available
        if self.cache:
            self.cache.set(query, self.expert_type, response_content)
        
        return response_content
    
    def _should_skip(self, state: MoEState) -> bool:
        """Check if this expert should skip execution"""
        return self.expert_type not in state.get('selected_experts', [])
