from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time
from langchain_core.messages import SystemMessage, HumanMessage

from .config import config, ExpertConfig
from ..llm.providers import LLMFactory
from ..utils.metrics import get_token_tracker

@dataclass
class AgentResult:
    """Standardized result returned by a micro-agent."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    duration_ms: int = 0

async def query_agent(agent_type: str, prompt: str, context_ids: Optional[List[str]] = None) -> AgentResult:
    """
    Spawns a transient micro-agent of the requested type, processes the prompt, and returns the result.
    
    Args:
        agent_type: The type of expert to spawn (e.g., 'technical', 'analytical', 'creative', 'general', 'critical-thinker')
        prompt: The task instruction or query.
        context_ids: Optional list of memory context IDs to retrieve and inject.
    """
    start_time = time.time()
    
    # 1. Retrieve config
    if agent_type not in config.expert_configs:
        # Fallback to general if unknown, or raise. We'll raise to keep it strict.
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(config.expert_configs.keys())}")
        
    expert_config = config.expert_configs[agent_type]
    
    # Check if context_ids are provided (memory integration to be added)
    # If we have context, we would look it up here and prepend it to the prompt.
    # For now we'll just mock the context retrieval part since Memory class isn't fully injected yet.
    # In sandbox.py, memory_search would have been used by the orchestrator script to get facts.
    
    full_prompt = prompt
    if context_ids:
        # This assumes the sandbox has already done the `memory_search`. If context_ids is passed,
        # it might just be the actual texts, or the orchestrator should pass the searched texts directly.
        # But per idea_v2, the orchestrator might pass raw context texts or we fetch them here.
        pass

    # 2. Instantiate LLM
    provider_type = expert_config.provider_type or config.get_provider_type()
    api_key = config.get_api_key(provider_type)
    
    llm_provider = LLMFactory.create_provider(
        provider_type=provider_type,
        api_key=api_key,
        config=expert_config.llm_config,
    )
    
    # 3. Prepare messages
    messages = [
        SystemMessage(content=expert_config.system_prompt),
        HumanMessage(content=full_prompt)
    ]
    
    # 4. Invoke LLM
    # Note: LLMProvider ainvoke expects a string or list of messages depending on the underlying langchain implementation.
    # We should pass a list of messages if supported, or a concatenated string.
    # Langchain models `invoke` and `ainvoke` support lists of messages.
    try:
        response = await llm_provider.ainvoke(messages)
        # response is an AIMessage in langchain
        response_text = response.content
        
        # We can extract token counts if the model returns them in response.response_metadata
        token_count = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_count = response.usage_metadata.get("total_tokens", 0)
        elif hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
            token_count = response.response_metadata["token_usage"].get("total_tokens", 0)
            
        tracker = get_token_tracker()
        tracker.record_from_response(f"agent_{agent_type}", expert_config.llm_config.model_name, response)
        
    except Exception as e:
        raise RuntimeError(f"Agent '{agent_type}' execution failed: {str(e)}")
        
    duration_ms = int((time.time() - start_time) * 1000)
    
    return AgentResult(
        text=response_text,
        metadata={"agent_type": agent_type, "model": expert_config.llm_config.model_name},
        token_count=token_count,
        duration_ms=duration_ms
    )
