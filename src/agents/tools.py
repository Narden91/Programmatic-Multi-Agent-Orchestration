import asyncio
from typing import Optional
from ..core.config import config
from .experts.generic import GenericExpert
from ..llm.providers import LLMFactory

async def spawn_expert(expert_type: str, prompt: str) -> str:
    """Spawns an expert of the given type to process a prompt."""
    if expert_type not in config.expert_configs:
        raise ValueError(f"Unknown expert type: {expert_type}. Available: {list(config.expert_configs.keys())}")
    
    expert_config = config.expert_configs[expert_type]
    provider_type = config.get_provider_type()
    api_key = getattr(config, f"{provider_type}_api_key")
    
    llm_provider = LLMFactory.create_provider(
        provider_type,
        api_key,
        expert_config.llm_config
    )
    
    # Instantiate the expert
    expert = GenericExpert(
        expert_type=expert_type,
        llm_provider=llm_provider,
        confidence_threshold=expert_config.confidence_threshold,
        cache=None  # Enable if needed
    )
    
    return await expert.aexecute(prompt)

async def query_technical_expert(prompt: str) -> str:
    """Query the technical expert agent programmatically"""
    return await spawn_expert('technical', prompt)

async def query_analytical_expert(prompt: str) -> str:
    """Query the analytical expert agent programmatically"""
    return await spawn_expert('analytical', prompt)

async def query_creative_expert(prompt: str) -> str:
    """Query the creative expert agent programmatically"""
    return await spawn_expert('creative', prompt)

async def query_general_expert(prompt: str) -> str:
    """Query the general expert agent programmatically"""
    return await spawn_expert('general', prompt)
