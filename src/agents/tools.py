import asyncio
from typing import Optional
from ..core.config import config
from .experts.generic import GenericExpert
from ..llm.providers import LLMFactory
from ..utils.cache import ResponseCache

# ---------------------------------------------------------------------------
# Thread-safe singleton cache for expert instances
# ---------------------------------------------------------------------------
_expert_instances: dict[str, GenericExpert] = {}
_expert_lock = asyncio.Lock()

# Module-level response cache (shared across experts)
_response_cache: Optional[ResponseCache] = None


def _get_response_cache() -> Optional[ResponseCache]:
    """Lazily initialise the shared response cache based on config."""
    global _response_cache
    if _response_cache is None and config.enable_cache:
        _response_cache = ResponseCache(
            ttl_seconds=config.cache_ttl,
            max_size=config.cache_max_size,
        )
    return _response_cache


async def spawn_expert(expert_type: str, prompt: str) -> str:
    """Spawns an expert of the given type to process a prompt.

    Uses an ``asyncio.Lock`` to avoid race conditions when multiple
    ``asyncio.gather`` calls instantiate experts concurrently.
    """
    if expert_type not in config.expert_configs:
        raise ValueError(
            f"Unknown expert type: {expert_type}. "
            f"Available: {list(config.expert_configs.keys())}"
        )

    async with _expert_lock:
        if expert_type not in _expert_instances:
            expert_config = config.expert_configs[expert_type]
            provider_type = config.get_provider_type()
            api_key = getattr(config, f"{provider_type}_api_key")

            llm_provider = LLMFactory.create_provider(
                provider_type,
                api_key,
                expert_config.llm_config,
            )

            _expert_instances[expert_type] = GenericExpert(
                expert_type=expert_type,
                llm_provider=llm_provider,
                confidence_threshold=expert_config.confidence_threshold,
                cache=_get_response_cache(),
            )

    expert = _expert_instances[expert_type]
    return await expert.aexecute(prompt)


async def query_technical_expert(prompt: str) -> str:
    """Query the technical expert agent programmatically."""
    return await spawn_expert("technical", prompt)


async def query_analytical_expert(prompt: str) -> str:
    """Query the analytical expert agent programmatically."""
    return await spawn_expert("analytical", prompt)


async def query_creative_expert(prompt: str) -> str:
    """Query the creative expert agent programmatically."""
    return await spawn_expert("creative", prompt)


async def query_general_expert(prompt: str) -> str:
    """Query the general expert agent programmatically."""
    return await spawn_expert("general", prompt)
