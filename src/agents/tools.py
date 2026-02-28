import asyncio
from typing import Callable, Dict, Optional
from ..core.config import config, ExpertConfig, LLMConfig
from .experts.generic import GenericExpert
from .registry import registry
from ..llm.providers import LLMFactory
from ..utils.cache import ResponseCache
from ..utils.tracing import get_tracer, TraceEvent, TraceKind

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

    Works with **any** expert type registered in the :pydata:`registry`.
    """
    if expert_type not in registry:
        raise ValueError(
            f"Unknown expert type: {expert_type}. "
            f"Registered: {registry.types}"
        )

    async with _expert_lock:
        if expert_type not in _expert_instances:
            # LLM config: prefer explicit config, fall back to orchestrator_config
            expert_cfg = config.expert_configs.get(expert_type)

            # Per-expert provider override (multi-provider mixing)
            provider_type = (
                expert_cfg.provider_type
                if expert_cfg and expert_cfg.provider_type
                else config.get_provider_type()
            )
            api_key = config.get_api_key(provider_type)

            llm_config = expert_cfg.llm_config if expert_cfg else config.orchestrator_config
            confidence = (
                expert_cfg.confidence_threshold if expert_cfg
                else registry.get(expert_type).confidence_threshold
            )

            llm_provider = LLMFactory.create_provider(
                provider_type, api_key, llm_config,
            )

            _expert_instances[expert_type] = GenericExpert(
                expert_type=expert_type,
                llm_provider=llm_provider,
                confidence_threshold=confidence,
                cache=_get_response_cache(),
            )

    expert = _expert_instances[expert_type]

    await get_tracer().emit(TraceEvent(
        kind=TraceKind.EXPERT_CALL_START.value,
        agent=expert_type,
        data={"prompt_length": len(prompt)},
    ))

    result = await expert.aexecute(prompt)

    await get_tracer().emit(TraceEvent(
        kind=TraceKind.EXPERT_CALL_END.value,
        agent=expert_type,
        data={"result_length": len(result)},
    ))

    return result


# ---------------------------------------------------------------------------
# Dynamic tool-function generation
# ---------------------------------------------------------------------------

def _make_tool_function(expert_type: str):
    """Create a named async tool function for *expert_type*."""
    async def _tool(prompt: str) -> str:
        return await spawn_expert(expert_type, prompt)
    _tool.__name__ = f"query_{expert_type}_expert"
    _tool.__qualname__ = f"query_{expert_type}_expert"
    return _tool


def get_tool_functions() -> Dict[str, Callable]:
    """Return ``{expert_type: async_fn}`` for every registered expert."""
    return {etype: _make_tool_function(etype) for etype in registry.types}


# Convenience aliases for the four built-in types --------------------------

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


# ---------------------------------------------------------------------------
# Public registration helper
# ---------------------------------------------------------------------------

def register_expert(
    expert_type: str,
    description: str,
    prompt_template: str,
    system_prompt: Optional[str] = None,
    confidence_threshold: float = 0.75,
    llm_config: Optional[LLMConfig] = None,
) -> None:
    """Register a **new** expert type at runtime.

    Updates both the :pydata:`registry` and the global ``MoEConfig`` so that
    the sandbox can instantiate the expert automatically.
    """
    sys_prompt = system_prompt or description
    registry.register(
        expert_type=expert_type,
        description=description,
        system_prompt=sys_prompt,
        prompt_template=prompt_template,
        confidence_threshold=confidence_threshold,
    )
    if expert_type not in config.expert_configs:
        config.expert_configs[expert_type] = ExpertConfig(
            name=expert_type,
            description=description,
            llm_config=llm_config or LLMConfig.from_env(expert_type.upper()),
            system_prompt=sys_prompt,
            confidence_threshold=confidence_threshold,
        )
