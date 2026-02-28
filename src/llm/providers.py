"""LLM provider abstraction supporting Groq, OpenAI, and Anthropic.

Each provider wraps the corresponding ``langchain-*`` chat model.  OpenAI and
Anthropic are **optional** — import errors are deferred until someone actually
tries to instantiate the provider, so ``langchain-openai`` / ``langchain-anthropic``
need only be installed when used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from ..core.config import LLMConfig


# ======================================================================
# Abstract base
# ======================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    model_name: str = "unknown"
    provider_name: str = "unknown"

    @abstractmethod
    def invoke(self, prompt: str) -> Any:
        """Invoke the LLM with a prompt"""
        pass

    @abstractmethod
    async def ainvoke(self, prompt: str) -> Any:
        """Asynchronously invoke the LLM with a prompt"""
        pass


# ======================================================================
# Concrete providers
# ======================================================================

class GroqProvider(LLMProvider):
    """Groq LLM provider (fast open-source inference)."""

    provider_name = "groq"

    def __init__(self, api_key: str, config: LLMConfig):
        from langchain_groq import ChatGroq

        self.model_name = config.model_name
        self.llm = ChatGroq(
            api_key=api_key,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    def invoke(self, prompt: str) -> Any:
        return self.llm.invoke(prompt)

    async def ainvoke(self, prompt: str) -> Any:
        return await self.llm.ainvoke(prompt)


class OpenAIProvider(LLMProvider):
    """OpenAI / Azure OpenAI provider (GPT-4o, GPT-4-turbo, etc.)."""

    provider_name = "openai"

    def __init__(self, api_key: str, config: LLMConfig):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for the OpenAI provider. "
                "Install it with: pip install langchain-openai"
            ) from exc

        self.model_name = config.model_name
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    def invoke(self, prompt: str) -> Any:
        return self.llm.invoke(prompt)

    async def ainvoke(self, prompt: str) -> Any:
        return await self.llm.ainvoke(prompt)


class AnthropicProvider(LLMProvider):
    """Anthropic provider (Claude 3.5 Sonnet, Opus, Haiku, etc.)."""

    provider_name = "anthropic"

    def __init__(self, api_key: str, config: LLMConfig):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "langchain-anthropic is required for the Anthropic provider. "
                "Install it with: pip install langchain-anthropic"
            ) from exc

        self.model_name = config.model_name
        self.llm = ChatAnthropic(
            api_key=api_key,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    def invoke(self, prompt: str) -> Any:
        return self.llm.invoke(prompt)

    async def ainvoke(self, prompt: str) -> Any:
        return await self.llm.ainvoke(prompt)


# ======================================================================
# Factory
# ======================================================================

class LLMFactory:
    """Factory for creating LLM providers.

    Supports built-in providers (``groq``, ``openai``, ``anthropic``) and
    allows runtime registration of custom providers via
    :meth:`register_provider`.
    """

    _providers: Dict[str, Type[LLMProvider]] = {
        "groq": GroqProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    @classmethod
    def register_provider(
        cls, name: str, provider_class: Type[LLMProvider]
    ) -> None:
        """Register a custom LLM provider class."""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def available_providers(cls) -> list[str]:
        """Return names of all registered provider types."""
        return list(cls._providers)

    @classmethod
    def create_provider(
        cls,
        provider_type: str,
        api_key: str,
        config: LLMConfig,
    ) -> LLMProvider:
        """Create an LLM provider instance."""
        provider_class = cls._providers.get(provider_type.lower())
        if not provider_class:
            raise ValueError(
                f"Unknown provider type: {provider_type}. "
                f"Available: {cls.available_providers()}"
            )
        return provider_class(api_key, config)