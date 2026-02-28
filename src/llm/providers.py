from abc import ABC, abstractmethod
from typing import Any
from langchain_groq import ChatGroq
from ..core.config import LLMConfig


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def invoke(self, prompt: str) -> Any:
        """Invoke the LLM with a prompt"""
        pass

    @abstractmethod
    async def ainvoke(self, prompt: str) -> Any:
        """Asynchronously invoke the LLM with a prompt"""
        pass


class GroqProvider(LLMProvider):
    """Groq LLM provider implementation"""
    
    def __init__(self, api_key: str, config: LLMConfig):
        """Initialize Groq provider"""
        self.llm = ChatGroq(
            api_key=api_key,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    
    def invoke(self, prompt: str) -> Any:
        """Invoke Groq LLM"""
        return self.llm.invoke(prompt)

    async def ainvoke(self, prompt: str) -> Any:
        """Asynchronously invoke Groq LLM"""
        return await self.llm.ainvoke(prompt)


class LLMFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(
        provider_type: str,
        api_key: str,
        config: LLMConfig
    ) -> LLMProvider:
        """Create an LLM provider instance"""
        providers = {
            "groq": GroqProvider,
        }
        
        provider_class = providers.get(provider_type.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        return provider_class(api_key, config)