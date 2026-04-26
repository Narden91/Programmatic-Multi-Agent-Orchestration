from types import SimpleNamespace

import pytest

from src.agents.base import BaseAgent
from src.core import agents as agents_module
from src.core.agents import query_agent
from src.core.config import LLMConfig, config
from src.llm.providers import LLMProvider


class _FallbackProvider(LLMProvider):
    provider_name = "dummy"

    def __init__(self, api_key: str, config: LLMConfig):
        self._api_key = api_key
        self._config = config
        self.model_name = config.model_name

    def invoke(self, prompt: str):
        if self.model_name == "llama-3.3-70b-versatile":
            raise Exception("rate_limit_exceeded")
        return SimpleNamespace(content="ok", response_metadata={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    async def ainvoke(self, prompt: str):
        if self.model_name == "llama-3.3-70b-versatile":
            raise Exception("rate_limit_exceeded")
        return SimpleNamespace(content="ok", response_metadata={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1}})


class _DummyAgent(BaseAgent):
    def execute(self, state):
        return {}


@pytest.mark.asyncio
async def test_base_agent_switches_to_fallback_model():
    agent = _DummyAgent(
        "Orchestrator",
        _FallbackProvider("key", LLMConfig("llama-3.3-70b-versatile")),
        max_retries=2,
    )

    response = await agent.ainvoke_with_retry("hello")

    assert response.content == "ok"
    assert agent.llm.model_name == "llama-3.1-8b-instant"


@pytest.mark.asyncio
async def test_query_agent_switches_expert_to_fallback_model(monkeypatch):
    agents_module._prompt_cache.clear()
    monkeypatch.setattr(config, "get_provider_type", lambda: "groq")
    monkeypatch.setattr(config, "get_api_key", lambda provider_type: "fake-key")

    calls = []

    class _Provider:
        def __init__(self, model_name: str):
            self.model_name = model_name

        async def ainvoke(self, messages):
            calls.append(self.model_name)
            if self.model_name == "llama-3.3-70b-versatile":
                raise Exception("rate_limit_exceeded")
            return SimpleNamespace(content="Fallback answer.", response_metadata={})

    monkeypatch.setattr(
        "src.core.agents.LLMFactory.create_provider",
        lambda provider_type, api_key, config: _Provider(config.model_name),
    )

    original_config = config.expert_configs["technical"].llm_config
    config.expert_configs["technical"].llm_config = LLMConfig("llama-3.3-70b-versatile")
    try:
        result = await query_agent("technical", "Explain fallback")
    finally:
        config.expert_configs["technical"].llm_config = original_config
        agents_module._prompt_cache.clear()

    assert calls == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    assert result.metadata["model"] == "llama-3.1-8b-instant"