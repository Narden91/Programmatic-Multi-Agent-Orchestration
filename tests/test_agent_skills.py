"""Unit tests for Karpathy, Caveman, and Ponytail agent skills and token budgeting."""

import pytest
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

from src.core.agents import query_agent, _SKILL_DIRECTIVES, _prompt_cache
from src.core.config import config, LLMConfig
from src.core.sandbox import CodeSandbox, SandboxPolicy


@pytest.fixture(autouse=True)
def clear_agent_cache():
    _prompt_cache.clear()
    yield
    _prompt_cache.clear()


@pytest.mark.asyncio
async def test_caveman_skill_directive_injected(monkeypatch):
    """Test that explicit skill='caveman' injects the token-saving caveman directive."""
    captured_messages = []

    class _MockProvider:
        def __init__(self, model_name: str):
            self.model_name = model_name

        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return SimpleNamespace(
                content="Dense facts: A=1, B=2.",
                response_metadata={"token_usage": {"total_tokens": 12}},
            )

    monkeypatch.setattr(
        "src.core.agents.LLMFactory.create_provider",
        lambda provider_type, api_key, config: _MockProvider(config.model_name),
    )

    result = await query_agent("technical", "Summarize protocol", skill="caveman")

    assert result.metadata["applied_skill"] == "caveman"
    assert "CAVEMAN TOKEN COMPRESSION" in captured_messages[0].content
    assert result.text == "Dense facts: A=1, B=2."


@pytest.mark.asyncio
async def test_ponytail_skill_directive_injected(monkeypatch):
    """Test that explicit skill='ponytail' injects the high-precision analytical directive."""
    captured_messages = []

    class _MockProvider:
        def __init__(self, model_name: str):
            self.model_name = model_name

        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return SimpleNamespace(
                content="First principles: Invariant holds.",
                response_metadata={"token_usage": {"total_tokens": 25}},
            )

    monkeypatch.setattr(
        "src.core.agents.LLMFactory.create_provider",
        lambda provider_type, api_key, config: _MockProvider(config.model_name),
    )

    result = await query_agent("analytical", "Verify theorem", skill="ponytail")

    assert result.metadata["applied_skill"] == "ponytail"
    assert "PONYTAIL PRECISION & RIGOR" in captured_messages[0].content


@pytest.mark.asyncio
async def test_low_weight_auto_activates_caveman(monkeypatch):
    """Test that setting weight < 0.4 automatically activates caveman compression."""
    captured_messages = []

    class _MockProvider:
        def __init__(self, model_name: str):
            self.model_name = model_name

        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return SimpleNamespace(
                content="Auxiliary check: OK",
                response_metadata={"token_usage": {"total_tokens": 8}},
            )

    monkeypatch.setattr(
        "src.core.agents.LLMFactory.create_provider",
        lambda provider_type, api_key, config: _MockProvider(config.model_name),
    )

    result = await query_agent("general", "Quick sanity check", weight=0.2)

    assert result.metadata["applied_skill"] == "caveman"
    assert result.metadata["weight"] == 0.2
    assert "CAVEMAN TOKEN COMPRESSION" in captured_messages[0].content


@pytest.mark.asyncio
async def test_high_weight_auto_activates_ponytail(monkeypatch):
    """Test that setting weight >= 0.8 automatically activates ponytail precision."""
    captured_messages = []

    class _MockProvider:
        def __init__(self, model_name: str):
            self.model_name = model_name

        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return SimpleNamespace(
                content="Primary architectural plan: Validated.",
                response_metadata={"token_usage": {"total_tokens": 40}},
            )

    monkeypatch.setattr(
        "src.core.agents.LLMFactory.create_provider",
        lambda provider_type, api_key, config: _MockProvider(config.model_name),
    )

    result = await query_agent("technical", "Design database schema", weight=0.9)

    assert result.metadata["applied_skill"] == "ponytail"
    assert result.metadata["weight"] == 0.9
    assert "PONYTAIL PRECISION & RIGOR" in captured_messages[0].content


@pytest.mark.asyncio
async def test_sandbox_executes_with_skills_and_weights(monkeypatch):
    """Test that the sandbox runtime successfully executes scripts with skill directives."""
    class _MockProvider:
        def __init__(self, model_name: str):
            self.model_name = model_name

        async def ainvoke(self, messages):
            sys_text = messages[0].content
            if "CAVEMAN" in sys_text:
                return SimpleNamespace(content="FACT: 42", response_metadata={"token_usage": {"total_tokens": 5}})
            return SimpleNamespace(content="The answer is 42.", response_metadata={"token_usage": {"total_tokens": 15}})

    monkeypatch.setattr(
        "src.core.agents.LLMFactory.create_provider",
        lambda provider_type, api_key, config: _MockProvider(config.model_name),
    )

    sandbox = CodeSandbox(isolate_process=False)
    code = """
async def orchestrate():
    aux = await query_agent("technical", "Verify answer", weight=0.1, skill="caveman")
    final = await query_agent("general", f"Synthesize: {aux.text}", skill="karpathy")
    return final.text
"""
    result = await sandbox.execute(code)
    assert result["result"] == "The answer is 42."
    assert "technical" in result["selected_experts"]
    assert "general" in result["selected_experts"]
