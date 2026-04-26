import json
from types import SimpleNamespace

import pytest

from src.core import agents as agents_module
from src.core.agents import AgentResult, query_agent
from src.core.config import config
from src.llm.prompts import OrchestratorPrompts


class _FakeProvider:
    def __init__(self, response_text: str):
        self.response_text = response_text

    async def ainvoke(self, messages):
        return SimpleNamespace(content=self.response_text, response_metadata={})


@pytest.fixture(autouse=True)
def clear_prompt_cache():
    agents_module._prompt_cache.clear()
    yield
    agents_module._prompt_cache.clear()


def test_agent_result_plain_text_fallback_creates_atom():
    result = AgentResult.from_response_text(
        "Binary search halves the search space.",
        agent_type="technical",
    )

    assert result.text == "Binary search halves the search space."
    assert result.metadata["response_format"] == "plain_text"
    assert len(result.atoms) == 1
    assert result.atoms[0].text == result.text


@pytest.mark.asyncio
async def test_query_agent_parses_structured_atoms(monkeypatch):
    payload = {
        "summary": "Binary search works on sorted arrays.",
        "atoms": [
            {
                "claim_id": "bs-1",
                "compressed_text": "Binary search repeatedly halves the interval.",
                "confidence": 0.91,
                "dependencies": [],
                "evidence_tags": ["algorithm"],
            },
            {
                "claim_id": "bs-2",
                "compressed_text": "It requires sorted input.",
                "confidence": 0.97,
                "dependencies": ["bs-1"],
                "evidence_tags": ["precondition"],
            },
        ],
    }

    monkeypatch.setattr(config, "get_provider_type", lambda: "groq")
    monkeypatch.setattr(config, "get_api_key", lambda provider_type: "fake-key")
    monkeypatch.setattr(
        "src.core.agents.LLMFactory.create_provider",
        lambda provider_type, api_key, config: _FakeProvider(json.dumps(payload)),
    )

    result = await query_agent("technical", "Explain binary search")

    assert result.text == payload["summary"]
    assert result.metadata["response_format"] == "semantic_atoms"
    assert len(result.atoms) == 2
    assert result.atoms[1].dependencies == ["bs-1"]


@pytest.mark.asyncio
async def test_query_agent_cache_keeps_atoms_and_zeroes_duration(monkeypatch):
    monkeypatch.setattr(config, "get_provider_type", lambda: "groq")
    monkeypatch.setattr(config, "get_api_key", lambda provider_type: "fake-key")
    monkeypatch.setattr(
        "src.core.agents.LLMFactory.create_provider",
        lambda provider_type, api_key, config: _FakeProvider("Compact explanation."),
    )

    first = await query_agent("technical", "Explain caching")
    second = await query_agent("technical", "Explain caching")

    assert len(first.atoms) == 1
    assert len(second.atoms) == 1
    assert second.text == first.text
    assert second.duration_ms == 0
    assert second.metadata == first.metadata


def test_orchestrator_prompt_mentions_optional_atoms_contract():
    prompt = OrchestratorPrompts.create_orchestration_prompt(
        query="Explain binary search",
        available_experts=["technical"],
    )

    assert "res.atoms" in prompt
    assert "atom_id" in prompt