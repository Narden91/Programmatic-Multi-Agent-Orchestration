from types import SimpleNamespace

import pytest

from src.agents.orchestrator import OrchestratorAgent
from src.core.config import MoEConfig, SecretStr
from src.core.state import create_initial_state


class StubLLM:
    model_name = "stub-model"

    def __init__(self, responses: list[str]):
        self._responses = responses
        self.calls = 0

    def _next(self):
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return SimpleNamespace(content=self._responses[idx])

    def invoke(self, prompt: str):
        return self._next()

    async def ainvoke(self, prompt: str):
        return self._next()


class DummyRegistry:
    def __init__(self, rows=None):
        self.rows = rows or []

    def search(self, query: str, top_k: int = 2):
        return self.rows[:top_k]


@pytest.mark.asyncio
async def test_candidate_search_prefers_balanced_parallel_script(monkeypatch):
    similar_rows = [
        {
            "task_description": "parallel technical and analytical comparison",
            "script_content": (
                "async def orchestrate():\n"
                "    t = query_agent('technical', 'Explain architecture')\n"
                "    a = query_agent('analytical', 'Compare alternatives')\n"
                "    t_res, a_res = await asyncio.gather(t, a)\n"
                "    return t_res.text + a_res.text\n"
            ),
            "score": 0.85,
            "similarity": 0.92,
        }
    ]

    monkeypatch.setattr(
        "src.agents.orchestrator.OrchestrationRegistry",
        lambda: DummyRegistry(similar_rows),
    )

    candidate_a = """```python
async def orchestrate():
    t = await query_agent('technical', 'Explain architecture')
    a = await query_agent('analytical', 'Compare alternatives')
    return t.text + a.text
```"""

    candidate_b = """```python
async def orchestrate():
    t_task = query_agent('technical', 'Explain architecture')
    a_task = query_agent('analytical', 'Compare alternatives')
    t, a = await asyncio.gather(t_task, a_task)
    return t.text + a.text
```"""

    candidate_c = """```python
result = 'missing orchestrate function'
```"""

    llm = StubLLM([candidate_a, candidate_b, candidate_c])
    agent = OrchestratorAgent(
        llm_provider=llm,
        available_experts=["technical", "analytical", "general"],
        candidate_count=3,
    )

    state = create_initial_state("Compare two database architectures")
    result = await agent.execute(state)

    assert "asyncio.gather" in result["generated_code"]
    selection = result["reasoning_steps"][0]["details"]["selection"]
    assert selection["selection_mode"] == "heuristic"
    assert selection["candidate_count"] >= 2


@pytest.mark.asyncio
async def test_retry_path_keeps_single_generation(monkeypatch):
    monkeypatch.setattr(
        "src.agents.orchestrator.OrchestrationRegistry",
        lambda: DummyRegistry([]),
    )

    fixed_candidate = """```python
async def orchestrate():
    return 'fixed'
```"""

    llm = StubLLM([fixed_candidate])
    agent = OrchestratorAgent(
        llm_provider=llm,
        available_experts=["technical"],
        candidate_count=4,
    )

    state = create_initial_state("Fix the failing script")
    state["code_execution_error"] = "NameError: foo is not defined"
    state["generated_code"] = "async def orchestrate():\n    return foo\n"

    result = await agent.execute(state)

    assert llm.calls == 1
    selection = result["reasoning_steps"][0]["details"]["selection"]
    assert selection["selection_mode"] == "single"
    assert result["generated_code"].startswith("async def orchestrate")


def test_config_rejects_invalid_candidate_count():
    cfg = MoEConfig(
        groq_api_key=SecretStr("fake-key"),
        orchestrator_candidate_count=0,
    )

    with pytest.raises(ValueError, match="ORCHESTRATOR_CANDIDATES"):
        cfg.validate()


@pytest.mark.asyncio
async def test_candidate_search_uses_atom_and_parallel_registry_bias(monkeypatch):
    similar_rows = [
        {
            "task_description": "parallel technical and analytical explanation",
            "script_content": (
                "async def orchestrate():\n"
                "    t_task = query_agent('technical', 'Explain architecture')\n"
                "    a_task = query_agent('analytical', 'Compare alternatives')\n"
                "    t, a = await asyncio.gather(t_task, a_task)\n"
                "    return t.text + a.text\n"
            ),
            "score": 0.92,
            "similarity": 0.96,
            "metadata": {
                "selected_experts": ["technical", "analytical"],
                "execution_plan": {
                    "experts_used": ["technical", "analytical"],
                    "has_sequential": False,
                    "has_parallel": True,
                    "gather_groups": 1,
                    "calls": [],
                },
                "trace_summary": {
                    "agent_span_count": 2,
                    "atom_count_total": 8,
                    "max_atom_count": 4,
                    "response_formats": ["semantic_atoms"],
                },
            },
        }
    ]

    monkeypatch.setattr(
        "src.agents.orchestrator.OrchestrationRegistry",
        lambda: DummyRegistry(similar_rows),
    )

    candidate_a = """```python
async def orchestrate():
    t = await query_agent('technical', 'Explain architecture')
    a = await query_agent('analytical', 'Compare alternatives')
    c = await query_agent('creative', 'Offer an analogy')
    return t.text + a.text + c.text
```"""

    candidate_b = """```python
async def orchestrate():
    t_task = query_agent('technical', 'Explain architecture')
    a_task = query_agent('analytical', 'Compare alternatives')
    t, a = await asyncio.gather(t_task, a_task)
    return t.text + a.text
```"""

    llm = StubLLM([candidate_a, candidate_b])
    agent = OrchestratorAgent(
        llm_provider=llm,
        available_experts=["technical", "analytical", "creative"],
        candidate_count=2,
    )

    state = create_initial_state("Compare two database architectures")
    result = await agent.execute(state)

    assert "asyncio.gather" in result["generated_code"]
    selection = result["reasoning_steps"][0]["details"]["selection"]
    assert selection["selected_features"]["registry_parallel_alignment"] == 1.0
    assert selection["selected_features"]["registry_atom_alignment"] > 0.9
