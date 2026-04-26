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
        self.prompts = []

    def _next(self):
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return SimpleNamespace(content=self._responses[idx])

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return self._next()

    async def ainvoke(self, prompt: str):
        self.prompts.append(prompt)
        return self._next()


class DummyRegistry:
    def __init__(self, rows=None, atom_rows=None):
        self.rows = rows or []
        self.atom_rows = atom_rows or []
        self.neighborhood_rows = []
        self.plan_rows = []
        self.search_atom_neighborhoods_calls = 0
        self.search_plan_motifs_calls = 0

    def search(self, query: str, top_k: int = 2):
        return self.rows[:top_k]

    def search_atoms(self, query: str, top_k: int = 4):
        return self.atom_rows[:top_k]

    def search_atom_neighborhoods(self, query: str, top_k: int = 2):
        self.search_atom_neighborhoods_calls += 1
        return self.neighborhood_rows[:top_k]

    def search_plan_motifs(self, query: str, top_k: int = 3):
        self.search_plan_motifs_calls += 1
        return self.plan_rows[:top_k]


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
        lambda *args, **kwargs: DummyRegistry(similar_rows),
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
    assert selection["pruned_candidate_count"] == 1


@pytest.mark.asyncio
async def test_retry_path_keeps_single_generation(monkeypatch):
    monkeypatch.setattr(
        "src.agents.orchestrator.OrchestrationRegistry",
        lambda *args, **kwargs: DummyRegistry([]),
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


def test_config_rejects_invalid_atom_few_shot_count():
    cfg = MoEConfig(
        groq_api_key=SecretStr("fake-key"),
        orchestrator_atom_few_shot_count=-1,
    )

    with pytest.raises(ValueError, match="ORCHESTRATOR_ATOM_FEW_SHOTS"):
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
        lambda *args, **kwargs: DummyRegistry(similar_rows),
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


@pytest.mark.asyncio
async def test_orchestrator_prompt_includes_atom_level_few_shots(monkeypatch):
    registry_rows = [
        {
            "task_description": "Explain search algorithms",
            "script_content": "async def orchestrate():\n    return 'ok'\n",
            "score": 0.8,
            "similarity": 0.9,
            "metadata": {},
        }
    ]
    atom_rows = [
        {
            "task_description": "Explain search algorithms",
            "agent_type": "technical",
            "confidence": 0.95,
            "dependencies": ["sorted-input"],
            "evidence_tags": ["algorithm"],
            "payload": {"text": "Binary search halves the interval.", "content_hash": "hash-1"},
        }
    ]

    monkeypatch.setattr(
        "src.agents.orchestrator.OrchestrationRegistry",
        lambda *args, **kwargs: DummyRegistry(registry_rows, atom_rows),
    )

    llm = StubLLM(["""```python
async def orchestrate():
    return 'ok'
```"""])
    agent = OrchestratorAgent(
        llm_provider=llm,
        available_experts=["technical"],
        candidate_count=1,
        script_few_shot_count=1,
        atom_few_shot_count=1,
    )

    state = create_initial_state("Explain binary search")
    await agent.execute(state)

    assert "Relevant semantic atoms retrieved" in llm.prompts[0]
    assert "Binary search halves the interval." in llm.prompts[0]


@pytest.mark.asyncio
async def test_orchestrator_prompt_includes_atom_graph_hints(monkeypatch):
    registry = DummyRegistry(
        rows=[
            {
                "task_description": "Explain search algorithms",
                "script_content": "async def orchestrate():\n    return 'ok'\n",
                "score": 0.8,
                "similarity": 0.9,
                "metadata": {},
            }
        ],
        atom_rows=[
            {
                "task_description": "Explain search algorithms",
                "agent_type": "technical",
                "confidence": 0.95,
                "dependencies": ["bs-1"],
                "evidence_tags": ["algorithm"],
                "payload": {"text": "It requires sorted data.", "content_hash": "hash-2"},
            }
        ],
    )
    registry.neighborhood_rows = [
        {
            "seed": {
                "task_description": "Explain search algorithms",
                "agent_type": "technical",
                "atom_id": "bs-2",
                "similarity": 0.97,
                "payload": {"atom_id": "bs-2", "text": "It requires sorted data."},
            },
            "neighbors": [
                {
                    "atom_id": "bs-1",
                    "payload": {"atom_id": "bs-1", "text": "Binary search halves the interval."},
                }
            ],
            "edges": [
                {"source_atom_id": "bs-2", "target_atom_id": "bs-1", "edge_type": "dependency"}
            ],
        }
    ]

    monkeypatch.setattr(
        "src.agents.orchestrator.OrchestrationRegistry",
        lambda *args, **kwargs: registry,
    )

    llm = StubLLM(["""```python
async def orchestrate():
    return 'ok'
```"""])
    agent = OrchestratorAgent(
        llm_provider=llm,
        available_experts=["technical"],
        candidate_count=1,
        script_few_shot_count=1,
        atom_few_shot_count=2,
    )

    state = create_initial_state("Explain binary search")
    result = await agent.execute(state)

    assert registry.search_atom_neighborhoods_calls == 1
    assert "Relevant atom graph neighborhoods" in llm.prompts[0]
    assert "bs-2 -> bs-1 (dependency)" in llm.prompts[0]
    assert result["reasoning_steps"][0]["details"]["graph_few_shot_count"] == 1


@pytest.mark.asyncio
async def test_orchestrator_prompt_includes_plan_motifs(monkeypatch):
    registry = DummyRegistry(
        rows=[
            {
                "task_description": "Explain search algorithms",
                "script_content": "async def orchestrate():\n    return 'ok'\n",
                "score": 0.8,
                "similarity": 0.9,
                "metadata": {},
            }
        ],
        atom_rows=[
            {
                "task_description": "Explain search algorithms",
                "agent_type": "technical",
                "confidence": 0.95,
                "dependencies": ["bs-1"],
                "evidence_tags": ["algorithm"],
                "payload": {"text": "It requires sorted data.", "content_hash": "hash-2"},
            }
        ],
    )
    registry.plan_rows = [
        {
            "script_id": 1,
            "motif_index": 0,
            "task_description": "Explain search algorithms",
            "motif_text": "parallel group 1 technical via query_agent",
            "expert_type": "technical",
            "is_parallel": True,
            "group_id": 1,
            "similarity": 0.94,
        }
    ]

    monkeypatch.setattr(
        "src.agents.orchestrator.OrchestrationRegistry",
        lambda *args, **kwargs: registry,
    )

    llm = StubLLM(["""```python
async def orchestrate():
    return 'ok'
```"""])
    agent = OrchestratorAgent(
        llm_provider=llm,
        available_experts=["technical"],
        candidate_count=1,
        script_few_shot_count=1,
        atom_few_shot_count=2,
    )

    state = create_initial_state("Explain binary search")
    result = await agent.execute(state)

    assert registry.search_plan_motifs_calls == 1
    assert "Relevant compressed plan motifs" in llm.prompts[0]
    assert "parallel group 1 technical via query_agent" in llm.prompts[0]
    assert result["reasoning_steps"][0]["details"]["plan_few_shot_count"] == 1


@pytest.mark.asyncio
async def test_candidate_generation_modes_are_explicitly_graph_biased(monkeypatch):
    registry = DummyRegistry(
        rows=[
            {
                "task_description": "Explain search algorithms",
                "script_content": "async def orchestrate():\n    return 'ok'\n",
                "score": 0.8,
                "similarity": 0.9,
                "metadata": {},
            }
        ],
        atom_rows=[
            {
                "task_description": "Explain search algorithms",
                "agent_type": "technical",
                "confidence": 0.95,
                "dependencies": ["bs-1"],
                "evidence_tags": ["algorithm"],
                "payload": {"text": "It requires sorted data.", "content_hash": "hash-2"},
            }
        ],
    )
    registry.neighborhood_rows = [
        {
            "seed": {
                "task_description": "Explain search algorithms",
                "agent_type": "technical",
                "atom_id": "bs-2",
                "similarity": 0.97,
                "payload": {"atom_id": "bs-2", "text": "It requires sorted data."},
            },
            "neighbors": [
                {
                    "atom_id": "bs-1",
                    "payload": {"atom_id": "bs-1", "text": "Binary search halves the interval."},
                }
            ],
            "edges": [
                {"source_atom_id": "bs-2", "target_atom_id": "bs-1", "edge_type": "dependency"}
            ],
        }
    ]
    registry.plan_rows = [
        {
            "script_id": 1,
            "motif_index": 0,
            "task_description": "Explain search algorithms",
            "motif_text": "parallel group 1 technical via query_agent",
            "expert_type": "technical",
            "is_parallel": True,
            "group_id": 1,
            "similarity": 0.94,
        }
    ]

    monkeypatch.setattr(
        "src.agents.orchestrator.OrchestrationRegistry",
        lambda *args, **kwargs: registry,
    )

    llm = StubLLM([
        """```python
async def orchestrate():
    return 'a'
```""",
        """```python
async def orchestrate():
    return 'b'
```""",
        """```python
async def orchestrate():
    return 'c'
```""",
    ])
    agent = OrchestratorAgent(
        llm_provider=llm,
        available_experts=["technical"],
        candidate_count=3,
        script_few_shot_count=1,
        atom_few_shot_count=2,
    )

    state = create_initial_state("Explain binary search")
    result = await agent.execute(state)

    selection = result["reasoning_steps"][0]["details"]["selection"]
    retrieval = result["metadata"]["retrieval"]

    assert any("Candidate Generation Mode: dependency_preserving_graph" in prompt for prompt in llm.prompts)
    assert any("Candidate Generation Mode: plan_motif_reuse" in prompt for prompt in llm.prompts)
    assert selection["graph_biased_modes"] >= 2
    assert retrieval["neighborhood_reuse_rate"] == 1.0


def test_candidate_scoring_penalizes_atomization_cost(monkeypatch):
    monkeypatch.setattr(
        "src.agents.orchestrator.OrchestrationRegistry",
        lambda *args, **kwargs: DummyRegistry([]),
    )

    agent = OrchestratorAgent(
        llm_provider=StubLLM(["""```python
async def orchestrate():
    return 'ok'
```"""]),
        available_experts=["technical", "analytical", "general"],
        candidate_count=1,
    )

    over_fragmented = """
async def orchestrate():
    technical_summary = await query_agent('technical', 'Explain architecture')
    analytical_summary = await query_agent('analytical', 'Compare alternatives')
    market_summary = await query_agent('general', 'Summarize the market')
    risk_summary = await query_agent('analytical', 'List the risks')
    return technical_summary.text + analytical_summary.text + market_summary.text + risk_summary.text
"""
    compact_parallel = """
async def orchestrate():
    technical_task = query_agent('technical', 'Explain architecture')
    analytical_task = query_agent('analytical', 'Compare alternatives')
    technical_summary, analytical_summary = await asyncio.gather(technical_task, analytical_task)
    return technical_summary.text + analytical_summary.text
"""

    fragmented_score, fragmented_details = agent._score_candidate(over_fragmented, [])
    compact_score, compact_details = agent._score_candidate(compact_parallel, [])

    assert fragmented_details["atomization_cost"] > compact_details["atomization_cost"]
    assert compact_score > fragmented_score


@pytest.mark.asyncio
async def test_candidate_search_prefers_higher_learned_registry_alignment(monkeypatch):
    similar_rows = [
        {
            "task_description": "compare systems with creative analogy",
            "script_content": "async def orchestrate():\n    return 'creative'\n",
            "score": 0.9,
            "similarity": 0.96,
            "learning_rank": 0.0,
            "metadata": {
                "selected_experts": ["technical", "creative"],
                "execution_plan": {
                    "experts_used": ["technical", "creative"],
                    "has_sequential": False,
                    "has_parallel": True,
                    "gather_groups": 1,
                    "calls": [],
                },
                "trace_summary": {"atom_count_total": 6, "response_formats": ["semantic_atoms"]},
            },
        },
        {
            "task_description": "compare systems analytically",
            "script_content": "async def orchestrate():\n    return 'analytical'\n",
            "score": 0.9,
            "similarity": 0.96,
            "learning_rank": 0.9,
            "metadata": {
                "selected_experts": ["technical", "analytical"],
                "execution_plan": {
                    "experts_used": ["technical", "analytical"],
                    "has_sequential": False,
                    "has_parallel": True,
                    "gather_groups": 1,
                    "calls": [],
                },
                "trace_summary": {"atom_count_total": 6, "response_formats": ["semantic_atoms"]},
            },
        },
    ]

    monkeypatch.setattr(
        "src.agents.orchestrator.OrchestrationRegistry",
        lambda *args, **kwargs: DummyRegistry(similar_rows),
    )

    candidate_a = """```python
async def orchestrate():
    technical_task = query_agent('technical', 'Explain architecture')
    creative_task = query_agent('creative', 'Offer an analogy')
    technical_summary, creative_summary = await asyncio.gather(technical_task, creative_task)
    return technical_summary.text + creative_summary.text
```"""

    candidate_b = """```python
async def orchestrate():
    technical_task = query_agent('technical', 'Explain architecture')
    analytical_task = query_agent('analytical', 'Compare alternatives')
    technical_summary, analytical_summary = await asyncio.gather(technical_task, analytical_task)
    return technical_summary.text + analytical_summary.text
```"""

    llm = StubLLM([candidate_a, candidate_b])
    agent = OrchestratorAgent(
        llm_provider=llm,
        available_experts=["technical", "analytical", "creative"],
        candidate_count=2,
    )

    state = create_initial_state("Compare two database architectures")
    result = await agent.execute(state)

    assert "analytical" in result["generated_code"]
    selection = result["reasoning_steps"][0]["details"]["selection"]
    assert selection["selected_features"]["registry_learning_alignment"] > 0.2


def test_graph_prior_rewards_candidates_matching_retrieved_structure(monkeypatch):
    monkeypatch.setattr(
        "src.agents.orchestrator.OrchestrationRegistry",
        lambda *args, **kwargs: DummyRegistry([]),
    )

    agent = OrchestratorAgent(
        llm_provider=StubLLM(["""```python
async def orchestrate():
    return 'ok'
```"""]),
        available_experts=["technical", "analytical", "creative"],
        candidate_count=1,
    )

    neighborhood_rows = [
        {
            "seed": {
                "agent_type": "analytical",
                "similarity": 0.97,
            },
            "neighbors": [{"atom_id": "n-1"}],
            "edges": [{"source_atom_id": "n-0", "target_atom_id": "n-1", "edge_type": "dependency"}],
        }
    ]
    plan_rows = [
        {
            "expert_type": "analytical",
            "is_parallel": True,
            "group_id": 1,
            "similarity": 0.95,
        }
    ]

    matching_candidate = """
async def orchestrate():
    technical_task = query_agent('technical', 'Explain architecture')
    analytical_task = query_agent('analytical', 'Compare alternatives')
    technical_summary, analytical_summary = await asyncio.gather(technical_task, analytical_task)
    return technical_summary.text + analytical_summary.text
"""
    mismatching_candidate = """
async def orchestrate():
    technical_task = query_agent('technical', 'Explain architecture')
    creative_task = query_agent('creative', 'Offer an analogy')
    technical_summary, creative_summary = await asyncio.gather(technical_task, creative_task)
    return technical_summary.text + creative_summary.text
"""

    matching_score, matching_details = agent._score_candidate(
        matching_candidate,
        [],
        neighborhood_rows=neighborhood_rows,
        plan_rows=plan_rows,
    )
    mismatching_score, mismatching_details = agent._score_candidate(
        mismatching_candidate,
        [],
        neighborhood_rows=neighborhood_rows,
        plan_rows=plan_rows,
    )

    assert matching_details["graph_prior"] > mismatching_details["graph_prior"]
    assert matching_details["graph_motif_expert_coverage"] == 1.0
    assert mismatching_details["graph_neighborhood_expert_coverage"] == 0.0
    assert matching_score > mismatching_score


@pytest.mark.asyncio
async def test_candidate_search_uses_graph_prior_in_selection(monkeypatch):
    registry = DummyRegistry(rows=[])
    registry.neighborhood_rows = [
        {
            "seed": {
                "task_description": "Compare system designs",
                "agent_type": "analytical",
                "atom_id": "dep-0",
                "similarity": 0.97,
                "payload": {"atom_id": "dep-0", "text": "Compare alternatives before synthesis."},
            },
            "neighbors": [
                {
                    "atom_id": "dep-1",
                    "payload": {"atom_id": "dep-1", "text": "Preserve dependency order."},
                }
            ],
            "edges": [
                {"source_atom_id": "dep-0", "target_atom_id": "dep-1", "edge_type": "dependency"}
            ],
        }
    ]
    registry.plan_rows = [
        {
            "script_id": 1,
            "motif_index": 0,
            "task_description": "Compare system designs",
            "motif_text": "parallel group 1 analytical via query_agent",
            "expert_type": "analytical",
            "is_parallel": True,
            "group_id": 1,
            "similarity": 0.96,
        }
    ]

    monkeypatch.setattr(
        "src.agents.orchestrator.OrchestrationRegistry",
        lambda *args, **kwargs: registry,
    )

    candidate_a = """```python
async def orchestrate():
    technical_task = query_agent('technical', 'Explain architecture')
    creative_task = query_agent('creative', 'Offer an analogy')
    technical_summary, creative_summary = await asyncio.gather(technical_task, creative_task)
    return technical_summary.text + creative_summary.text
```"""
    candidate_b = """```python
async def orchestrate():
    technical_task = query_agent('technical', 'Explain architecture')
    analytical_task = query_agent('analytical', 'Compare alternatives')
    technical_summary, analytical_summary = await asyncio.gather(technical_task, analytical_task)
    return technical_summary.text + analytical_summary.text
```"""

    llm = StubLLM([candidate_a, candidate_b])
    agent = OrchestratorAgent(
        llm_provider=llm,
        available_experts=["technical", "analytical", "creative"],
        candidate_count=2,
        atom_few_shot_count=2,
    )

    state = create_initial_state("Compare two system designs")
    result = await agent.execute(state)

    assert "analytical" in result["generated_code"]
    selection = result["reasoning_steps"][0]["details"]["selection"]
    assert selection["selected_features"]["graph_prior"] > 0.2
    assert selection["selected_features"]["graph_motif_expert_coverage"] == 1.0
