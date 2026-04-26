import pytest
import os
from src.core.registry import OrchestrationRegistry


class _FakeEmbeddingModel:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, value):
        if isinstance(value, list):
            return [self.mapping[item] for item in value]
        return self.mapping[value]

@pytest.fixture
def temp_registry(tmp_path):
    db_file = tmp_path / "test_registry.db"
    # use a fast model or mock the encoder
    registry = OrchestrationRegistry(db_path=str(db_file), model_name="all-MiniLM-L6-v2")
    yield registry
    if db_file.exists():
        os.remove(db_file)

def test_store_and_search(temp_registry):
    # Store simple script
    script1 = "async def orchestrate(): return 'math'"
    temp_registry.store_script("Calculate the sum of 5 and 10", script1, score=1.0)
    
    script2 = "async def orchestrate(): return 'creative'"
    temp_registry.store_script("Write a poem about the sea", script2, score=0.8)
    
    # Search should return the math script
    results = temp_registry.search("What is 2 + 2?", top_k=1)
    
    assert len(results) == 1
    assert "math" in results[0]["script_content"]
    assert results[0]["score"] == 1.0


def test_store_and_search_preserves_metadata(temp_registry):
    temp_registry.store_script(
        "Explain binary search",
        "async def orchestrate(): return 'ok'",
        score=0.9,
        metadata={
            "selected_experts": ["technical"],
            "trace_summary": {"atom_count_total": 2},
        },
    )

    results = temp_registry.search("How does binary search work?", top_k=1)

    assert len(results) == 1
    assert results[0]["metadata"]["selected_experts"] == ["technical"]
    assert results[0]["metadata"]["trace_summary"]["atom_count_total"] == 2


def test_store_script_merges_learning_updates_for_same_script(temp_registry):
    script = "async def orchestrate(): return 'ok'"

    temp_registry.store_script(
        "Explain binary search",
        script,
        score=0.9,
        metadata={
            "outcome": "success",
            "execution_metrics": {
                "retry_count": 0,
                "total_tokens": 100,
                "neighborhood_reuse_rate": 1.0,
                "plan_reuse_rate": 0.5,
            },
        },
    )
    temp_registry.store_script(
        "Explain binary search",
        script,
        score=0.3,
        metadata={
            "outcome": "error",
            "error": "boom",
            "execution_metrics": {
                "retry_count": 2,
                "total_tokens": 300,
                "neighborhood_reuse_rate": 0.0,
                "plan_reuse_rate": 0.0,
            },
        },
    )

    results = temp_registry.search("Explain binary search", top_k=1)

    assert len(results) == 1
    assert results[0]["execution_count"] == 2
    learning = results[0]["metadata"]["learning"]
    assert learning["observations"] == 2
    assert learning["success_count"] == 1
    assert learning["failure_count"] == 1
    assert learning["success_rate"] == 0.5
    assert learning["mean_retry_count"] == 1.0
    assert learning["mean_total_tokens"] == 200.0
    assert learning["mean_neighborhood_reuse_rate"] == 0.5


def test_search_prefers_successful_learned_script_when_similarity_ties(temp_registry):
    temp_registry.model = _FakeEmbeddingModel({
        "Explain binary search": [1.0, 0.0],
    })

    temp_registry.store_script(
        "Explain binary search",
        "async def orchestrate(): return 'failure'",
        score=0.0,
        metadata={
            "outcome": "error",
            "execution_metrics": {"retry_count": 2, "total_tokens": 450},
        },
    )
    temp_registry.store_script(
        "Explain binary search",
        "async def orchestrate(): return 'success'",
        score=0.95,
        metadata={
            "outcome": "success",
            "execution_metrics": {
                "retry_count": 0,
                "total_tokens": 90,
                "neighborhood_reuse_rate": 1.0,
                "plan_reuse_rate": 1.0,
            },
        },
    )

    results = temp_registry.search("Explain binary search", top_k=2)

    assert len(results) == 2
    assert "success" in results[0]["script_content"]
    assert results[0]["learning_rank"] > results[1]["learning_rank"]


def test_store_and_fetch_full_atom_payloads(temp_registry):
    script_id = temp_registry.store_script(
        "Explain binary search",
        "async def orchestrate(): return 'ok'",
        score=0.9,
        atom_payloads=[
            {
                "span_name": "query_agent_technical",
                "agent_type": "technical",
                "response_format": "semantic_atoms",
                "atom_index": 0,
                "payload": {
                    "atom_id": "bs-1",
                    "text": "Binary search halves the interval.",
                    "confidence": 0.93,
                    "dependencies": [],
                    "evidence_tags": ["algorithm"],
                    "metadata": {"source": "unit-test"},
                    "content_hash": "abcd1234",
                },
            },
            {
                "span_name": "query_agent_technical",
                "agent_type": "technical",
                "response_format": "semantic_atoms",
                "atom_index": 1,
                "payload": {
                    "atom_id": "bs-2",
                    "text": "It requires sorted data.",
                    "confidence": 0.98,
                    "dependencies": ["bs-1"],
                    "evidence_tags": ["precondition"],
                    "metadata": {"source": "unit-test"},
                    "content_hash": "efgh5678",
                },
            },
        ],
    )

    atoms = temp_registry.get_script_atoms(script_id)

    assert len(atoms) == 2
    assert atoms[0]["agent_type"] == "technical"
    assert atoms[0]["payload"]["text"] == "Binary search halves the interval."
    assert atoms[1]["dependencies"] == ["bs-1"]
    assert atoms[1]["payload"]["metadata"]["source"] == "unit-test"


def test_search_atoms_uses_vectorized_lookup(temp_registry):
    temp_registry.model = _FakeEmbeddingModel({
        "Explain binary search": [1.0, 0.0],
        "Binary search halves the interval.": [1.0, 0.0],
        "Write a poem": [0.0, 1.0],
        "Poems use rhythm.": [0.0, 1.0],
    })

    temp_registry.store_script(
        "Explain binary search",
        "async def orchestrate(): return 'ok'",
        atom_payloads=[
            {
                "span_name": "query_agent_technical",
                "agent_type": "technical",
                "response_format": "semantic_atoms",
                "atom_index": 0,
                "payload": {
                    "atom_id": "bs-1",
                    "text": "Binary search halves the interval.",
                    "confidence": 0.95,
                    "dependencies": [],
                    "evidence_tags": ["algorithm"],
                    "content_hash": "hash-1",
                },
            },
        ],
    )
    temp_registry.store_script(
        "Write a poem",
        "async def orchestrate(): return 'poem'",
        atom_payloads=[
            {
                "span_name": "query_agent_creative",
                "agent_type": "creative",
                "response_format": "semantic_atoms",
                "atom_index": 0,
                "payload": {
                    "atom_id": "poem-1",
                    "text": "Poems use rhythm.",
                    "confidence": 0.82,
                    "dependencies": [],
                    "evidence_tags": ["creative"],
                    "content_hash": "hash-2",
                },
            },
        ],
    )

    rows = temp_registry.search_atoms("Explain binary search", top_k=1)

    assert len(rows) == 1
    assert rows[0]["agent_type"] == "technical"
    assert rows[0]["payload"]["text"] == "Binary search halves the interval."
    assert rows[0]["similarity"] > 0.9


def test_search_atom_neighborhoods_expands_dependency_edges(temp_registry):
    temp_registry.model = _FakeEmbeddingModel({
        "Explain binary search": [1.0, 0.0],
        "Binary search halves the interval.": [0.8, 0.2],
        "It requires sorted data.": [1.0, 0.0],
    })

    script_id = temp_registry.store_script(
        "Explain binary search",
        "async def orchestrate(): return 'ok'",
        atom_payloads=[
            {
                "span_name": "query_agent_technical",
                "agent_type": "technical",
                "response_format": "semantic_atoms",
                "atom_index": 0,
                "payload": {
                    "atom_id": "bs-1",
                    "text": "Binary search halves the interval.",
                    "confidence": 0.95,
                    "dependencies": [],
                    "evidence_tags": ["algorithm"],
                    "content_hash": "hash-1",
                },
            },
            {
                "span_name": "query_agent_technical",
                "agent_type": "technical",
                "response_format": "semantic_atoms",
                "atom_index": 1,
                "payload": {
                    "atom_id": "bs-2",
                    "text": "It requires sorted data.",
                    "confidence": 0.98,
                    "dependencies": ["bs-1"],
                    "evidence_tags": ["precondition"],
                    "content_hash": "hash-2",
                },
            },
        ],
    )

    edges = temp_registry.get_atom_edges(script_id)
    neighborhoods = temp_registry.search_atom_neighborhoods("Explain binary search", top_k=1)

    assert edges == [{"source_atom_id": "bs-2", "target_atom_id": "bs-1", "edge_type": "dependency"}]
    assert len(neighborhoods) == 1
    assert neighborhoods[0]["seed"]["atom_id"] == "bs-2"
    assert neighborhoods[0]["neighbors"][0]["atom_id"] == "bs-1"
    assert neighborhoods[0]["edges"][0]["target_atom_id"] == "bs-1"


def test_store_and_search_plan_motifs(temp_registry):
    temp_registry.model = _FakeEmbeddingModel({
        "Explain binary search": [1.0, 0.0],
        "parallel group 1 technical via query_agent": [1.0, 0.0],
    })

    script_id = temp_registry.store_script(
        "Explain binary search",
        "async def orchestrate(): return 'ok'",
        metadata={
            "execution_plan": {
                "experts_used": ["technical"],
                "has_sequential": False,
                "has_parallel": True,
                "gather_groups": 1,
                "calls": [
                    {
                        "expert": "technical",
                        "function": "query_agent",
                        "line": 2,
                        "parallel": True,
                        "group": 1,
                    }
                ],
            }
        },
    )

    motifs = temp_registry.get_plan_motifs(script_id)
    rows = temp_registry.search_plan_motifs("Explain binary search", top_k=1)

    assert motifs[0]["motif_text"] == "parallel group 1 technical via query_agent"
    assert motifs[0]["is_parallel"] is True
    assert rows[0]["motif_text"] == "parallel group 1 technical via query_agent"
    assert rows[0]["similarity"] > 0.9

