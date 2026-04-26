import pytest
import os
from src.core.registry import OrchestrationRegistry

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

