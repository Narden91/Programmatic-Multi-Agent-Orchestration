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

