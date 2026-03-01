import pytest
import asyncio
from src.core.memory import EphemeralMemory

@pytest.mark.asyncio
async def test_ephemeral_memory_store_and_search():
    memory = EphemeralMemory(model_name="all-MiniLM-L6-v2")
    
    # Store items
    await memory.store("fact_1", "The capital of France is Paris.", metadata={"type": "geography"})
    await memory.store("fact_2", "Quantum mechanics is a fundamental theory in physics.", metadata={"type": "science"})
    
    # Search
    results = await memory.search("What is the capital of France?", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "fact_1"
    
    # Compress Context
    compressed = await memory.compress_context("What is the capital of France?", top_k=1)
    assert "Paris" in compressed
    
    memory.cleanup()
