"""
Integration tests for the complete MoE system
"""

import pytest
import os
import asyncio
from src.core.config import MoEConfig, SecretStr
from src.core.state import create_initial_state
from src.graph.builder import MoEGraphBuilder


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set"
)
class TestMoEIntegration:
    """Integration tests requiring actual API key"""
    
        
    def test_full_pipeline_technical_query(self):
        """Test full pipeline with technical query"""
        async def run():
            config = MoEConfig(groq_api_key=SecretStr(os.getenv("GROQ_API_KEY")))
            builder = MoEGraphBuilder(config)
            graph = builder.build()
            
            state = create_initial_state("What is a binary search tree?")
            return await graph.ainvoke(state)
            
        result = asyncio.run(run())
        
        assert result['query'] == "What is a binary search tree?"
        assert len(result['selected_experts']) > 0
        assert len(result['expert_responses']) > 0
        assert result['final_answer'] != ""
        assert len(result['reasoning_steps']) > 0
    
    def test_full_pipeline_creative_query(self):
        """Test full pipeline with creative query"""
        async def run():
            config = MoEConfig(groq_api_key=SecretStr(os.getenv("GROQ_API_KEY")))
            builder = MoEGraphBuilder(config)
            graph = builder.build()
            
            state = create_initial_state("Write a haiku about AI")
            return await graph.ainvoke(state)
            
        result = asyncio.run(run())
        
        assert "creative" in result['selected_experts']
        assert result['final_answer'] != ""