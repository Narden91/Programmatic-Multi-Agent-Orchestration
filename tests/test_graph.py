"""
Tests for graph construction
"""

import pytest
from unittest.mock import Mock, patch
from src.graph.builder import MoEGraphBuilder
from src.core.config import MoEConfig, SecretStr


class TestMoEGraphBuilder:
    """Tests for MoEGraphBuilder"""
    
    @patch('src.graph.builder.LLMFactory.create_provider')
    def test_graph_builder_initializes_all_agents(self, mock_factory):
        """Test that graph builder initializes orchestrator and code executor"""
        mock_llm = Mock()
        mock_factory.return_value = mock_llm
        
        config = MoEConfig(groq_api_key=SecretStr("test_key"))
        builder = MoEGraphBuilder(config)
        
        # Check the programmatic orchestration agents are initialized
        assert 'orchestrator' in builder.agents
        assert 'code_executor' in builder.agents
    
    @patch('src.graph.builder.LLMFactory.create_provider')
    def test_graph_builds_successfully(self, mock_factory):
        """Test that graph builds without errors"""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="test"))
        mock_factory.return_value = mock_llm
        
        config = MoEConfig(groq_api_key=SecretStr("test_key"))
        builder = MoEGraphBuilder(config)
        
        # Should not raise
        graph = builder.build()
        assert graph is not None