"""
Unit tests for agent modules
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.agents.router import RouterAgent
from src.agents.experts.generic import GenericExpert
from src.core.state import create_initial_state


class TestRouterAgent:
    """Tests for RouterAgent"""
    
    def test_router_selects_technical_expert(self):
        """Test that router selects technical expert for code questions"""
        # Mock LLM
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="technical"))
        
        # Create router
        router = RouterAgent(mock_llm, ["technical", "creative", "analytical", "general"])
        
        # Test state
        state = create_initial_state("Explain binary search algorithm")
        
        # Execute
        result = router.execute(state)
        
        # Assert
        assert "technical" in result['selected_experts']
        assert result['query_type'] == "technical"
    
    def test_router_selects_multiple_experts(self):
        """Test that router can select multiple experts"""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="technical,creative"))
        
        router = RouterAgent(mock_llm, ["technical", "creative", "analytical", "general"])
        state = create_initial_state("Explain recursion with creative analogies")
        
        result = router.execute(state)
        
        assert len(result['selected_experts']) == 2
        assert "technical" in result['selected_experts']
        assert "creative" in result['selected_experts']
        assert result['query_type'] == "multi-expert"


class TestGenericExpert:
    """Tests for GenericExpert (replaces type-specific expert classes)"""
    
    def test_technical_expert_generates_response(self):
        """Test that technical expert generates response when selected"""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Binary search is..."))
        
        expert = GenericExpert(expert_type="technical", llm_provider=mock_llm, confidence_threshold=0.85)
        
        state = create_initial_state("Explain binary search")
        state['selected_experts'] = ["technical"]
        
        result = expert.execute(state)
        
        assert "technical" in result['expert_responses']
        assert result['expert_responses']['technical'] == "Binary search is..."
        assert result['confidence_scores']['technical'] == 0.85
    
    def test_technical_expert_skips_when_not_selected(self):
        """Test that technical expert skips when not selected"""
        mock_llm = Mock()
        expert = GenericExpert(expert_type="technical", llm_provider=mock_llm)
        
        state = create_initial_state("Write a poem")
        state['selected_experts'] = ["creative"]
        
        result = expert.execute(state)
        
        assert result == {}
    
    def test_creative_expert_generates_response(self):
        """Test that creative expert generates response when selected"""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Once upon a time..."))
        
        expert = GenericExpert(expert_type="creative", llm_provider=mock_llm)
        
        state = create_initial_state("Write a haiku")
        state['selected_experts'] = ["creative"]
        
        result = expert.execute(state)
        
        assert "creative" in result['expert_responses']
        assert result['confidence_scores']['creative'] == 0.75  # default threshold
    
    def test_invalid_expert_type_raises(self):
        """Test that invalid expert type raises ValueError"""
        mock_llm = Mock()
        with pytest.raises(ValueError, match="Invalid expert type"):
            GenericExpert(expert_type="nonexistent", llm_provider=mock_llm)