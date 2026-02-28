"""
Unit tests for agent modules
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.agents.experts.generic import GenericExpert
from src.core.state import create_initial_state


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