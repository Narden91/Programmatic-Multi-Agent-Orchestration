"""
Tests for the programmatic orchestration pipeline (Orchestrator -> CodeExecutor)
"""

import os
import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.core.config import MoEConfig
from src.core.state import create_initial_state
from src.graph.builder import MoEGraphBuilder
from src.agents.orchestrator import OrchestratorAgent, CodeExecutionAgent


class TestOrchestratorAgent:
    """Unit tests for OrchestratorAgent"""

    def test_generates_code_from_query(self):
        """Test that the orchestrator produces generated_code in its output"""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(
            content='```python\nasync def orchestrate():\n    return "hello"\n```'
        ))

        agent = OrchestratorAgent(mock_llm, ["technical", "creative"])
        state = create_initial_state("Explain recursion")
        result = agent.execute(state)

        assert "generated_code" in result
        assert "orchestrate" in result["generated_code"]
        assert len(result["reasoning_steps"]) > 0

    def test_retry_prompt_on_code_failure(self):
        """Test that orchestrator uses a retry prompt when code_execution_error is set"""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(
            content='```python\nasync def orchestrate():\n    return "fixed"\n```'
        ))

        agent = OrchestratorAgent(mock_llm, ["technical"])
        state = create_initial_state("Explain recursion")
        state["code_execution_error"] = "NameError: name 'foo' is not defined"

        result = agent.execute(state)

        # The prompt sent to the LLM should mention the previous error
        prompt_arg = mock_llm.invoke.call_args[0][0]
        assert "NameError" in prompt_arg
        assert "generated_code" in result


class TestCodeExecutionAgent:
    """Unit tests for CodeExecutionAgent"""

    def test_successful_execution_populates_state(self):
        """Test that a successful sandbox run returns answer + expert metadata"""
        agent = CodeExecutionAgent(None)

        state = create_initial_state("test")
        state["generated_code"] = (
            'async def orchestrate():\n'
            '    result = await query_technical_expert("What is AI?")\n'
            '    return result\n'
        )

        async def run():
            return await agent.aexecute(state)

        with patch.object(
            agent.sandbox, "execute",
            new_callable=AsyncMock,
            return_value={
                "result": "AI is artificial intelligence.",
                "selected_experts": ["technical"],
                "expert_responses": {"technical": "AI is artificial intelligence."},
            },
        ):
            result = asyncio.run(run())

        assert result["final_answer"] == "AI is artificial intelligence."
        assert "technical" in result["selected_experts"]
        assert "technical" in result["expert_responses"]
        assert result["code_execution_error"] == ""

    def test_failed_execution_records_error(self):
        """Test that a sandbox error is captured without crashing"""
        agent = CodeExecutionAgent(None)

        state = create_initial_state("test")
        state["generated_code"] = "invalid code"

        async def run():
            return await agent.aexecute(state)

        with patch.object(
            agent.sandbox, "execute",
            new_callable=AsyncMock,
            side_effect=SyntaxError("invalid syntax"),
        ):
            result = asyncio.run(run())

        assert "invalid syntax" in result["code_execution_error"]
        assert result["code_execution_iterations"] == 1


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set",
)
class TestOrchestratorIntegration:
    """Integration test for the full programmatic orchestration loop"""

    def test_end_to_end_orchestration(self):
        """Test that orchestrator generates code, sandbox executes it, and state is complete"""
        async def run():
            config = MoEConfig(groq_api_key=os.getenv("GROQ_API_KEY"))
            config.validate()

            builder = MoEGraphBuilder(config)
            graph = builder.build()

            state = create_initial_state(
                "Explain the concept of quantum entanglement. "
                "Give a creative analogy and a technical explanation."
            )
            return await graph.ainvoke(state)

        result = asyncio.run(run())

        assert result.get("generated_code"), "Orchestrator should produce code"
        assert result.get("code_execution_iterations", 0) >= 1
        assert result.get("final_answer"), "Pipeline should produce a final answer"
        assert len(result.get("selected_experts", [])) > 0, "At least one expert should be called"
