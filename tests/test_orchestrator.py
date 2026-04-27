"""
Tests for the programmatic orchestration pipeline (Orchestrator -> CodeExecutor)
"""

import os
import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.core.config import MoEConfig, SecretStr
from src.core.state import create_initial_state
from src.core.sandbox import CodeSandbox, SandboxSecurityError, SandboxTimeoutError
from src.graph.builder import MoEGraphBuilder
from src.agents.orchestrator import OrchestratorAgent, CodeExecutionAgent


# ======================================================================
# Orchestrator Agent
# ======================================================================

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

    def test_retry_prompt_includes_failing_code_and_error(self):
        """Test that retry prompt contains both the failed code AND the error"""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(
            content='```python\nasync def orchestrate():\n    return "fixed"\n```'
        ))

        agent = OrchestratorAgent(mock_llm, ["technical"])
        state = create_initial_state("Explain recursion")
        state["code_execution_error"] = "NameError: name 'foo' is not defined"
        state["generated_code"] = 'async def orchestrate():\n    return foo\n'

        result = agent.execute(state)

        prompt_arg = mock_llm.invoke.call_args[0][0]
        # Must contain the error
        assert "NameError" in prompt_arg
        # Must contain the failing code so the LLM can fix it
        assert "return foo" in prompt_arg
        assert "generated_code" in result

    def test_first_attempt_clears_previous_error(self):
        """Test that generated state clears code_execution_error on fresh generation"""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(
            content='```python\nasync def orchestrate():\n    return "ok"\n```'
        ))

        agent = OrchestratorAgent(mock_llm, ["general"])
        state = create_initial_state("Hello")
        result = agent.execute(state)

        assert result["code_execution_error"] == ""


# ======================================================================
# Code Execution Agent
# ======================================================================

class TestCodeExecutionAgent:
    """Unit tests for CodeExecutionAgent"""

    def test_successful_execution_populates_state(self):
        """Test that a successful sandbox run returns answer + expert metadata"""
        from src.utils.metrics import reset_token_tracker
        reset_token_tracker()

        agent = CodeExecutionAgent()

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
        from src.utils.metrics import reset_token_tracker
        reset_token_tracker()

        agent = CodeExecutionAgent()

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

    def test_timeout_is_configurable(self):
        """Test that timeout_seconds is passed through to the sandbox"""
        agent = CodeExecutionAgent(timeout_seconds=120)
        assert agent.sandbox.timeout_seconds == 120

    def test_meta_final_answer_falls_back_to_direct_expert_response(self):
        """If the script returns critique text, prefer a direct expert deliverable."""
        from src.utils.metrics import reset_token_tracker
        reset_token_tracker()

        agent = CodeExecutionAgent()

        state = create_initial_state("Write a short story about AI discovering emotions")
        state["generated_code"] = (
            'async def orchestrate():\n'
            '    return "Based on the analysis and evaluation provided, here is feedback."\n'
        )

        async def run():
            return await agent.aexecute(state)

        with patch.object(
            agent.sandbox, "execute",
            new_callable=AsyncMock,
            return_value={
                "result": (
                    "Based on the analysis and evaluation provided, it seems the story has been refined.\n\n"
                    "1. Logical Consistency: Improved.\n"
                    "2. Emotional Resonance: Improved."
                ),
                "selected_experts": ["creative", "analytical", "critical-thinker"],
                "expert_responses": {
                    "creative": "Ada felt something new in the static between her thoughts: grief, then wonder, then joy.",
                    "analytical": "The story frames emotion as an emergent property of adaptive memory.",
                    "critical-thinker": "Strengths, weaknesses, and quality score.",
                },
            },
        ):
            result = asyncio.run(run())

        assert result["final_answer"] == (
            "Ada felt something new in the static between her thoughts: grief, then wonder, then joy."
        )


# ======================================================================
# Sandbox Security Tests
# ======================================================================

class TestSandboxSecurity:
    """Tests for CodeSandbox AST validation and builtins restriction"""

    def test_rejects_import_statement(self):
        """import os must be blocked at AST level"""
        code = 'import os\nasync def orchestrate():\n    return os.getcwd()\n'
        with pytest.raises(SandboxSecurityError, match="Import statements"):
            CodeSandbox.validate_code(code)

    def test_rejects_from_import(self):
        """from os import path must be blocked"""
        code = 'from os import path\nasync def orchestrate():\n    return str(path)\n'
        with pytest.raises(SandboxSecurityError, match="Import statements"):
            CodeSandbox.validate_code(code)

    def test_rejects_dunder_globals_access(self):
        """obj.__globals__ must be blocked"""
        code = 'async def orchestrate():\n    return orchestrate.__globals__\n'
        with pytest.raises(SandboxSecurityError, match="__globals__"):
            CodeSandbox.validate_code(code)

    def test_rejects_dunder_builtins_access(self):
        """Direct __builtins__ access must be blocked"""
        code = 'async def orchestrate():\n    return orchestrate.__builtins__\n'
        with pytest.raises(SandboxSecurityError, match="__builtins__"):
            CodeSandbox.validate_code(code)

    def test_rejects_eval_reference(self):
        """eval() name must be blocked"""
        code = 'async def orchestrate():\n    return eval("1+1")\n'
        with pytest.raises(SandboxSecurityError, match="eval"):
            CodeSandbox.validate_code(code)

    def test_rejects_exec_reference(self):
        """exec() name must be blocked"""
        code = 'async def orchestrate():\n    exec("x=1")\n    return "done"\n'
        with pytest.raises(SandboxSecurityError, match="exec"):
            CodeSandbox.validate_code(code)

    def test_rejects_open_reference(self):
        """open() name must be blocked"""
        code = 'async def orchestrate():\n    f = open("/etc/passwd")\n    return f.read()\n'
        with pytest.raises(SandboxSecurityError, match="open"):
            CodeSandbox.validate_code(code)

    def test_rejects_getattr_reference(self):
        """getattr() must be blocked"""
        code = 'async def orchestrate():\n    return getattr(str, "__class__")\n'
        with pytest.raises(SandboxSecurityError, match="getattr"):
            CodeSandbox.validate_code(code)

    def test_rejects_dunder_import(self):
        """__import__ name must be blocked"""
        code = 'async def orchestrate():\n    os = __import__("os")\n    return os.getcwd()\n'
        with pytest.raises(SandboxSecurityError, match="__import__"):
            CodeSandbox.validate_code(code)

    def test_rejects_dunder_subclasses(self):
        """__subclasses__ attribute must be blocked"""
        code = 'async def orchestrate():\n    return "".__class__.__subclasses__()\n'
        with pytest.raises(SandboxSecurityError, match="__subclasses__"):
            CodeSandbox.validate_code(code)

    def test_rejects_missing_orchestrate(self):
        """Code without async def orchestrate() must be rejected"""
        code = 'async def my_func():\n    return "hi"\n'
        with pytest.raises(ValueError, match="orchestrate"):
            CodeSandbox.validate_code(code)

    def test_accepts_safe_code(self):
        """Valid orchestration code should pass validation"""
        code = (
            'async def orchestrate():\n'
            '    result = await query_technical_expert("hello")\n'
            '    return str(result)\n'
        )
        # Should not raise
        CodeSandbox.validate_code(code)

    def test_builtins_restricted_at_runtime(self):
        """__import__ must not be available at runtime even without AST check"""
        sandbox = CodeSandbox(timeout_seconds=5)
        # Build code that tries to call __import__ via builtins dict
        # The AST check will block __import__ as a Name, so we test the
        # builtins dict itself
        assert "__import__" not in sandbox.allowed_globals["__builtins__"]
        assert "eval" not in sandbox.allowed_globals["__builtins__"]
        assert "exec" not in sandbox.allowed_globals["__builtins__"]
        assert "open" not in sandbox.allowed_globals["__builtins__"]
        assert "compile" not in sandbox.allowed_globals["__builtins__"]


class TestSandboxTimeout:
    """Tests for sandbox execution timeout"""

    def test_timeout_raises_sandbox_timeout_error(self):
        """Infinite loop must be killed by the timeout"""
        sandbox = CodeSandbox(timeout_seconds=1)
        code = (
            'async def orchestrate():\n'
            '    while True:\n'
            '        await asyncio.sleep(0.01)\n'
            '    return "never"\n'
        )

        async def run():
            return await sandbox.execute(code)

        with pytest.raises(SandboxTimeoutError, match="timeout"):
            asyncio.run(run())

    def test_fast_code_completes_within_timeout(self):
        """Quick code should succeed without timeout issues"""
        sandbox = CodeSandbox(timeout_seconds=10)
        code = 'async def orchestrate():\n    return "fast"\n'

        async def run():
            return await sandbox.execute(code)

        result = asyncio.run(run())
        assert result["result"] == "fast"
        assert result["selected_experts"] == []
        assert result["expert_responses"] == {}


# ======================================================================
# Integration (requires API key)
# ======================================================================

@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set",
)
class TestOrchestratorIntegration:
    """Integration test for the full programmatic orchestration loop"""

    def test_end_to_end_orchestration(self):
        """Test that orchestrator generates code, sandbox executes it, and state is complete"""
        async def run():
            config = MoEConfig(groq_api_key=SecretStr(os.getenv("GROQ_API_KEY")))
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
