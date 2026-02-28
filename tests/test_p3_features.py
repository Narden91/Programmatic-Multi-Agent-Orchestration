"""Tests for P3 features: tracing, multi-provider, memory, benchmarks."""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# ======================================================================
# P3.1 — Tracing
# ======================================================================

from src.utils.tracing import (
    Tracer,
    TraceEvent,
    TraceKind,
    get_tracer,
    reset_tracer,
)


class TestTraceEvent:
    def test_to_dict(self):
        ev = TraceEvent(kind="test.event", agent="A", data={"x": 1})
        d = ev.to_dict()
        assert d["kind"] == "test.event"
        assert d["agent"] == "A"
        assert d["data"] == {"x": 1}
        assert "timestamp" in d

    def test_frozen(self):
        ev = TraceEvent(kind="x")
        with pytest.raises(AttributeError):
            ev.kind = "y"  # type: ignore[misc]


class TestTracer:
    def test_emit_sync_records_history(self):
        tracer = Tracer()
        ev = TraceEvent(kind="a")
        tracer.emit_sync(ev)
        assert len(tracer.history) == 1
        assert tracer.history[0].kind == "a"

    @pytest.mark.asyncio
    async def test_emit_async(self):
        tracer = Tracer()
        await tracer.emit(TraceEvent(kind="b", agent="X"))
        assert len(tracer.history) == 1

    @pytest.mark.asyncio
    async def test_subscribe(self):
        tracer = Tracer()
        received = []

        async def consumer():
            async for ev in tracer.subscribe():
                received.append(ev)

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        await tracer.emit(TraceEvent(kind="e1"))
        await tracer.emit(TraceEvent(kind="e2"))
        await asyncio.sleep(0.01)
        await tracer.close()
        await task

        assert len(received) == 2
        assert received[0].kind == "e1"
        assert received[1].kind == "e2"

    def test_reset(self):
        tracer = Tracer()
        tracer.emit_sync(TraceEvent(kind="x"))
        tracer.reset()
        assert tracer.history == []

    @pytest.mark.asyncio
    async def test_close_flag(self):
        tracer = Tracer()
        assert not tracer.closed
        await tracer.close()
        assert tracer.closed


class TestTracerSingleton:
    def test_get_tracer_returns_same_instance(self):
        reset_tracer()
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_reset_tracer_creates_new(self):
        t1 = get_tracer()
        reset_tracer()
        t2 = get_tracer()
        assert t1 is not t2


class TestTraceKind:
    def test_enum_values(self):
        assert TraceKind.ORCHESTRATOR_START.value == "orchestrator.start"
        assert TraceKind.SANDBOX_SUCCESS.value == "sandbox.success"
        assert TraceKind.EXPERT_CALL_END.value == "expert.call_end"


# ======================================================================
# P3.2 — Multi-Provider
# ======================================================================

from src.llm.providers import (
    LLMProvider,
    GroqProvider,
    OpenAIProvider,
    AnthropicProvider,
    LLMFactory,
)
from src.core.config import LLMConfig, ExpertConfig


class TestLLMFactory:
    def test_available_providers(self):
        providers = LLMFactory.available_providers()
        assert "groq" in providers
        assert "openai" in providers
        assert "anthropic" in providers

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMFactory.create_provider("nonexistent", "key", LLMConfig("m"))

    def test_register_custom_provider(self):
        class DummyProvider(LLMProvider):
            provider_name = "dummy"
            def __init__(self, api_key, config):
                self.model_name = config.model_name
            def invoke(self, prompt): return "ok"
            async def ainvoke(self, prompt): return "ok"

        LLMFactory.register_provider("dummy", DummyProvider)
        assert "dummy" in LLMFactory.available_providers()
        p = LLMFactory.create_provider("dummy", "k", LLMConfig("dm"))
        assert p.model_name == "dm"

        # Cleanup
        del LLMFactory._providers["dummy"]

    def test_openai_import_error(self):
        """OpenAIProvider raises helpful ImportError when langchain-openai missing."""
        with patch.dict("sys.modules", {"langchain_openai": None}):
            with pytest.raises(ImportError, match="langchain-openai"):
                OpenAIProvider("key", LLMConfig("gpt-4o"))

    def test_anthropic_import_error(self):
        """AnthropicProvider raises helpful ImportError when langchain-anthropic missing."""
        with patch.dict("sys.modules", {"langchain_anthropic": None}):
            with pytest.raises(ImportError, match="langchain-anthropic"):
                AnthropicProvider("key", LLMConfig("claude-3-5-sonnet"))


class TestExpertConfigProviderType:
    def test_default_provider_type_empty(self):
        ec = ExpertConfig(
            name="test", description="d",
            llm_config=LLMConfig("m"), system_prompt="s",
        )
        assert ec.provider_type == ""

    def test_custom_provider_type(self):
        ec = ExpertConfig(
            name="test", description="d",
            llm_config=LLMConfig("gpt-4o"), system_prompt="s",
            provider_type="openai",
        )
        assert ec.provider_type == "openai"


# ======================================================================
# P3.3 — Multi-Turn Memory
# ======================================================================

from src.utils.memory import ConversationMemory, Turn


class TestTurn:
    def test_to_dict_roundtrip(self):
        t = Turn(query="q", answer="a", experts_used=["technical"])
        d = t.to_dict()
        t2 = Turn.from_dict(d)
        assert t2.query == "q"
        assert t2.answer == "a"
        assert t2.experts_used == ["technical"]


class TestConversationMemory:
    def test_add_and_len(self):
        mem = ConversationMemory(max_turns=5)
        mem.add("q1", "a1")
        mem.add("q2", "a2")
        assert len(mem) == 2

    def test_sliding_window(self):
        mem = ConversationMemory(max_turns=2)
        mem.add("q1", "a1")
        mem.add("q2", "a2")
        mem.add("q3", "a3")
        assert len(mem) == 2
        assert mem.turns[0].query == "q2"
        assert mem.turns[1].query == "q3"

    def test_last_turn(self):
        mem = ConversationMemory()
        assert mem.last_turn is None
        mem.add("q", "a")
        assert mem.last_turn is not None
        assert mem.last_turn.query == "q"

    def test_format_context_empty(self):
        mem = ConversationMemory()
        assert mem.format_context() == ""

    def test_format_context_nonempty(self):
        mem = ConversationMemory()
        mem.add("What is Python?", "A programming language.")
        ctx = mem.format_context()
        assert "Conversation History" in ctx
        assert "What is Python?" in ctx
        assert "programming language" in ctx

    def test_clear(self):
        mem = ConversationMemory()
        mem.add("q", "a")
        mem.clear()
        assert len(mem) == 0

    def test_persistence(self, tmp_path):
        path = tmp_path / "conv.json"
        mem = ConversationMemory(persist_path=path)
        mem.add("q1", "a1", experts_used=["technical"])
        mem.add("q2", "a2")

        # Reload
        mem2 = ConversationMemory(persist_path=path)
        assert len(mem2) == 2
        assert mem2.turns[0].query == "q1"
        assert mem2.turns[0].experts_used == ["technical"]

    def test_persistence_handles_corrupt(self, tmp_path):
        path = tmp_path / "conv.json"
        path.write_text("{{not json")
        mem = ConversationMemory(persist_path=path)
        assert len(mem) == 0

    def test_format_context_max_chars(self):
        mem = ConversationMemory()
        for i in range(50):
            mem.add(f"Question {i}", "A" * 200)
        ctx = mem.format_context(max_chars=500)
        assert len(ctx) <= 600  # +header overhead


# ======================================================================
# P3.4 — Benchmark Suite
# ======================================================================

from benchmarks.suite import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkReport,
    BenchmarkSuite,
    STANDARD_CASES,
    create_standard_suite,
)


class TestBenchmarkCase:
    def test_matches_filter_by_name(self):
        bc = BenchmarkCase(name="single_technical", query="x")
        assert bc.matches_filter("technical")
        assert not bc.matches_filter("creative")

    def test_matches_filter_by_tag(self):
        bc = BenchmarkCase(name="x", query="y", tags=["routing"])
        assert bc.matches_filter("routing")

    def test_matches_filter_by_query(self):
        bc = BenchmarkCase(name="x", query="explain GIL")
        assert bc.matches_filter("GIL")


class TestBenchmarkReport:
    def _make_result(self, success=True, expected=None, used=None):
        return BenchmarkResult(
            case=BenchmarkCase(
                name="t", query="q",
                expected_experts=expected or [],
            ),
            success=success,
            elapsed_seconds=1.0,
            experts_used=used or [],
        )

    def test_counts(self):
        r = BenchmarkReport(results=[
            self._make_result(success=True),
            self._make_result(success=False),
            self._make_result(success=True),
        ])
        assert r.passed == 2
        assert r.failed == 1

    def test_expert_accuracy(self):
        r = BenchmarkReport(results=[
            self._make_result(expected=["technical"], used=["technical", "general"]),
            self._make_result(expected=["creative"], used=["general"]),
        ])
        assert r.expert_accuracy == 50.0

    def test_summary(self):
        r = BenchmarkReport(results=[self._make_result()])
        s = r.summary()
        assert s["total_cases"] == 1
        assert "per_case" in s

    def test_pretty_print(self):
        r = BenchmarkReport(results=[self._make_result()])
        text = r.pretty_print()
        assert "BENCHMARK REPORT" in text
        assert "PASS" in text


class TestBenchmarkSuite:
    def test_add_and_cases(self):
        suite = BenchmarkSuite()
        suite.add(BenchmarkCase(name="a", query="q"))
        assert len(suite.cases) == 1

    @pytest.mark.asyncio
    async def test_run_all_with_mock_graph(self):
        """Run the suite against a mock graph that always succeeds."""
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "final_answer": "test answer",
            "selected_experts": ["technical"],
            "token_usage": {"total_tokens": 100},
        }

        suite = BenchmarkSuite()
        suite.add(BenchmarkCase(
            name="test", query="q",
            expected_experts=["technical"],
        ))
        report = await suite.run_all(mock_graph)
        assert report.passed == 1
        assert report.failed == 0

    @pytest.mark.asyncio
    async def test_run_all_handles_exception(self):
        mock_graph = AsyncMock()
        mock_graph.ainvoke.side_effect = RuntimeError("boom")

        suite = BenchmarkSuite()
        suite.add(BenchmarkCase(name="fail", query="q"))
        report = await suite.run_all(mock_graph)
        assert report.failed == 1
        assert "boom" in report.results[0].error

    @pytest.mark.asyncio
    async def test_filter(self):
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "final_answer": "ok",
            "selected_experts": [],
            "token_usage": {},
        }

        suite = BenchmarkSuite()
        suite.add(BenchmarkCase(name="alpha", query="q"))
        suite.add(BenchmarkCase(name="beta", query="q"))
        report = await suite.run_all(mock_graph, filter_pattern="alpha")
        assert len(report.results) == 1
        assert report.results[0].case.name == "alpha"


class TestStandardSuite:
    def test_standard_cases_not_empty(self):
        assert len(STANDARD_CASES) >= 5

    def test_create_standard_suite(self):
        suite = create_standard_suite()
        assert len(suite.cases) == len(STANDARD_CASES)


# ======================================================================
# P3 integration: state has conversation_context field
# ======================================================================

from src.core.state import create_initial_state


class TestStateConversationContext:
    def test_initial_state_has_conversation_context(self):
        state = create_initial_state("test")
        assert "conversation_context" in state
        assert state["conversation_context"] == ""


# ======================================================================
# P3 integration: orchestrator prompt includes context
# ======================================================================

from src.llm.prompts import OrchestratorPrompts


class TestOrchestratorPromptContext:
    def test_prompt_without_context(self):
        p = OrchestratorPrompts.create_orchestration_prompt(
            query="test", available_experts=["general"],
        )
        assert "Conversation History" not in p

    def test_prompt_with_context(self):
        p = OrchestratorPrompts.create_orchestration_prompt(
            query="test",
            available_experts=["general"],
            conversation_context="## Conversation History\n**User:** hi\n**Answer:** hello",
        )
        assert "Conversation History" in p
        assert "Take the conversation history" in p
