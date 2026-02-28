"""
Tests for P2 features: ExpertRegistry, ScriptBank, code_analyzer, TokenTracker.
"""

import asyncio
import json
import os
import tempfile

import pytest
from src.agents.registry import ExpertRegistry, ExpertSpec, registry
from src.utils.script_bank import ScriptBank, ScriptRecord
from src.utils.code_analyzer import analyze_code, ExecutionPlan, ExpertCall
from src.utils.metrics import TokenTracker, TokenRecord, reset_token_tracker, get_token_tracker


# ======================================================================
# ExpertRegistry
# ======================================================================

class TestExpertRegistry:

    def test_default_experts_registered(self):
        """The four built-in experts should be pre-registered."""
        assert "technical" in registry
        assert "creative" in registry
        assert "analytical" in registry
        assert "general" in registry
        assert len(registry) == 4

    def test_register_and_unregister(self):
        r = ExpertRegistry()
        r.register(
            expert_type="legal",
            description="law and compliance",
            system_prompt="You are a legal expert.",
            prompt_template='Legal query: "{query}"',
        )
        assert "legal" in r
        assert len(r) == 1
        assert r.types == ["legal"]
        r.unregister("legal")
        assert "legal" not in r
        assert len(r) == 0

    def test_create_prompt(self):
        r = ExpertRegistry()
        r.register(
            expert_type="test",
            description="testing",
            system_prompt="sys",
            prompt_template='Prompt for: "{query}"',
        )
        result = r.create_prompt("test", "hello world")
        assert result == 'Prompt for: "hello world"'

    def test_create_prompt_unknown_type_raises(self):
        r = ExpertRegistry()
        with pytest.raises(ValueError, match="Unknown expert type"):
            r.create_prompt("nonexistent", "hello")

    def test_descriptions(self):
        r = ExpertRegistry()
        r.register("a", "desc-a", "sys", "tmpl")
        r.register("b", "desc-b", "sys", "tmpl")
        assert r.descriptions() == {"a": "desc-a", "b": "desc-b"}

    def test_get_returns_spec(self):
        spec = registry.get("technical")
        assert isinstance(spec, ExpertSpec)
        assert spec.expert_type == "technical"
        assert spec.confidence_threshold == 0.85

    def test_overwrite_on_re_register(self):
        r = ExpertRegistry()
        r.register("x", "old", "sys", "tmpl", 0.5)
        r.register("x", "new", "sys2", "tmpl2", 0.9)
        assert r.get("x").description == "new"
        assert r.get("x").confidence_threshold == 0.9


# ======================================================================
# ScriptBank
# ======================================================================

class TestScriptBank:

    def test_record_and_size(self):
        bank = ScriptBank()
        bank.record("q1", "code1", ["technical"], True)
        bank.record("q2", "code2", ["creative"], False)
        assert bank.size == 2

    def test_find_similar(self):
        bank = ScriptBank()
        bank.record("Explain quicksort algorithm", "code1", ["technical"], True)
        bank.record("Write a haiku about spring", "code2", ["creative"], True)
        bank.record("Describe merge sort algorithm", "code3", ["technical"], True)

        results = bank.find_similar("How does quicksort work?", top_k=2)
        # "quicksort" and "algorithm" overlap → code1 should be top
        assert len(results) >= 1
        assert results[0].code == "code1"

    def test_find_similar_only_success(self):
        bank = ScriptBank()
        bank.record("Explain recursion in Python", "bad_code", ["technical"], False)
        bank.record("Explain recursion in Python", "good_code", ["technical"], True)

        results = bank.find_similar("Explain recursion", only_success=True)
        assert all(r.success for r in results)
        assert results[0].code == "good_code"

    def test_max_size_eviction(self):
        bank = ScriptBank(max_size=3)
        for i in range(5):
            bank.record(f"query {i}", f"code_{i}", [], True)
        assert bank.size == 3

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            bank1 = ScriptBank(persist_path=path)
            bank1.record("explain recursion", "code_recurse", ["tech"], True)
            assert bank1.size == 1

            # Reload from disk
            bank2 = ScriptBank(persist_path=path)
            assert bank2.size == 1
            assert bank2.find_similar("explain recursion")[0].code == "code_recurse"
        finally:
            os.unlink(path)

    def test_clear(self):
        bank = ScriptBank()
        bank.record("q", "c", [], True)
        bank.clear()
        assert bank.size == 0


# ======================================================================
# Code Analyzer
# ======================================================================

class TestCodeAnalyzer:

    def test_sequential_calls(self):
        code = (
            'async def orchestrate():\n'
            '    a = await query_technical_expert("hi")\n'
            '    b = await query_creative_expert("hi")\n'
            '    return a + b\n'
        )
        plan = analyze_code(code)
        assert plan.has_sequential
        assert not plan.has_parallel
        assert plan.experts_used == ["technical", "creative"]
        assert len(plan.sequential_calls) == 2

    def test_parallel_calls(self):
        code = (
            'async def orchestrate():\n'
            '    a, b = await asyncio.gather(\n'
            '        query_technical_expert("hi"),\n'
            '        query_analytical_expert("hi"),\n'
            '    )\n'
            '    return a + b\n'
        )
        plan = analyze_code(code)
        assert plan.has_parallel
        assert plan.gather_groups == 1
        assert len(plan.parallel_calls) == 2
        assert set(plan.experts_used) == {"technical", "analytical"}

    def test_mixed_sequential_and_parallel(self):
        code = (
            'async def orchestrate():\n'
            '    a, b = await asyncio.gather(\n'
            '        query_technical_expert("hi"),\n'
            '        query_creative_expert("hi"),\n'
            '    )\n'
            '    c = await query_general_expert(a + b)\n'
            '    return c\n'
        )
        plan = analyze_code(code)
        assert plan.has_parallel
        assert plan.has_sequential
        assert plan.gather_groups == 1
        assert len(plan.calls) == 3

    def test_syntax_error_returns_empty_plan(self):
        plan = analyze_code("this is not python {{{{")
        assert plan.calls == []
        assert not plan.has_parallel
        assert not plan.has_sequential

    def test_no_expert_calls(self):
        code = 'async def orchestrate():\n    return "hello"\n'
        plan = analyze_code(code)
        assert plan.experts_used == []

    def test_to_dict_serialisable(self):
        code = (
            'async def orchestrate():\n'
            '    r = await query_general_expert("hi")\n'
            '    return r\n'
        )
        plan = analyze_code(code)
        d = plan.to_dict()
        assert isinstance(d, dict)
        assert d["experts_used"] == ["general"]
        # Must be JSON-serialisable
        assert json.dumps(d)


# ======================================================================
# TokenTracker
# ======================================================================

class TestTokenTracker:

    def test_record_and_summary(self):
        tracker = TokenTracker()
        tracker.record("Orchestrator", "llama-3.3-70b-versatile", 100, 50)
        tracker.record("Technical Expert", "llama-3.3-70b-versatile", 200, 80)

        assert tracker.total_input_tokens == 300
        assert tracker.total_output_tokens == 130
        assert tracker.total_tokens == 430

        s = tracker.summary()
        assert s["total_tokens"] == 430
        assert len(s["by_agent"]) == 2
        assert s["by_agent"]["Orchestrator"]["calls"] == 1

    def test_cost_estimation(self):
        tracker = TokenTracker()
        tracker.record("A", "llama-3.3-70b-versatile", 1_000_000, 1_000_000)
        # Expected: 1M * 0.59/1M + 1M * 0.79/1M = 1.38
        assert abs(tracker.total_cost - 1.38) < 0.01

    def test_unknown_model_zero_cost(self):
        tracker = TokenTracker()
        tracker.record("A", "unknown-model", 1000, 1000)
        assert tracker.total_cost == 0.0

    def test_set_pricing(self):
        tracker = TokenTracker()
        tracker.set_pricing("my-model", 1.0, 2.0)
        tracker.record("A", "my-model", 1_000_000, 1_000_000)
        assert abs(tracker.total_cost - 3.0) < 0.01

    def test_record_from_response_with_proper_metadata(self):
        tracker = TokenTracker()

        class FakeResponse:
            response_metadata = {
                "token_usage": {
                    "prompt_tokens": 42,
                    "completion_tokens": 18,
                }
            }

        tracker.record_from_response("Agent", "model", FakeResponse())
        assert tracker.total_input_tokens == 42
        assert tracker.total_output_tokens == 18

    def test_record_from_response_with_mock_is_noop(self):
        """Mocked responses should not crash the tracker."""
        from unittest.mock import Mock

        tracker = TokenTracker()
        tracker.record_from_response("Agent", "model", Mock())
        assert tracker.total_tokens == 0

    def test_reset(self):
        tracker = TokenTracker()
        tracker.record("A", "m", 100, 50)
        tracker.reset()
        assert tracker.total_tokens == 0

    def test_module_level_singleton(self):
        t1 = get_token_tracker()
        t2 = get_token_tracker()
        assert t1 is t2

    def test_reset_token_tracker_returns_fresh(self):
        old = get_token_tracker()
        old.record("X", "m", 1, 1)
        new = reset_token_tracker()
        assert new is not old
        assert new.total_tokens == 0


# ======================================================================
# Dynamic expert registration (integration)
# ======================================================================

class TestDynamicExpertRegistration:

    def test_register_expert_adds_to_registry_and_config(self):
        from src.agents.tools import register_expert
        from src.agents.registry import registry as r
        from src.core.config import config

        register_expert(
            expert_type="_test_dynamic",
            description="test description",
            prompt_template='Test: "{query}"',
        )

        try:
            assert "_test_dynamic" in r
            assert "_test_dynamic" in config.expert_configs
            assert r.get("_test_dynamic").description == "test description"
        finally:
            # Clean up
            r.unregister("_test_dynamic")
            config.expert_configs.pop("_test_dynamic", None)
