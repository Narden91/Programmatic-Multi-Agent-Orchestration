import asyncio
from unittest.mock import AsyncMock

import pytest

from benchmarks.suite import BenchmarkCase, BenchmarkSuite
from src.core.config import MoEConfig, SecretStr, apply_model_override


@pytest.mark.asyncio
async def test_run_all_supports_repeats():
    graph = AsyncMock()
    graph.ainvoke.return_value = {
        "final_answer": "ok",
        "selected_experts": ["technical"],
        "token_usage": {"total_tokens": 42},
        "metadata": {"retrieval": {"neighborhood_reuse_rate": 0.5}},
    }

    suite = BenchmarkSuite()
    suite.add(BenchmarkCase(name="repeat_case", query="Explain caching"))

    report = await suite.run_all(graph, repeats=3)

    assert len(report.results) == 3
    assert [r.repeat_index for r in report.results] == [1, 2, 3]

    summary = report.summary()
    by_case = summary["by_case"]
    assert len(by_case) == 1
    assert by_case[0]["name"] == "repeat_case"
    assert by_case[0]["runs"] == 3
    assert summary["neighborhood_reuse_rate_mean"] == 0.5


def test_pretty_print_shows_aggregates_for_repeats():
    from benchmarks.suite import BenchmarkReport, BenchmarkResult

    case = BenchmarkCase(name="x", query="q")
    report = BenchmarkReport(results=[
        BenchmarkResult(case=case, success=True, elapsed_seconds=1.0, repeat_index=1),
        BenchmarkResult(case=case, success=True, elapsed_seconds=2.0, repeat_index=2),
    ])

    text = report.pretty_print()

    assert "AGGREGATES (BY CASE)" in text
    assert "runs=2" in text


@pytest.mark.asyncio
async def test_run_variant_slice_compares_success_retries_and_tokens():
    baseline = AsyncMock()
    baseline.ainvoke.return_value = {
        "final_answer": "ok",
        "selected_experts": ["technical"],
        "token_usage": {"total_tokens": 120},
        "code_execution_iterations": 2,
        "metadata": {"retrieval": {"neighborhood_reuse_rate": 0.0}},
    }

    metadata_bias = AsyncMock()
    metadata_bias.ainvoke.return_value = {
        "final_answer": "ok",
        "selected_experts": ["technical"],
        "token_usage": {"total_tokens": 80},
        "code_execution_iterations": 1,
        "metadata": {"retrieval": {"neighborhood_reuse_rate": 1.0}},
    }

    suite = BenchmarkSuite()
    suite.add(BenchmarkCase(name="slice_case", query="Explain caching"))

    comparison = await suite.run_variant_slice(
        {"baseline": baseline, "metadata_bias": metadata_bias},
        repeats=2,
    )

    summary = comparison.summary()
    assert summary["variants"][0]["metrics"]["retries_mean"] == 1
    assert summary["variants"][1]["metrics"]["tokens_mean"] == 80.0
    assert summary["deltas"][0]["delta_retries_mean"] == -1
    assert summary["deltas"][0]["delta_tokens_mean"] == -40.0
    assert summary["deltas"][0]["delta_neighborhood_reuse_rate_mean"] == 1.0

    text = comparison.pretty_print()
    assert "BENCHMARK SLICE REPORT" in text
    assert "DELTAS VS BASELINE" in text


def test_pretty_print_shows_family_aggregates_when_present():
    from benchmarks.suite import BenchmarkReport, BenchmarkResult

    case = BenchmarkCase(name="warm_case", query="q", family="search")
    report = BenchmarkReport(results=[
        BenchmarkResult(case=case, success=True, elapsed_seconds=1.0, neighborhood_reuse_rate=1.0),
    ])

    text = report.pretty_print()

    assert "FAMILY AGGREGATES" in text


@pytest.mark.asyncio
async def test_run_all_records_cancelled_runs_as_failures():
    graph = AsyncMock()
    graph.ainvoke.side_effect = asyncio.CancelledError("rate limit")

    suite = BenchmarkSuite()
    suite.add(BenchmarkCase(name="cancelled_case", query="Explain caching"))

    report = await suite.run_all(graph)

    assert len(report.results) == 1
    assert report.results[0].success is False
    assert "cancelled" in report.results[0].error


def test_apply_model_override_updates_all_expert_models():
    cfg = MoEConfig(groq_api_key=SecretStr("fake-key"))

    apply_model_override(cfg, "llama-3.1-8b-instant")

    assert cfg.orchestrator_config.model_name == "llama-3.1-8b-instant"
    assert all(
        expert.llm_config.model_name == "llama-3.1-8b-instant"
        for expert in cfg.expert_configs.values()
    )
