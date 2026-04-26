from unittest.mock import AsyncMock

import pytest

from benchmarks.suite import BenchmarkCase, BenchmarkSuite


@pytest.mark.asyncio
async def test_run_all_supports_repeats():
    graph = AsyncMock()
    graph.ainvoke.return_value = {
        "final_answer": "ok",
        "selected_experts": ["technical"],
        "token_usage": {"total_tokens": 42},
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
