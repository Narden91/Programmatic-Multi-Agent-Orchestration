"""Benchmark harness for the MoE pipeline.

Runs a set of standardised queries, records timing / token usage, and
produces a structured report.  Designed to be executed from the CLI::

    python -m benchmarks.run            # run all benchmarks
    python -m benchmarks.run --filter routing   # substring filter

Or imported programmatically::

    from benchmarks.suite import BenchmarkSuite, BenchmarkCase
    suite = BenchmarkSuite()
    suite.add(BenchmarkCase(query="…", expected_experts=["technical"]))
    report = await suite.run_all(graph)
"""

from __future__ import annotations

import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.metrics import get_token_tracker, reset_token_tracker


# ======================================================================
# Data model
# ======================================================================

@dataclass
class BenchmarkCase:
    """A single benchmark scenario."""

    name: str
    query: str
    expected_experts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def matches_filter(self, pattern: str) -> bool:
        pattern_lower = pattern.lower()
        return (
            pattern_lower in self.name.lower()
            or pattern_lower in self.query.lower()
            or any(pattern_lower in t.lower() for t in self.tags)
        )


@dataclass
class BenchmarkResult:
    """Result of running a single :class:`BenchmarkCase`."""

    case: BenchmarkCase
    success: bool
    elapsed_seconds: float
    token_summary: Dict[str, Any] = field(default_factory=dict)
    experts_used: List[str] = field(default_factory=list)
    answer_snippet: str = ""
    error: str = ""
    repeat_index: int = 1
    retry_count: int = 0


@dataclass
class BenchmarkReport:
    """Aggregated report across all benchmark cases."""

    results: List[BenchmarkResult] = field(default_factory=list)
    total_elapsed: float = 0.0

    # ---- Derived stats -----------------------------------------------

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    @property
    def expert_accuracy(self) -> float:
        """% of cases where the actual experts matched expectations."""
        if not self.results:
            return 0.0
        correct = sum(
            1 for r in self.results
            if set(r.case.expected_experts) <= set(r.experts_used)
        )
        return correct / len(self.results) * 100

    @property
    def success_rate_pct(self) -> float:
        if not self.results:
            return 0.0
        return self.passed / len(self.results) * 100

    @property
    def mean_tokens(self) -> float:
        tokens = [
            int(r.token_summary.get("total_tokens", 0))
            for r in self.results
            if r.token_summary
        ]
        return statistics.mean(tokens) if tokens else 0.0

    @property
    def mean_retries(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.retry_count for r in self.results)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_cases": len(self.results),
            "passed": self.passed,
            "failed": self.failed,
            "expert_accuracy_pct": round(self.expert_accuracy, 1),
            "success_rate_pct": round(self.success_rate_pct, 1),
            "retries_mean": round(self.mean_retries, 3),
            "tokens_mean": round(self.mean_tokens, 1),
            "total_elapsed_seconds": round(self.total_elapsed, 2),
            "by_case": self.case_aggregates(),
            "per_case": [
                {
                    "name": r.case.name,
                    "repeat_index": r.repeat_index,
                    "success": r.success,
                    "elapsed": round(r.elapsed_seconds, 2),
                    "experts_used": r.experts_used,
                    "expected_experts": r.case.expected_experts,
                    "tokens": r.token_summary.get("total_tokens", 0),
                    "retry_count": r.retry_count,
                    "error": r.error[:200] if r.error else "",
                }
                for r in self.results
            ],
        }

    def case_aggregates(self) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[BenchmarkResult]] = {}
        for result in self.results:
            grouped.setdefault(result.case.name, []).append(result)

        rows: List[Dict[str, Any]] = []
        for name, items in grouped.items():
            elapsed = [r.elapsed_seconds for r in items]
            tokens = [
                int(r.token_summary.get("total_tokens", 0))
                for r in items
                if r.token_summary
            ]
            pass_rate = sum(1 for r in items if r.success) / len(items) * 100
            rows.append({
                "name": name,
                "runs": len(items),
                "pass_rate_pct": round(pass_rate, 1),
                "elapsed_mean": round(statistics.mean(elapsed), 3),
                "elapsed_stdev": round(statistics.pstdev(elapsed), 3) if len(elapsed) > 1 else 0.0,
                "retries_mean": round(statistics.mean(r.retry_count for r in items), 3),
                "tokens_mean": round(statistics.mean(tokens), 1) if tokens else 0.0,
            })

        rows.sort(key=lambda r: r["name"])
        return rows

    def pretty_print(self) -> str:
        lines = [
            "=" * 60,
            " BENCHMARK REPORT",
            "=" * 60,
            f"  Total cases   : {len(self.results)}",
            f"  Passed        : {self.passed}",
            f"  Failed        : {self.failed}",
            f"  Success rate  : {self.success_rate_pct:.1f}%",
            f"  Expert acc.   : {self.expert_accuracy:.1f}%",
            f"  Mean retries  : {self.mean_retries:.2f}",
            f"  Mean tokens   : {self.mean_tokens:.1f}",
            f"  Total time    : {self.total_elapsed:.2f}s",
            "-" * 60,
        ]
        for r in self.results:
            mark = "PASS" if r.success else "FAIL"
            lines.append(
                f"  [{mark}] {r.case.name:30s}  "
                f"{r.elapsed_seconds:.2f}s  experts={r.experts_used}"
            )
            if r.error:
                lines.append(f"         ERROR: {r.error[:120]}")

        if self.results and max(r.repeat_index for r in self.results) > 1:
            lines.append("-" * 60)
            lines.append(" AGGREGATES (BY CASE)")
            for row in self.case_aggregates():
                lines.append(
                    "  {name:30s} runs={runs:<2d} pass={pass_rate_pct:5.1f}% "
                    "elapsed={elapsed_mean:.2f}s±{elapsed_stdev:.2f} retries={retries_mean:.2f} tokens={tokens_mean:.1f}".format(**row)
                )

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class BenchmarkVariantRun:
    name: str
    report: BenchmarkReport


@dataclass
class BenchmarkComparisonReport:
    variants: List[BenchmarkVariantRun] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        if not self.variants:
            return {"variants": [], "deltas": []}

        baseline = self.variants[0]
        baseline_metrics = {
            "success_rate_pct": baseline.report.success_rate_pct,
            "retries_mean": baseline.report.mean_retries,
            "tokens_mean": baseline.report.mean_tokens,
        }

        variants = []
        deltas = []
        for variant in self.variants:
            metrics = {
                "success_rate_pct": round(variant.report.success_rate_pct, 1),
                "retries_mean": round(variant.report.mean_retries, 3),
                "tokens_mean": round(variant.report.mean_tokens, 1),
                "failed": variant.report.failed,
            }
            variants.append({"name": variant.name, "metrics": metrics})
            if variant is baseline:
                continue
            deltas.append({
                "name": variant.name,
                "delta_success_rate_pct": round(variant.report.success_rate_pct - baseline_metrics["success_rate_pct"], 1),
                "delta_retries_mean": round(variant.report.mean_retries - baseline_metrics["retries_mean"], 3),
                "delta_tokens_mean": round(variant.report.mean_tokens - baseline_metrics["tokens_mean"], 1),
            })

        return {"variants": variants, "deltas": deltas}

    def pretty_print(self) -> str:
        lines = [
            "=" * 60,
            " BENCHMARK SLICE REPORT",
            "=" * 60,
        ]
        summary = self.summary()
        for item in summary["variants"]:
            metrics = item["metrics"]
            lines.append(
                "  {name:18s} success={success_rate_pct:5.1f}% retries={retries_mean:.2f} tokens={tokens_mean:.1f} failed={failed}".format(
                    name=item["name"],
                    **metrics,
                )
            )

        if summary["deltas"]:
            lines.append("-" * 60)
            lines.append(" DELTAS VS BASELINE")
            for delta in summary["deltas"]:
                lines.append(
                    "  {name:18s} success={delta_success_rate_pct:+.1f} retries={delta_retries_mean:+.2f} tokens={delta_tokens_mean:+.1f}".format(**delta)
                )

        lines.append("=" * 60)
        return "\n".join(lines)


# ======================================================================
# Suite runner
# ======================================================================

class BenchmarkSuite:
    """Collection of :class:`BenchmarkCase` objects with a runner."""

    def __init__(self) -> None:
        self._cases: List[BenchmarkCase] = []

    def add(self, case: BenchmarkCase) -> None:
        self._cases.append(case)

    def add_many(self, cases: List[BenchmarkCase]) -> None:
        self._cases.extend(cases)

    @property
    def cases(self) -> List[BenchmarkCase]:
        return list(self._cases)

    async def run_all(
        self,
        graph: Any,
        *,
        filter_pattern: str = "",
        repeats: int = 1,
    ) -> BenchmarkReport:
        """Execute every matching case against the compiled *graph*.

        Parameters
        ----------
        graph:
            A compiled LangGraph (the return of ``MoEGraphBuilder.build()``).
        filter_pattern:
            When non-empty, only cases matching this substring are run.
        """
        from src.core.state import create_initial_state

        cases = (
            [c for c in self._cases if c.matches_filter(filter_pattern)]
            if filter_pattern
            else self._cases
        )

        repeats = max(int(repeats), 1)

        report = BenchmarkReport()
        suite_start = time.time()

        for case in cases:
            for repeat_index in range(1, repeats + 1):
                reset_token_tracker()
                state = create_initial_state(case.query)
                t0 = time.time()
                try:
                    result_state = await graph.ainvoke(state)
                    elapsed = time.time() - t0
                    report.results.append(BenchmarkResult(
                        case=case,
                        success=True,
                        elapsed_seconds=elapsed,
                        token_summary=result_state.get("token_usage", {}),
                        experts_used=result_state.get("selected_experts", []),
                        answer_snippet=(result_state.get("final_answer", "")[:200]),
                        repeat_index=repeat_index,
                        retry_count=max(int(result_state.get("code_execution_iterations", 0)) - 1, 0),
                    ))
                except Exception as exc:
                    elapsed = time.time() - t0
                    report.results.append(BenchmarkResult(
                        case=case,
                        success=False,
                        elapsed_seconds=elapsed,
                        error=str(exc),
                        repeat_index=repeat_index,
                    ))

        report.total_elapsed = time.time() - suite_start
        return report

    async def run_variant_slice(
        self,
        graphs: Dict[str, Any],
        *,
        filter_pattern: str = "",
        repeats: int = 1,
    ) -> BenchmarkComparisonReport:
        variants: List[BenchmarkVariantRun] = []
        for name, graph in graphs.items():
            report = await self.run_all(graph, filter_pattern=filter_pattern, repeats=repeats)
            variants.append(BenchmarkVariantRun(name=name, report=report))
        return BenchmarkComparisonReport(variants=variants)


# ======================================================================
# Standard benchmark cases (curated set)
# ======================================================================

STANDARD_CASES: List[BenchmarkCase] = [
    BenchmarkCase(
        name="single_technical",
        query="Explain how Python's GIL works and its implications for multi-threading.",
        expected_experts=["technical"],
        tags=["single", "routing"],
    ),
    BenchmarkCase(
        name="single_creative",
        query="Write a short poem about the beauty of mathematics.",
        expected_experts=["creative"],
        tags=["single", "routing"],
    ),
    BenchmarkCase(
        name="single_analytical",
        query="Compare REST and GraphQL APIs: pros, cons, and when to use each.",
        expected_experts=["analytical"],
        tags=["single", "routing"],
    ),
    BenchmarkCase(
        name="multi_tech_analytical",
        query="Explain the technical architecture of a recommendation engine and analytically compare collaborative filtering vs content-based filtering.",
        expected_experts=["technical", "analytical"],
        tags=["multi", "routing", "parallel"],
    ),
    BenchmarkCase(
        name="multi_creative_general",
        query="Write a creative product pitch for a new AI coding assistant and provide general context about the AI tools market.",
        expected_experts=["creative", "general"],
        tags=["multi", "routing"],
    ),
    BenchmarkCase(
        name="all_experts",
        query="For a startup building an AI-powered education platform: provide technical architecture, creative branding ideas, analytical market comparison, and a general overview of the EdTech landscape.",
        expected_experts=["technical", "creative", "analytical", "general"],
        tags=["multi", "routing", "comprehensive"],
    ),
    BenchmarkCase(
        name="sequential_reasoning",
        query="Explain what a transformer architecture is, then analytically compare it to RNNs.",
        expected_experts=["technical", "analytical"],
        tags=["sequential", "reasoning"],
    ),
    BenchmarkCase(
        name="ambiguous_query",
        query="Tell me about Python.",
        expected_experts=["general"],
        tags=["ambiguous", "routing"],
    ),
]


def create_standard_suite() -> BenchmarkSuite:
    """Return a :class:`BenchmarkSuite` pre-loaded with standard cases."""
    suite = BenchmarkSuite()
    suite.add_many(STANDARD_CASES)
    return suite
