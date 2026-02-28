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

    def summary(self) -> Dict[str, Any]:
        return {
            "total_cases": len(self.results),
            "passed": self.passed,
            "failed": self.failed,
            "expert_accuracy_pct": round(self.expert_accuracy, 1),
            "total_elapsed_seconds": round(self.total_elapsed, 2),
            "per_case": [
                {
                    "name": r.case.name,
                    "success": r.success,
                    "elapsed": round(r.elapsed_seconds, 2),
                    "experts_used": r.experts_used,
                    "expected_experts": r.case.expected_experts,
                    "tokens": r.token_summary.get("total_tokens", 0),
                    "error": r.error[:200] if r.error else "",
                }
                for r in self.results
            ],
        }

    def pretty_print(self) -> str:
        lines = [
            "=" * 60,
            " BENCHMARK REPORT",
            "=" * 60,
            f"  Total cases   : {len(self.results)}",
            f"  Passed        : {self.passed}",
            f"  Failed        : {self.failed}",
            f"  Expert acc.   : {self.expert_accuracy:.1f}%",
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

        report = BenchmarkReport()
        suite_start = time.time()

        for case in cases:
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
                ))
            except Exception as exc:
                elapsed = time.time() - t0
                report.results.append(BenchmarkResult(
                    case=case,
                    success=False,
                    elapsed_seconds=elapsed,
                    error=str(exc),
                ))

        report.total_elapsed = time.time() - suite_start
        return report


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
