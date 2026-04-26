"""CLI entry-point for the benchmark suite.

Usage::

    python -m benchmarks.run
    python -m benchmarks.run --filter routing
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import sys

from benchmarks.suite import create_standard_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MoE benchmark suite")
    parser.add_argument(
        "--filter", type=str, default="",
        help="Substring filter for benchmark names / tags",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override LLM model name",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="How many times to run each benchmark case",
    )
    parser.add_argument(
        "--selection-bias-slice",
        action="store_true",
        help="Compare baseline routing against atom few-shot + metadata-biased candidate selection",
    )
    args = parser.parse_args()

    # Late import so benchmarks module can be imported without side-effects
    from src.core.config import MoEConfig, SecretStr
    from src.graph.builder import MoEGraphBuilder

    cfg = MoEConfig()
    if args.model:
        cfg.orchestrator_config.model_name = args.model

    suite = create_standard_suite()
    repeats = max(args.repeats, 1)

    if args.selection_bias_slice:
        baseline_cfg = deepcopy(cfg)
        baseline_cfg.enable_atom_few_shot_retrieval = False
        baseline_cfg.enable_metadata_selection_bias = False
        baseline_cfg.orchestrator_candidate_count = max(baseline_cfg.orchestrator_candidate_count, 2)

        biased_cfg = deepcopy(cfg)
        biased_cfg.enable_atom_few_shot_retrieval = True
        biased_cfg.enable_metadata_selection_bias = True
        biased_cfg.orchestrator_candidate_count = max(biased_cfg.orchestrator_candidate_count, 2)

        comparison = asyncio.run(
            suite.run_variant_slice(
                {
                    "baseline": MoEGraphBuilder(baseline_cfg).build(),
                    "metadata_bias": MoEGraphBuilder(biased_cfg).build(),
                },
                filter_pattern=args.filter,
                repeats=repeats,
            )
        )
        print(comparison.pretty_print())
        any_failures = any(variant.report.failed for variant in comparison.variants)
        sys.exit(0 if not any_failures else 1)

    builder = MoEGraphBuilder(cfg)
    graph = builder.build()
    report = asyncio.run(
        suite.run_all(
            graph,
            filter_pattern=args.filter,
            repeats=repeats,
        )
    )

    print(report.pretty_print())

    # Exit with non-zero if any failures
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
