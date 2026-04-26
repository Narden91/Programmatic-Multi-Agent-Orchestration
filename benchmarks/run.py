"""CLI entry-point for the benchmark suite.

Usage::

    python -m benchmarks.run
    python -m benchmarks.run --filter routing
"""

from __future__ import annotations

import argparse
import asyncio
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
    args = parser.parse_args()

    # Late import so benchmarks module can be imported without side-effects
    from src.core.config import MoEConfig, SecretStr
    from src.graph.builder import MoEGraphBuilder

    cfg = MoEConfig()
    if args.model:
        cfg.orchestrator_config.model_name = args.model

    builder = MoEGraphBuilder(cfg)
    graph = builder.build()

    suite = create_standard_suite()
    report = asyncio.run(
        suite.run_all(
            graph,
            filter_pattern=args.filter,
            repeats=max(args.repeats, 1),
        )
    )

    print(report.pretty_print())

    # Exit with non-zero if any failures
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
