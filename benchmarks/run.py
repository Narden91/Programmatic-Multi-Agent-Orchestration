"""CLI entry-point for the benchmark suite.

Usage::

    python -m benchmarks.run
    python -m benchmarks.run --filter routing
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

from benchmarks.suite import create_standard_suite


def _emit_benchmark_outputs(
    payload: dict,
    *,
    output_json: str | None,
    plot_output: str | None,
    plot_title: str,
) -> None:
    if not output_json and not plot_output:
        return

    from benchmarks.plotting import render_benchmark_plot, write_benchmark_payload

    if output_json:
        json_path = write_benchmark_payload(payload, output_json)
        print(f"Wrote benchmark JSON to {json_path}")

    if plot_output:
        plot_path = render_benchmark_plot(payload, plot_output, title=plot_title or None)
        print(f"Wrote benchmark plot to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MoE benchmark suite")
    parser.add_argument(
        "--filter", type=str, default="",
        help="Substring filter for benchmark names / tags",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override benchmark model name for orchestrator and experts",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="How many times to run each benchmark case",
    )
    parser.add_argument(
        "--output-json", type=str, default="",
        help="Optional path to write a machine-readable benchmark JSON summary",
    )
    parser.add_argument(
        "--plot-output", type=str, default="",
        help="Optional path to render a benchmark PNG/PDF/SVG comparison plot",
    )
    parser.add_argument(
        "--plot-title", type=str, default="",
        help="Optional custom title for the generated benchmark plot",
    )
    parser.add_argument(
        "--selection-bias-slice",
        action="store_true",
        help="Compare baseline routing against atom few-shot + metadata-biased candidate selection",
    )
    parser.add_argument(
        "--warm-task-slice",
        action="store_true",
        help=(
            "Compare baseline routing against graph-aware retrieval on warm-task "
            "benchmark families"
        ),
    )
    args = parser.parse_args()

    if args.selection_bias_slice and args.warm_task_slice:
        parser.error("Choose either --selection-bias-slice or --warm-task-slice, not both.")

    # Late import so benchmarks module can be imported without side-effects
    from benchmarks.plotting import build_comparison_payload, build_report_payload
    from src.core.config import MoEConfig, apply_model_override
    from src.core.config import config as runtime_config
    from src.graph.builder import MoEGraphBuilder

    cfg = MoEConfig()
    if args.model:
        apply_model_override(cfg, args.model)
        apply_model_override(runtime_config, args.model)

    suite = create_standard_suite()
    repeats = max(args.repeats, 1)

    if args.selection_bias_slice or args.warm_task_slice:
        with tempfile.TemporaryDirectory(prefix="moe-bench-") as temp_dir:
            temp_root = Path(temp_dir)
            baseline_cfg = deepcopy(cfg)
            baseline_cfg.registry_db_path = str(temp_root / "baseline.sqlite")
            baseline_cfg.enable_atom_few_shot_retrieval = False
            baseline_cfg.enable_metadata_selection_bias = False

            graph_cfg = deepcopy(cfg)
            graph_cfg.registry_db_path = str(temp_root / "graph.sqlite")
            graph_cfg.enable_atom_few_shot_retrieval = True
            graph_cfg.enable_metadata_selection_bias = True

            if args.warm_task_slice:
                baseline_cfg.orchestrator_candidate_count = max(
                    baseline_cfg.orchestrator_candidate_count,
                    3,
                )
                graph_cfg.orchestrator_candidate_count = max(
                    graph_cfg.orchestrator_candidate_count,
                    3,
                )
                variant_names = {
                    "baseline": MoEGraphBuilder(baseline_cfg).build(),
                    "graph_retrieval": MoEGraphBuilder(graph_cfg).build(),
                }
                filter_pattern = args.filter or "warm"
            else:
                baseline_cfg.orchestrator_candidate_count = max(
                    baseline_cfg.orchestrator_candidate_count,
                    2,
                )
                graph_cfg.orchestrator_candidate_count = max(
                    graph_cfg.orchestrator_candidate_count,
                    2,
                )
                variant_names = {
                    "baseline": MoEGraphBuilder(baseline_cfg).build(),
                    "metadata_bias": MoEGraphBuilder(graph_cfg).build(),
                }
                filter_pattern = args.filter

            comparison = asyncio.run(
                suite.run_variant_slice(
                    variant_names,
                    filter_pattern=filter_pattern,
                    repeats=repeats,
                )
            )
            print(comparison.pretty_print())
            payload = build_comparison_payload(
                comparison,
                slice_name="warm_task" if args.warm_task_slice else "selection_bias",
                filter_pattern=filter_pattern,
                model_name=cfg.orchestrator_config.model_name,
                repeats=repeats,
            )
            _emit_benchmark_outputs(
                payload,
                output_json=args.output_json or None,
                plot_output=args.plot_output or None,
                plot_title=args.plot_title,
            )
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
    payload = build_report_payload(
        report,
        filter_pattern=args.filter,
        model_name=cfg.orchestrator_config.model_name,
        repeats=repeats,
    )
    _emit_benchmark_outputs(
        payload,
        output_json=args.output_json or None,
        plot_output=args.plot_output or None,
        plot_title=args.plot_title,
    )

    # Exit with non-zero if any failures
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
