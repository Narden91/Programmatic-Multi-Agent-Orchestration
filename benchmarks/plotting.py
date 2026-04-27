"""Machine-readable benchmark exports and publication-style comparison plots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


_METRIC_SPECS = (
    {
        "key": "tokens_mean",
        "label": "Query Cost\nMean Tokens",
        "delta_key": "delta_tokens_mean",
        "lower_is_better": True,
    },
    {
        "key": "elapsed_mean_seconds",
        "label": "Latency\nMean Seconds",
        "delta_key": "delta_elapsed_mean_seconds",
        "lower_is_better": True,
    },
    {
        "key": "retries_mean",
        "label": "Retries\nMean Count",
        "delta_key": "delta_retries_mean",
        "lower_is_better": True,
    },
    {
        "key": "success_rate_pct",
        "label": "Success Rate\nPercent",
        "delta_key": "delta_success_rate_pct",
        "lower_is_better": False,
    },
    {
        "key": "neighborhood_reuse_rate_mean",
        "label": "Neighborhood Reuse\nMean Rate",
        "delta_key": "delta_neighborhood_reuse_rate_mean",
        "lower_is_better": False,
    },
)

_COLOR_CYCLE = (
    "#6C757D",
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
)


def build_report_payload(
    report: Any,
    *,
    filter_pattern: str = "",
    model_name: Optional[str] = None,
    repeats: int = 1,
    label: str = "current",
) -> Dict[str, Any]:
    return {
        "mode": "report",
        "filter": filter_pattern,
        "model": model_name,
        "repeats": repeats,
        "variants": [
            {
                "name": label,
                "summary": report.summary(),
            }
        ],
    }


def build_comparison_payload(
    comparison: Any,
    *,
    slice_name: str,
    filter_pattern: str = "",
    model_name: Optional[str] = None,
    repeats: int = 1,
) -> Dict[str, Any]:
    return {
        "mode": "comparison",
        "slice": slice_name,
        "filter": filter_pattern,
        "model": model_name,
        "repeats": repeats,
        "comparison": comparison.summary(),
        "variants": [
            {
                "name": variant.name,
                "summary": variant.report.summary(),
            }
            for variant in comparison.variants
        ],
    }


def write_benchmark_payload(payload: Dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_benchmark_payload(input_path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(input_path).read_text(encoding="utf-8"))


def render_benchmark_plot(
    payload: Dict[str, Any],
    output_path: str | Path,
    *,
    title: Optional[str] = None,
) -> Path:
    variants = payload.get("variants") or []
    if not variants:
        raise ValueError("Benchmark payload does not contain any variants.")

    first_summary = variants[0].get("summary") or {}
    metric_specs = [spec for spec in _METRIC_SPECS if spec["key"] in first_summary]
    if not metric_specs:
        raise ValueError("Benchmark payload does not contain plot-friendly metrics.")

    columns = 3 if len(metric_specs) > 4 else 2
    rows = math.ceil(len(metric_specs) / columns)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.4 * columns, 3.1 * rows),
        constrained_layout=True,
    )
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    fig.suptitle(title or _default_title(payload), fontsize=15, fontweight="bold")
    subtitle = _build_subtitle(payload)
    if subtitle:
        fig.text(0.5, 0.985, subtitle, ha="center", va="top", fontsize=9, color="#4A4A4A")
    fig.text(
        0.5,
        0.01,
        "Lower is better for cost, latency, and retries. Higher is better for success and reuse.",
        ha="center",
        fontsize=9,
        color="#4A4A4A",
    )

    for axis, spec in zip(axes_list, metric_specs):
        _plot_metric_axis(axis, payload, variants, spec)

    for axis in axes_list[len(metric_specs):]:
        fig.delaxes(axis)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark comparison plots from JSON summaries")
    parser.add_argument("input_json", type=str, help="Path to a benchmark JSON summary")
    parser.add_argument("--output", required=True, type=str, help="Path to the output plot file")
    parser.add_argument("--title", default="", type=str, help="Optional custom plot title")
    args = parser.parse_args()

    payload = load_benchmark_payload(args.input_json)
    output_path = render_benchmark_plot(payload, args.output, title=args.title or None)
    print(f"Wrote benchmark plot to {output_path}")


def _build_subtitle(payload: Dict[str, Any]) -> str:
    parts = []
    if payload.get("slice"):
        parts.append(str(payload["slice"]).replace("_", " "))
    if payload.get("filter"):
        parts.append(f"filter={payload['filter']}")
    if payload.get("model"):
        parts.append(f"model={payload['model']}")
    if payload.get("repeats"):
        parts.append(f"repeats={payload['repeats']}")
    return " | ".join(parts)


def _default_title(payload: Dict[str, Any]) -> str:
    if payload.get("mode") == "comparison":
        slice_name = str(payload.get("slice") or "benchmark").replace("_", " ").title()
        return f"{slice_name} Benchmark Comparison"
    return "Benchmark Summary"


def _plot_metric_axis(axis: Any, payload: Dict[str, Any], variants: Sequence[Dict[str, Any]], spec: Dict[str, Any]) -> None:
    labels = [str(variant.get("name", "variant")).replace("_", " ") for variant in variants]
    values = [float((variant.get("summary") or {}).get(spec["key"], 0.0)) for variant in variants]
    colors = [_COLOR_CYCLE[index % len(_COLOR_CYCLE)] for index in range(len(variants))]
    positions = list(range(len(labels)))
    bars = axis.bar(positions, values, color=colors, width=0.62)

    axis.set_title(spec["label"], fontweight="bold")
    axis.set_xticks(positions)
    rotation = 15 if any(len(label) > 12 for label in labels) else 0
    axis.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
    axis.yaxis.grid(True, linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    max_value = max(values) if values else 0.0
    upper_bound = _upper_bound(spec["key"], max_value)
    axis.set_ylim(0, upper_bound)
    offset = max(upper_bound * 0.045, 0.03)

    for bar, value, label in zip(bars, values, labels):
        delta = _lookup_delta(payload, label.replace(" ", "_"), spec["delta_key"])
        text = _format_metric_value(value, spec["key"])
        if delta is not None:
            text = f"{text}\nΔ {_format_delta_value(delta, spec['key'])}"

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height(), 0.0) + offset,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=_delta_color(delta, spec["lower_is_better"]),
        )


def _upper_bound(metric_key: str, max_value: float) -> float:
    if metric_key == "success_rate_pct":
        return max(100.0, max_value * 1.15 or 100.0)
    if metric_key == "neighborhood_reuse_rate_mean":
        return max(1.0, max_value * 1.15 or 1.0)
    if max_value <= 0:
        return 1.0
    return max_value * 1.2


def _lookup_delta(payload: Dict[str, Any], variant_name: str, delta_key: str) -> Optional[float]:
    comparison = payload.get("comparison") or {}
    for item in comparison.get("deltas", []):
        if item.get("name") == variant_name:
            return float(item.get(delta_key, 0.0))
    return None


def _format_metric_value(value: float, metric_key: str) -> str:
    if metric_key == "success_rate_pct":
        return f"{value:.1f}%"
    if metric_key == "tokens_mean":
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _format_delta_value(delta: float, metric_key: str) -> str:
    if metric_key == "success_rate_pct":
        return f"{delta:+.1f} pts"
    if metric_key == "tokens_mean":
        return f"{delta:+.1f}"
    if abs(delta) >= 10:
        return f"{delta:+.2f}"
    return f"{delta:+.3f}"


def _delta_color(delta: Optional[float], lower_is_better: bool) -> str:
    if delta is None:
        return "#222222"
    improved = delta < 0 if lower_is_better else delta > 0
    return "#0B6E4F" if improved else "#A22C29"


if __name__ == "__main__":
    main()