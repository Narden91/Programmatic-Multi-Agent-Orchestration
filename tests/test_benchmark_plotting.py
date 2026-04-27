from benchmarks.plotting import (
    build_comparison_payload,
    build_report_payload,
    load_benchmark_payload,
    render_benchmark_plot,
    write_benchmark_payload,
)
from benchmarks.suite import (
    BenchmarkCase,
    BenchmarkComparisonReport,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkVariantRun,
)


def _make_report(
    name: str,
    *,
    tokens: int,
    elapsed: float,
    retries: int,
    reuse: float,
    success: bool = True,
) -> BenchmarkReport:
    case = BenchmarkCase(name=name, query="Explain caching")
    return BenchmarkReport(
        results=[
            BenchmarkResult(
                case=case,
                success=success,
                elapsed_seconds=elapsed,
                token_summary={"total_tokens": tokens},
                retry_count=retries,
                neighborhood_reuse_rate=reuse,
            )
        ],
        total_elapsed=elapsed,
    )


def test_build_benchmark_payloads_include_variant_summaries(tmp_path):
    baseline = _make_report("baseline_case", tokens=120, elapsed=2.0, retries=1, reuse=0.0)
    metadata_bias = _make_report("metadata_case", tokens=80, elapsed=1.4, retries=0, reuse=1.0)

    comparison = BenchmarkComparisonReport(
        variants=[
            BenchmarkVariantRun(name="baseline", report=baseline),
            BenchmarkVariantRun(name="metadata_bias", report=metadata_bias),
        ]
    )

    payload = build_comparison_payload(
        comparison,
        slice_name="selection_bias",
        filter_pattern="routing",
        model_name="llama-3.1-8b-instant",
        repeats=3,
    )
    assert payload["comparison"]["deltas"][0]["delta_tokens_mean"] == -40.0
    assert payload["variants"][1]["summary"]["tokens_mean"] == 80.0

    payload_path = write_benchmark_payload(payload, tmp_path / "selection_bias.json")
    loaded = load_benchmark_payload(payload_path)
    assert loaded["slice"] == "selection_bias"
    assert loaded["variants"][0]["name"] == "baseline"

    report_payload = build_report_payload(
        baseline,
        filter_pattern="single_technical",
        model_name="llama-3.1-8b-instant",
        repeats=1,
    )
    assert report_payload["variants"][0]["summary"]["tokens_mean"] == 120.0


def test_render_benchmark_plot_writes_output(tmp_path):
    baseline = _make_report("baseline_case", tokens=120, elapsed=2.0, retries=1, reuse=0.0)
    graph = _make_report("graph_case", tokens=70, elapsed=1.2, retries=0, reuse=0.8)
    comparison = BenchmarkComparisonReport(
        variants=[
            BenchmarkVariantRun(name="baseline", report=baseline),
            BenchmarkVariantRun(name="graph_retrieval", report=graph),
        ]
    )

    payload = build_comparison_payload(
        comparison,
        slice_name="warm_task",
        filter_pattern="warm",
        model_name="llama-3.1-8b-instant",
        repeats=5,
    )

    output_path = render_benchmark_plot(payload, tmp_path / "warm_task.png")

    assert output_path.exists()
    assert output_path.stat().st_size > 0