# 🧪 Benchmark Suite & Evaluation Guide

The benchmark suite evaluates programmatic multi-agent routing across multiple dimensions: **latency, token efficiency, candidate recovery rate, storage footprint, and graph neighborhood reuse**.

---

## 🚀 Running Benchmarks

### 1. Instant Local Mock Mode (No API keys required)
Run standard benchmark queries locally with synthetic deterministic responses:

```bash
uv run python -m benchmarks.run --mock-llm --repeats 2 --output-json artifacts/benchmark_results.json
```

### 2. Compression Slice Evaluation
Compare uncompressed storage against entropy-budgeted motif dictionary compression:

```bash
uv run python -m benchmarks.run --mock-llm --compression-slice --repeats 2 --output-json artifacts/compression_benchmark.json --plot-output artifacts/compression_comparison.png
```

### 3. Selection-Bias Slice
Compare baseline single-candidate routing against atom few-shot + metadata-biased candidate search:

```bash
uv run python -m benchmarks.run --selection-bias-slice --repeats 3 --output-json artifacts/selection_bias.json --plot-output artifacts/selection_bias.png
```

### 4. Warm-Task Graph Retrieval Slice
Evaluate neighborhood reuse and latency on warm-task query families (e.g. `binary_search`, `transformers`):

```bash
uv run python -m benchmarks.run --warm-task-slice --filter warm --repeats 2 --output-json artifacts/warm_task.json --plot-output artifacts/warm_task.png
```

---

## 📈 Interpreting Benchmark Metrics

| Metric | Description | Desired Direction |
| :--- | :--- | :--- |
| **Query Cost (`tokens_mean`)** | Total prompt and completion tokens consumed per request | 🔽 Lower is better |
| **Latency (`elapsed_mean_seconds`)** | Wall-clock execution time from query arrival to final response | 🔽 Lower is better |
| **Retries (`retries_mean`)** | Number of retry/repair attempts triggered by sandbox errors | 🔽 Lower is better |
| **Success Rate (`success_rate_pct`)** | Percentage of benchmark cases that successfully executed | 🔼 Higher is better |
| **Recovery Rate (`recovery_rate_pct`)** | Percentage of runs requiring retries that ultimately succeeded | 🔼 Higher is better |
| **Neighborhood Reuse (`neighborhood_reuse_rate_mean`)** | Fraction of retrieved graph edges reused during script generation | 🔼 Higher is better |
| **Storage Footprint (`storage_bytes_mean`)** | Average script byte size stored in the persistent registry | 🔽 Lower is better |
| **Space Savings (`space_savings_pct`)** | Percentage storage savings achieved via dictionary coding | 🔼 Higher is better |

---

## 📊 Generating Comparison Plots

Generate publication-grade PNG/SVG/PDF figures directly from any benchmark output JSON:

```bash
uv run python -m benchmarks.plotting artifacts/compression_benchmark.json --output artifacts/compression_comparison.png --title "Entropy-Budgeted Compression Evaluation"
```
