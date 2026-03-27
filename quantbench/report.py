"""
Report generation for QuantBench.

Produces visual comparison charts and formatted tables from benchmark results.
All visualizations use matplotlib/seaborn for high-quality figures suitable
for README embeds, papers, and presentations.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from tabulate import tabulate

from quantbench.benchmark import BenchmarkResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Style Configuration
# ──────────────────────────────────────────────────────────
COLORS = {
    "baseline": "#4C72B0",
    "dynamic": "#55A868",
    "static": "#C44E52",
    "gptq": "#8172B3",
    "onnx": "#CCB974",
}

LABELS = {
    "baseline": "FP32 Baseline",
    "dynamic": "Dynamic INT8",
    "static": "Static PTQ INT8",
    "gptq": "GPTQ INT4",
    "onnx": "ONNX Runtime INT8",
}


def _setup_style():
    """Configure matplotlib/seaborn for clean, modern charts."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
    })


def generate_comparison_table(
    results: List[BenchmarkResult],
    include_perplexity: bool = True,
) -> str:
    """Generate a formatted markdown comparison table.

    Args:
        results: List of benchmark results.
        include_perplexity: Whether to include perplexity column.

    Returns:
        Formatted markdown table string.
    """
    headers = [
        "Method",
        "Precision",
        "Model Size (MB)",
        "Compression",
        "Throughput (tok/s)",
        "TTFT (ms)",
        "Per-Token (ms)",
        "Peak Mem (MB)",
    ]

    if include_perplexity:
        headers.append("Perplexity")

    rows = []
    for r in results:
        row = [
            LABELS.get(r.method, r.method),
            r.precision,
            f"{r.model_size_mb:.1f}" if r.model_size_mb > 0 else f"{r.model_disk_mb:.1f}",
            f"{r.compression_ratio:.2f}x",
            f"{r.tokens_per_second:.1f} ± {r.std_tokens_per_second:.1f}",
            f"{r.time_to_first_token_ms:.1f}",
            f"{r.avg_time_per_token_ms:.1f}",
            f"{r.peak_memory_mb:.0f}",
        ]
        if include_perplexity:
            if r.perplexity > 0:
                row.append(f"{r.perplexity:.2f}")
            else:
                row.append("—")
        rows.append(row)

    table = tabulate(rows, headers=headers, tablefmt="github")
    return table


def plot_model_size_comparison(
    results: List[BenchmarkResult],
    save_path: str,
    title: Optional[str] = None,
):
    """Bar chart comparing model sizes across quantization methods.

    Args:
        results: Benchmark results.
        save_path: Path to save the chart image.
        title: Optional chart title.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [LABELS.get(r.method, r.method) for r in results]
    sizes = [r.model_size_mb if r.model_size_mb > 0 else r.model_disk_mb for r in results]
    colors = [COLORS.get(r.method, "#999999") for r in results]

    bars = ax.bar(methods, sizes, color=colors, edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sizes) * 0.02,
            f"{size:.1f} MB",
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )

    # Add compression ratios
    if len(results) > 0 and results[0].method == "baseline":
        baseline_size = sizes[0]
        for bar, size in zip(bars[1:], sizes[1:]):
            ratio = baseline_size / max(size, 0.01)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{ratio:.1f}x smaller",
                ha="center", va="center",
                fontsize=9, color="white", fontweight="bold",
            )

    ax.set_ylabel("Model Size (MB)")
    ax.set_title(title or "Model Size Comparison Across Quantization Methods")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Model size chart saved to {save_path}")


def plot_throughput_comparison(
    results: List[BenchmarkResult],
    save_path: str,
    title: Optional[str] = None,
):
    """Bar chart comparing throughput (tokens/second) with error bars.

    Args:
        results: Benchmark results.
        save_path: Path to save the chart image.
        title: Optional chart title.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [LABELS.get(r.method, r.method) for r in results]
    throughputs = [r.tokens_per_second for r in results]
    stds = [r.std_tokens_per_second for r in results]
    colors = [COLORS.get(r.method, "#999999") for r in results]

    bars = ax.bar(
        methods, throughputs, yerr=stds,
        color=colors, edgecolor="white", linewidth=1.5,
        capsize=5, error_kw={"linewidth": 1.5},
    )

    for bar, tps in zip(bars, throughputs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(throughputs) * 0.03,
            f"{tps:.1f}",
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )

    ax.set_ylabel("Tokens per Second")
    ax.set_title(title or "Inference Throughput Comparison")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Throughput chart saved to {save_path}")


def plot_latency_breakdown(
    results: List[BenchmarkResult],
    save_path: str,
    title: Optional[str] = None,
):
    """Grouped bar chart showing TTFT and per-token latency.

    Shows both Time to First Token and Average Time Per Token
    side by side for each method.

    Args:
        results: Benchmark results.
        save_path: Path to save the chart.
        title: Optional chart title.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = [LABELS.get(r.method, r.method) for r in results]
    ttfts = [r.time_to_first_token_ms for r in results]
    tpts = [r.avg_time_per_token_ms for r in results]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, ttfts, width, label="Time to First Token (ms)",
                   color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width/2, tpts, width, label="Avg Per-Token (ms)",
                   color="#55A868", edgecolor="white")

    for bar, val in zip(bars1, ttfts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    for bar, val in zip(bars2, tpts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Latency (ms)")
    ax.set_title(title or "Latency Breakdown: Time to First Token vs Per-Token")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Latency breakdown chart saved to {save_path}")


def plot_perplexity_comparison(
    results: List[BenchmarkResult],
    save_path: str,
    title: Optional[str] = None,
):
    """Bar chart showing perplexity across quantization methods.

    Lower is better. Includes a dashed reference line for the baseline.

    Args:
        results: Benchmark results with perplexity.
        save_path: Path to save the chart.
        title: Optional chart title.
    """
    _setup_style()

    # Filter results with valid perplexity
    valid_results = [r for r in results if r.perplexity > 0 and r.perplexity < 1e6]
    if not valid_results:
        logger.warning("No valid perplexity measurements to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [LABELS.get(r.method, r.method) for r in valid_results]
    ppls = [r.perplexity for r in valid_results]
    colors = [COLORS.get(r.method, "#999999") for r in valid_results]

    bars = ax.bar(methods, ppls, color=colors, edgecolor="white", linewidth=1.5)

    # Add baseline reference line
    baseline_ppl = next(
        (r.perplexity for r in valid_results if r.method == "baseline"), None
    )
    if baseline_ppl:
        ax.axhline(y=baseline_ppl, color="#4C72B0", linestyle="--",
                   alpha=0.7, label=f"Baseline PPL: {baseline_ppl:.2f}")

    for bar, ppl in zip(bars, ppls):
        delta = ""
        if baseline_ppl and ppl != baseline_ppl:
            d = ppl - baseline_ppl
            delta = f"\n(+{d:.2f})" if d > 0 else f"\n({d:.2f})"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ppls) * 0.02,
            f"{ppl:.2f}{delta}",
            ha="center", va="bottom", fontweight="bold", fontsize=10,
        )

    ax.set_ylabel("Perplexity (↓ lower is better)")
    ax.set_title(title or "Perplexity Impact of Quantization (WikiText-2)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if baseline_ppl:
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Perplexity chart saved to {save_path}")


def plot_memory_comparison(
    results: List[BenchmarkResult],
    save_path: str,
    title: Optional[str] = None,
):
    """Bar chart comparing peak memory usage.

    Args:
        results: Benchmark results.
        save_path: Path to save the chart.
        title: Optional chart title.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [LABELS.get(r.method, r.method) for r in results]
    memories = [r.peak_memory_mb for r in results]
    colors = [COLORS.get(r.method, "#999999") for r in results]

    bars = ax.bar(methods, memories, color=colors, edgecolor="white", linewidth=1.5)

    for bar, mem in zip(bars, memories):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(memories) * 0.02,
            f"{mem:.0f} MB",
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )

    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title(title or "Peak Memory Usage During Inference")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Memory chart saved to {save_path}")


def plot_radar_chart(
    results: List[BenchmarkResult],
    save_path: str,
    title: Optional[str] = None,
):
    """Radar/spider chart showing multi-dimensional comparison.

    Normalizes all metrics to 0-1 range and plots them on a radar chart.
    Great for quick visual comparison of trade-offs.

    Dimensions:
    - Compression (higher = better)
    - Throughput (higher = better)
    - Latency (lower = better, inverted)
    - Memory efficiency (lower = better, inverted)
    - Accuracy (lower perplexity = better, inverted)
    """
    _setup_style()

    categories = ["Compression", "Throughput", "Low Latency", "Mem Efficiency", "Accuracy"]
    N = len(categories)

    # Extract and normalize metrics
    compressions = [r.compression_ratio for r in results]
    throughputs = [r.tokens_per_second for r in results]
    latencies = [r.avg_time_per_token_ms for r in results]
    memories = [r.peak_memory_mb if r.peak_memory_mb > 0 else r.model_size_mb for r in results]
    perplexities = [r.perplexity if r.perplexity > 0 else 100 for r in results]

    def normalize(values, invert=False):
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return [0.5] * len(values)
        normed = [(v - min_v) / (max_v - min_v) for v in values]
        if invert:
            normed = [1 - v for v in normed]
        return normed

    norm_data = {
        "Compression": normalize(compressions),
        "Throughput": normalize(throughputs),
        "Low Latency": normalize(latencies, invert=True),
        "Mem Efficiency": normalize(memories, invert=True),
        "Accuracy": normalize(perplexities, invert=True),
    }

    # Plot
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, r in enumerate(results):
        values = [norm_data[cat][i] for cat in categories]
        values += values[:1]
        color = COLORS.get(r.method, f"C{i}")
        label = LABELS.get(r.method, r.method)

        ax.plot(angles, values, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(title or "Quantization Trade-off Radar", pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Radar chart saved to {save_path}")


def generate_full_report(
    results: List[BenchmarkResult],
    save_dir: str,
    model_name: str = "",
) -> str:
    """Generate a complete report with all charts and tables.

    Creates:
    - comparison_table.md: Markdown table
    - model_size.png: Size comparison bar chart
    - throughput.png: Throughput comparison bar chart
    - latency.png: Latency breakdown chart
    - perplexity.png: Perplexity comparison chart
    - memory.png: Memory usage chart
    - radar.png: Trade-off radar chart
    - report.md: Full markdown report combining everything

    Args:
        results: List of benchmark results.
        save_dir: Directory to save all report artifacts.
        model_name: Model name for titles.

    Returns:
        Path to the generated report.md file.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    title_prefix = f"{model_name} — " if model_name else ""

    # Generate table
    table = generate_comparison_table(results)
    table_path = save_path / "comparison_table.md"
    table_path.write_text(table)

    # Generate charts
    plot_model_size_comparison(
        results, str(save_path / "model_size.png"),
        title=f"{title_prefix}Model Size Comparison"
    )
    plot_throughput_comparison(
        results, str(save_path / "throughput.png"),
        title=f"{title_prefix}Inference Throughput"
    )
    plot_latency_breakdown(
        results, str(save_path / "latency.png"),
        title=f"{title_prefix}Latency Breakdown"
    )
    plot_perplexity_comparison(
        results, str(save_path / "perplexity.png"),
        title=f"{title_prefix}Perplexity Impact"
    )
    plot_memory_comparison(
        results, str(save_path / "memory.png"),
        title=f"{title_prefix}Peak Memory Usage"
    )
    plot_radar_chart(
        results, str(save_path / "radar.png"),
        title=f"{title_prefix}Quantization Trade-offs"
    )

    # Generate markdown report
    report_lines = [
        f"# QuantBench Results: {model_name}\n",
        f"## Comparison Table\n",
        table,
        "\n",
        "## Model Size Comparison\n",
        "![Model Size](model_size.png)\n",
        "## Inference Throughput\n",
        "![Throughput](throughput.png)\n",
        "## Latency Breakdown\n",
        "![Latency](latency.png)\n",
        "## Perplexity Impact\n",
        "![Perplexity](perplexity.png)\n",
        "## Peak Memory Usage\n",
        "![Memory](memory.png)\n",
        "## Trade-off Radar\n",
        "![Radar](radar.png)\n",
    ]

    # Add generated text samples
    report_lines.append("\n## Generated Text Samples\n")
    for r in results:
        if r.generated_text:
            label = LABELS.get(r.method, r.method)
            report_lines.append(f"### {label}\n")
            report_lines.append(f"```\n{r.generated_text}\n```\n")

    report_content = "\n".join(report_lines)
    report_path = save_path / "report.md"
    report_path.write_text(report_content)

    logger.info(f"Full report generated at {report_path}")
    print(f"\n{'='*60}")
    print(f"  Report saved to: {report_path}")
    print(f"  Charts saved to: {save_path}")
    print(f"{'='*60}\n")
    print(table)

    return str(report_path)
