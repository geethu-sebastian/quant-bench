"""
Benchmark harness for QuantBench.

Measures throughput (tokens/sec), latency (time to first token, per-token),
peak memory usage, and model size for each quantization variant.

All benchmarks run on CPU to simulate edge/on-device deployment.
"""

import gc
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    # Identity
    model_name: str = ""
    method: str = ""  # e.g., "baseline", "dynamic", "gptq", "onnx"
    precision: str = ""  # e.g., "FP32", "INT8", "INT4"

    # Model size
    model_size_mb: float = 0.0  # In-memory model size
    model_disk_mb: float = 0.0  # On-disk model size
    param_count: int = 0

    # Throughput & Latency
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0
    avg_time_per_token_ms: float = 0.0
    total_generation_time_s: float = 0.0
    num_tokens_generated: int = 0

    # Memory
    peak_memory_mb: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_delta_mb: float = 0.0

    # Quality
    perplexity: float = 0.0
    generated_text: str = ""

    # Metadata
    num_runs: int = 0
    std_tokens_per_second: float = 0.0
    std_time_per_token_ms: float = 0.0
    compression_ratio: float = 1.0  # vs. FP32 baseline

    def to_dict(self) -> dict:
        return asdict(self)


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def benchmark_pytorch_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str = "The future of artificial intelligence on edge devices is",
    max_new_tokens: int = 100,
    num_warmup_runs: int = 2,
    num_benchmark_runs: int = 5,
    method_name: str = "baseline",
    model_name: str = "",
) -> BenchmarkResult:
    """Benchmark a PyTorch model (FP32 or quantized).

    Measures:
    1. Time to First Token (TTFT): Time from prompt submission to first output token.
       Critical metric for user-perceived latency in chat applications.
    2. Average Time Per Token (ATPT): Mean time to generate each subsequent token.
       Determines the streaming speed of text generation.
    3. Throughput: Tokens generated per second (1000 / ATPT).
    4. Peak Memory: Maximum RSS memory during generation.

    The benchmark uses autoregressive generation with greedy decoding
    (temperature=0) for reproducible results.

    Args:
        model: PyTorch model (FP32 or quantized).
        tokenizer: Model tokenizer.
        prompt: Input prompt for generation.
        max_new_tokens: Max tokens to generate per run.
        num_warmup_runs: Warmup runs to stabilize CPU caches and JIT.
        num_benchmark_runs: Actual measured runs.
        method_name: Name of the quantization method.
        model_name: Name of the model.

    Returns:
        BenchmarkResult with all metrics populated.
    """
    result = BenchmarkResult(
        model_name=model_name,
        method=method_name,
        num_runs=num_benchmark_runs,
    )

    # Get model stats
    from quantbench.models import get_model_memory_footprint
    model_stats = get_model_memory_footprint(model)
    result.model_size_mb = model_stats["total_mb"]
    result.param_count = model_stats["param_count"]

    # Determine precision string
    precision_map = {
        "baseline": "FP32",
        "dynamic": "INT8 (dynamic)",
        "static": "INT8 (static)",
        "gptq": "INT4 (GPTQ)",
        "onnx": "INT8 (ONNX)",
    }
    result.precision = precision_map.get(method_name, method_name)

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    model.eval()

    # ─── Warmup Runs ─────────────────────────────────────
    logger.info(f"[{method_name}] Running {num_warmup_runs} warmup iterations...")
    for _ in range(num_warmup_runs):
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(max_new_tokens, 20),
                do_sample=False,
            )
    gc.collect()

    # ─── Benchmark Runs ──────────────────────────────────
    logger.info(f"[{method_name}] Running {num_benchmark_runs} benchmark iterations...")

    all_tps = []  # tokens per second per run
    all_tpt = []  # time per token per run
    all_ttft = []  # time to first token per run
    all_total_times = []

    for run_idx in range(num_benchmark_runs):
        memory_before = get_process_memory_mb()

        # Measure time to first token
        ttft_start = time.perf_counter()

        with torch.no_grad():
            # Generate one token to measure TTFT
            first_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
            )

        ttft = (time.perf_counter() - ttft_start) * 1000  # ms
        all_ttft.append(ttft)

        # Full generation
        gen_start = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        gen_end = time.perf_counter()
        gen_time = gen_end - gen_start

        memory_after = get_process_memory_mb()

        # Calculate metrics
        num_generated = output.shape[1] - input_ids.shape[1]
        tps = num_generated / max(gen_time, 1e-9)
        tpt = (gen_time / max(num_generated, 1)) * 1000  # ms per token

        all_tps.append(tps)
        all_tpt.append(tpt)
        all_total_times.append(gen_time)

        result.peak_memory_mb = max(result.peak_memory_mb, memory_after)
        result.memory_before_mb = memory_before
        result.memory_after_mb = memory_after
        result.memory_delta_mb = max(result.memory_delta_mb, memory_after - memory_before)

        if run_idx == num_benchmark_runs - 1:
            result.num_tokens_generated = num_generated
            result.generated_text = tokenizer.decode(
                output[0][input_ids.shape[1]:], skip_special_tokens=True
            )

        logger.debug(
            f"  Run {run_idx + 1}: {tps:.1f} tok/s, "
            f"TTFT: {ttft:.1f}ms, TPT: {tpt:.1f}ms"
        )

    # Aggregate results
    result.tokens_per_second = float(np.mean(all_tps))
    result.std_tokens_per_second = float(np.std(all_tps))
    result.time_to_first_token_ms = float(np.mean(all_ttft))
    result.avg_time_per_token_ms = float(np.mean(all_tpt))
    result.std_time_per_token_ms = float(np.std(all_tpt))
    result.total_generation_time_s = float(np.mean(all_total_times))

    logger.info(
        f"[{method_name}] Results: "
        f"{result.tokens_per_second:.1f} ± {result.std_tokens_per_second:.1f} tok/s | "
        f"TTFT: {result.time_to_first_token_ms:.1f}ms | "
        f"TPT: {result.avg_time_per_token_ms:.1f}ms | "
        f"Memory: {result.model_size_mb:.1f}MB model, {result.peak_memory_mb:.0f}MB peak"
    )

    return result


def benchmark_onnx_model(
    onnx_model_path: str,
    tokenizer: AutoTokenizer,
    prompt: str = "The future of artificial intelligence on edge devices is",
    max_new_tokens: int = 100,
    num_warmup_runs: int = 2,
    num_benchmark_runs: int = 5,
    model_name: str = "",
    method_name: str = "onnx",
) -> BenchmarkResult:
    """Benchmark an ONNX model using ONNX Runtime.

    Uses CPUExecutionProvider for consistent CPU-only benchmarking.

    Args:
        onnx_model_path: Path to the ONNX model file.
        tokenizer: Model tokenizer.
        prompt: Input prompt.
        max_new_tokens: Max tokens to generate.
        num_warmup_runs: Warmup iterations.
        num_benchmark_runs: Measured iterations.
        model_name: Model name for labeling.
        method_name: Method name for labeling.

    Returns:
        BenchmarkResult with ONNX-specific metrics.
    """
    from quantbench.quantizers.onnx_quant import run_onnx_inference

    result = BenchmarkResult(
        model_name=model_name,
        method=method_name,
        precision="INT8 (ONNX)" if "quantized" in onnx_model_path else "FP32 (ONNX)",
        num_runs=num_benchmark_runs,
    )

    # Model size on disk
    result.model_disk_mb = Path(onnx_model_path).stat().st_size / (1024 ** 2)

    # Warmup
    logger.info(f"[{method_name}] Running {num_warmup_runs} warmup iterations...")
    for _ in range(num_warmup_runs):
        _ = run_onnx_inference(
            onnx_model_path, tokenizer, prompt, max_new_tokens=5
        )

    # Benchmark
    logger.info(f"[{method_name}] Running {num_benchmark_runs} benchmark iterations...")
    all_tps = []

    for run_idx in range(num_benchmark_runs):
        memory_before = get_process_memory_mb()

        inference_result = run_onnx_inference(
            onnx_model_path, tokenizer, prompt, max_new_tokens=max_new_tokens
        )

        memory_after = get_process_memory_mb()

        tps = inference_result["tokens_per_second"]
        all_tps.append(tps)

        result.peak_memory_mb = max(result.peak_memory_mb, memory_after)
        result.memory_delta_mb = max(result.memory_delta_mb, memory_after - memory_before)

        if run_idx == num_benchmark_runs - 1:
            result.generated_text = inference_result["generated_text"]
            result.num_tokens_generated = inference_result["num_tokens"]
            result.total_generation_time_s = inference_result["total_time_s"]

        logger.debug(f"  Run {run_idx + 1}: {tps:.1f} tok/s")

    result.tokens_per_second = float(np.mean(all_tps))
    result.std_tokens_per_second = float(np.std(all_tps))
    result.avg_time_per_token_ms = 1000.0 / max(result.tokens_per_second, 1e-9)
    result.std_time_per_token_ms = float(np.std([1000.0 / max(t, 1e-9) for t in all_tps]))

    logger.info(
        f"[{method_name}] Results: "
        f"{result.tokens_per_second:.1f} ± {result.std_tokens_per_second:.1f} tok/s | "
        f"Disk: {result.model_disk_mb:.1f}MB | "
        f"Peak Memory: {result.peak_memory_mb:.0f}MB"
    )

    return result


def compute_compression_ratios(
    results: List[BenchmarkResult],
) -> List[BenchmarkResult]:
    """Compute compression ratios relative to the FP32 baseline.

    Args:
        results: List of benchmark results (first should be baseline).

    Returns:
        Same list with compression_ratio populated.
    """
    baseline = None
    for r in results:
        if r.method == "baseline":
            baseline = r
            break

    if baseline is None:
        logger.warning("No baseline result found; compression ratios set to 1.0")
        return results

    for r in results:
        if r.model_size_mb > 0 and baseline.model_size_mb > 0:
            r.compression_ratio = baseline.model_size_mb / r.model_size_mb
        elif r.model_disk_mb > 0 and baseline.model_size_mb > 0:
            r.compression_ratio = baseline.model_size_mb / r.model_disk_mb

    return results


def save_results(
    results: List[BenchmarkResult],
    save_dir: str,
    filename: str = "benchmark_results.json",
):
    """Save benchmark results to JSON.

    Args:
        results: List of BenchmarkResult objects.
        save_dir: Directory to save results.
        filename: Output filename.
    """
    save_path = Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [r.to_dict() for r in results],
    }

    # Add system info
    from quantbench.config import get_system_info
    data["system_info"] = get_system_info()

    with open(save_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Results saved to {save_path}")
    return str(save_path)


def load_results(results_path: str) -> dict:
    """Load benchmark results from JSON."""
    with open(results_path, "r") as f:
        return json.load(f)
