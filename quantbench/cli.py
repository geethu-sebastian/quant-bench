"""
QuantBench CLI — Command-line interface for running benchmarks.

Usage:
    # Quick benchmark with default model (SmolLM2-135M)
    python -m quantbench.cli --model SmolLM2-135M --methods baseline dynamic

    # Full benchmark with all methods
    python -m quantbench.cli --model GPT2 --methods all

    # Skip perplexity evaluation for faster results
    python -m quantbench.cli --model SmolLM2-135M --methods baseline dynamic --no-perplexity

    # Custom prompt and token count
    python -m quantbench.cli --model GPT2 --methods baseline dynamic \\
        --prompt "Edge AI enables" --max-tokens 50

    # List available models
    python -m quantbench.cli --list-models
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
from rich.panel import Panel

from quantbench.config import (
    BenchmarkConfig,
    SUPPORTED_MODELS,
    QUANTIZATION_METHODS,
    get_system_info,
)
from quantbench.models import (
    load_model_and_tokenizer,
    load_model,
    load_tokenizer,
    get_model_memory_footprint,
    cleanup_model,
)
from quantbench.benchmark import (
    BenchmarkResult,
    benchmark_pytorch_model,
    benchmark_onnx_model,
    compute_compression_ratios,
    save_results,
)
from quantbench.evaluate import evaluate_perplexity, evaluate_perplexity_onnx
from quantbench.report import generate_full_report, generate_comparison_table

console = Console()


def setup_logging(verbose: bool = True):
    """Configure rich logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


def list_models():
    """Display available models in a rich table."""
    table = Table(title="Available Models", show_lines=True)
    table.add_column("Name", style="bold cyan")
    table.add_column("HuggingFace ID")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Recommended", justify="center")
    table.add_column("Description")

    for name, info in SUPPORTED_MODELS.items():
        rec = "✅" if info.get("recommended", False) else "⚠️ (tight on 8GB)"
        table.add_row(
            name,
            info["hf_id"],
            str(info["approx_size_mb"]),
            rec,
            info["description"],
        )

    console.print(table)


def list_methods():
    """Display available quantization methods."""
    table = Table(title="Quantization Methods", show_lines=True)
    table.add_column("Method", style="bold cyan")
    table.add_column("Description")
    table.add_column("Calibration", justify="center")

    for name, info in QUANTIZATION_METHODS.items():
        cal = "Required" if info["requires_calibration"] else "Not needed"
        table.add_row(name, info["description"], cal)

    console.print(table)


def run_benchmark(config: BenchmarkConfig):
    """Execute the full benchmark pipeline.

    Pipeline:
    1. Load baseline FP32 model
    2. For each quantization method:
       a. Apply quantization
       b. Benchmark throughput & latency
       c. Evaluate perplexity (optional)
       d. Clean up memory
    3. Compute compression ratios
    4. Generate report with charts & tables
    """
    total_start = time.time()

    console.print(Panel.fit(
        f"[bold cyan]QuantBench[/] — LLM Quantization Benchmark\n\n"
        f"Model: [bold]{config.model_name}[/] ({config.model_hf_id})\n"
        f"Methods: [bold]{', '.join(config.methods)}[/]\n"
        f"Tokens: {config.max_new_tokens} | "
        f"Runs: {config.num_benchmark_runs} | "
        f"Perplexity: {'✅' if config.eval_perplexity else '❌'}\n"
        f"Output: {config.results_dir}",
        title="Configuration",
    ))

    # Print system info
    sys_info = get_system_info()
    console.print(f"\n[dim]System: {sys_info['cpu_brand']} | "
                  f"RAM: {sys_info['ram_total_gb']}GB | "
                  f"PyTorch: {sys_info.get('pytorch_version', 'N/A')}[/dim]\n")

    results = []

    # ─── Step 1: Load tokenizer (shared across all methods) ──
    console.print("[bold]Step 1:[/] Loading tokenizer...")
    tokenizer = load_tokenizer(config.model_name)

    # ─── Step 2: Baseline (FP32) ─────────────────────────────
    if "baseline" in config.methods or "all" in config.methods:
        console.print("\n[bold blue]━━━ FP32 Baseline ━━━[/]")

        model = load_model(config.model_name, dtype=torch.float32)

        result = benchmark_pytorch_model(
            model=model,
            tokenizer=tokenizer,
            prompt=config.prompt,
            max_new_tokens=config.max_new_tokens,
            num_warmup_runs=config.num_warmup_runs,
            num_benchmark_runs=config.num_benchmark_runs,
            method_name="baseline",
            model_name=config.model_name,
        )

        if config.eval_perplexity:
            console.print("  Evaluating perplexity...")
            ppl_result = evaluate_perplexity(
                model, tokenizer,
                max_samples=config.perplexity_max_samples,
                stride=config.perplexity_stride,
            )
            result.perplexity = ppl_result["perplexity"]

        results.append(result)
        cleanup_model(model)

    # ─── Step 3: Dynamic Quantization (INT8) ─────────────────
    if "dynamic" in config.methods or "all" in config.methods:
        console.print("\n[bold green]━━━ PyTorch Dynamic Quantization (INT8) ━━━[/]")

        model = load_model(config.model_name, dtype=torch.float32)

        from quantbench.quantizers.dynamic_quant import apply_dynamic_quantization
        quantized_model = apply_dynamic_quantization(model, dtype=torch.qint8)
        cleanup_model(model)

        result = benchmark_pytorch_model(
            model=quantized_model,
            tokenizer=tokenizer,
            prompt=config.prompt,
            max_new_tokens=config.max_new_tokens,
            num_warmup_runs=config.num_warmup_runs,
            num_benchmark_runs=config.num_benchmark_runs,
            method_name="dynamic",
            model_name=config.model_name,
        )

        if config.eval_perplexity:
            console.print("  Evaluating perplexity...")
            ppl_result = evaluate_perplexity(
                quantized_model, tokenizer,
                max_samples=config.perplexity_max_samples,
                stride=config.perplexity_stride,
            )
            result.perplexity = ppl_result["perplexity"]

        results.append(result)
        cleanup_model(quantized_model)

    # ─── Step 4: Static Quantization (INT8 PTQ) ──────────────
    if "static" in config.methods or "all" in config.methods:
        console.print("\n[bold red]━━━ PyTorch Static PTQ (INT8) ━━━[/]")

        model = load_model(config.model_name, dtype=torch.float32)

        from quantbench.quantizers.static_quant import apply_static_quantization
        quantized_model = apply_static_quantization(
            model, tokenizer,
            num_calibration_samples=min(50, config.perplexity_max_samples),
        )
        cleanup_model(model)

        result = benchmark_pytorch_model(
            model=quantized_model,
            tokenizer=tokenizer,
            prompt=config.prompt,
            max_new_tokens=config.max_new_tokens,
            num_warmup_runs=config.num_warmup_runs,
            num_benchmark_runs=config.num_benchmark_runs,
            method_name="static",
            model_name=config.model_name,
        )

        if config.eval_perplexity:
            console.print("  Evaluating perplexity...")
            ppl_result = evaluate_perplexity(
                quantized_model, tokenizer,
                max_samples=config.perplexity_max_samples,
                stride=config.perplexity_stride,
            )
            result.perplexity = ppl_result["perplexity"]

        results.append(result)
        cleanup_model(quantized_model)

    # ─── Step 5: GPTQ Quantization (INT4) ────────────────────
    if "gptq" in config.methods or "all" in config.methods:
        console.print("\n[bold magenta]━━━ GPTQ Quantization (INT4) ━━━[/]")

        from quantbench.quantizers.gptq_quant import (
            check_gptq_available,
            apply_gptq_quantization,
        )

        if check_gptq_available():
            gptq_save_path = str(Path(config.results_dir) / "gptq_model")

            gptq_model = apply_gptq_quantization(
                model_name=config.model_name,
                tokenizer=tokenizer,
                bits=config.gptq_bits,
                save_path=gptq_save_path,
            )

            # GPTQ models have a different inference API
            result = BenchmarkResult(
                model_name=config.model_name,
                method="gptq",
                precision=f"INT{config.gptq_bits} (GPTQ)",
                num_runs=config.num_benchmark_runs,
            )

            # Benchmark GPTQ model
            try:
                result = benchmark_pytorch_model(
                    model=gptq_model.model if hasattr(gptq_model, 'model') else gptq_model,
                    tokenizer=tokenizer,
                    prompt=config.prompt,
                    max_new_tokens=config.max_new_tokens,
                    num_warmup_runs=config.num_warmup_runs,
                    num_benchmark_runs=config.num_benchmark_runs,
                    method_name="gptq",
                    model_name=config.model_name,
                )
            except Exception as e:
                console.print(f"  [yellow]GPTQ benchmark failed: {e}[/]")
                # Use disk size as fallback
                from quantbench.models import get_model_size_on_disk
                result.model_disk_mb = get_model_size_on_disk(gptq_save_path)

            if config.eval_perplexity:
                try:
                    console.print("  Evaluating perplexity...")
                    inner_model = gptq_model.model if hasattr(gptq_model, 'model') else gptq_model
                    ppl_result = evaluate_perplexity(
                        inner_model, tokenizer,
                        max_samples=config.perplexity_max_samples,
                        stride=config.perplexity_stride,
                    )
                    result.perplexity = ppl_result["perplexity"]
                except Exception as e:
                    console.print(f"  [yellow]GPTQ perplexity eval failed: {e}[/]")

            results.append(result)
            cleanup_model(gptq_model)
        else:
            console.print("  [yellow]⚠ auto-gptq not installed, skipping GPTQ[/]")

    # ─── Step 6: ONNX Runtime Quantization (INT8) ────────────
    if "onnx" in config.methods or "all" in config.methods:
        console.print("\n[bold yellow]━━━ ONNX Runtime Quantization (INT8) ━━━[/]")

        from quantbench.quantizers.onnx_quant import (
            check_onnx_available,
            export_to_onnx,
            export_with_optimum,
            apply_onnx_dynamic_quantization,
        )

        if check_onnx_available():
            onnx_dir = str(Path(config.results_dir) / "onnx")
            onnx_model_path = None

            # Try Optimum export first (better compatibility), fall back to torch
            try:
                console.print("  Exporting to ONNX via Optimum...")
                export_dir = export_with_optimum(config.model_name, onnx_dir)
                # Find the ONNX file
                onnx_files = list(Path(export_dir).glob("*.onnx"))
                if onnx_files:
                    onnx_model_path = str(onnx_files[0])
            except Exception as e:
                console.print(f"  [dim]Optimum export failed ({e}), trying torch.onnx.export...[/]")

            if onnx_model_path is None:
                try:
                    model = load_model(config.model_name, dtype=torch.float32)
                    onnx_model_path = export_to_onnx(model, tokenizer, onnx_dir)
                    cleanup_model(model)
                except Exception as e:
                    console.print(f"  [red]ONNX export failed: {e}[/]")

            if onnx_model_path:
                # Apply dynamic quantization to the ONNX model
                console.print("  Applying ONNX dynamic quantization...")
                quantized_onnx_path = apply_onnx_dynamic_quantization(onnx_model_path)

                # Benchmark the quantized ONNX model
                result = benchmark_onnx_model(
                    onnx_model_path=quantized_onnx_path,
                    tokenizer=tokenizer,
                    prompt=config.prompt,
                    max_new_tokens=config.max_new_tokens,
                    num_warmup_runs=config.num_warmup_runs,
                    num_benchmark_runs=config.num_benchmark_runs,
                    model_name=config.model_name,
                    method_name="onnx",
                )

                if config.eval_perplexity:
                    try:
                        console.print("  Evaluating ONNX perplexity...")
                        ppl_result = evaluate_perplexity_onnx(
                            quantized_onnx_path, tokenizer,
                            max_samples=min(50, config.perplexity_max_samples),
                        )
                        result.perplexity = ppl_result["perplexity"]
                    except Exception as e:
                        console.print(f"  [yellow]ONNX perplexity eval failed: {e}[/]")

                results.append(result)
        else:
            console.print("  [yellow]⚠ onnxruntime not installed, skipping ONNX[/]")

    # ─── Step 7: Compute compression ratios & generate report ─
    console.print("\n[bold]━━━ Generating Report ━━━[/]")

    results = compute_compression_ratios(results)

    # Save raw results
    save_results(results, config.results_dir)

    # Generate visual report
    report_path = generate_full_report(
        results,
        save_dir=config.results_dir,
        model_name=config.model_name,
    )

    total_time = time.time() - total_start

    console.print(Panel.fit(
        f"[bold green]Benchmark complete![/]\n\n"
        f"Total time: {total_time:.1f}s\n"
        f"Methods benchmarked: {len(results)}\n"
        f"Results: {config.results_dir}\n"
        f"Report: {report_path}",
        title="✅ Done",
    ))

    return results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="quantbench",
        description="QuantBench — LLM Quantization Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m quantbench.cli --model SmolLM2-135M --methods baseline dynamic
  python -m quantbench.cli --model GPT2 --methods all
  python -m quantbench.cli --model GPT2 --methods baseline dynamic --no-perplexity
  python -m quantbench.cli --list-models
  python -m quantbench.cli --list-methods
        """,
    )

    parser.add_argument(
        "--model", type=str, default="SmolLM2-135M",
        help="Model name (from --list-models) or HuggingFace model ID",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["baseline", "dynamic"],
        choices=["baseline", "dynamic", "static", "gptq", "onnx", "all"],
        help="Quantization methods to benchmark",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="The future of artificial intelligence on edge devices is",
        help="Prompt for text generation benchmark",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100,
        help="Max tokens to generate per benchmark run",
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=2,
        help="Number of warmup runs before benchmarking",
    )
    parser.add_argument(
        "--benchmark-runs", type=int, default=5,
        help="Number of benchmark runs for statistics",
    )
    parser.add_argument(
        "--gptq-bits", type=int, default=4, choices=[2, 3, 4, 8],
        help="Bit width for GPTQ quantization",
    )
    parser.add_argument(
        "--no-perplexity", action="store_true",
        help="Skip perplexity evaluation (faster)",
    )
    parser.add_argument(
        "--perplexity-samples", type=int, default=200,
        help="Max samples for perplexity evaluation",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Custom output directory for results",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--list-methods", action="store_true",
        help="List available quantization methods and exit",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Verbose output",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    setup_logging(verbose=not args.quiet)

    if args.list_models:
        list_models()
        sys.exit(0)

    if args.list_methods:
        list_methods()
        sys.exit(0)

    # Resolve 'all' methods
    methods = args.methods
    if "all" in methods:
        methods = ["baseline", "dynamic", "static", "gptq", "onnx"]

    # Build config
    config = BenchmarkConfig(
        model_name=args.model,
        methods=methods,
        gptq_bits=args.gptq_bits,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        num_warmup_runs=args.warmup_runs,
        num_benchmark_runs=args.benchmark_runs,
        eval_perplexity=not args.no_perplexity,
        perplexity_max_samples=args.perplexity_samples,
        results_dir=args.output_dir,
        verbose=not args.quiet,
    )

    # Run benchmark
    results = run_benchmark(config)

    return results


if __name__ == "__main__":
    main()
