# QuantBench — LLM Quantization Benchmark Suite

<div align="center">

**A comprehensive benchmarking tool for evaluating quantization techniques on Large Language Models, optimized for CPU-only edge deployment.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.17+-green.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🎯 Overview

QuantBench systematically compares **4 quantization techniques** across multiple LLMs, measuring:

| Metric | Description |
|--------|-------------|
| **Model Size** | In-memory and on-disk footprint (MB) |
| **Throughput** | Tokens generated per second (tok/s) |
| **Time to First Token** | Latency before first output token (ms) |
| **Per-Token Latency** | Average time per subsequent token (ms) |
| **Peak Memory** | Maximum RSS memory during inference (MB) |
| **Perplexity** | Model quality on WikiText-2 test set (↓ lower = better) |

### Quantization Methods Compared

| Method | Precision | Calibration | Key Technique |
|--------|-----------|-------------|---------------|
| **FP32 Baseline** | 32-bit float | — | No compression (reference) |
| **PyTorch Dynamic** | INT8 weights, FP32 activations | None needed | `torch.ao.quantization.quantize_dynamic` — weights pre-quantized, activations quantized at runtime |
| **PyTorch Static PTQ** | INT8 weights + activations | WikiText-2 train | Observer-based calibration with HistogramObserver (activations) and PerChannelMinMaxObserver (weights) |
| **GPTQ** | INT4 weights, FP32 activations | WikiText-2 train | Second-order Hessian-based error compensation — minimizes ‖WX − ŴX‖² layer-by-layer |
| **ONNX Runtime** | INT8 (dynamic) | None needed | Export to ONNX → apply ONNX Runtime quantization → deploy via CPUExecutionProvider |

## 📐 Theoretical Background

### Why Quantization?

Neural networks are typically trained in **FP32** (32 bits per parameter). Quantization reduces this to INT8 (8 bits) or INT4 (4 bits):

```
FP32: 32 bits → 4 bytes per parameter
INT8:  8 bits → 1 byte per parameter  → 4× compression
INT4:  4 bits → 0.5 bytes per parameter → 8× compression
```

The key insight: neural networks are **over-parameterized** and robust to small numerical perturbations. With careful calibration, lower-precision representations closely approximate full-precision behavior.

### Quantization Formula

For a weight value `w` in range [min, max], quantization to `b` bits:

```
scale = (max - min) / (2^b - 1)
zero_point = round(-min / scale)
w_quantized = clamp(round(w / scale) + zero_point, 0, 2^b - 1)
```

### Method Comparison

**Dynamic Quantization** — Simplest approach. Weights are statically quantized to INT8; activations remain FP32 until inference, where they're quantized per-tensor on each forward pass. No calibration data needed.

**Static Post-Training Quantization (PTQ)** — Both weights and activations are quantized to INT8. Uses a calibration dataset to pre-compute optimal scale/zero-point for activations via observers. Faster inference than dynamic (no runtime quantization overhead).

**GPTQ** — Weight-only quantization to INT4 using second-order information. For each layer with weight matrix W and input activations X:
1. Compute Hessian: H = 2·X·Xᵀ + λI
2. Quantize weights column-by-column
3. Compensate remaining weights: δ = -(w_q - quant(w_q)) / H⁻¹_{qq} · H⁻¹_{:,q}
4. This minimizes the layer reconstruction error ‖WX − ŴX‖²

**ONNX Runtime Quantization** — Export model to ONNX (the universal ML interchange format), then apply ONNX Runtime's built-in quantization. Enables deployment on any ONNX-compatible runtime, including Qualcomm QNN for Snapdragon NPUs.

> Reference: [Model Compression for On-Device AI](https://aman.ai/primers/ai/model-compression/)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Linux (tested on Ubuntu 22.04)
- 4GB+ RAM (8GB recommended)
- No GPU required — all benchmarks run on CPU

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/quantbench.git
cd quantbench

# Run the automated setup (creates venv, installs CPU PyTorch + all deps)
bash setup.sh

# Activate the virtual environment
source venv/bin/activate
```

### Run Your First Benchmark

```bash
# Quick benchmark — SmolLM2-135M with dynamic quantization (fastest)
python -m quantbench.cli --model SmolLM2-135M --methods baseline dynamic

# Full benchmark — all methods
python -m quantbench.cli --model SmolLM2-135M --methods baseline dynamic static onnx

# Skip perplexity for faster results
python -m quantbench.cli --model GPT2 --methods baseline dynamic static --no-perplexity

# Run all benchmarks (SmolLM2-135M + GPT2 with all methods)
bash scripts/run_all.sh

# Quick mode (no perplexity, fewer runs)
bash scripts/run_all.sh --quick
```

### CLI Options

```
python -m quantbench.cli --help

Options:
  --model MODEL         Model name or HuggingFace ID (default: SmolLM2-135M)
  --methods METHODS     Quantization methods: baseline dynamic static gptq onnx all
  --prompt TEXT          Input prompt for generation benchmark
  --max-tokens N        Max tokens to generate (default: 100)
  --warmup-runs N       Warmup iterations (default: 2)
  --benchmark-runs N    Measured iterations (default: 5)
  --gptq-bits {2,3,4,8} GPTQ bit width (default: 4)
  --no-perplexity       Skip perplexity evaluation
  --perplexity-samples  Max samples for perplexity (default: 200)
  --output-dir DIR      Custom output directory
  --list-models         Show available models
  --list-methods        Show available quantization methods
```

### Available Models

| Model | Parameters | FP32 Size | RAM Required | Recommended |
|-------|-----------|-----------|--------------|-------------|
| SmolLM2-135M | 135M | ~270 MB | ~2 GB | ✅ Best for testing |
| SmolLM2-360M | 360M | ~720 MB | ~3 GB | ✅ |
| GPT2 | 124M | ~500 MB | ~2.5 GB | ✅ Well-known baseline |
| GPT2-Medium | 355M | ~1.4 GB | ~4 GB | ⚠️ |
| TinyLlama-1.1B | 1.1B | ~2.2 GB | ~6 GB | ⚠️ Tight on 8GB |

## 📊 Output

After running, QuantBench generates a complete report in `results/<model_name>/`:

```
results/smollm2_135m/
├── benchmark_results.json    # Raw data with system info
├── report.md                 # Full markdown report
├── comparison_table.md       # Formatted comparison table
├── model_size.png            # Size comparison chart
├── throughput.png            # Throughput comparison chart
├── latency.png               # TTFT vs per-token latency
├── perplexity.png            # Accuracy impact chart
├── memory.png                # Peak memory chart
└── radar.png                 # Multi-dimensional trade-off radar
```

## 🏗 Architecture

```
quantbench/
├── quantbench/
│   ├── __init__.py           # Package init
│   ├── __main__.py           # python -m quantbench entry point
│   ├── cli.py                # CLI with argparse + rich console output
│   ├── config.py             # Model registry, method definitions, BenchmarkConfig
│   ├── models.py             # HuggingFace model loading with memory management
│   ├── benchmark.py          # Throughput, latency, memory profiling harness
│   ├── evaluate.py           # Sliding-window perplexity on WikiText-2
│   ├── report.py             # Chart generation (matplotlib/seaborn) + markdown
│   └── quantizers/
│       ├── dynamic_quant.py  # PyTorch Dynamic Quantization (INT8)
│       ├── static_quant.py   # PyTorch Static PTQ (INT8) with calibration
│       ├── gptq_quant.py     # GPTQ via AutoGPTQ (INT4/INT8)
│       └── onnx_quant.py     # ONNX export + ONNX Runtime quantization
├── scripts/
│   └── run_all.sh            # One-command full benchmark
├── results/                  # Auto-generated benchmark outputs
├── setup.sh                  # Linux setup script
└── requirements.txt          # Python dependencies
```

### Pipeline Flow

```
                    ┌─────────────┐
                    │  CLI Input  │
                    │ --model     │
                    │ --methods   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Load Model  │
                    │ + Tokenizer │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼───┐ ┌──────▼───┐ ┌──────▼───┐
       │ Dynamic  │ │ Static   │ │  ONNX    │  ...
       │ Quant    │ │ PTQ      │ │  Export  │
       └──────┬───┘ └──────┬───┘ └──────┬───┘
              │            │            │
       ┌──────▼───┐ ┌──────▼───┐ ┌──────▼───┐
       │Benchmark │ │Benchmark │ │Benchmark │
       │ Throughp.│ │ Throughp.│ │ Throughp.│
       │ Latency  │ │ Latency  │ │ Latency  │
       │ Memory   │ │ Memory   │ │ Memory   │
       └──────┬───┘ └──────┬───┘ └──────┬───┘
              │            │            │
       ┌──────▼───┐ ┌──────▼───┐ ┌──────▼───┐
       │Perplexity│ │Perplexity│ │Perplexity│
       │  Eval    │ │  Eval    │ │  Eval    │
       └──────┬───┘ └──────┬───┘ └──────┬───┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │  Generate   │
                    │  Report     │
                    │ Charts +    │
                    │ Tables      │
                    └─────────────┘
```

## 🔑 Key Concepts Demonstrated

This project demonstrates hands-on knowledge of:

- **Model Compression Techniques**: Dynamic quantization, static PTQ, GPTQ, ONNX quantization
- **Quantization Theory**: Scale/zero-point computation, per-channel vs per-tensor, symmetric vs asymmetric
- **AI Framework Proficiency**: PyTorch (`torch.ao.quantization`), ONNX Runtime, HuggingFace Transformers
- **Performance Engineering**: CPU benchmark methodology, warmup strategies, statistical reporting
- **LLM Internals**: Autoregressive generation, tokenization, perplexity evaluation
- **Cross-Platform Deployment**: ONNX export for hardware-agnostic model serving
- **Edge/On-Device AI**: Memory-aware loading, INT4/INT8 inference on constrained hardware

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

## 📚 References

- [Model Compression for On-Device AI](https://aman.ai/primers/ai/model-compression/) — Comprehensive primer on quantization, pruning, distillation, and low-rank decomposition
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Frantar et al., 2023
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Runtime Quantization Guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [HuggingFace Optimum](https://huggingface.co/docs/optimum/) — ONNX export and optimization
