"""
Configuration for QuantBench.

Defines supported models, quantization methods, and benchmark parameters.
All models are chosen to fit within 8GB RAM on CPU.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────
# Supported Models (CPU-friendly, <8GB RAM)
# ──────────────────────────────────────────────────────────
SUPPORTED_MODELS = {
    "SmolLM2-135M": {
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "description": "135M parameter model — minimal RAM, fast benchmarks",
        "approx_size_mb": 270,
        "recommended": True,
    },
    "SmolLM2-360M": {
        "hf_id": "HuggingFaceTB/SmolLM2-360M",
        "description": "360M parameter model — good balance of size and quality",
        "approx_size_mb": 720,
        "recommended": True,
    },
    "TinyLlama-1.1B": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "1.1B parameter chat model — largest feasible on 8GB RAM",
        "approx_size_mb": 2200,
        "recommended": False,  # Tight on 8GB RAM
    },
    "GPT2": {
        "hf_id": "gpt2",
        "description": "124M parameter classic model — great baseline",
        "approx_size_mb": 500,
        "recommended": True,
    },
    "GPT2-Medium": {
        "hf_id": "gpt2-medium",
        "description": "355M parameter model",
        "approx_size_mb": 1400,
        "recommended": False,
    },
}


# ──────────────────────────────────────────────────────────
# Quantization Methods
# ──────────────────────────────────────────────────────────
QUANTIZATION_METHODS = {
    "baseline": {
        "description": "FP32 baseline (no quantization)",
        "requires_calibration": False,
    },
    "dynamic": {
        "description": "PyTorch Dynamic Quantization (INT8) — no calibration needed",
        "requires_calibration": False,
    },
    "gptq": {
        "description": "GPTQ (INT4/INT8) — post-training weight quantization with second-order correction",
        "requires_calibration": True,
    },
    "onnx": {
        "description": "ONNX Runtime Dynamic Quantization (INT8) — cross-platform deployment",
        "requires_calibration": False,
    },
}


# ──────────────────────────────────────────────────────────
# Benchmark Configuration
# ──────────────────────────────────────────────────────────
@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    # Model
    model_name: str = "SmolLM2-135M"
    model_hf_id: Optional[str] = None  # Auto-resolved from model_name

    # Quantization
    methods: list = field(default_factory=lambda: ["baseline", "dynamic"])
    gptq_bits: int = 4  # 4 or 8 for GPTQ

    # Benchmark parameters
    prompt: str = "The future of artificial intelligence on edge devices is"
    max_new_tokens: int = 100
    num_warmup_runs: int = 2
    num_benchmark_runs: int = 5
    batch_size: int = 1

    # Perplexity evaluation
    eval_perplexity: bool = True
    perplexity_dataset: str = "wikitext"
    perplexity_config: str = "wikitext-2-raw-v1"
    perplexity_split: str = "test"
    perplexity_max_samples: int = 200  # Limit for CPU speed
    perplexity_stride: int = 512

    # Output
    results_dir: Optional[str] = None  # Auto-set based on model name
    save_charts: bool = True
    verbose: bool = True

    def __post_init__(self):
        """Resolve model HF ID and results directory."""
        if self.model_hf_id is None:
            if self.model_name in SUPPORTED_MODELS:
                self.model_hf_id = SUPPORTED_MODELS[self.model_name]["hf_id"]
            else:
                # Assume it's a direct HF model ID
                self.model_hf_id = self.model_name

        if self.results_dir is None:
            safe_name = self.model_name.replace("/", "_").replace("-", "_").lower()
            self.results_dir = str(RESULTS_DIR / safe_name)

        Path(self.results_dir).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────
# System Info Helper
# ──────────────────────────────────────────────────────────
def get_system_info() -> dict:
    """Gather system information for benchmark reports."""
    import platform
    import psutil
    import cpuinfo

    cpu_info = cpuinfo.get_cpu_info()
    mem = psutil.virtual_memory()

    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_brand": cpu_info.get("brand_raw", "Unknown"),
        "cpu_arch": cpu_info.get("arch", "Unknown"),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_available_gb": round(mem.available / (1024**3), 2),
    }

    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["torch_num_threads"] = torch.get_num_threads()
    except ImportError:
        info["pytorch_version"] = "not installed"

    try:
        import onnxruntime
        info["onnxruntime_version"] = onnxruntime.__version__
    except ImportError:
        info["onnxruntime_version"] = "not installed"

    return info
