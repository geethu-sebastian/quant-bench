"""
PyTorch Dynamic Quantization (INT8).

Dynamic quantization is the simplest quantization workflow:
- Weights are quantized ahead of time (statically) to INT8
- Activations are quantized dynamically at runtime before each computation
- No calibration dataset is required
- Particularly effective for Linear, LSTM, and GRU layers

This maps to the "Dynamic / Runtime Quantization" workflow described in
the model compression primer (aman.ai/primers/ai/model-compression/).

Reference:
    PyTorch docs: https://pytorch.org/docs/stable/quantization.html
    torch.ao.quantization.quantize_dynamic
"""

import copy
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.ao.quantization as tq
from transformers import AutoModelForCausalLM

from quantbench.models import get_model_memory_footprint

logger = logging.getLogger(__name__)


def apply_dynamic_quantization(
    model: AutoModelForCausalLM,
    dtype: torch.dtype = torch.qint8,
    target_layers: Optional[set] = None,
) -> AutoModelForCausalLM:
    """Apply PyTorch dynamic quantization to a model.

    Dynamic quantization converts weight tensors from FP32 to INT8
    at model load time. Activations are dynamically quantized during
    inference (computed in FP32 then quantized before matrix multiply).

    This requires NO calibration data and is the fastest method to apply.

    Args:
        model: A loaded PyTorch model in eval mode.
        dtype: Quantized dtype, typically torch.qint8 (8-bit integer).
        target_layers: Set of nn.Module types to quantize.
                       Defaults to {nn.Linear} if None.

    Returns:
        The dynamically quantized model (new object, original unchanged).

    Technical Notes:
        - Only nn.Linear (and optionally nn.LSTM, nn.GRU) layers are quantized
        - Weights: FP32 → INT8 (static, at quantization time)
        - Activations: Remain FP32, quantized on-the-fly per inference call
        - Scale & zero-point for activations computed per-tensor at runtime
        - No retraining or fine-tuning involved (post-training method)
    """
    if target_layers is None:
        target_layers = {torch.nn.Linear}

    logger.info(
        f"Applying dynamic quantization (dtype={dtype}) "
        f"to layers: {[l.__name__ for l in target_layers]}"
    )

    # Get pre-quantization stats
    pre_stats = get_model_memory_footprint(model)

    start_time = time.time()

    # Deep copy to preserve original model
    model_copy = copy.deepcopy(model)

    # Apply dynamic quantization
    quantized_model = tq.quantize_dynamic(
        model_copy,
        qconfig_spec=target_layers,
        dtype=dtype,
    )

    elapsed = time.time() - start_time

    # Get post-quantization stats
    post_stats = get_model_memory_footprint(quantized_model)

    logger.info(
        f"Dynamic quantization complete in {elapsed:.2f}s | "
        f"Size: {pre_stats['total_mb']:.1f}MB → {post_stats['total_mb']:.1f}MB | "
        f"Compression: {pre_stats['total_mb'] / max(post_stats['total_mb'], 0.01):.2f}x"
    )

    return quantized_model


def save_dynamic_quantized(
    model: AutoModelForCausalLM,
    save_path: str,
) -> str:
    """Save a dynamically quantized model using torch.save.

    Note: Dynamic quantized models use PyTorch's native serialization
    since they contain quantized tensor types not supported by
    HuggingFace's save_pretrained.

    Args:
        model: The quantized model.
        save_path: Directory to save the model.

    Returns:
        Path to the saved model file.
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_file = save_dir / "quantized_model.pt"
    torch.save(model.state_dict(), model_file)

    logger.info(f"Dynamic quantized model saved to {model_file}")
    return str(model_file)


def get_quantization_summary(
    original_model: AutoModelForCausalLM,
    quantized_model: AutoModelForCausalLM,
) -> dict:
    """Generate a summary comparing original and quantized models.

    Returns:
        Dict with original_mb, quantized_mb, compression_ratio,
        quantized_layers_count, total_layers_count.
    """
    orig_stats = get_model_memory_footprint(original_model)
    quant_stats = get_model_memory_footprint(quantized_model)

    # Count quantized vs total linear layers
    total_linear = 0
    quantized_linear = 0
    for name, module in quantized_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            total_linear += 1
        elif isinstance(module, torch.ao.nn.quantized.dynamic.Linear):
            quantized_linear += 1

    return {
        "method": "PyTorch Dynamic Quantization",
        "precision": "INT8 (weights), FP32 (activations at runtime)",
        "original_mb": orig_stats["total_mb"],
        "quantized_mb": quant_stats["total_mb"],
        "compression_ratio": orig_stats["total_mb"] / max(quant_stats["total_mb"], 0.01),
        "quantized_layers": quantized_linear,
        "total_linear_layers": total_linear + quantized_linear,
        "requires_calibration": False,
        "requires_gpu": False,
    }
