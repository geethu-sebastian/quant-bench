"""
Model loading utilities for QuantBench.

Handles downloading, caching, and loading HuggingFace models
with memory-conscious settings for CPU-only environments.
"""

import gc
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from quantbench.config import SUPPORTED_MODELS, MODELS_DIR

logger = logging.getLogger(__name__)


def resolve_model_id(model_name: str) -> str:
    """Resolve a short model name to its HuggingFace model ID.

    Args:
        model_name: Either a key from SUPPORTED_MODELS (e.g., 'SmolLM2-135M')
                    or a direct HuggingFace model ID (e.g., 'gpt2').

    Returns:
        The full HuggingFace model identifier.
    """
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]["hf_id"]
    return model_name


def get_model_info(model_name: str) -> dict:
    """Get metadata about a model.

    Returns:
        Dict with keys: hf_id, description, approx_size_mb, recommended, param_count
    """
    hf_id = resolve_model_id(model_name)
    info = SUPPORTED_MODELS.get(model_name, {}).copy()
    info["hf_id"] = hf_id

    try:
        config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        # Estimate parameter count from config
        if hasattr(config, "num_parameters"):
            info["param_count"] = config.num_parameters
        elif hasattr(config, "n_embd") and hasattr(config, "n_layer"):
            # Rough estimate for GPT-style models
            d = config.n_embd
            L = config.n_layer
            V = getattr(config, "vocab_size", 50257)
            info["param_count_estimate"] = V * d + L * (12 * d * d) + V * d
    except Exception as e:
        logger.debug(f"Could not fetch config for param estimation: {e}")

    return info


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load the tokenizer for a given model.

    Args:
        model_name: Short name or HuggingFace model ID.

    Returns:
        The loaded tokenizer with padding configured.
    """
    hf_id = resolve_model_id(model_name)
    logger.info(f"Loading tokenizer for {hf_id}...")

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=True,
    )

    # Ensure pad token is set (required for batched generation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def load_model(
    model_name: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    low_memory: bool = True,
) -> AutoModelForCausalLM:
    """Load a HuggingFace causal LM model for CPU inference.

    Args:
        model_name: Short name or HuggingFace model ID.
        dtype: torch.float32 (default) or torch.float16.
        device: Target device ('cpu').
        low_memory: If True, use memory-efficient loading strategies.

    Returns:
        The loaded model in eval mode.
    """
    hf_id = resolve_model_id(model_name)
    logger.info(f"Loading model {hf_id} (dtype={dtype}, device={device})...")

    # Force garbage collection before loading
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    load_kwargs = {
        "pretrained_model_name_or_path": hf_id,
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "device_map": None,  # Explicit CPU placement
    }

    if low_memory:
        load_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    model = model.to(device)
    model.eval()

    # Log model stats
    param_count = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    logger.info(
        f"Model loaded: {param_count:,} parameters, "
        f"{size_mb:.1f} MB in memory ({dtype})"
    )

    return model


def load_model_and_tokenizer(
    model_name: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Convenience function to load both model and tokenizer.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, dtype=dtype, device=device)
    return model, tokenizer


def get_model_size_on_disk(model_path: str) -> float:
    """Calculate the total size of a saved model on disk in MB.

    Args:
        model_path: Path to the saved model directory.

    Returns:
        Total size in megabytes.
    """
    total_size = 0
    model_dir = Path(model_path)

    if not model_dir.exists():
        return 0.0

    for f in model_dir.rglob("*"):
        if f.is_file():
            total_size += f.stat().st_size

    return total_size / (1024 ** 2)


def get_model_memory_footprint(model: torch.nn.Module) -> dict:
    """Calculate the in-memory footprint of a loaded model.

    Returns:
        Dict with total_mb, param_mb, buffer_mb, param_count.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    param_count = sum(p.numel() for p in model.parameters())

    return {
        "total_mb": (param_size + buffer_size) / (1024 ** 2),
        "param_mb": param_size / (1024 ** 2),
        "buffer_mb": buffer_size / (1024 ** 2),
        "param_count": param_count,
    }


def cleanup_model(model: Optional[torch.nn.Module] = None):
    """Release model from memory and force garbage collection.

    Important for CPU-constrained environments to prevent OOM
    when loading multiple quantized variants sequentially.
    """
    if model is not None:
        del model
    gc.collect()
    logger.debug("Model cleaned up, garbage collected.")
