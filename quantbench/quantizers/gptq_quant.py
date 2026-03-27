"""
GPTQ Quantization (INT4 / INT8 weight-only).

GPTQ is a post-training quantization method designed for large language models.
It uses approximate second-order information (Hessian-based) to minimize the
layer-wise reconstruction error when quantizing weights.

Key properties:
- Weight-only quantization (activations remain in FP16/FP32)
- Uses a small calibration dataset to compute Hessian information
- Quantizes weights column-by-column with error compensation
- Achieves INT4 quantization with minimal perplexity degradation
- Particularly effective for transformer architectures

This maps to the "GPTQ: Quantization with Second-Order Error Compensation"
section in the model compression primer (aman.ai/primers/ai/model-compression/).

References:
    Paper: "GPTQ: Accurate Post-Training Quantization for
           Generative Pre-trained Transformers" (Frantar et al., 2023)
    Implementation: AutoGPTQ (https://github.com/PanQiWei/AutoGPTQ)
"""

import logging
import time
from pathlib import Path
from typing import Optional, List

import torch
from transformers import AutoTokenizer

from quantbench.models import resolve_model_id, get_model_memory_footprint

logger = logging.getLogger(__name__)

# Flag for availability
GPTQ_AVAILABLE = False
try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    GPTQ_AVAILABLE = True
except ImportError:
    logger.warning(
        "auto-gptq not installed. GPTQ quantization will not be available. "
        "Install with: pip install auto-gptq"
    )


def check_gptq_available() -> bool:
    """Check if AutoGPTQ is installed and available."""
    return GPTQ_AVAILABLE


def prepare_calibration_data(
    tokenizer: AutoTokenizer,
    num_samples: int = 128,
    max_length: int = 512,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
) -> List[dict]:
    """Prepare calibration data for GPTQ quantization.

    GPTQ requires a small calibration dataset to compute the Hessian
    (second-order sensitivity information) for each layer. The Hessian
    H = 2 * X * X^T captures how sensitive each weight is to quantization.

    Args:
        tokenizer: The model's tokenizer.
        num_samples: Number of calibration samples (128 is typical).
        max_length: Maximum sequence length per sample.
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration.
        split: Dataset split to use.

    Returns:
        List of tokenized examples (dicts with 'input_ids' and 'attention_mask').
    """
    from datasets import load_dataset

    logger.info(
        f"Preparing calibration data: {num_samples} samples from "
        f"{dataset_name}/{dataset_config} ({split} split)"
    )

    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Filter out empty texts
    texts = [
        item["text"]
        for item in dataset
        if item["text"].strip() and len(item["text"]) > 50
    ][:num_samples]

    # Tokenize
    calibration_data = []
    for text in texts:
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        calibration_data.append(
            {
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
            }
        )

    logger.info(f"Prepared {len(calibration_data)} calibration samples")
    return calibration_data


def apply_gptq_quantization(
    model_name: str,
    tokenizer: AutoTokenizer,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = False,
    num_calibration_samples: int = 128,
    calibration_max_length: int = 512,
    save_path: Optional[str] = None,
) -> object:
    """Apply GPTQ quantization to a model.

    GPTQ Process (per layer):
    1. Collect input activations X from calibration data
    2. Compute Hessian: H = 2 * X * X^T + λI
    3. For each column q of weight matrix W:
       a. Quantize: w_hat = quant(w_q)
       b. Compute error: δ = -(w_q - w_hat) / H^{-1}_{qq}
       c. Update remaining weights: W_{:,q+1:} += δ * H^{-1}_{:,q}
    4. This minimizes ‖WX - Ŵ X‖² (layer-wise reconstruction error)

    Args:
        model_name: Short name or HuggingFace model ID.
        tokenizer: The model's tokenizer.
        bits: Quantization bits (4 or 8). INT4 gives ~4x compression.
        group_size: Number of weights sharing the same scale/zero-point.
                    128 is standard; smaller = better accuracy, larger model.
        desc_act: If True, quantize in order of descending activation magnitude.
                  Better accuracy but slower and incompatible with some kernels.
        num_calibration_samples: Number of calibration examples.
        calibration_max_length: Max token length of calibration samples.
        save_path: If provided, save the quantized model here.

    Returns:
        The GPTQ-quantized model.
    """
    if not GPTQ_AVAILABLE:
        raise RuntimeError(
            "auto-gptq is not installed. Install with: pip install auto-gptq"
        )

    hf_id = resolve_model_id(model_name)

    logger.info(
        f"Starting GPTQ quantization: model={hf_id}, bits={bits}, "
        f"group_size={group_size}, desc_act={desc_act}"
    )

    # Configure GPTQ quantization
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=True,  # Symmetric quantization
    )

    # Load model for quantization
    logger.info("Loading model for GPTQ quantization...")
    start_time = time.time()

    model = AutoGPTQForCausalLM.from_pretrained(
        hf_id,
        quantize_config=quantize_config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Prepare calibration data
    calibration_data = prepare_calibration_data(
        tokenizer=tokenizer,
        num_samples=num_calibration_samples,
        max_length=calibration_max_length,
    )

    # Run GPTQ quantization
    logger.info(
        f"Running GPTQ quantization with {len(calibration_data)} "
        f"calibration samples (this may take a while on CPU)..."
    )

    model.quantize(calibration_data)

    elapsed = time.time() - start_time
    logger.info(f"GPTQ quantization completed in {elapsed:.1f}s")

    # Save if requested
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_quantized(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        logger.info(f"GPTQ model saved to {save_dir}")

    return model


def load_gptq_model(
    model_path: str,
    device: str = "cpu",
) -> object:
    """Load a pre-quantized GPTQ model.

    This can load:
    - Models quantized by this tool
    - Pre-quantized models from HuggingFace Hub (e.g., TheBloke models)

    Args:
        model_path: Path to local GPTQ model or HuggingFace model ID.
        device: Target device.

    Returns:
        The loaded GPTQ model.
    """
    if not GPTQ_AVAILABLE:
        raise RuntimeError("auto-gptq is not installed.")

    logger.info(f"Loading GPTQ model from {model_path}...")

    model = AutoGPTQForCausalLM.from_quantized(
        model_path,
        device=device,
        use_safetensors=True,
        trust_remote_code=True,
    )

    return model


def get_quantization_summary(bits: int, group_size: int) -> dict:
    """Generate a summary of GPTQ quantization settings.

    Returns:
        Dict with method details and theoretical compression info.
    """
    # Theoretical compression: FP32 (32 bits) → INT4 (4 bits) = 8x
    # With group_size overhead: each group needs scale (FP16) + zero_point (FP16)
    overhead = 32 / group_size  # bits of overhead per weight
    effective_bits = bits + overhead
    compression_ratio = 32 / effective_bits

    return {
        "method": f"GPTQ (INT{bits})",
        "precision": f"INT{bits} weights (group_size={group_size}), FP32 activations",
        "theoretical_compression": f"{compression_ratio:.2f}x",
        "bits_per_weight": bits,
        "group_size": group_size,
        "effective_bits_per_weight": round(effective_bits, 2),
        "requires_calibration": True,
        "requires_gpu": False,  # Runs on CPU (slower but works)
        "calibration_dataset": "WikiText-2",
    }
