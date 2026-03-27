"""
ONNX Runtime Quantization (INT8).

Exports a PyTorch model to ONNX format, then applies ONNX Runtime's
built-in quantization — enabling cross-platform deployment on any
hardware that supports ONNX (including Qualcomm Snapdragon via QNN EP).

ONNX quantization supports two modes:
1. Dynamic quantization: No calibration needed, quantizes weights & activations
   dynamically (similar to PyTorch dynamic quantization).
2. Static quantization: Uses calibration data for optimal activation ranges.

This is particularly relevant to the Qualcomm JD because:
- ONNX is the standard interchange format for Snapdragon AI deployment
- Qualcomm AI Engine Direct (QNN) consumes ONNX models
- ONNX Runtime supports CPU, GPU, and NPU execution providers

References:
    ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
    Qualcomm AI Hub: https://aihub.qualcomm.com/
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantbench.models import get_model_memory_footprint

logger = logging.getLogger(__name__)

# Check ONNX Runtime availability
ONNX_AVAILABLE = False
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        quantize_dynamic,
        quantize_static,
        QuantType,
        CalibrationDataReader,
    )
    ONNX_AVAILABLE = True
except ImportError:
    logger.warning(
        "onnxruntime or onnx not installed. ONNX quantization unavailable. "
        "Install with: pip install onnx onnxruntime optimum[onnxruntime]"
    )


def check_onnx_available() -> bool:
    """Check if ONNX Runtime is available."""
    return ONNX_AVAILABLE


def export_to_onnx(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    save_path: str,
    opset_version: int = 14,
    max_length: int = 128,
) -> str:
    """Export a PyTorch model to ONNX format.

    The ONNX export traces the model's computation graph with sample inputs,
    creating a portable representation that can be optimized by any ONNX-
    compatible runtime (ONNX Runtime, TensorRT, Qualcomm QNN, etc.).

    Args:
        model: The PyTorch model to export.
        tokenizer: The model's tokenizer (for creating dummy inputs).
        save_path: Directory to save the ONNX model.
        opset_version: ONNX opset version (14+ recommended for transformers).
        max_length: Sequence length for the dummy input.

    Returns:
        Path to the exported ONNX model file.
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = save_dir / "model.onnx"

    logger.info(f"Exporting model to ONNX (opset={opset_version})...")

    # Create dummy inputs
    dummy_text = "The quick brown fox jumps over the lazy dog"
    dummy_input = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    # Export with dynamic axes for variable sequence length
    start_time = time.time()

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    elapsed = time.time() - start_time

    # Validate the exported model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    model_size_mb = onnx_path.stat().st_size / (1024 ** 2)
    logger.info(
        f"ONNX export complete in {elapsed:.1f}s | "
        f"Model size: {model_size_mb:.1f} MB | Path: {onnx_path}"
    )

    return str(onnx_path)


def export_with_optimum(
    model_name: str,
    save_path: str,
) -> str:
    """Export using HuggingFace Optimum for better ONNX compatibility.

    Optimum handles complex model architectures (KV cache, attention masks)
    that torch.onnx.export may struggle with.

    Args:
        model_name: HuggingFace model ID or local path.
        save_path: Directory to save the ONNX model.

    Returns:
        Path to the export directory.
    """
    try:
        from optimum.onnxruntime import ORTModelForCausalLM

        logger.info(f"Exporting {model_name} to ONNX via Optimum...")

        from quantbench.models import resolve_model_id
        hf_id = resolve_model_id(model_name)

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        model = ORTModelForCausalLM.from_pretrained(
            hf_id,
            export=True,
            trust_remote_code=True,
        )
        model.save_pretrained(str(save_dir))

        logger.info(f"Optimum ONNX export saved to {save_dir}")
        return str(save_dir)

    except ImportError:
        logger.warning("optimum not installed, falling back to torch.onnx.export")
        raise
    except Exception as e:
        logger.warning(f"Optimum export failed: {e}, falling back to torch.onnx.export")
        raise


def apply_onnx_dynamic_quantization(
    onnx_model_path: str,
    save_path: Optional[str] = None,
) -> str:
    """Apply ONNX Runtime dynamic quantization to an ONNX model.

    Dynamic quantization in ONNX Runtime:
    - Quantizes weights to INT8 at load time
    - Activations are quantized dynamically per inference call
    - No calibration data needed
    - Supports MatMul, Attention, Conv, and other common ops

    Args:
        onnx_model_path: Path to the FP32 ONNX model.
        save_path: Path for the quantized model. Auto-generated if None.

    Returns:
        Path to the quantized ONNX model.
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError("ONNX Runtime not installed.")

    if save_path is None:
        p = Path(onnx_model_path)
        save_path = str(p.parent / f"{p.stem}_quantized_dynamic{p.suffix}")

    logger.info(f"Applying ONNX dynamic quantization to {onnx_model_path}...")
    start_time = time.time()

    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=save_path,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    )

    elapsed = time.time() - start_time

    original_mb = Path(onnx_model_path).stat().st_size / (1024 ** 2)
    quantized_mb = Path(save_path).stat().st_size / (1024 ** 2)

    logger.info(
        f"ONNX dynamic quantization complete in {elapsed:.1f}s | "
        f"Size: {original_mb:.1f}MB → {quantized_mb:.1f}MB | "
        f"Compression: {original_mb / max(quantized_mb, 0.01):.2f}x"
    )

    return save_path


class WikiTextCalibrationReader(CalibrationDataReader):
    """Calibration data reader for ONNX static quantization.

    Provides batches of tokenized WikiText-2 data to the ONNX Runtime
    quantizer for computing optimal activation scale/zero-point values.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_samples: int = 50,
        max_length: int = 128,
    ):
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [
            item["text"]
            for item in dataset
            if item["text"].strip() and len(item["text"]) > 50
        ][:num_samples]

        self.data = []
        for text in texts:
            tokenized = tokenizer(
                text,
                return_tensors="np",
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )
            self.data.append({
                "input_ids": tokenized["input_ids"].astype(np.int64),
                "attention_mask": tokenized["attention_mask"].astype(np.int64),
            })

        self.index = 0

    def get_next(self):
        if self.index >= len(self.data):
            return None
        data = self.data[self.index]
        self.index += 1
        return data

    def rewind(self):
        self.index = 0


def apply_onnx_static_quantization(
    onnx_model_path: str,
    tokenizer: AutoTokenizer,
    save_path: Optional[str] = None,
    num_calibration_samples: int = 50,
) -> str:
    """Apply ONNX Runtime static quantization with calibration.

    Args:
        onnx_model_path: Path to the FP32 ONNX model.
        tokenizer: Tokenizer for generating calibration data.
        save_path: Output path. Auto-generated if None.
        num_calibration_samples: Number of calibration samples.

    Returns:
        Path to the quantized ONNX model.
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError("ONNX Runtime not installed.")

    if save_path is None:
        p = Path(onnx_model_path)
        save_path = str(p.parent / f"{p.stem}_quantized_static{p.suffix}")

    logger.info(f"Applying ONNX static quantization to {onnx_model_path}...")

    calibration_reader = WikiTextCalibrationReader(
        tokenizer=tokenizer,
        num_samples=num_calibration_samples,
    )

    start_time = time.time()

    try:
        quantize_static(
            model_input=onnx_model_path,
            model_output=save_path,
            calibration_data_reader=calibration_reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            per_channel=True,
        )
    except Exception as e:
        logger.warning(
            f"ONNX static quantization failed: {e}. "
            f"Falling back to dynamic quantization."
        )
        return apply_onnx_dynamic_quantization(onnx_model_path, save_path)

    elapsed = time.time() - start_time

    original_mb = Path(onnx_model_path).stat().st_size / (1024 ** 2)
    quantized_mb = Path(save_path).stat().st_size / (1024 ** 2)

    logger.info(
        f"ONNX static quantization complete in {elapsed:.1f}s | "
        f"Size: {original_mb:.1f}MB → {quantized_mb:.1f}MB | "
        f"Compression: {original_mb / max(quantized_mb, 0.01):.2f}x"
    )

    return save_path


def run_onnx_inference(
    onnx_model_path: str,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> dict:
    """Run inference using ONNX Runtime for benchmarking.

    Uses Optimum's ORTModelForCausalLM to properly handle 
    KV cache (past_key_values) and position_ids automatically.

    Args:
        onnx_model_path: Path to the ONNX model.
        tokenizer: The tokenizer.
        prompt: Input prompt text.
        max_new_tokens: Number of tokens to generate.

    Returns:
        Dict with generated_text, num_tokens, total_time_s, tokens_per_second.
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError("ONNX Runtime not installed.")

    from optimum.onnxruntime import ORTModelForCausalLM
    import torch

    model_dir = Path(onnx_model_path).parent.as_posix()
    file_name = Path(onnx_model_path).name

    model = ORTModelForCausalLM.from_pretrained(
        model_dir,
        file_name=file_name,
        provider="CPUExecutionProvider",
        use_cache=True,
        trust_remote_code=True
    )

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Autoregressive generation using HuggingFace's generate() loop
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    total_time = time.time() - start_time
    
    num_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return {
        "generated_text": generated_text,
        "num_tokens": num_generated,
        "total_time_s": total_time,
        "tokens_per_second": num_generated / max(total_time, 1e-6),
    }


def get_quantization_summary(
    original_onnx_path: str,
    quantized_onnx_path: str,
    method: str = "dynamic",
) -> dict:
    """Generate a summary of ONNX quantization."""
    original_mb = Path(original_onnx_path).stat().st_size / (1024 ** 2)
    quantized_mb = Path(quantized_onnx_path).stat().st_size / (1024 ** 2)

    return {
        "method": f"ONNX Runtime {'Dynamic' if method == 'dynamic' else 'Static'} Quantization",
        "precision": "INT8 (weights" + (", activations" if method == "static" else "") + ")",
        "format": "ONNX",
        "original_mb": original_mb,
        "quantized_mb": quantized_mb,
        "compression_ratio": original_mb / max(quantized_mb, 0.01),
        "requires_calibration": method == "static",
        "requires_gpu": False,
        "cross_platform": True,
        "supported_eps": ["CPUExecutionProvider", "QNNExecutionProvider (Snapdragon)"],
    }
