"""
Perplexity evaluation for QuantBench.

Perplexity measures how well a language model predicts a held-out test set.
Lower perplexity = better model quality. It is the standard metric for
evaluating the accuracy impact of quantization on language models.

Formula:
    PPL = exp( -1/N * Σ log P(token_i | context_i) )

Where:
    - N = total number of tokens
    - P(token_i | context_i) = model's predicted probability of token_i
      given all preceding tokens as context

A perplexity of 1.0 = perfect prediction (impossible in practice)
Typical LLM perplexity on WikiText-2: 5-30 depending on model size

For quantization evaluation:
    - Baseline FP32 perplexity: ~X
    - Quantized perplexity: ~X + Δ
    - If Δ < 0.5: excellent quantization quality
    - If Δ < 2.0: acceptable quantization quality
    - If Δ > 5.0: significant accuracy degradation
"""

import logging
import math
import time
from typing import Optional

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def evaluate_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: int = 200,
    stride: int = 512,
    max_length: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Evaluate perplexity of a model on a dataset.

    Uses the sliding window approach recommended by HuggingFace:
    - Concatenate all text into a single long sequence
    - Use a sliding window of size `max_length` with stride `stride`
    - Compute NLL loss for each window, masking already-seen tokens
    - Final perplexity = exp(mean NLL)

    This approach handles the fact that transformers have a fixed context
    window and avoids the per-sentence boundary effects.

    Args:
        model: The model to evaluate (FP32 or quantized).
        tokenizer: The model's tokenizer.
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset config name.
        split: Dataset split ('test' is standard for evaluation).
        max_samples: Maximum number of text samples to use.
        stride: Sliding window stride in tokens.
        max_length: Context window size. Defaults to model's max_position_embeddings.
        verbose: Show progress bar.

    Returns:
        Dict with perplexity, avg_nll, num_tokens, eval_time_s.
    """
    logger.info(
        f"Evaluating perplexity on {dataset_name}/{dataset_config} "
        f"(split={split}, max_samples={max_samples})"
    )

    start_time = time.time()

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Concatenate all text
    texts = [item["text"] for item in dataset if item["text"].strip()][:max_samples]
    full_text = "\n\n".join(texts)

    # Tokenize the entire text
    encodings = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )

    input_ids = encodings["input_ids"]
    seq_len = input_ids.shape[1]

    # Determine context window
    if max_length is None:
        if hasattr(model.config, "max_position_embeddings"):
            max_length = model.config.max_position_embeddings
        else:
            max_length = 1024

    # Cap max_length to avoid OOM on CPU
    max_length = min(max_length, 1024)

    logger.info(
        f"Evaluating on {seq_len} tokens with "
        f"window={max_length}, stride={stride}"
    )

    # Sliding window evaluation
    nlls = []
    total_tokens = 0

    num_windows = max(1, (seq_len - max_length) // stride + 1)
    iterator = range(0, seq_len - 1, stride)
    if verbose:
        iterator = tqdm(iterator, desc="Perplexity eval", total=num_windows)

    model.eval()

    for begin_loc in iterator:
        end_loc = min(begin_loc + max_length, seq_len)
        target_len = end_loc - begin_loc - 1

        if target_len <= 0:
            break

        input_chunk = input_ids[:, begin_loc:end_loc]

        # Create labels: mask everything before the stride position
        # so we don't count previously-seen tokens
        target_ids = input_chunk.clone()
        if begin_loc > 0:
            # Mask first (max_length - stride) tokens as they were seen before
            overlap = max_length - stride
            if overlap > 0:
                target_ids[:, :overlap] = -100

        with torch.no_grad():
            try:
                outputs = model(input_ids=input_chunk, labels=target_ids)
                nll = outputs.loss.item()
            except Exception as e:
                # For quantized models that don't support labels
                outputs = model(input_ids=input_chunk)
                logits = outputs.logits

                # Manual cross-entropy calculation
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_chunk[:, 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
                nll = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).item()

        if not math.isnan(nll) and not math.isinf(nll):
            nlls.append(nll)
            total_tokens += target_len

        if end_loc >= seq_len:
            break

    eval_time = time.time() - start_time

    # Calculate perplexity
    if len(nlls) == 0:
        logger.error("No valid NLL values computed!")
        return {
            "perplexity": float("inf"),
            "avg_nll": float("inf"),
            "num_tokens": 0,
            "eval_time_s": eval_time,
        }

    avg_nll = float(np.mean(nlls))
    perplexity = math.exp(avg_nll)

    logger.info(
        f"Perplexity: {perplexity:.2f} | "
        f"Avg NLL: {avg_nll:.4f} | "
        f"Tokens evaluated: {total_tokens:,} | "
        f"Time: {eval_time:.1f}s"
    )

    return {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "num_tokens": total_tokens,
        "num_windows": len(nlls),
        "eval_time_s": eval_time,
        "dataset": f"{dataset_name}/{dataset_config}",
        "split": split,
        "max_length": max_length,
        "stride": stride,
    }


def evaluate_perplexity_onnx(
    onnx_model_path: str,
    tokenizer: AutoTokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: int = 100,
    max_length: int = 512,
    verbose: bool = True,
) -> dict:
    """Evaluate perplexity of an ONNX model.

    Uses Optimum's ORTModelForCausalLM.

    Args:
        onnx_model_path: Path to the ONNX model.
        tokenizer: The tokenizer.
        dataset_name: Dataset name.
        dataset_config: Dataset config.
        split: Dataset split.
        max_samples: Max text samples.
        max_length: Max sequence length per window.
        verbose: Show progress.

    Returns:
        Dict with perplexity metrics.
    """
    from optimum.onnxruntime import ORTModelForCausalLM
    from pathlib import Path

    logger.info(f"Evaluating ONNX model perplexity on {dataset_name}/{dataset_config}...")

    start_time = time.time()

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    texts = [item["text"] for item in dataset if item["text"].strip()][:max_samples]
    full_text = "\n\n".join(texts)

    encodings = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"]
    seq_len = input_ids.shape[1]

    # Create ONNX session via Optimum
    model_dir = Path(onnx_model_path).parent.as_posix()
    file_name = Path(onnx_model_path).name

    model = ORTModelForCausalLM.from_pretrained(
        model_dir,
        file_name=file_name,
        provider="CPUExecutionProvider",
        use_cache=False,
        trust_remote_code=True
    )

    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
    stride = max_length // 2

    num_windows = max(1, (seq_len - max_length) // stride + 1)
    iterator = range(0, min(seq_len - 1, max_samples * max_length), stride)
    if verbose:
        iterator = tqdm(iterator, desc="ONNX perplexity eval", total=min(num_windows, max_samples))

    for begin_loc in iterator:
        end_loc = min(begin_loc + max_length, seq_len)
        if end_loc - begin_loc < 10:
            break

        chunk_ids = input_ids[:, begin_loc:end_loc]

        try:
            with torch.no_grad():
                outputs = model(input_ids=chunk_ids)
            
            logits = outputs.logits
            labels = chunk_ids[0, 1:]
            shift_logits = logits[0, :-1, :].contiguous()

            nll = loss_fct(shift_logits, labels).item()

            if not math.isnan(nll) and not math.isinf(nll):
                nlls.append(nll)
        except Exception as e:
            logger.debug(f"Error in ONNX perplexity window: {e}")
            continue

        if end_loc >= seq_len:
            break

    eval_time = time.time() - start_time

    if len(nlls) == 0:
        return {
            "perplexity": float("inf"),
            "avg_nll": float("inf"),
            "num_tokens": 0,
            "eval_time_s": eval_time,
        }

    avg_nll = float(np.mean(nlls))
    perplexity = math.exp(min(avg_nll, 100))  # Cap to prevent overflow

    logger.info(f"ONNX Perplexity: {perplexity:.2f} | Avg NLL: {avg_nll:.4f}")

    return {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "num_windows": len(nlls),
        "eval_time_s": eval_time,
        "dataset": f"{dataset_name}/{dataset_config}",
    }
