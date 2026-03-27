#!/bin/bash
# QuantBench — Run All Benchmarks
# Runs benchmarks for all recommended models with all available methods
#
# Usage:
#   bash scripts/run_all.sh              # Full benchmark
#   bash scripts/run_all.sh --quick      # Quick mode (no perplexity)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate venv if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

QUICK_MODE=false
if [ "$1" = "--quick" ]; then
    QUICK_MODE=true
    echo "[INFO] Quick mode: skipping perplexity evaluation"
fi

echo "============================================"
echo "  QuantBench — Full Benchmark Suite"
echo "============================================"
echo ""

# ─── Benchmark 1: SmolLM2-135M (smallest, fastest) ───────
echo "━━━ Benchmarking SmolLM2-135M ━━━"
if [ "$QUICK_MODE" = true ]; then
    python -m quantbench.cli \
        --model SmolLM2-135M \
        --methods baseline dynamic static onnx \
        --max-tokens 50 \
        --benchmark-runs 3 \
        --no-perplexity
else
    python -m quantbench.cli \
        --model SmolLM2-135M \
        --methods baseline dynamic static onnx \
        --max-tokens 100 \
        --benchmark-runs 5 \
        --perplexity-samples 100
fi

echo ""

# ─── Benchmark 2: GPT2 (well-known baseline) ─────────────
echo "━━━ Benchmarking GPT2 ━━━"
if [ "$QUICK_MODE" = true ]; then
    python -m quantbench.cli \
        --model GPT2 \
        --methods baseline dynamic static onnx \
        --max-tokens 50 \
        --benchmark-runs 3 \
        --no-perplexity
else
    python -m quantbench.cli \
        --model GPT2 \
        --methods baseline dynamic static onnx \
        --max-tokens 100 \
        --benchmark-runs 5 \
        --perplexity-samples 100
fi

echo ""
echo "============================================"
echo "  All benchmarks complete!"
echo "  Results saved in: results/"
echo "============================================"
echo ""

# List generated reports
echo "Generated reports:"
find results/ -name "report.md" -type f 2>/dev/null || echo "  (none found)"
echo ""
echo "Generated charts:"
find results/ -name "*.png" -type f 2>/dev/null || echo "  (none found)"
