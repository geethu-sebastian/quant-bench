#!/bin/bash
# QuantBench — Setup Script
# Run this on your Linux machine to set up the project

set -e

echo "============================================"
echo "  QuantBench — Setup"
echo "============================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "[INFO] Python version: $PYTHON_VERSION"

REQUIRED_MAJOR=3
REQUIRED_MINOR=9
ACTUAL_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
ACTUAL_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$ACTUAL_MAJOR" -lt "$REQUIRED_MAJOR" ] || { [ "$ACTUAL_MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$ACTUAL_MINOR" -lt "$REQUIRED_MINOR" ]; }; then
    echo "[ERROR] Python 3.9+ required. Found: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo ""
echo "[1/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "[2/4] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU-only)
echo ""
echo "[3/4] Installing PyTorch (CPU-only)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
echo ""
echo "[4/4] Installing project dependencies..."
pip install -r requirements.txt

# Create output directories
mkdir -p results models

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Activate the environment:"
echo "    source venv/bin/activate"
echo ""
echo "  Run a quick benchmark:"
echo "    python -m quantbench.cli --model SmolLM2-135M --methods dynamic"
echo ""
echo "  Run all benchmarks:"
echo "    bash scripts/run_all.sh"
echo "============================================"
