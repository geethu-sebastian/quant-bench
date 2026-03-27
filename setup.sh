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

# Attempt to install python3-venv on Debian/Ubuntu systems if needed
if command -v apt-get &> /dev/null; then
    if ! python3 -m venv -h &> /dev/null; then
        echo "[INFO] python3-venv module is missing. Attempting to install it..."
        apt-get update -y
        # Try generic first, then specific version based on current python version
        apt-get install -y python3-venv "python${ACTUAL_MAJOR}.${ACTUAL_MINOR}-venv" || true
    fi
fi

# Create virtual environment
echo ""
echo "[1/4] Creating virtual environment..."
if ! python3 -m venv venv; then
    echo "[ERROR] Failed to create virtual environment."
    echo "Please manually run: apt install python3.12-venv (or your specific version)"
    echo "Then re-run this setup script."
    exit 1
fi
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
echo "[4/5] Installing general project dependencies..."
pip install -r requirements.txt

echo ""
echo "[5/5] Compiling AutoGPTQ for CPU (this may take a minute)..."
# AutoGPTQ tries to build CUDA extensions by default and fails in isolated environments
# and on CPU-only machines. We explicitly disable CUDA and use the already-installed torch.
BUILD_CUDA_EXT=0 pip install auto-gptq --no-build-isolation

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
