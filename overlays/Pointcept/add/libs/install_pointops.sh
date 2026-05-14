#!/bin/bash
# Install script for pointops and pointops2 with proper environment setup

set -e

# Activate conda environment if not already activated
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: CONDA_PREFIX not set. Please activate the conda environment first."
    exit 1
fi

# Set up CUDA environment variables
export PATH="$CONDA_PREFIX/nvvm/bin:$PATH"
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"

echo "Setting up CUDA environment:"
echo "  PATH includes: $CONDA_PREFIX/nvvm/bin"
echo "  CUDA_HOME: $CUDA_HOME"
echo ""

# Verify cicc is accessible
if [ -f "$CONDA_PREFIX/nvvm/bin/cicc" ]; then
    echo "✓ Found cicc at $CONDA_PREFIX/nvvm/bin/cicc"
else
    echo "⚠ Warning: cicc not found at $CONDA_PREFIX/nvvm/bin/cicc"
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install pointops
echo ""
echo "Installing pointops..."
cd "$SCRIPT_DIR/pointops"
pip install --no-build-isolation -e . || {
    echo "Failed to install pointops"
    exit 1
}

# Install pointops2
echo ""
echo "Installing pointops2..."
cd "$SCRIPT_DIR/pointops2"
pip install --no-build-isolation -e . || {
    echo "Failed to install pointops2"
    exit 1
}

echo ""
echo "✓ All pointops libraries installed successfully!"

