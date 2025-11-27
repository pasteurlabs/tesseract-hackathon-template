#!/bin/bash
# Build all tesseracts in the project
# Assumes tesseract CLI is installed

set -e  # Exit on error

echo "========================================="
echo "Building Tesseract Hackathon Template"
echo "========================================="
echo ""

# Check if tesseract CLI is available
if ! command -v tesseract &> /dev/null; then
    echo "Error: tesseract CLI not found. Please install tesseract-core first."
    echo "Visit: https://github.com/pasteurlabs/tesseract-core"
    exit 1
fi

# Build scaler_jax
echo "[1/2] Building scaler..."
cd tesseracts/scaler
tesseract build .
cd ../..
echo "✓ scaler built successfully"
echo ""

# Build dotproduct_jax
echo "[2/2] Building dotproduct..."
cd tesseracts/dotproduct
tesseract build .
cd ../..
echo "✓ dotproduct built successfully"
echo ""

echo "========================================="
echo "✓ All tesseracts built successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Run 'python main.py' to see the demo pipeline"
echo "  2. Or use the tesseracts in your Python code with tesseract_utils.py"
