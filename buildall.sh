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


for tess_dir in tesseracts/*/
do
    echo "Building ${tess_dir}"
    tesseract build ${tess_dir}
    echo "✓ ${tess_dir} built successfully"
    echo ""
done


echo "========================================="
echo "✓ All tesseracts built successfully!"
echo "========================================="
