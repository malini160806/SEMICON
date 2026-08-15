#!/usr/bin/env bash
# Evaluate the trained model on a directory of test images.
# Usage: bash scripts/evaluate.sh <input_dir> <output_dir> [weights_path]
set -euo pipefail
cd "$(dirname "$0")/.."

INPUT_DIR="${1:-datasets/Test_NoisyLR}"
OUTPUT_DIR="${2:-outputs/test_restored}"
WEIGHTS="${3:-weights/best_model.pth}"

python evaluate.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --weights "$WEIGHTS"
