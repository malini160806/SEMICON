#!/usr/bin/env bash
# Train the KLA/i4C image restoration model.
# Usage: bash scripts/train.sh [extra --set overrides...]
set -euo pipefail
cd "$(dirname "$0")/.."

python train.py --config config/config.yaml "$@"
