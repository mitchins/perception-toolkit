#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/perception-bench-py312/bin/python}"
DEVICE="${DEVICE:-mps}"
GROUND_TRUTH="${GROUND_TRUTH:-benchmarks/image_caption_ground_truth.json}"
IMAGE_DIR="${IMAGE_DIR:-test_resources}"
OFFLINE="${OFFLINE:-0}"
RESUME="${RESUME:-1}"
LIMIT="${LIMIT:-}"
MOONDREAM_MODEL_ID="${MOONDREAM_MODEL_ID:-vikhyatk/moondream2}"
FLORENCE_MODEL_ID="${FLORENCE_MODEL_ID:-microsoft/Florence-2-base}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

common_args=(
  --device "$DEVICE"
  --ground-truth "$GROUND_TRUTH"
  --image-dir "$IMAGE_DIR"
)

if [ "$OFFLINE" = "1" ]; then
  common_args+=(--offline)
fi
if [ "$RESUME" = "1" ]; then
  common_args+=(--resume)
fi
if [ -n "$LIMIT" ]; then
  common_args+=(--limit "$LIMIT")
fi

run_eval() {
  local label="$1"
  shift
  echo
  echo "==> ${label}"
  "$PYTHON_BIN" benchmarks/caption_eval.py "${common_args[@]}" "$@"
}

run_eval \
  "Florence base :: more-detailed-caption" \
  --backend florence-local \
  --model-id "$FLORENCE_MODEL_ID" \
  --florence-task more-detailed-caption \
  --write-predictions benchmarks/florence2_base_more_detailed_caption_predictions.json

run_eval \
  "Moondream2 :: official :: caption-long" \
  --backend moondream-local \
  --model-id "$MOONDREAM_MODEL_ID" \
  --moondream-loader official \
  --moondream-mode caption-long \
  --write-predictions benchmarks/moondream2_official_caption_long_predictions.json

run_eval \
  "Moondream2 :: official :: query" \
  --backend moondream-local \
  --model-id "$MOONDREAM_MODEL_ID" \
  --moondream-loader official \
  --moondream-mode query \
  --write-predictions benchmarks/moondream2_official_query_predictions.json

echo
echo "==> Florence vs Moondream leaderboard"
"$PYTHON_BIN" benchmarks/caption_eval.py \
  --predictions \
  benchmarks/florence2_base_more_detailed_caption_predictions.json \
  benchmarks/moondream2_official_caption_long_predictions.json \
  benchmarks/moondream2_official_query_predictions.json \
  --leaderboard
