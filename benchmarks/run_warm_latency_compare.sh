#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/perception-bench-py312/bin/python}"
DEVICE="${DEVICE:-mps}"
GROUND_TRUTH="${GROUND_TRUTH:-benchmarks/image_caption_ground_truth.json}"
IMAGE_DIR="${IMAGE_DIR:-test_resources}"
OFFLINE="${OFFLINE:-0}"
LIMIT="${LIMIT:-}"
WARMUP_PASSES="${WARMUP_PASSES:-1}"
MEASURE_PASSES="${MEASURE_PASSES:-1}"
MOONDREAM_MODE="${MOONDREAM_MODE:-query}"
MOONDREAM_COMPILE="${MOONDREAM_COMPILE:-1}"
MOONDREAM_MODEL_ID="${MOONDREAM_MODEL_ID:-vikhyatk/moondream2}"
WRITE_JSON="${WRITE_JSON:-}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

cmd=(
  "$PYTHON_BIN"
  benchmarks/warm_latency.py
  --device "$DEVICE"
  --ground-truth "$GROUND_TRUTH"
  --image-dir "$IMAGE_DIR"
  --warmup-passes "$WARMUP_PASSES"
  --measure-passes "$MEASURE_PASSES"
  --moondream-mode "$MOONDREAM_MODE"
  --moondream-model-id "$MOONDREAM_MODEL_ID"
)

if [ "$MOONDREAM_COMPILE" = "1" ]; then
  cmd+=(--moondream-compile)
fi

if [ "$OFFLINE" = "1" ]; then
  cmd+=(--offline)
fi

if [ -n "$LIMIT" ]; then
  cmd+=(--limit "$LIMIT")
fi

if [ -n "$WRITE_JSON" ]; then
  cmd+=(--write-json "$WRITE_JSON")
fi

"${cmd[@]}"
