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
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

default_models=(
  "microsoft/Florence-2-base"
  "microsoft/Florence-2-base-ft"
  "microsoft/Florence-2-large"
  "microsoft/Florence-2-large-ft"
)

if [ "$#" -gt 0 ]; then
  models=("$@")
else
  models=("${default_models[@]}")
fi

IFS=' ' read -r -a tasks <<< "${TASKS:-caption detailed-caption more-detailed-caption}"

slugify() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed 's#/#_#g; s#-#_#g; s/florence_2/florence2/g; s/[^a-z0-9_]/_/g; s/__*/_/g; s/^_//; s/_$//'
}

prediction_files=()

for model_id in "${models[@]}"; do
  model_slug="$(slugify "${model_id##*/}")"

  for task in "${tasks[@]}"; do
    task_slug="$(slugify "$task")"
    output_path="benchmarks/${model_slug}_${task_slug}_predictions.json"
    prediction_files+=("$output_path")

    cmd=(
      "$PYTHON_BIN"
      benchmarks/caption_eval.py
      --backend florence-local
      --device "$DEVICE"
      --model-id "$model_id"
      --florence-task "$task"
      --ground-truth "$GROUND_TRUTH"
      --image-dir "$IMAGE_DIR"
      --write-predictions "$output_path"
    )

    if [ "$OFFLINE" = "1" ]; then
      cmd+=(--offline)
    fi
    if [ "$RESUME" = "1" ]; then
      cmd+=(--resume)
    fi
    if [ -n "$LIMIT" ]; then
      cmd+=(--limit "$LIMIT")
    fi

    echo
    echo "==> ${model_id} :: ${task}"
    "${cmd[@]}"
  done
done

echo
echo "==> Florence leaderboard"
"$PYTHON_BIN" benchmarks/caption_eval.py --predictions "${prediction_files[@]}" --leaderboard
