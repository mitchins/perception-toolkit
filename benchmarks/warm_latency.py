#!/usr/bin/env python3
"""
Warm latency benchmark for the main image-captioning contenders.

This measures steady-state latency after an untimed warmup pass so the results
better reflect a sidecar process that keeps models loaded.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.caption_eval import (
    DEFAULT_MODEL_IDS,
    FlorenceLocalBackend,
    MoondreamBackend,
    configure_hf_runtime,
    load_ground_truth,
)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_image_paths(image_dir: Path, ground_truth: Path | None, limit: int | None) -> list[Path]:
    if ground_truth and ground_truth.exists():
        payload = load_ground_truth(ground_truth)
        image_specs = payload["images"]
        if limit:
            image_specs = image_specs[:limit]
        return [image_dir / image_spec["filename"] for image_spec in image_specs]

    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    if limit:
        image_paths = image_paths[:limit]
    return image_paths


def build_backend(kind: str, model_id: str, device: str, moondream_mode: str, moondream_compile: bool):
    if kind == "florence-base":
        return FlorenceLocalBackend(
            model_id=model_id,
            device=device,
            task_name="more-detailed-caption",
        )
    if kind == "florence-large":
        return FlorenceLocalBackend(
            model_id=model_id,
            device=device,
            task_name="more-detailed-caption",
        )
    if kind == "moondream2":
        return MoondreamBackend(
            model_id=model_id,
            device=device,
            patched=False,
            inference_mode=moondream_mode,
            compile_model=moondream_compile,
        )
    raise ValueError(f"Unsupported benchmark kind: {kind}")


def resolve_backend_device(kind: str, backend: Any) -> str:
    if kind.startswith("florence"):
        return getattr(backend.florence, "_device", getattr(backend, "device", "unknown"))
    return getattr(backend, "device", "unknown")


def release_backend(kind: str, backend: Any) -> None:
    try:
        if kind.startswith("florence"):
            backend.florence._model = None
            backend.florence._processor = None
            backend.florence._device = "cpu"
        if hasattr(backend, "model"):
            backend.model = None
        if hasattr(backend, "clip_model"):
            backend.clip_model = None
        if hasattr(backend, "gpt_model"):
            backend.gpt_model = None
    finally:
        del backend
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                mps = getattr(torch, "mps", None)
                if mps and hasattr(mps, "empty_cache"):
                    mps.empty_cache()
        except Exception:
            pass


def run_pass(backend: Any, image_paths: list[Path]) -> list[dict[str, Any]]:
    timings: list[dict[str, Any]] = []
    for image_path in image_paths:
        started = time.perf_counter()
        backend.caption(image_path)
        elapsed = time.perf_counter() - started
        timings.append(
            {
                "filename": image_path.name,
                "seconds": elapsed,
            }
        )
    return timings


def benchmark_model(
    *,
    label: str,
    kind: str,
    model_id: str,
    device: str,
    image_paths: list[Path],
    warmup_passes: int,
    measure_passes: int,
    moondream_mode: str,
    moondream_compile: bool,
) -> dict[str, Any]:
    init_started = time.perf_counter()
    backend = build_backend(
        kind=kind,
        model_id=model_id,
        device=device,
        moondream_mode=moondream_mode,
        moondream_compile=moondream_compile,
    )
    init_seconds = time.perf_counter() - init_started

    warmup_started = time.perf_counter()
    for _ in range(warmup_passes):
        run_pass(backend, image_paths)
    warmup_seconds = time.perf_counter() - warmup_started

    resolved_device = resolve_backend_device(kind, backend)

    measured: list[dict[str, Any]] = []
    for _ in range(measure_passes):
        measured.extend(run_pass(backend, image_paths))

    all_times = [item["seconds"] for item in measured]
    result = {
        "label": label,
        "kind": kind,
        "model_id": model_id,
        "requested_device": device,
        "resolved_device": resolved_device,
        "init_seconds": init_seconds,
        "warmup_seconds": warmup_seconds,
        "timed_images": len(all_times),
        "steady_total_seconds": sum(all_times),
        "steady_avg_seconds": average(all_times),
        "steady_p50_seconds": percentile(all_times, 50),
        "steady_p95_seconds": percentile(all_times, 95),
        "per_image": measured,
    }

    release_backend(kind, backend)
    return result


def print_report(results: list[dict[str, Any]]) -> None:
    print("Warm Latency Leaderboard")
    print("Model\tResolvedDevice\tInit(s)\tWarmup(s)\tAvg(s)\tP50(s)\tP95(s)\tImages")
    for item in sorted(results, key=lambda row: (row["steady_avg_seconds"], row["label"])):
        print(
            f"{item['label']}\t"
            f"{item['resolved_device']}\t"
            f"{item['init_seconds']:.2f}\t"
            f"{item['warmup_seconds']:.2f}\t"
            f"{item['steady_avg_seconds']:.3f}\t"
            f"{item['steady_p50_seconds']:.3f}\t"
            f"{item['steady_p95_seconds']:.3f}\t"
            f"{item['timed_images']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure warmed-up caption latency across the main benchmark contenders.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("test_resources"),
        help="Directory of benchmark images.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("benchmarks/image_caption_ground_truth.json"),
        help="Ground-truth file used for deterministic image ordering.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Requested torch device for all contenders.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force Hugging Face offline mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of images to use.",
    )
    parser.add_argument(
        "--warmup-passes",
        type=int,
        default=1,
        help="Untimed warmup passes across the full image set.",
    )
    parser.add_argument(
        "--measure-passes",
        type=int,
        default=1,
        help="Timed measurement passes across the full image set.",
    )
    parser.add_argument(
        "--moondream-mode",
        choices=["query", "caption-short", "caption-normal", "caption-long"],
        default="query",
        help="Moondream mode to benchmark.",
    )
    parser.add_argument(
        "--moondream-model-id",
        default=DEFAULT_MODEL_IDS["moondream-local"],
        help="Moondream model id override for the latency benchmark.",
    )
    parser.add_argument(
        "--moondream-compile",
        action="store_true",
        help="Call model.compile() after loading Moondream.",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        help="Optional JSON output path for raw timing data.",
    )
    args = parser.parse_args()

    configure_hf_runtime(args.offline)

    image_paths = load_image_paths(args.image_dir, args.ground_truth, args.limit)
    if not image_paths:
        print("No benchmark images found.", file=sys.stderr)
        return 1

    contenders = [
        {
            "label": "florence-base-more-detailed",
            "kind": "florence-base",
            "model_id": "microsoft/Florence-2-base",
        },
        {
            "label": "florence-large-more-detailed",
            "kind": "florence-large",
            "model_id": "microsoft/Florence-2-large",
        },
        {
            "label": f"{args.moondream_model_id.split('/')[-1]}-{args.moondream_mode}{'-compiled' if args.moondream_compile else ''}",
            "kind": "moondream2",
            "model_id": args.moondream_model_id,
        },
    ]

    results = []
    for contender in contenders:
        print(
            f"[timing] {contender['label']} on {args.device} "
            f"(warmup_passes={args.warmup_passes}, measure_passes={args.measure_passes})...",
            file=sys.stderr,
            flush=True,
        )
        result = benchmark_model(
            label=contender["label"],
            kind=contender["kind"],
            model_id=contender["model_id"],
            device=args.device,
            image_paths=image_paths,
            warmup_passes=args.warmup_passes,
            measure_passes=args.measure_passes,
            moondream_mode=args.moondream_mode,
            moondream_compile=args.moondream_compile,
        )
        results.append(result)

    print_report(results)

    if args.write_json:
        payload = {
            "image_count": len(image_paths),
            "image_names": [path.name for path in image_paths],
            "warmup_passes": args.warmup_passes,
            "measure_passes": args.measure_passes,
            "results": results,
        }
        args.write_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
