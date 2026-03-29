#!/usr/bin/env python3
"""
Small sanity-check harness for screenshot classification plus Grounding DINO UI proposals.

This is intentionally lightweight. It is meant for development guardrails rather
than rigorous benchmarking. It runs the current screenshot-oriented stack on a
small set of local images, records timings and top detections, and applies a few
basic checks such as:
  - classifier predicts "screenshot"
  - classifier confidence clears a minimum
  - grounding returns at least N UI proposals

It requires an inference-capable environment such as the local `perception`
conda env and may download the Grounding DINO checkpoint on first use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception_api import classifier, grounding
from perception_api.config import reload_config

DEFAULT_IMAGES = [
    PROJECT_ROOT / "test_resources" / "duolingo_korean_unit.png",
    PROJECT_ROOT / "test_resources" / "Screenshot 2026-03-29 at 12.22.45.png",
    PROJECT_ROOT / "test_resources" / "Screenshot 2026-03-29 at 12.23.31.png",
    PROJECT_ROOT / "test_resources" / "Screenshot 2026-03-29 at 12.31.04.png",
]

DEFAULT_PROMPT = (
    "button . icon . menu . tab . input field . search box . text . "
    "toolbar . dialog . list item ."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small screenshot sanity-check against the current classifier + Grounding DINO stack."
    )
    parser.add_argument(
        "--image",
        action="append",
        type=Path,
        default=[],
        help="Local image to evaluate. Can be repeated. Defaults to the current screenshot seed set.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Grounding prompt to use for all images.",
    )
    parser.add_argument(
        "--grounding-model-id",
        default=os.environ.get("PERCEPTION_GROUNDING_MODEL_ID", "IDEA-Research/grounding-dino-tiny"),
        help="Grounding DINO model ID.",
    )
    parser.add_argument(
        "--grounding-device",
        default=os.environ.get("PERCEPTION_GROUNDING_DEVICE", "auto"),
        help="Grounding backend device.",
    )
    parser.add_argument(
        "--classifier-model-path",
        default=os.environ.get("PERCEPTION_CLASSIFIER_MODEL_PATH", "v2.8_draft_single_model_mcp"),
        help="Classifier model path or export directory.",
    )
    parser.add_argument(
        "--classifier-device",
        default=os.environ.get("PERCEPTION_CLASSIFIER_DEVICE", "auto"),
        help="Classifier backend device.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.20,
        help="Grounding DINO box threshold.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.20,
        help="Grounding DINO text threshold.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=10,
        help="Max UI proposals to keep per image.",
    )
    parser.add_argument(
        "--include-ocr-context",
        action="store_true",
        help="Attach overlapping OCR snippets to the grounded UI elements.",
    )
    parser.add_argument(
        "--expected-label",
        default="screenshot",
        help="Expected classifier label for all test images.",
    )
    parser.add_argument(
        "--min-classifier-confidence",
        type=float,
        default=0.90,
        help="Minimum acceptable classifier confidence.",
    )
    parser.add_argument(
        "--min-elements",
        type=int,
        default=2,
        help="Minimum acceptable number of UI proposals.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to benchmarks/ground_ui_eval_<timestamp>.json",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Exit 0 even when one or more checks fail.",
    )
    return parser.parse_args()


def resolve_images(args: argparse.Namespace) -> list[Path]:
    if args.image:
        return args.image
    return list(DEFAULT_IMAGES)


def configure_backends(args: argparse.Namespace) -> None:
    os.environ["PERCEPTION_CLASSIFIER_ENABLED"] = "true"
    os.environ["PERCEPTION_CLASSIFIER_MODEL_PATH"] = args.classifier_model_path
    os.environ["PERCEPTION_CLASSIFIER_DEVICE"] = args.classifier_device
    os.environ["PERCEPTION_GROUNDING_ENABLED"] = "true"
    os.environ["PERCEPTION_GROUNDING_MODEL_ID"] = args.grounding_model_id
    os.environ["PERCEPTION_GROUNDING_DEVICE"] = args.grounding_device
    reload_config()


def area_fraction(bbox: list[float], width: int, height: int) -> float | None:
    if len(bbox) != 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = bbox
    box_width = max(0.0, float(x2) - float(x1))
    box_height = max(0.0, float(y2) - float(y1))
    if box_width <= 0.0 or box_height <= 0.0:
        return 0.0
    return round((box_width * box_height) / (width * height), 4)


def evaluate_image(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    record: dict[str, Any] = {
        "file": path.name,
        "path": str(path),
        "exists": path.is_file(),
        "passed": False,
        "checks": {},
    }
    if not path.is_file():
        record["error"] = "Missing file"
        return record

    classifier_result = classifier.classify_image(path)
    grounding_result = grounding.detect_ui_elements(
        path,
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_detections=args.max_detections,
        include_ocr_context=args.include_ocr_context,
    )

    width = height = None
    top_boxes: list[dict[str, Any]] = []
    if grounding_result["elements"]:
        try:
            from PIL import Image

            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            width = height = None

    for item in grounding_result["elements"][:5]:
        top_boxes.append(
            {
                "label": item["label"],
                "confidence": item["confidence"],
                "bbox": item["bbox"],
                "bbox_area_fraction": area_fraction(item["bbox"], width or 0, height or 0),
                "ocr_text": item.get("ocr_text", ""),
            }
        )

    checks = {
        "classifier_label_ok": classifier_result["label"] == args.expected_label,
        "classifier_confidence_ok": classifier_result["confidence"] >= args.min_classifier_confidence,
        "grounding_elements_ok": len(grounding_result["elements"]) >= args.min_elements,
    }

    record.update(
        {
            "classifier": {
                "label": classifier_result["label"],
                "confidence": round(float(classifier_result["confidence"]), 4),
                "device": classifier_result["device"],
                "total_ms": round(float(classifier_result["total_ms"]), 1),
                "low_confidence": bool(classifier_result["low_confidence"]),
            },
            "grounding": {
                "prompt_used": grounding_result["prompt_used"],
                "device": grounding_result["device"],
                "elapsed_ms": round(float(grounding_result["elapsed_ms"]), 1),
                "element_count": len(grounding_result["elements"]),
                "ocr_used": bool(grounding_result["ocr_used"]),
                "top_elements": top_boxes,
            },
            "checks": checks,
            "passed": all(checks.values()),
        }
    )
    return record


def output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "benchmarks" / f"ground_ui_eval_{timestamp}.json"


def main() -> int:
    args = parse_args()
    images = resolve_images(args)
    configure_backends(args)

    results: list[dict[str, Any]] = []
    for path in images:
        try:
            results.append(evaluate_image(path, args))
        except Exception as exc:
            results.append(
                {
                    "file": path.name,
                    "path": str(path),
                    "exists": path.is_file(),
                    "passed": False,
                    "error": str(exc),
                    "checks": {},
                }
            )

    summary = {
        "total": len(results),
        "passed": sum(1 for item in results if item.get("passed")),
        "failed": sum(1 for item in results if not item.get("passed")),
        "expected_label": args.expected_label,
        "min_classifier_confidence": args.min_classifier_confidence,
        "min_elements": args.min_elements,
        "prompt": args.prompt,
        "grounding_model_id": args.grounding_model_id,
        "include_ocr_context": args.include_ocr_context,
    }

    payload = {
        "summary": summary,
        "results": results,
    }

    target = output_path(args)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote results to {target}")
    for item in results:
        status = "PASS" if item.get("passed") else "FAIL"
        if "error" in item:
            print(f"{status} {item['file']}: error={item['error']}")
            continue
        classifier_result = item["classifier"]
        grounding_result = item["grounding"]
        failed_checks = [name for name, ok in item["checks"].items() if not ok]
        suffix = f" failed_checks={','.join(failed_checks)}" if failed_checks else ""
        print(
            f"{status} {item['file']}: "
            f"classifier={classifier_result['label']}@{classifier_result['confidence']:.2f} "
            f"elements={grounding_result['element_count']} "
            f"classifier_ms={classifier_result['total_ms']:.1f} "
            f"grounding_ms={grounding_result['elapsed_ms']:.1f}"
            f"{suffix}"
        )

    if summary["failed"] and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
