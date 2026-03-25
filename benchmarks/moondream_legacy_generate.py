#!/usr/bin/env python3
"""
Generate benchmark predictions using the legacy Moondream ONNX/Moonfile runtime.

Run this from the disposable `.venv-moondream05` environment, then score the
output with the main `benchmarks/caption_eval.py --predictions ...` flow.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import moondream as md


DEFAULT_QUERY_PROMPT = (
    "Describe what matters in this image."
)


def load_ground_truth(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
        return {
            str(item["filename"]): str(item["caption"])
            for item in payload["predictions"]
            if isinstance(item, dict) and "filename" in item and "caption" in item
        }
    if isinstance(payload, dict):
        return {str(k): str(v) for k, v in payload.items()}
    raise ValueError(f"Unsupported predictions file format: {path}")


def write_predictions(path: Path, model_name: str, predictions: dict[str, str]) -> None:
    payload = {
        "model": model_name,
        "predictions": [
            {"filename": filename, "caption": caption}
            for filename, caption in predictions.items()
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_image(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            subprocess.run(
                [
                    "python3",
                    "-c",
                    (
                        "from PIL import Image; import sys; "
                        "img = Image.open(sys.argv[1]).convert('RGB'); "
                        "img.save(sys.argv[2], format='PNG')"
                    ),
                    str(path),
                    str(tmp_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            image = Image.open(tmp_path).convert("RGB")
        finally:
            tmp_path.unlink(missing_ok=True)
        return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate predictions with legacy Moondream 0.5B runtime.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/moondream/moondream-0_5b-int8.mf"),
        help="Path to a local Moondream Moonfile model.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("test_resources"),
        help="Directory containing benchmark images.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("benchmarks/image_caption_ground_truth.json"),
        help="Ground-truth file used for image ordering.",
    )
    parser.add_argument(
        "--mode",
        choices=["query", "caption-short", "caption-normal"],
        default="query",
        help="Legacy Moondream generation mode to use.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_QUERY_PROMPT,
        help="Prompt to use for query mode.",
    )
    parser.add_argument(
        "--write-predictions",
        type=Path,
        required=True,
        help="Output JSON file for predictions.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing predictions file if present.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of images to process.",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    image_specs = ground_truth["images"]
    if args.limit:
        image_specs = image_specs[: args.limit]

    predictions: dict[str, str] = {}
    if args.resume and args.write_predictions.exists():
        predictions = load_predictions(args.write_predictions)

    model_name = f"moondream-legacy:{args.model_path.name}:{args.mode}"
    model = md.vl(model=str(args.model_path))

    for index, image_spec in enumerate(image_specs, start=1):
        filename = image_spec["filename"]
        image_path = args.image_dir / filename
        if filename in predictions:
            print(f"[{index}/{len(image_specs)}] skipping {filename} (already present)", file=sys.stderr, flush=True)
            continue

        print(f"[{index}/{len(image_specs)}] captioning {filename}...", file=sys.stderr, flush=True)
        image = load_image(image_path)
        if args.mode == "query":
            result = model.query(image, args.prompt)["answer"]
        else:
            length = args.mode.removeprefix("caption-").replace("normal", "normal")
            result = model.caption(image, length=length)["caption"]

        predictions[filename] = clean_text(str(result))
        write_predictions(args.write_predictions, model_name, predictions)

    print(f"Wrote {len(predictions)} predictions to {args.write_predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
