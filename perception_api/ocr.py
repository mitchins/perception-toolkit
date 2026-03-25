"""
RapidOCR backend for the perception sidecar.

Provides structured OCR output with line text, confidences, and quadrilateral
boxes. This is the preferred OCR path over generative caption models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from perception_api.config import get_config

log = logging.getLogger(__name__)

_engine: Any = None
_loaded: bool = False


def is_available() -> bool:
    """Check if the OCR backend is enabled in config."""
    return get_config().ocr.enabled


def ensure_loaded() -> None:
    """Load the RapidOCR engine if not already loaded."""
    global _engine, _loaded

    if _loaded:
        return

    cfg = get_config().ocr
    if not cfg.enabled:
        raise RuntimeError("OCR backend is disabled in configuration.")

    try:
        from rapidocr import RapidOCR

        started = perf_counter()
        log.info("Loading RapidOCR engine ...")
        _engine = RapidOCR()
        _loaded = True
        log.info("RapidOCR engine loaded successfully in %.2fs.", perf_counter() - started)
    except Exception as e:
        log.error("Failed to load RapidOCR engine: %s", e)
        raise


def extract_text(
    image_path: Path,
    threshold: float | None = None,
    *,
    return_word_box: bool = False,
) -> dict[str, Any]:
    """
    Run OCR on an image and return structured text results.

    Returns:
      {
        "full_text": str,
        "lines": [{"text": str, "confidence": float, "bbox": [[x, y], ...]}],
        "elapsed_ms": float,
      }
    """
    ensure_loaded()

    cfg = get_config().ocr
    score_threshold = _clamp_threshold(cfg.score_threshold_default if threshold is None else threshold)

    started = perf_counter()
    result = _engine(str(image_path), return_word_box=return_word_box)
    elapsed_ms = (perf_counter() - started) * 1000.0

    if result is None:
        log.info("RapidOCR inference complete image=%s lines=0 total_ms=%.1f", image_path.name, elapsed_ms)
        return {"full_text": "", "lines": [], "elapsed_ms": elapsed_ms}

    boxes_raw = getattr(result, "boxes", None)
    txts_raw = getattr(result, "txts", None)
    scores_raw = getattr(result, "scores", None)
    boxes = list(boxes_raw) if boxes_raw is not None else []
    txts = list(txts_raw) if txts_raw is not None else []
    scores = list(scores_raw) if scores_raw is not None else []

    lines: list[dict[str, Any]] = []
    for idx, text in enumerate(txts):
        text = str(text).strip()
        if not text:
            continue
        confidence = float(scores[idx]) if idx < len(scores) else 0.0
        if confidence < score_threshold:
            continue
        bbox = _normalize_box(boxes[idx]) if idx < len(boxes) else []
        lines.append(
            {
                "text": text,
                "confidence": round(confidence, 4),
                "bbox": bbox,
            }
        )

    full_text = "\n".join(line["text"] for line in lines)
    log.info(
        "RapidOCR inference complete image=%s lines=%d total_ms=%.1f",
        image_path.name,
        len(lines),
        elapsed_ms,
    )
    return {"full_text": full_text, "lines": lines, "elapsed_ms": elapsed_ms}


def format_text_for_llm(full_text: str, lines: list[dict[str, Any]]) -> str:
    """Format OCR results as concise text for LLM consumption."""
    if not lines:
        return "No readable text was found above the OCR confidence threshold."

    text_lines = ["Exact OCR transcription from the image:"]
    if full_text:
        text_lines.append(full_text)
    else:
        for line in lines:
            text_lines.append(line["text"])

    text_lines.append("")
    text_lines.append("Line-level OCR confidence:")
    for line in lines:
        text_lines.append(f"- {line['text']} ({line['confidence']:.2f})")

    return "\n".join(text_lines)


def _normalize_box(box: Any) -> list[list[float]]:
    """Convert a box-like object into a JSON-friendly quadrilateral."""
    if box is None:
        return []
    normalized: list[list[float]] = []
    for point in box:
        try:
            normalized.append([round(float(point[0]), 1), round(float(point[1]), 1)])
        except (IndexError, KeyError, TypeError, ValueError):
            continue
    return normalized


def _clamp_threshold(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
