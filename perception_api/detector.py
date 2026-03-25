"""
Optional object detector backend for the perception sidecar.

Uses Ultralytics YOLO models to provide closed-vocabulary object detections,
counts, confidences, and rough bounding boxes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

from perception_api.config import get_config
from perception_api.devices import resolve_torch_device

log = logging.getLogger(__name__)

_model: Any = None
_device: str = "cpu"
_loaded: bool = False

_IRREGULAR_PLURALS = {
    "person": "people",
}


def is_available() -> bool:
    """Check if the detector backend is enabled in config."""
    return get_config().detector.enabled


def ensure_loaded() -> None:
    """Load the detector model if not already loaded."""
    global _model, _device, _loaded

    if _loaded:
        return

    cfg = get_config().detector
    if not cfg.enabled:
        raise RuntimeError("Detector backend is disabled in configuration.")

    try:
        from ultralytics import YOLO

        started = perf_counter()
        _device = resolve_torch_device(cfg.device, log)
        log.info("Loading detector model %s on %s ...", cfg.model_id, _device)
        _model = YOLO(cfg.model_id)
        _loaded = True
        log.info(
            "Detector model loaded successfully in %.2fs.",
            perf_counter() - started,
        )
    except Exception as e:
        log.error("Failed to load detector model: %s", e)
        raise


def detect_image(
    image_path: Path,
    threshold: float | None = None,
    iou_threshold: float | None = None,
    max_detections: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run detector inference on an image.

    Returns a list of raw detection dicts:
      {"label": str, "confidence": float, "bbox": [x1, y1, x2, y2]}
    """
    ensure_loaded()

    cfg = get_config().detector
    conf = _clamp_threshold(cfg.confidence_threshold_default if threshold is None else threshold)
    iou = _clamp_threshold(cfg.iou_threshold_default if iou_threshold is None else iou_threshold)
    max_det = int(cfg.max_detections if max_detections is None else max_detections)
    max_det = max(1, max_det)

    started = perf_counter()
    results = _model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        device=_device,
        max_det=max_det,
        verbose=False,
    )
    total_ms = (perf_counter() - started) * 1000.0

    if not results:
        log.info(
            "Detector inference complete image=%s device=%s detections=0 total_ms=%.1f",
            image_path.name,
            _device,
            total_ms,
        )
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        log.info(
            "Detector inference complete image=%s device=%s detections=0 total_ms=%.1f",
            image_path.name,
            _device,
            total_ms,
        )
        return []

    names = getattr(result, "names", {}) or getattr(_model, "names", {})
    detections: list[dict[str, Any]] = []
    for cls_id, conf_score, bbox in zip(
        boxes.cls.tolist(),
        boxes.conf.tolist(),
        boxes.xyxy.tolist(),
    ):
        label = _resolve_label(names, int(cls_id))
        detections.append(
            {
                "label": label,
                "confidence": float(conf_score),
                "bbox": [round(float(value), 1) for value in bbox],
            }
        )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    log.info(
        "Detector inference complete image=%s device=%s detections=%d total_ms=%.1f",
        image_path.name,
        _device,
        len(detections),
        total_ms,
    )
    return detections


def summarize_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group detections by label and emit count/max-confidence summaries."""
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "max_confidence": 0.0})

    for item in detections:
        label = str(item["label"])
        confidence = float(item["confidence"])
        grouped[label]["count"] += 1
        grouped[label]["max_confidence"] = max(grouped[label]["max_confidence"], confidence)

    summary = [
        {
            "label": label,
            "count": values["count"],
            "max_confidence": round(values["max_confidence"], 4),
        }
        for label, values in grouped.items()
    ]
    summary.sort(key=lambda item: (-item["count"], item["label"]))
    return summary


def format_detections_for_llm(
    grouped: list[dict[str, Any]],
    detections: list[dict[str, Any]],
) -> str:
    """Format detection results as concise text for LLM consumption."""
    if not detections:
        return "No supported detector objects were found above the confidence threshold."

    summary = ", ".join(
        f"{item['count']} {_pluralize(item['label'], item['count'])}"
        for item in grouped
    )

    lines = [f"Detected objects: {summary}."]
    lines.append("Detections:")
    for item in detections[:20]:
        bbox = item["bbox"]
        lines.append(
            f"- {item['label']} ({item['confidence']:.2f}) bbox={bbox}"
        )
    return "\n".join(lines)


def _resolve_label(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _clamp_threshold(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _pluralize(label: str, count: int) -> str:
    if count == 1:
        return label
    if label in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[label]
    if label.endswith("s"):
        return label
    return f"{label}s"
