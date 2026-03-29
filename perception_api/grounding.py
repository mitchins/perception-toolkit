"""
Grounding DINO backend for lightweight UI element proposals.

This backend is intended as an exploratory screenshot/UI tool. It produces
open-vocabulary region proposals from a prompt and can attach rough OCR
context from overlapping text lines when the OCR backend is enabled.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

from perception_api import ocr
from perception_api.config import get_config
from perception_api.devices import resolve_torch_device
from perception_api.image_codecs import load_image_rgb

log = logging.getLogger(__name__)

_processor: Any = None
_model: Any = None
_device: str = "cpu"
_loaded: bool = False
_runtime_info: dict[str, Any] = {}
_MAX_BOX_AREA_FRACTION = 0.85
_MAX_FULLSPAN_WIDTH_FRACTION = 0.97
_MAX_FULLSPAN_HEIGHT_FRACTION = 0.97
_MAX_SECTION_WIDTH_FRACTION = 0.90
_MAX_SECTION_HEIGHT_FRACTION = 0.60
_MIN_CONFIDENCE_FOR_SECTION_BOX = 0.35


def is_available() -> bool:
    """Check if the Grounding DINO backend is enabled in config."""
    return get_config().grounding.enabled


def get_runtime_info() -> dict[str, Any]:
    """Return lightweight runtime diagnostics for the grounding backend."""
    cfg = get_config().grounding
    return {
        "loaded": _loaded,
        "model_id": cfg.model_id,
        "device": _device,
        **_runtime_info,
    }


def ensure_loaded() -> None:
    """Load the Grounding DINO model if not already loaded."""
    global _processor, _model, _device, _loaded, _runtime_info

    if _loaded:
        return

    cfg = get_config().grounding
    if not cfg.enabled:
        raise RuntimeError("Grounding backend is disabled in configuration.")

    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        started = perf_counter()
        _device = resolve_torch_device(cfg.device, log)
        log.info("Loading grounding model %s on %s ...", cfg.model_id, _device)
        _processor = AutoProcessor.from_pretrained(cfg.model_id)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.model_id)
        _model = _model.to(_device).eval()
        _loaded = True
        _runtime_info = {
            "torch_version": getattr(torch, "__version__", ""),
            "load_ms": round((perf_counter() - started) * 1000.0, 1),
        }
        log.info(
            "Grounding model loaded successfully model=%s device=%s load_ms=%.1f",
            cfg.model_id,
            _device,
            _runtime_info["load_ms"],
        )
    except Exception as e:
        log.error("Failed to load grounding model: %s", e)
        raise


def detect_ui_elements(
    image_path: Path,
    prompt: str = "",
    box_threshold: float | None = None,
    text_threshold: float | None = None,
    max_detections: int | None = None,
    include_ocr_context: bool | None = None,
) -> dict[str, Any]:
    """
    Run lightweight screenshot/UI grounding and return structured region proposals.
    """
    ensure_loaded()

    import torch

    cfg = get_config().grounding
    prompt_used = _normalize_prompt(prompt or cfg.prompt_default)
    box_thresh = _clamp_threshold(
        cfg.box_threshold_default if box_threshold is None else box_threshold
    )
    text_thresh = _clamp_threshold(
        cfg.text_threshold_default if text_threshold is None else text_threshold
    )
    max_det = max(1, int(cfg.max_detections if max_detections is None else max_detections))
    use_ocr_context = (
        cfg.include_ocr_context_default if include_ocr_context is None else include_ocr_context
    )

    started = perf_counter()
    image = load_image_rgb(image_path)
    target_sizes = [image.size[::-1]]
    inputs = _processor(images=image, text=prompt_used, return_tensors="pt")
    input_ids = inputs["input_ids"]
    model_inputs = {
        key: value.to(_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = _model(**model_inputs)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The key `labels` is will return integer ids.*Use `text_labels` instead.*",
            category=FutureWarning,
            module=r"transformers\.models\.grounding_dino\.processing_grounding_dino",
        )
        result = _processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=target_sizes,
        )[0]

    labels = result.get("text_labels", result.get("labels", []))
    elements = []
    for label, score, box in zip(
        labels,
        result.get("scores", []),
        result.get("boxes", []),
    ):
        confidence = round(float(score), 4)
        bbox = [round(float(value), 1) for value in box.tolist()]
        if _is_oversized_box(bbox, image.size, confidence):
            continue
        elements.append(
            {
                "label": str(label).strip(),
                "confidence": confidence,
                "bbox": bbox,
                "ocr_text": "",
            }
        )

    elements.sort(key=lambda item: item["confidence"], reverse=True)
    elements = elements[:max_det]

    ocr_used = False
    if use_ocr_context and elements and ocr.is_available():
        try:
            ocr_result = ocr.extract_text(image_path)
            _attach_ocr_context(elements, ocr_result.get("lines", []))
            ocr_used = True
        except Exception as e:
            log.warning("Grounding OCR context failed image=%s error=%s", image_path.name, e)

    total_ms = (perf_counter() - started) * 1000.0
    log.info(
        "Grounding inference complete image=%s device=%s elements=%d ocr_context=%s total_ms=%.1f",
        image_path.name,
        _device,
        len(elements),
        ocr_used,
        total_ms,
    )
    return {
        "prompt_used": prompt_used,
        "elements": elements,
        "ocr_used": ocr_used,
        "elapsed_ms": total_ms,
        "device": _device,
    }


def format_ui_elements_for_llm(
    prompt_used: str,
    elements: list[dict[str, Any]],
) -> str:
    """Format UI grounding results as concise text for LLM consumption."""
    if not elements:
        return (
            "No UI elements matched the grounding prompt above the current thresholds.\n"
            f"Prompt used: {prompt_used}"
        )

    lines = [
        f"Grounded UI elements using prompt: {prompt_used}",
        "Elements:",
    ]
    for idx, item in enumerate(elements[:50], start=1):
        line = (
            f"- #{idx} {item['label']} ({item['confidence']:.2f}) "
            f"bbox={item['bbox']}"
        )
        if item.get("ocr_text"):
            line += f" text={item['ocr_text']!r}"
        lines.append(line)
    return "\n".join(lines)


def _attach_ocr_context(elements: list[dict[str, Any]], lines: list[dict[str, Any]]) -> None:
    """Attach rough OCR snippets to grounded elements using line-center overlap."""
    for element in elements:
        x1, y1, x2, y2 = element["bbox"]
        matched_texts: list[str] = []
        for line in lines:
            bbox = line.get("bbox", [])
            if not bbox:
                continue
            center_x, center_y = _box_center(bbox)
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                text = str(line.get("text", "")).strip()
                if text:
                    matched_texts.append(text)
        if matched_texts:
            element["ocr_text"] = " | ".join(dict.fromkeys(matched_texts))


def _box_center(quad: list[list[float]]) -> tuple[float, float]:
    xs = [float(point[0]) for point in quad]
    ys = [float(point[1]) for point in quad]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _normalize_prompt(value: str) -> str:
    text = " ".join(str(value).strip().split())
    return text if text.endswith(".") else f"{text} ."


def _clamp_threshold(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _is_oversized_box(
    bbox: list[float],
    image_size: tuple[int, int],
    confidence: float,
) -> bool:
    """Drop near-full-image boxes that are usually low-value UI proposals."""
    if len(bbox) != 4:
        return False

    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        return False

    x1, y1, x2, y2 = bbox
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    if box_width <= 0 or box_height <= 0:
        return True

    width_fraction = box_width / image_width
    height_fraction = box_height / image_height
    area_fraction = (box_width * box_height) / (image_width * image_height)
    return (
        area_fraction >= _MAX_BOX_AREA_FRACTION
        or (
            width_fraction >= _MAX_FULLSPAN_WIDTH_FRACTION
            and height_fraction >= _MAX_FULLSPAN_HEIGHT_FRACTION
        )
        or (
            confidence < _MIN_CONFIDENCE_FOR_SECTION_BOX
            and width_fraction >= _MAX_SECTION_WIDTH_FRACTION
            and height_fraction >= _MAX_SECTION_HEIGHT_FRACTION
        )
    )
