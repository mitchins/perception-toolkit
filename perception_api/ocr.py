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
_fallback_engines: dict[str, Any] = {}
_failed_fallback_langs: set[str] = set()


def is_available() -> bool:
    """Check if the OCR backend is enabled in config."""
    return get_config().ocr.enabled


def get_runtime_info() -> dict[str, Any]:
    """Return OCR runtime/provider information for diagnostics."""
    cfg = get_config().ocr
    info: dict[str, Any] = {
        "loaded": _loaded,
        "use_coreml": cfg.use_coreml,
        "fallback_rec_langs": list(cfg.fallback_rec_langs),
        "available_providers": [],
        "session_providers": {},
        "loaded_fallback_rec_langs": sorted(_fallback_engines.keys()),
        "failed_fallback_rec_langs": sorted(_failed_fallback_langs),
    }
    try:
        import onnxruntime as ort

        info["available_providers"] = list(ort.get_available_providers())
    except Exception as e:
        info["available_providers_error"] = str(e)

    if _loaded and _engine is not None:
        info["session_providers"] = {
            "det": _session_providers(getattr(_engine, "text_det", None)),
            "cls": _session_providers(getattr(_engine, "text_cls", None)),
            "rec": _session_providers(getattr(_engine, "text_rec", None)),
        }
    return info


def ensure_loaded() -> None:
    """Load the RapidOCR engine if not already loaded."""
    global _engine, _loaded

    if _loaded:
        return

    cfg = get_config().ocr
    if not cfg.enabled:
        raise RuntimeError("OCR backend is disabled in configuration.")

    try:
        _engine = _load_engine(cfg.rec_lang, role="primary")
        _loaded = True
        runtime_info = get_runtime_info()
        session_providers = runtime_info.get("session_providers", {})
        log.info(
            "RapidOCR session providers det=%s cls=%s rec=%s",
            session_providers.get("det", []),
            session_providers.get("cls", []),
            session_providers.get("rec", []),
        )
        if cfg.use_coreml and not any(
            "CoreMLExecutionProvider" in providers
            for providers in session_providers.values()
        ):
            log.warning(
                "RapidOCR requested CoreML, but no OCR session reported CoreMLExecutionProvider. "
                "The runtime likely fell back to CPU. session_providers=%s",
                session_providers,
            )
        log.info("RapidOCR engine loaded successfully.")
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
    primary_result = _engine(str(image_path), return_word_box=return_word_box)
    primary_candidates = _extract_candidates(primary_result, cfg.rec_lang)
    fallback_candidates = _run_fallback_passes(
        image_path,
        return_word_box=return_word_box,
        fallback_rec_langs=_configured_fallback_rec_langs(cfg),
    )
    merged_lines = _merge_multilingual_candidates(
        primary_candidates + fallback_candidates,
        score_threshold=score_threshold,
    )
    elapsed_ms = (perf_counter() - started) * 1000.0

    full_text = "\n".join(line["text"] for line in merged_lines)
    log.info(
        "RapidOCR inference complete image=%s primary_candidates=%d fallback_candidates=%d lines=%d total_ms=%.1f",
        image_path.name,
        len(primary_candidates),
        len(fallback_candidates),
        len(merged_lines),
        elapsed_ms,
    )
    return {"full_text": full_text, "lines": merged_lines, "elapsed_ms": elapsed_ms}


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


def _load_engine(rec_lang: str, *, role: str) -> Any:
    """Create a RapidOCR engine for the requested recognition language."""
    from rapidocr import RapidOCR
    import onnxruntime as ort

    cfg = get_config().ocr
    started = perf_counter()
    params = _build_engine_params(rec_lang)
    providers = ort.get_available_providers()
    log.info("Loading RapidOCR %s engine ...", role)
    log.info(
        "RapidOCR %s config det_lang=%s rec_lang=%s rec_version=%s fallback_rec_langs=%s use_coreml=%s providers=%s",
        role,
        cfg.det_lang,
        rec_lang,
        cfg.rec_version,
        cfg.fallback_rec_langs,
        cfg.use_coreml,
        providers,
    )
    if cfg.use_coreml:
        log.info(
            "RapidOCR CoreML config format=%s compute_units=%s require_static_input_shapes=%s "
            "enable_on_subgraphs=%s specialization_strategy=%s profile_compute_plan=%s "
            "allow_low_precision_accumulation_on_gpu=%s model_cache_directory=%s",
            cfg.coreml_model_format,
            cfg.coreml_compute_units,
            cfg.coreml_require_static_input_shapes,
            cfg.coreml_enable_on_subgraphs,
            cfg.coreml_specialization_strategy,
            cfg.coreml_profile_compute_plan,
            cfg.coreml_allow_low_precision_accumulation_on_gpu,
            cfg.coreml_model_cache_directory,
        )
    engine = RapidOCR(params=params)
    log.info("RapidOCR %s engine loaded in %.2fs.", role, perf_counter() - started)
    return engine


def _build_engine_params(rec_lang: str) -> dict[str, Any]:
    """Build RapidOCR params for a recognition language."""
    from rapidocr import LangDet, LangRec, OCRVersion

    cfg = get_config().ocr
    params: dict[str, Any] = {
        "Det.lang_type": _parse_enum(LangDet, cfg.det_lang, "ocr.det_lang"),
        "Rec.lang_type": _parse_enum(LangRec, rec_lang, "ocr.rec_lang"),
        "Rec.ocr_version": _parse_enum(OCRVersion, cfg.rec_version, "ocr.rec_version"),
    }
    if cfg.use_coreml:
        params.update(
            {
                "EngineConfig.onnxruntime.use_coreml": True,
                "EngineConfig.onnxruntime.coreml_ep_cfg.ModelFormat": cfg.coreml_model_format,
                "EngineConfig.onnxruntime.coreml_ep_cfg.MLComputeUnits": cfg.coreml_compute_units,
                "EngineConfig.onnxruntime.coreml_ep_cfg.RequireStaticInputShapes": cfg.coreml_require_static_input_shapes,
                "EngineConfig.onnxruntime.coreml_ep_cfg.EnableOnSubgraphs": cfg.coreml_enable_on_subgraphs,
                "EngineConfig.onnxruntime.coreml_ep_cfg.SpecializationStrategy": cfg.coreml_specialization_strategy,
                "EngineConfig.onnxruntime.coreml_ep_cfg.ProfileComputePlan": cfg.coreml_profile_compute_plan,
                "EngineConfig.onnxruntime.coreml_ep_cfg.AllowLowPrecisionAccumulationOnGPU": cfg.coreml_allow_low_precision_accumulation_on_gpu,
                "EngineConfig.onnxruntime.coreml_ep_cfg.ModelCacheDirectory": cfg.coreml_model_cache_directory,
            }
        )
    return params


def _configured_fallback_rec_langs(cfg) -> list[str]:
    """Return ordered, de-duplicated fallback recognizers excluding the primary one."""
    seen: set[str] = set()
    ordered: list[str] = []
    for rec_lang in cfg.fallback_rec_langs:
        rec_lang = str(rec_lang).strip()
        if not rec_lang or rec_lang == cfg.rec_lang or rec_lang in seen:
            continue
        seen.add(rec_lang)
        ordered.append(rec_lang)
    return ordered


def _ensure_fallback_engine(rec_lang: str) -> Any | None:
    """Load a fallback recognizer once, or skip it permanently after a failure."""
    if rec_lang in _fallback_engines:
        return _fallback_engines[rec_lang]
    if rec_lang in _failed_fallback_langs:
        return None

    try:
        engine = _load_engine(rec_lang, role=f"fallback:{rec_lang}")
    except Exception as e:
        _failed_fallback_langs.add(rec_lang)
        log.warning("Skipping OCR fallback recognizer rec_lang=%s: %s", rec_lang, e)
        return None

    _fallback_engines[rec_lang] = engine
    return engine


def _run_fallback_passes(
    image_path: Path,
    *,
    return_word_box: bool,
    fallback_rec_langs: list[str],
) -> list[dict[str, Any]]:
    """Run configured fallback recognizers and return raw candidate lines."""
    candidates: list[dict[str, Any]] = []
    if not fallback_rec_langs:
        return candidates

    for rec_lang in fallback_rec_langs:
        engine = _ensure_fallback_engine(rec_lang)
        if engine is None:
            continue
        try:
            started = perf_counter()
            result = engine(str(image_path), return_word_box=return_word_box)
            pass_candidates = _extract_candidates(result, rec_lang)
            log.info(
                "RapidOCR fallback inference complete image=%s rec_lang=%s candidates=%d total_ms=%.1f",
                image_path.name,
                rec_lang,
                len(pass_candidates),
                (perf_counter() - started) * 1000.0,
            )
            candidates.extend(pass_candidates)
        except Exception as e:
            log.warning(
                "RapidOCR fallback inference failed image=%s rec_lang=%s error=%s",
                image_path.name,
                rec_lang,
                e,
            )
    return candidates


def _extract_candidates(result: Any, rec_lang: str) -> list[dict[str, Any]]:
    """Convert a RapidOCR result object into line candidates."""
    if result is None:
        return []

    boxes_raw = getattr(result, "boxes", None)
    txts_raw = getattr(result, "txts", None)
    scores_raw = getattr(result, "scores", None)
    boxes = list(boxes_raw) if boxes_raw is not None else []
    txts = list(txts_raw) if txts_raw is not None else []
    scores = list(scores_raw) if scores_raw is not None else []

    candidates: list[dict[str, Any]] = []
    for idx, text in enumerate(txts):
        text = str(text).strip()
        if not text:
            continue
        confidence = float(scores[idx]) if idx < len(scores) else 0.0
        bbox = _normalize_box(boxes[idx]) if idx < len(boxes) else []
        candidates.append(
            {
                "text": text,
                "confidence": round(confidence, 4),
                "bbox": bbox,
                "rec_lang": rec_lang,
                "script": _script_family(text),
            }
        )
    return candidates


def _merge_multilingual_candidates(
    candidates: list[dict[str, Any]],
    *,
    score_threshold: float,
) -> list[dict[str, Any]]:
    """Choose the best candidate per text region, then order/merge rows."""
    if not candidates:
        return []

    clusters: list[list[dict[str, Any]]] = []
    for candidate in candidates:
        matched_cluster: list[dict[str, Any]] | None = None
        for cluster in clusters:
            if _same_text_region(candidate["bbox"], cluster[0]["bbox"]):
                matched_cluster = cluster
                break
        if matched_cluster is None:
            clusters.append([candidate])
        else:
            matched_cluster.append(candidate)

    selected: list[dict[str, Any]] = []
    for cluster in clusters:
        best = max(cluster, key=_candidate_score)
        if best["confidence"] < score_threshold:
            continue
        selected.append(
            {
                "text": best["text"],
                "confidence": best["confidence"],
                "bbox": best["bbox"],
            }
        )

    return _merge_rows(selected)


def _candidate_score(candidate: dict[str, Any]) -> float:
    """Rank competing recognizer outputs for the same line box."""
    text = candidate["text"]
    score = float(candidate["confidence"])
    script = candidate.get("script") or _script_family(text)
    rec_lang = candidate.get("rec_lang", "")

    if _looks_like_noise(text):
        score -= 0.75
    if script in {"hangul", "han", "japanese", "latin"}:
        score += 0.05
    if rec_lang == "korean" and script == "hangul":
        score += 0.20
    elif rec_lang == "japan" and script in {"japanese", "han"}:
        score += 0.20
    elif rec_lang == "ch" and script in {"han", "latin", "mixed"}:
        score += 0.10
    elif rec_lang in {"en", "latin"} and script == "latin":
        score += 0.10

    return score


def _merge_rows(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group line fragments into simple top-to-bottom rows."""
    if not lines:
        return []

    ordered = sorted(lines, key=lambda line: (_center_y(line["bbox"]), _x_min(line["bbox"])))
    rows: list[list[dict[str, Any]]] = []

    for line in ordered:
        target_row: list[dict[str, Any]] | None = None
        for row in rows:
            if _same_row(line, row):
                target_row = row
                break
        if target_row is None:
            rows.append([line])
        else:
            target_row.append(line)

    merged_rows: list[dict[str, Any]] = []
    for row in rows:
        row_items = sorted(row, key=lambda line: (_x_min(line["bbox"]), _center_x(line["bbox"])))
        row_text = " ".join(item["text"] for item in row_items if item["text"]).strip()
        if not row_text:
            continue
        merged_rows.append(
            {
                "text": row_text,
                "confidence": round(sum(item["confidence"] for item in row_items) / len(row_items), 4),
                "bbox": _union_bbox([item["bbox"] for item in row_items]),
            }
        )

    return merged_rows


def _same_text_region(box_a: list[list[float]], box_b: list[list[float]]) -> bool:
    """Heuristic for matching the same detected text line across recognizers."""
    rect_a = _bbox_rect(box_a)
    rect_b = _bbox_rect(box_b)
    if rect_a is None or rect_b is None:
        return False

    if _bbox_iou(rect_a, rect_b) >= 0.30:
        return True

    center_y_delta = abs(_center_y(box_a) - _center_y(box_b))
    center_x_delta = abs(_center_x(box_a) - _center_x(box_b))
    return (
        _vertical_overlap_ratio(rect_a, rect_b) >= 0.60
        and center_y_delta <= max(_height(rect_a), _height(rect_b)) * 0.60
        and center_x_delta <= max(_width(rect_a), _width(rect_b)) * 0.60
    )


def _same_row(line: dict[str, Any], row: list[dict[str, Any]]) -> bool:
    """Return True when a line belongs in an existing row cluster."""
    row_box = _union_bbox([item["bbox"] for item in row])
    rect_line = _bbox_rect(line["bbox"])
    rect_row = _bbox_rect(row_box)
    if rect_line is None or rect_row is None:
        return False

    return (
        _vertical_overlap_ratio(rect_line, rect_row) >= 0.45
        or abs(_center_y(line["bbox"]) - _center_y(row_box)) <= max(_height(rect_line), _height(rect_row)) * 0.60
    )


def _union_bbox(boxes: list[list[list[float]]]) -> list[list[float]]:
    """Return a rectangular union of one or more quadrilateral boxes."""
    rects = [_bbox_rect(box) for box in boxes]
    rects = [rect for rect in rects if rect is not None]
    if not rects:
        return []

    min_x = min(rect[0] for rect in rects)
    min_y = min(rect[1] for rect in rects)
    max_x = max(rect[2] for rect in rects)
    max_y = max(rect[3] for rect in rects)
    return [
        [round(min_x, 1), round(min_y, 1)],
        [round(max_x, 1), round(min_y, 1)],
        [round(max_x, 1), round(max_y, 1)],
        [round(min_x, 1), round(max_y, 1)],
    ]


def _bbox_rect(box: list[list[float]]) -> tuple[float, float, float, float] | None:
    """Convert a quadrilateral into min/max rectangle bounds."""
    if not box:
        return None
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
    except (IndexError, TypeError, ValueError):
        return None
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(
    rect_a: tuple[float, float, float, float],
    rect_b: tuple[float, float, float, float],
) -> float:
    """Return rectangle intersection-over-union."""
    left = max(rect_a[0], rect_b[0])
    top = max(rect_a[1], rect_b[1])
    right = min(rect_a[2], rect_b[2])
    bottom = min(rect_a[3], rect_b[3])
    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    area_a = _width(rect_a) * _height(rect_a)
    area_b = _width(rect_b) * _height(rect_b)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _vertical_overlap_ratio(
    rect_a: tuple[float, float, float, float],
    rect_b: tuple[float, float, float, float],
) -> float:
    """Return vertical overlap normalized by the shorter height."""
    top = max(rect_a[1], rect_b[1])
    bottom = min(rect_a[3], rect_b[3])
    overlap = max(0.0, bottom - top)
    denom = max(1.0, min(_height(rect_a), _height(rect_b)))
    return overlap / denom


def _width(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[2] - rect[0])


def _height(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[3] - rect[1])


def _x_min(box: list[list[float]]) -> float:
    rect = _bbox_rect(box)
    return rect[0] if rect is not None else 0.0


def _center_x(box: list[list[float]]) -> float:
    rect = _bbox_rect(box)
    if rect is None:
        return 0.0
    return (rect[0] + rect[2]) / 2.0


def _center_y(box: list[list[float]]) -> float:
    rect = _bbox_rect(box)
    if rect is None:
        return 0.0
    return (rect[1] + rect[3]) / 2.0


def _looks_like_noise(text: str) -> bool:
    """Return True for common OCR garbage fragments."""
    stripped = "".join(ch for ch in text if not ch.isspace())
    if not stripped:
        return True

    meaningful = 0
    for ch in stripped:
        if ch.isdigit():
            meaningful += 1
            continue
        if _char_script_family(ch) in {"latin", "han", "hangul", "japanese"}:
            meaningful += 1

    if meaningful == 0:
        return True

    script = _script_family(text)
    return len(stripped) <= 2 and meaningful <= 1 and script in {"latin", "other", "unknown"}


def _script_family(text: str) -> str:
    """Return a coarse script family for a text fragment."""
    families = {
        family
        for family in (_char_script_family(ch) for ch in text)
        if family != "neutral"
    }
    if not families:
        return "unknown"
    if "hangul" in families and len(families) == 1:
        return "hangul"
    if "japanese" in families:
        return "japanese"
    if "han" in families and len(families) == 1:
        return "han"
    if "latin" in families and len(families) == 1:
        return "latin"
    if len(families) > 1:
        return "mixed"
    return next(iter(families))


def _char_script_family(ch: str) -> str:
    """Return a coarse script family for a single character."""
    code = ord(ch)
    if ch.isspace() or ch.isdigit():
        return "neutral"
    if 0x0041 <= code <= 0x024F:
        return "latin"
    if 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F or 0xAC00 <= code <= 0xD7AF:
        return "hangul"
    if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
        return "japanese"
    if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
        return "han"
    if ch.isalnum():
        return "other"
    return "neutral"


def _session_providers(component: Any) -> list[str]:
    """Extract ONNX Runtime providers from a RapidOCR component session."""
    infer_session = getattr(component, "session", None)
    ort_session = getattr(infer_session, "session", None)
    if ort_session is None or not hasattr(ort_session, "get_providers"):
        return []
    try:
        return list(ort_session.get_providers())
    except Exception:
        return []


def _parse_enum(enum_cls, raw_value: str, field_name: str):
    """Convert a config string into the enum type RapidOCR expects."""
    try:
        return enum_cls(raw_value)
    except ValueError as e:
        allowed = ", ".join(item.value for item in enum_cls)
        raise ValueError(f"Invalid {field_name} '{raw_value}'. Allowed values: {allowed}") from e
