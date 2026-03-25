"""
Perception Sidecar — FastAPI application.

All heavy inference lives here. Open WebUI calls this service over HTTP.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from perception_api import detector, florence, ocr, tagger
from perception_api.attachments import (
    cleanup_expired_scopes,
    find_latest_scope,
    get_or_create_scope,
    get_scope,
    remove_scope,
)
from perception_api.config import get_config, reload_config
from perception_api.schemas import (
    AttachmentInfo,
    CapabilitiesResponse,
    CapabilityAction,
    DetectionCountEntry,
    DetectionEntry,
    DetectionResponse,
    DetectRequest,
    ErrorResponse,
    HealthResponse,
    OCRLineEntry,
    OCRRequest,
    OCRResponse,
    InspectRequest,
    InspectResponse,
    ListRequest,
    ListResponse,
    ResolveScopeRequest,
    ResolveScopeResponse,
    TagEntry,
    TagRequest,
    TagResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("perception_api")
API_SCHEMA_VERSION = "perception-sidecar-2026-03-25"


def _ms(start: float) -> float:
    """Return elapsed milliseconds since start."""
    return (perf_counter() - start) * 1000.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    cfg = get_config()
    log.info("Perception sidecar starting on %s:%d", cfg.host, cfg.port)
    log.info(
        "Backends — Florence: %s | OCR: %s | WD-14: %s | Detector: %s | Classifier: %s",
        cfg.florence.enabled,
        cfg.ocr.enabled,
        cfg.wd14.enabled,
        cfg.detector.enabled,
        cfg.classifier.enabled,
    )
    # Pre-load Florence if enabled (avoids cold start on first request)
    if cfg.florence.enabled:
        try:
            florence.ensure_loaded()
        except Exception as e:
            log.warning("Florence pre-load failed (will retry on first request): %s", e)
    if cfg.ocr.enabled:
        try:
            ocr.ensure_loaded()
        except Exception as e:
            log.warning("OCR pre-load failed (will retry on first request): %s", e)
    if cfg.detector.enabled:
        try:
            detector.ensure_loaded()
        except Exception as e:
            log.warning("Detector pre-load failed (will retry on first request): %s", e)
    yield
    log.info("Perception sidecar shutting down.")


app = FastAPI(
    title="Perception Sidecar",
    version="0.1.0",
    description="Local perception inference service for the Perception Sandbox PoC.",
    lifespan=lifespan,
)


# ── Health / Status ───────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    cfg = get_config()
    return HealthResponse(
        status="ok",
        api_schema_version=API_SCHEMA_VERSION,
        backends={
            "florence": cfg.florence.enabled,
            "ocr": cfg.ocr.enabled,
            "wd14": cfg.wd14.enabled,
            "detector": cfg.detector.enabled,
            "classifier": cfg.classifier.enabled,
        },
    )


@app.get("/config")
async def config_summary():
    cfg = get_config()
    return {
        "florence": {
            "enabled": cfg.florence.enabled,
            "model_id": cfg.florence.model_id,
            "device": cfg.florence.device,
        },
        "wd14": {
            "enabled": cfg.wd14.enabled,
            "model_id": cfg.wd14.model_id,
            "threshold_default": cfg.wd14.threshold_default,
            "device": cfg.wd14.device,
        },
        "ocr": {
            "enabled": cfg.ocr.enabled,
            "score_threshold_default": cfg.ocr.score_threshold_default,
        },
        "classifier": {
            "enabled": cfg.classifier.enabled,
        },
        "detector": {
            "enabled": cfg.detector.enabled,
            "model_id": cfg.detector.model_id,
            "confidence_threshold_default": cfg.detector.confidence_threshold_default,
            "iou_threshold_default": cfg.detector.iou_threshold_default,
            "max_detections": cfg.detector.max_detections,
            "device": cfg.detector.device,
        },
        "sandbox": {
            "base_path": cfg.sandbox.base_path,
            "ttl_seconds": cfg.sandbox.ttl_seconds,
        },
    }


@app.get("/backends")
async def backends_status():
    cfg = get_config()
    return {
        "florence": {
            "enabled": cfg.florence.enabled,
            "loaded": florence._model is not None,
        },
        "wd14": {
            "enabled": cfg.wd14.enabled,
            "loaded": tagger._loaded,
        },
        "ocr": {
            "enabled": cfg.ocr.enabled,
            "loaded": ocr._loaded,
        },
        "detector": {
            "enabled": cfg.detector.enabled,
            "loaded": detector._loaded,
        },
        "classifier": {
            "enabled": cfg.classifier.enabled,
            "loaded": False,  # stub
        },
    }


@app.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities():
    """Describe which perception actions are currently available."""
    cfg = get_config()
    backend_status = {
        "florence": cfg.florence.enabled,
        "ocr": cfg.ocr.enabled,
        "wd14": cfg.wd14.enabled,
        "detector": cfg.detector.enabled,
        "classifier": cfg.classifier.enabled,
    }
    actions = [
        CapabilityAction(
            name="list_attachments",
            enabled=True,
            description="List the files staged for the current turn.",
            recommended_for=["start of turn", "identify active image", "multi-image disambiguation"],
        ),
        CapabilityAction(
            name="inspect_image.general",
            enabled=cfg.florence.enabled,
            description="Describe what matters in an image.",
            recommended_for=["general description", "what is this", "non-photographic/style cues"],
            notes=(
                f"Backed by Florence model {cfg.florence.model_id}."
                if cfg.florence.enabled
                else "Disabled because the Florence backend is off."
            ),
        ),
        CapabilityAction(
            name="inspect_image.ocr",
            enabled=cfg.ocr.enabled or cfg.florence.enabled,
            description="Extract visible text from an image.",
            recommended_for=["OCR", "read labels", "read signs", "read UI text"],
            notes=(
                f"Uses RapidOCR with default threshold {cfg.ocr.score_threshold_default:.2f}."
                if cfg.ocr.enabled
                else "Falls back to the Florence backend when RapidOCR is disabled."
            ),
        ),
        CapabilityAction(
            name="inspect_image.regions",
            enabled=cfg.florence.enabled,
            description="Answer a region-focused query about an image.",
            recommended_for=["where is X", "region lookup", "localized visual queries"],
            notes="Requires a query string describing what to ground.",
        ),
        CapabilityAction(
            name="extract_text",
            enabled=cfg.ocr.enabled,
            description="Extract visible text with OCR and return line boxes and confidences.",
            recommended_for=["OCR", "read labels", "read signs", "UI text", "receipts", "menus"],
            notes=(
                f"Uses RapidOCR with default threshold {cfg.ocr.score_threshold_default:.2f}."
                if cfg.ocr.enabled
                else "Disabled because the OCR backend is off."
            ),
        ),
        CapabilityAction(
            name="tag_image",
            enabled=cfg.wd14.enabled,
            description="Return label-style tags for an image.",
            recommended_for=["tag-like labels", "attributes", "booru-style tags"],
            notes=(
                f"Uses WD-14 threshold default {cfg.wd14.threshold_default:.2f}."
                if cfg.wd14.enabled
                else "Disabled because the WD-14 tagger backend is off."
            ),
        ),
        CapabilityAction(
            name="detect_objects",
            enabled=cfg.detector.enabled,
            description="Return closed-vocabulary object detections with counts and rough boxes.",
            recommended_for=["count objects", "inventory common objects", "corroborate a caption"],
            notes=(
                f"Uses detector model {cfg.detector.model_id} with default threshold {cfg.detector.confidence_threshold_default:.2f}."
                if cfg.detector.enabled
                else "Disabled because the detector backend is off."
            ),
        ),
    ]
    display_lines = [
        "Perception capabilities available right now:",
        f"- schema_version: {API_SCHEMA_VERSION}",
        *[
            f"- {action.name}: {'enabled' if action.enabled else 'disabled'} — {action.description}"
            for action in actions
        ],
        "Recommended default workflow:",
        "- First call list_attachments() for the current turn.",
        "- Then use inspect_image(..., intent='general') unless the user specifically wants OCR, region grounding, object counts, or tags.",
        "- Use extract_text(...) first when the user wants the actual text read from an image.",
        "- Avoid inspect_image(..., intent='general') for OCR-heavy requests unless layout or other non-text visual context is also needed.",
        "- Use detect_objects(...) when the user asks how many people, cars, pizzas, or other common detector categories are present.",
    ]
    return CapabilitiesResponse(
        api_schema_version=API_SCHEMA_VERSION,
        backend_status=backend_status,
        actions=actions,
        display_text="\n".join(display_lines),
    )


# ── Attachment Staging ────────────────────────────────────────────────

@app.post("/attachments/stage")
async def stage_attachment(
    session_id: str = Form(...),
    turn_id: str = Form(...),
    logical_name: str = Form(...),
    mime_type: str = Form(None),
    file: UploadFile = File(...),
):
    """Stage an uploaded file into the scoped sandbox."""
    started = perf_counter()
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    if len(data) > 50 * 1024 * 1024:  # 50 MB limit
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit.")

    scope = get_or_create_scope(session_id, turn_id)
    try:
        meta = scope.stage_file(logical_name, data, mime_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    log.info(
        "stage_attachment session=%s turn=%s name=%s size_bytes=%d took_ms=%.1f",
        session_id,
        turn_id,
        logical_name,
        len(data),
        _ms(started),
    )

    return {"status": "staged", "attachment": meta.to_dict()}


@app.post("/attachments/list", response_model=ListResponse)
async def list_attachments(req: ListRequest):
    """List attachments in a scoped sandbox."""
    started = perf_counter()
    scope = get_scope(req.session_id, req.turn_id)
    if scope is None:
        log.info(
            "list_attachments session=%s turn=%s count=0 took_ms=%.1f",
            req.session_id,
            req.turn_id,
            _ms(started),
        )
        return ListResponse(attachments=[], display_text="No attachments available for this turn.")

    items = scope.list_attachments()
    att_infos = [
        AttachmentInfo(
            logical_name=m.logical_name,
            mime_type=m.mime_type,
            width=m.width,
            height=m.height,
            size_bytes=m.size_bytes,
        )
        for m in items
    ]
    display_lines = [m.to_display() for m in items]
    display = "Available attachments for this turn:\n" + "\n".join(display_lines) if display_lines else "No attachments available for this turn."

    log.info(
        "list_attachments session=%s turn=%s count=%d took_ms=%.1f",
        req.session_id,
        req.turn_id,
        len(att_infos),
        _ms(started),
    )

    return ListResponse(attachments=att_infos, display_text=display)


@app.post("/attachments/resolve", response_model=ResolveScopeResponse)
async def resolve_attachment_scope(req: ResolveScopeRequest):
    """Resolve the newest matching attachment scope for a session."""
    started = perf_counter()
    scope = find_latest_scope(req.session_id, req.logical_name)
    if scope is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No matching attachment scope found for session '{req.session_id}'."
                if not req.logical_name
                else f"No matching attachment scope found for session '{req.session_id}' and name '{req.logical_name}'."
            ),
        )

    log.info(
        "resolve_scope session=%s logical_name=%s turn=%s took_ms=%.1f",
        req.session_id,
        req.logical_name or "<any>",
        scope.turn_id,
        _ms(started),
    )
    return ResolveScopeResponse(
        session_id=scope.session_id,
        turn_id=scope.turn_id,
        logical_name=req.logical_name,
    )


# ── Inspect (Florence-2) ─────────────────────────────────────────────

@app.post("/inspect", response_model=InspectResponse)
async def inspect_image(req: InspectRequest):
    """Run perception on an image using Florence-2."""
    started = perf_counter()
    if req.intent == "ocr" and not (ocr.is_available() or florence.is_available()):
        raise HTTPException(status_code=503, detail="No OCR-capable backend is enabled.")
    if req.intent != "ocr" and not florence.is_available():
        raise HTTPException(status_code=503, detail="Florence backend is not enabled.")

    scope = get_scope(req.session_id, req.turn_id)
    if scope is None:
        raise HTTPException(status_code=404, detail="No sandbox scope found for this session/turn.")

    path = scope.resolve_path(req.logical_name)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Attachment '{req.logical_name}' not found in sandbox.",
        )

    try:
        if req.intent == "ocr" and ocr.is_available():
            ocr_result = ocr.extract_text(path)
            result_text = ocr.format_text_for_llm(ocr_result["full_text"], ocr_result["lines"])
            backend_used = "rapidocr"
        else:
            result_text = florence.run_inference(path, req.intent, req.query)
            backend_used = "florence"
    except Exception as e:
        log.error(
            "inspect_image failed session=%s turn=%s name=%s intent=%s took_ms=%.1f error=%s",
            req.session_id,
            req.turn_id,
            req.logical_name,
            req.intent,
            _ms(started),
            e,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    log.info(
        "inspect_image session=%s turn=%s name=%s intent=%s backend=%s took_ms=%.1f",
        req.session_id,
        req.turn_id,
        req.logical_name,
        req.intent,
        backend_used,
        _ms(started),
    )

    return InspectResponse(
        logical_name=req.logical_name,
        intent=req.intent,
        result_text=result_text,
        backend_used=backend_used,
    )


# ── OCR (RapidOCR) ───────────────────────────────────────────────────

@app.post("/ocr", response_model=OCRResponse)
async def extract_text(req: OCRRequest):
    """Run OCR on an image and return extracted text lines."""
    started = perf_counter()
    if not ocr.is_available():
        raise HTTPException(
            status_code=503,
            detail="OCR backend is not enabled. Enable it in config.yaml to use extract_text.",
        )

    scope = get_scope(req.session_id, req.turn_id)
    if scope is None:
        raise HTTPException(status_code=404, detail="No sandbox scope found for this session/turn.")

    path = scope.resolve_path(req.logical_name)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Attachment '{req.logical_name}' not found in sandbox.",
        )

    try:
        result = ocr.extract_text(path, req.threshold)
    except Exception as e:
        log.error(
            "extract_text failed session=%s turn=%s name=%s took_ms=%.1f error=%s",
            req.session_id,
            req.turn_id,
            req.logical_name,
            _ms(started),
            e,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"OCR error: {e}")

    display = ocr.format_text_for_llm(result["full_text"], result["lines"])

    log.info(
        "extract_text session=%s turn=%s name=%s lines=%d backend=rapidocr took_ms=%.1f",
        req.session_id,
        req.turn_id,
        req.logical_name,
        len(result["lines"]),
        _ms(started),
    )

    return OCRResponse(
        logical_name=req.logical_name,
        full_text=result["full_text"],
        lines=[OCRLineEntry(**line) for line in result["lines"]],
        display_text=display,
        backend_used="rapidocr",
    )


# ── Detect (YOLO) ────────────────────────────────────────────────────

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(req: DetectRequest):
    """Run detector inference on an image and return grouped object counts."""
    started = perf_counter()
    if not detector.is_available():
        raise HTTPException(
            status_code=503,
            detail="Detector backend is not enabled. Enable it in config.yaml to use detect_objects.",
        )

    scope = get_scope(req.session_id, req.turn_id)
    if scope is None:
        raise HTTPException(status_code=404, detail="No sandbox scope found for this session/turn.")

    path = scope.resolve_path(req.logical_name)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Attachment '{req.logical_name}' not found in sandbox.",
        )

    try:
        detections = detector.detect_image(
            path,
            req.threshold,
            req.iou_threshold,
            req.max_detections,
        )
    except Exception as e:
        log.error(
            "detect_objects failed session=%s turn=%s name=%s took_ms=%.1f error=%s",
            req.session_id,
            req.turn_id,
            req.logical_name,
            _ms(started),
            e,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Detector error: {e}")

    grouped = detector.summarize_detections(detections)
    display = detector.format_detections_for_llm(grouped, detections)

    log.info(
        "detect_objects session=%s turn=%s name=%s grouped=%d detections=%d backend=detector took_ms=%.1f",
        req.session_id,
        req.turn_id,
        req.logical_name,
        len(grouped),
        len(detections),
        _ms(started),
    )

    return DetectionResponse(
        logical_name=req.logical_name,
        object_counts=[DetectionCountEntry(**item) for item in grouped],
        detections=[DetectionEntry(**item) for item in detections],
        display_text=display,
        backend_used="detector",
    )


# ── Tag (WD-14) ──────────────────────────────────────────────────────

@app.post("/tag", response_model=TagResponse)
async def tag_image(req: TagRequest):
    """Run WD-14 tagging on an image."""
    started = perf_counter()
    if not tagger.is_available():
        raise HTTPException(
            status_code=503,
            detail="Tagger backend is not enabled. Enable it in config.yaml to use tag_image.",
        )

    scope = get_scope(req.session_id, req.turn_id)
    if scope is None:
        raise HTTPException(status_code=404, detail="No sandbox scope found for this session/turn.")

    path = scope.resolve_path(req.logical_name)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Attachment '{req.logical_name}' not found in sandbox.",
        )

    try:
        tags = tagger.tag_image(path, req.threshold)
    except Exception as e:
        log.error(
            "tag_image failed session=%s turn=%s name=%s took_ms=%.1f error=%s",
            req.session_id,
            req.turn_id,
            req.logical_name,
            _ms(started),
            e,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Tagger error: {e}")

    tag_entries = [TagEntry(tag=t, confidence=c) for t, c in tags]
    display = tagger.format_tags_for_llm(tags)

    log.info(
        "tag_image session=%s turn=%s name=%s tags=%d took_ms=%.1f",
        req.session_id,
        req.turn_id,
        req.logical_name,
        len(tag_entries),
        _ms(started),
    )

    return TagResponse(
        logical_name=req.logical_name,
        tags=tag_entries,
        display_text=display,
    )


# ── Cleanup ───────────────────────────────────────────────────────────

@app.post("/attachments/cleanup")
async def cleanup_scope(session_id: str = Form(...), turn_id: str = Form(...)):
    """Clean up a specific sandbox scope."""
    remove_scope(session_id, turn_id)
    return {"status": "cleaned"}


@app.post("/attachments/cleanup-expired")
async def cleanup_expired():
    """Clean up all expired sandbox scopes based on TTL."""
    count = cleanup_expired_scopes()
    return {"status": "ok", "removed": count}


# ── Entry point ───────────────────────────────────────────────────────

def main():
    cfg = get_config()
    uvicorn.run(
        "perception_api.main:app",
        host=cfg.host,
        port=cfg.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
