"""
Perception Sidecar — FastAPI application.

All heavy inference lives here. Open WebUI calls this service over HTTP.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from perception_api import florence, tagger
from perception_api.attachments import (
    cleanup_expired_scopes,
    get_or_create_scope,
    get_scope,
    remove_scope,
)
from perception_api.config import get_config, reload_config
from perception_api.schemas import (
    AttachmentInfo,
    ErrorResponse,
    HealthResponse,
    InspectRequest,
    InspectResponse,
    ListRequest,
    ListResponse,
    TagEntry,
    TagRequest,
    TagResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("perception_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    cfg = get_config()
    log.info("Perception sidecar starting on %s:%d", cfg.host, cfg.port)
    log.info(
        "Backends — Florence: %s | WD-14: %s | Classifier: %s",
        cfg.florence.enabled,
        cfg.wd14.enabled,
        cfg.classifier.enabled,
    )
    # Pre-load Florence if enabled (avoids cold start on first request)
    if cfg.florence.enabled:
        try:
            florence.ensure_loaded()
        except Exception as e:
            log.warning("Florence pre-load failed (will retry on first request): %s", e)
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
        backends={
            "florence": cfg.florence.enabled,
            "wd14": cfg.wd14.enabled,
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
        "classifier": {
            "enabled": cfg.classifier.enabled,
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
        "classifier": {
            "enabled": cfg.classifier.enabled,
            "loaded": False,  # stub
        },
    }


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

    return {"status": "staged", "attachment": meta.to_dict()}


@app.post("/attachments/list", response_model=ListResponse)
async def list_attachments(req: ListRequest):
    """List attachments in a scoped sandbox."""
    scope = get_scope(req.session_id, req.turn_id)
    if scope is None:
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

    return ListResponse(attachments=att_infos, display_text=display)


# ── Inspect (Florence-2) ─────────────────────────────────────────────

@app.post("/inspect", response_model=InspectResponse)
async def inspect_image(req: InspectRequest):
    """Run perception on an image using Florence-2."""
    if not florence.is_available():
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
        result_text = florence.run_inference(path, req.intent, req.query)
    except Exception as e:
        log.error("Florence inference error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return InspectResponse(
        logical_name=req.logical_name,
        intent=req.intent,
        result_text=result_text,
        backend_used="florence",
    )


# ── Tag (WD-14) ──────────────────────────────────────────────────────

@app.post("/tag", response_model=TagResponse)
async def tag_image(req: TagRequest):
    """Run WD-14 tagging on an image."""
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
        log.error("Tagger inference error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tagger error: {e}")

    tag_entries = [TagEntry(tag=t, confidence=c) for t, c in tags]
    display = tagger.format_tags_for_llm(tags)

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
