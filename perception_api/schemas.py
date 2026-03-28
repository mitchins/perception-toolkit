"""
Pydantic schemas for the perception sidecar API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request schemas ───────────────────────────────────────────────────

class InspectRequest(BaseModel):
    session_id: str
    turn_id: str
    logical_name: str
    intent: str = Field(default="general", pattern="^(general|ocr|regions)$")
    query: str = ""


class TagRequest(BaseModel):
    session_id: str
    turn_id: str
    logical_name: str
    threshold: float | None = None  # None → use backend default


class DetectRequest(BaseModel):
    session_id: str
    turn_id: str
    logical_name: str
    threshold: float | None = None
    iou_threshold: float | None = None
    max_detections: int | None = None


class OCRRequest(BaseModel):
    session_id: str
    turn_id: str
    logical_name: str
    threshold: float | None = None


class StageRequest(BaseModel):
    session_id: str
    turn_id: str
    logical_name: str
    mime_type: str | None = None
    # image data is sent as file upload, not in JSON body


class ListRequest(BaseModel):
    session_id: str
    turn_id: str


class ResolveScopeRequest(BaseModel):
    session_id: str
    logical_name: str | None = None


# ── Response schemas ──────────────────────────────────────────────────

class CapabilityAction(BaseModel):
    name: str
    enabled: bool
    description: str
    recommended_for: list[str] = []
    notes: str = ""


class AttachmentInfo(BaseModel):
    logical_name: str
    mime_type: str
    width: int | None = None
    height: int | None = None
    size_bytes: int = 0
    auto_media_type: str | None = None
    auto_media_confidence: float | None = None
    auto_media_total_ms: float | None = None
    auto_media_device: str | None = None
    auto_media_low_confidence: bool = False
    decode_warning: str | None = None


class ListResponse(BaseModel):
    attachments: list[AttachmentInfo]
    display_text: str  # pre-formatted text for LLM consumption


class ResolveScopeResponse(BaseModel):
    session_id: str
    turn_id: str
    logical_name: str | None = None


class InspectResponse(BaseModel):
    logical_name: str
    intent: str
    result_text: str  # concise textual output for the LLM
    backend_used: str = ""


class TagEntry(BaseModel):
    tag: str
    confidence: float


class DetectionEntry(BaseModel):
    label: str
    confidence: float
    bbox: list[float]


class DetectionCountEntry(BaseModel):
    label: str
    count: int
    max_confidence: float


class OCRLineEntry(BaseModel):
    text: str
    confidence: float
    bbox: list[list[float]]


class TagResponse(BaseModel):
    logical_name: str
    tags: list[TagEntry]
    display_text: str


class OCRResponse(BaseModel):
    logical_name: str
    full_text: str
    lines: list[OCRLineEntry]
    display_text: str
    backend_used: str = ""


class DetectionResponse(BaseModel):
    logical_name: str
    object_counts: list[DetectionCountEntry]
    detections: list[DetectionEntry]
    display_text: str
    backend_used: str = ""


class CapabilitiesResponse(BaseModel):
    api_schema_version: str = ""
    backend_status: dict[str, bool]
    actions: list[CapabilityAction]
    display_text: str


class HealthResponse(BaseModel):
    status: str = "ok"
    api_schema_version: str = ""
    backends: dict[str, bool] = {}


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
