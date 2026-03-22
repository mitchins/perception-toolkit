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


class StageRequest(BaseModel):
    session_id: str
    turn_id: str
    logical_name: str
    mime_type: str | None = None
    # image data is sent as file upload, not in JSON body


class ListRequest(BaseModel):
    session_id: str
    turn_id: str


# ── Response schemas ──────────────────────────────────────────────────

class AttachmentInfo(BaseModel):
    logical_name: str
    mime_type: str
    width: int | None = None
    height: int | None = None
    size_bytes: int = 0


class ListResponse(BaseModel):
    attachments: list[AttachmentInfo]
    display_text: str  # pre-formatted text for LLM consumption


class InspectResponse(BaseModel):
    logical_name: str
    intent: str
    result_text: str  # concise textual output for the LLM
    backend_used: str = ""


class TagEntry(BaseModel):
    tag: str
    confidence: float


class TagResponse(BaseModel):
    logical_name: str
    tags: list[TagEntry]
    display_text: str


class HealthResponse(BaseModel):
    status: str = "ok"
    backends: dict[str, bool] = {}


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
