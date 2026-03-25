"""
title: Perception Tools
author: Mitchell Currie
author_url: https://github.com/mitchins/perception-toolkit
funding_url: https://github.com/mitchins/perception-toolkit
version: 0.1
"""

"""
Open WebUI Tools — Perception Sandbox.

Model-callable tools that proxy to the perception sidecar.
These are lightweight HTTP clients — no inference runs here.
"""

import base64
import io
import logging
import os
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, Field

log = logging.getLogger("perception_tools")
log.setLevel(logging.INFO)

SIDECAR_URL = os.environ.get("PERCEPTION_SIDECAR_URL", "http://localhost:8200")
WEBUI_URL = os.environ.get("OPENWEBUI_BASE_URL", "http://localhost:8080")
EXPECTED_SIDECAR_SCHEMA = "perception-sidecar-2026-03-25"


class Tools:
    """
    Perception Sandbox tools exposed to the reasoning model in Open WebUI.

    The model can call these tools to inspect image attachments
    that were staged by the inlet filter.
    """

    class Valves(BaseModel):
        """User-configurable settings exposed in Open WebUI."""
        sidecar_url: str = Field(
            default=SIDECAR_URL,
            description="Perception sidecar base URL.",
        )
        webui_url: str = Field(
            default=WEBUI_URL,
            description="Open WebUI base URL for resolving uploaded file links.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def get_perception_capabilities(self) -> str:
        """
        Describe which perception actions are currently available.

        Use this to discover which image-analysis actions are enabled before
        choosing a tool. Especially useful when the backend set may change over
        time as the sidecar becomes more extensible.
        """
        log.info("Perception tools get_perception_capabilities called.")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.valves.sidecar_url}/capabilities")
                if resp.status_code == 200:
                    data = resp.json()
                    display_text = data.get("display_text", "No capability summary returned.")
                    schema_version = data.get("api_schema_version", "")
                    if schema_version:
                        if schema_version != EXPECTED_SIDECAR_SCHEMA:
                            return (
                                f"{display_text}\n"
                                f"Compatibility warning: tool expects schema {EXPECTED_SIDECAR_SCHEMA}, "
                                f"but sidecar reports {schema_version}."
                            )
                        return f"{display_text}\nSchema version: {schema_version}"
                    return (
                        f"{display_text}\n"
                        "Compatibility note: sidecar did not report an API schema version. "
                        "It may be an older revision."
                    )
                return f"Error fetching perception capabilities: {resp.status_code}"
        except httpx.RequestError as e:
            log.error("Sidecar request failed: %s", e)
            return "Perception sidecar is not reachable. Cannot fetch capabilities."

    async def list_attachments(
        self,
        scope: str = "turn",
        __metadata__: dict[str, Any] | None = None,
        __messages__: list[dict[str, Any]] | None = None,
        __chat_id__: str | None = None,
        __message_id__: str | None = None,
    ) -> str:
        """
        List attachments available in the current perception sandbox.

        Returns a concise manifest of staged files for this turn.

        :param scope: Scope of attachments to list. Currently only "turn" is supported.
        :return: Text listing of available attachments.
        """
        session_id, turn_id = _extract_scope(__metadata__)
        used_fallback_scope = False
        if not session_id or not turn_id:
            used_fallback_scope = True
            session_id, turn_id = _derive_scope(__metadata__, __chat_id__, __message_id__)
            await _ensure_inline_images_staged(
                self.valves.sidecar_url,
                self.valves.webui_url,
                session_id,
                turn_id,
                __messages__,
            )
        if not session_id or not turn_id:
            return "No perception sandbox is active for this turn. No attachments are available."
        log.info(
            "Perception tools list_attachments called: session=%s turn=%s scope_source=%s",
            session_id,
            turn_id,
            "fallback" if used_fallback_scope else "metadata",
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.valves.sidecar_url}/attachments/list",
                    json={"session_id": session_id, "turn_id": turn_id},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("display_text", "No attachments found.")
                else:
                    return f"Error listing attachments: {resp.status_code}"
        except httpx.RequestError as e:
            log.error("Sidecar request failed: %s", e)
            return "Perception sidecar is not reachable. Cannot list attachments."

    async def inspect_image(
        self,
        name: str,
        intent: str = "general",
        query: str = "",
        __metadata__: dict[str, Any] | None = None,
        __messages__: list[dict[str, Any]] | None = None,
        __chat_id__: str | None = None,
        __message_id__: str | None = None,
    ) -> str:
        """
        Inspect an image attachment using the perception backend.

        Analyses the named image and returns a textual result.
        Prefer extract_text(...) when the user mainly wants visible text read
        from a screenshot, document, or UI image. Use this tool's general
        intent for non-text visual description.

        :param name: Logical filename of the attachment (e.g. "image_1.jpg").
        :param intent: Analysis type — "general" for description, "ocr" for text extraction, "regions" for region analysis.
        :param query: Optional query for region-based analysis (used with intent="regions").
        :return: Textual analysis result from the perception backend.
        """
        session_id, turn_id = _extract_scope(__metadata__)
        used_fallback_scope = False
        if not session_id or not turn_id:
            used_fallback_scope = True
            session_id, turn_id = _derive_scope(__metadata__, __chat_id__, __message_id__)
            await _ensure_inline_images_staged(
                self.valves.sidecar_url,
                self.valves.webui_url,
                session_id,
                turn_id,
                __messages__,
            )
        if not session_id or not turn_id:
            return "No perception sandbox is active. Cannot inspect image."

        if intent not in ("general", "ocr", "regions"):
            return f"Invalid intent '{intent}'. Use 'general', 'ocr', or 'regions'."
        log.info(
            "Perception tools inspect_image called: name=%s intent=%s session=%s turn=%s scope_source=%s",
            name,
            intent,
            session_id,
            turn_id,
            "fallback" if used_fallback_scope else "metadata",
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.valves.sidecar_url}/inspect",
                    json={
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "logical_name": name,
                        "intent": intent,
                        "query": query,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("result_text", "No result returned.")
                elif resp.status_code == 404:
                    return f"Attachment '{name}' not found in the sandbox. Use list_attachments() to see available files."
                elif resp.status_code == 503:
                    return "The perception backend is not available. The Florence model may not be enabled."
                else:
                    detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    return f"Inspection failed ({resp.status_code}): {detail}"
        except httpx.RequestError as e:
            log.error("Sidecar request failed: %s", e)
            return "Perception sidecar is not reachable. Cannot inspect image."

    async def detect_objects(
        self,
        name: str,
        threshold: float | None = None,
        iou_threshold: float | None = None,
        max_detections: int | None = None,
        __metadata__: dict[str, Any] | None = None,
        __messages__: list[dict[str, Any]] | None = None,
        __chat_id__: str | None = None,
        __message_id__: str | None = None,
    ) -> str:
        """
        Detect common objects in an image attachment.

        Returns grouped counts plus raw detector hits for the named image.

        :param name: Logical filename of the attachment (e.g. "image_1.jpg").
        :param threshold: Optional detector confidence threshold override.
        :param iou_threshold: Optional detector IoU threshold override.
        :param max_detections: Optional cap on returned detections.
        :return: Textual detector result from the perception backend.
        """
        session_id, turn_id = _extract_scope(__metadata__)
        used_fallback_scope = False
        if not session_id or not turn_id:
            used_fallback_scope = True
            session_id, turn_id = _derive_scope(__metadata__, __chat_id__, __message_id__)
            await _ensure_inline_images_staged(
                self.valves.sidecar_url,
                self.valves.webui_url,
                session_id,
                turn_id,
                __messages__,
            )
        if not session_id or not turn_id:
            return "No perception sandbox is active. Cannot detect objects."

        log.info(
            "Perception tools detect_objects called: name=%s session=%s turn=%s scope_source=%s",
            name,
            session_id,
            turn_id,
            "fallback" if used_fallback_scope else "metadata",
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.valves.sidecar_url}/detect",
                    json={
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "logical_name": name,
                        "threshold": threshold,
                        "iou_threshold": iou_threshold,
                        "max_detections": max_detections,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("display_text", "No detection result returned.")
                elif resp.status_code == 404:
                    return f"Attachment '{name}' not found in the sandbox. Use list_attachments() to see available files."
                elif resp.status_code == 503:
                    return "The detector backend is not available. Enable it in the sidecar config to use detect_objects."
                else:
                    detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    return f"Detection failed ({resp.status_code}): {detail}"
        except httpx.RequestError as e:
            log.error("Sidecar request failed: %s", e)
            return "Perception sidecar is not reachable. Cannot detect objects."

    async def extract_text(
        self,
        name: str,
        threshold: float | None = None,
        __metadata__: dict[str, Any] | None = None,
        __messages__: list[dict[str, Any]] | None = None,
        __chat_id__: str | None = None,
        __message_id__: str | None = None,
    ) -> str:
        """
        Extract visible text from an image attachment using OCR.

        Returns OCR text lines with confidences for the named image.
        Prefer this tool when the user asks to read, transcribe, quote, or
        explain text that appears in an image.

        :param name: Logical filename of the attachment (e.g. "image_1.jpg").
        :param threshold: Optional OCR confidence threshold override.
        :return: Textual OCR result from the perception backend.
        """
        session_id, turn_id = _extract_scope(__metadata__)
        used_fallback_scope = False
        if not session_id or not turn_id:
            used_fallback_scope = True
            session_id, turn_id = _derive_scope(__metadata__, __chat_id__, __message_id__)
            await _ensure_inline_images_staged(
                self.valves.sidecar_url,
                self.valves.webui_url,
                session_id,
                turn_id,
                __messages__,
            )
        if not session_id or not turn_id:
            return "No perception sandbox is active. Cannot extract text."

        log.info(
            "Perception tools extract_text called: name=%s session=%s turn=%s scope_source=%s",
            name,
            session_id,
            turn_id,
            "fallback" if used_fallback_scope else "metadata",
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.valves.sidecar_url}/ocr",
                    json={
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "logical_name": name,
                        "threshold": threshold,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("display_text", "No OCR result returned.")
                elif resp.status_code == 404:
                    # Backward compatibility: older sidecars routed OCR through
                    # /inspect with intent="ocr" and did not expose /ocr yet.
                    legacy_resp = await client.post(
                        f"{self.valves.sidecar_url}/inspect",
                        json={
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "logical_name": name,
                            "intent": "ocr",
                            "query": "",
                        },
                    )
                    if legacy_resp.status_code == 200:
                        legacy_data = legacy_resp.json()
                        legacy_result = legacy_data.get("result_text", "No OCR result returned.")
                        return (
                            "Compatibility warning: connected sidecar does not expose the /ocr route "
                            f"expected by tool schema {EXPECTED_SIDECAR_SCHEMA}; falling back to legacy "
                            "OCR via /inspect. Update the sidecar and re-import the tool/filter files.\n\n"
                            f"{legacy_result}"
                        )
                    if legacy_resp.status_code == 404:
                        return f"Attachment '{name}' not found in the sandbox. Use list_attachments() to see available files."
                    if legacy_resp.status_code == 503:
                        return "The OCR backend is not available. Enable it in the sidecar config to use extract_text."
                    legacy_detail = (
                        legacy_resp.json().get("detail", legacy_resp.text)
                        if legacy_resp.headers.get("content-type", "").startswith("application/json")
                        else legacy_resp.text
                    )
                    return f"OCR failed ({legacy_resp.status_code}): {legacy_detail}"
                elif resp.status_code == 503:
                    return "The OCR backend is not available. Enable it in the sidecar config to use extract_text."
                else:
                    detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    return f"OCR failed ({resp.status_code}): {detail}"
        except httpx.RequestError as e:
            log.error("Sidecar request failed: %s", e)
            return "Perception sidecar is not reachable. Cannot extract text."

# ── Helpers ───────────────────────────────────────────────────────────

def _extract_scope(metadata: dict[str, Any] | None) -> tuple[str, str]:
    """Extract perception sandbox scope IDs from Open WebUI metadata."""
    if not metadata:
        return "", ""
    session_id = metadata.get("perception_session_id", "")
    turn_id = metadata.get("perception_turn_id", "")
    return session_id, turn_id


def _derive_scope(
    metadata: dict[str, Any] | None,
    chat_id: str | None,
    message_id: str | None,
) -> tuple[str, str]:
    """Derive a stable fallback scope from standard Open WebUI metadata."""
    metadata = metadata or {}
    session_id = (
        chat_id
        or metadata.get("chat_id", "")
        or metadata.get("session_id", "")
    )
    turn_id = (
        message_id
        or metadata.get("message_id", "")
        or metadata.get("session_id", "")
    )
    return session_id, turn_id


async def _ensure_inline_images_staged(
    sidecar_url: str,
    webui_url: str,
    session_id: str,
    turn_id: str,
    messages: list[dict[str, Any]] | None,
) -> None:
    """Stage current-turn image_url payloads when filter metadata is missing."""
    if not session_id or not turn_id or not messages:
        return

    inline_images = _collect_current_turn_images(messages)
    if not inline_images:
        log.info("Perception tools fallback staging: no current-turn image_url parts found.")
        return
    log.info(
        "Perception tools fallback staging: current_turn_images=%d names=%s",
        len(inline_images),
        ",".join(item["logical_name"] for item in inline_images),
    )

    existing = await _list_scope_attachment_names(sidecar_url, session_id, turn_id)
    next_index = len(existing)

    for item in inline_images:
        logical_name = item["logical_name"]
        if logical_name in existing:
            continue

        next_index += 1
        if not logical_name:
            logical_name = f"image_{next_index}.jpg"
        log.info(
            "Perception tools fallback staging image: name=%s source_kind=%s",
            logical_name,
            _source_kind(item["source_url"]),
        )
        staged = await _stage_data_url(
            sidecar_url,
            webui_url,
            session_id,
            turn_id,
            logical_name,
            item["source_url"],
        )
        if staged:
            existing.add(logical_name)


def _collect_current_turn_images(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Collect image_url data only from the latest user turn."""
    images: list[dict[str, str]] = []
    image_counter = 0
    current_user_msg = _get_last_user_message(messages)
    if current_user_msg is None:
        return images

    content = current_user_msg.get("content")
    if not isinstance(content, list):
        return images

    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue

        image_url = part.get("image_url", {})
        url = image_url.get("url", "") if isinstance(image_url, dict) else ""
        if not url:
            continue

        image_counter += 1
        images.append(
            {
                "logical_name": _guess_logical_name(url, image_counter),
                "source_url": url,
            }
        )

    return images


def _get_last_user_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the latest user message in the chat."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    return None


async def _list_scope_attachment_names(
    sidecar_url: str,
    session_id: str,
    turn_id: str,
) -> set[str]:
    """Return logical names already staged in the current scope."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{sidecar_url}/attachments/list",
                json={"session_id": session_id, "turn_id": turn_id},
            )
            if resp.status_code != 200:
                return set()
            data = resp.json()
            return {
                item.get("logical_name", "")
                for item in data.get("attachments", [])
                if item.get("logical_name")
            }
    except httpx.RequestError:
        return set()


async def _stage_data_url(
    sidecar_url: str,
    webui_url: str,
    session_id: str,
    turn_id: str,
    logical_name: str,
    source_url: str,
) -> bool:
    """Stage either a data URL or a URL-backed image in the sidecar sandbox."""
    try:
        if source_url.startswith("data:"):
            header, encoded = source_url.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
            raw = base64.b64decode(encoded)
        else:
            fetch_url = urljoin(webui_url, source_url) if source_url.startswith("/") else source_url
            async with httpx.AsyncClient(timeout=30.0) as client:
                download = await client.get(fetch_url)
                download.raise_for_status()
                mime_type = download.headers.get("content-type", "application/octet-stream").split(";")[0]
                raw = download.content
        log.info(
            "Perception tools staging attachment: name=%s source_kind=%s mime=%s size_bytes=%d",
            logical_name,
            _source_kind(source_url),
            mime_type,
            len(raw),
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{sidecar_url}/attachments/stage",
                data={
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "logical_name": logical_name,
                    "mime_type": mime_type,
                },
                files={"file": (logical_name, io.BytesIO(raw), mime_type)},
            )
            return resp.status_code == 200
    except Exception as e:
        log.error("Failed to stage inline image %s: %s", logical_name, e)
        return False


def _guess_logical_name(url: str, index: int) -> str:
    """Guess a stable logical name from a URL or fall back to a numbered image."""
    if url and not url.startswith("data:"):
        path = urlparse(url).path
        candidate = path.rsplit("/", 1)[-1]
        if candidate:
            return candidate
    return f"image_{index}.jpg"


def _source_kind(url: str) -> str:
    """Return a short label describing the image source."""
    if not url:
        return "missing"
    if url.startswith("data:"):
        return "data_url"
    return "url"
