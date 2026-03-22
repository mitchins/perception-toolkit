"""
Open WebUI Inlet Filter — Perception Sandbox.

Lightweight filter that:
  1. Detects image attachments on incoming messages.
  2. Stages them into the perception sidecar's scoped sandbox via HTTP.
  3. Strips raw image payloads from the model context.
  4. Injects a system hint so the model knows tools are available.

No heavy inference runs here. The sidecar owns all perception logic.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import time
import uuid
from typing import Any

import httpx

log = logging.getLogger("perception_filter")

SIDECAR_URL = os.environ.get("PERCEPTION_SIDECAR_URL", "http://localhost:8200")

SYSTEM_HINT = """[PERCEPTION SANDBOX ACTIVE]
Attachments are available in a scoped sandbox for this turn.
Use list_attachments() to inspect available files.
Use inspect_image(name, intent, query) to analyse image content.
  - intent="general" for a detailed description
  - intent="ocr" for text extraction
  - intent="regions" with an optional query for region-level analysis
Use tag_image(name, threshold) to get ranked tags (if tagger backend is enabled).
Do NOT attempt to view raw image data directly. Use the tools above instead."""


class Filter:
    """
    Open WebUI inlet filter for the Perception Sandbox.

    Intercepts image attachments, stages them to the sidecar sandbox,
    and injects a system hint for the model.
    """

    class Valves:
        """User-configurable settings exposed in Open WebUI."""
        sidecar_url: str = SIDECAR_URL
        enabled: bool = True
        max_file_size_mb: int = 50

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(self, body: dict[str, Any], __user__: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Process incoming request before it reaches the model.

        Detects image attachments, stages them in the sidecar sandbox,
        strips raw image data, and injects a system hint.
        """
        if not self.valves.enabled:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        # Generate scope identifiers
        user_id = ""
        if __user__:
            user_id = __user__.get("id", "")
        session_id = _session_id(user_id, body)
        turn_id = _turn_id()

        # Scan messages for image content (typically in the last user message)
        staged_any = False
        image_counter = 0

        for msg in messages:
            if msg.get("role") != "user":
                continue

            content = msg.get("content")
            if not isinstance(content, list):
                continue

            new_content_parts: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_url = part.get("image_url", {})
                    url = image_url.get("url", "") if isinstance(image_url, dict) else ""

                    if url.startswith("data:"):
                        image_counter += 1
                        logical_name = f"image_{image_counter}.jpg"
                        staged = await _stage_data_url(
                            self.valves.sidecar_url,
                            session_id,
                            turn_id,
                            logical_name,
                            url,
                            self.valves.max_file_size_mb,
                        )
                        if staged:
                            staged_any = True
                            # Replace image with a text note
                            new_content_parts.append({
                                "type": "text",
                                "text": f"[Attachment staged: {logical_name}]",
                            })
                            continue

                # Keep non-image parts as-is
                new_content_parts.append(part)

            msg["content"] = new_content_parts

        # Also check for Open WebUI-style file attachments in metadata
        metadata = body.get("metadata", {})
        files = metadata.get("files", [])
        if files:
            for file_info in files:
                file_data = file_info.get("data", {})
                file_url = file_data.get("url", "") or file_info.get("url", "")
                file_name = file_data.get("name", "") or file_info.get("name", "")
                file_type = file_data.get("type", "") or file_info.get("type", "")

                if not _is_image_type(file_type) and not _is_image_name(file_name):
                    continue

                image_counter += 1
                logical_name = file_name or f"image_{image_counter}.jpg"

                if file_url.startswith("data:"):
                    staged = await _stage_data_url(
                        self.valves.sidecar_url, session_id, turn_id,
                        logical_name, file_url, self.valves.max_file_size_mb,
                    )
                    if staged:
                        staged_any = True

        if not staged_any:
            return body

        # Store scope IDs in metadata so tools can find the sandbox
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["perception_session_id"] = session_id
        body["metadata"]["perception_turn_id"] = turn_id

        # Inject system hint
        _inject_system_hint(messages)

        return body


# ── Helpers ───────────────────────────────────────────────────────────

def _session_id(user_id: str, body: dict) -> str:
    """Derive a stable session ID."""
    chat_id = body.get("metadata", {}).get("chat_id", "")
    if chat_id:
        return _safe_id(chat_id)
    raw = f"{user_id}-{id(body)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _turn_id() -> str:
    """Generate a unique turn ID."""
    return uuid.uuid4().hex[:12]


def _safe_id(value: str) -> str:
    """Ensure ID is filesystem-safe."""
    safe = "".join(c for c in value if c.isalnum() or c in "-_")
    return safe[:128] or hashlib.sha256(value.encode()).hexdigest()[:16]


def _is_image_type(mime: str) -> bool:
    return mime.startswith("image/") if mime else False


def _is_image_name(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"))


async def _stage_data_url(
    sidecar_url: str,
    session_id: str,
    turn_id: str,
    logical_name: str,
    data_url: str,
    max_mb: int,
) -> bool:
    """Decode a data: URL and POST it to the sidecar staging endpoint."""
    try:
        # Parse data URL: data:<mime>;base64,<data>
        header, encoded = data_url.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
        raw = base64.b64decode(encoded)

        if len(raw) > max_mb * 1024 * 1024:
            log.warning("Attachment %s exceeds %d MB limit, skipping.", logical_name, max_mb)
            return False

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
            if resp.status_code == 200:
                log.info("Staged %s in sidecar sandbox.", logical_name)
                return True
            else:
                log.error("Sidecar staging failed (%d): %s", resp.status_code, resp.text)
                return False

    except Exception as e:
        log.error("Failed to stage attachment %s: %s", logical_name, e)
        return False


def _inject_system_hint(messages: list[dict]) -> None:
    """Add perception system hint to the message list."""
    # Check if hint already present
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and "PERCEPTION SANDBOX ACTIVE" in content:
                return

    # Prepend system hint
    messages.insert(0, {"role": "system", "content": SYSTEM_HINT})
