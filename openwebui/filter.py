"""
title: Perception Filter
author: Mitchell Currie
author_url: https://github.com/mitchins/perception-toolkit
funding_url: https://github.com/mitchins/perception-toolkit
version: 0.1
"""

"""
Open WebUI Inlet Filter — Perception Sandbox.

Lightweight filter that:
  1. Detects image attachments on incoming messages.
  2. Stages them into the perception sidecar's scoped sandbox via HTTP.
  3. Strips raw image payloads from the model context.
  4. Injects a system hint so the model knows tools are available.

No heavy inference runs here. The sidecar owns all perception logic.
"""

import base64
import hashlib
import io
import logging
import os
import uuid
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, Field

log = logging.getLogger("perception_filter")
log.setLevel(logging.INFO)

SIDECAR_URL = os.environ.get("PERCEPTION_SIDECAR_URL", "http://localhost:8200")
WEBUI_URL = os.environ.get("OPENWEBUI_BASE_URL", "http://localhost:8080")

SYSTEM_HINT = """[PERCEPTION SANDBOX ACTIVE]
Attachments are available in a scoped sandbox for this turn.
If the user uploaded an image in this turn, treat that as the active image for phrases like "this image".
Use get_perception_capabilities() first if you are unsure which actions are available.
Use list_attachments() first to inspect the files available for the current turn.
Use inspect_image(name, intent, query) to analyse image content.
  - intent="general" for a detailed description
  - intent="ocr" for text extraction
  - intent="regions" with an optional query for region-level analysis
If the user asks to read, transcribe, quote, or explain text from an image, use extract_text(name, threshold) first.
Do not call inspect_image(name, intent="general") for OCR-heavy requests unless the user also needs non-text visual context like layout, style, or surrounding objects.
Use detect_objects(name, threshold, iou_threshold, max_detections) when the user wants object counts or a concrete inventory of common detected objects.
Use get_perception_capabilities() to see whether any optional specialist actions are enabled.
Do NOT attempt to view raw image data directly. Use the tools above instead."""


class Filter:
    """
    Open WebUI inlet filter for the Perception Sandbox.

    Intercepts image attachments, stages them to the sidecar sandbox,
    and injects a system hint for the model.
    """

    class Valves(BaseModel):
        """User-configurable settings exposed in Open WebUI."""
        priority: int = Field(
            default=0,
            description="Priority for filter execution. Lower values run first.",
        )
        sidecar_url: str = Field(
            default=SIDECAR_URL,
            description="Perception sidecar base URL.",
        )
        webui_url: str = Field(
            default=WEBUI_URL,
            description="Open WebUI base URL for resolving uploaded file links.",
        )
        enabled: bool = Field(
            default=True,
            description="Enable or disable perception attachment staging.",
        )
        max_file_size_mb: int = Field(
            default=50,
            description="Maximum staged attachment size in megabytes.",
        )
        debug_breadcrumbs: bool = Field(
            default=True,
            description="Emit visible debug breadcrumbs when the filter inlet runs.",
        )

    def __init__(self):
        # Tell Open WebUI this filter owns file/image handling so it can
        # disengage default upload processing paths where supported.
        self.file_handler = True
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __metadata__: dict[str, Any] | None = None,
        __event_emitter__=None,
    ) -> dict[str, Any]:
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

        metadata = body.get("metadata", {})
        top_level_files = body.get("files", [])
        metadata_files = metadata.get("files", []) if isinstance(metadata, dict) else []
        interface = ""
        chat_id = ""
        if isinstance(__metadata__, dict):
            interface = __metadata__.get("interface", "") or ""
            chat_id = __metadata__.get("chat_id", "") or ""
        print(
            "[PerceptionFilter] inlet hit "
            f"interface={interface or 'unknown'} "
            f"messages={len(messages)} "
            f"top_files={len(top_level_files) if isinstance(top_level_files, list) else 0} "
            f"metadata_files={len(metadata_files) if isinstance(metadata_files, list) else 0}"
        )
        log.info(
            "Perception filter inlet: interface=%s chat_id=%s messages=%d top_files=%d metadata_files=%d",
            interface or "unknown",
            chat_id or "none",
            len(messages),
            len(top_level_files) if isinstance(top_level_files, list) else 0,
            len(metadata_files) if isinstance(metadata_files, list) else 0,
        )
        if self.valves.debug_breadcrumbs and callable(__event_emitter__):
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": (
                            "PerceptionFilter inlet hit: "
                            f"messages={len(messages)} "
                            f"files={len(top_level_files) if isinstance(top_level_files, list) else 0}"
                        ),
                        "done": True,
                        "hidden": False,
                    },
                }
            )

        # Generate scope identifiers
        user_id = ""
        if __user__:
            user_id = __user__.get("id", "")
        session_id = _session_id(user_id, body)
        turn_id = _turn_id()

        # Scan messages for image content (typically in the last user message)
        staged_any = False
        image_counter = 0
        saw_image_attachment = False
        staged_names: list[str] = []
        saw_non_inline_image_payload = _body_contains_non_inline_image_payloads(body)
        print(
            "[PerceptionFilter] payload scan "
            f"non_inline_image_payload={saw_non_inline_image_payload}"
        )
        log.info(
            "Perception filter payload scan: non_inline_image_payload=%s",
            saw_non_inline_image_payload,
        )
        current_user_msg = _get_last_user_message(messages)
        if current_user_msg is not None:
            content = current_user_msg.get("content")
            if isinstance(content, list):
                log.info(
                    "Perception filter current user turn: content_parts=%d image_parts=%d",
                    len(content),
                    sum(
                        1
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "image_url"
                    ),
                )
                new_content_parts: list[Any] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        saw_image_attachment = True
                        image_url = part.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                        image_counter += 1
                        logical_name = _guess_logical_name(url, image_counter)
                        log.info(
                            "Perception filter staging message image: name=%s source_kind=%s",
                            logical_name,
                            _source_kind(url),
                        )
                        staged = await _stage_image_source(
                            self.valves.sidecar_url,
                            self.valves.webui_url,
                            session_id,
                            turn_id,
                            logical_name,
                            url,
                            self.valves.max_file_size_mb,
                        )
                        if staged:
                            staged_any = True
                            staged_names.append(logical_name)
                            new_content_parts.append({
                                "type": "text",
                                "text": f"[Attachment staged: {logical_name}]",
                            })
                        else:
                            new_content_parts.append({
                                "type": "text",
                                "text": f"[Attachment detected but staging failed: {logical_name}]",
                            })
                        continue

                    # Keep non-image parts as-is
                    new_content_parts.append(part)

                current_user_msg["content"] = new_content_parts

        # Open WebUI file attachments can arrive either at body["files"]
        # or nested under body["metadata"]["files"], depending on version/path.
        files = _collect_file_entries(body)
        if files:
            log.info("Perception filter found %d uploaded file entries.", len(files))
            for file_info in files:
                file_url, file_name, file_type = _extract_file_fields(file_info)

                if not _is_image_type(file_type) and not _is_image_name(file_name):
                    continue

                saw_image_attachment = True
                image_counter += 1
                logical_name = file_name or _guess_logical_name(file_url, image_counter)
                log.info(
                    "Perception filter staging file entry: name=%s mime=%s source_kind=%s",
                    logical_name,
                    file_type or "<unknown>",
                    _source_kind(file_url),
                )
                staged = await _stage_image_source(
                    self.valves.sidecar_url,
                    self.valves.webui_url,
                    session_id,
                    turn_id,
                    logical_name,
                    file_url,
                    self.valves.max_file_size_mb,
                )
                if staged:
                    staged_any = True
                    staged_names.append(logical_name)

        if saw_image_attachment:
            _strip_image_file_entries(body)
        if saw_image_attachment or saw_non_inline_image_payload:
            _scrub_raw_image_payloads(body)
            log.info(
                "Perception filter scrubbed payloads: saw_image_attachment=%s non_inline_image_payload=%s",
                saw_image_attachment,
                saw_non_inline_image_payload,
            )
            if saw_non_inline_image_payload and not saw_image_attachment:
                log.info(
                    "Perception filter scrubbed non-inline image payloads from an alternate request shape."
                )

        if not staged_any:
            return body

        # Store scope IDs in metadata so tools can find the sandbox
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["perception_session_id"] = session_id
        body["metadata"]["perception_turn_id"] = turn_id

        # Inject system hint
        _inject_system_hint(messages)
        log.info(
            "Perception filter activated sandbox: session=%s turn=%s staged=%s",
            session_id,
            turn_id,
            ",".join(dict.fromkeys(staged_names)),
        )

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
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".avif"))


def _collect_file_entries(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect uploaded file records from the common Open WebUI request shapes."""
    metadata = body.get("metadata", {})
    file_entries: list[dict[str, Any]] = []

    top_level_files = body.get("files", [])
    if isinstance(top_level_files, list):
        file_entries.extend(item for item in top_level_files if isinstance(item, dict))

    metadata_files = metadata.get("files", [])
    if isinstance(metadata_files, list):
        file_entries.extend(item for item in metadata_files if isinstance(item, dict))

    return file_entries


def _get_last_user_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the latest user message in the chat."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    return None


def _extract_file_fields(file_info: dict[str, Any]) -> tuple[str, str, str]:
    """Normalize file record fields across Open WebUI versions."""
    file_data = file_info.get("data", {})
    if not isinstance(file_data, dict):
        file_data = {}

    file_url = (
        file_data.get("url", "")
        or file_info.get("url", "")
        or file_info.get("src", "")
    )
    file_name = (
        file_data.get("name", "")
        or file_info.get("name", "")
        or file_info.get("filename", "")
    )
    file_type = (
        file_data.get("type", "")
        or file_info.get("type", "")
        or file_info.get("mime_type", "")
    )
    return file_url, file_name, file_type


def _strip_image_file_entries(body: dict[str, Any]) -> None:
    """Remove image file entries so the default model file pipeline cannot see them."""
    files = body.get("files")
    if isinstance(files, list):
        body["files"] = [item for item in files if not _file_entry_is_image(item)]

    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        metadata_files = metadata.get("files")
        if isinstance(metadata_files, list):
            metadata["files"] = [item for item in metadata_files if not _file_entry_is_image(item)]


def _file_entry_is_image(file_info: Any) -> bool:
    """Return True when a file entry looks like an image attachment."""
    if not isinstance(file_info, dict):
        return False
    _, file_name, file_type = _extract_file_fields(file_info)
    return _is_image_type(file_type) or _is_image_name(file_name)


def _scrub_raw_image_payloads(obj: Any) -> Any:
    """
    Recursively remove non-inline image payloads from alternate Open WebUI request shapes.

    The obvious `messages[*].content` and top-level `files` paths are handled
    earlier. This is a last-resort scrub for duplicate copies used in model
    continuation requests.
    """
    if isinstance(obj, list):
        scrubbed: list[Any] = []
        for item in obj:
            if _looks_like_non_inline_image_content_part(item):
                scrubbed.append(
                    {"type": "text", "text": "[Image attachment available via perception tools]"}
                )
                continue
            if _file_entry_is_image(item):
                continue
            scrubbed.append(_scrub_raw_image_payloads(item))
        return scrubbed

    if isinstance(obj, dict):
        if _looks_like_non_inline_image_content_part(obj):
            return {"type": "text", "text": "[Image attachment available via perception tools]"}

        for key, value in list(obj.items()):
            if key in {"files", "images"} and isinstance(value, list):
                obj[key] = [
                    _scrub_raw_image_payloads(item)
                    for item in value
                    if not _file_entry_is_image(item)
                ]
                continue
            obj[key] = _scrub_raw_image_payloads(value)
        return obj

    return obj


def _body_contains_non_inline_image_payloads(obj: Any) -> bool:
    """Return True when a request body still contains non-data image payloads."""
    if isinstance(obj, list):
        return any(_body_contains_non_inline_image_payloads(item) for item in obj)

    if isinstance(obj, dict):
        if _looks_like_non_inline_image_content_part(obj):
            return True

        if _file_entry_is_image(obj):
            file_url, _, _ = _extract_file_fields(obj)
            if file_url and not file_url.startswith("data:"):
                return True

        return any(_body_contains_non_inline_image_payloads(value) for value in obj.values())

    return False


def _looks_like_non_inline_image_content_part(value: Any) -> bool:
    """Return True when a value looks like an image content part with a non-data URL."""
    if not isinstance(value, dict):
        return False
    if value.get("type") != "image_url":
        return False
    url = _extract_image_content_url(value)
    return bool(url) and not url.startswith("data:")


def _extract_image_content_url(value: dict[str, Any]) -> str:
    """Extract the URL from an OpenAI-style image content part."""
    image_url = value.get("image_url", {})
    if isinstance(image_url, dict):
        return image_url.get("url", "")
    if isinstance(image_url, str):
        return image_url
    return ""


def _source_kind(url: str) -> str:
    """Return a short label describing the image source."""
    if not url:
        return "missing"
    if url.startswith("data:"):
        return "data_url"
    return "url"


def _guess_logical_name(url: str, index: int) -> str:
    """Guess a stable logical name from a URL or fall back to a numbered image."""
    if url and not url.startswith("data:"):
        path = urlparse(url).path
        candidate = path.rsplit("/", 1)[-1]
        if candidate and _is_image_name(candidate):
            return candidate
    return f"image_{index}.jpg"


async def _stage_image_source(
    sidecar_url: str,
    webui_url: str,
    session_id: str,
    turn_id: str,
    logical_name: str,
    source_url: str,
    max_mb: int,
) -> bool:
    """Stage either an inline data URL or a URL-backed image upload."""
    if source_url.startswith("data:"):
        return await _stage_data_url(
            sidecar_url,
            session_id,
            turn_id,
            logical_name,
            source_url,
            max_mb,
        )
    if not source_url:
        log.warning("Attachment %s had no source URL.", logical_name)
        return False
    return await _stage_fetch_url(
        sidecar_url,
        webui_url,
        session_id,
        turn_id,
        logical_name,
        source_url,
        max_mb,
    )


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


async def _stage_fetch_url(
    sidecar_url: str,
    webui_url: str,
    session_id: str,
    turn_id: str,
    logical_name: str,
    source_url: str,
    max_mb: int,
) -> bool:
    """Fetch an image by URL from Open WebUI and stage it in the sidecar."""
    try:
        fetch_url = urljoin(webui_url, source_url) if source_url.startswith("/") else source_url
        async with httpx.AsyncClient(timeout=30.0) as client:
            download = await client.get(fetch_url)
            download.raise_for_status()
            mime_type = download.headers.get("content-type", "application/octet-stream").split(";")[0]
            raw = download.content

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
                log.info("Fetched and staged %s from %s.", logical_name, fetch_url)
                return True
            log.error("Sidecar staging failed (%d): %s", resp.status_code, resp.text)
            return False
    except Exception as e:
        log.error("Failed to fetch attachment %s from %s: %s", logical_name, source_url, e)
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

