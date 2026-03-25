"""
Scoped attachment sandbox for the perception sidecar.

Manages per-session/per-turn file staging.
The model only sees logical file names — never raw filesystem paths.
All access is validated against a manifest per scope.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from perception_api.config import get_config

log = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"}


@dataclass
class AttachmentMeta:
    """Metadata for a single staged attachment."""
    logical_name: str
    mime_type: str
    width: int | None = None
    height: int | None = None
    size_bytes: int = 0
    staged_path: str = ""  # internal-only, never exposed to model
    staged_at: float = field(default_factory=time.time)

    def to_display(self) -> str:
        """Human-readable single-line summary for tool output."""
        size_str = _format_bytes(self.size_bytes)
        dims = f"{self.width}x{self.height}" if self.width and self.height else "unknown dims"
        return f"- {self.logical_name} ({self.mime_type}, {dims}, {size_str})"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("staged_path", None)  # never leak internal path
        return d


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / (1024 * 1024):.1f} MB"


class SandboxScope:
    """
    A scoped virtual sandbox for one session/turn.

    Files are stored on disk under:
        <base>/<session_id>/<turn_id>/

    A manifest tracks which logical names map to real staged files.
    """

    def __init__(self, session_id: str, turn_id: str) -> None:
        cfg = get_config()
        self.session_id = _sanitize_id(session_id)
        self.turn_id = _sanitize_id(turn_id)
        self.base = Path(cfg.sandbox.base_path) / self.session_id / self.turn_id
        self.base.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, AttachmentMeta] = {}

    def stage_file(self, logical_name: str, data: bytes, mime_type: str | None = None) -> AttachmentMeta:
        """Stage raw bytes into the sandbox under a logical name."""
        safe_name = _sanitize_filename(logical_name)
        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(safe_name)
            mime_type = mime_type or "application/octet-stream"

        if mime_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError(f"Unsupported file type: {mime_type}")

        dest = self.base / safe_name
        dest.write_bytes(data)

        width, height = _probe_image_dims(dest)

        meta = AttachmentMeta(
            logical_name=safe_name,
            mime_type=mime_type,
            width=width,
            height=height,
            size_bytes=len(data),
            staged_path=str(dest),
        )
        self._manifest[safe_name] = meta
        log.info("Staged attachment %s (%s, %d bytes)", safe_name, mime_type, len(data))
        return meta

    def stage_from_path(self, source_path: str | Path, logical_name: str | None = None) -> AttachmentMeta:
        """Stage a file already on disk (e.g. from Open WebUI upload dir)."""
        src = Path(source_path)
        if not src.is_file():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        name = logical_name or src.name
        data = src.read_bytes()
        mime_type, _ = mimetypes.guess_type(src.name)
        return self.stage_file(name, data, mime_type)

    def list_attachments(self) -> list[AttachmentMeta]:
        return list(self._manifest.values())

    def get_attachment(self, logical_name: str) -> AttachmentMeta | None:
        safe = _sanitize_filename(logical_name)
        return self._manifest.get(safe)

    def resolve_path(self, logical_name: str) -> Path | None:
        """
        Resolve a logical attachment name to its real filesystem path.
        Returns None if the name is not in the manifest.
        This is used internally by the sidecar — never exposed to the model.
        """
        meta = self.get_attachment(logical_name)
        if meta is None:
            return None
        p = Path(meta.staged_path)
        # Double-check the file is actually under our sandbox base
        try:
            p.resolve().relative_to(self.base.resolve())
        except ValueError:
            log.warning("Path traversal attempt blocked: %s", logical_name)
            return None
        if not p.is_file():
            return None
        return p

    def cleanup(self) -> None:
        """Remove all staged files for this scope."""
        if self.base.exists():
            shutil.rmtree(self.base, ignore_errors=True)
            log.info("Cleaned up sandbox scope %s/%s", self.session_id, self.turn_id)
        self._manifest.clear()


# ── In-memory scope registry ─────────────────────────────────────────

_scopes: dict[str, SandboxScope] = {}


def get_or_create_scope(session_id: str, turn_id: str) -> SandboxScope:
    key = f"{_sanitize_id(session_id)}:{_sanitize_id(turn_id)}"
    if key not in _scopes:
        _scopes[key] = SandboxScope(session_id, turn_id)
    return _scopes[key]


def get_scope(session_id: str, turn_id: str) -> SandboxScope | None:
    key = f"{_sanitize_id(session_id)}:{_sanitize_id(turn_id)}"
    return _scopes.get(key)


def find_latest_scope(session_id: str, logical_name: str | None = None) -> SandboxScope | None:
    """Return the newest non-empty scope for a session, optionally requiring a logical name."""
    safe_session = _sanitize_id(session_id)
    candidates: list[tuple[float, SandboxScope]] = []

    for scope in _scopes.values():
        if scope.session_id != safe_session or not scope._manifest:
            continue
        if logical_name and scope.get_attachment(logical_name) is None:
            continue
        latest_staged_at = max(meta.staged_at for meta in scope._manifest.values())
        candidates.append((latest_staged_at, scope))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def describe_session_scopes(session_id: str) -> list[dict[str, Any]]:
    """Return a debug-friendly summary of scopes currently held for a session."""
    safe_session = _sanitize_id(session_id)
    summaries: list[dict[str, Any]] = []

    for scope in _scopes.values():
        if scope.session_id != safe_session:
            continue
        attachment_names = sorted(scope._manifest.keys())
        latest_staged_at = max(
            (meta.staged_at for meta in scope._manifest.values()),
            default=0.0,
        )
        summaries.append(
            {
                "turn_id": scope.turn_id,
                "attachment_names": attachment_names,
                "attachment_count": len(attachment_names),
                "latest_staged_at": latest_staged_at,
            }
        )

    summaries.sort(key=lambda item: item["latest_staged_at"], reverse=True)
    return summaries


def remove_scope(session_id: str, turn_id: str) -> None:
    key = f"{_sanitize_id(session_id)}:{_sanitize_id(turn_id)}"
    scope = _scopes.pop(key, None)
    if scope:
        scope.cleanup()


def cleanup_expired_scopes() -> int:
    """Remove scopes whose files are older than TTL. Returns count removed."""
    cfg = get_config()
    ttl = cfg.sandbox.ttl_seconds
    now = time.time()
    expired_keys = []
    for key, scope in _scopes.items():
        if scope._manifest:
            oldest = min(m.staged_at for m in scope._manifest.values())
            if now - oldest > ttl:
                expired_keys.append(key)
        else:
            # Empty scope — check directory mtime
            if scope.base.exists() and (now - scope.base.stat().st_mtime) > ttl:
                expired_keys.append(key)
    for key in expired_keys:
        scope = _scopes.pop(key)
        scope.cleanup()
    if expired_keys:
        log.info("Cleaned up %d expired sandbox scopes", len(expired_keys))
    return len(expired_keys)


# ── Helpers ───────────────────────────────────────────────────────────

def _sanitize_id(value: str) -> str:
    """Sanitize a session/turn ID to prevent path traversal."""
    # Allow only alphanumeric, hyphens, underscores
    safe = "".join(c for c in value if c.isalnum() or c in "-_")
    if not safe:
        safe = hashlib.sha256(value.encode()).hexdigest()[:16]
    return safe[:128]


def _sanitize_filename(name: str) -> str:
    """Sanitize a logical filename — no path separators, no traversal."""
    # Strip any directory components
    name = Path(name).name
    # Remove anything but alphanumeric, dots, hyphens, underscores
    safe = "".join(c for c in name if c.isalnum() or c in ".-_")
    if not safe:
        safe = "unnamed"
    return safe[:255]


def _probe_image_dims(path: Path) -> tuple[int | None, int | None]:
    """Try to read image dimensions without heavy dependencies."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return None, None
