"""
Shared image codec checks and decode helpers for perception backends.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image, features

log = logging.getLogger("perception_api.codecs")


def get_pillow_codec_support() -> dict[str, dict[str, Any]]:
    """Return lightweight codec support info from the active Pillow runtime."""
    registered = Image.registered_extensions()
    return {
        "jpeg": {
            "registered": registered.get(".jpg") == "JPEG" or registered.get(".jpeg") == "JPEG",
            "feature": _safe_check_feature("jpg"),
        },
        "png": {
            "registered": registered.get(".png") == "PNG",
            "feature": _safe_check_feature("zlib"),
        },
        "webp": {
            "registered": registered.get(".webp") == "WEBP",
            "feature": _safe_check_feature("webp"),
        },
        "avif": {
            "registered": registered.get(".avif") == "AVIF" or registered.get(".avifs") == "AVIF",
            "feature": _safe_check_feature("avif"),
        },
    }


def probe_image_info(path: Path) -> tuple[int | None, int | None, str | None]:
    """
    Probe dimensions and return a decode warning when Pillow can identify a file
    but not fully decode it.
    """
    try:
        with Image.open(path) as img:
            size = img.size
            try:
                img.load()
                return size[0], size[1], None
            except Exception as exc:
                return size[0], size[1], format_decode_error(path, exc)
    except Exception as exc:
        return None, None, format_decode_error(path, exc)


def load_image_rgb(path: Path) -> Image.Image:
    """Load an image and convert it to RGB with a clearer codec failure message."""
    try:
        with Image.open(path) as img:
            img.load()
            return img.convert("RGB")
    except Exception as exc:
        raise RuntimeError(format_decode_error(path, exc)) from exc


def format_decode_error(path: Path, exc: Exception) -> str:
    """Return a human-readable decode error with codec installation hints."""
    ext = path.suffix.lower()
    message = str(exc).strip() or exc.__class__.__name__

    if "No codec available" in message:
        if ext in {".avif", ".avifs"}:
            return (
                "AVIF decode failed: Pillow sees the file type but no AVIF decoder is available at runtime. "
                "Install an AVIF-capable Pillow codec in the perception environment "
                "(for example `pillow-avif-plugin`) or rebuild Pillow with libavif support."
            )
        if ext == ".webp":
            return (
                "WEBP decode failed: Pillow sees the file type but no WEBP decoder is available at runtime. "
                "Reinstall Pillow with libwebp support in the perception environment."
            )

    if ext in {".avif", ".avifs"}:
        return f"AVIF image decode failed: {message}"
    if ext == ".webp":
        return f"WEBP image decode failed: {message}"
    return f"Image decode failed: {message}"


def run_startup_codec_self_check(sample_root: Path | None = None) -> None:
    """
    Log lightweight codec support visible from the active Pillow runtime.
    This is best-effort and must never affect startup.
    """
    support = get_pillow_codec_support()
    log.info("Pillow codec support: %s", support)

    # Deliberately avoid probing local sample files here. Actual file-based
    # codec diagnostics live behind an explicit diagnostics access point.

def collect_codec_diagnostics(sample_root: Path | None = None) -> dict[str, Any]:
    """
    Return richer codec diagnostics, including an actual decode attempt for one
    local WEBP and AVIF sample when such files are available.
    """
    support = get_pillow_codec_support()
    root = sample_root or Path(__file__).resolve().parents[1] / "test_resources"
    diagnostics: dict[str, Any] = {
        "support": support,
        "sample_root": str(root),
        "sample_root_exists": root.exists(),
        "sample_checks": {},
    }
    if not root.exists():
        return diagnostics

    for extension in (".webp", ".avif"):
        sample = _find_codec_sample(root, extension)
        if sample is None:
            diagnostics["sample_checks"][extension] = {"found": False}
            continue

        start = perf_counter()
        check: dict[str, Any] = {
            "found": True,
            "sample": sample.name,
        }
        try:
            image = load_image_rgb(sample)
            try:
                width, height = image.size
            finally:
                image.close()
            check.update(
                {
                    "ok": True,
                    "width": width,
                    "height": height,
                    "elapsed_ms": round((perf_counter() - start) * 1000.0, 1),
                }
            )
        except Exception as exc:
            check.update(
                {
                    "ok": False,
                    "elapsed_ms": round((perf_counter() - start) * 1000.0, 1),
                    "error": str(exc),
                }
            )
        diagnostics["sample_checks"][extension] = check

    return diagnostics


def _safe_check_feature(name: str) -> bool | None:
    try:
        return bool(features.check(name))
    except Exception:
        return None


def _find_codec_sample(root: Path, extension: str) -> Path | None:
    """Return the first matching sample file for a codec self-check."""
    for path in sorted(root.rglob(f"*{extension}")):
        if path.is_file():
            return path
    return None
