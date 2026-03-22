"""
Torch device selection helpers for sidecar backends.
"""

from __future__ import annotations

import logging


def resolve_torch_device(preferred: str, log: logging.Logger | None = None) -> str:
    """Resolve a requested torch device to an available runtime device."""
    import torch

    requested = (preferred or "auto").strip().lower()

    if requested in {"auto", "default"}:
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return preferred
        if log:
            log.warning("Requested torch device %s is unavailable; falling back to cpu.", preferred)
        return "cpu"

    if requested.startswith("mps"):
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if log:
            log.warning("Requested torch device %s is unavailable; falling back to cpu.", preferred)
        return "cpu"

    if requested.startswith("cpu"):
        return "cpu"

    if log:
        log.warning("Unknown torch device %s; falling back to cpu.", preferred)
    return "cpu"


def preferred_dtype_for_device(device: str):
    """Choose a conservative default dtype for a torch device."""
    import torch

    if device.startswith("cuda") or device.startswith("mps"):
        return torch.float16
    return torch.float32
