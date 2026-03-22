"""
Optional tagger backend for the perception sidecar.

Supports WD-14 style taggers. Only loaded if enabled in config.
This module is a sidecar concern — Open WebUI never touches it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from perception_api.config import WD14Config, get_config
from perception_api.devices import resolve_torch_device

log = logging.getLogger(__name__)

_model: Any = None
_labels: list[str] = []
_device: str = "cpu"
_loaded: bool = False


def is_available() -> bool:
    """Check if the tagger backend is enabled in config."""
    return get_config().wd14.enabled


def ensure_loaded() -> None:
    """Load the WD-14 tagger model if not already loaded."""
    global _model, _labels, _device, _loaded

    if _loaded:
        return

    cfg = get_config().wd14
    if not cfg.enabled:
        raise RuntimeError("WD-14 tagger backend is disabled in configuration.")

    try:
        import torch
        import timm
        from huggingface_hub import hf_hub_download
        import pandas as pd

        _device = resolve_torch_device(cfg.device, log)
        log.info("Loading WD-14 tagger model %s on %s ...", cfg.model_id, _device)

        _model = timm.create_model(
            f"hf_hub:{cfg.model_id}",
            pretrained=True,
        ).to(_device).eval()

        # Load tag labels
        labels_path = hf_hub_download(cfg.model_id, filename="selected_tags.csv")
        df = pd.read_csv(labels_path)
        _labels = df["name"].tolist()

        _loaded = True
        log.info("WD-14 tagger loaded successfully (%d tags).", len(_labels))

    except Exception as e:
        log.error("Failed to load WD-14 tagger: %s", e)
        raise


def tag_image(image_path: Path, threshold: float | None = None) -> list[tuple[str, float]]:
    """
    Run WD-14 tagging on an image.

    Args:
        image_path: Path to the image file (sandbox-validated).
        threshold: Minimum confidence threshold. None → use config default.

    Returns:
        List of (tag, confidence) tuples, sorted by descending confidence.
    """
    import torch
    import numpy as np
    from PIL import Image
    from timm.data import resolve_data_config, create_transform

    ensure_loaded()

    cfg = get_config().wd14
    if threshold is None:
        threshold = cfg.threshold_default
    threshold = max(0.0, min(1.0, threshold))  # clamp

    image = Image.open(image_path).convert("RGB")

    # Build transform from model config
    transform = create_transform(**resolve_data_config(_model.pretrained_cfg))
    tensor = transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    results = []
    for i, prob in enumerate(probs):
        if prob >= threshold and i < len(_labels):
            results.append((_labels[i], float(prob)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def format_tags_for_llm(tags: list[tuple[str, float]]) -> str:
    """Format tag results as concise text for LLM consumption."""
    if not tags:
        return "No tags met the confidence threshold."
    lines = [f"- {tag} ({conf:.2f})" for tag, conf in tags]
    return "Image tags:\n" + "\n".join(lines)
