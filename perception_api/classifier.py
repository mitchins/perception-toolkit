"""
Optional lightweight media-type classifier backend for the perception sidecar.

This backend is intended for quick attachment indexing and routing hints rather
than end-user-facing semantic analysis. It currently supports a small closed set
of visual media classes from a local timm checkpoint export.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from perception_api.config import get_config
from perception_api.devices import resolve_torch_device
from perception_api.image_codecs import load_image_rgb

log = logging.getLogger(__name__)

_model: Any = None
_device: str = "cpu"
_loaded: bool = False
_runtime_info: dict[str, Any] = {}
_class_names: list[str] = []
_input_size: int = 336
_resize_size: int = 384
_mean: list[float] = [0.485, 0.456, 0.406]
_std: list[float] = [0.229, 0.224, 0.225]


def is_available() -> bool:
    """Check if the classifier backend is enabled in config."""
    return get_config().classifier.enabled


def get_runtime_info() -> dict[str, Any]:
    """Return lightweight runtime info for diagnostics."""
    return dict(_runtime_info)


def ensure_loaded() -> None:
    """Load the classifier model if not already loaded."""
    global _model, _device, _loaded, _runtime_info, _class_names, _input_size, _resize_size, _mean, _std

    if _loaded:
        return

    cfg = get_config().classifier
    if not cfg.enabled:
        raise RuntimeError("Classifier backend is disabled in configuration.")

    import torch
    import timm

    started = perf_counter()
    checkpoint_path, deploy_config = _resolve_bundle(cfg.model_path)

    _device = resolve_torch_device(cfg.device, log)
    _class_names = _extract_class_names(deploy_config, checkpoint_path.parent)
    _input_size = int(deploy_config.get("input_size") or deploy_config.get("eval_size") or 336)
    _resize_size = int(round(_input_size * 1.143))
    normalization = deploy_config.get("normalization", {}) or {}
    _mean = [float(value) for value in normalization.get("mean", _mean)]
    _std = [float(value) for value in normalization.get("std", _std)]

    model_name = str(deploy_config.get("model", "tf_efficientnet_b0"))
    log.info(
        "Loading classifier model %s from %s on %s ...",
        model_name,
        checkpoint_path,
        _device,
    )

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(_class_names),
    )

    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(state, model)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(_device).eval()

    _model = model
    _loaded = True
    load_ms = (perf_counter() - started) * 1000.0
    _runtime_info = {
        "checkpoint_path": str(checkpoint_path),
        "device": _device,
        "class_names": list(_class_names),
        "input_size": _input_size,
        "resize_size": _resize_size,
        "load_ms": round(load_ms, 1),
    }
    log.info(
        "Classifier model loaded successfully in %.1f ms with %d classes.",
        load_ms,
        len(_class_names),
    )


def classify_image(image_path: Path) -> dict[str, Any]:
    """
    Run closed-set media-type classification on an image.

    Returns a dict containing the predicted label, confidence, timing, and device.
    """
    import torch

    ensure_loaded()

    cfg = get_config().classifier
    started = perf_counter()
    image = load_image_rgb(image_path)
    tensor = _preprocess_image(image).to(_device)
    preprocess_ms = (perf_counter() - started) * 1000.0

    infer_started = perf_counter()
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        confidence, class_index = torch.max(probs, dim=0)
    inference_ms = (perf_counter() - infer_started) * 1000.0
    total_ms = (perf_counter() - started) * 1000.0

    index = int(class_index.item())
    label = _class_names[index] if 0 <= index < len(_class_names) else str(index)
    score = float(confidence.item())
    low_confidence = score < float(cfg.threshold_default)

    result = {
        "label": label,
        "confidence": score,
        "class_index": index,
        "device": _device,
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "total_ms": total_ms,
        "low_confidence": low_confidence,
    }
    log.info(
        "Classifier inference complete image=%s device=%s label=%s confidence=%.3f total_ms=%.1f",
        image_path.name,
        _device,
        label,
        score,
        total_ms,
    )
    return result


def _resolve_bundle(model_path: str) -> tuple[Path, dict[str, Any]]:
    raw_path = Path(model_path)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        candidates.extend([project_root / raw_path, Path.cwd() / raw_path])

    target = None
    for candidate in candidates:
        if candidate.exists():
            target = candidate.resolve()
            break

    if target is None:
        raise FileNotFoundError(f"Classifier model path not found: {model_path}")

    bundle_dir = target if target.is_dir() else target.parent
    deploy_config_path = bundle_dir / "deploy_config.json"
    if not deploy_config_path.is_file():
        raise FileNotFoundError(f"Classifier deploy_config.json not found in {bundle_dir}")

    deploy_config = json.loads(deploy_config_path.read_text(encoding="utf-8"))
    checkpoint_name = str(deploy_config.get("checkpoint_file", "best_model.pth"))
    checkpoint_path = target if target.is_file() else bundle_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

    return checkpoint_path, deploy_config


def _extract_class_names(deploy_config: dict[str, Any], bundle_dir: Path) -> list[str]:
    class_names = deploy_config.get("class_names")
    if isinstance(class_names, list) and class_names:
        return [str(item) for item in class_names]

    labels_path = bundle_dir / "labels.json"
    if labels_path.is_file():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        class_names = labels.get("class_names")
        if isinstance(class_names, list) and class_names:
            return [str(item) for item in class_names]

    raise ValueError("Classifier export is missing class_names metadata.")


def _extract_state_dict(state: Any, model: Any) -> Any:
    if isinstance(state, dict):
        if isinstance(state.get("state_dict"), dict):
            state_dict = _strip_module_prefix(state["state_dict"])
        elif isinstance(state.get("model_state_dict"), dict):
            state_dict = _strip_module_prefix(state["model_state_dict"])
        else:
            state_dict = _strip_module_prefix(state)
    else:
        state_dict = state

    if isinstance(state_dict, dict) and any(str(key).startswith("_") for key in state_dict.keys()):
        return _remap_legacy_efficientnet_keys(state_dict, model)
    return state_dict


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if not all(isinstance(key, str) for key in state_dict.keys()):
        return state_dict
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def _remap_legacy_efficientnet_keys(state_dict: dict[str, Any], model: Any) -> dict[str, Any]:
    """Translate older timm EfficientNet key names into current grouped-stage names."""
    block_index_map = _build_block_index_map(model)
    has_expand: dict[int, bool] = {}

    for key in state_dict.keys():
        if not isinstance(key, str) or not key.startswith("_blocks."):
            continue
        parts = key.split(".", 2)
        if len(parts) < 3:
            continue
        flat_idx = int(parts[1])
        has_expand.setdefault(flat_idx, False)
        if parts[2].startswith("_expand_conv."):
            has_expand[flat_idx] = True

    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            remapped[key] = value
            continue

        if key.endswith(".num_batches_tracked"):
            continue

        mapped_key = _map_top_level_key(key)
        if mapped_key is not None:
            remapped[mapped_key] = value
            continue

        if not key.startswith("_blocks."):
            remapped[key] = value
            continue

        parts = key.split(".", 2)
        if len(parts) < 3:
            remapped[key] = value
            continue

        flat_idx = int(parts[1])
        suffix = parts[2]
        stage_idx, block_idx = block_index_map[flat_idx]
        prefix = f"blocks.{stage_idx}.{block_idx}."
        expand = has_expand.get(flat_idx, False)

        if suffix.startswith("_expand_conv."):
            remapped[prefix + "conv_pw." + suffix[len("_expand_conv."):]] = value
        elif suffix.startswith("_bn0."):
            remapped[prefix + "bn1." + suffix[len("_bn0."):]] = value
        elif suffix.startswith("_depthwise_conv."):
            remapped[prefix + "conv_dw." + suffix[len("_depthwise_conv."):]] = value
        elif suffix.startswith("_bn1."):
            bn_name = "bn2" if expand else "bn1"
            remapped[prefix + bn_name + "." + suffix[len("_bn1."):]] = value
        elif suffix.startswith("_se_reduce."):
            remapped[prefix + "se.conv_reduce." + suffix[len("_se_reduce."):]] = value
        elif suffix.startswith("_se_expand."):
            remapped[prefix + "se.conv_expand." + suffix[len("_se_expand."):]] = value
        elif suffix.startswith("_project_conv."):
            proj_name = "conv_pwl" if expand else "conv_pw"
            remapped[prefix + proj_name + "." + suffix[len("_project_conv."):]] = value
        elif suffix.startswith("_bn2."):
            bn_name = "bn3" if expand else "bn2"
            remapped[prefix + bn_name + "." + suffix[len("_bn2."):]] = value
        else:
            remapped[key] = value

    return remapped


def _map_top_level_key(key: str) -> str | None:
    top_level_mappings = (
        ("_conv_stem.", "conv_stem."),
        ("_bn0.", "bn1."),
        ("_conv_head.", "conv_head."),
        ("_bn1.", "bn2."),
        ("_fc.", "classifier."),
    )
    for old_prefix, new_prefix in top_level_mappings:
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]
    return None


def _build_block_index_map(model: Any) -> dict[int, tuple[int, int]]:
    mapping: dict[int, tuple[int, int]] = {}
    flat_idx = 0
    for stage_idx, stage in enumerate(getattr(model, "blocks", [])):
        for block_idx, _ in enumerate(stage):
            mapping[flat_idx] = (stage_idx, block_idx)
            flat_idx += 1
    if not mapping:
        raise RuntimeError("Unable to derive EfficientNet block layout for classifier remapping.")
    return mapping


def _preprocess_image(image: Image.Image):
    import torch

    resized = _resize_preserving_aspect(image, _resize_size)
    cropped = _center_crop(resized, _input_size)
    raw = torch.tensor(bytearray(cropped.tobytes()), dtype=torch.uint8)
    tensor = raw.view(cropped.height, cropped.width, 3).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor(_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_std, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


def _resize_preserving_aspect(image: Image.Image, target_short_side: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions.")

    if width < height:
        new_width = target_short_side
        new_height = round((height / width) * target_short_side)
    else:
        new_height = target_short_side
        new_width = round((width / height) * target_short_side)

    return image.resize((new_width, new_height), Image.Resampling.BICUBIC)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    right = min(width, left + size)
    bottom = min(height, top + size)
    return image.crop((left, top, right, bottom))
