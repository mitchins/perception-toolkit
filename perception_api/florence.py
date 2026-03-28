"""
Florence-2 perception backend.

Handles model loading and inference for the perception sidecar.
Maps high-level intents (general, ocr, regions) to Florence-2 task tokens.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from perception_api.config import FlorenceConfig, get_config
from perception_api.devices import preferred_dtype_for_device, resolve_torch_device
from perception_api.image_codecs import load_image_rgb

log = logging.getLogger(__name__)

# Florence-2 task token mappings — internal only, never exposed to model
_GENERAL_TASK_TOKENS: dict[str, str] = {
    "caption": "<CAPTION>",
    "detailed": "<DETAILED_CAPTION>",
    "more_detailed": "<MORE_DETAILED_CAPTION>",
}

_INTENT_TO_TASKS: dict[str, list[str]] = {
    "ocr": ["<OCR>"],
    "regions": ["<DENSE_REGION_CAPTION>"],
}

# Lazy-loaded model and processor
_model: Any = None
_processor: Any = None
_device: str = "cpu"


def _load_model(cfg: FlorenceConfig) -> None:
    """Load Florence-2 model and processor. Called once on first inference."""
    global _model, _processor, _device

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    started = perf_counter()
    _device = resolve_torch_device(cfg.device, log)
    log.info("Loading Florence-2 model %s on %s ...", cfg.model_id, _device)

    dtype = preferred_dtype_for_device(_device)
    local_files_only = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"

    # attn_implementation="eager" required for Florence-2 compatibility with
    # transformers >= 4.46 — the model never implemented _supports_sdpa and
    # Microsoft has not released an update. Eager attention is correct here.
    _model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(_device).eval()

    _processor = AutoProcessor.from_pretrained(
        cfg.model_id,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )

    log.info(
        "Florence-2 model loaded successfully in %.2fs.",
        perf_counter() - started,
    )


def is_available() -> bool:
    """Check if the Florence backend is enabled in config."""
    return get_config().florence.enabled


def ensure_loaded() -> None:
    """Ensure the model is loaded. Safe to call multiple times."""
    if _model is None:
        cfg = get_config().florence
        if not cfg.enabled:
            raise RuntimeError("Florence backend is disabled in configuration.")
        _load_model(cfg)


def run_inference(image_path: Path, intent: str, query: str = "") -> str:
    """
    Run Florence-2 inference on an image.

    Args:
        image_path: Path to the image file on disk (sandbox-validated).
        intent: One of "general", "ocr", "regions".
        query: Optional text query for grounding tasks.

    Returns:
        Concise text result suitable for LLM consumption.
    """
    import torch
    ensure_loaded()
    cfg = get_config().florence
    started = perf_counter()

    tasks = _tasks_for_intent(intent, cfg.general_task)
    if not tasks:
        raise ValueError(f"Unknown intent: {intent}")

    image_started = perf_counter()
    image = load_image_rgb(image_path)
    image_ms = (perf_counter() - image_started) * 1000.0
    results: list[str] = []

    for task_token in tasks:
        task_started = perf_counter()
        prompt = task_token
        if query and intent == "regions":
            prompt = f"{task_token} {query}"

        processor_started = perf_counter()
        inputs = _processor(text=prompt, images=image, return_tensors="pt")
        inputs = _prepare_inputs(inputs)
        processor_ms = (perf_counter() - processor_started) * 1000.0

        generate_started = perf_counter()
        with torch.no_grad():
            generated_ids = _model.generate(
                **inputs,
                max_new_tokens=max(32, int(cfg.max_new_tokens)),
                num_beams=max(1, int(cfg.num_beams)),
                # Florence-2 custom code still trips over empty KV caches on
                # newer transformers releases; disable caching for stability.
                use_cache=False,
            )
        generate_ms = (perf_counter() - generate_started) * 1000.0

        # Trim input tokens — post_process_generation expects only new tokens
        decode_started = perf_counter()
        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        generated_text = _processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]
        parsed = _processor.post_process_generation(
            generated_text, task=task_token, image_size=image.size
        )
        decode_ms = (perf_counter() - decode_started) * 1000.0

        text = _format_parsed_output(task_token, parsed)
        if intent == "general":
            text = _annotate_general_caption(text)
        results.append(text)

        log.info(
            "Florence task=%s image=%s device=%s image_load_ms=%.1f processor_ms=%.1f generate_ms=%.1f decode_ms=%.1f total_ms=%.1f",
            task_token,
            image_path.name,
            _device,
            image_ms,
            processor_ms,
            generate_ms,
            decode_ms,
            (perf_counter() - task_started) * 1000.0,
        )

    log.info(
        "Florence inference complete image=%s intent=%s total_ms=%.1f",
        image_path.name,
        intent,
        (perf_counter() - started) * 1000.0,
    )
    return "\n".join(results)


def _prepare_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Move processor outputs to the model device and align float dtypes."""
    import torch

    model_dtype = getattr(_model, "dtype", None)
    prepared: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and model_dtype is not None:
                prepared[key] = value.to(device=_device, dtype=model_dtype)
            else:
                prepared[key] = value.to(_device)
        else:
            prepared[key] = value
    return prepared


def _tasks_for_intent(intent: str, general_task: str) -> list[str] | None:
    """Resolve high-level intent into Florence task tokens."""
    if intent == "general":
        task_name = (general_task or "detailed").strip().lower()
        task_token = _GENERAL_TASK_TOKENS.get(task_name, _GENERAL_TASK_TOKENS["detailed"])
        return [task_token]
    return _INTENT_TO_TASKS.get(intent)


def _format_parsed_output(task_token: str, parsed: Any) -> str:
    """Convert Florence-2 parsed output into clean text for the LLM."""
    if isinstance(parsed, str):
        return parsed.strip()

    if isinstance(parsed, dict):
        # Florence-2 returns dict keyed by task token
        value = parsed.get(task_token, parsed)

        if isinstance(value, str):
            return value.strip()

        if isinstance(value, dict):
            # Dense region captions: {bboxes: [...], labels: [...]}
            bboxes = value.get("bboxes", [])
            labels = value.get("labels", [])
            if labels:
                lines = []
                for i, label in enumerate(labels):
                    bbox_str = ""
                    if i < len(bboxes):
                        b = bboxes[i]
                        bbox_str = f" [region: {b}]" if b else ""
                    lines.append(f"- {label}{bbox_str}")
                return "\n".join(lines)

            # OCR with regions
            text_lines = value.get("text", [])
            if isinstance(text_lines, list):
                return "\n".join(str(t) for t in text_lines)

            return str(value)

        if isinstance(value, list):
            return "\n".join(str(v) for v in value)

    return str(parsed).strip()


def _annotate_general_caption(text: str) -> str:
    """Add a conservative caveat to Florence captions for LLM consumption."""
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    return (
        f"{cleaned} Note: Florence-2 can misidentify image medium/style; "
        "this may be an illustration, render, or other non-photographic image."
    )
