#!/usr/bin/env python3
"""
Small caption-eval harness for the seed images in test_resources/.

The benchmark scores captions by whether they surface important distinctions
such as medium, subject, color treatment, text, and composition.
It is intentionally dimension-based rather than reference-caption based.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HF_MODULES_CACHE_DIR = PROJECT_ROOT / ".hf_modules_cache"
HF_MODULES_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_MODULES_CACHE", str(HF_MODULES_CACHE_DIR))

from perception_api.devices import resolve_torch_device

IMPORTANCE_WEIGHTS = {
    "critical": 3.0,
    "major": 2.0,
    "minor": 1.0,
}

SMOLVLM_SALIENCE_PROMPT = (
    "Describe this image for a downstream tool-using model. Mention the most "
    "decision-relevant distinctions first: whether it is a photograph, "
    "illustration, render, texture/background asset, black-and-white vs color, "
    "visible text, main subject, and unusual style or composition. Keep it concise."
)

MOONDREAM_SALIENCE_PROMPT = (
    "Describe what matters in this image for a downstream tool-using model. "
    "Mention only the image facts that are actually important, such as whether "
    "it is a photograph, illustration, render, graphic, or texture/background "
    "asset; whether it is black-and-white; any visible text; the main subject; "
    "and any unusually important visual distinction. Keep it concise and answer "
    "only with the description."
)

DEFAULT_MODEL_IDS = {
    "florence-local": "microsoft/Florence-2-base",
    "blip-local": "Salesforce/blip-image-captioning-base",
    "smolvlm-local": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "moondream-local": "vikhyatk/moondream2",
    "clipcap-local": "Hamza66628/clip-prefix-caption-conceptual",
    "paligemma-local": "google/paligemma-3b-mix-448",
}

DEFAULT_MOONDREAM_QUERY_SETTINGS = {"max_tokens": 96, "temperature": 0.0, "top_p": 0.3}
DEFAULT_MOONDREAM_CAPTION_SETTINGS = {"max_tokens": 96, "temperature": 0.0, "top_p": 0.3}
DEFAULT_PALIGEMMA_MAX_NEW_TOKENS = 96
FLORENCE_TASK_TOKENS = {
    "caption": "<CAPTION>",
    "detailed-caption": "<DETAILED_CAPTION>",
    "more-detailed-caption": "<MORE_DETAILED_CAPTION>",
}
DEFAULT_FLORENCE_TASK = "more-detailed-caption"
DEFAULT_PALIGEMMA_CAPTION_PROMPT = "<image> caption en"
DEFAULT_PALIGEMMA_QUERY_PROMPT = (
    "<image> Describe what matters in this image for a downstream tool-using model. "
    "Mention only the image facts that are actually important, such as whether "
    "it is a photograph, illustration, render, graphic, or texture/background "
    "asset; whether it is black-and-white; any visible text; the main subject; "
    "and any unusually important visual distinction. Keep it concise and answer "
    "only with the description."
)
DEFAULT_CLIPCAP_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_CLIPCAP_GPT_MODEL_ID = "gpt2"
DEFAULT_CLIPCAP_CHECKPOINT_FILES = (
    "model.pt",
    "conceptual_weights.pt",
    "coco_weights.pt",
    "pytorch_model.bin",
)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    normalized = normalize_text(text)
    return len(normalized.split()) if normalized else 0


def phrase_matches(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    tokens = normalized_phrase.split()
    if tokens and tokens[-1].isalpha() and not tokens[-1].endswith("s"):
        tokens[-1] = rf"{re.escape(tokens[-1])}s?"
    else:
        tokens[-1] = re.escape(tokens[-1])
    if len(tokens) > 1:
        pattern = r"\b" + r"\s+".join([re.escape(token) for token in normalized_phrase.split()[:-1]] + [tokens[-1]]) + r"\b"
    else:
        pattern = r"\b" + tokens[-1] + r"\b"
    return re.search(pattern, normalized_text) is not None


def load_ground_truth(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "predictions" in data and isinstance(data["predictions"], list):
            return {
                item["filename"]: item["caption"]
                for item in data["predictions"]
                if "filename" in item and "caption" in item
            }
        return {str(k): str(v) for k, v in data.items()}

    if isinstance(data, list):
        return {
            item["filename"]: item["caption"]
            for item in data
            if isinstance(item, dict) and "filename" in item and "caption" in item
        }

    raise ValueError(f"Unsupported predictions file format: {path}")


def write_predictions(path: Path, model_name: str, predictions: dict[str, str]) -> None:
    payload = {
        "model": model_name,
        "predictions": [
            {"filename": filename, "caption": caption}
            for filename, caption in predictions.items()
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def prepare_model_inputs(inputs: dict[str, Any], device: str, model_dtype: Any | None = None) -> dict[str, Any]:
    import torch

    prepared: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and model_dtype is not None:
                prepared[key] = value.to(device=device, dtype=model_dtype)
            else:
                prepared[key] = value.to(device)
        else:
            prepared[key] = value
    return prepared


class FlorenceLocalBackend:
    def __init__(self, model_id: str, device: str, task_name: str):
        from perception_api import florence
        from perception_api.config import get_config

        self.florence = florence
        cfg = get_config()
        cfg.florence.enabled = True
        cfg.florence.model_id = model_id
        cfg.florence.device = device

        florence._model = None
        florence._processor = None
        florence._device = "cpu"

        self.model_id = model_id
        self.device = device
        self.task_name = task_name
        self.task_token = FLORENCE_TASK_TOKENS[task_name]

    def caption(self, image_path: Path) -> str:
        import torch
        from PIL import Image

        florence = self.florence
        florence.ensure_loaded()

        image = Image.open(image_path).convert("RGB")
        task_token = self.task_token
        inputs = florence._processor(text=task_token, images=image, return_tensors="pt")
        inputs = florence._prepare_inputs(inputs)

        with torch.no_grad():
            generated_ids = florence._model.generate(
                **inputs,
                max_new_tokens=1024,
                num_beams=3,
                use_cache=False,
            )

        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        generated_text = florence._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False
        )[0]
        parsed = florence._processor.post_process_generation(
            generated_text,
            task=task_token,
            image_size=image.size,
        )
        return florence._format_parsed_output(task_token, parsed)


class BlipLocalBackend:
    def __init__(self, model_id: str, device: str):
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.model_id = model_id
        self.device = resolve_torch_device(device)
        self.processor = BlipProcessor.from_pretrained(model_id, local_files_only=True)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_id,
            local_files_only=True,
        ).to(self.device).eval()

    def caption(self, image_path: Path) -> str:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = prepare_model_inputs(inputs, self.device, getattr(self.model, "dtype", None))
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=80)
        return self.processor.decode(output[0], skip_special_tokens=True).strip()


class SmolVLMBackend:
    def __init__(self, model_id: str, device: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.device = resolve_torch_device(device)
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            local_files_only=True,
        ).to(self.device).eval()

    def caption(self, image_path: Path) -> str:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SMOLVLM_SALIENCE_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt")
        inputs = prepare_model_inputs(inputs, self.device, getattr(self.model, "dtype", None))
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
            )
        generated = output[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def resolve_checkpoint_path(
    model_id: str,
    candidate_filenames: tuple[str, ...],
    *,
    local_only: bool,
) -> Path:
    from transformers.utils.hub import cached_file

    model_path = Path(model_id)
    if model_path.is_file():
        return model_path
    if model_path.is_dir():
        for filename in candidate_filenames:
            candidate = model_path / filename
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No checkpoint file found in {model_path}. Tried: {', '.join(candidate_filenames)}"
        )

    for filename in candidate_filenames:
        try:
            checkpoint_path = cached_file(model_id, filename, local_files_only=True)
            return Path(checkpoint_path)
        except Exception:
            continue

    if not local_only:
        for filename in candidate_filenames:
            try:
                checkpoint_path = cached_file(model_id, filename, local_files_only=False)
                return Path(checkpoint_path)
            except Exception:
                continue

    raise FileNotFoundError(
        f"No checkpoint file found for {model_id}. Tried: {', '.join(candidate_filenames)}"
    )


class ClipCapMLP(torch.nn.Module):
    def __init__(self, sizes: tuple[int, ...], bias: bool = True):
        import torch.nn as nn

        super().__init__()
        layers = []
        for index in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[index], sizes[index + 1], bias=bias))
            if index < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ClipCapModel(torch.nn.Module):
    def __init__(
        self,
        prefix_length: int,
        *,
        gpt_model_id: str,
        local_only: bool,
        prefix_size: int = 512,
    ):
        from transformers import GPT2LMHeadModel

        super().__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(
            gpt_model_id,
            local_files_only=local_only,
        )
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:
            self.clip_project = torch.nn.Linear(
                prefix_size,
                self.gpt_embedding_size * prefix_length,
            )
        else:
            self.clip_project = ClipCapMLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )


def generate_clipcap_beam(
    model: ClipCapModel,
    tokenizer: Any,
    *,
    embed: Any,
    beam_size: int,
    entry_length: int,
    temperature: float = 1.0,
    stop_token: str = ".",
) -> list[str]:
    import numpy as np
    import torch

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

    with torch.no_grad():
        generated = embed
        for _ in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens = next_tokens.permute(1, 0)
                scores = scores.squeeze(0)
                tokens = next_tokens if tokens is None else torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    return [output_texts[i] for i in order]


class ClipCapBackend:
    def __init__(
        self,
        model_id: str,
        device: str,
        *,
        local_only: bool,
        prefix_length: int,
        beam_size: int,
        entry_length: int,
        clip_model_id: str,
        gpt_model_id: str,
    ):
        from transformers import CLIPImageProcessor, CLIPModel, GPT2Tokenizer

        self.device = resolve_torch_device(device)
        self.prefix_length = prefix_length
        self.beam_size = beam_size
        self.entry_length = entry_length

        checkpoint_path = resolve_checkpoint_path(
            model_id,
            DEFAULT_CLIPCAP_CHECKPOINT_FILES,
            local_only=local_only,
        )

        self.processor = CLIPImageProcessor.from_pretrained(
            clip_model_id,
            local_files_only=local_only,
        )
        self.clip_model = CLIPModel.from_pretrained(
            clip_model_id,
            local_files_only=local_only,
        ).to(self.device).eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            gpt_model_id,
            local_files_only=local_only,
        )
        self.model = ClipCapModel(
            prefix_length=prefix_length,
            gpt_model_id=gpt_model_id,
            local_only=local_only,
        ).to(self.device).eval()

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self.model.load_state_dict(checkpoint, strict=False)

    def caption(self, image_path: Path) -> str:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.float32)

        with torch.no_grad():
            prefix = self.clip_model.get_image_features(pixel_values=pixel_values).to(
                self.device, dtype=torch.float32
            )
            prefix_embed = self.model.clip_project(prefix).reshape(1, self.prefix_length, -1)
            candidates = generate_clipcap_beam(
                self.model,
                self.tokenizer,
                embed=prefix_embed,
                beam_size=self.beam_size,
                entry_length=self.entry_length,
            )
        return candidates[0].strip()


class PaliGemmaBackend:
    def __init__(
        self,
        model_id: str,
        device: str,
        *,
        local_only: bool,
        inference_mode: str,
        max_new_tokens: int,
        prompt_override: str | None,
    ):
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        self.device = resolve_torch_device(device)
        self.inference_mode = inference_mode
        self.max_new_tokens = max_new_tokens
        self.prompt_override = prompt_override
        self.local_only = local_only

        if self.device == "cpu":
            dtype = torch.float32
        elif self.device == "mps":
            dtype = torch.bfloat16
        else:
            dtype = torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=local_only,
        )
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            local_files_only=local_only,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

    def _prompt(self) -> str:
        if self.prompt_override:
            return self.prompt_override
        if self.inference_mode == "query":
            return DEFAULT_PALIGEMMA_QUERY_PROMPT
        return DEFAULT_PALIGEMMA_CAPTION_PROMPT

    def caption(self, image_path: Path) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        prompt = self._prompt()
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )
        inputs = prepare_model_inputs(inputs, self.device, getattr(self.model, "dtype", None))

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_trimmed = generated_ids[:, input_length:]
        if generated_trimmed.numel() == 0:
            generated_trimmed = generated_ids
        text = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return text


def resolve_local_tokenizer_json(model_id: str) -> Path:
    from transformers.utils.hub import cached_file

    model_path = Path(model_id)
    if model_path.exists():
        tokenizer_path = model_path / "tokenizer.json"
        if tokenizer_path.exists():
            return tokenizer_path
        raise FileNotFoundError(f"No tokenizer.json found under local model path: {model_path}")

    tokenizer_path = cached_file(model_id, "tokenizer.json", local_files_only=True)
    return Path(tokenizer_path)


def resolve_local_model_dir(model_id: str) -> Path:
    from transformers.utils.hub import cached_file

    model_path = Path(model_id)
    if model_path.exists():
        return model_path

    config_path = cached_file(model_id, "config.json", local_files_only=True)
    return Path(config_path).parent


def ensure_moondream_py39_dynamic_module(model_id: str) -> None:
    if sys.version_info >= (3, 10):
        return

    snapshot_dir = resolve_local_model_dir(model_id)
    if snapshot_dir.name == "snapshots":
        raise ValueError(f"Expected a snapshot revision directory, got: {snapshot_dir}")

    if "/" in model_id:
        namespace, repo_name = model_id.split("/", 1)
    else:
        namespace, repo_name = "_local", Path(model_id).name

    dynamic_root = HF_MODULES_CACHE_DIR / "transformers_modules"
    target_dir = dynamic_root / namespace / repo_name / snapshot_dir.name
    target_dir.mkdir(parents=True, exist_ok=True)

    for package_dir in (
        dynamic_root,
        dynamic_root / namespace,
        dynamic_root / namespace / repo_name,
        target_dir,
    ):
        init_file = package_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")

    for source_path in snapshot_dir.glob("*.py"):
        text = source_path.read_text(encoding="utf-8")
        if source_path.name == "hf_moondream.py":
            text = text.replace(
                "from typing import Union",
                "from typing import Optional, Union",
            )
            text = text.replace(
                "device: torch.device | None = None",
                "device: Optional[torch.device] = None",
            )
        (target_dir / source_path.name).write_text(text, encoding="utf-8")


@contextlib.contextmanager
def moondream_local_tokenizer_patch(model_id: str):
    from tokenizers import Tokenizer

    local_tokenizer_json = resolve_local_tokenizer_json(model_id)
    original_from_pretrained = Tokenizer.from_pretrained

    def patched_from_pretrained(identifier: str, *args: Any, **kwargs: Any):
        if identifier == "moondream/starmie-v1":
            return Tokenizer.from_file(str(local_tokenizer_json))
        return original_from_pretrained(identifier, *args, **kwargs)

    Tokenizer.from_pretrained = patched_from_pretrained
    try:
        yield
    finally:
        Tokenizer.from_pretrained = original_from_pretrained


class MoondreamBackend:
    def __init__(
        self,
        model_id: str,
        device: str,
        *,
        patched: bool,
        inference_mode: str,
    ):
        from transformers import AutoModelForCausalLM
        import torch

        self.model_id = model_id
        self.device = resolve_torch_device(device)
        self.patched = patched
        self.inference_mode = inference_mode
        if patched:
            ensure_moondream_py39_dynamic_module(model_id)
        if self.device == "cpu":
            dtype = torch.float32
        elif self.device == "mps":
            dtype = torch.bfloat16
        else:
            dtype = torch.bfloat16

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if patched:
            model_kwargs["local_files_only"] = True

        if patched:
            with moondream_local_tokenizer_patch(model_id):
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **model_kwargs,
                ).to(self.device).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs,
            ).to(self.device).eval()
        patch_moondream_device_mixing(self.model)

    def caption(self, image_path: Path) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        if self.inference_mode == "query":
            result = self.model.query(
                image,
                MOONDREAM_SALIENCE_PROMPT,
                settings=DEFAULT_MOONDREAM_QUERY_SETTINGS,
            )
            return str(result.get("answer", "")).strip()

        caption_length = self.inference_mode.removeprefix("caption-")
        result = self.model.caption(
            image,
            length=caption_length,
            settings=DEFAULT_MOONDREAM_CAPTION_SETTINGS,
        )
        return str(result.get("caption", "")).strip()


def configure_hf_runtime(offline: bool) -> None:
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def patch_moondream_device_mixing(model: Any) -> None:
    import importlib
    import torch.nn.functional as F

    module_parts = model.__class__.__module__.split(".")
    if len(module_parts) < 2:
        return

    vision_module_name = ".".join(module_parts[:-1] + ["vision"])
    vision_module = importlib.import_module(vision_module_name)

    def safe_adaptive_avg_pool2d(input_tensor, output_size):
        if input_tensor.device.type == "mps":
            return F.adaptive_avg_pool2d(input_tensor.to("cpu"), output_size).to(input_tensor.device)
        return F.adaptive_avg_pool2d(input_tensor, output_size)

    vision_module.adaptive_avg_pool2d = safe_adaptive_avg_pool2d


def build_backend(args: argparse.Namespace):
    model_id = args.model_id or DEFAULT_MODEL_IDS[args.backend]
    if args.backend == "florence-local":
        return FlorenceLocalBackend(
            model_id=model_id,
            device=args.device,
            task_name=args.florence_task,
        )
    if args.backend == "blip-local":
        return BlipLocalBackend(model_id=model_id, device=args.device)
    if args.backend == "clipcap-local":
        return ClipCapBackend(
            model_id=model_id,
            device=args.device,
            local_only=args.offline,
            prefix_length=args.clipcap_prefix_length,
            beam_size=args.clipcap_beam_size,
            entry_length=args.clipcap_entry_length,
            clip_model_id=args.clipcap_clip_model_id,
            gpt_model_id=args.clipcap_gpt_model_id,
        )
    if args.backend == "paligemma-local":
        return PaliGemmaBackend(
            model_id=model_id,
            device=args.device,
            local_only=args.offline,
            inference_mode=args.paligemma_mode,
            max_new_tokens=args.paligemma_max_new_tokens,
            prompt_override=args.paligemma_prompt,
        )
    if args.backend == "smolvlm-local":
        return SmolVLMBackend(model_id=model_id, device=args.device)
    if args.backend == "moondream-local":
        return MoondreamBackend(
            model_id=model_id,
            device=args.device,
            patched=args.moondream_loader == "patched",
            inference_mode=args.moondream_mode,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def evaluate_caption(caption: str, image_spec: dict[str, Any]) -> dict[str, Any]:
    normalized_caption = normalize_text(caption)
    dimensions = []
    total_weight = 0.0
    matched_weight = 0.0
    contradicted_weight = 0.0
    ambiguous_weight = 0.0

    for dim in image_spec["dimensions"]:
        importance = dim["importance"]
        policy = dim.get("policy", "must_mention")
        weight = IMPORTANCE_WEIGHTS[importance]
        total_weight += weight

        positive_hits = [
            phrase for phrase in dim.get("positives", []) if phrase_matches(normalized_caption, phrase)
        ]
        negative_hits = [
            phrase for phrase in dim.get("negatives", []) if phrase_matches(normalized_caption, phrase)
        ]

        if policy == "avoid_wrong":
            if negative_hits:
                status = "contradicted"
                score = -weight
                contradicted_weight += weight
            else:
                status = "ok"
                score = 0.0
        elif positive_hits and negative_hits:
            status = "ambiguous"
            score = -0.5 * weight
            ambiguous_weight += weight
        elif positive_hits:
            status = "matched"
            score = weight
            matched_weight += weight
        elif negative_hits:
            status = "contradicted"
            score = -weight
            contradicted_weight += weight
        else:
            status = "missing"
            score = 0.0

        dimensions.append(
            {
                "name": dim["name"],
                "importance": importance,
                "policy": policy,
                "expected": dim["expected"],
                "status": status,
                "score": score,
                "positive_hits": positive_hits,
                "negative_hits": negative_hits,
            }
        )

    raw_score = matched_weight - contradicted_weight - (0.5 * ambiguous_weight)
    final_score = max(0.0, raw_score / total_weight) if total_weight else 0.0
    critical_total = sum(
        1
        for dim in dimensions
        if dim["importance"] == "critical" and dim["policy"] != "avoid_wrong"
    )
    critical_matched = sum(
        1
        for dim in dimensions
        if dim["importance"] == "critical"
        and dim["policy"] != "avoid_wrong"
        and dim["status"] == "matched"
    )
    critical_problem_count = sum(
        1
        for dim in dimensions
        if dim["importance"] == "critical" and dim["status"] in {"contradicted", "ambiguous"}
    )

    return {
        "caption": caption,
        "word_count": word_count(caption),
        "score": final_score,
        "matched_weight": matched_weight,
        "contradicted_weight": contradicted_weight,
        "ambiguous_weight": ambiguous_weight,
        "signal_density": matched_weight / max(word_count(caption), 1),
        "critical_total": critical_total,
        "critical_matched": critical_matched,
        "critical_problem_count": critical_problem_count,
        "dimensions": dimensions,
    }


def evaluate_predictions(
    image_specs: list[dict[str, Any]],
    predictions: dict[str, str],
) -> dict[str, Any]:
    results = {
        image_spec["filename"]: evaluate_caption(predictions[image_spec["filename"]], image_spec)
        for image_spec in image_specs
    }
    image_count = len(image_specs)
    average_score = sum(result["score"] for result in results.values()) / image_count if image_count else 0.0
    total_critical = sum(result["critical_total"] for result in results.values())
    matched_critical = sum(result["critical_matched"] for result in results.values())
    problem_critical = sum(result["critical_problem_count"] for result in results.values())
    average_words = sum(result["word_count"] for result in results.values()) / image_count if image_count else 0.0
    average_density = sum(result["signal_density"] for result in results.values()) / image_count if image_count else 0.0

    critical_gap_counts: dict[str, int] = {}
    for result in results.values():
        for dim in result["dimensions"]:
            if dim["importance"] != "critical":
                continue
            if dim["status"] in {"missing", "contradicted", "ambiguous"}:
                critical_gap_counts[dim["name"]] = critical_gap_counts.get(dim["name"], 0) + 1

    top_critical_gaps = sorted(
        critical_gap_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:5]

    return {
        "results": results,
        "average_score": average_score,
        "total_critical": total_critical,
        "matched_critical": matched_critical,
        "problem_critical": problem_critical,
        "average_words": average_words,
        "average_density": average_density,
        "top_critical_gaps": top_critical_gaps,
    }


def print_report(
    model_name: str,
    image_specs: list[dict[str, Any]],
    predictions: dict[str, str],
    evaluation: dict[str, Any],
) -> None:
    results = evaluation["results"]

    print(f"Model: {model_name}")
    print(f"Images: {len(image_specs)}")
    print(f"What matters score: {evaluation['average_score']:.3f}")
    print(f"Critical coverage: {evaluation['matched_critical']}/{evaluation['total_critical']}")
    print(f"Critical problems: {evaluation['problem_critical']}")
    print(f"Average words: {evaluation['average_words']:.1f}")
    print(f"Signal density: {evaluation['average_density']:.3f}")
    if evaluation["top_critical_gaps"]:
        gap_bits = ", ".join(f"{name}({count})" for name, count in evaluation["top_critical_gaps"])
        print(f"Top critical gaps: {gap_bits}")
    print()

    for image_spec in image_specs:
        filename = image_spec["filename"]
        result = results[filename]
        print(f"[{filename}] score={result['score']:.3f}")
        print(f"Caption: {predictions[filename]}")

        matched = [d for d in result["dimensions"] if d["status"] == "matched"]
        missing = [d for d in result["dimensions"] if d["status"] == "missing"]
        contradicted = [d for d in result["dimensions"] if d["status"] == "contradicted"]
        ambiguous = [d for d in result["dimensions"] if d["status"] == "ambiguous"]

        if matched:
            matched_bits = ", ".join(
                f"{d['name']} ({'/'.join(d['positive_hits'])})" for d in matched
            )
            print(f"Matched: {matched_bits}")
        if missing:
            missing_bits = ", ".join(
                f"{d['name']}[{d['importance']}]" for d in missing
            )
            print(f"Missing: {missing_bits}")
        if contradicted:
            contradicted_bits = ", ".join(
                f"{d['name']} ({'/'.join(d['negative_hits'])})" for d in contradicted
            )
            print(f"Contradicted: {contradicted_bits}")
        if ambiguous:
            ambiguous_bits = ", ".join(
                f"{d['name']} (+{'/'.join(d['positive_hits'])}; -{'/'.join(d['negative_hits'])})"
                for d in ambiguous
            )
            print(f"Ambiguous: {ambiguous_bits}")
        print()


def print_leaderboard(leaderboard: list[dict[str, Any]]) -> None:
    print("What Matters Leaderboard")
    print("Model\tScore\tCritical\tProblems\tAvgWords\tDensity\tTopCriticalGaps")
    for item in sorted(
        leaderboard,
        key=lambda row: (-row["average_score"], -row["matched_critical"], row["average_words"]),
    ):
        gaps = ",".join(f"{name}:{count}" for name, count in item["top_critical_gaps"]) or "-"
        critical = f"{item['matched_critical']}/{item['total_critical']}"
        print(
            f"{item['model_name']}\t"
            f"{item['average_score']:.3f}\t"
            f"{critical}\t"
            f"{item['problem_critical']}\t"
            f"{item['average_words']:.1f}\t"
            f"{item['average_density']:.3f}\t"
            f"{gaps}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate caption outputs against hand-labeled dimensions.")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("benchmarks/image_caption_ground_truth.json"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("test_resources"),
    )
    parser.add_argument(
        "--backend",
        choices=["florence-local", "blip-local", "clipcap-local", "paligemma-local", "smolvlm-local", "moondream-local"],
        help="Generate captions with a local backend instead of loading a predictions file.",
    )
    parser.add_argument(
        "--model-id",
        help="Optional model id override for the selected backend.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device string for the selected backend.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force Hugging Face offline mode for locally cached model assets.",
    )
    parser.add_argument(
        "--florence-task",
        choices=list(FLORENCE_TASK_TOKENS.keys()),
        default=DEFAULT_FLORENCE_TASK,
        help="Florence caption task token to benchmark.",
    )
    parser.add_argument(
        "--moondream-loader",
        choices=["official", "patched"],
        default="official",
        help="Use the official Moondream loading path or the local patched offline loader.",
    )
    parser.add_argument(
        "--moondream-mode",
        choices=["query", "caption-short", "caption-normal", "caption-long"],
        default="query",
        help="Which Moondream API to benchmark.",
    )
    parser.add_argument(
        "--clipcap-prefix-length",
        type=int,
        default=10,
        help="ClipCap prefix length used by the checkpoint.",
    )
    parser.add_argument(
        "--clipcap-beam-size",
        type=int,
        default=5,
        help="Beam size for ClipCap decoding.",
    )
    parser.add_argument(
        "--clipcap-entry-length",
        type=int,
        default=67,
        help="Maximum generated token count for ClipCap decoding.",
    )
    parser.add_argument(
        "--clipcap-clip-model-id",
        default=DEFAULT_CLIPCAP_CLIP_MODEL_ID,
        help="CLIP backbone repo id used by ClipCap.",
    )
    parser.add_argument(
        "--clipcap-gpt-model-id",
        default=DEFAULT_CLIPCAP_GPT_MODEL_ID,
        help="GPT decoder repo id used by ClipCap.",
    )
    parser.add_argument(
        "--paligemma-mode",
        choices=["caption", "query"],
        default="caption",
        help="Which PaliGemma prompt style to benchmark.",
    )
    parser.add_argument(
        "--paligemma-max-new-tokens",
        type=int,
        default=DEFAULT_PALIGEMMA_MAX_NEW_TOKENS,
        help="Maximum generated token count for PaliGemma decoding.",
    )
    parser.add_argument(
        "--paligemma-prompt",
        help="Optional prompt override for PaliGemma.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        help="Existing predictions JSON file to evaluate.",
    )
    parser.add_argument(
        "--write-predictions",
        type=Path,
        help="Optional output file for generated predictions.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a generated backend run from an existing predictions file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of images to evaluate.",
    )
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="When evaluating multiple prediction files, print a comparison leaderboard.",
    )
    args = parser.parse_args()

    if not args.backend and not args.predictions:
        parser.error("Provide either --backend or --predictions.")

    configure_hf_runtime(args.offline)

    ground_truth = load_ground_truth(args.ground_truth)
    image_specs = ground_truth["images"]
    if args.limit:
        image_specs = image_specs[: args.limit]

    if args.predictions:
        if len(args.predictions) > 1 or args.leaderboard:
            leaderboard = []
            for path in args.predictions:
                predictions = load_predictions(path)
                missing_predictions = [
                    image_spec["filename"]
                    for image_spec in image_specs
                    if image_spec["filename"] not in predictions
                ]
                if missing_predictions:
                    print(
                        f"Missing predictions for {path}: {', '.join(missing_predictions)}",
                        file=sys.stderr,
                    )
                    return 1
                evaluation = evaluate_predictions(image_specs, predictions)
                leaderboard.append(
                    {
                        "model_name": path.stem,
                        **{k: v for k, v in evaluation.items() if k != "results"},
                    }
                )
            print_leaderboard(leaderboard)
            return 0

        predictions = load_predictions(args.predictions[0])
        model_name = args.predictions[0].stem
    else:
        backend = build_backend(args)
        model_id = args.model_id or DEFAULT_MODEL_IDS[args.backend]
        model_name = f"{args.backend}:{model_id}"
        if args.backend == "florence-local":
            model_name = f"{model_name}:{args.florence_task}"
        elif args.backend == "moondream-local":
            model_name = f"{model_name}:{args.moondream_loader}:{args.moondream_mode}"
        elif args.backend == "paligemma-local":
            model_name = f"{model_name}:{args.paligemma_mode}"
        predictions = {}
        if args.resume and args.write_predictions and args.write_predictions.exists():
            predictions = load_predictions(args.write_predictions)
        for index, image_spec in enumerate(image_specs, start=1):
            image_path = args.image_dir / image_spec["filename"]
            if image_spec["filename"] in predictions:
                print(
                    f"[{index}/{len(image_specs)}] skipping {image_spec['filename']} (already present)",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            print(
                f"[{index}/{len(image_specs)}] captioning {image_spec['filename']}...",
                file=sys.stderr,
                flush=True,
            )
            predictions[image_spec["filename"]] = backend.caption(image_path)
            if args.write_predictions:
                write_predictions(args.write_predictions, model_name, predictions)

    missing_predictions = [
        image_spec["filename"] for image_spec in image_specs if image_spec["filename"] not in predictions
    ]
    if missing_predictions:
        print("Missing predictions for:", ", ".join(missing_predictions), file=sys.stderr)
        return 1

    evaluation = evaluate_predictions(image_specs, predictions)
    print_report(model_name, image_specs, predictions, evaluation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
