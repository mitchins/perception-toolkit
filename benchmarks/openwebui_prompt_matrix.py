#!/usr/bin/env python3
"""
Run a prompt-shape matrix against a local Open WebUI chat endpoint.

This is meant to benchmark small-model tool routing on image tasks. It sends the
same image attachment with a series of prompt phrasings and records the raw
responses plus a few coarse heuristics such as refusal/tool-echo detection.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "test_resources" / "duolingo_korean_unit.png"
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "benchmarks" / "openwebui_prompt_matrix_prompts.json"

TOOL_ECHO_PATTERN = re.compile(r"\b(?:extract_text|list_attachments|inspect_image|detect_objects)\s*\(")
REFUSAL_PATTERNS = (
    re.compile(r"\bi(?:'m| am)? unable to (?:view|process|access|interpret) images?\b", re.IGNORECASE),
    re.compile(r"\bi cannot (?:view|process|access|interpret) (?:the )?(?:image|attached image)\b", re.IGNORECASE),
    re.compile(r"\bcan't access or interpret the image\b", re.IGNORECASE),
    re.compile(r"\bplease provide (?:the )?text(?:ual)? description\b", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark prompt-shape sensitivity for image tasks through local Open WebUI."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENWEBUI_BASE_URL", "http://localhost:8080"),
        help="Base URL for Open WebUI.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENWEBUI_MODEL", ""),
        help="Model ID to target. Can also be set via OPENWEBUI_MODEL.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Local image file to attach to every prompt.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="JSON file containing prompt specs.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("OPENWEBUI_TOKEN", ""),
        help="Bearer token or API key. Can also be set via OPENWEBUI_TOKEN.",
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("OPENWEBUI_EMAIL", ""),
        help="Email used to sign in when no token is provided.",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("OPENWEBUI_PASSWORD", ""),
        help="Password used to sign in when no token is provided.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to benchmarks/openwebui_prompt_matrix_<timestamp>.json",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to run each prompt.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models visible to the current user and exit.",
    )
    return parser.parse_args()


def ensure_model_requested(args: argparse.Namespace) -> None:
    if not args.list_models and not args.model:
        raise SystemExit(
            "Model ID is required. Pass --model or set OPENWEBUI_MODEL."
        )


def image_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def load_prompt_specs(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("prompts", [])
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unsupported prompts file shape in {path}")

    prompt_specs: list[dict[str, str]] = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, str):
            prompt_specs.append(
                {
                    "id": f"prompt_{idx}",
                    "label": f"Prompt {idx}",
                    "prompt": item,
                }
            )
            continue

        if not isinstance(item, dict):
            raise ValueError(f"Prompt entry #{idx} must be a string or object")

        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"Prompt entry #{idx} is missing 'prompt'")

        prompt_specs.append(
            {
                "id": str(item.get("id", f"prompt_{idx}")),
                "label": str(item.get("label", item.get("id", f"Prompt {idx}"))),
                "prompt": prompt,
            }
        )

    return prompt_specs


def request_json(
    url: str,
    *,
    method: str,
    timeout: float,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any, str]:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib_request.Request(url, data=body, headers=request_headers, method=method)

    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            status = response.status
    except urllib_error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        status = exc.code
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc

    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = {"raw_text": raw}

    return status, parsed, raw


def authenticate(base_url: str, token: str, email: str, password: str, timeout: float) -> str:
    if token:
        return token

    if not email or not password:
        raise RuntimeError(
            "Authentication required. Pass --token, or set --email/--password (or OPENWEBUI_EMAIL/OPENWEBUI_PASSWORD)."
        )

    status, payload, _ = request_json(
        f"{base_url.rstrip('/')}/api/v1/auths/signin",
        method="POST",
        timeout=timeout,
        payload={"email": email, "password": password},
    )
    if status >= 400:
        raise RuntimeError(f"Signin failed ({status}): {payload}")
    resolved_token = payload.get("token", "")
    if not resolved_token:
        raise RuntimeError("Signin succeeded but no token was returned.")
    return resolved_token


def fetch_models(base_url: str, token: str, timeout: float) -> list[dict[str, Any]]:
    status, payload, _ = request_json(
        f"{base_url.rstrip('/')}/api/models",
        method="GET",
        timeout=timeout,
        headers={"Authorization": f"Bearer {token}"},
    )
    if status >= 400:
        raise RuntimeError(f"/api/models failed ({status}): {payload}")
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise RuntimeError("Unexpected /api/models response shape")


def build_chat_body(model: str, prompt: str, image_data_url: str) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:12]
    return {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "features": {},
        "background_tasks": {},
        "chat_id": f"local:prompt-matrix-{run_id}",
        "session_id": f"prompt-matrix-{run_id}",
    }


def extract_text_content(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue
                if item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(part for part in parts if part)

    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            return value["content"]
        if isinstance(value.get("text"), str):
            return value["text"]

    return ""


def extract_response_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            text = extract_text_content(message)
            if text:
                return text
            return extract_text_content(choice.get("text"))

    if "message" in payload:
        text = extract_text_content(payload.get("message"))
        if text:
            return text

    if "content" in payload:
        text = extract_text_content(payload.get("content"))
        if text:
            return text

    return ""


def classify_response(text: str) -> dict[str, Any]:
    refusal = any(pattern.search(text) for pattern in REFUSAL_PATTERNS)
    tool_echo = bool(TOOL_ECHO_PATTERN.search(text))

    if tool_echo:
        bucket = "tool_echo"
    elif refusal:
        bucket = "image_refusal"
    elif text.strip():
        bucket = "answered"
    else:
        bucket = "empty"

    return {
        "bucket": bucket,
        "refusal": refusal,
        "tool_echo": tool_echo,
    }


def run_prompt(
    *,
    base_url: str,
    token: str,
    model: str,
    prompt_spec: dict[str, str],
    image_data_url: str,
    timeout: float,
) -> dict[str, Any]:
    body = build_chat_body(model=model, prompt=prompt_spec["prompt"], image_data_url=image_data_url)
    started = time.perf_counter()
    status, payload, raw = request_json(
        f"{base_url.rstrip('/')}/api/chat/completions",
        method="POST",
        timeout=timeout,
        headers={"Authorization": f"Bearer {token}"},
        payload=body,
    )
    elapsed = time.perf_counter() - started

    result: dict[str, Any] = {
        "id": prompt_spec["id"],
        "label": prompt_spec["label"],
        "prompt": prompt_spec["prompt"],
        "elapsed_seconds": elapsed,
        "http_status": status,
    }

    result["response"] = payload

    if status >= 400:
        result["assistant_text"] = extract_response_text(payload) or raw
        result["classification"] = {
            "bucket": "http_error",
            "refusal": False,
            "tool_echo": False,
        }
        return result

    assistant_text = extract_response_text(payload)
    result["assistant_text"] = assistant_text
    result["classification"] = classify_response(assistant_text)
    return result


def shorten(text: str, limit: int = 96) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def default_output_path() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "benchmarks" / f"openwebui_prompt_matrix_{timestamp}.json"


def print_summary(results: list[dict[str, Any]]) -> None:
    print("Prompt Matrix Results")
    print("ID\tBucket\tHTTP\tSeconds\tPreview")
    for item in results:
        print(
            f"{item['id']}\t"
            f"{item['classification']['bucket']}\t"
            f"{item['http_status']}\t"
            f"{item['elapsed_seconds']:.2f}\t"
            f"{shorten(item['assistant_text'])}"
        )


def main() -> int:
    args = parse_args()
    ensure_model_requested(args)

    if not args.image.exists():
        raise SystemExit(f"Image file not found: {args.image}")
    if not args.prompts_file.exists():
        raise SystemExit(f"Prompts file not found: {args.prompts_file}")
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    prompt_specs = load_prompt_specs(args.prompts_file)
    image_data_url = image_to_data_url(args.image)
    token = authenticate(args.base_url, args.token, args.email, args.password, args.timeout)

    if args.list_models:
        models = fetch_models(args.base_url, token, args.timeout)
        print(json.dumps(models, indent=2))
        return 0

    results: list[dict[str, Any]] = []
    for iteration in range(args.repeat):
        for prompt_spec in prompt_specs:
            run_result = run_prompt(
                base_url=args.base_url,
                token=token,
                model=args.model,
                prompt_spec=prompt_spec,
                image_data_url=image_data_url,
                timeout=args.timeout,
            )
            run_result["iteration"] = iteration + 1
            results.append(run_result)

    output_path = args.output or default_output_path()
    output_payload = {
        "base_url": args.base_url,
        "model": args.model,
        "image": str(args.image.resolve()),
        "prompts_file": str(args.prompts_file.resolve()),
        "repeat": args.repeat,
        "results": results,
    }
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print_summary(results)
    print(f"\nSaved raw results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
