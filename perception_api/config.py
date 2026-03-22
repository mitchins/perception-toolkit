"""
Perception sidecar configuration.

Loads backend configuration from config.yaml with environment variable overrides.
All backend settings (models, thresholds, devices) live here — not in Open WebUI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FlorenceConfig:
    enabled: bool = True
    model_id: str = "microsoft/Florence-2-base"
    device: str = "auto"


@dataclass
class WD14Config:
    enabled: bool = False
    model_id: str = "SmilingWolf/wd-v1-4-convnextv2"
    threshold_default: float = 0.35
    device: str = "auto"


@dataclass
class ClassifierConfig:
    enabled: bool = False
    model_path: str = "/models/custom_classifier.onnx"
    threshold_default: float = 0.60
    device: str = "auto"


@dataclass
class SandboxConfig:
    base_path: str = "/tmp/perception"
    ttl_seconds: int = 3600


@dataclass
class PerceptionConfig:
    florence: FlorenceConfig = field(default_factory=FlorenceConfig)
    wd14: WD14Config = field(default_factory=WD14Config)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    host: str = "0.0.0.0"
    port: int = 8200


def _merge_env_overrides(cfg: PerceptionConfig) -> None:
    """Apply environment variable overrides on top of file config."""
    if v := os.environ.get("PERCEPTION_FLORENCE_ENABLED"):
        cfg.florence.enabled = v.lower() in ("1", "true", "yes")
    if v := os.environ.get("PERCEPTION_FLORENCE_MODEL_ID"):
        cfg.florence.model_id = v
    if v := os.environ.get("PERCEPTION_FLORENCE_DEVICE"):
        cfg.florence.device = v

    if v := os.environ.get("PERCEPTION_WD14_ENABLED"):
        cfg.wd14.enabled = v.lower() in ("1", "true", "yes")
    if v := os.environ.get("PERCEPTION_WD14_MODEL_ID"):
        cfg.wd14.model_id = v
    if v := os.environ.get("PERCEPTION_WD14_THRESHOLD"):
        cfg.wd14.threshold_default = float(v)
    if v := os.environ.get("PERCEPTION_WD14_DEVICE"):
        cfg.wd14.device = v

    if v := os.environ.get("PERCEPTION_CLASSIFIER_ENABLED"):
        cfg.classifier.enabled = v.lower() in ("1", "true", "yes")
    if v := os.environ.get("PERCEPTION_CLASSIFIER_MODEL_PATH"):
        cfg.classifier.model_path = v
    if v := os.environ.get("PERCEPTION_CLASSIFIER_THRESHOLD"):
        cfg.classifier.threshold_default = float(v)

    if v := os.environ.get("PERCEPTION_SANDBOX_BASE"):
        cfg.sandbox.base_path = v
    if v := os.environ.get("PERCEPTION_SANDBOX_TTL"):
        cfg.sandbox.ttl_seconds = int(v)

    if v := os.environ.get("PERCEPTION_HOST"):
        cfg.host = v
    if v := os.environ.get("PERCEPTION_PORT"):
        cfg.port = int(v)


def _dict_to_backend_config(section: str, data: dict[str, Any]) -> Any:
    """Convert a raw dict from YAML into the appropriate backend config dataclass."""
    mapping = {
        "florence": FlorenceConfig,
        "wd14": WD14Config,
        "classifier": ClassifierConfig,
    }
    cls = mapping.get(section)
    if cls is None:
        return None
    # Only pass keys the dataclass accepts
    valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def load_config(config_path: str | Path | None = None) -> PerceptionConfig:
    """
    Load configuration from YAML file, then apply env overrides.

    Resolution order:
      1. Defaults in dataclass
      2. config.yaml values
      3. Environment variable overrides
    """
    cfg = PerceptionConfig()

    if config_path is None:
        # Look for config.yaml in the project root (one level up from this file)
        candidates = [
            Path(__file__).resolve().parent.parent / "config.yaml",
            Path("/app/config.yaml"),
            Path("config.yaml"),
        ]
        for c in candidates:
            if c.is_file():
                config_path = c
                break

    if config_path is not None:
        path = Path(config_path)
        if path.is_file():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}

            backends = raw.get("backends", {})
            for section_name, section_data in backends.items():
                if not isinstance(section_data, dict):
                    continue
                obj = _dict_to_backend_config(section_name, section_data)
                if obj is not None:
                    setattr(cfg, section_name, obj)

            sandbox_data = raw.get("sandbox", {})
            if isinstance(sandbox_data, dict):
                if "base_path" in sandbox_data:
                    cfg.sandbox.base_path = sandbox_data["base_path"]
                if "ttl_seconds" in sandbox_data:
                    cfg.sandbox.ttl_seconds = int(sandbox_data["ttl_seconds"])

            if "host" in raw:
                cfg.host = raw["host"]
            if "port" in raw:
                cfg.port = int(raw["port"])

    _merge_env_overrides(cfg)
    return cfg


# Module-level singleton — loaded once at import time, can be reloaded.
_config: PerceptionConfig | None = None


def get_config() -> PerceptionConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: str | Path | None = None) -> PerceptionConfig:
    global _config
    _config = load_config(config_path)
    return _config
