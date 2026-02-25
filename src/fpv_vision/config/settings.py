"""
Pydantic-based settings with YAML + dotenv support.

Priority (highest → lowest):
  1. Environment variables / .env file
  2. configs/default.yaml
  3. Hard-coded defaults below
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_CONFIG_FILE = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


def _load_yaml_defaults() -> dict:
    if _CONFIG_FILE.exists():
        with _CONFIG_FILE.open() as f:
            return yaml.safe_load(f) or {}
    return {}


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_")

    weights: str = "yolo11n.pt"
    device: str = ""
    confidence: float = 0.5
    iou_threshold: float = 0.45
    imgsz: int = 640


class TrainingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRAIN_")

    epochs: int = 100
    batch_size: int = 16
    lr0: float = 0.01
    optimizer: str = "auto"
    patience: int = 50
    save_period: int = 10
    project: str = "runs/train"
    name: str = "fpv_v1"
    pretrained: bool = True
    augment: bool = True


class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATA_")

    dir: str = "data/fpv"
    yaml: str = "data/fpv/data.yaml"


class InferenceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERENCE_")

    source: str = "0"
    show: bool = True
    save: bool = False
    save_dir: str = "runs/detect"


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    data: DataSettings = DataSettings()
    inference: InferenceSettings = InferenceSettings()

    classes: list[str] = [
        "motor",
        "flight_controller",
        "esc",
        "camera",
        "vtx",
        "receiver",
        "propeller",
        "lipo_battery",
        "xt60_connector",
        "capacitor",
    ]

    @field_validator("classes", mode="before")
    @classmethod
    def _load_classes_from_yaml(cls, v: list[str]) -> list[str]:
        """Allow classes to be overridden via the YAML config file."""
        raw = _load_yaml_defaults()
        return raw.get("classes", v)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
