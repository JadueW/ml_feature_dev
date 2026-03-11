from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.core import DatasetBundle, PipelineConfig


def load_dataset_bundle(path: str | Path) -> DatasetBundle:
    return DatasetBundle.from_serialized(joblib.load(Path(path)))


def save_dataset_bundle(bundle: DatasetBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.to_serialized(), path)


def ensure_output_dir(config: PipelineConfig) -> Path:
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_plot_path(config: PipelineConfig, stem: str) -> Path:
    return ensure_output_dir(config) / f"{stem}.{config.output.plot_format}"


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
