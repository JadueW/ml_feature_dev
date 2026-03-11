from __future__ import annotations

from pathlib import Path
from src.cross_time import run_cross_time_experiment
from src.core import DatasetBundle, PipelineConfig
from src.models.ml_feature_model import SplitData, TrainingResult
from src.workflows import (
    build_feature_dataset_from_raw,
    load_dataset_bundle,
    print_training_summary,
    run_training_pipeline,
    train_feature_decoder,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PROCESSED_DATA_PATH = DATA_DIR / "processed" / "test.pkl"

__all__ = [
    "DatasetBundle",
    "PipelineConfig",
    "SplitData",
    "TrainingResult",
    "build_feature_dataset_from_raw",
    "load_dataset_bundle",
    "print_training_summary",
    "run_training_pipeline",
    "train_feature_decoder",
    "DEFAULT_PROCESSED_DATA_PATH",
]


def main() -> None:
    config = PipelineConfig()
    feature_bundle = load_dataset_bundle(DEFAULT_PROCESSED_DATA_PATH)
    best_result, _, _ = run_training_pipeline(feature_bundle, config=config)
    print_training_summary(best_result)


if __name__ == "__main__":
    config = PipelineConfig()
    feature_bundle = build_feature_dataset_from_raw(
        "../data/raw/WHTJYY-subdural-fineMovement-day1-1.pkl",
        "../data/processed/day1_1_features.pkl",
        config=config,
    )
    best_result, all_results, split = run_training_pipeline(
        feature_bundle,
        config=config
    )
    print_training_summary(best_result)
    summary = run_cross_time_experiment(
        raw_dir="data/raw",
        config=config,
        group_by="day",
        subject="WHTJYY",
        task_type="fineMovement",
        cache_dir="data/processed/cross_time_cache",
    )
    print(summary)
