from __future__ import annotations

from pathlib import Path

import joblib

from src.core import DatasetBundle, PipelineConfig
from src.models.ml_feature_model import FeatureModel, SplitData, TrainingResult

from .data_io import ensure_output_dir
from .reporting import run_visual_report, write_training_report_files


def train_feature_decoder(
    feature_bundle: DatasetBundle | dict,
    config: PipelineConfig | None = None,
    model_output_path: str | Path | None = None,
) -> tuple[TrainingResult, dict[int, TrainingResult], SplitData]:
    config = config or PipelineConfig()
    model = FeatureModel(feature_bundle)
    split = model.create_split(config.split)
    best_result, all_results = model.train_eval_splits(
        split.X_train,
        split.y_train,
        split.X_test,
        split.y_test,
        n_splits=config.model.repeat_splits,
        model_config=config.model,
    )

    if model_output_path is None:
        model_output_path = ensure_output_dir(config) / config.output.model_filename
    else:
        model_output_path = Path(model_output_path)
        model_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_result.best_model, model_output_path)
    return best_result, all_results, split


def run_training_pipeline(
    feature_bundle: DatasetBundle,
    config: PipelineConfig | None = None,
) -> tuple[TrainingResult, dict[int, TrainingResult], SplitData]:
    config = config or PipelineConfig()
    best_result, all_results, split = train_feature_decoder(feature_bundle, config=config)
    write_training_report_files(best_result, config)
    run_visual_report(best_result, feature_bundle, split, config=config)
    return best_result, all_results, split
