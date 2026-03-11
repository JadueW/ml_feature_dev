from __future__ import annotations

from pathlib import Path

import joblib

from src.core import DatasetBundle, PipelineConfig, reconstruct_binary_task_dataset
from src.featureExtract.feature_extract import FeatureExtractor
from src.preprocess.preprocessor import Preprocessor

from .data_io import save_dataset_bundle


def build_feature_dataset_from_raw(
    raw_data_path: str | Path,
    output_path: str | Path,
    config: PipelineConfig | None = None,
) -> DatasetBundle:
    config = config or PipelineConfig()
    raw_data = joblib.load(Path(raw_data_path))
    reconstructed = reconstruct_binary_task_dataset(raw_data, n_channels=config.n_channels)
    preprocessed = Preprocessor(reconstructed).transform(config.preprocess)
    feature_bundle, _ = FeatureExtractor.from_config(preprocessed.fs, config.features).transform_bundle(preprocessed)
    save_dataset_bundle(feature_bundle, output_path)
    return feature_bundle
