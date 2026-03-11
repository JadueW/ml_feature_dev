from __future__ import annotations

import numpy as np

from src.core import DatasetBundle


class DataChecker:
    @staticmethod
    def check_inputs(X: np.ndarray, y: np.ndarray, expected_n_features: int | None = None) -> None:
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must both be numpy arrays")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got {y.ndim}D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if expected_n_features is not None and X.shape[1] != expected_n_features:
            raise ValueError(f"Expected {expected_n_features} features, got {X.shape[1]}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf")
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or inf")

        classes = np.unique(y)
        if not set(classes).issubset({0, 1}):
            raise ValueError(f"y must be binary 0/1, got {classes}")

    @staticmethod
    def check_feature_bundle(feature_bundle: DatasetBundle | dict) -> None:
        if not isinstance(feature_bundle, DatasetBundle):
            feature_bundle = DatasetBundle.from_serialized(feature_bundle)

        layout = feature_bundle.get_feature_layout()
        expected_n_features = layout.n_features if layout is not None else None

        for class_id in feature_bundle.class_ids:
            X = feature_bundle.get_class_data(class_id)
            y = np.full(X.shape[0], class_id, dtype=int)
            DataChecker.check_inputs(X, y, expected_n_features=expected_n_features)
