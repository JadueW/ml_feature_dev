from __future__ import annotations

import numpy as np
import shap

from src.core import DatasetBundle, FeatureLayout
from src.visualize.visualizer import Visualizer

try:
    from shap import Explainer, KernelExplainer, LinearExplainer
except ImportError:
    Explainer = shap.Explainer
    KernelExplainer = shap.KernelExplainer
    LinearExplainer = shap.LinearExplainer


class ShapAnalyzer:
    DEFAULT_BAND_NAMES = ("lmp", "beta", "high_gamma")

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_layout: FeatureLayout | None = None,
        n_channels: int = 128,
        n_bands: int = 3,
        n_types: int = 1,
        band_names: tuple[str, ...] | None = None,
        feature_types: tuple[str, ...] | None = None,
    ):
        if feature_layout is None:
            band_names = tuple(band_names or self.DEFAULT_BAND_NAMES[:n_bands])
            feature_types = tuple(feature_types or ("feature",))
            feature_layout = FeatureLayout(
                n_channels=n_channels,
                band_names=band_names,
                feature_types=feature_types,
            )

        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.layout = feature_layout
        self.n_channels = feature_layout.n_channels
        self.n_bands = feature_layout.n_bands
        self.n_types = feature_layout.n_types
        self.band_names = list(feature_layout.band_names)
        self.feature_types = list(feature_layout.feature_types)
        self.n_features = feature_layout.n_features
        self.feature_names = self._build_feature_names()
        self.shap_values_train = None
        self.shap_values_test = None
        self.explainer = None

        if X_train.shape[1] != self.n_features:
            raise ValueError(f"X_train feature count mismatch: got {X_train.shape[1]}, expected {self.n_features}")
        if X_test.shape[1] != self.n_features:
            raise ValueError(f"X_test feature count mismatch: got {X_test.shape[1]}, expected {self.n_features}")

    @classmethod
    def from_feature_bundle(cls, model, X_train: np.ndarray, X_test: np.ndarray, feature_bundle: DatasetBundle):
        layout = feature_bundle.get_feature_layout()
        if layout is None:
            raise ValueError("feature_bundle metadata does not contain feature_layout")
        return cls(model, X_train, X_test, feature_layout=layout)

    def _build_feature_names(self) -> list[str]:
        feature_names = []
        for feature_type in self.feature_types:
            for channel_idx in range(self.n_channels):
                for band_name in self.band_names:
                    feature_names.append(f"ch{channel_idx}_{feature_type}_{band_name}")
        return feature_names

    def _make_explainer(self, background: np.ndarray, use_kernel: bool):
        if use_kernel:
            return KernelExplainer(self.model.predict_proba, background)
        if hasattr(self.model, "named_steps") and hasattr(self.model, "predict_proba"):
            return Explainer(self.model.predict_proba, background)
        try:
            return LinearExplainer(self.model, background)
        except Exception:
            return Explainer(self.model.predict_proba, background)

    def compute_shap(self, background_size: int = 100, use_kernel: bool = False):
        if len(self.X_train) > background_size:
            indices = np.random.choice(len(self.X_train), background_size, replace=False)
            background = self.X_train[indices]
        else:
            background = self.X_train

        self.explainer = self._make_explainer(background, use_kernel=use_kernel)
        try:
            self.shap_values_train = self.explainer.shap_values(self.X_train)
        except Exception:
            self.shap_values_train = self.explainer(self.X_train)

        try:
            self.shap_values_test = self.explainer.shap_values(self.X_test)
        except Exception:
            self.shap_values_test = self.explainer(self.X_test)
        return self

    @staticmethod
    def _extract_shap_array(shap_values):
        values = shap_values.values if hasattr(shap_values, "values") else shap_values
        if values.ndim == 3:
            values = values[:, :, 1]
        return values

    def get_shap_arrays(self, data: str = "test") -> tuple[np.ndarray, np.ndarray]:
        shap_values = self.shap_values_train if data == "train" else self.shap_values_test
        X_values = self.X_train if data == "train" else self.X_test
        return self._extract_shap_array(shap_values), X_values

    def plot_channel_importance(self, data: str = "test", **kwargs):
        shap_arr, _ = self.get_shap_arrays(data)
        return Visualizer.plot_shap_channel_importance(
            shap_arr,
            self.n_channels,
            self.n_bands,
            self.n_types,
            data_name=data,
            **kwargs,
        )

    def plot_band_importance(self, data: str = "test", **kwargs):
        shap_arr, _ = self.get_shap_arrays(data)
        return Visualizer.plot_shap_band_importance(
            shap_arr,
            self.n_channels,
            self.n_bands,
            self.n_types,
            separate_types=False,
            data_name=data,
            band_names=self.band_names,
            **kwargs,
        )

    def plot_summary(self, data: str = "test", max_display: int = 20, **kwargs):
        shap_arr, X_arr = self.get_shap_arrays(data)
        return Visualizer.plot_shap_summary(
            shap_arr,
            X_arr,
            feature_names=self.feature_names,
            max_display=max_display,
            data_name=data,
            **kwargs,
        )
