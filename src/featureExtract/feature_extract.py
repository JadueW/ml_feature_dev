from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt, welch

from src.core import DatasetBundle, FeatureConfig, FeatureLayout


class FeatureExtractor:
    FEATURE_NAMES = ("lmp", "beta", "high_gamma")

    def __init__(
        self,
        fs: float = 2000,
        config: FeatureConfig | None = None,
    ):
        self.config = config or FeatureConfig()
        self.fs = fs
        self.nperseg = self.config.nperseg
        self.noverlap = self.config.noverlap
        self.default_parallel = self.config.parallel
        self.default_n_jobs = self.config.n_jobs
        self.use_log_power = self.config.use_log_power
        self.total_power_range = self.config.total_power_range
        self.lmp_lowpass_cutoff = self.config.lmp_lowpass_cutoff
        self.bands = dict(self.config.bands)
        self.band_names = list(self.FEATURE_NAMES)
        self._freqs = None
        self._band_masks = None
        self._total_mask = None
        self._lmp_ba = None

    @classmethod
    def from_config(cls, fs: float, config: FeatureConfig) -> "FeatureExtractor":
        return cls(fs=fs, config=config)

    def _prepare_filters(self) -> None:
        dummy_data = np.zeros((1, max(self.nperseg * 2, 2048)))
        freqs, _ = welch(dummy_data, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, axis=-1)
        self._freqs = freqs
        self._band_masks = {
            band_name: (freqs >= low) & (freqs <= high)
            for band_name, (low, high) in self.bands.items()
        }
        low, high = self.total_power_range
        self._total_mask = (freqs >= low) & (freqs <= high)
        normal_cutoff = self.lmp_lowpass_cutoff / (0.5 * self.fs)
        self._lmp_ba = butter(4, normal_cutoff, btype="low")

    def _compute_sample_features(self, sample: np.ndarray) -> np.ndarray:
        if self._freqs is None:
            self._prepare_filters()

        b, a = self._lmp_ba
        lmp_signal = filtfilt(b, a, sample, axis=-1)
        lmp_feature = np.mean(lmp_signal, axis=-1)

        _, psd = welch(
            sample,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            axis=-1,
        )
        eps = np.finfo(float).eps
        total_power = np.mean(psd[:, self._total_mask], axis=1)
        total_power = np.maximum(total_power, eps)

        feature_matrix = np.zeros((sample.shape[0], len(self.FEATURE_NAMES)), dtype=float)
        feature_matrix[:, 0] = lmp_feature

        for feature_idx, band_name in enumerate(("beta", "high_gamma"), start=1):
            band_power = np.mean(psd[:, self._band_masks[band_name]], axis=1)
            band_power = np.maximum(band_power, eps)
            feature_matrix[:, feature_idx] = np.log10(band_power) if self.use_log_power else band_power / total_power

        return feature_matrix

    def compute_features(
        self,
        data: np.ndarray,
        parallel: bool | None = None,
        n_jobs: int | None = None,
    ) -> np.ndarray:
        if data.ndim != 3:
            raise ValueError(f"Expected data to be 3D (samples, channels, timepoints), got {data.shape}")

        parallel = self.default_parallel if parallel is None else parallel
        n_jobs = self.default_n_jobs if n_jobs is None else n_jobs
        n_samples = data.shape[0]
        features = np.zeros((n_samples, data.shape[1], len(self.FEATURE_NAMES)), dtype=float)

        if parallel:
            results = Parallel(n_jobs=n_jobs)(delayed(self._compute_sample_features)(data[idx]) for idx in range(n_samples))
            for idx, feature_matrix in enumerate(results):
                features[idx] = feature_matrix
        else:
            for idx in range(n_samples):
                features[idx] = self._compute_sample_features(data[idx])

        return features.reshape(n_samples, -1)

    def transform_bundle(
        self,
        processed_bundle: DatasetBundle | dict,
        parallel: bool | None = None,
        n_jobs: int | None = None,
    ) -> tuple[DatasetBundle, FeatureLayout]:
        if not isinstance(processed_bundle, DatasetBundle):
            processed_bundle = DatasetBundle.from_serialized(processed_bundle)

        feature_datasets = {}
        for class_id in processed_bundle.class_ids:
            feature_datasets[class_id] = self.compute_features(
                processed_bundle.get_class_data(class_id),
                parallel=parallel,
                n_jobs=n_jobs,
            )

        reference_shape = processed_bundle.get_class_data(processed_bundle.class_ids[0]).shape
        layout = FeatureLayout(
            n_channels=reference_shape[1],
            band_names=tuple(self.FEATURE_NAMES),
            feature_types=("feature",),
        )
        metadata = {
            "feature_layout": layout.to_metadata(),
            "feature_extractor": {
                "feature_names": list(self.FEATURE_NAMES),
                "bands": {name: list(bounds) for name, bounds in self.bands.items()},
                "lmp_lowpass_cutoff": self.lmp_lowpass_cutoff,
                "nperseg": self.nperseg,
                "noverlap": self.noverlap,
                "use_log_power": self.use_log_power,
            },
        }
        return processed_bundle.replace_datasets(feature_datasets, metadata=metadata), layout

    def extract_features_and_labels(self, processed_datasets, labels=None):
        feature_dict = {}
        if labels is None:
            labels = list(processed_datasets.keys())
        for class_idx in labels:
            feature_dict[class_idx] = self.compute_features(processed_datasets[class_idx])
        return feature_dict
