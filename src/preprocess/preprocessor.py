from __future__ import annotations

import time

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

from src.core import DatasetBundle, PreprocessConfig


class Preprocessor:
    def __init__(self, dataset_bundle: DatasetBundle | dict):
        if isinstance(dataset_bundle, DatasetBundle):
            self.bundle = dataset_bundle
        else:
            self.bundle = DatasetBundle.from_serialized(dataset_bundle)

        self.fs = self.bundle.fs
        self.label_mapping = self.bundle.label_mapping

    @staticmethod
    def common_average_reference(data: np.ndarray) -> np.ndarray:
        mean_chan = np.mean(data, axis=1, keepdims=True)
        return data - mean_chan

    def _design_notch_sos(self, f0: float, bandwidth: float) -> np.ndarray:
        quality_factor = f0 / bandwidth
        b, a = iirnotch(f0, quality_factor, self.fs)
        return tf2sos(b, a)

    def _design_bandpass_sos(self, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        return butter(order, [low, high], btype="band", output="sos")

    def transform(self, config: PreprocessConfig) -> DatasetBundle:
        start = time.time()

        notch_sos_list = [self._design_notch_sos(freq, config.notch_bandwidth) for freq in config.notch_frequencies]
        total_notch_sos = np.vstack(notch_sos_list) if notch_sos_list else None
        bandpass_sos = self._design_bandpass_sos(config.lowcut, config.highcut, order=config.bandpass_order)
        combined_sos = np.vstack([total_notch_sos, bandpass_sos]) if total_notch_sos is not None else bandpass_sos

        processed = {}
        for class_id in self.bundle.class_ids:
            data = self.bundle.get_class_data(class_id)
            if data.ndim != 3:
                raise ValueError(
                    f"Expected class {class_id} data to be 3D (samples, channels, timepoints), got {data.shape}"
                )
            filtered = sosfiltfilt(combined_sos, data, axis=-1)
            processed[class_id] = self.common_average_reference(filtered)

        elapsed = time.time() - start
        metadata = {
            "preprocess": {
                "notch_frequencies": list(config.notch_frequencies),
                "lowcut": config.lowcut,
                "highcut": config.highcut,
                "notch_bandwidth": config.notch_bandwidth,
                "bandpass_order": config.bandpass_order,
                "elapsed_seconds": round(elapsed, 3),
            }
        }
        return self.bundle.replace_datasets(processed, metadata=metadata)

    def preprocess(
        self,
        f0_list,
        lowcut: float = 1,
        highcut: float = 200,
        bw: float = 2,
        bandpass_order: int = 4,
    ) -> dict[int, np.ndarray]:
        config = PreprocessConfig(
            notch_frequencies=tuple(float(freq) for freq in f0_list),
            lowcut=float(lowcut),
            highcut=float(highcut),
            notch_bandwidth=float(bw),
            bandpass_order=int(bandpass_order),
        )
        return self.transform(config).datasets
