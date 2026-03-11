from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


DEFAULT_FEATURE_NAMES = ("lmp", "beta", "high_gamma")
DEFAULT_FEATURE_TYPES = ("feature",)


@dataclass(frozen=True)
class FeatureLayout:
    n_channels: int
    band_names: tuple[str, ...]
    feature_types: tuple[str, ...] = DEFAULT_FEATURE_TYPES

    @property
    def n_bands(self) -> int:
        return len(self.band_names)

    @property
    def n_types(self) -> int:
        return len(self.feature_types)

    @property
    def n_features(self) -> int:
        return self.n_channels * self.n_bands * self.n_types

    def to_metadata(self) -> dict[str, Any]:
        return {
            "n_channels": self.n_channels,
            "band_names": list(self.band_names),
            "feature_types": list(self.feature_types),
        }

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any] | None) -> "FeatureLayout | None":
        if not payload:
            return None
        return cls(
            n_channels=int(payload["n_channels"]),
            band_names=tuple(payload["band_names"]),
            feature_types=tuple(payload.get("feature_types", DEFAULT_FEATURE_TYPES)),
        )


@dataclass(frozen=True)
class DatasetBundle:
    datasets: dict[int, np.ndarray]
    label_mapping: dict[int, str]
    fs: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = {int(class_id): np.asarray(values) for class_id, values in self.datasets.items()}
        object.__setattr__(self, "datasets", normalized)
        object.__setattr__(self, "label_mapping", {int(class_id): name for class_id, name in self.label_mapping.items()})
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_serialized(cls, payload: Mapping[str, Any]) -> "DatasetBundle":
        required_keys = {"datasets", "label_mapping", "fs"}
        missing = required_keys.difference(payload)
        if missing:
            raise KeyError(f"Serialized dataset is missing keys: {sorted(missing)}")

        return cls(
            datasets={int(class_id): np.asarray(values) for class_id, values in payload["datasets"].items()},
            label_mapping={int(class_id): str(name) for class_id, name in payload["label_mapping"].items()},
            fs=float(payload["fs"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_serialized(self) -> dict[str, Any]:
        return {
            "datasets": self.datasets,
            "label_mapping": self.label_mapping,
            "fs": self.fs,
            "metadata": self.metadata,
        }

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.datasets))

    @property
    def n_classes(self) -> int:
        return len(self.datasets)

    def get_class_counts(self) -> dict[int, int]:
        return {class_id: int(self.datasets[class_id].shape[0]) for class_id in self.class_ids}

    def get_class_data(self, class_id: int) -> np.ndarray:
        return self.datasets[int(class_id)]

    def replace_datasets(self, datasets: Mapping[int, np.ndarray], metadata: Mapping[str, Any] | None = None) -> "DatasetBundle":
        next_metadata = dict(self.metadata)
        if metadata:
            next_metadata.update(metadata)
        return DatasetBundle(
            datasets={int(class_id): np.asarray(values) for class_id, values in datasets.items()},
            label_mapping=self.label_mapping,
            fs=self.fs,
            metadata=next_metadata,
        )

    def stack(self) -> tuple[np.ndarray, np.ndarray]:
        features = []
        labels = []
        for class_id in self.class_ids:
            class_data = self.datasets[class_id]
            features.append(class_data)
            labels.append(np.full(class_data.shape[0], class_id, dtype=int))
        return np.vstack(features), np.concatenate(labels)

    def get_feature_layout(self) -> FeatureLayout | None:
        layout = FeatureLayout.from_metadata(self.metadata.get("feature_layout"))
        if layout is not None:
            return layout

        if not self.datasets:
            return None

        sample = self.datasets[self.class_ids[0]]
        if sample.ndim != 2:
            return None

        inferred_bands = len(DEFAULT_FEATURE_NAMES)
        inferred_types = len(DEFAULT_FEATURE_TYPES)
        if sample.shape[1] % (inferred_bands * inferred_types) != 0:
            return None

        return FeatureLayout(
            n_channels=sample.shape[1] // (inferred_bands * inferred_types),
            band_names=DEFAULT_FEATURE_NAMES,
            feature_types=DEFAULT_FEATURE_TYPES,
        )


def reconstruct_binary_task_dataset(
    raw_data: Mapping[str, Any],
    *,
    rest_index: int = 0,
    n_channels: int = 128,
    negative_label: str = "non_task",
    positive_label: str = "task",
) -> DatasetBundle:
    raw_datasets = raw_data["datasets"]
    if len(raw_datasets) < 2:
        raise ValueError("Expected at least 2 class segments in raw_data['datasets']")

    if isinstance(raw_datasets, Mapping):
        dataset_items = [(int(class_id), np.asarray(segment)) for class_id, segment in raw_datasets.items()]
    else:
        dataset_items = [(idx, np.asarray(segment)) for idx, segment in enumerate(raw_datasets)]

    dataset_items.sort(key=lambda item: item[0])
    available_class_ids = [class_id for class_id, _ in dataset_items]
    if rest_index not in available_class_ids:
        raise KeyError(f"rest_index={rest_index} not found in datasets keys: {available_class_ids}")

    normalized_segments = {}
    for class_id, segment in dataset_items:
        if segment.ndim != 3:
            raise ValueError(
                f"Expected class {class_id} segment to be 3D (samples, channels, timepoints), got {segment.shape}"
            )
        normalized_segments[class_id] = segment[:, :n_channels, :]

    rest_data = normalized_segments[rest_index]
    task_segments = [segment for class_id, segment in normalized_segments.items() if class_id != rest_index]
    if not task_segments:
        raise ValueError("No task segments found after removing rest class")
    task_data = np.concatenate(task_segments, axis=0)

    return DatasetBundle(
        datasets={0: rest_data, 1: task_data},
        label_mapping={0: negative_label, 1: positive_label},
        fs=float(raw_data["fs"]),
        metadata={"source": "reconstructed_binary_task_dataset"},
    )


def merge_dataset_bundles(
    bundles: list[DatasetBundle],
    metadata: Mapping[str, Any] | None = None,
) -> DatasetBundle:
    if not bundles:
        raise ValueError("bundles must not be empty")

    first = bundles[0]
    merged = {}
    for class_id in first.class_ids:
        merged[class_id] = np.vstack([bundle.get_class_data(class_id) for bundle in bundles])

    merged_metadata = dict(first.metadata)
    if metadata:
        merged_metadata.update(metadata)

    return DatasetBundle(
        datasets=merged,
        label_mapping=first.label_mapping,
        fs=first.fs,
        metadata=merged_metadata,
    )
