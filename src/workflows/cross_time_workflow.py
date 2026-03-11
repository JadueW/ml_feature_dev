from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np

from src.core import DatasetBundle, PipelineConfig, merge_dataset_bundles
from src.models.ml_feature_model import FeatureModel, TrainingResult
from src.visualize.visualizer import Visualizer

from .data_io import ensure_output_dir, load_dataset_bundle, save_json
from .feature_workflow import build_feature_dataset_from_raw


FILENAME_PATTERN = re.compile(
    r"(?P<subject>[^-]+)-(?P<implant>[^-]+)-(?P<task_type>[^-]+)-day(?P<day>\d+)-(?P<recording>\d+)\.pkl$"
)


@dataclass(frozen=True)
class RecordingInfo:
    path: Path
    subject: str
    implant: str
    task_type: str
    day: int
    recording: int

    @property
    def recording_id(self) -> str:
        return f"day{self.day}-rec{self.recording}"

    @property
    def day_key(self) -> str:
        return f"day{self.day}"


@dataclass
class CrossTimeFoldResult:
    holdout_group: str
    train_recordings: list[str]
    test_recordings: list[str]
    training_result: TrainingResult

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "holdout_group": self.holdout_group,
            "train_recordings": self.train_recordings,
            "test_recordings": self.test_recordings,
            "result": self.training_result.to_serializable_dict(),
        }


def discover_raw_recordings(
    raw_dir: str | Path,
    subject: str | None = None,
    task_type: str | None = None,
) -> list[RecordingInfo]:
    raw_dir = Path(raw_dir)
    recordings = []
    for path in sorted(raw_dir.glob("*.pkl")):
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            continue
        info = RecordingInfo(
            path=path,
            subject=match.group("subject"),
            implant=match.group("implant"),
            task_type=match.group("task_type"),
            day=int(match.group("day")),
            recording=int(match.group("recording")),
        )
        if subject is not None and info.subject != subject:
            continue
        if task_type is not None and info.task_type != task_type:
            continue
        recordings.append(info)

    if not recordings:
        raise FileNotFoundError("No raw recordings matched the requested filters")
    return recordings


def group_recordings(recordings: list[RecordingInfo], group_by: str) -> dict[str, list[RecordingInfo]]:
    grouped: dict[str, list[RecordingInfo]] = defaultdict(list)
    for recording in recordings:
        if group_by == "day":
            key = recording.day_key
        elif group_by == "recording":
            key = recording.recording_id
        else:
            raise ValueError(f"Unsupported group_by: {group_by}")
        grouped[key].append(recording)
    return dict(grouped)


def load_or_build_feature_bundle(recording: RecordingInfo, cache_dir: str | Path, config: PipelineConfig) -> DatasetBundle:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = cache_dir / f"{recording.path.stem}_features.pkl"
    if bundle_path.exists():
        return load_dataset_bundle(bundle_path)
    return build_feature_dataset_from_raw(recording.path, bundle_path, config=config)


def _compute_mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def summarize_fold_results(fold_results: list[CrossTimeFoldResult]) -> dict[str, Any]:
    test_accuracy = [fold.training_result.test.accuracy for fold in fold_results]
    test_auc = [fold.training_result.test.auc for fold in fold_results]
    test_balanced_accuracy = [fold.training_result.test.balanced_accuracy for fold in fold_results]
    test_f1 = [fold.training_result.test.f1 for fold in fold_results]

    acc_mean, acc_std = _compute_mean_std(test_accuracy)
    auc_mean, auc_std = _compute_mean_std(test_auc)
    bal_mean, bal_std = _compute_mean_std(test_balanced_accuracy)
    f1_mean, f1_std = _compute_mean_std(test_f1)

    return {
        "n_folds": len(fold_results),
        "test_accuracy_mean": acc_mean,
        "test_accuracy_std": acc_std,
        "test_auc_mean": auc_mean,
        "test_auc_std": auc_std,
        "test_balanced_accuracy_mean": bal_mean,
        "test_balanced_accuracy_std": bal_std,
        "test_f1_mean": f1_mean,
        "test_f1_std": f1_std,
        "folds": [fold.to_serializable_dict() for fold in fold_results],
    }


def save_cross_time_fold_report(
    fold_result: CrossTimeFoldResult,
    output_dir: str | Path,
    config: PipelineConfig,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_result = fold_result.training_result
    Visualizer.plot_stage_metric_summary(
        training_result.train.to_summary_dict(),
        training_result.validation_metrics,
        training_result.test.to_summary_dict(),
        output_path=output_dir / f"stage_metrics.{config.output.plot_format}" if config.output.save_plots else None,
        show=config.output.show_plots,
    )
    Visualizer.plot_cv_metric_detail(
        training_result.cv_summary["auc"],
        "auc",
        output_path=output_dir / f"cv_auc.{config.output.plot_format}" if config.output.save_plots else None,
        show=config.output.show_plots,
    )
    Visualizer.plot_cv_metric_detail(
        training_result.cv_summary["accuracy"],
        "accuracy",
        output_path=output_dir / f"cv_accuracy.{config.output.plot_format}" if config.output.save_plots else None,
        show=config.output.show_plots,
    )
    Visualizer.plot_auc(
        training_result.test.fpr,
        training_result.test.tpr,
        training_result.test.auc,
        output_path=output_dir / f"roc_curve.{config.output.plot_format}" if config.output.save_plots else None,
        show=config.output.show_plots,
    )
    Visualizer.plot_confusion_matrix(
        training_result.test.confusion_matrix,
        output_path=output_dir / f"confusion_matrix.{config.output.plot_format}" if config.output.save_plots else None,
        show=config.output.show_plots,
    )
    save_json(output_dir / "metrics_summary.json", fold_result.to_serializable_dict())


def save_cross_time_summary_report(
    fold_results: list[CrossTimeFoldResult],
    summary: dict[str, Any],
    config: PipelineConfig,
) -> None:
    output_dir = ensure_output_dir(config)
    save_json(output_dir / "cross_time_summary.json", summary)

    holdout_labels = [fold.holdout_group for fold in fold_results]
    test_accuracies = [fold.training_result.test.accuracy for fold in fold_results]
    test_aucs = [fold.training_result.test.auc for fold in fold_results]
    Visualizer.plot_holdout_metric_overview(
        holdout_labels,
        test_accuracies,
        "accuracy",
        output_path=output_dir / f"cross_time_accuracy.{config.output.plot_format}" if config.output.save_plots else None,
        show=config.output.show_plots,
    )
    Visualizer.plot_holdout_metric_overview(
        holdout_labels,
        test_aucs,
        "auc",
        output_path=output_dir / f"cross_time_auc.{config.output.plot_format}" if config.output.save_plots else None,
        show=config.output.show_plots,
    )


def run_cross_time_experiment(
    raw_dir: str | Path,
    config: PipelineConfig,
    *,
    group_by: str = "day",
    cache_dir: str | Path = "data/processed/cross_time_cache",
    subject: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    recordings = discover_raw_recordings(raw_dir, subject=subject, task_type=task_type)
    grouped = group_recordings(recordings, group_by=group_by)
    feature_bundles = {
        recording.recording_id: load_or_build_feature_bundle(recording, cache_dir=cache_dir, config=config)
        for recording in recordings
    }

    output_dir = ensure_output_dir(config)
    fold_results: list[CrossTimeFoldResult] = []

    for holdout_group, test_recordings in grouped.items():
        train_recordings = [recording for group, items in grouped.items() if group != holdout_group for recording in items]
        if not train_recordings:
            continue

        train_bundle = merge_dataset_bundles(
            [feature_bundles[recording.recording_id] for recording in train_recordings],
            metadata={"holdout_group": holdout_group, "split": "train"},
        )
        test_bundle = merge_dataset_bundles(
            [feature_bundles[recording.recording_id] for recording in test_recordings],
            metadata={"holdout_group": holdout_group, "split": "test"},
        )

        X_train, y_train = train_bundle.stack()
        X_test, y_test = test_bundle.stack()
        training_result = FeatureModel(train_bundle).train_eval_one_split(
            X_train,
            y_train,
            X_test,
            y_test,
            model_config=config.model,
        )
        fold_result = CrossTimeFoldResult(
            holdout_group=holdout_group,
            train_recordings=[recording.recording_id for recording in train_recordings],
            test_recordings=[recording.recording_id for recording in test_recordings],
            training_result=training_result,
        )
        fold_results.append(fold_result)
        save_cross_time_fold_report(fold_result, output_dir / holdout_group, config)

    summary = summarize_fold_results(fold_results)
    summary["group_by"] = group_by
    summary["subject"] = subject
    summary["task_type"] = task_type
    save_cross_time_summary_report(fold_results, summary, config)
    return summary
