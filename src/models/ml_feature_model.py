from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import trange

from src.core import DatasetBundle, ModelConfig, SplitConfig

warnings.filterwarnings("ignore")


SCORING_MAP = {
    "auc": "roc_auc",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
}


@dataclass(frozen=True)
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    auc: float
    balanced_accuracy: float
    f1: float
    confusion_matrix: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray

    def to_summary_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "balanced_accuracy": self.balanced_accuracy,
            "f1": self.f1,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "bal_acc": self.balanced_accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1": self.f1,
            "cm": self.confusion_matrix,
            "confusion_matrix": self.confusion_matrix,
            "fpr": self.fpr,
            "tpr": self.tpr,
        }

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "balanced_accuracy": self.balanced_accuracy,
            "f1": self.f1,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "fpr": self.fpr.tolist(),
            "tpr": self.tpr.tolist(),
        }


@dataclass
class TrainingResult:
    train: ClassificationMetrics
    validation_metrics: dict[str, float]
    test: ClassificationMetrics
    best_params: dict[str, Any]
    cv_best_score: float
    cv_results: dict[str, Any]
    cv_summary: dict[str, dict[str, Any]]
    best_model: Pipeline
    best_index: int
    refit_metric: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": {**self.train.to_dict(), "best_params": self.best_params, "cv_best_score": self.cv_best_score},
            "validation": self.validation_metrics,
            "test": self.test.to_dict(),
            "cv_results": self.cv_results,
            "cv_summary": self.cv_summary,
            "auc": self.test.auc,
            "accuracy": self.test.accuracy,
            "bal_acc": self.test.balanced_accuracy,
            "f1": self.test.f1,
            "cm": self.test.confusion_matrix,
            "best_params": self.best_params,
            "best_score": self.cv_best_score,
            "best_index": self.best_index,
            "refit_metric": self.refit_metric,
            "fpr": self.test.fpr,
            "tpr": self.test.tpr,
            "best_model": self.best_model,
        }

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "train": self.train.to_serializable_dict(),
            "validation": self.validation_metrics,
            "test": self.test.to_serializable_dict(),
            "best_params": self.best_params,
            "cv_best_score": self.cv_best_score,
            "cv_summary": self.cv_summary,
            "best_index": self.best_index,
            "refit_metric": self.refit_metric,
        }


class FeatureModel:
    def __init__(self, dataset_bundle: DatasetBundle | dict):
        if isinstance(dataset_bundle, DatasetBundle):
            self.bundle = dataset_bundle
        else:
            self.bundle = DatasetBundle.from_serialized(dataset_bundle)

        if self.bundle.n_classes != 2:
            raise ValueError(f"FeatureModel currently expects a binary dataset, got {self.bundle.n_classes} classes")

    def _resolve_train_counts(self, config: SplitConfig) -> dict[int, int]:
        class_counts = self.bundle.get_class_counts()

        if config.strategy == "min":
            samples_per_class = min(class_counts.values())
            return {class_id: samples_per_class for class_id in self.bundle.class_ids}

        if config.strategy == "ratio":
            return {class_id: int(count * config.ratio) for class_id, count in class_counts.items()}

        if config.strategy == "fixed":
            if config.samples_per_class is None:
                raise ValueError("samples_per_class is required when strategy='fixed'")
            smallest_class = min(class_counts.values())
            if config.samples_per_class > smallest_class:
                raise ValueError(
                    f"samples_per_class={config.samples_per_class} exceeds smallest class size {smallest_class}"
                )
            return {class_id: config.samples_per_class for class_id in self.bundle.class_ids}

        raise ValueError(f"Unknown split strategy: {config.strategy}")

    def create_split(self, config: SplitConfig | None = None) -> SplitData:
        config = config or SplitConfig()
        train_counts = self._resolve_train_counts(config)
        rng = np.random.default_rng(42)

        train_parts = []
        test_parts = []
        train_labels = []
        test_labels = []

        for class_id in self.bundle.class_ids:
            class_data = self.bundle.get_class_data(class_id)
            train_count = train_counts[class_id]
            if train_count < 0 or train_count > class_data.shape[0]:
                raise ValueError(f"Invalid train_count={train_count} for class {class_id}")
            if train_count == 0 or train_count == class_data.shape[0]:
                raise ValueError(
                    f"Class {class_id} must keep at least one sample in both train and test splits, got train_count={train_count}"
                )

            indices = np.arange(class_data.shape[0])
            if config.shuffle_within_class:
                indices = rng.permutation(indices)
            train_indices = indices[:train_count]
            test_indices = indices[train_count:]

            train_parts.append(class_data[train_indices])
            test_parts.append(class_data[test_indices])
            train_labels.append(np.full(train_indices.shape[0], class_id, dtype=int))
            test_labels.append(np.full(test_indices.shape[0], class_id, dtype=int))

        return SplitData(
            X_train=np.vstack(train_parts),
            y_train=np.concatenate(train_labels),
            X_test=np.vstack(test_parts),
            y_test=np.concatenate(test_labels),
        )

    def train_test_split_manual(
        self,
        strategy: str = "min",
        ratio: float = 0.8,
        m: int | None = None,
        split_config: SplitConfig | None = None,
    ):
        if split_config is None:
            split_config = SplitConfig(strategy=strategy, ratio=ratio, samples_per_class=m)
        split = self.create_split(split_config)
        return split.X_train, split.y_train, split.X_test, split.y_test

    def make_pipeline(self, config: ModelConfig | None = None) -> Pipeline:
        config = config or ModelConfig()
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=config.max_iter,
            class_weight=config.class_weight,
            random_state=config.random_state,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    def make_param_grid(self, config: ModelConfig | None = None) -> dict:
        config = config or ModelConfig()
        return {
            "clf__C": list(config.c_values),
            "clf__l1_ratio": list(config.l1_ratios),
        }

    @staticmethod
    def _evaluate_split(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> ClassificationMetrics:
        predictions = (probabilities >= threshold).astype(int)
        auc = roc_auc_score(y_true, probabilities)
        accuracy = accuracy_score(y_true, predictions)
        balanced_accuracy = balanced_accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)
        cm = confusion_matrix(y_true, predictions)
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        return ClassificationMetrics(
            accuracy=float(accuracy),
            auc=float(auc),
            balanced_accuracy=float(balanced_accuracy),
            f1=float(f1),
            confusion_matrix=cm,
            fpr=fpr,
            tpr=tpr,
        )

    @staticmethod
    def _build_scoring(metrics: tuple[str, ...]) -> dict[str, str]:
        unsupported = [metric for metric in metrics if metric not in SCORING_MAP]
        if unsupported:
            raise ValueError(f"Unsupported scoring metrics: {unsupported}")
        return {metric: SCORING_MAP[metric] for metric in metrics}

    @staticmethod
    def _build_cv_summary(cv_results: dict[str, Any], metrics: tuple[str, ...], best_index: int) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
        summary: dict[str, dict[str, Any]] = {}
        validation_metrics: dict[str, float] = {}
        param_labels = [f"P{idx + 1}" for idx in range(len(cv_results["params"]))]

        for metric in metrics:
            mean_train = np.asarray(cv_results[f"mean_train_{metric}"], dtype=float)
            std_train = np.asarray(cv_results[f"std_train_{metric}"], dtype=float)
            mean_validation = np.asarray(cv_results[f"mean_test_{metric}"], dtype=float)
            std_validation = np.asarray(cv_results[f"std_test_{metric}"], dtype=float)

            fold_train = []
            fold_validation = []
            fold_idx = 0
            while f"split{fold_idx}_train_{metric}" in cv_results:
                fold_train.append(float(cv_results[f"split{fold_idx}_train_{metric}"][best_index]))
                fold_validation.append(float(cv_results[f"split{fold_idx}_test_{metric}"][best_index]))
                fold_idx += 1

            summary[metric] = {
                "param_labels": param_labels,
                "mean_train": mean_train.tolist(),
                "std_train": std_train.tolist(),
                "mean_validation": mean_validation.tolist(),
                "std_validation": std_validation.tolist(),
                "best_train_folds": fold_train,
                "best_validation_folds": fold_validation,
                "best_train_mean": float(mean_train[best_index]),
                "best_validation_mean": float(mean_validation[best_index]),
                "selected_index": best_index,
                "selected_label": param_labels[best_index],
            }
            validation_metrics[metric] = float(mean_validation[best_index])

        return summary, validation_metrics

    def train_eval_one_split(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_config: ModelConfig | None = None,
    ) -> TrainingResult:
        model_config = model_config or ModelConfig()
        pipeline = self.make_pipeline(model_config)
        param_grid = self.make_param_grid(model_config)
        inner_cv = StratifiedKFold(
            n_splits=model_config.inner_cv_splits,
            shuffle=True,
            random_state=model_config.random_state,
        )

        scoring = self._build_scoring(model_config.scoring_metrics)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit=model_config.refit_metric,
            cv=inner_cv,
            verbose=0,
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_index = int(search.best_index_)

        train_prob = best_model.predict_proba(X_train)[:, 1]
        test_prob = best_model.predict_proba(X_test)[:, 1]
        cv_summary, validation_metrics = self._build_cv_summary(search.cv_results_, model_config.scoring_metrics, best_index)

        return TrainingResult(
            train=self._evaluate_split(y_train, train_prob, model_config.threshold),
            validation_metrics=validation_metrics,
            test=self._evaluate_split(y_test, test_prob, model_config.threshold),
            best_params=search.best_params_,
            cv_best_score=float(search.best_score_),
            cv_results=search.cv_results_,
            cv_summary=cv_summary,
            best_model=best_model,
            best_index=best_index,
            refit_metric=model_config.refit_metric,
        )

    def train_eval_splits(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_splits: int | None = None,
        model_config: ModelConfig | None = None,
    ) -> tuple[TrainingResult, dict[int, TrainingResult]]:
        model_config = model_config or ModelConfig()
        n_splits = model_config.repeat_splits if n_splits is None else n_splits

        all_results = {}
        for split_idx in trange(n_splits, desc="Training splits"):
            all_results[split_idx] = self.train_eval_one_split(
                X_train,
                y_train,
                X_test,
                y_test,
                model_config=model_config,
            )

        best_result = max(all_results.values(), key=lambda result: result.cv_best_score)
        return best_result, all_results


