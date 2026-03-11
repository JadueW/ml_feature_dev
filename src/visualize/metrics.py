from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .style import apply_plot_style, finalize_figure


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "balanced_accuracy": "Balanced Accuracy",
    "f1": "F1",
}


def plot_stage_metric_summary(
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    test_metrics: Mapping[str, float],
    output_path: str | Path | None = None,
    show: bool = True,
):
    apply_plot_style()
    metric_names = ["accuracy", "auc", "balanced_accuracy", "f1"]
    stages = ["Train", "Validation", "Test"]
    values = np.array([
        [train_metrics[name] for name in metric_names],
        [validation_metrics[name] for name in metric_names],
        [test_metrics[name] for name in metric_names],
    ])

    x = np.arange(len(metric_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#1A2CA3", "#2e7d32", "#C40C0C"]

    for idx, stage_name in enumerate(stages):
        bars = ax.bar(x + (idx - 1) * width, values[idx], width, label=stage_name, color=colors[idx], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[name] for name in metric_names])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Train / Validation / Test Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return finalize_figure(fig, output_path=output_path, show=show)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    output_path: str | Path | None = None,
    show: bool = True,
):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#C40C0C", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#455a64", lw=1.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return finalize_figure(fig, output_path=output_path, show=show)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    labels: tuple[str, str] = ("Neg.", "Pos."),
    output_path: str | Path | None = None,
    show: bool = True,
):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="coolwarm",
        cbar=False,
        xticklabels=list(labels),
        yticklabels=list(labels),
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return finalize_figure(fig, output_path=output_path, show=show)


def plot_cv_metric_detail(
    cv_metric_summary: Mapping[str, object],
    metric_name: str,
    output_path: str | Path | None = None,
    show: bool = True,
):
    apply_plot_style()
    param_labels = list(cv_metric_summary["param_labels"])
    mean_train = np.asarray(cv_metric_summary["mean_train"], dtype=float)
    std_train = np.asarray(cv_metric_summary["std_train"], dtype=float)
    mean_validation = np.asarray(cv_metric_summary["mean_validation"], dtype=float)
    std_validation = np.asarray(cv_metric_summary["std_validation"], dtype=float)
    best_train_folds = np.asarray(cv_metric_summary["best_train_folds"], dtype=float)
    best_validation_folds = np.asarray(cv_metric_summary["best_validation_folds"], dtype=float)
    best_index = int(cv_metric_summary.get("selected_index", int(np.argmax(mean_validation))))

    x = np.arange(len(param_labels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    ax1.errorbar(x, mean_train, yerr=std_train, fmt="o-", color="#1A2CA3", capsize=4, label="Train (CV)")
    ax1.errorbar(x, mean_validation, yerr=std_validation, fmt="s-", color="#C40C0C", capsize=4, label="Validation (CV)")
    ax1.axvline(best_index, color="#2e7d32", linestyle="--", linewidth=1.5, label=f"Best Param: {param_labels[best_index]}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_labels)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))
    ax1.set_title(f"CV Parameter Search: {METRIC_LABELS.get(metric_name, metric_name)}")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="lower right")

    folds = np.arange(1, len(best_train_folds) + 1)
    ax2.plot(folds, best_train_folds, marker="o", color="#1A2CA3", label="Train Fold Score")
    ax2.plot(folds, best_validation_folds, marker="s", color="#C40C0C", label="Validation Fold Score")
    ax2.set_xticks(folds)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))
    ax2.set_title(f"Best Parameter Fold-by-Fold: {METRIC_LABELS.get(metric_name, metric_name)}")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="lower right")

    fig.tight_layout()
    return finalize_figure(fig, output_path=output_path, show=show)



def plot_holdout_metric_overview(
    holdout_labels: list[str],
    metric_values: list[float],
    metric_name: str,
    output_path: str | Path | None = None,
    show: bool = True,
):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(max(10, len(holdout_labels) * 1.4), 5))
    bars = ax.bar(np.arange(len(holdout_labels)), metric_values, color="#1A2CA3", alpha=0.85)
    ax.set_xticks(np.arange(len(holdout_labels)))
    ax.set_xticklabels(holdout_labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))
    ax.set_title(f"Holdout {METRIC_LABELS.get(metric_name, metric_name)} by Fold")
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return finalize_figure(fig, output_path=output_path, show=show)
