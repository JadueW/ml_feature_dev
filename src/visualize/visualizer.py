from __future__ import annotations

from pathlib import Path

from .metrics import (
    plot_confusion_matrix,
    plot_cv_metric_detail,
    plot_holdout_metric_overview,
    plot_roc_curve,
    plot_stage_metric_summary,
)
from .shap_plots import (
    plot_shap_band_importance,
    plot_shap_channel_band_heatmap,
    plot_shap_channel_importance,
    plot_shap_channel_profile,
    plot_shap_direction_summary,
    plot_shap_summary,
    plot_shap_type_importance,
)
from .style import apply_plot_style


class Visualizer:
    BAND_NAMES = ["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma"]

    @staticmethod
    def initialize_style() -> None:
        apply_plot_style()

    @staticmethod
    def plot_auc(fpr, tpr, auc, output_path: str | Path | None = None, show: bool = True):
        return plot_roc_curve(fpr=fpr, tpr=tpr, auc=auc, output_path=output_path, show=show)

    @staticmethod
    def plot_confusion_matrix(confusion_matrix, output_path: str | Path | None = None, show: bool = True):
        return plot_confusion_matrix(confusion_matrix=confusion_matrix, output_path=output_path, show=show)

    @staticmethod
    def plot_stage_metric_summary(train_metrics, validation_metrics, test_metrics, output_path: str | Path | None = None, show: bool = True):
        return plot_stage_metric_summary(
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            output_path=output_path,
            show=show,
        )

    @staticmethod
    def plot_cv_metric_detail(cv_metric_summary, metric_name: str, output_path: str | Path | None = None, show: bool = True):
        return plot_cv_metric_detail(
            cv_metric_summary=cv_metric_summary,
            metric_name=metric_name,
            output_path=output_path,
            show=show,
        )

    @staticmethod
    def plot_holdout_metric_overview(holdout_labels, metric_values, metric_name: str, output_path: str | Path | None = None, show: bool = True):
        return plot_holdout_metric_overview(
            holdout_labels=holdout_labels,
            metric_values=metric_values,
            metric_name=metric_name,
            output_path=output_path,
            show=show,
        )

    @staticmethod
    def plot_shap_channel_importance(*args, **kwargs):
        return plot_shap_channel_importance(*args, **kwargs)

    @staticmethod
    def plot_shap_band_importance(*args, **kwargs):
        return plot_shap_band_importance(*args, **kwargs)

    @staticmethod
    def plot_shap_type_importance(*args, **kwargs):
        return plot_shap_type_importance(*args, **kwargs)

    @staticmethod
    def plot_shap_channel_band_heatmap(*args, **kwargs):
        return plot_shap_channel_band_heatmap(*args, **kwargs)

    @staticmethod
    def plot_shap_direction_summary(*args, **kwargs):
        return plot_shap_direction_summary(*args, **kwargs)

    @staticmethod
    def plot_shap_channel_profile(*args, **kwargs):
        return plot_shap_channel_profile(*args, **kwargs)

    @staticmethod
    def plot_shap_summary(*args, **kwargs):
        return plot_shap_summary(*args, **kwargs)
