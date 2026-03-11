from __future__ import annotations

from src.core import DatasetBundle, PipelineConfig
from src.models.ml_feature_model import SplitData, TrainingResult
from src.shap_analysis.shap_analysis import ShapAnalyzer
from src.visualize.visualizer import Visualizer

from .data_io import build_plot_path, ensure_output_dir, save_json


def write_training_report_files(training_result: TrainingResult, config: PipelineConfig) -> None:
    if not config.output.save_metrics_json:
        return

    output_dir = ensure_output_dir(config)
    save_json(output_dir / config.output.metrics_filename, training_result.to_serializable_dict())
    save_json(output_dir / config.output.cv_filename, training_result.cv_summary)


def run_visual_report(
    training_result: TrainingResult,
    feature_bundle: DatasetBundle,
    split: SplitData,
    config: PipelineConfig | None = None,
) -> None:
    config = config or PipelineConfig()
    show = config.output.show_plots

    Visualizer.plot_stage_metric_summary(
        train_metrics=training_result.train.to_summary_dict(),
        validation_metrics=training_result.validation_metrics,
        test_metrics=training_result.test.to_summary_dict(),
        output_path=build_plot_path(config, config.output.summary_figure_filename) if config.output.save_plots else None,
        show=show,
    )
    Visualizer.plot_cv_metric_detail(
        training_result.cv_summary["auc"],
        "auc",
        output_path=build_plot_path(config, config.output.cv_auc_figure_filename) if config.output.save_plots else None,
        show=show,
    )
    Visualizer.plot_cv_metric_detail(
        training_result.cv_summary["accuracy"],
        "accuracy",
        output_path=build_plot_path(config, config.output.cv_accuracy_figure_filename) if config.output.save_plots else None,
        show=show,
    )
    Visualizer.plot_auc(
        training_result.test.fpr,
        training_result.test.tpr,
        training_result.test.auc,
        output_path=build_plot_path(config, config.output.roc_figure_filename) if config.output.save_plots else None,
        show=show,
    )
    Visualizer.plot_confusion_matrix(
        training_result.test.confusion_matrix,
        output_path=build_plot_path(config, config.output.confusion_matrix_filename) if config.output.save_plots else None,
        show=show,
    )

    if not config.shap.enabled:
        return

    analyzer = ShapAnalyzer.from_feature_bundle(
        training_result.best_model,
        split.X_train,
        split.X_test,
        feature_bundle,
    )
    analyzer.compute_shap(
        background_size=config.shap.background_size,
        use_kernel=config.shap.use_kernel,
    )
    analyzer.plot_channel_importance(
        output_path=build_plot_path(config, "shap_channel_importance") if config.output.save_plots else None,
        show=show,
    )
    analyzer.plot_band_importance(
        output_path=build_plot_path(config, "shap_band_importance") if config.output.save_plots else None,
        show=show,
    )
    analyzer.plot_summary(
        max_display=config.shap.summary_max_display,
        output_path=build_plot_path(config, "shap_summary") if config.output.save_plots else None,
        show=show,
    )


def print_training_summary(training_result: TrainingResult) -> None:
    print("Train metrics:", training_result.train.to_summary_dict())
    print("Validation metrics:", training_result.validation_metrics)
    print("Test metrics:", training_result.test.to_summary_dict())
    print("Best params:", training_result.best_params)
    print("Best CV score:", training_result.cv_best_score)
