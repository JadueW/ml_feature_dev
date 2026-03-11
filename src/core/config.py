from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


DEFAULT_FEATURE_BANDS = {
    "beta": (13, 30),
    "high_gamma": (70, 150),
}


@dataclass(frozen=True)
class PreprocessConfig:
    notch_frequencies: tuple[float, ...] = (50.0, 100.0, 150.0)
    lowcut: float = 1.0
    highcut: float = 200.0
    notch_bandwidth: float = 2.0
    bandpass_order: int = 4


@dataclass(frozen=True)
class FeatureConfig:
    bands: Mapping[str, tuple[float, float]] = field(default_factory=lambda: DEFAULT_FEATURE_BANDS.copy())
    lmp_lowpass_cutoff: float = 4.0
    nperseg: int = 1024
    noverlap: int = 512
    parallel: bool = True
    n_jobs: int = -1
    use_log_power: bool = True
    total_power_range: tuple[float, float] = (1.0, 150.0)


@dataclass(frozen=True)
class SplitConfig:
    strategy: str = "ratio"
    ratio: float = 0.8
    samples_per_class: int | None = None
    shuffle_within_class: bool = False


@dataclass(frozen=True)
class ModelConfig:
    c_values: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    l1_ratios: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    inner_cv_splits: int = 5
    repeat_splits: int = 1
    max_iter: int = 20000
    class_weight: str = "balanced"
    threshold: float = 0.5
    random_state: int = 42
    refit_metric: str = "auc"
    scoring_metrics: tuple[str, ...] = ("auc", "accuracy", "balanced_accuracy", "f1")


@dataclass(frozen=True)
class ShapConfig:
    enabled: bool = True
    background_size: int = 100
    use_kernel: bool = False
    summary_max_display: int = 30
    top_k_channels: int = 10


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str = "outputs/latest"
    plot_format: str = "png"
    save_plots: bool = True
    show_plots: bool = False
    save_metrics_json: bool = True
    metrics_filename: str = "metrics_summary.json"
    cv_filename: str = "cv_summary.json"
    summary_figure_filename: str = "stage_metrics"
    cv_auc_figure_filename: str = "cv_auc"
    cv_accuracy_figure_filename: str = "cv_accuracy"
    roc_figure_filename: str = "roc_curve"
    confusion_matrix_filename: str = "confusion_matrix"
    model_filename: str = "ml_decoder_model.pkl"


@dataclass(frozen=True)
class PipelineConfig:
    n_channels: int = 128
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    shap: ShapConfig = field(default_factory=ShapConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
