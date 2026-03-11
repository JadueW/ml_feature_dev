from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap

from .style import apply_plot_style, finalize_figure


DEFAULT_BAND_NAMES = ("lmp", "beta", "high_gamma")


def _reshape_shap_values(shap_values, n_channels: int, n_bands: int, n_types: int) -> np.ndarray:
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    if shap_values.ndim == 2:
        return shap_values.reshape(-1, n_channels, n_bands, n_types)
    return shap_values


def plot_shap_channel_importance(
    shap_values,
    n_channels: int = 128,
    n_bands: int = 6,
    n_types: int = 2,
    data_name: str = "test",
    output_path: str | Path | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (12, 7),
):
    apply_plot_style()
    shap_reshaped = _reshape_shap_values(shap_values, n_channels, n_bands, n_types)
    channel_importance = np.mean(np.abs(shap_reshaped), axis=(0, 2, 3))
    sorted_idx = np.argsort(channel_importance)[::-1]
    sorted_importance = channel_importance[sorted_idx]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(n_channels), sorted_importance, color="#1A2CA3")
    ax.set_xlabel("Channel (sorted)")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title(f"Channel Importance ({data_name})")
    ax.grid(axis="y", alpha=0.25)
    for idx in range(min(10, n_channels)):
        ax.text(idx, sorted_importance[idx] + sorted_importance.max() * 0.01, str(sorted_idx[idx]), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    finalize_figure(fig, output_path=output_path, show=show)
    return sorted_idx, sorted_importance


def plot_shap_band_importance(
    shap_values,
    n_channels: int = 128,
    n_bands: int = 6,
    n_types: int = 2,
    separate_types: bool = True,
    data_name: str = "test",
    band_names: tuple[str, ...] | list[str] | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (10, 6),
):
    apply_plot_style()
    band_names = tuple(band_names or DEFAULT_BAND_NAMES[:n_bands])
    shap_reshaped = _reshape_shap_values(shap_values, n_channels, n_bands, n_types)
    fig, ax = plt.subplots(figsize=figsize)

    if separate_types:
        imp_abs = np.mean(np.abs(shap_reshaped[:, :, :, 0]), axis=(0, 1))
        imp_rel = np.mean(np.abs(shap_reshaped[:, :, :, 1]), axis=(0, 1))
        x = np.arange(n_bands)
        width = 0.35
        ax.bar(x - width / 2, imp_abs, width, label="abs", color="#1A2CA3")
        ax.bar(x + width / 2, imp_rel, width, label="rel", color="#C40C0C")
        ax.legend()
        result = (imp_abs, imp_rel)
    else:
        imp_all = np.mean(np.abs(shap_reshaped), axis=(0, 1, 3))
        ax.bar(np.arange(n_bands), imp_all, color="#1A2CA3")
        result = imp_all

    ax.set_xticks(np.arange(n_bands))
    ax.set_xticklabels(band_names)
    ax.set_xlabel("Band")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title(f"Band Importance ({data_name})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    finalize_figure(fig, output_path=output_path, show=show)
    return result


def plot_shap_type_importance(
    shap_values,
    n_channels: int = 128,
    n_bands: int = 6,
    n_types: int = 2,
    data_name: str = "test",
    output_path: str | Path | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (6, 5),
):
    apply_plot_style()
    shap_reshaped = _reshape_shap_values(shap_values, n_channels, n_bands, n_types)
    imp_abs = float(np.mean(np.abs(shap_reshaped[:, :, :, 0])))
    imp_rel = float(np.mean(np.abs(shap_reshaped[:, :, :, 1])))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(["abs", "rel"], [imp_abs, imp_rel], color=["#1A2CA3", "#C40C0C"])
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title(f"Type Importance ({data_name})")
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(imp_abs, imp_rel) * 0.01, f"{bar.get_height():.4f}", ha="center", va="bottom")
    fig.tight_layout()
    finalize_figure(fig, output_path=output_path, show=show)
    return imp_abs, imp_rel


def plot_shap_channel_band_heatmap(
    shap_values,
    n_channels: int = 128,
    n_bands: int = 6,
    n_types: int = 2,
    type_idx: int | None = None,
    data_name: str = "test",
    band_names: tuple[str, ...] | list[str] | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (14, 8),
):
    apply_plot_style()
    band_names = tuple(band_names or DEFAULT_BAND_NAMES[:n_bands])
    shap_reshaped = _reshape_shap_values(shap_values, n_channels, n_bands, n_types)

    if type_idx is None:
        heatmap_data = np.mean(np.abs(shap_reshaped), axis=(0, 3))
        title_suffix = "merged"
    else:
        heatmap_data = np.mean(np.abs(shap_reshaped[:, :, :, type_idx]), axis=0)
        title_suffix = "abs" if type_idx == 0 else "rel"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(heatmap_data.T, cmap="viridis", cbar_kws={"label": "Mean |SHAP|"}, ax=ax)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Band")
    ax.set_yticks(np.arange(n_bands) + 0.5)
    ax.set_yticklabels(band_names)
    ax.set_title(f"Channel-Band Heatmap ({title_suffix}, {data_name})")
    fig.tight_layout()
    finalize_figure(fig, output_path=output_path, show=show)
    return heatmap_data


def plot_shap_direction_summary(
    shap_values,
    n_channels: int = 128,
    n_bands: int = 6,
    n_types: int = 2,
    top_k_channels: int = 10,
    data_name: str = "test",
    output_path: str | Path | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (14, 8),
):
    apply_plot_style()
    shap_reshaped = _reshape_shap_values(shap_values, n_channels, n_bands, n_types)
    channel_mean = np.mean(shap_reshaped, axis=(0, 2, 3))
    top_channels = np.argsort(np.abs(channel_mean))[-top_k_channels:][::-1]
    box_data = [shap_reshaped[:, ch, :, :].reshape(-1) for ch in top_channels]
    labels = [f"Ch{ch}\n({channel_mean[ch]:.3f})" for ch in top_channels]

    fig, ax = plt.subplots(figsize=figsize)
    box = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
    for idx, ch in enumerate(top_channels):
        box["boxes"][idx].set_facecolor("#2e7d32" if channel_mean[ch] > 0 else "#C40C0C")
        box["boxes"][idx].set_alpha(0.65)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Channel")
    ax.set_ylabel("SHAP value")
    ax.set_title(f"Top Channel SHAP Distribution ({data_name})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    finalize_figure(fig, output_path=output_path, show=show)
    return top_channels, channel_mean[top_channels]


def plot_shap_channel_profile(
    shap_values,
    channel_idx: int,
    n_channels: int = 128,
    n_bands: int = 6,
    n_types: int = 2,
    data_name: str = "test",
    band_names: tuple[str, ...] | list[str] | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (12, 5),
):
    apply_plot_style()
    band_names = tuple(band_names or DEFAULT_BAND_NAMES[:n_bands])
    shap_reshaped = _reshape_shap_values(shap_values, n_channels, n_bands, n_types)
    channel_values = shap_reshaped[:, channel_idx, :, :]
    avg_values = np.mean(channel_values, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    bands = np.arange(n_bands)
    width = 0.35
    ax1.bar(bands - width / 2, avg_values[:, 0], width, label="abs", color="#1A2CA3")
    ax1.bar(bands + width / 2, avg_values[:, 1], width, label="rel", color="#C40C0C")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(bands)
    ax1.set_xticklabels(band_names)
    ax1.set_title(f"Channel {channel_idx} Average SHAP")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.25)

    ax2.boxplot(channel_values.reshape(-1), vert=False, showfliers=False)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_title(f"Channel {channel_idx} SHAP Distribution")
    ax2.set_yticks([])

    fig.suptitle(f"Channel Profile ({data_name})")
    fig.tight_layout()
    finalize_figure(fig, output_path=output_path, show=show)
    return avg_values


def plot_shap_summary(
    shap_values,
    X,
    feature_names=None,
    max_display: int = 20,
    data_name: str = "test",
    output_path: str | Path | None = None,
    show: bool = True,
):
    apply_plot_style()
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    if shap_values.ndim == 4:
        shap_values = shap_values.reshape(shap_values.shape[0], -1)
    elif shap_values.ndim == 3 and shap_values.shape[2] == 2:
        shap_values = shap_values[:, :, 1]

    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=max_display, show=False)
    plt.title(f"SHAP Summary ({data_name})")
    plt.tight_layout()
    return finalize_figure(fig, output_path=output_path, show=show)

