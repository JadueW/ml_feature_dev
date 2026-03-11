from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif", "SimHei", "Microsoft YaHei UI"],
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "axes.unicode_minus": False,
    "font.size": 14,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "axes.prop_cycle": plt.cycler(
        color=["#1A2CA3", "#C40C0C", "#2e7d32", "#f57c00", "#00695c", "#6a1b9a"]
    ),
}


_STYLE_APPLIED = False


def apply_plot_style() -> None:
    global _STYLE_APPLIED
    if not _STYLE_APPLIED:
        plt.rcParams.update(PLOT_STYLE)
        _STYLE_APPLIED = True


def finalize_figure(fig, output_path: str | Path | None = None, show: bool = True):
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
