"""
fig_stability_heatmap.pdf: Results section 4.2.

This is a heatmap of mean temporal Spearman rho at epsilon = 0.05. The layout
is frameworks (rows) x datasets (columns). Each cell is annotated with the
numeric value. The aggregation is the scale-invariant rank-then-mean. The
cells are uniformly dark across both frameworks. No dataset-framework
combination shows low rank agreement once the code averages per-model rankings
instead of raw SHAP magnitudes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from figlib.style import abort_empty, apply_base_style, save_fig
from figlib.datasets import DATASETS, DS_ORDER
from figlib.data import load_eps_sensitivity

NAME = "fig_stability_heatmap.pdf"


def build() -> Path:
    print(f"Building {NAME} ...")
    apply_base_style()

    frameworks = ["AutoGluon", "H2O"]
    fw_keys    = ["ag", "h2o"]
    target_eps = 0.05

    # frameworks (rows) x datasets (columns): wide-short layout
    matrix = np.full((len(frameworks), len(DS_ORDER)), np.nan)

    for j, ds in enumerate(DS_ORDER):
        runs = DATASETS[ds]
        for i, fw in enumerate(fw_keys):
            run = runs[fw]
            if run is None:
                continue
            df = load_eps_sensitivity(run)
            if df is None or df.empty:
                continue
            row = df[np.isclose(df["epsilon"], target_eps)]
            if row.empty:
                continue
            matrix[i, j] = row["avg_spearman"].mean()

    fig, ax = plt.subplots(figsize=(9, 2.7))
    if np.isnan(matrix).all():
        return abort_empty(fig, NAME, 0)

    # mask NaN for colormap but keep white cell
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="YlGnBu", vmin=0.3, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(DS_ORDER)))
    ax.set_xticklabels(DS_ORDER, fontsize=9)
    ax.set_yticks(range(len(frameworks)))
    ax.set_yticklabels(frameworks, fontsize=10)
    ax.tick_params(length=0)

    for i in range(len(frameworks)):
        for j in range(len(DS_ORDER)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=9, color="#111111")
            else:
                text_color = "white" if val > 0.72 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        shrink=0.5, pad=0.28, aspect=30)
    cbar.set_label("Mean Spearman $\\rho$  ($\\varepsilon$ = 0.05)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Feature importance stability across datasets and frameworks",
                 fontsize=11, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout()
    return save_fig(fig, NAME)
