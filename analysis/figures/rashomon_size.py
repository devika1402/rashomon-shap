"""
fig_rashomon_size.pdf: Results section 4.1.

This figure plots the mean number of models in the empirical Rashomon set
against epsilon. It compares AutoGluon and H2O, with one panel per dataset.
AutoGluon sets stay small. H2O sets grow substantially. They approach the
20-model cap at wider tolerances.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from figlib.style import C_AG, C_H2O, abort_empty, apply_base_style, save_fig
from figlib.datasets import DATASETS, DS_ORDER, EPS_VALS
from figlib.data import load_rashomon_models

NAME = "fig_rashomon_size.pdf"


def build() -> Path:
    print(f"Building {NAME} ...")
    apply_base_style()

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6))
    axes = axes.flatten()
    n_series = 0

    for ax, ds in zip(axes, DS_ORDER):
        runs = DATASETS[ds]

        for fw, color, label, ls in [
            ("ag",  C_AG,  "AutoGluon", "-"),
            ("h2o", C_H2O, "H2O", "--"),
        ]:
            run = runs[fw]
            if run is None:
                continue
            df = load_rashomon_models(run)
            if df is None or df.empty:
                continue
            n_series += 1

            # count per (eps, seed, split_id) then average across combinations
            counts = (df.groupby(["eps", "seed", "split_id"])["model"]
                        .count()
                        .groupby("eps")
                        .mean()
                        .reindex(EPS_VALS)
                        .fillna(0)
                        .reset_index())
            counts.columns = ["eps", "n_models"]

            ax.plot(counts["eps"], counts["n_models"],
                    color=color, linestyle=ls, linewidth=1.8,
                    marker="s", markersize=4, label=label, zorder=3)

        ax.set_title(ds, pad=4)
        ax.set_xlabel("$\\varepsilon$", labelpad=2)
        ax.set_ylabel("Rashomon set size" if ax in (axes[0], axes[3]) else "")
        ax.set_xlim(-0.01, 0.32)
        ax.set_xticks(EPS_VALS)
        ax.set_xticklabels(["0.02", "0.05", "0.10", "0.20", "0.30"], fontsize=8)
        ax.tick_params(axis="both", length=3)
        ax.set_ylim(bottom=0)

    if n_series == 0:
        return abort_empty(fig, NAME, n_series)

    handles = [
        mpatches.Patch(color=C_AG,  label="AutoGluon (best_quality)"),
        mpatches.Patch(color=C_H2O, label="H2O"),
    ]
    axes[1].legend(handles=handles, loc="upper left", fontsize=8)

    fig.suptitle(
        "Rashomon set size growth with $\\varepsilon$ threshold",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return save_fig(fig, NAME)
