"""
fig_stability_vs_size.pdf: Results section 4.4 (RQ2).

This figure plots mean temporal Spearman rho against mean Rashomon set size at
epsilon = 0.05. It shows one point per dataset x framework. Stability is flat
in set size. The H2O points fall among the AutoGluon points. This holds even
though the H2O sets are several times larger.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from figlib.style import C_AG, C_H2O, abort_empty, apply_base_style, save_fig
from figlib.datasets import DATASETS, DS_ORDER
from figlib.data import load_eps_sensitivity, load_rashomon_models

NAME = "fig_stability_vs_size.pdf"
EPS = 0.05

OFFSETS_AG = {"Electricity": (6, -9), "ETTh1": (6, 4), "ETTh2": (6, 4),
              "ETTm1": (6, -9), "M4 Monthly": (6, 4), "Cable Demand": (6, -9)}
OFFSETS_H2O = {"Electricity": (5, 6), "ETTh1": (5, -10), "ETTh2": (5, 6),
               "ETTm1": (5, 6), "M4 Monthly": (5, -10), "Cable Demand": (2, -12)}


def _rho(run, aggregator):
    df = load_eps_sensitivity(run, aggregator=aggregator)
    if df is None or df.empty:
        return None
    row = df[np.isclose(df["epsilon"], EPS)]
    return None if row.empty else row["avg_spearman"].mean()


def build() -> Path:
    print(f"Building {NAME} ...")
    apply_base_style()

    fig, ax = plt.subplots(figsize=(7, 3.9))
    any_point = False

    for ds in DS_ORDER:
        runs = DATASETS[ds]
        for fw, color, marker in [("ag", C_AG, "o"), ("h2o", C_H2O, "s")]:
            run = runs[fw]
            if run is None:
                continue
            df_rm = load_rashomon_models(run)
            rho = _rho(run, "rank_then_mean")
            if df_rm is None or rho is None:
                continue

            n_models = (df_rm[np.isclose(df_rm["eps"], EPS)]
                        .groupby(["seed", "split_id"])["model"].count().mean())
            any_point = True

            ax.scatter(n_models, rho, color=color, marker=marker, s=70, zorder=4)
            off = OFFSETS_AG[ds] if fw == "ag" else OFFSETS_H2O[ds]
            ax.annotate(ds, (n_models, rho), xytext=off,
                        textcoords="offset points", fontsize=7.5, color="#111111")

    if not any_point:
        return abort_empty(fig, NAME, 0)

    ax.set_xlabel(f"Rashomon set size  (mean models at $\\varepsilon$ = {EPS:.2f})",
                  fontsize=10)
    ax.set_ylabel(f"Mean temporal Spearman $\\rho$  ($\\varepsilon$ = {EPS:.2f})",
                  fontsize=10)
    ax.set_ylim(0.6, 1.0)
    ax.tick_params(axis="both", length=3)

    handles = [
        mlines.Line2D([], [], color=C_AG, marker="o", linestyle="none",
                      markersize=8, label="AutoGluon"),
        mlines.Line2D([], [], color=C_H2O, marker="s", linestyle="none",
                      markersize=8, label="H2O AutoML"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower left")

    fig.tight_layout()
    return save_fig(fig, NAME)
