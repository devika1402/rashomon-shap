"""
fig_epsilon_sensitivity.pdf: Results section 4.4.

This figure plots mean temporal Spearman rho against epsilon. It shows one
panel per dataset. AutoGluon is blue and H2O AutoML is gold. Every profile is
flat or gently rising. No framework shows a stability step change at any
tolerance.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from figlib.style import C_AG, C_H2O, abort_empty, apply_base_style, save_fig
from figlib.datasets import DATASETS, DS_ORDER, EPS_VALS
from figlib.data import load_eps_sensitivity

NAME = "fig_epsilon_sensitivity.pdf"


def _profile(run, aggregator):
    df = load_eps_sensitivity(run, aggregator=aggregator)
    if df is None or df.empty:
        return None, None
    # One row per epsilon. The per-cell std across the consecutive-split
    # comparisons (seeds x split pairs) is precomputed as std_spearman.
    agg = df.set_index("epsilon").reindex(EPS_VALS)
    return agg["avg_spearman"].values, agg["std_spearman"].fillna(0).values


def build() -> Path:
    print(f"Building {NAME} ...")
    apply_base_style()

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.8), sharey=True)
    axes = axes.flatten()
    n_series = 0

    for ax, ds in zip(axes, DS_ORDER):
        runs = DATASETS[ds]

        for fw, color, label, ls in [
            ("ag",  C_AG,  "AutoGluon", "-"),
            ("h2o", C_H2O, "H2O AutoML", "--"),
        ]:
            run = runs[fw]
            if run is None:
                continue
            mu, sigma = _profile(run, "rank_then_mean")
            if mu is None:
                continue
            n_series += 1
            ms = 6 if fw == "h2o" else 4
            ax.plot(EPS_VALS, mu, color=color, linestyle=ls, linewidth=1.8,
                    marker="o", markersize=ms, label=label, zorder=3)
            ax.fill_between(EPS_VALS, mu - sigma, mu + sigma,
                            color=color, alpha=0.12, zorder=2)

        ax.set_title(ds, pad=4)
        ax.set_xlabel("$\\varepsilon$", labelpad=2)
        ax.set_ylabel("Mean temporal Spearman $\\rho$"
                      if ax in (axes[0], axes[3]) else "")
        ax.set_ylim(0.5, 1.02)
        ax.set_xlim(-0.01, 0.32)
        ax.set_xticks(EPS_VALS)
        ax.set_xticklabels(["0.02", "0.05", "0.10", "0.20", "0.30"], fontsize=8)
        ax.axhline(1.0, color="#CCCCCC", linewidth=0.6, linestyle=":")
        ax.tick_params(axis="both", length=3)

    if n_series == 0:
        return abort_empty(fig, NAME, n_series)

    handles = [
        mlines.Line2D([], [], color=C_AG, linewidth=1.8, marker="o",
                      markersize=4, label="AutoGluon"),
        mlines.Line2D([], [], color=C_H2O, linewidth=1.8, linestyle="--",
                      marker="o", markersize=6, label="H2O AutoML"),
    ]
    axes[1].legend(handles=handles, loc="lower right", fontsize=7.5)

    fig.tight_layout()
    return save_fig(fig, NAME)
