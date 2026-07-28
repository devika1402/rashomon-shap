"""
fig_robustness_bars.pdf: Results section 4.6.

This figure plots H2O mean pairwise Spearman rho at epsilon = 0.05. It uses
three model subsets: full set, GLM removed, and tree-exact only. It shows one
gold hue at three intensities. All values come from the robustness CSVs that
analysis/robustness_shap_method_check.py writes. Regenerating those CSVs
updates this figure automatically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory

from figlib.style import (C_H2O, INK, MUTED, HAIR, WIDTH, ROOT,
                          alpha_fill, apply_ink_style, save_fig)

NAME = "fig_robustness_bars.pdf"
DS_ORDER = ["ETTh1", "ETTh2", "ETTm1", "Electricity", "M4 Monthly", "Cable Demand"]
CSV_DIR = ROOT / "analysis" / "robustness_tree_only"

# The dominant mechanism per dataset does not come from a single CSV cell. It
# compares the GLM-removal effect (tree_approx minus full) against the
# Saabas-removal effect (tree_exact minus tree_approx). The code below derives
# it from the same summary table that the bars use.


def _load() -> pd.DataFrame | None:
    p_sum = CSV_DIR / "h2o_summary.csv"
    p_cmp = CSV_DIR / "h2o_comparison.csv"
    if not p_sum.exists() or not p_cmp.exists():
        return None
    summary = pd.read_csv(p_sum)
    comparison = pd.read_csv(p_cmp)[["dataset", "label", "delta_vs_full", "flagged"]]
    return summary.merge(comparison, on=["dataset", "label"], how="left")


def build() -> Path:
    print(f"Building {NAME} ...")
    apply_ink_style()

    df = _load()
    if df is None:
        print(f"  [skip] {NAME}: robustness CSVs missing. "
              "Run analysis/robustness_shap_method_check.py first.")
        return None

    wide = df.pivot(index="dataset", columns="label",
                    values="mean_spearman").reindex(DS_ORDER)
    flags = (df[df["label"] == "tree_exact"]
             .set_index("dataset")["flagged"].reindex(DS_ORDER))
    exact_delta = (df[df["label"] == "tree_exact"]
                   .set_index("dataset")["delta_vs_full"].reindex(DS_ORDER))
    n_ok = (df[df["label"] == "tree_exact"]
            .set_index("dataset")["n_ok"].reindex(DS_ORDER))

    full_vals = wide["full_set"].tolist()
    tree_approx_vals = wide["tree_approx"].tolist()
    tree_exact_vals = wide["tree_exact"].tolist()

    glm_effect = wide["tree_approx"] - wide["full_set"]
    saabas_effect = wide["tree_exact"] - wide["tree_approx"]
    low_n = n_ok <= 2

    datasets, deltas, delta_shown, drivers, driver_shown = DS_ORDER, [], [], [], []
    for ds in DS_ORDER:
        deltas.append(f"{exact_delta[ds]:+.2f}")
        delta_shown.append(bool(flags[ds]))
        drv = "GLM" if glm_effect[ds] > saabas_effect[ds] else "Saabas"
        drivers.append(drv + (r"$\dagger$" if low_n[ds] else ""))
        driver_shown.append(bool(flags[ds]))

    c_full   = alpha_fill(C_H2O, 0.22)
    c_approx = alpha_fill(C_H2O, 0.58)
    c_exact  = alpha_fill(C_H2O, 1.00)
    ec_full  = alpha_fill(C_H2O, 0.65)
    ec_app   = alpha_fill(C_H2O, 0.80)

    Y_MIN, Y_MAX = 0.60, 1.00
    n  = len(datasets)
    bw = 0.22
    gap = bw * 0.18
    total = 3 * bw + 2 * gap
    offs = [-total / 2 + bw / 2, 0.0, total / 2 - bw / 2]
    x = np.arange(n, dtype=float)

    fig, ax = plt.subplots(figsize=(WIDTH, 3.7))

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(HAIR)
    ax.spines["bottom"].set_color(HAIR)

    for vals, fc, ec, lw, dx in [
        (full_vals,        c_full,   ec_full, 0.8, offs[0]),
        (tree_approx_vals, c_approx, ec_app,  0.6, offs[1]),
        (tree_exact_vals,  c_exact,  "none",  0.0, offs[2]),
    ]:
        kw = dict(width=bw, bottom=Y_MIN, color=fc, zorder=3,
                  linewidth=0 if ec == "none" else lw)
        if ec != "none":
            kw["edgecolor"] = ec
        ax.bar(x + dx, [v - Y_MIN for v in vals], **kw)

    for i in range(n):
        if not delta_shown[i]:
            continue
        ax.text(x[i] + offs[2], tree_exact_vals[i] + 0.008, deltas[i],
                ha="center", va="bottom", fontsize=8, color=INK)

    ax.set_ylim(Y_MIN, Y_MAX + 0.04)
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"Mean pairwise Spearman $\rho$  ($\varepsilon$ = 0.05)",
                  fontsize=10, labelpad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([])

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i in range(n):
        ax.text(i, -0.10, datasets[i], ha="center", va="top",
                fontsize=9, color=INK, transform=trans)
        if driver_shown[i]:
            ax.text(i, -0.20, drivers[i], ha="center", va="top",
                    fontsize=8.5, color=MUTED, style="italic", transform=trans)

    leg = [
        mpatches.Patch(facecolor=c_full,   edgecolor=ec_full, linewidth=0.8,
                       label="Full set"),
        mpatches.Patch(facecolor=c_approx, edgecolor=ec_app,  linewidth=0.6,
                       label="Tree-approx (GLM removed)"),
        mpatches.Patch(facecolor=c_exact,  label="Tree-exact (GBM and XGBoost only)"),
    ]
    ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.40),
              ncol=3, fontsize=8.5, handlelength=1.4)

    low = [ds for ds in DS_ORDER if low_n[ds]]
    if low:
        counts = ", ".join(f"{ds} ($n_\\mathrm{{ok}}$ = {int(n_ok[ds])})" for ds in low)
        fig.text(0.5, 0.005,
                 rf"$\dagger$ few evaluable split-seed combinations for tree-exact: {counts}",
                 ha="center", va="bottom", fontsize=7.5, color=MUTED, style="italic")

    fig.subplots_adjust(bottom=0.28, top=0.96, left=0.11, right=0.97)
    return save_fig(fig, NAME)
