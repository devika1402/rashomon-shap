"""
fig_shap_cv_groups.pdf: Results section 4.3 (RQ1).

This figure plots mean SHAP-CV by feature group at epsilon = 0.05. The
AutoGluon cluster is on the left and the H2O cluster is on the right. It shows
one hue per framework at three intensities (target lags / cov / calendar).

Every H2O run is computed over exact-TreeSHAP families only (GBM, XGBoost).
An H2O Rashomon set mixes exact TreeSHAP with Saabas (DRF, XRT) and
permutation (GLM) attributions. Their magnitudes are not on a comparable
scale. At eps = 0.05 an XRT model reaches 6e10 on Electricity and 1.1e7 on
ETTm1. The exact families reach roughly 58 and 4. A coefficient of variation
pooled over such a mixture measures the scale gap and the SHAP method mix. It
does not measure model disagreement. AutoGluon runs need no filter. All their
models use one method (permutation SHAP).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from figlib.style import (C_AG, C_H2O, HAIR, INK, MUTED, WIDTH,
                          alpha_fill, apply_ink_style, save_fig)
from figlib.datasets import DATASETS, DS_ORDER
from figlib.data import compute_group_cv

NAME = "fig_shap_cv_groups.pdf"
EPS_TARGET = 0.05
GROUP_ORDER = ["target_lag", "cov", "calendar"]


def build() -> Path:
    print(f"Building {NAME} ...")
    apply_ink_style()

    ag_combos, h2o_combos = [], []
    excluded = []
    for ds in DS_ORDER:
        for fw, color, lst in [("ag", C_AG, ag_combos), ("h2o", C_H2O, h2o_combos)]:
            run = DATASETS[ds][fw]
            if not run:
                continue
            data, n_cells, n_singleton = compute_group_cv(
                run, EPS_TARGET, GROUP_ORDER,
                tree_exact_only=(fw == "h2o"), with_cells=True,
            )
            # Singleton cells carry no cross-model spread. They are excluded
            # (see compute_group_cv). A run whose cells are all singletons has
            # nothing to measure. Report it. Do not drop it without a sign.
            if data is None:
                excluded.append(
                    f"{fw.upper()} {ds}: not evaluable, all {n_singleton} "
                    f"(split, seed) cells are singleton Rashomon sets"
                )
                continue
            lst.append({"ds": ds, "fw": fw, "color": color, "data": data,
                        "n_cells": n_cells, "n_singleton": n_singleton})

    for line in excluded:
        print(f"  [excluded] {line}")

    all_combos = ag_combos + h2o_combos
    if not all_combos:
        print(f"  [skip] {NAME}: no raw_importance.csv found. Run from project root.")
        return None
    n_ag, n_h2o = len(ag_combos), len(h2o_combos)
    GAP = 1.2
    x_ag = np.arange(n_ag, dtype=float)
    x_h2o = np.arange(n_h2o, dtype=float) + n_ag + GAP
    x_all = np.concatenate([x_ag, x_h2o])
    bw = 0.24
    offsets = np.array([-bw, 0.0, bw])

    max_top = max(m + s for combo in all_combos for m, s in combo["data"].values())

    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))

    for combo, xi in zip(all_combos, x_all):
        data, fw_color = combo["data"], combo["color"]
        for gi, grp in enumerate(GROUP_ORDER):
            if grp not in data:
                continue
            mean_val, std_val = data[grp]
            if grp == "target_lag":
                face, ec = alpha_fill(fw_color, 1.00), "none"
            elif grp == "cov":
                face, ec = alpha_fill(fw_color, 0.58), fw_color
            else:
                face, ec = alpha_fill(fw_color, 0.28), fw_color
            ax.bar(xi + offsets[gi], mean_val, yerr=std_val,
                   width=bw, facecolor=face,
                   edgecolor=ec, linewidth=0.7 if ec != "none" else 0,
                   error_kw=dict(ecolor=MUTED, capsize=2, linewidth=0.7), zorder=3)

    ax.set_ylim(0, max_top * 1.18)

    if n_ag and n_h2o:
        ax.axvline(n_ag + GAP / 2, color=HAIR, linewidth=1.0, zorder=1)
    ax.axhline(0.3, color=HAIR, linewidth=0.9, linestyle="--", zorder=2)
    ax.text(x_all[-1] + 0.45, 0.3, "CV = 0.3", va="center", fontsize=8, color=MUTED)

    header_y = max_top * 1.10
    if n_ag:
        ax.text(float(np.mean(x_ag)), header_y, "AutoGluon", ha="center",
                fontsize=10.5, fontweight="bold", color=C_AG)
    if n_h2o:
        ax.text(float(np.mean(x_h2o)), header_y, "H2O AutoML", ha="center",
                fontsize=10.5, fontweight="bold", color="#b9930a")

    ax.set_xticks(x_all)
    ax.set_xticklabels([c["ds"] for c in all_combos], fontsize=8.5,
                       rotation=22, ha="right")
    ax.set_xlim(-0.6, x_all[-1] + 0.7)
    ax.tick_params(axis="both", length=0)

    ax.set_ylabel(r"Mean SHAP-CV  ($\varepsilon$ = 0.05)", fontsize=10)

    leg = [
        mpatches.Patch(facecolor=(0.3, 0.3, 0.3, 1.00), label="Target lags"),
        mpatches.Patch(facecolor=(0.3, 0.3, 0.3, 0.58), edgecolor="#555", linewidth=0.7,
                       label="Covariate lags"),
        mpatches.Patch(facecolor=(0.3, 0.3, 0.3, 0.28), edgecolor="#555", linewidth=0.7,
                       label="Calendar features"),
    ]
    ax.legend(handles=leg, loc="upper left", fontsize=8.5, title="Feature group",
              title_fontsize=8.5)

    fig.subplots_adjust(left=0.10, right=0.985, top=0.93, bottom=0.22)
    return save_fig(fig, NAME)
