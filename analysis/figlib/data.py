"""
figlib.data: result-CSV loaders and feature-group logic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .style import ROOT

RESULTS = ROOT / "results"


# Per-run loaders ───────────────────────────────────────────────────────────

def load_eps_sensitivity(run: str, aggregator: str = "rank_then_mean") -> pd.DataFrame | None:
    """Read stability against epsilon for one run.

    ``rank_then_mean`` is the default. It reads the scale-invariant aggregation
    that ``analysis/recompute_stability_rank_agg.py`` writes. This aggregation
    ranks the features within each model. It then averages the ranks across the
    Rashomon set. The thesis reports this aggregation.

    ``mean_then_rank`` reads the magnitude-weighted aggregation. This
    aggregation ranks the mean of the raw mean-absolute-SHAP magnitudes. The
    magnitudes are not comparable across explainers. For example, H2O's
    Saabas-explained XRT can exceed the exact-TreeSHAP families on
    Electricity by many orders of magnitude. One model can therefore determine the aggregate ranking. We keep
    this option only to reproduce that artefact. The 2026-07-13 stability
    regeneration overwrote the per-run ``04_stability/epsilon_sensitivity.csv``
    with rank-then-mean values. The magnitude-weighted profiles that remain are
    in the provenance CSV. This CSV is keyed by run and comes from
    ``analysis/emit_provenance_csvs.py``. This function reads it.
    """
    if aggregator == "rank_then_mean":
        p = RESULTS / run / "04_stability" / "epsilon_sensitivity_rankagg.csv"
        if not p.exists():
            return None
        return pd.read_csv(p)
    if aggregator == "mean_then_rank":
        p = ROOT / "analysis" / "epsilon_sensitivity_meanrank.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        df = df[df["run"] == run]
        return df if not df.empty else None
    raise ValueError(f"unknown aggregator: {aggregator!r}")


def load_rashomon_models(run: str) -> pd.DataFrame | None:
    p = RESULTS / run / "05_rashomon" / "rashomon_models.csv"
    if not p.exists() or p.stat().st_size < 10:
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


# Feature-group classification ──────────────────────────────────────────────

def classify_feature(name: str) -> str:
    """Map a feature column name to one of: target_lag, cov, calendar, other.

    ``trend_lag1`` is a covariate. It is not a calendar feature. It is
    Electricity's only covariate.

    Branch order matters. Test the explicit calendar list BEFORE the ``_lag1``
    heuristic below. Calendar features are themselves lagged (``hour_lag1``,
    ``month_lag1``, ``day_of_week_lag1``). If the heuristic ran first, it would
    put them in ``cov``. The calendar branch would then be unreachable. Every
    lagged calendar feature would be misfiled without any warning.
    """
    if name.startswith("target_lag"):
        return "target_lag"
    if name in ("year", "month_lag1", "day_of_week_lag1", "hour_lag1",
                "month_sin", "month_cos", "cal_month_lag1", "cal_quarter_lag1",
                "cal_year_lag1"):
        return "calendar"
    if name.startswith("cov_") or (name.endswith("_lag1") and not name.startswith("target")):
        return "cov"
    return "other"


# These H2O families give contributions from exact TreeSHAP. DRF and XRT use
# the Saabas approximation. GLM uses a permutation explainer. Their
# attributions are since not on a comparable scale.
TREE_EXACT_FAMILIES = ("GBM", "XGBoost")


def compute_group_cv(run: str, eps_target: float = 0.05,
                     group_order=("target_lag", "cov", "calendar"),
                     tree_exact_only: bool = False,
                     min_models: int = 2,
                     with_cells: bool = False) -> dict | None:
    """Mean within-feature SHAP coefficient of variation per feature group.

    This function reads ``03_importance/raw_importance.csv``. It restricts the
    rows to ``eps_target``. For each (split, seed) it computes the per-feature
    CV. The CV is the std over the mean of global importance across the
    Rashomon set. It then averages the CVs to a per-group
    (mean, std-across-combinations) tuple. It returns ``None`` if there is no
    data.

    This CV variant uses population std (ddof=0) over the mean, guarded by
    mean > 0. It differs from eq:shap_cv in src/importance_aggregation.py.
    That equation uses sample std and an additive 1e-10 in the denominator.

    Set ``tree_exact_only`` to restrict the rows to ``TREE_EXACT_FAMILIES``.
    This is required for every H2O run. Saabas (DRF, XRT) and permutation (GLM)
    attributions share no magnitude scale with exact TreeSHAP. A single XRT
    model reaches 6e10 on Electricity and 1.1e7 on ETTm1. The exact families
    reach only tens or units. The mixed magnitudes drive the pooled CV towards
    sqrt(n) and erase the between-group signal.

    ``min_models`` (default 2) excludes singleton cells. A (split, seed) cell
    whose Rashomon set holds one model has no cross-model spread. Its std is 0,
    so every group scores CV = 0. Averaging those zeros in treats them as
    measurements of perfect agreement. They are not measurements at all. The
    zeros also drag the group means down. The effect is severe where singletons
    dominate. AutoGluon Cable Demand is 9 of 12 cells. AutoGluon ETTm1 is 6 of
    9. AutoGluon ETTh1 and M4 Monthly are singletons in every cell, so they are
    not evaluable and return None. Restoring the guard moves AutoGluon Cable
    Demand's target-lag CV from 0.057 to 0.227. It moves H2O ETTh1's from 0.273
    to 0.351.

    Returns ``{group: (mean, std_across_cells)}``. With ``with_cells`` it
    returns a ``(dict, n_usable_cells, n_singleton_cells)`` triple.
    """
    group_order = list(group_order)
    path = RESULTS / run / "03_importance" / "raw_importance.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[np.isclose(df["eps"], eps_target)]
    if df.empty:
        return None

    if tree_exact_only:
        family = df["model"].str.split("_").str[0]
        df = df[family.isin(TREE_EXACT_FAMILIES)]
        if df.empty:
            return None

    df["group"] = df["feature"].apply(classify_feature)
    df = df[df["group"].isin(group_order)]

    records = []
    n_usable, n_singleton = 0, 0
    for (split_id, seed), grp in df.groupby(["split_id", "seed"]):
        # A cell with fewer than min_models models has no cross-model spread.
        # Its std is 0. That would enter the average as a spurious CV = 0.
        if grp["model"].nunique() < min_models:
            n_singleton += 1
            continue
        n_usable += 1

        fs = (grp.groupby(["feature", "group"])["global_importance"]
                 .agg(std=lambda x: x.std(ddof=0), mean="mean")
                 .reset_index())
        fs["cv"] = np.where(fs["mean"] > 0, fs["std"] / fs["mean"], np.nan)
        for g, sub in fs.groupby("group"):
            cv = sub["cv"].dropna()
            if len(cv):
                records.append({"group": g, "cv": cv.mean()})

    if not records:
        return (None, 0, n_singleton) if with_cells else None

    res = pd.DataFrame(records)
    out = {}
    for g in group_order:
        sub = res[res["group"] == g]["cv"].dropna()
        if len(sub):
            out[g] = (sub.mean(), sub.std(ddof=1) if len(sub) > 1 else 0.0)
    out = out or None
    return (out, n_usable, n_singleton) if with_cells else out
