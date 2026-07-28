"""
Stability metrics for SHAP-based feature importance rankings.

This module computes the three pairwise stability metrics of the thesis
(ch. 3.8). These are Spearman rho (eq:spearman), Kendall tau-b (eq:kendall),
and top-k Jaccard similarity (eq:jaccard). It evaluates them across consecutive
temporal splits and across epsilon thresholds.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap as scipy_bootstrap

from importance_aggregation import (  # noqa: F401  (re-export kept for old callers)
    NON_FEATURE_COLUMNS,
    drop_non_feature_rows,
    compute_rank_stability_matrix,
)


# =============================================================================
# Core Correlation Functions
# =============================================================================

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-12
    return float((a * b).sum() / denom)


def spearman_from_ranks(rank_a: pd.Series, rank_b: pd.Series) -> float:
    """
    Return the Spearman rank correlation of two rank series, in [-1, 1].

    Thesis eq:spearman: rho = Corr(rank(a), rank(b)). The function computes it
    as the Pearson correlation of the rank vectors. A feature may be present in
    one ranking only. The function then assigns it rank max + 1 in the other
    ranking. Disappearing features therefore count as disagreement. The
    function does not drop them silently.
    """
    idx = rank_a.index.union(rank_b.index)
    a = rank_a.reindex(idx).fillna(rank_a.max() + 1).values
    b = rank_b.reindex(idx).fillna(rank_b.max() + 1).values
    return _pearson(a, b)


def kendall_tau_from_ranks(rank_a: pd.Series, rank_b: pd.Series) -> Tuple[float, float]:
    """
    Return (tau-b, p-value) for two rank series.

    Thesis eq:kendall: tau_b = (C - D) / sqrt((C + D + T_a)(C + D + T_b)). Here
    C and D count the concordant and discordant pairs. T_a and T_b count the
    ties in each ranking (Kendall, 1945). The function handles missing features
    as in spearman_from_ranks.
    """
    idx = rank_a.index.union(rank_b.index)
    a = rank_a.reindex(idx).fillna(rank_a.max() + 1).values
    b = rank_b.reindex(idx).fillna(rank_b.max() + 1).values
    tau, pval = stats.kendalltau(a, b)
    return float(tau), float(pval)


def topk_jaccard(
    rank_a: pd.Series,
    rank_b: pd.Series,
    k: int | None = None,
    top_percent: float = 0.30,
) -> float:
    """
    Return the Jaccard similarity of the top-k feature sets, in [0, 1].

    Thesis eq:jaccard: J(A, B) = |A intersect B| / |A union B|. Here A and B
    are the top-k features of each ranking. k = round(0.30 x p), where p is the
    number of features (tab:jaccard_k: M4 Monthly p=11 -> k=3, Electricity
    p=13 -> k=4, ETT p=18 -> k=5, Cable Demand p=40 -> k=12). The rankings must
    contain features only. The caller must exclude bookkeeping columns such as
    item_id (see NON_FEATURE_COLUMNS). Otherwise k inflates.

    Args:
        rank_a: First ranking (index=feature, values=rank)
        rank_b: Second ranking
        k: Fixed number of top features. When None, uses top_percent instead.
        top_percent: Fraction of features to use when k is None (default 0.30).
    """
    if k is None:
        # Top 30% of the smaller feature set. This is symmetric. k does not
        # depend on which ranking is passed as rank_a or rank_b.
        k = max(1, round(top_percent * min(len(rank_a), len(rank_b))))
    a_top = set(rank_a.nsmallest(k).index)
    b_top = set(rank_b.nsmallest(k).index)
    intersection = len(a_top & b_top)
    union = len(a_top | b_top)
    return float(intersection / (union + 1e-12))


# =============================================================================
# Temporal Stability Analysis
# =============================================================================

def stability_over_splits(
    rank_df: pd.DataFrame,
    k: int | None = None,
) -> Dict[str, Any]:
    """
    Compute stability metrics across consecutive temporal splits.

    This measures how consistent feature rankings are over time. It uses
    several metrics:
    - Spearman rho: Rank correlation (monotonic relationship)
    - Kendall tau: Concordance, the fraction of pairwise agreements (more robust)
    - Top-k Jaccard: Overlap of the most important features

    By default, Jaccard uses the top 30% of features (k=None).

    Args:
        rank_df: DataFrame with columns: split_id, seed, eps, feature, rank
        k: Fixed top-k for Jaccard. When None, uses top 30% of feature count.

    Returns:
        Dictionary with 'consecutive' list containing per-pair metrics
    """
    # item_id is a series identifier. The thesis defines k over the p tabular
    # features (tab:jaccard_k). AutoGluon models receive this column. SHAP
    # produces an attribution for it. The function must therefore exclude
    # item_id here at the metric layer. Keeping it inflates k. It also lets
    # item_id occupy top-k slots.
    rank_df = drop_non_feature_rows(rank_df)

    out = {"consecutive": []}

    for (seed, eps), sub in rank_df.groupby(["seed", "eps"]):
        splits = sorted(sub["split_id"].unique())
        rank_map = {}
        for s in splits:
            r = sub[sub["split_id"] == s].set_index("feature")["rank"]
            rank_map[s] = r

        for a, b in zip(splits[:-1], splits[1:]):
            sp = spearman_from_ranks(rank_map[a], rank_map[b])
            tau, tau_pval = kendall_tau_from_ranks(rank_map[a], rank_map[b])
            jac = topk_jaccard(rank_map[a], rank_map[b], k=k)

            out["consecutive"].append({
                "seed": int(seed),
                "eps": float(eps),
                "split_a": int(a),
                "split_b": int(b),
                "spearman": sp,
                "kendall_tau": tau,
                "kendall_pval": tau_pval,
                "topk_jaccard": jac,
            })

    return out


def _bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    rng: int = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (ci_low, ci_high) bootstrap CI, or (None, None) if n < 2."""
    if len(values) < 2:
        return None, None
    result = scipy_bootstrap(
        (values,),
        statistic,
        n_resamples=n_resamples,
        confidence_level=confidence,
        random_state=rng,
        method="percentile",
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def compute_stability_summary(stability: Dict[str, Any]) -> pd.DataFrame:
    """
    Summarise stability metrics across all comparisons.

    This aggregates Spearman rho, Kendall tau, and Jaccard similarity across
    splits. It includes 95% bootstrap confidence intervals for the Spearman rho
    mean.

    Args:
        stability: Output from stability_over_splits()

    Returns:
        DataFrame with mean/std/CI for each (seed, eps) combination
    """
    if not stability.get("consecutive"):
        return pd.DataFrame()

    df = pd.DataFrame(stability["consecutive"])
    agg_dict = {
        "spearman": ["mean", "std", "min", "max"],
        "topk_jaccard": ["mean", "std", "min", "max"],
    }

    if "kendall_tau" in df.columns:
        agg_dict["kendall_tau"] = ["mean", "std", "min", "max"]

    summary = df.groupby(["seed", "eps"]).agg(agg_dict).reset_index()
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    ci_rows = []
    for (seed, eps), grp in df.groupby(["seed", "eps"]):
        lo, hi = _bootstrap_ci(grp["spearman"].values)
        ci_rows.append({"seed": seed, "eps": eps, "spearman_ci95_low": lo, "spearman_ci95_high": hi})
    ci_df = pd.DataFrame(ci_rows)
    summary = summary.merge(ci_df, on=["seed", "eps"], how="left")

    return summary


# =============================================================================
# Grouped/Cross-Entity Stability
# =============================================================================

def compute_grouped_stability(
    rank_df: pd.DataFrame,
    group_col: str = "item_id",
    panel_df: pd.DataFrame = None,
    k: int | None = None,
) -> Dict[str, Any]:
    """
    Compute stability metrics within and across groups (e.g., plants, regions).

    This analyses whether feature importance rankings are more stable within
    groups than across groups. It is useful for multi-entity time series.

    By default, Jaccard uses the top 30% of features (k=None).

    Args:
        rank_df: DataFrame with columns: split_id, seed, eps, feature, rank
        group_col: Column name identifying groups
        panel_df: Optional panel data with group assignments
        k: Fixed top-k for Jaccard. When None, uses top 30% of feature count.

    Returns:
        Dictionary with within_group, across_groups, and group_comparison metrics
    """
    results = {
        "within_group": [],
        "across_groups": {},
        "group_comparison": {}
    }

    if group_col not in rank_df.columns and panel_df is None:
        overall = stability_over_splits(rank_df, k=k)
        if overall.get("consecutive"):
            consecutive = pd.DataFrame(overall["consecutive"])
            results["across_groups"] = {
                "avg_spearman": float(consecutive["spearman"].mean()),
                "avg_jaccard": float(consecutive["topk_jaccard"].mean()),
                "std_spearman": float(consecutive["spearman"].std()),
                "n_comparisons": len(consecutive)
            }
        return results

    groups = rank_df[group_col].unique() if group_col in rank_df.columns else []
    within_spearman = []
    within_jaccard = []

    for group in groups:
        group_data = rank_df[rank_df[group_col] == group]
        if len(group_data) < 2:
            continue

        group_stability = stability_over_splits(group_data, k=k)
        if group_stability.get("consecutive"):
            for item in group_stability["consecutive"]:
                item["group"] = group
                results["within_group"].append(item)
                within_spearman.append(item["spearman"])
                within_jaccard.append(item["topk_jaccard"])

    overall = stability_over_splits(rank_df, k=k)
    if overall.get("consecutive"):
        consecutive = pd.DataFrame(overall["consecutive"])
        results["across_groups"] = {
            "avg_spearman": float(consecutive["spearman"].mean()),
            "avg_jaccard": float(consecutive["topk_jaccard"].mean()),
            "std_spearman": float(consecutive["spearman"].std()),
            "n_comparisons": len(consecutive)
        }

    if within_spearman:
        results["group_comparison"] = {
            "within_avg_spearman": float(np.mean(within_spearman)),
            "within_std_spearman": float(np.std(within_spearman)),
            "within_avg_jaccard": float(np.mean(within_jaccard)),
            "across_avg_spearman": results["across_groups"].get("avg_spearman", np.nan),
            "across_avg_jaccard": results["across_groups"].get("avg_jaccard", np.nan),
            "within_more_stable": float(np.mean(within_spearman)) > results["across_groups"].get("avg_spearman", 0)
        }

    return results


# =============================================================================
# Epsilon Sensitivity Analysis
# =============================================================================

def compute_epsilon_sensitivity(
    rank_df: pd.DataFrame,
    eps_list: List[float],
    k: int | None = None,
) -> pd.DataFrame:
    """
    Analyse how stability metrics change across different epsilon thresholds.

    This shows the trade-off between Rashomon set size and explanation
    stability.

    By default, Jaccard uses the top 30% of features (k=None).

    Args:
        rank_df: DataFrame with ranks at different epsilon values
        eps_list: List of epsilon values to analyse
        k: Fixed top-k for Jaccard. When None, uses top 30% of feature count.

    Returns:
        DataFrame with stability metrics per epsilon
    """
    results = []

    for eps in eps_list:
        eps_data = rank_df[rank_df['eps'] == eps]
        if len(eps_data) == 0:
            continue

        stability = stability_over_splits(eps_data, k=k)
        if not stability.get("consecutive"):
            continue

        consecutive = pd.DataFrame(stability["consecutive"])
        n_models = eps_data.groupby(['split_id', 'seed'])['feature'].count().values

        result_dict = {
            'epsilon': eps,
            'avg_spearman': float(consecutive['spearman'].mean()),
            'std_spearman': float(consecutive['spearman'].std()),
            'avg_jaccard': float(consecutive['topk_jaccard'].mean()),
            'std_jaccard': float(consecutive['topk_jaccard'].std()),
            'n_comparisons': len(consecutive),
            'avg_features': float(np.mean(n_models)) if len(n_models) > 0 else 0
        }

        if 'kendall_tau' in consecutive.columns:
            result_dict['avg_kendall'] = float(consecutive['kendall_tau'].mean())
            result_dict['std_kendall'] = float(consecutive['kendall_tau'].std())

        results.append(result_dict)

    return pd.DataFrame(results)

