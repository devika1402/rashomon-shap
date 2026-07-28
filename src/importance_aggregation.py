"""
Aggregation of per-model SHAP importances across the Rashomon set.

This module computes the per-feature spread metrics of the thesis (ch. 3.8).
These are SHAP-range (eq:shap_range) and SHAP-CV (eq:shap_cv). It reports them
as Model Class Reliance (MCR) style bounds. It also computes ranks and
distribution data for plotting.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# item_id identifies which series a row belongs to. It is bookkeeping only.
# The thesis feature counts (tab:jaccard_k) are defined over the p tabular
# features only. AutoGluon models receive this column. SHAP produces an
# attribution for it. Every metric in this module must drop item_id before it
# aggregates or ranks.
NON_FEATURE_COLUMNS = ("item_id",)


def drop_non_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return df without rows whose 'feature' is a bookkeeping column."""
    if "feature" not in df.columns:
        return df
    return df[~df["feature"].isin(NON_FEATURE_COLUMNS)]


# =============================================================================
# Global Importance Aggregation
# =============================================================================

def aggregate_global_importances(
    global_imp_df: pd.DataFrame,
    quantiles: List[float],
) -> pd.DataFrame:
    """
    Aggregate global SHAP importances across models in Rashomon sets.

    This computes per-feature statistics: the mean, quantiles, and coefficient
    of variation. These statistics quantify the uncertainty.

    Args:
        global_imp_df: DataFrame with columns:
            split_id, seed, eps, model, feature, global_importance
        quantiles: Three quantiles for uncertainty bands (e.g., [0.1, 0.5, 0.9])

    Returns:
        DataFrame with aggregated statistics per (split_id, seed, eps, feature)
    """
    q = sorted([float(x) for x in quantiles])
    if len(q) != 3:
        raise ValueError("Provide exactly 3 quantiles (e.g., [0.1, 0.5, 0.9])")

    global_imp_df = drop_non_feature_rows(global_imp_df)
    g = global_imp_df.groupby(["split_id", "seed", "eps", "feature"])["global_importance"]

    mean = g.mean()
    std = g.std(ddof=1)
    q0 = g.quantile(q[0])
    q1 = g.quantile(q[1])
    q2 = g.quantile(q[2])
    count = g.count()

    # Thesis eq:shap_cv: SHAP-CV_j = sigma(I_j) / (|mean(I_j)| + 1e-10).
    # This is the coefficient of variation of mean-absolute-SHAP importance
    # across models in the Rashomon set.
    rdi_cv = std / (mean.abs() + 1e-10)

    out = pd.DataFrame({
        "mean_importance": mean,
        f"q{int(q[0]*100)}": q0,
        f"q{int(q[1]*100)}": q1,
        f"q{int(q[2]*100)}": q2,
        "rdi_cv": rdi_cv,
        "n_models": count,
    }).reset_index()

    return out


def mean_rank_per_split(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute feature ranks within each (split_id, seed, eps) context.

    The function ranks features by descending mean_importance. Rank 1 is the
    most important feature.

    WARNING (scale sensitivity). This function ranks the MEAN OF RAW SHAP
    MAGNITUDES across the models of a Rashomon set. Mean-absolute-SHAP values
    are comparable across models only when those models share an explainer.
    H2O's DRF/XRT models use Saabas path attributions. These attributions are
    unbounded. On Electricity, a single XRT model can exceed the exact-TreeSHAP
    families by many orders of magnitude. That one model therefore determines the arithmetic
    mean and this ranking. Use rank_then_mean_per_split() whenever a Rashomon
    set mixes explainers. This function remains here to reproduce the
    magnitude-weighted aggregation.

    Args:
        summary_df: Output from aggregate_global_importances()

    Returns:
        DataFrame with added 'rank' column
    """
    df = summary_df.copy()
    df["rank"] = df.groupby(["split_id", "seed", "eps"])["mean_importance"].rank(
        ascending=False,
        method="average"
    )
    return df


def rank_then_mean_per_split(global_imp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale-invariant feature ranking across the models of a Rashomon set.

    The function ranks each model's features first. Rank 1 is the most
    important feature for that model. It then averages the ranks across models
    within each (split_id, seed, eps) context. The ranking is applied per
    model. A model whose attributions use a different numeric scale
    contributes its ordering and nothing more. No single model can therefore
    dominate the aggregate.

    This is the aggregation the thesis reports for stability metrics. It is the
    scale-safe counterpart of mean_rank_per_split(). The two agree when the
    Rashomon set uses a single explainer, as under AutoGluon. They diverge when
    the set mixes explainers, as under H2O AutoML.

    Args:
        global_imp_df: DataFrame with columns:
            split_id, seed, eps, model, feature, global_importance

    Returns:
        DataFrame with columns split_id, seed, eps, feature, mean_rank, rank,
        n_models. 'rank' re-ranks the averaged ranks. Rank 1 is the most
        important feature. This matches the contract of mean_rank_per_split().
    """
    df = drop_non_feature_rows(global_imp_df)

    # Rank features WITHIN each model. Descending: rank 1 = most important.
    df = df.copy()
    df["within_model_rank"] = df.groupby(
        ["split_id", "seed", "eps", "model"]
    )["global_importance"].rank(ascending=False, method="average")

    g = df.groupby(["split_id", "seed", "eps", "feature"])["within_model_rank"]
    out = pd.DataFrame({
        "mean_rank": g.mean(),
        "n_models": g.count(),
    }).reset_index()

    # Re-rank the averaged ranks: lowest mean_rank = most important = rank 1.
    out["rank"] = out.groupby(["split_id", "seed", "eps"])["mean_rank"].rank(
        ascending=True,
        method="average"
    )
    return out


def compute_importance_distribution_data(
    importance_df: pd.DataFrame,
    split_id: int = 0,
    seed: int = 0,
    eps: float = 0.05,
    top_k: int = 15
) -> pd.DataFrame:
    """
    Prepare data for violin/beeswarm plots of importance distributions.

    This returns per-model importance values for the top features. The values
    plot the full distribution rather than summary statistics alone.

    This shows whether importance distributions are:
    - Unimodal: Models agree on approximate importance
    - Bimodal: Two clusters of models with different strategies
    - Uniform: High uncertainty, no consensus

    Args:
        importance_df: DataFrame with per-model importance values
        split_id: Temporal split
        seed: Random seed
        eps: Rashomon tolerance
        top_k: Number of top features to include

    Returns:
        DataFrame with columns: feature, model, importance
    """
    df = drop_non_feature_rows(importance_df)
    df = df[
        (df['split_id'] == split_id) &
        (df['seed'] == seed) &
        (df['eps'] == eps)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    mean_importance = df.groupby('feature')['global_importance'].mean()
    top_features = mean_importance.nlargest(top_k).index.tolist()

    result = df[df['feature'].isin(top_features)][
        ['feature', 'model', 'global_importance']
    ].rename(columns={'global_importance': 'importance'})

    result['feature'] = pd.Categorical(
        result['feature'],
        categories=top_features,
        ordered=True
    )
    result = result.sort_values('feature')

    return result


# =============================================================================
# Rank Stability Matrix
# =============================================================================

def compute_rank_stability_matrix(
    rank_df: pd.DataFrame,
    top_k: int = 15,
    eps: float = 0.05,
    seed: int = 0
) -> pd.DataFrame:
    """
    Compute rank stability matrix for heatmap visualisation.

    This creates a matrix. Rows are the top features and columns are the
    splits. Each cell value shows the feature's rank in that split.

    Args:
        rank_df: DataFrame with columns: split_id, seed, eps, feature, rank
        top_k: Number of top features to include (by average rank)
        eps: Epsilon threshold to analyse
        seed: Random seed to analyse

    Returns:
        DataFrame with features as rows, splits as columns, ranks as values
    """
    df = drop_non_feature_rows(rank_df)
    df = df[
        (df['seed'] == seed) &
        (df['eps'] == eps)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    splits = sorted(df['split_id'].unique())
    avg_rank = df.groupby('feature')['rank'].mean()
    top_features = avg_rank.nsmallest(top_k).index.tolist()

    matrix_data = {}
    for split in splits:
        split_data = df[df['split_id'] == split].set_index('feature')['rank']
        matrix_data[f'split_{split}'] = split_data.reindex(top_features)

    matrix = pd.DataFrame(matrix_data, index=top_features)
    matrix['mean_rank'] = matrix.mean(axis=1)
    split_cols = [c for c in matrix.columns if c.startswith('split_')]
    matrix['rank_std'] = matrix[split_cols].std(axis=1)
    matrix['rank_range'] = matrix[split_cols].max(axis=1) - matrix[split_cols].min(axis=1)
    matrix = matrix.sort_values('mean_rank')

    return matrix
