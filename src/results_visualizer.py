"""
Reporting and visualisation for Rashomon x SHAP experiments.

Visualizations include:
- Feature importance with uncertainty bands (bar chart)
- Stability over time (line plots)
- Importance distributions across Rashomon set (violin/beeswarm)
- Rank stability heatmap (features × splits)
- Cross-model agreement matrix (model × model correlation)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("rashomon_shap")

# Configure matplotlib style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

# Use a clean seaborn style for consistency
sns.set_style("whitegrid")


def plot_importance_bands(
    summary_df: pd.DataFrame,
    outpath: Path,
    split_id: int,
    seed: int,
    eps: float,
    topk: int = 12,
    q_low_name: str = "q10",
    q_high_name: str = "q90",
    title: Optional[str] = None,
) -> None:
    """Plot feature importance with uncertainty bands."""
    df = summary_df[
        (summary_df["split_id"] == split_id) &
        (summary_df["seed"] == seed) &
        (summary_df["eps"] == eps)
    ].copy()
    
    if df.empty:
        logger.warning(f"No data for split={split_id}, seed={seed}, eps={eps}")
        return
    
    df = df.sort_values("mean_importance", ascending=False).head(topk)
    df = df.sort_values("mean_importance", ascending=True)
    
    qcols = [c for c in df.columns if c.startswith("q") and c[1:].isdigit()]
    low = q_low_name if q_low_name in qcols else (sorted(qcols)[0] if qcols else None)
    high = q_high_name if q_high_name in qcols else (sorted(qcols)[-1] if qcols else None)
    
    means = df["mean_importance"].values
    features = df["feature"].astype(str).values
    
    if low and high:
        left_err = np.maximum(means - df[low].values, 0)
        right_err = np.maximum(df[high].values - means, 0)
        xerr = [left_err, right_err]
    else:
        xerr = None
    
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(df))))
    ax.barh(features, means, xerr=xerr, capsize=3, color="steelblue", alpha=0.8)
    ax.set_xlabel("Global SHAP Importance (mean |SHAP|)")
    ax.set_title(title or f"Rashomon SHAP (split={split_id}, seed={seed}, eps={eps})")
    ax.xaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_stability_over_time(
    stability: Dict,
    outpath: Path,
    eps: float,
    seed: int = 0,
) -> None:
    """Plot stability metrics across splits."""
    if not stability.get("consecutive"):
        return
    
    df = pd.DataFrame(stability["consecutive"])
    df = df[(df["eps"] == eps) & (df["seed"] == seed)]
    
    if df.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(df["split_b"], df["spearman"], "o-", color="blue")
    ax1.set_xlabel("Split")
    ax1.set_ylabel("Spearman Correlation")
    ax1.set_title("Rank Correlation Stability")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df["split_b"], df["topk_jaccard"], "o-", color="green")
    ax2.set_xlabel("Split")
    ax2.set_ylabel("Top-k Jaccard Similarity")
    ax2.set_title("Top Feature Overlap")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Phase 2: Within-Set Agreement Visualizations
# =============================================================================

def plot_importance_violin(
    dist_df: pd.DataFrame,
    outpath: Path,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> None:
    """
    Plot violin/beeswarm plot of importance distributions across Rashomon set.

    This plot shows the full distribution of importance values for each feature.
    It goes beyond summary statistics. This reveals:
    - Unimodal: Models agree on approximate importance
    - Bimodal: Two clusters with different strategies
    - Wide spread: High uncertainty

    Args:
        dist_df: DataFrame with columns: feature, model, importance
            (output from compute_importance_distribution_data)
        outpath: Path to save figure
        title: Optional plot title
        figsize: Figure dimensions

    Example:
        >>> from importance_aggregation import compute_importance_distribution_data
        >>> dist_df = compute_importance_distribution_data(imp_df, eps=0.05)
        >>> plot_importance_violin(dist_df, Path("figures/violin_eps0.05.png"))
    """
    if dist_df.empty:
        logger.warning("Empty DataFrame for violin plot")
        return

    feature_order = (
        dist_df.groupby('feature')['importance']
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Create violin plot with individual points
    # Use seaborn for nicer aesthetics
    sns.violinplot(
        data=dist_df,
        x='importance',
        y='feature',
        order=feature_order,
        inner=None,  # No inner representation
        color='steelblue',
        alpha=0.3,
        ax=ax
    )

    # Overlay individual points (beeswarm-like)
    sns.stripplot(
        data=dist_df,
        x='importance',
        y='feature',
        order=feature_order,
        color='darkblue',
        alpha=0.6,
        size=4,
        jitter=True,
        ax=ax
    )

    # Add mean markers
    means = dist_df.groupby('feature')['importance'].mean().reindex(feature_order)
    for i, (feat, mean_val) in enumerate(means.items()):
        ax.plot(mean_val, i, 'r|', markersize=12, markeredgewidth=2)

    ax.set_xlabel("SHAP Importance (mean |SHAP|)")
    ax.set_ylabel("")
    ax.set_title(title or "Feature Importance Distribution Across Rashomon Set")

    # Add legend for mean marker
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='|', color='r', linestyle='None',
               markersize=10, markeredgewidth=2, label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved violin plot to {outpath}")


def plot_rank_stability_heatmap(
    rank_matrix: pd.DataFrame,
    outpath: Path,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
) -> None:
    """
    Plot heatmap showing feature ranks across temporal splits.

    This heatmap shows which features keep stable rankings. It also shows which features change.
    Color intensity shows rank (darker = higher rank = more important).

    Args:
        rank_matrix: DataFrame with features as rows, splits as columns.
            (output from compute_rank_stability_matrix)
        outpath: Path to save figure
        title: Optional plot title
        figsize: Figure dimensions

    Interpretation:
        - Consistent color across row: Stable feature
        - Color changes across row: Unstable feature
        - Row with mostly dark colors: Consistently important
    """
    if rank_matrix.empty:
        logger.warning("Empty DataFrame for rank heatmap")
        return

    # Select only split columns for heatmap
    split_cols = [c for c in rank_matrix.columns if c.startswith('split_')]
    if not split_cols:
        logger.warning("No split columns found in rank matrix")
        return

    # Prepare data matrix
    data = rank_matrix[split_cols].copy()

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap with inverted colormap (low rank = important = dark)
    # Use diverging colormap centered around median rank
    n_features = len(data)
    vmax = n_features

    # Create custom colormap: dark blue (important) to light (unimportant)
    cmap = sns.color_palette("Blues_r", as_cmap=True)

    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt='.0f',
        vmin=1,
        vmax=min(20, vmax),  # Cap at 20 for readability
        cbar_kws={'label': 'Feature Rank (1 = most important)'},
        linewidths=0.5,
        linecolor='white'
    )

    ax.set_xlabel("Temporal Split")
    ax.set_ylabel("Feature")
    ax.set_title(title or "Feature Rank Stability Across Splits")

    # Add stability metrics annotation
    if 'rank_std' in rank_matrix.columns:
        for i, (idx, row) in enumerate(rank_matrix.iterrows()):
            # Add std annotation on right side
            ax.text(
                len(split_cols) + 0.3, i + 0.5,
                f"σ={row['rank_std']:.1f}",
                va='center', fontsize=8, color='gray'
            )

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved rank stability heatmap to {outpath}")


def plot_epsilon_stability_comparison(
    epsilon_df: pd.DataFrame,
    outpath: Path,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> None:
    """
    Plot stability metrics across epsilon thresholds.

    This figure shows how Spearman ρ, Kendall τ, and Jaccard change with tolerance.
    It helps answer this question: "Does a larger Rashomon set reduce stability?"

    Args:
        epsilon_df: DataFrame with stability metrics per epsilon.
            (output from compute_epsilon_sensitivity)
        outpath: Path to save figure
        title: Optional plot title
        figsize: Figure dimensions
    """
    if epsilon_df.empty:
        logger.warning("Empty epsilon sensitivity DataFrame")
        return

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Spearman correlation
    ax1 = axes[0]
    ax1.bar(
        range(len(epsilon_df)),
        epsilon_df['avg_spearman'],
        yerr=epsilon_df['std_spearman'],
        capsize=4,
        color='steelblue',
        alpha=0.7
    )
    ax1.set_xticks(range(len(epsilon_df)))
    ax1.set_xticklabels([f"ε={e:.2f}" for e in epsilon_df['epsilon']])
    ax1.set_ylabel("Spearman ρ")
    ax1.set_title("Rank Correlation")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='High stability')
    ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Moderate')

    # Plot 2: Kendall tau (if available)
    ax2 = axes[1]
    if 'avg_kendall' in epsilon_df.columns:
        ax2.bar(
            range(len(epsilon_df)),
            epsilon_df['avg_kendall'],
            yerr=epsilon_df.get('std_kendall', 0),
            capsize=4,
            color='teal',
            alpha=0.7
        )
        ax2.set_ylabel("Kendall τ")
        ax2.set_title("Concordance")
    else:
        # Fall back to Jaccard
        ax2.bar(
            range(len(epsilon_df)),
            epsilon_df['avg_jaccard'],
            yerr=epsilon_df['std_jaccard'],
            capsize=4,
            color='teal',
            alpha=0.7
        )
        ax2.set_ylabel("Top-k Jaccard")
        ax2.set_title("Top Feature Overlap")
    ax2.set_xticks(range(len(epsilon_df)))
    ax2.set_xticklabels([f"ε={e:.2f}" for e in epsilon_df['epsilon']])
    ax2.set_ylim(0, 1.05)

    # Plot 3: Jaccard
    ax3 = axes[2]
    ax3.bar(
        range(len(epsilon_df)),
        epsilon_df['avg_jaccard'],
        yerr=epsilon_df['std_jaccard'],
        capsize=4,
        color='coral',
        alpha=0.7
    )
    ax3.set_xticks(range(len(epsilon_df)))
    ax3.set_xticklabels([f"ε={e:.2f}" for e in epsilon_df['epsilon']])
    ax3.set_ylabel("Top-k Jaccard")
    ax3.set_title("Top Feature Overlap")
    ax3.set_ylim(0, 1.05)

    plt.suptitle(title or "Stability Metrics vs. Rashomon Tolerance (ε)")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved epsilon stability comparison to {outpath}")
