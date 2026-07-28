"""
Rashomon set construction from AutoML leaderboard metrics.

This implements the thesis inclusion criterion (eq:rashomon):
R(eps) = { m : MAE_val(m) <= MAE_val(m*) x (1 + eps) }.
Here m* is the best model on the validation split (Fisher et al., 2019;
Marx et al., 2020). The set is empirical. The code filters it from the models
that the AutoML search actually trained. It therefore approximates the
theoretical Rashomon set rather than enumerating it.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def rashomon_sets_from_val_metrics(
    val_metrics: pd.DataFrame,
    eps_list: List[float],
    max_models: int,
    metric_col: str = "mae",
    lower_is_better: bool = True,
) -> Dict[float, pd.DataFrame]:
    """
    Construct Rashomon sets at multiple epsilon thresholds.
    
    A Rashomon set R(ε) contains models m where:
        metric(m) ≤ best_metric × (1 + ε)  [if lower is better]
        metric(m) ≥ best_metric × (1 - ε)  [if higher is better]
    
    Args:
        val_metrics: DataFrame with 'model' and metric columns
        eps_list: List of epsilon thresholds (e.g., [0.01, 0.02, 0.05])
        max_models: Maximum models to include per set
        metric_col: Column name for the metric
        lower_is_better: True for metrics like MAE, False for accuracy
    
    Returns:
        Dictionary mapping epsilon -> DataFrame of models in that set
    
    Example:
        >>> sets = rashomon_sets_from_val_metrics(
        ...     val_metrics, eps_list=[0.01, 0.05], max_models=20
        ... )
        >>> len(sets[0.05])  # Models within 5% of best
        15
    """
    if val_metrics.empty:
        return {eps: pd.DataFrame() for eps in eps_list}
    
    best = val_metrics[metric_col].min() if lower_is_better else val_metrics[metric_col].max()
    
    out = {}
    for eps in eps_list:
        eps_f = float(eps)
        
        if lower_is_better:
            # Thesis eq:rashomon: MAE_val(m) <= MAE_val(m*) x (1 + eps).
            thresh = best * (1.0 + eps_f)
            subset = val_metrics[val_metrics[metric_col] <= thresh].copy()
            subset = subset.sort_values(metric_col, ascending=True)
        else:
            thresh = best * (1.0 - eps_f)
            subset = val_metrics[val_metrics[metric_col] >= thresh].copy()
            subset = subset.sort_values(metric_col, ascending=False)
        
        subset = subset.head(max_models).reset_index(drop=True)
        out[eps_f] = subset
    
    return out


def get_union_models(
    sets: Dict[float, pd.DataFrame],
    model_col: str = "model"
) -> List[str]:
    """
    Get union of all models across all Rashomon sets.
    
    Useful for computing SHAP once for all models that might be
    in any Rashomon set.
    """
    all_models = set()
    for df in sets.values():
        if not df.empty:
            all_models.update(df[model_col].tolist())
    return sorted(all_models)
