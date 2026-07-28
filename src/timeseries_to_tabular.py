"""
Tabularisation: Convert panel time series to supervised learning format.

This transforms time-indexed panel data into feature matrices for AutoML
tabular models. It supports:
- Target lags
- Covariate lags
- Calendar features
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_supervised(
    panel: pd.DataFrame,
    horizon: int,
    target_lags: int,
    cov_lag: int,
) -> pd.DataFrame:
    """
    Convert panel data to supervised learning format.
    
    For each row at origin time t, the function creates a prediction problem:
    - Label: target at t+horizon
    - Features: target lags (1..target_lags), lagged covariates, calendar
    
    Args:
        panel: DataFrame with columns [item_id, timestamp, target, cov_0, ...]
        horizon: Prediction horizon (predict t+horizon)
        target_lags: Number of target lags to include (1..target_lags)
        cov_lag: Lag for covariates
    
    Returns:
        DataFrame ready for ML with columns:
            item_id, timestamp, label, target_lag1..lagN, 
            cov_X_lag{cov_lag}, month_sin, month_cos, year
    
    Example:
        >>> sup = build_supervised(panel, horizon=1, target_lags=6, cov_lag=1)
        >>> print(sup.columns.tolist()[:5])
        ['item_id', 'timestamp', 'label', 'target_lag1', 'target_lag2']
    """
    df = panel.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["item_id", "timestamp"]).copy()
    
    id_col = "item_id"
    time_col = "timestamp"
    y_col = "target"
    
    # Identify covariate columns
    cov_cols = [c for c in df.columns if c not in {id_col, time_col, y_col}]
    
    # Group by series for lagging
    g = df.groupby(id_col, sort=False)
    
    # Start output with identifiers
    out = df[[id_col, time_col]].copy()
    
    # Label: future target
    out["label"] = g[y_col].shift(-horizon)

    # Label timestamp: when the prediction actually occurs (t+horizon)
    # Critical for preventing horizon leakage in rolling-origin splits
    out["label_timestamp"] = g[time_col].shift(-horizon)
    
    # Target lags (using vectorised shift)
    for k in range(1, target_lags + 1):
        out[f"{y_col}_lag{k}"] = g[y_col].shift(k)
    
    # Covariate lags
    for c in cov_cols:
        out[f"{c}_lag{cov_lag}"] = g[c].shift(cov_lag)
    
    # Calendar features (cyclical encoding for month).
    # This uses label_timestamp (t+h). It encodes the FORECAST month, not the
    # observation month. Rationale: for horizon-1 forecasting, the forecast
    # timestamp is known at prediction time (no leakage). The "forecast month"
    # seasonality is the operationally relevant signal. For hourly/15-min data
    # (ETT, Electricity) the offset is negligible. label_timestamp and
    # origin_time share the same month/year. For monthly data (M4, Cable
    # Demand) the SHAP importance of month_sin/month_cos reflects the seasonal
    # effect of the TARGET period, not the observation period.
    ts = out["label_timestamp"]
    out["month_sin"] = np.sin(2.0 * np.pi * ts.dt.month / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * ts.dt.month / 12.0)
    out["year"] = ts.dt.year
    
    # Handle infinities
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows missing required columns (including label_timestamp for proper splitting)
    must_have = ["label", "label_timestamp"] + [f"{y_col}_lag{k}" for k in range(1, target_lags + 1)]
    out = out.dropna(subset=must_have).reset_index(drop=True)
    
    # Convert item_id to category for efficient storage
    out[id_col] = out[id_col].astype("category")
    
    return out


def get_feature_names(sup_df: pd.DataFrame) -> list[str]:
    """
    Extract feature column names from the supervised DataFrame.

    This excludes the identifiers (item_id, timestamp, label_timestamp) and the
    label.
    """
    exclude = {"item_id", "timestamp", "label", "label_timestamp"}
    return [c for c in sup_df.columns if c not in exclude]
