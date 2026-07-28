"""
AutoGluon training wrapper with automatic hardware detection.

Handles:
- GPU/CPU auto-detection
- Mac (MPS) compatibility
- Model caching and resumption
- Consistent metric computation
"""
from __future__ import annotations

import gc
import inspect
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# =============================================================================
# Suppress verbose warnings from AutoGluon and its dependencies
# =============================================================================
warnings.filterwarnings("ignore", message=".*load_learner.*pickle.*")
warnings.filterwarnings("ignore", category=UserWarning, module="fastai")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Import AutoGluon with suppressed warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from autogluon.tabular import TabularPredictor

logger = logging.getLogger("rashomon_shap")

Num = Union[int, str]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))



def encode_non_numeric(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert non-numeric columns to integer codes with consistent mapping.
    
    The function uses one shared category mapping across train, val, and
    test. This gives consistent encoding. AutoGluon can handle categoricals
    by itself. This step still makes the pipeline fully numeric. A fully
    numeric pipeline improves cross-model compatibility.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        label: Name of label column to exclude
    
    Returns:
        Tuple of encoded (train, val, test) DataFrames
    """
    dfs = [train_df.copy(), val_df.copy(), test_df.copy()]
    feature_cols = [c for c in train_df.columns if c != label]
    
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(train_df[c]):
            # Fit the category mapping on the training data only. Unseen
            # categories in val or test get code -1. This is the pandas
            # default for unknown categories.
            cats = pd.Categorical(dfs[0][c].astype("category")).categories

            # Apply consistent encoding
            for d in dfs:
                d[c] = pd.Categorical(d[c], categories=cats).codes.astype("int32")
    
    return dfs[0], dfs[1], dfs[2]


def train_one_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_dir: Path,
    time_limit_s: int,
    presets: str,
    seed: int,
    eval_metric: str = "mae",
    num_gpus: Num = "auto",
    num_cpus: Num = "auto",
    fit_strategy: str = "sequential",
    force_retrain: bool = False,
    verbosity: int = 2,
    excluded_model_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Train AutoGluon models for one split with smart resource handling.
    
    Features:
    - Automatic model caching and resumption
    - GPU/CPU/MPS detection
    - Consistent metric computation
    
    Args:
        train_df: Training data with 'label' column
        val_df: Validation data
        test_df: Test data
        model_dir: Directory to save/load models
        time_limit_s: Training time limit in seconds
        presets: AutoGluon presets ('best_quality', 'high_quality', 'medium_quality', etc.)
        seed: Random seed
        eval_metric: Metric for evaluation ('mae', 'rmse', etc.)
        num_gpus: Number of GPUs (int or 'auto')
        num_cpus: Number of CPUs (int or 'auto')
        fit_strategy: 'sequential' or 'parallel'
        force_retrain: If True, ignore cached models
        verbosity: AutoGluon verbosity level (0-4)
        excluded_model_types: Model types to exclude from training
    
    Returns:
        Dictionary containing:
            - predictor: Trained AutoGluon predictor
            - leaderboard: Model leaderboard DataFrame
            - val_metrics: Per-model validation metrics
            - test_metrics: Per-model test metrics
            - train_used, val_used, test_used: Processed DataFrames
    """
    label = "label"

    # Drop the timestamp columns to avoid time-index exploitation.
    # It is important to drop label_timestamp. It is a future-derived column.
    drop_cols = ["timestamp", "label_timestamp"]
    train2 = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    val2 = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])
    test2 = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    
    # Encode non-numeric features
    train2, val2, test2 = encode_non_numeric(train2, val2, test2, label=label)
    
    model_dir.mkdir(parents=True, exist_ok=True)
    has_val = val2 is not None and not val2.empty

    predictor = None
    
    # Try to resume from cached models
    if not force_retrain:
        try:
            if any(model_dir.iterdir()):
                predictor = TabularPredictor.load(str(model_dir))
                logger.info(f"Loaded cached predictor from {model_dir}")
        except Exception as e:
            logger.debug(f"Could not load cached predictor: {e}")
            predictor = None
    
    # Train if no cached predictor
    if predictor is None:
        predictor = TabularPredictor(
            label=label,
            eval_metric=eval_metric,
            path=str(model_dir),
            verbosity=verbosity,
        )
        
        # Build fit kwargs dynamically based on AutoGluon version
        fit_sig = inspect.signature(predictor.fit).parameters
        fit_kwargs: Dict[str, Any] = {}
        ag_args_fit: Dict[str, Any] = {"random_seed": seed}
        use_bag_holdout_supported = "use_bag_holdout" in fit_sig
        
        # Handle GPU settings
        if "num_gpus" in fit_sig:
            fit_kwargs["num_gpus"] = num_gpus if num_gpus != "auto" else None
        elif num_gpus != "auto" and num_gpus > 0:
            ag_args_fit["num_gpus"] = num_gpus
        
        # Handle CPU settings
        if "num_cpus" in fit_sig:
            fit_kwargs["num_cpus"] = num_cpus if num_cpus != "auto" else None
        elif num_cpus != "auto":
            ag_args_fit["num_cpus"] = num_cpus
        
        # Fit strategy (AutoGluon >= 1.5)
        if "fit_strategy" in fit_sig:
            fit_kwargs["fit_strategy"] = fit_strategy
        
        # Excluded models
        if excluded_model_types:
            fit_kwargs["excluded_model_types"] = excluded_model_types
        
        train_for_fit = train2
        tuning_for_fit = val2 if has_val else None

        if has_val:
            if use_bag_holdout_supported:
                fit_kwargs["use_bag_holdout"] = True
            else:
                logger.warning(
                    "use_bag_holdout not supported by this AutoGluon version; "
                    "merging val into train. Validation MAE will not reflect a held-out set."
                )
                train_for_fit = pd.concat([train2, val2], ignore_index=True)
                tuning_for_fit = None
        
        # Disable DyStack: it deadlocks on shared-node Ray cluster setups
        if "dynamic_stacking" in fit_sig:
            fit_kwargs["dynamic_stacking"] = False

        # Ray worker registration deadlocks on shared-node cluster setups.
        # Hide ray from sys.modules during fit(). Then AutoGluon falls back to
        # SequentialLocalFoldFittingStrategy. This is the same path that works
        # on Mac. Set AUTOGLUON_USE_RAY=1 to skip this workaround.
        import sys
        _blocked_ray: Dict[str, Any] = {}
        if os.environ.get("AUTOGLUON_USE_RAY", "0") != "1":
            for _k in list(sys.modules.keys()):
                if _k == "ray" or _k.startswith("ray."):
                    _blocked_ray[_k] = sys.modules.pop(_k)
            sys.modules["ray"] = None  # type: ignore[assignment]
            logger.info("Ray hidden; sequential bag fitting with full CPU/GPU per fold")

        logger.info(
            f"Training AutoGluon: time_limit={time_limit_s}s, "
            f"presets={presets}, seed={seed}"
        )

        try:
            predictor.fit(
                train_data=train_for_fit,
                tuning_data=tuning_for_fit,
                time_limit=time_limit_s,
                presets=presets,
                ag_args_fit=ag_args_fit,
                **fit_kwargs,
            )
        finally:
            if _blocked_ray:
                sys.modules.pop("ray", None)
                sys.modules.update(_blocked_ray)
        
        # Clean up memory
        gc.collect()
    
    lb = predictor.leaderboard(val2, silent=True)
    
    y_val = val2[label].values
    y_test = test2[label].values
    models = predictor.model_names()
    
    val_rows, test_rows = [], []
    
    for m in models:
        try:
            pred_val = predictor.predict(val2, model=m).values
            pred_test = predictor.predict(test2, model=m).values
            
            val_rows.append({
                "model": m,
                "mae": mae(y_val, pred_val),
                "rmse": rmse(y_val, pred_val),
            })
            test_rows.append({
                "model": m,
                "mae": mae(y_test, pred_test),
                "rmse": rmse(y_test, pred_test),
            })
        except Exception as e:
            logger.warning(f"Failed to get predictions for model {m}: {e}")
            continue
    
    return {
        "predictor": predictor,
        "leaderboard": lb,
        "val_metrics": pd.DataFrame(val_rows).sort_values("mae"),
        "test_metrics": pd.DataFrame(test_rows).sort_values("mae"),
        "train_used": train2,
        "val_used": val2,
        "test_used": test2,
    }


def get_model_info(predictor: TabularPredictor) -> pd.DataFrame:
    """
    Extract detailed information about trained models.
    
    Useful for understanding model composition in Rashomon sets.
    """
    rows = []
    for model_name in predictor.model_names():
        try:
            info = predictor.model_info(model_name)
            rows.append({
                "model": model_name,
                "model_type": info.get("model_type", "unknown"),
                "fit_time": info.get("fit_time", None),
                "pred_time_val": info.get("pred_time_val", None),
                "n_features": info.get("num_features", None),
            })
        except Exception:
            rows.append({"model": model_name})
    
    return pd.DataFrame(rows)
