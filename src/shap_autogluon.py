"""
SHAP value computation for AutoGluon models.

The code uses TreeExplainer where the underlying model supports it. A chunked
permutation explainer covers everything else, such as bagged ensembles and
neural nets. The code caches results to disk per model.
"""
from __future__ import annotations

import gc
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import shap

logger = logging.getLogger("rashomon_shap")


@dataclass
class ShapConfig:
    """Configuration for SHAP computation."""
    background_size: int
    explain_size: int
    max_evals: int
    prefer_tree: bool = True
    enable_chunking: bool = True
    chunk_size: int = 100
    show_progress: bool = True
    n_jobs: int = 1  # Parallel jobs for multiple models
    
    def __post_init__(self):
        if self.background_size <= 0:
            raise ValueError("background_size must be positive")
        if self.explain_size <= 0:
            raise ValueError("explain_size must be positive")


@dataclass
class ShapResult:
    """Container for SHAP computation results."""
    feature_names: List[str]
    shap_values: np.ndarray  # Shape: (n_samples, n_features)
    base_values: np.ndarray  # Shape: (n_samples,)
    global_importance: np.ndarray  # Shape: (n_features,)
    explainer_type: str  # 'tree' or 'permutation'
    model_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialisation."""
        return {
            "feature_names": self.feature_names,
            "shap_values": self.shap_values,
            "base_values": self.base_values,
            "global_importance": self.global_importance,
            "explainer": self.explainer_type,
            "model_name": self.model_name,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShapResult":
        """Create from dictionary."""
        return cls(
            feature_names=list(d["feature_names"]),
            shap_values=np.asarray(d["shap_values"]),
            base_values=np.asarray(d["base_values"]),
            global_importance=np.asarray(d["global_importance"]),
            explainer_type=str(d.get("explainer", "unknown")),
            model_name=str(d.get("model_name", "")),
        )


def _sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample DataFrame with fallback for small datasets."""
    if len(df) == 0:
        raise ValueError("Cannot sample from empty DataFrame")
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def _get_trainer(predictor) -> Optional[Any]:
    """
    Access AutoGluon's internal trainer.
    
    The API differs across versions. The function tries several paths.
    """
    # Try direct access
    trainer = getattr(predictor, "_trainer", None)
    if trainer is not None:
        return trainer
    
    # Try via learner (older versions)
    learner = getattr(predictor, "_learner", None)
    if learner is not None:
        return getattr(learner, "_trainer", None)
    
    return None


def _try_tree_explainer(
    predictor,
    model_name: str,
    X_bg: pd.DataFrame,
    X_exp: pd.DataFrame,
    show_progress: bool = True,
) -> Optional[ShapResult]:
    """
    Attempt TreeExplainer for tree-based models.
    
    TreeExplainer is much faster than permutation-based methods for supported
    models (XGBoost, LightGBM, CatBoost, sklearn trees).
    
    This returns None if the model is not tree-based, or if extraction fails.
    """
    trainer = _get_trainer(predictor)
    if trainer is None:
        return None
    
    try:
        ag_model = trainer.load_model(model_name)
    except Exception as e:
        logger.debug(f"Could not load model {model_name}: {e}")
        return None
    
    base_model = getattr(ag_model, "model", None)
    if base_model is None:
        base_model = getattr(ag_model, "_model", None)
    if base_model is None:
        return None
    
    mod = getattr(base_model.__class__, "__module__", "")
    supported = ("xgboost", "lightgbm", "catboost", "sklearn.ensemble", "sklearn.tree")
    if not any(k in mod for k in supported):
        return None
    
    try:
        if show_progress:
            logger.info(f"  Using TreeExplainer for {model_name}")
        
        explainer = shap.TreeExplainer(base_model, data=X_bg)
        sv = explainer.shap_values(X_exp)
        
        # Multi-output explainers return a list. Regression uses the first.
        if isinstance(sv, list):
            sv = sv[0]
        
        shap_vals = np.asarray(sv)
        if shap_vals.ndim != 2:
            logger.warning(f"Unexpected SHAP shape: {shap_vals.shape}")
            return None
        
        expected = getattr(explainer, "expected_value", 0.0)
        if isinstance(expected, (list, np.ndarray)):
            expected = float(np.asarray(expected).flatten()[0])
        base_vals = np.full(shap_vals.shape[0], float(expected))
        
        # Thesis eq:global_importance: I_j = (1/n) * sum_i |phi_j(x_i)|.
        # This is the mean absolute SHAP value per feature (Lundberg and Lee,
        # 2017).
        global_imp = np.mean(np.abs(shap_vals), axis=0)
        
        return ShapResult(
            feature_names=list(X_bg.columns),
            shap_values=shap_vals,
            base_values=base_vals,
            global_importance=global_imp,
            explainer_type="tree",
            model_name=model_name,
        )
        
    except Exception as e:
        logger.debug(f"TreeExplainer failed for {model_name}: {e}")
        return None


def _compute_shap_permutation(
    predictor,
    model_name: str,
    X_bg: pd.DataFrame,
    X_exp: pd.DataFrame,
    cfg: ShapConfig,
) -> ShapResult:
    """
    Compute SHAP using permutation explainer with memory-efficient chunking.
    
    This is a fallback method. It works for any model type.
    """
    feature_names = list(X_bg.columns)
    
    def predict_fn(X) -> np.ndarray:
        """Prediction wrapper for SHAP."""
        if isinstance(X, pd.DataFrame):
            Xdf = X
        else:
            Xdf = pd.DataFrame(np.asarray(X), columns=feature_names)
        return predictor.predict(Xdf, model=model_name).values
    
    masker = shap.maskers.Independent(X_bg)
    explainer = shap.Explainer(predict_fn, masker, algorithm="permutation")
    
    # Chunking bounds peak memory on large explain sets.
    values_list: List[np.ndarray] = []
    base_list: List[np.ndarray] = []
    
    chunk_size = cfg.chunk_size if cfg.enable_chunking else len(X_exp)
    n_chunks = (len(X_exp) + chunk_size - 1) // chunk_size
    
    iterator = range(n_chunks)
    if cfg.show_progress:
        iterator = tqdm(
            iterator,
            desc=f"  SHAP [{model_name[:20]}]",
            unit="chunk",
            leave=False,
        )
    
    for i in iterator:
        start = i * chunk_size
        end = min(start + chunk_size, len(X_exp))
        chunk = X_exp.iloc[start:end]
        
        try:
            sv = explainer(chunk, max_evals=cfg.max_evals)
            values_list.append(np.asarray(sv.values))
            base_list.append(np.asarray(sv.base_values).flatten())
            
            del sv
            if cfg.enable_chunking:
                gc.collect()
                
        except Exception as e:
            logger.warning(f"SHAP chunk {i+1}/{n_chunks} failed: {e}")
            continue
    
    if not values_list:
        raise RuntimeError(f"All SHAP chunks failed for {model_name}")
    
    # Aggregate
    shap_vals = np.vstack(values_list)
    base_vals = np.concatenate(base_list)
    global_imp = np.mean(np.abs(shap_vals), axis=0)
    
    return ShapResult(
        feature_names=feature_names,
        shap_values=shap_vals,
        base_values=base_vals,
        global_importance=global_imp,
        explainer_type="permutation",
        model_name=model_name,
    )


def compute_shap_for_model(
    predictor,
    model_name: str,
    X_bg: pd.DataFrame,
    X_exp: pd.DataFrame,
    cfg: ShapConfig,
) -> ShapResult:
    """
    Compute SHAP values for a single model.
    
    This selects the best explainer automatically:
    1. TreeExplainer for tree-based models (fast)
    2. PermutationExplainer as fallback (universal)
    
    Args:
        predictor: Trained AutoGluon predictor
        model_name: Name of model to explain
        X_bg: Background dataset for SHAP
        X_exp: Dataset to explain
        cfg: SHAP configuration
    
    Returns:
        ShapResult with all computed values
    
    Raises:
        ValueError: If input validation fails
        RuntimeError: If SHAP computation fails
    """
    # Validate inputs
    if X_bg is None or X_exp is None:
        raise ValueError("Background and explanation datasets cannot be None")
    
    if len(X_bg) == 0 or len(X_exp) == 0:
        raise ValueError("Datasets must be non-empty")
    
    if list(X_bg.columns) != list(X_exp.columns):
        raise ValueError("Feature columns must match between datasets")
    
    # Try TreeExplainer first
    if cfg.prefer_tree:
        result = _try_tree_explainer(
            predictor, model_name, X_bg, X_exp, cfg.show_progress
        )
        if result is not None:
            return result
    
    # Fall back to permutation
    if cfg.show_progress:
        logger.info(f"  Using PermutationExplainer for {model_name}")
    
    return _compute_shap_permutation(predictor, model_name, X_bg, X_exp, cfg)


def compute_shap_for_models(
    predictor,
    model_names: List[str],
    X_bg: pd.DataFrame,
    X_exp: pd.DataFrame,
    cfg: ShapConfig,
    cache_dir: Optional[Path] = None,
) -> Dict[str, ShapResult]:
    """
    Compute SHAP for multiple models with optional caching.
    
    Args:
        predictor: Trained AutoGluon predictor
        model_names: List of model names
        X_bg: Background dataset
        X_exp: Explanation dataset
        cfg: SHAP configuration
        cache_dir: Directory for caching results (optional)
    
    Returns:
        Dictionary mapping model_name -> ShapResult
    """
    results: Dict[str, ShapResult] = {}
    
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    if cfg.show_progress:
        model_iter = tqdm(model_names, desc="Computing SHAP", unit="model")
    else:
        model_iter = model_names
    
    for model_name in model_iter:
        if cache_dir is not None:
            cache_file = cache_dir / f"{_safe_name(model_name)}.npz"
            if cache_file.exists():
                try:
                    result = _load_shap_cache(cache_file)
                    result.model_name = model_name
                    results[model_name] = result
                    logger.debug(f"Loaded cached SHAP for {model_name}")
                    continue
                except Exception as e:
                    logger.warning(f"Cache load failed for {model_name}: {e}")
        
        # Compute
        try:
            result = compute_shap_for_model(
                predictor, model_name, X_bg, X_exp, cfg
            )
            results[model_name] = result
            
            if cache_dir is not None:
                _save_shap_cache(cache_file, result)
                
        except Exception as e:
            logger.error(f"SHAP failed for {model_name}: {e}")
            continue
    
    return results


def _safe_name(s: str) -> str:
    """Convert string to safe filename."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)


def _save_shap_cache(path: Path, result: ShapResult) -> None:
    """Save SHAP result to cache file."""
    np.savez_compressed(
        path,
        feature_names=np.asarray(result.feature_names, dtype=object),
        shap_values=result.shap_values.astype(np.float32),  # Save space
        base_values=result.base_values.astype(np.float32),
        global_importance=result.global_importance.astype(np.float32),
        explainer=np.asarray(result.explainer_type, dtype=object),
        model_name=np.asarray(result.model_name, dtype=object),
    )


def _load_shap_cache(path: Path) -> ShapResult:
    """Load SHAP result from cache file."""
    z = np.load(path, allow_pickle=True)
    return ShapResult(
        feature_names=[str(x) for x in z["feature_names"].tolist()],
        shap_values=z["shap_values"].astype(np.float64),
        base_values=z["base_values"].astype(np.float64),
        global_importance=z["global_importance"].astype(np.float64),
        explainer_type=str(z["explainer"].item()) if "explainer" in z.files else "unknown",
        model_name=str(z["model_name"].item()) if "model_name" in z.files else "",
    )
