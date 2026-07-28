"""
SHAP value computation for H2O AutoML models.

Strategy
--------
Tree models (GBM, DRF, XGBoost, XRT):
    Use H2O's native predict_contributions(). This gives exact TreeSHAP for
    GBM/XGBoost. It gives approximate Saabas-style attributions for DRF/XRT
    (Lundberg et al., 2020). It is fast. It needs no separate background
    dataset. The two methods are not on a comparable magnitude scale. For this
    reason the downstream analyses stratify by family (thesis ch. 3.7).

Non-tree models (GLM, StackedEnsemble), or any tree model where
predict_contributions() fails:
    Fall back to shap.PermutationExplainer wrapping model.predict().

    The shap masker requires purely numeric input. A string or category column
    passed to shap.maskers.Independent raises a TypeError. So the code splits
    feature_cols into numeric_cols and cat_cols. It gives numeric_cols to the
    masker and explainer. It fixes cat_cols to the background mode inside the
    predict wrapper. H2O then still receives the full feature set it was trained
    on. Note: the H2O trainer already drops item_id before training. So item_id
    never appears in feature_cols or in any contribution output.

Output
------
Returns the same ShapResult objects as shap_autogluon.py. The downstream
aggregation and stability code therefore needs no framework-specific handling.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from shap_autogluon import (
    ShapConfig,
    ShapResult,
    _load_shap_cache,
    _safe_name,
    _save_shap_cache,
)
from h2o_trainer import h2o_to_pandas

logger = logging.getLogger("rashomon_shap")

# These H2O model families support predict_contributions().
# GBM and XGBoost give exact TreeSHAP.
# DRF and XRT give approximate Saabas-style contributions (H2O >= 3.32).
_TREE_SHAP_PREFIXES = ("GBM_", "DRF_", "XGBoost_", "XRT_")


def _is_tree_model(model_id: str) -> bool:
    """Return True if this H2O model family supports predict_contributions."""
    return any(model_id.startswith(p) for p in _TREE_SHAP_PREFIXES)


# =============================================================================
# Helper: separate numeric from categorical columns
# =============================================================================

def _split_numeric_cat(feature_cols: List[str], X_ref: pd.DataFrame):
    """
    Split feature_cols into (numeric_cols, cat_cols) by dtype in X_ref.

    numeric_cols : safe to pass to shap.maskers.Independent and numpy arrays.
    cat_cols     : string or category dtype. The code handles these separately.
    """
    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(X_ref[c])
    ]
    cat_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, cat_cols


# =============================================================================
# Native H2O SHAP  (tree models via predict_contributions)
# =============================================================================

def _compute_h2o_native_shap(
    model,
    feature_cols: List[str],
    X_exp: pd.DataFrame,
    model_id: str,
) -> Optional[ShapResult]:
    """
    Call H2O's predict_contributions() to get TreeSHAP values.

    H2O returns one column per feature plus a "BiasTerm" intercept column.
    For DRF/XRT the values are approximate (Saabas method). For GBM/XGBoost
    the values are exact.

    Returns None if the model does not support contributions. The caller
    then invokes _compute_h2o_permutation_shap as a fallback.

    Args:
        model        : H2O model object retrieved via h2o.get_model()
        feature_cols : All feature columns the model was trained on
        X_exp        : pandas DataFrame of rows to explain
        model_id     : H2O model ID string (used only for logging)
    """
    try:
        import h2o

        # Upload only feature columns (no label) to H2O.
        # reset_index gives a clean 0-based row index. This avoids H2O
        # frame alignment issues when X_exp is a slice of a larger frame.
        X_h = h2o.H2OFrame(X_exp[feature_cols].reset_index(drop=True))

        # predict_contributions returns an H2OFrame: one col per feature
        # (matching training column names) + "BiasTerm".
        contrib_h  = model.predict_contributions(X_h)
        contrib_df = h2o_to_pandas(contrib_h)

        # Free H2O memory immediately. These frames can be large.
        h2o.remove(X_h)
        h2o.remove(contrib_h)

        # Extract the intercept / bias term.
        if "BiasTerm" in contrib_df.columns:
            bias = contrib_df["BiasTerm"].values.astype(np.float64)
        else:
            bias = np.zeros(len(contrib_df))

        # H2O preserves the original column names. So the intersection should
        # equal feature_cols minus "BiasTerm". feature_cols comes from the
        # trainer. The trainer already excludes item_id.
        shap_cols = [c for c in feature_cols if c in contrib_df.columns]

        if not shap_cols:
            # H2O returned a contributions frame whose column names do not
            # match feature_cols. Log enough detail to diagnose.
            logger.warning(
                f"Native SHAP column mismatch for {model_id[:50]}.\n"
                f"  Expected (first 8): {feature_cols[:8]}\n"
                f"  Got from H2O (first 8): {list(contrib_df.columns)[:8]}"
            )
            return None

        shap_vals  = contrib_df[shap_cols].values.astype(np.float64)
        # Thesis eq:global_importance: I_j = (1/n) * sum_i |phi_j(x_i)|,
        # the mean absolute SHAP value per feature (Lundberg and Lee, 2017).
        global_imp = np.mean(np.abs(shap_vals), axis=0)

        return ShapResult(
            feature_names=shap_cols,
            shap_values=shap_vals,
            base_values=bias,
            global_importance=global_imp,
            explainer_type="h2o_tree",
            model_name=model_id,
        )

    except Exception as e:
        # Use WARNING level, not debug. This makes intermittent DRF/XRT
        # failures visible in the run log for diagnosis.
        logger.warning(f"Native SHAP failed for {model_id[:50]}: {e}")
        return None


# =============================================================================
# Permutation SHAP fallback  (GLM, StackedEnsemble, or failed tree models)
# =============================================================================

def _compute_h2o_permutation_shap(
    model,
    feature_cols: List[str],
    X_bg: pd.DataFrame,
    X_exp: pd.DataFrame,
    cfg: ShapConfig,
    model_id: str,
) -> ShapResult:
    """
    Permutation SHAP for H2O models that do not support predict_contributions.

    The shap.maskers.Independent masker needs a purely numeric array. So the
    code partitions feature_cols into numeric_cols and cat_cols. The masker and
    explainer use numeric_cols. cat_cols hold the string or category dtype. The
    predict wrapper receives only numeric_cols from SHAP. It reconstructs the
    full feature set. It fixes each cat_col to its most frequent background
    value. H2O then always receives the complete column set it was trained on.
    SHAP reports results for numeric_cols only.

    Args:
        model        : H2O model object
        feature_cols : All feature columns used in training (numeric + cat)
        X_bg         : Background DataFrame (used to build masker and to
                       obtain mode values for categorical columns)
        X_exp        : DataFrame of rows to explain
        cfg          : ShapConfig with chunk_size, max_evals, etc.
        model_id     : H2O model ID string (logging only)
    """
    import h2o
    import shap

    # ------------------------------------------------------------------
    # 1. Separate numeric from non-numeric (categorical/string) columns.
    # ------------------------------------------------------------------
    numeric_cols, cat_cols = _split_numeric_cat(feature_cols, X_bg)

    if cat_cols:
        logger.debug(
            f"  Permutation SHAP [{model_id[:40]}]: "
            f"{len(numeric_cols)} numeric cols, "
            f"{len(cat_cols)} categorical fixed to mode: {cat_cols}"
        )

    # Pre-compute the mode of each categorical column from the background data.
    # The code inserts this fixed value into every prediction call. H2O then
    # gets a valid full-feature DataFrame. This holds regardless of which
    # features SHAP masks.
    cat_fixed = {
        col: X_bg[col].mode().iloc[0]
        for col in cat_cols
    }

    # ------------------------------------------------------------------
    # 2. Prediction wrapper. It takes a numeric-only numpy array from SHAP.
    #    It reconstructs the full feature DataFrame. It calls H2O predict.
    # ------------------------------------------------------------------
    def predict_fn(X_numpy: np.ndarray) -> np.ndarray:
        """
        shap.Explainer calls this wrapper for every masked sample batch.

        X_numpy shape: (n_samples, len(numeric_cols))
        Returns: float64 array of shape (n_samples,)
        """
        X_df = pd.DataFrame(X_numpy, columns=numeric_cols)

        # H2O trained on the categorical columns too. So re-attach them before
        # every predict, at their fixed background mode.
        for col, val in cat_fixed.items():
            X_df[col] = val

        X_df = X_df[feature_cols]

        X_h   = h2o.H2OFrame(X_df)
        preds = (
            h2o_to_pandas(model.predict(X_h))["predict"]
                 .values
                 .astype(np.float64)
        )
        h2o.remove(X_h)
        return preds

    # ------------------------------------------------------------------
    # 3. Build masker and explainer on numeric columns only.
    # ------------------------------------------------------------------
    # The explicit float64 cast avoids implicit type issues inside SHAP.
    bg_numeric  = X_bg[numeric_cols].astype(np.float64).reset_index(drop=True)
    exp_numeric = X_exp[numeric_cols].astype(np.float64).reset_index(drop=True)

    masker    = shap.maskers.Independent(bg_numeric)
    explainer = shap.Explainer(predict_fn, masker, algorithm="permutation")

    # ------------------------------------------------------------------
    # 4. Compute SHAP in chunks to limit peak memory usage.
    # ------------------------------------------------------------------
    values_list: list = []
    base_list:   list = []

    chunk_size = cfg.chunk_size if cfg.enable_chunking else len(exp_numeric)
    n_chunks   = max(1, (len(exp_numeric) + chunk_size - 1) // chunk_size)

    for i in range(n_chunks):
        chunk = exp_numeric.iloc[i * chunk_size : (i + 1) * chunk_size]
        try:
            sv = explainer(chunk, max_evals=cfg.max_evals)
            values_list.append(np.asarray(sv.values,      dtype=np.float64))
            base_list.append(  np.asarray(sv.base_values, dtype=np.float64).flatten())
            del sv
            gc.collect()
        except Exception as e:
            logger.warning(
                f"SHAP chunk {i + 1}/{n_chunks} failed for {model_id[:50]}: {e}"
            )

    if not values_list:
        raise RuntimeError(f"All SHAP chunks failed for {model_id}")

    shap_vals = np.vstack(values_list)
    base_vals = np.concatenate(base_list)

    return ShapResult(
        feature_names=numeric_cols,
        shap_values=shap_vals,
        base_values=base_vals,
        # Thesis eq:global_importance: I_j = (1/n) * sum_i |phi_j(x_i)|,
        # the mean absolute SHAP value per feature (Lundberg and Lee, 2017).
        global_importance=np.mean(np.abs(shap_vals), axis=0),
        explainer_type="permutation",
        model_name=model_id,
    )


# =============================================================================
# Public entry point
# =============================================================================

def compute_shap_for_h2o_models(
    model_ids: List[str],
    feature_cols: List[str],
    X_bg: pd.DataFrame,
    X_exp: pd.DataFrame,
    cfg: ShapConfig,
    cache_dir: Optional[Path] = None,
) -> Dict[str, ShapResult]:
    """
    Compute SHAP for a list of H2O model IDs.

    For each model:
      - Tree models (GBM, DRF, XGBoost, XRT): attempt H2O native
        predict_contributions().  If that returns None (e.g. DRF in
        some H2O configurations), fall back to permutation SHAP.
      - All other models (GLM, StackedEnsemble): use permutation SHAP
        directly.

    The permutation fallback handles mixed numeric/categorical feature_cols.
    See _compute_h2o_permutation_shap for details.

    Returns Dict[model_id -> ShapResult]. This is the same interface as
    shap_autogluon.compute_shap_for_models(). All downstream code is therefore
    framework-agnostic.

    Args:
        model_ids    : H2O model ID strings (from leaderboard["model_id"])
        feature_cols : Feature column names the models were trained on
                       (the trainer excludes item_id and timestamps).
        X_bg         : Background DataFrame for the permutation masker.
        X_exp        : DataFrame of rows to explain (subset of test set).
        cfg          : ShapConfig (background_size, chunk_size, max_evals...)
        cache_dir    : Optional directory to cache computed .npz results.
    """
    import h2o

    results: Dict[str, ShapResult] = {}

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for mid in model_ids:

        # ── check on-disk cache ────────────────────────────────────────
        if cache_dir is not None:
            cache_file = cache_dir / f"{_safe_name(mid)}.npz"
            if cache_file.exists():
                try:
                    result = _load_shap_cache(cache_file)
                    result.model_name = mid
                    results[mid] = result
                    logger.debug(f"Loaded cached SHAP for {mid[:50]}")
                    continue
                except Exception:
                    pass  # recompute if cache is corrupt

        # ── compute SHAP ───────────────────────────────────────────────
        try:
            model = h2o.get_model(mid)

            result = None

            # cfg.prefer_tree=False forces every model through the permutation
            # explainer, tree families included. A single attribution method
            # then produces the whole Rashomon set. The single-explainer
            # confirmation runs use this (configs/h2o/h2o_perm_*.yaml). They test
            # whether explainer heterogeneity causes the magnitude-scale artefact.
            if _is_tree_model(mid) and cfg.prefer_tree:
                logger.info(f"  H2O native SHAP  [{mid[:50]}]")
                result = _compute_h2o_native_shap(model, feature_cols, X_exp, mid)

                if result is None:
                    # predict_contributions failed or returned mismatched columns.
                    # _compute_h2o_native_shap already logged the warning.
                    logger.info(
                        f"  Native SHAP unavailable for [{mid[:50]}], "
                        f"falling back to permutation SHAP"
                    )

            if result is None:
                logger.info(f"  Permutation SHAP [{mid[:50]}]")
                result = _compute_h2o_permutation_shap(
                    model, feature_cols, X_bg, X_exp, cfg, mid
                )

            results[mid] = result

            if cache_dir is not None:
                _save_shap_cache(cache_file, result)

        except Exception as e:
            logger.error(f"SHAP failed for {mid}: {e}")

    return results
