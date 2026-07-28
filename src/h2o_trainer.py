"""
H2O AutoML training wrapper.

It provides the same output interface as autogluon_trainer.py. The Rashomon
set construction, the aggregation, and the rest of the pipeline therefore
work without changes.

Key difference from AutoGluon:
  H2O AutoML trains many hyperparameter variants of each model family
  (GBM, XGBoost, DRF, GLM) within the time budget. It produces 10 to 20
  distinct models, typically. These Rashomon sets are larger than
  AutoGluon's.

Installation:
    pip install h2o
    # Also requires Java 8+ on PATH. Check with: java -version
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("rashomon_shap")

# Track whether H2O has been initialised in this process
_H2O_INITIALIZED: bool = False

# Detect multi-thread conversion support once at import time.
# H2O's as_data_frame(use_multi_thread=True) requires both polars and pyarrow.
# If either is missing we silently fall back to single-thread.
try:
    import polars  # noqa: F401
    import pyarrow  # noqa: F401
    _H2O_MULTITHREAD: bool = True
except ImportError:
    _H2O_MULTITHREAD: bool = False


def h2o_to_pandas(frame) -> "pd.DataFrame":
    """
    Convert an H2OFrame to a pandas DataFrame.

    It uses use_multi_thread=True when both polars and pyarrow are installed.
    This suppresses the H2ODependencyWarning that H2O emits for single-thread
    conversions. It falls back to single-thread silently when dependencies are
    missing.
    """
    return frame.as_data_frame(use_multi_thread=_H2O_MULTITHREAD)


# =============================================================================
# H2O Lifecycle
# =============================================================================

def ensure_h2o(max_mem_size: str = "4G", nthreads: int = -1) -> None:
    """
    Start the H2O cluster if it is not already running.

    It is safe to call this function many times. The server can die since the
    last init, for example on an OOM crash. This function then resets the flag
    and restarts automatically.

    Args:
        max_mem_size: JVM heap size, e.g. "4G", "8G". Increase if OOM errors.
        nthreads: CPU threads for H2O (-1 = use all available).
    """
    global _H2O_INITIALIZED
    if _H2O_INITIALIZED:
        # Verify the server is actually alive before returning.
        try:
            import h2o
            h2o.cluster()
            return
        except Exception:
            _H2O_INITIALIZED = False
            logger.warning("H2O server died; restarting...")
    try:
        import h2o

        # Isolate this process's H2O cluster.
        #
        # h2o.init() with no name/port auto-discovers peer H2O JVMs on the same
        # host. It then MERGES with them into one multi-node cloud. Sometimes
        # several jobs of this pipeline run on the same compute node. Those jobs
        # then share a single cluster. They overwrite each other's frames and
        # models. They die with "Local server has died unexpectedly" as soon as
        # one job shuts its JVM down.
        #
        # A unique cluster name confines membership to this process. H2O only
        # joins a cloud of the same name. A name-derived port keeps two
        # co-scheduled jobs off the same socket. The code falls back to the pid
        # when it is not under SLURM.
        uniq = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        cluster_name = f"h2o_{uniq}"
        # Ephemeral-range port, deterministic per job, away from H2O's 54321 default.
        port = 40000 + (int(uniq) % 20000)

        # nthreads=-1 makes H2O spawn one thread per core on the whole machine.
        # This ignores the SLURM allocation. Several jobs on one node then
        # oversubscribe the CPUs many times over. Respect the allocation when
        # SLURM provides it.
        if nthreads == -1:
            slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            if slurm_cpus:
                nthreads = int(slurm_cpus)

        h2o.init(
            max_mem_size=max_mem_size,
            nthreads=nthreads,
            name=cluster_name,
            port=port,
            verbose=False,
        )
        h2o.no_progress()
        _H2O_INITIALIZED = True

        cloud_size = h2o.cluster().cloud_size
        logger.info(
            f"H2O initialized: {cloud_size} node(s), mem={max_mem_size}, "
            f"cluster={cluster_name}, port={port}"
        )
        if cloud_size != 1:
            raise RuntimeError(
                f"H2O formed a {cloud_size}-node cluster; expected exactly 1. "
                f"This process has merged with another job's H2O JVM and the "
                f"results would be corrupt. Aborting."
            )
    except Exception as e:
        raise RuntimeError(
            f"H2O initialization failed.\n"
            f"  Install: pip install h2o\n"
            f"  Requires: Java 8+ on PATH (check: java -version)\n"
            f"  Error: {e}"
        ) from e


def shutdown_h2o() -> None:
    """Shut down the H2O cluster. Call it once at the end of the experiment."""
    global _H2O_INITIALIZED
    if not _H2O_INITIALIZED:
        return
    try:
        import h2o
        h2o.cluster().shutdown()
        _H2O_INITIALIZED = False
        logger.info("H2O cluster shut down")
    except Exception as e:
        logger.warning(f"H2O shutdown: {e}")


# =============================================================================
# Metrics
# =============================================================================

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# =============================================================================
# Training
# =============================================================================

def train_h2o_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_models: int = 20,
    max_runtime_secs: int = 360,
    seed: int = 0,
    sort_metric: str = "MAE",
    nfolds: int = 0,
    exclude_algos: Optional[List[str]] = None,
    max_mem_size: str = "4G",
    label: str = "label",
) -> Dict[str, Any]:
    """
    Train H2O AutoML for one (split, seed) pair.

    With nfolds=0, H2O uses the validation_frame for leaderboard scoring.
    Individual model families (GBM, XGBoost, DRF, GLM) produce multiple
    hyperparameter variants. This gives 10 to 20 models within the time budget.

    Note: nfolds=0 disables StackedEnsemble (it needs CV fold predictions).
    To enable StackedEnsemble, set nfolds=5. That uses random CV folds instead
    of temporal splits. Random folds leak future data in time series.

    Returns dict with keys:
        predictor     : H2OAutoML instance
        leaderboard   : pd.DataFrame with all model IDs and leaderboard metrics
        val_metrics   : pd.DataFrame with columns [model, mae, rmse], sorted by mae
        test_metrics  : pd.DataFrame with columns [model, mae, rmse], sorted by mae
        train_used    : processed train DataFrame (timestamp cols dropped)
        val_used      : processed val DataFrame
        test_used     : processed test DataFrame
        _feature_cols : list of feature column names
    """
    import h2o
    from h2o.automl import H2OAutoML

    ensure_h2o(max_mem_size=max_mem_size)

    # Drop timestamp and ID columns before training.
    # item_id and series_id are grouping identifiers, not predictors. H2O
    # would drop them anyway with a "bad or constant columns" warning.
    # Dropping them here silences that warning. It also keeps feature_cols clean
    # for downstream SHAP (no need to handle them as special cat_cols).
    drop_cols = ["timestamp", "label_timestamp", "item_id", "series_id"]
    train2 = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns]).copy()
    val2   = val_df.drop(  columns=[c for c in drop_cols if c in val_df.columns]).copy()
    test2  = test_df.drop( columns=[c for c in drop_cols if c in test_df.columns]).copy()

    feature_cols = [c for c in train2.columns if c != label]

    # Upload to H2O JVM
    logger.info("  Uploading data to H2O...")
    train_h = h2o.H2OFrame(train2)
    val_h   = h2o.H2OFrame(val2)
    test_h  = h2o.H2OFrame(test2)

    # Configure AutoML
    aml = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        seed=seed,
        sort_metric=sort_metric,
        stopping_metric=sort_metric,
        nfolds=nfolds,
        exclude_algos=exclude_algos or [],
        keep_cross_validation_predictions=False,
        verbosity="warn",
    )

    logger.info(
        f"  H2O AutoML: max_models={max_models}, "
        f"max_runtime_secs={max_runtime_secs}, seed={seed}"
    )

    aml.train(
        x=feature_cols,
        y=label,
        training_frame=train_h,
        validation_frame=val_h if nfolds == 0 else None,
    )

    lb_df = h2o_to_pandas(aml.leaderboard)
    n_trained = len(lb_df)
    logger.info(f"  H2O trained {n_trained} models")
    if n_trained == 0:
        raise RuntimeError("H2O AutoML produced no models. Check time limit and data size.")

    # Compute val and test MAE for every model.
    # We do this manually to match the AutoGluon pipeline.
    y_val  = val2[label].values
    y_test = test2[label].values
    val_rows, test_rows = [], []

    for mid in lb_df["model_id"].tolist():
        try:
            model = h2o.get_model(mid)
            p_val  = h2o_to_pandas(model.predict(val_h))["predict"].values
            p_test = h2o_to_pandas(model.predict(test_h))["predict"].values
            val_rows.append(  {"model": mid, "mae": _mae(y_val,  p_val),  "rmse": _rmse(y_val,  p_val)})
            test_rows.append( {"model": mid, "mae": _mae(y_test, p_test), "rmse": _rmse(y_test, p_test)})
        except Exception as e:
            logger.warning(f"  Prediction failed for {mid}: {e}")

    val_metrics  = pd.DataFrame(val_rows).sort_values("mae").reset_index(drop=True)
    test_metrics = pd.DataFrame(test_rows).sort_values("mae").reset_index(drop=True)

    if val_metrics.empty:
        raise RuntimeError("No models produced valid predictions.")

    best_name = val_metrics.iloc[0]["model"]
    logger.info(f"  Best val MAE: {val_metrics.iloc[0]['mae']:.4f}  ({best_name})")

    # Free the training frame from the JVM. Keep val_h and test_h for SHAP.
    h2o.remove(train_h)

    return {
        "predictor":     aml,
        "leaderboard":   lb_df,
        "val_metrics":   val_metrics,
        "test_metrics":  test_metrics,
        "train_used":    train2,
        "val_used":      val2,
        "test_used":     test2,
        "_val_h2o":      val_h,
        "_test_h2o":     test_h,
        "_feature_cols": feature_cols,
    }


def cleanup_h2o_frames(*frames) -> None:
    """Remove H2O frames from JVM memory."""
    try:
        import h2o
        for f in frames:
            if f is not None:
                h2o.remove(f)
    except Exception:
        pass


def cleanup_h2o_models(leaderboard_df: "pd.DataFrame") -> None:
    """
    Remove all trained models for a completed split from JVM memory.

    Call this after SHAP computation finishes for a split. Without it,
    model objects accumulate across splits and exhaust the JVM heap. The
    server then crashes partway through a multi-seed run.
    """
    try:
        import h2o
        model_ids = leaderboard_df["model_id"].tolist()
        for mid in model_ids:
            try:
                h2o.remove(mid)
            except Exception:
                pass
        logger.debug(f"Removed {len(model_ids)} H2O models from JVM")
    except Exception as e:
        logger.warning(f"cleanup_h2o_models: {e}")
