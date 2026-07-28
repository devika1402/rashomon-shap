#!/usr/bin/env python3
"""
End-to-end H2O AutoML pipeline. It mirrors autogluon_pipeline.py exactly.

H2O trains many hyperparameter variants per model family (GBM, DRF,
XGBoost, GLM) within the time budget. A run produces 10 to 20 models,
typically. This gives architecturally diverse Rashomon sets.

Usage:
    python src/h2o_pipeline.py --config configs/h2o/h2o_electricity.yaml

Prerequisites:
    pip install h2o
    java -version   # must be Java 8+ on PATH

The output directory layout (05_rashomon/, 04_stability/, etc.) matches
autogluon_pipeline.py results. All downstream analysis scripts therefore
work unchanged.
"""
from __future__ import annotations

import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import gc
import logging
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from pipeline_utils import set_seeds, make_run_dir, save_json, setup_logging, detect_hardware
from timeseries_to_tabular import build_supervised
from temporal_cross_validation import build_splits, slice_split, get_split_sizes
from rashomon_set_builder import rashomon_sets_from_val_metrics, get_union_models
from shap_autogluon import ShapConfig

# H2O-specific modules
from h2o_trainer import train_h2o_split, shutdown_h2o, cleanup_h2o_frames, cleanup_h2o_models
from shap_h2o import compute_shap_for_h2o_models

# Shared pipeline stages
from pipeline_stage_runner import (
    setup_output_structure, generate_readme,
    load_real_data, load_benchmark_data,
    generate_final_report,
)

from stability_metrics import (
    stability_over_splits, compute_stability_summary,
    compute_grouped_stability, compute_epsilon_sensitivity,
)
from importance_aggregation import (
    aggregate_global_importances, rank_then_mean_per_split,
    compute_importance_distribution_data, compute_rank_stability_matrix,
)

from results_visualizer import (
    plot_importance_bands, plot_stability_over_time,
    plot_importance_violin, plot_rank_stability_heatmap,
    plot_epsilon_stability_comparison,
)

logger: Optional[logging.Logger] = None


def parse_args():
    parser = argparse.ArgumentParser(description="Rashomon × SHAP Pipeline (H2O AutoML)")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore SHAP cache")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP (for debugging)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser.parse_args()


def main():
    global logger
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seeds(cfg["project"]["random_seed"])
    run_dir = make_run_dir(cfg["project"]["outdir"], cfg["project"].get("run_name"))
    logger = setup_logging(run_dir, getattr(logging, args.log_level))

    logger.info("=" * 70)
    logger.info("RASHOMON × SHAP PIPELINE  [H2O AutoML]")
    logger.info(f"Run directory: {run_dir}")
    logger.info("=" * 70)

    hw = detect_hardware()
    logger.info(f"Hardware: {hw['cpu_count']} CPUs, CUDA={hw['has_cuda']}, MPS={hw['has_mps']}")
    dirs = setup_output_structure(run_dir)

    save_json(cfg, run_dir / "config.json")
    save_json(hw, run_dir / "hardware.json")

    if args.dry_run:
        logger.info("DRY RUN: Config valid. Exiting.")
        return

    try:
        # =====================================================================
        # STEP 1: Load Data
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 1: Load Data")
        logger.info("=" * 70)

        source = cfg["data"]["source"]
        if source == "real":
            panel, metadata = load_real_data(cfg, dirs["data"])
            id_col     = cfg["data"]["real"].get("id_col",     "item_id")
            time_col   = cfg["data"]["real"].get("time_col",   "timestamp")
            target_col = cfg["data"]["real"].get("target_col", "target")
        elif source == "benchmark":
            panel, metadata = load_benchmark_data(cfg, dirs["data"])
            id_col, time_col, target_col = "item_id", "timestamp", "target"
        else:
            raise ValueError(f"Unknown source: {source}")

        panel.to_parquet(dirs["data"] / "panel.parquet", index=False)

        # =====================================================================
        # STEP 2: Tabularize
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 2: Tabularize")
        logger.info("=" * 70)

        panel_std = panel.rename(columns={id_col: "item_id", time_col: "timestamp", target_col: "target"})
        sup = build_supervised(panel_std, cfg["data"]["horizon"], cfg["data"]["target_lags"], cfg["data"]["cov_lag"])
        sup.to_parquet(dirs["data"] / "supervised.parquet", index=False)
        logger.info(f"Supervised shape: {sup.shape}")

        # =====================================================================
        # STEP 3: Splits
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 3: Build splits")
        logger.info("=" * 70)

        splits = build_splits(
            sup,
            cfg["splits"]["n_splits"],
            cfg["splits"]["min_train_time"],
            cfg["splits"]["val_size"],
            cfg["splits"]["test_size"],
        )
        get_split_sizes(sup, splits).to_csv(dirs["data"] / "splits.csv", index=False)
        logger.info(f"Created {len(splits)} splits")
        for sp in splits:
            logger.info(f"  {sp}")

        # =====================================================================
        # STEP 4: Train (H2O) + Rashomon + SHAP
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 4: Train (H2O AutoML) + Rashomon + SHAP")
        logger.info("=" * 70)

        h2o_cfg    = cfg["h2o"]
        eps_list   = [float(e) for e in cfg["rashomon"]["eps_list"]]
        seeds      = [int(s) for s in h2o_cfg["seeds"]]
        shap_cfg   = ShapConfig(
            background_size = cfg["shap"]["background_size"],
            explain_size    = cfg["shap"]["explain_size"],
            max_evals       = cfg["shap"]["max_evals"],
            prefer_tree     = cfg["shap"].get("prefer_tree", True),
            enable_chunking = True,
            chunk_size      = cfg["shap"].get("chunk_size", 50),
            show_progress   = True,
        )

        global_imp_rows: list = []
        alt_imp_rows:    list = []
        metrics_rows:    list = []
        rashomon_rows:   list = []

        # dual_explainer: explain every model a SECOND time. The second pass
        # uses the opposite prefer_tree setting. The two passes see identical
        # models, identical explained rows and identical Rashomon sets. Any
        # difference between them therefore comes from the explainer alone.
        # The single-explainer confirmation runs use this flag. They reproduce
        # the mixed-explainer aggregation artefact within one run, not against
        # a separately trained run.
        dual_explainer = bool(cfg["shap"].get("dual_explainer", False))
        total = len(seeds) * len(splits)

        for i, seed in enumerate(seeds):
            for j, split in enumerate(splits):
                current = i * len(splits) + j + 1
                logger.info(f"\n[{current}/{total}] Seed={seed}, Split={split.split_id}")

                try:
                    train_df, val_df, test_df = slice_split(sup, split)
                    split_dir = dirs["models"] / f"seed_{seed}" / f"split_{split.split_id}"
                    split_dir.mkdir(parents=True, exist_ok=True)

                    # ---------------------------------------------------------
                    # H2O training
                    # ---------------------------------------------------------
                    logger.info("  Training with H2O AutoML...")
                    train_out = train_h2o_split(
                        train_df, val_df, test_df,
                        max_models       = h2o_cfg["max_models"],
                        max_runtime_secs = h2o_cfg["max_runtime_secs"],
                        seed             = seed,
                        sort_metric      = h2o_cfg.get("sort_metric", "MAE"),
                        nfolds           = h2o_cfg.get("nfolds", 0),
                        exclude_algos    = h2o_cfg.get("exclude_algos", []),
                        max_mem_size     = h2o_cfg.get("max_mem_size", "4G"),
                    )

                    val_metrics  = train_out["val_metrics"]
                    test_metrics = train_out["test_metrics"]
                    feature_cols = train_out["_feature_cols"]
                    logger.info(
                        f"  Trained {len(val_metrics)} models, "
                        f"best MAE: {val_metrics['mae'].min():.4f}"
                    )

                    # ---------------------------------------------------------
                    # Rashomon sets
                    # ---------------------------------------------------------
                    rash_sets = rashomon_sets_from_val_metrics(
                        val_metrics, eps_list,
                        max_models=h2o_cfg["max_models"],
                    )
                    for eps, sub in rash_sets.items():
                        logger.info(f"  ε={eps}: {len(sub)} models in Rashomon set")
                        for _, r in sub.iterrows():
                            rashomon_rows.append({
                                "seed": seed, "split_id": split.split_id,
                                "eps": eps, "model": r["model"], "val_mae": r["mae"],
                            })

                    if args.skip_shap:
                        cleanup_h2o_frames(train_out["_val_h2o"], train_out["_test_h2o"])
                        cleanup_h2o_models(train_out["leaderboard"])
                        continue

                    # ---------------------------------------------------------
                    # SHAP for Rashomon set members
                    # ---------------------------------------------------------
                    X_train = train_out["train_used"].drop(
                        columns=[c for c in ["label", "timestamp", "label_timestamp"]
                                 if c in train_out["train_used"].columns]
                    )
                    X_test = train_out["test_used"].drop(
                        columns=[c for c in ["label", "timestamp", "label_timestamp"]
                                 if c in train_out["test_used"].columns]
                    )
                    _shap_seed = 42
                    X_bg  = X_train.sample(n=min(shap_cfg.background_size, len(X_train)), random_state=_shap_seed)
                    X_exp = X_test.sample( n=min(shap_cfg.explain_size,    len(X_test)),  random_state=_shap_seed)

                    union_models = get_union_models(rash_sets)
                    logger.info(f"  Computing SHAP for {len(union_models)} models...")

                    shap_results = compute_shap_for_h2o_models(
                        model_ids    = union_models,
                        feature_cols = feature_cols,
                        X_bg         = X_bg,
                        X_exp        = X_exp,
                        cfg          = shap_cfg,
                        cache_dir    = split_dir / "shap_cache" if not args.force_recompute else None,
                    )

                    # Second pass over the SAME models with the opposite routing.
                    # The primary setting is prefer_tree=False. This pass then
                    # reproduces the original mixed-explainer behaviour. It uses
                    # native TreeSHAP/Saabas for tree families and permutation
                    # for GLM.
                    alt_results: Dict[str, Any] = {}
                    if dual_explainer:
                        alt_cfg = replace(shap_cfg, prefer_tree=not shap_cfg.prefer_tree)
                        logger.info(
                            f"  Dual-explainer pass: re-explaining the same "
                            f"{len(union_models)} models with prefer_tree={alt_cfg.prefer_tree}"
                        )
                        alt_results = compute_shap_for_h2o_models(
                            model_ids    = union_models,
                            feature_cols = feature_cols,
                            X_bg         = X_bg,
                            X_exp        = X_exp,
                            cfg          = alt_cfg,
                            cache_dir    = (split_dir / "shap_cache_alt"
                                            if not args.force_recompute else None),
                        )

                    for eps, sub in rash_sets.items():
                        for _, r in sub.iterrows():
                            m = str(r["model"])
                            if m not in shap_results:
                                continue
                            res = shap_results[m]
                            for feat, imp in zip(res.feature_names, res.global_importance):
                                global_imp_rows.append({
                                    "split_id": split.split_id, "seed": seed, "eps": eps,
                                    "model": m, "feature": feat, "global_importance": float(imp),
                                    "explainer_type": res.explainer_type,
                                })
                            if m in alt_results:
                                alt = alt_results[m]
                                for feat, imp in zip(alt.feature_names, alt.global_importance):
                                    alt_imp_rows.append({
                                        "split_id": split.split_id, "seed": seed, "eps": eps,
                                        "model": m, "feature": feat,
                                        "global_importance": float(imp),
                                        "explainer_type": alt.explainer_type,
                                    })
                            test_row = test_metrics[test_metrics["model"] == m]
                            test_mae = float(test_row["mae"].iloc[0]) if not test_row.empty else float("nan")
                            metrics_rows.append({
                                "split_id": split.split_id, "seed": seed, "eps": eps,
                                "model": m, "val_mae": float(r["mae"]), "test_mae": test_mae,
                            })

                    # Free H2O frames and trained models from JVM.
                    # Remove the models explicitly. Without this step they
                    # accumulate across splits and exhaust the JVM heap.
                    cleanup_h2o_frames(train_out["_val_h2o"], train_out["_test_h2o"])
                    cleanup_h2o_models(train_out["leaderboard"])
                    del train_out, shap_results, alt_results
                    gc.collect()

                except Exception as e:
                    logger.error(f"  Split failed: {e}\n{traceback.format_exc()}")

        # =====================================================================
        # STEP 5: Save & Aggregate
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 5: Save & Aggregate")
        logger.info("=" * 70)

        global_imp_df = pd.DataFrame(global_imp_rows)
        metrics_df    = pd.DataFrame(metrics_rows)
        rashomon_df   = pd.DataFrame(rashomon_rows)

        global_imp_df.to_csv(dirs["importance"] / "raw_importance.csv",   index=False)

        # Alternate-explainer attributions for the same models/rows/Rashomon sets.
        # The downstream stability code reads raw_importance.csv only. This file
        # lets us compute the explainer contrast without a second training run.
        if alt_imp_rows:
            alt_imp_df = pd.DataFrame(alt_imp_rows)
            alt_imp_df.to_csv(dirs["importance"] / "raw_importance_alt.csv", index=False)
            logger.info(
                f"Dual-explainer: wrote raw_importance_alt.csv "
                f"({len(alt_imp_df)} rows, explainers: "
                f"{sorted(alt_imp_df['explainer_type'].unique())})"
            )
        rashomon_df.to_csv(  dirs["rashomon"]   / "rashomon_models.csv",  index=False)
        metrics_df.to_csv(   dirs["rashomon"]   / "model_metrics.csv",    index=False)

        logger.info(f"Saved {len(global_imp_df)} importance records")

        if global_imp_df.empty:
            logger.warning("No SHAP results produced. Check logs above.")
            return

        summary_df = aggregate_global_importances(global_imp_df, cfg["shap"]["quantiles"])
        summary_df.to_csv(dirs["importance"] / "aggregated_summary.csv", index=False)

        # =====================================================================
        # STEP 6: Stability & Reports
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 6: Reports")
        logger.info("=" * 70)

        # Rank within each model, then average the ranks (eq:rank_aggregation).
        # The other option ranks the mean of raw SHAP magnitudes. That lets a
        # single model on a different numeric scale determine the aggregate.
        # H2O sets mix exact TreeSHAP with unbounded Saabas attributions. This
        # matters here.
        rank_df = rank_then_mean_per_split(global_imp_df)
        rank_df.to_csv(dirs["importance"] / "feature_ranks.csv", index=False)

        # Top 30% of features used for Jaccard computation (k=None uses dynamic default)
        stability = stability_over_splits(rank_df)
        save_json(stability, dirs["stability"] / "temporal_stability.json")

        stability_summary = compute_stability_summary(stability)
        if not stability_summary.empty:
            stability_summary.to_csv(dirs["stability"] / "stability_summary.csv", index=False)

        # Top 30% of features used for Jaccard computation (k=None uses dynamic default)
        eps_sensitivity = compute_epsilon_sensitivity(rank_df, eps_list)
        if not eps_sensitivity.empty:
            eps_sensitivity.to_csv(dirs["stability"] / "epsilon_sensitivity.csv", index=False)

        if "item_id" in rank_df.columns or "item_id" in sup.columns:
            # Top 30% of features used for Jaccard computation (k=None uses dynamic default)
            grouped_stability = compute_grouped_stability(rank_df, group_col="item_id")
            save_json(grouped_stability, dirs["stability"] / "grouped_stability.json")

        # --- Phase 1: Rashomon Set Characterization ---
        logger.info("Phase 1: Rashomon Set Characterization...")
        for eps in eps_list:
            eps_str = f"eps_{eps:.2f}".replace(".", "_")

            plot_importance_bands(
                summary_df,
                dirs["fig_importance"] / f"importance_{eps_str}.png",
                splits[-1].split_id, seeds[0], eps, cfg["report"]["topk_features"],
            )

        # --- Phase 2: Within-Set Agreement ---
        # The importance-distribution metric in this phase describes one Rashomon
        # set. That set is the final split at the first seed, per epsilon. It does
        # not average over splits or seeds. The temporal stability metrics in
        # Phase 3 do.
        logger.info("Phase 2: Within-Set Agreement...")
        for eps in eps_list:
            eps_str = f"eps_{eps:.2f}".replace(".", "_")

            dist_df = compute_importance_distribution_data(
                global_imp_df, split_id=splits[-1].split_id, seed=seeds[0], eps=eps,
                top_k=cfg["report"]["topk_features"],
            )
            if not dist_df.empty:
                plot_importance_violin(
                    dist_df,
                    dirs["fig_importance"] / f"violin_{eps_str}.png",
                    title=f"Feature Importance Distribution (ε={eps})",
                )

        # --- Phase 3: Temporal Stability ---
        logger.info("Phase 3: Temporal Stability...")
        for eps in eps_list:
            eps_str = f"eps_{eps:.2f}".replace(".", "_")
            rank_matrix = compute_rank_stability_matrix(
                rank_df, top_k=cfg["report"]["topk_features"], eps=eps, seed=seeds[0]
            )
            if not rank_matrix.empty:
                rank_matrix.to_csv(dirs["rank_matrices"] / f"{eps_str}.csv")
                plot_rank_stability_heatmap(
                    rank_matrix,
                    dirs["fig_stability"] / f"rank_heatmap_{eps_str}.png",
                    title=f"Feature Rank Stability Across Splits (ε={eps})",
                )

        if not eps_sensitivity.empty:
            plot_epsilon_stability_comparison(
                eps_sensitivity,
                dirs["fig_stability"] / "epsilon_comparison.png",
                title="Stability Metrics vs. Rashomon Tolerance",
            )

        if stability.get("consecutive"):
            plot_stability_over_time(
                stability,
                dirs["fig_stability"] / "temporal_stability.png",
                eps=eps_list[1] if len(eps_list) > 1 else eps_list[0],
                seed=seeds[0],
            )

        logger.info("All visualizations generated.")

        # Final report
        report = generate_final_report(
            dirs["reports"], cfg, summary_df, metrics_df, stability, dirs["root"].name
        )
        generate_readme(dirs, cfg)
        logger.info(f"\n{report}")

        logger.info("=" * 70)
        logger.info("COMPLETE!")
        logger.info(f"Results: {run_dir}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    finally:
        # Always shut down H2O cleanly
        shutdown_h2o()


if __name__ == "__main__":
    main()
