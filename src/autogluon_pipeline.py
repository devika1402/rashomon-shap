"""
End-to-end AutoGluon pipeline: data loading, temporal splits, training,
Rashomon set construction, SHAP computation, and stability analysis.

Usage:
    python src/autogluon_pipeline.py --config configs/ag/bq/electricity.yaml
"""
from __future__ import annotations

# Suppress warnings BEFORE importing
import warnings
import os
warnings.filterwarnings("ignore", message=".*load_learner.*pickle.*")
warnings.filterwarnings("ignore", category=UserWarning, module="fastai")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["JAX_PLATFORMS"] = ""

import argparse
import gc
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Local imports
from pipeline_utils import set_seeds, make_run_dir, save_json, setup_logging, detect_hardware, get_optimal_resources
from timeseries_to_tabular import build_supervised, get_feature_names
from temporal_cross_validation import build_splits, slice_split, get_split_sizes
from autogluon_trainer import train_one_split
from rashomon_set_builder import rashomon_sets_from_val_metrics, get_union_models
from shap_autogluon import ShapConfig, compute_shap_for_models

# Pipeline stages (data loading, output organization, reports)
from pipeline_stage_runner import (
    setup_output_structure,
    load_real_data, load_benchmark_data,
    generate_final_report,
)

# Aggregation and stability metrics
from stability_metrics import (
    stability_over_splits, compute_stability_summary,
    compute_grouped_stability, compute_epsilon_sensitivity,
)
from importance_aggregation import (
    aggregate_global_importances, rank_then_mean_per_split,
    compute_importance_distribution_data, compute_rank_stability_matrix,
)

# Visualizations
from results_visualizer import (
    plot_importance_bands, plot_stability_over_time,
    plot_importance_violin, plot_rank_stability_heatmap,
    plot_epsilon_stability_comparison,
)

logger: Optional[logging.Logger] = None


def parse_args():
    parser = argparse.ArgumentParser(description="Rashomon × SHAP Pipeline")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore cache")
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
    logger.info("RASHOMON × SHAP PIPELINE")
    logger.info(f"Run directory: {run_dir}")
    logger.info("=" * 70)

    hw = detect_hardware()
    logger.info(f"Hardware: {hw['cpu_count']} CPUs, CUDA={hw['has_cuda']}, MPS={hw['has_mps']}")
    resources = get_optimal_resources(cfg["automl"].get("num_gpus", "auto"), cfg["automl"].get("num_cpus", "auto"))
    dirs = setup_output_structure(run_dir)

    save_json(cfg, run_dir / "config.json")
    save_json(hw, run_dir / "hardware.json")

    if args.dry_run:
        logger.info("DRY RUN: Config valid. Exiting.")
        return

    try:
        # STEP 1: Load Data
        logger.info("=" * 70)
        logger.info("STEP 1: Load Data")
        logger.info("=" * 70)

        source = cfg["data"]["source"]
        if source == "real":
            panel, metadata = load_real_data(cfg, dirs['data'])
            id_col = cfg["data"]["real"].get("id_col", "item_id")
            time_col = cfg["data"]["real"].get("time_col", "timestamp")
            target_col = cfg["data"]["real"].get("target_col", "target")
        elif source == "benchmark":
            panel, metadata = load_benchmark_data(cfg, dirs['data'])
            id_col, time_col, target_col = "item_id", "timestamp", "target"
        else:
            raise ValueError(f"Unknown source: {source}. Supported: real, benchmark")

        panel.to_parquet(dirs['data'] / "panel.parquet", index=False)

        # STEP 2: Tabularize
        logger.info("=" * 70)
        logger.info("STEP 2: Tabularize")
        logger.info("=" * 70)

        panel_std = panel.rename(columns={id_col: "item_id", time_col: "timestamp", target_col: "target"})
        sup = build_supervised(panel_std, cfg["data"]["horizon"], cfg["data"]["target_lags"], cfg["data"]["cov_lag"])
        sup.to_parquet(dirs['data'] / "supervised.parquet", index=False)
        logger.info(f"Supervised shape: {sup.shape}")

        # STEP 3: Splits
        logger.info("=" * 70)
        logger.info("STEP 3: Build splits")
        logger.info("=" * 70)

        splits = build_splits(sup, cfg["splits"]["n_splits"], cfg["splits"]["min_train_time"], cfg["splits"]["val_size"], cfg["splits"]["test_size"])
        get_split_sizes(sup, splits).to_csv(dirs['data'] / "splits.csv", index=False)
        logger.info(f"Created {len(splits)} splits")
        for sp in splits:
            logger.info(f"  {sp}")

        # STEP 4: Train + Rashomon + SHAP
        logger.info("=" * 70)
        logger.info("STEP 4: Train + Rashomon + SHAP")
        logger.info("=" * 70)

        eps_list = [float(e) for e in cfg["rashomon"]["eps_list"]]
        seeds = [int(s) for s in cfg["automl"]["seeds"]]
        shap_cfg = ShapConfig(
            background_size=cfg["shap"]["background_size"], explain_size=cfg["shap"]["explain_size"],
            max_evals=cfg["shap"]["max_evals"],
            prefer_tree=cfg["shap"].get("prefer_tree", True), enable_chunking=True,
            chunk_size=cfg["shap"].get("chunk_size", 100), show_progress=True,
        )

        global_imp_rows, metrics_rows, rashomon_rows = [], [], []
        total = len(seeds) * len(splits)

        for i, seed in enumerate(seeds):
            for j, split in enumerate(splits):
                current = i * len(splits) + j + 1
                logger.info(f"\n[{current}/{total}] Seed={seed}, Split={split.split_id}")

                try:
                    train_df, val_df, test_df = slice_split(sup, split)
                    split_dir = dirs['models'] / f"seed_{seed}" / f"split_{split.split_id}"
                    split_dir.mkdir(parents=True, exist_ok=True)

                    logger.info("  Training...")
                    train_out = train_one_split(
                        train_df, val_df, test_df, model_dir=split_dir / "models",
                        time_limit_s=cfg["automl"]["time_limit_s"], presets=cfg["automl"]["presets"],
                        seed=seed, eval_metric=cfg["automl"]["eval_metric"],
                        num_gpus=resources["num_gpus"], num_cpus=resources["num_cpus"],
                        fit_strategy=cfg["automl"].get("fit_strategy", "sequential"),
                        force_retrain=args.force_recompute,
                        verbosity=cfg["automl"].get("verbosity", 2),
                    )

                    predictor, val_metrics, test_metrics = train_out["predictor"], train_out["val_metrics"], train_out["test_metrics"]
                    logger.info(f"  Trained {len(val_metrics)} models, best MAE: {val_metrics['mae'].min():.4f}")

                    rash_sets = rashomon_sets_from_val_metrics(val_metrics, eps_list, cfg["automl"]["max_models_per_rashomon"])
                    for eps, sub in rash_sets.items():
                        logger.info(f"  ε={eps}: {len(sub)} models")
                        for _, r in sub.iterrows():
                            rashomon_rows.append({"seed": seed, "split_id": split.split_id, "eps": eps, "model": r["model"], "val_mae": r["mae"]})

                    if args.skip_shap:
                        continue

                    # Drop timestamp columns and label before SHAP
                    drop = ["timestamp", "label", "label_timestamp"]
                    X_train = train_out["train_used"].drop(columns=[c for c in drop if c in train_out["train_used"].columns])
                    X_test = train_out["test_used"].drop(columns=[c for c in drop if c in train_out["test_used"].columns])
                    # A fixed seed (42) separates the SHAP sample selection from the
                    # model training seed. This keeps the background and explanation
                    # sets identical across seeds for the same (split, data). Valid
                    # cross-seed SHAP comparisons need this.
                    _shap_seed = 42
                    X_bg = X_train.sample(n=min(shap_cfg.background_size, len(X_train)), random_state=_shap_seed)
                    X_exp = X_test.sample(n=min(shap_cfg.explain_size, len(X_test)), random_state=_shap_seed)

                    union_models = get_union_models(rash_sets)
                    logger.info(f"  SHAP for {len(union_models)} models...")
                    shap_results = compute_shap_for_models(predictor, union_models, X_bg, X_exp, shap_cfg, split_dir / "shap_cache")

                    for eps, sub in rash_sets.items():
                        for _, r in sub.iterrows():
                            m = str(r["model"])
                            if m not in shap_results:
                                continue
                            res = shap_results[m]
                            for feat, imp in zip(res.feature_names, res.global_importance):
                                global_imp_rows.append({"split_id": split.split_id, "seed": seed, "eps": eps, "model": m, "feature": feat, "global_importance": float(imp)})
                            test_mae = float(test_metrics[test_metrics["model"] == m]["mae"].iloc[0])
                            metrics_rows.append({"split_id": split.split_id, "seed": seed, "eps": eps, "model": m, "val_mae": float(r["mae"]), "test_mae": test_mae})

                    del predictor, train_out, shap_results
                    gc.collect()
                except Exception as e:
                    logger.error(f"  Failed: {e}")

        # STEP 5: Save & Aggregate
        logger.info("=" * 70)
        logger.info("STEP 5: Save & Aggregate")
        logger.info("=" * 70)

        global_imp_df = pd.DataFrame(global_imp_rows)
        metrics_df = pd.DataFrame(metrics_rows)
        rashomon_df = pd.DataFrame(rashomon_rows)

        global_imp_df.to_csv(dirs['importance'] / "raw_importance.csv", index=False)
        rashomon_df.to_csv(dirs['rashomon'] / "rashomon_models.csv", index=False)
        metrics_df.to_csv(dirs['rashomon'] / "model_metrics.csv", index=False)

        logger.info(f"Saved {len(global_imp_df)} importance records")

        if global_imp_df.empty:
            logger.warning("No SHAP results!")
            return

        summary_df = aggregate_global_importances(global_imp_df, cfg["shap"]["quantiles"])
        summary_df.to_csv(dirs['importance'] / "aggregated_summary.csv", index=False)

        # STEP 6: Stability & Reports
        logger.info("=" * 70)
        logger.info("STEP 6: Reports")
        logger.info("=" * 70)

        # Rank the features within each model. Then average the ranks
        # (eq:rank_aggregation). If we rank the mean of raw SHAP magnitudes
        # instead, one model on a different numeric scale can determine the
        # aggregate. See the warning on mean_rank_per_split().
        rank_df = rank_then_mean_per_split(global_imp_df)
        rank_df.to_csv(dirs['importance'] / "feature_ranks.csv", index=False)

        # The Jaccard computation uses the top 30% of features. k=None uses the dynamic default.
        stability = stability_over_splits(rank_df)
        save_json(stability, dirs['stability'] / "temporal_stability.json")

        stability_summary = compute_stability_summary(stability)
        if not stability_summary.empty:
            stability_summary.to_csv(dirs['stability'] / "stability_summary.csv", index=False)

        # Advanced Rashomon Metrics
        logger.info("Computing advanced Rashomon metrics...")

        # The Jaccard computation uses the top 30% of features. k=None uses the dynamic default.
        eps_sensitivity = compute_epsilon_sensitivity(rank_df, eps_list)
        if not eps_sensitivity.empty:
            eps_sensitivity.to_csv(dirs['stability'] / "epsilon_sensitivity.csv", index=False)
            logger.info(f"  Epsilon sensitivity: {len(eps_sensitivity)} thresholds analysed")

        if "item_id" in rank_df.columns or "item_id" in sup.columns:
            # The Jaccard computation uses the top 30% of features. k=None uses the dynamic default.
            grouped_stability = compute_grouped_stability(rank_df, group_col="item_id")
            save_json(grouped_stability, dirs['stability'] / "grouped_stability.json")
            if grouped_stability.get("group_comparison"):
                logger.info(f"  Grouped stability computed: within vs across comparison available")

        # PHASE 1: Rashomon Set Characterization
        logger.info("Phase 1: Rashomon Set Characterization...")

        for eps in eps_list:
            eps_str = f"eps_{eps:.2f}".replace(".", "_")

            plot_importance_bands(
                summary_df,
                dirs['fig_importance'] / f"importance_{eps_str}.png",
                splits[-1].split_id, seeds[0], eps, cfg["report"]["topk_features"]
            )

        # PHASE 2: Within-Set Agreement
        # The importance-distribution metric in this phase describes one Rashomon
        # set. This set is the final split at the first seed, for each epsilon.

        logger.info("Phase 2: Within-Set Agreement...")

        for eps in eps_list:
            eps_str = f"eps_{eps:.2f}".replace(".", "_")

            dist_df = compute_importance_distribution_data(
                global_imp_df, split_id=splits[-1].split_id, seed=seeds[0], eps=eps,
                top_k=cfg["report"]["topk_features"]
            )
            if not dist_df.empty:
                plot_importance_violin(
                    dist_df,
                    dirs['fig_importance'] / f"violin_{eps_str}.png",
                    title=f"Feature Importance Distribution (ε={eps})"
                )

        # PHASE 3: Temporal Stability
        logger.info("Phase 3: Temporal Stability...")

        for eps in eps_list:
            eps_str = f"eps_{eps:.2f}".replace(".", "_")

            rank_matrix = compute_rank_stability_matrix(
                rank_df, top_k=cfg["report"]["topk_features"], eps=eps, seed=seeds[0]
            )
            if not rank_matrix.empty:
                rank_matrix.to_csv(dirs['rank_matrices'] / f"{eps_str}.csv")
                plot_rank_stability_heatmap(
                    rank_matrix,
                    dirs['fig_stability'] / f"rank_heatmap_{eps_str}.png",
                    title=f"Feature Rank Stability Across Splits (ε={eps})"
                )
                if 'rank_std' in rank_matrix.columns:
                    unstable = rank_matrix.nlargest(3, 'rank_std')
                    logger.info(f"  ε={eps}: Most unstable: {', '.join(unstable.index[:3].tolist())}")

        if not eps_sensitivity.empty:
            plot_epsilon_stability_comparison(
                eps_sensitivity,
                dirs['fig_stability'] / "epsilon_comparison.png",
                title="Stability Metrics vs. Rashomon Tolerance"
            )

        if stability.get("consecutive"):
            plot_stability_over_time(
                stability,
                dirs['fig_stability'] / "temporal_stability.png",
                eps=eps_list[1] if len(eps_list) > 1 else eps_list[0],
                seed=seeds[0]
            )

        logger.info("All visualizations generated.")

        # Generate reports
        report = generate_final_report(dirs['reports'], cfg, summary_df, metrics_df, stability, dirs['root'].name)
        logger.info(f"\n{report}")

        logger.info("=" * 70)
        logger.info("COMPLETE!")
        logger.info(f"Results: {run_dir}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
