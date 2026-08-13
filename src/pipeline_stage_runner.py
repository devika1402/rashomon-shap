"""
Pipeline stage functions.

These functions separate the concerns below:
- Data loading (real, benchmark)
- Output directory organisation
- Report generation

The main (run_all.py) imports these stages.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from benchmark_datasets import M4Dataset, ElectricityDataset, ETTDataset

logger = logging.getLogger("rashomon_shap")


# =============================================================================
# Output Directory Organization
# =============================================================================

def setup_output_structure(run_dir: Path) -> Dict[str, Path]:
    """
    Create an organised output directory structure.

    Returns a dictionary mapping logical names to paths for consistent saving.

    Directory structure:
        results/<run_name>/
        ├── README.md                 # Auto-generated navigation guide
        ├── config.json
        │
        ├── 01_data/                  # Input data artifacts
        ├── 02_models/                # Training artifacts (per seed/split)
        ├── 03_importance/            # SHAP importance results
        ├── 04_stability/             # Stability analysis
        │   └── rank_matrices/        # Per-epsilon rank stability
        ├── 05_rashomon/              # Rashomon set analysis
        ├── 06_figures/               # All visualizations
        │   ├── importance/
        │   ├── stability/
        │   └── rashomon/
        └── 07_reports/               # Summary reports
    """
    dirs = {
        'root': run_dir,
        'data': run_dir / '01_data',
        'models': run_dir / '02_models',
        'importance': run_dir / '03_importance',
        'stability': run_dir / '04_stability',
        'rank_matrices': run_dir / '04_stability' / 'rank_matrices',
        'rashomon': run_dir / '05_rashomon',
        'figures': run_dir / '06_figures',
        'fig_importance': run_dir / '06_figures' / 'importance',
        'fig_stability': run_dir / '06_figures' / 'stability',
        'fig_rashomon': run_dir / '06_figures' / 'rashomon',
        'reports': run_dir / '07_reports',
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def generate_readme(dirs: Dict[str, Path], cfg: Dict) -> None:
    """Generate a README.md with navigation guide for the results."""
    readme = f"""# Rashomon×SHAP Results

**Run:** {dirs['root'].name}
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Quick Navigation

### Key Results

| What you want | Where to look |
|---------------|---------------|
| **Top features** | `03_importance/feature_ranks.csv` |
| **Stability summary** | `04_stability/stability_summary.csv` |
| **Quick visual** | `06_figures/importance/importance_eps_0_05.png` |

### Directory Guide

```
{dirs['root'].name}/
│
├── 01_data/                 # Input data
│   ├── panel.parquet        # Raw panel data
│   ├── supervised.parquet   # Tabularized features
│   └── splits.csv           # Train/val/test splits
│
├── 02_models/               # Trained models (per seed/split)
│
├── 03_importance/           # SHAP importance
│   ├── raw_importance.csv   # Per-model SHAP values
│   ├── aggregated_summary.csv  # Mean + quantiles
│   └── feature_ranks.csv    # Rankings per context
│
├── 04_stability/            # Stability metrics
│   ├── temporal_stability.json  # Spearman, Kendall, Jaccard
│   ├── stability_summary.csv    # Aggregated by (seed, eps)
│   ├── epsilon_sensitivity.csv  # Stability vs epsilon
│   └── rank_matrices/       # Feature ranks across splits
│
├── 05_rashomon/             # Rashomon set analysis
│   ├── rashomon_models.csv  # Models in each set
│   └── model_metrics.csv    # MAE per model
│
├── 06_figures/              # Visualizations
│   ├── importance/          # Bar charts, violin plots
│   ├── stability/           # Heatmaps, comparisons
│   └── rashomon/            # Diversity plots
│
└── 07_reports/              #  summaries
    └── final_report.txt
```

### Interpretation Guide

**Stability Thresholds:**
- **Spearman ρ > 0.90**: Highly stable, report with confidence
- **Spearman ρ 0.70-0.90**: Moderate, report top features with uncertainty
- **Spearman ρ < 0.70**: Variable, avoid definitive claims

**Epsilon (ε) Effects:**
- Larger ε = more models in Rashomon set
- Check `epsilon_sensitivity.csv` for stability trends
"""
    with open(dirs['reports'] / 'README.md', 'w') as f:
        f.write(readme)

    with open(dirs['root'] / 'README.md', 'w') as f:
        f.write(readme)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_real_data(cfg: Dict, data_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load preprocessed panel_final.parquet data."""
    data_cfg = cfg["data"]["real"]
    panel_path = Path(data_cfg["panel_path"])

    if not panel_path.exists():
        raise FileNotFoundError(
            f"Panel not found: {panel_path}\n"
            "Run scripts/preprocess_cable_demand.py first."
        )

    logger.info(f"Loading real data from {panel_path}")
    panel = pd.read_parquet(panel_path)
    id_col = data_cfg.get("id_col", "item_id")
    time_col = data_cfg.get("time_col", "timestamp")
    target_col = data_cfg.get("target_col", "target")

    # Handle index-based columns
    if id_col not in panel.columns or time_col not in panel.columns:
        if isinstance(panel.index, pd.MultiIndex):
            if set([id_col, time_col]).issubset(panel.index.names):
                panel = panel.reset_index()
            elif panel.index.names == [None, None] and panel.index.nlevels == 2:
                panel.index = panel.index.set_names([id_col, time_col])
                panel = panel.reset_index()
        else:
            if panel.index.name in [id_col, time_col]:
                panel = panel.reset_index()

    missing = [c for c in [id_col, time_col, target_col] if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    panel[time_col] = pd.to_datetime(panel[time_col])
    panel = panel.sort_values([id_col, time_col]).reset_index(drop=True)

    metadata = None
    meta_path = data_cfg.get("metadata_path")
    if meta_path and Path(meta_path).exists():
        meta_path = Path(meta_path)
        metadata = pd.read_parquet(meta_path) if str(meta_path).endswith(".parquet") else pd.read_csv(meta_path)
        logger.info(f"Loaded metadata: {len(metadata)} features")

    _generate_data_report(panel, metadata, data_dir, id_col, time_col, target_col)
    return panel, metadata


def load_benchmark_data(cfg: Dict, data_dir: Path) -> Tuple[pd.DataFrame, None]:
    """Load benchmark dataset for methodology validation.

    Supports: M4, Electricity, ETT datasets.
    """
    bench_cfg = cfg["data"]["benchmark"]
    name = bench_cfg["name"]
    n_series = bench_cfg.get("n_series", 50)
    cache_dir = Path(bench_cfg.get("cache_dir", "data/benchmark_cache"))

    logger.info(f"Loading benchmark dataset: {name}")

    if name.startswith("M4"):
        frequency = name.split("_")[1] if "_" in name else "Monthly"
        dataset = M4Dataset(frequency=frequency, n_series=n_series, cache_dir=cache_dir)
    elif name.lower() == "electricity":
        dataset = ElectricityDataset(n_series=n_series, cache_dir=cache_dir)
    elif name.lower().startswith("ett"):
        variant = name if name.startswith("ETT") else "ETTh1"
        dataset = ETTDataset(variant=variant, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown benchmark dataset: {name}. "
                        f"Supported: M4_Monthly, M4_Yearly, Electricity, ETTh1, ETTm1")

    panel = dataset.load()

    rename_map = {}
    if "series_id" in panel.columns and "item_id" not in panel.columns:
        rename_map["series_id"] = "item_id"
    if "time_idx" in panel.columns and "timestamp" not in panel.columns:
        freq = cfg["data"].get("freq", "D")
        if freq in ["H", "h"]:
            panel["timestamp"] = pd.to_datetime(panel["time_idx"], unit="h", origin="2000-01-01")
        elif freq in ["M", "MS"]:
            panel["timestamp"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(panel["time_idx"] * 30, unit="D")
        else:
            panel["timestamp"] = pd.to_datetime(panel["time_idx"], unit="D", origin="2000-01-01")
        panel = panel.drop(columns=["time_idx"])

    if rename_map:
        panel = panel.rename(columns=rename_map)

    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    _generate_data_report(panel, None, data_dir, "item_id", "timestamp", "target")
    logger.info(f"Loaded benchmark: {len(panel)} rows, {panel['item_id'].nunique()} series")

    return panel, None


def _generate_data_report(panel, metadata, data_dir, id_col, time_col, target_col):
    """Generate data quality report."""
    lines = ["=" * 70, "DATA QUALITY REPORT", "=" * 70, ""]

    n_series = panel[id_col].nunique()
    series_ids = panel[id_col].unique().tolist()
    lengths = panel.groupby(id_col)[time_col].count()
    feature_cols = [c for c in panel.columns if c not in [id_col, time_col, target_col]]

    lines.append(f"Shape: {panel.shape}")
    lines.append(f"Series: {n_series} ({series_ids})")
    lines.append(f"Time range: {panel[time_col].min()} to {panel[time_col].max()}")
    lines.append(f"Observations per series: {lengths.iloc[0]} (balanced: {lengths.min() == lengths.max()})")
    lines.append(f"Features: {len(feature_cols)}")
    lines.append("")

    lines.append("-" * 40)
    lines.append("TARGET STATISTICS BY SERIES")
    lines.append("-" * 40)
    target_stats = panel.groupby(id_col)[target_col].agg(['mean', 'std', 'min', 'max']).round(2)
    lines.append(target_stats.to_string())
    lines.append("")

    lines.append("-" * 40)
    lines.append("TOP FEATURE-TARGET CORRELATIONS")
    lines.append("-" * 40)
    corrs = {c: panel[[target_col, c]].corr().iloc[0, 1] for c in feature_cols if panel[c].notna().sum() > 10}
    for feat, c in sorted(corrs.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)[:15]:
        lines.append(f"  {feat}: {c:.3f}")

    if metadata is not None:
        lines.append("")
        lines.append("-" * 40)
        lines.append("FEATURE METADATA (sample)")
        lines.append("-" * 40)
        lines.append(metadata.head(10).to_string())

    lines.append("\n" + "=" * 70)

    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "data_report.txt", "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Data: {n_series} series × {lengths.iloc[0]} months × {len(feature_cols)} features")


# =============================================================================
# Report Generation
# =============================================================================

def generate_final_report(report_dir: Path, cfg, summary_df, metrics_df, stability, run_name: str):
    """Generate final summary report with robust error handling."""
    try:
        lines = ["=" * 70, "RASHOMON × SHAP EXPERIMENT REPORT", "=" * 70, ""]
        lines.append(f"Run: {run_name}")
        lines.append(f"Data source: {cfg['data']['source']}")
        lines.append("")

        # Forecast performance with defensive checks
        if not metrics_df.empty:
            lines.append("-" * 40)
            lines.append("FORECAST PERFORMANCE")
            lines.append("-" * 40)
            try:
                if 'val_mae' in metrics_df.columns and 'test_mae' in metrics_df.columns:
                    lines.append(f"Best validation MAE: {metrics_df['val_mae'].min():.4f}")
                    lines.append(f"Best test MAE: {metrics_df['test_mae'].min():.4f}")
                    lines.append(f"Models evaluated: {len(metrics_df)}")
                else:
                    lines.append(f"Models evaluated: {len(metrics_df)} (metric columns missing)")
            except Exception as e:
                lines.append(f"ERROR computing performance metrics: {e}")
            lines.append("")

        # Top features with defensive checks
        if not summary_df.empty:
            lines.append("-" * 40)
            lines.append("TOP FEATURES (by mean SHAP importance)")
            lines.append("-" * 40)
            try:
                if 'feature' in summary_df.columns and 'mean_importance' in summary_df.columns:
                    top = summary_df.groupby("feature")["mean_importance"].mean().sort_values(ascending=False).head(15)
                    for i, (f, v) in enumerate(top.items(), 1):
                        lines.append(f"  {i:2d}. {f}: {v:.4f}")
                else:
                    lines.append("ERROR: Required columns missing from summary_df")
            except Exception as e:
                lines.append(f"ERROR computing top features: {e}")
            lines.append("")

        # Stability metrics with defensive checks
        if stability and stability.get("consecutive"):
            lines.append("-" * 40)
            lines.append("STABILITY METRICS")
            lines.append("-" * 40)
            try:
                df = pd.DataFrame(stability["consecutive"])
                if not df.empty and 'spearman' in df.columns:
                    lines.append(f"Avg Spearman ρ: {df['spearman'].mean():.3f}")
                    if 'kendall_tau' in df.columns:
                        lines.append(f"Avg Kendall τ: {df['kendall_tau'].mean():.3f}")
                    if 'topk_jaccard' in df.columns:
                        lines.append(f"Avg Top-k Jaccard: {df['topk_jaccard'].mean():.3f}")
                    lines.append("")
                    lines.append("Interpretation:")
                    avg_sp = df['spearman'].mean()
                    if avg_sp > 0.9:
                        lines.append("  → Highly stable rankings: report with confidence")
                    elif avg_sp > 0.7:
                        lines.append("  → Moderately stable: report top features with uncertainty bands")
                    else:
                        lines.append("  → Variable rankings: avoid definitive claims about feature order")
                else:
                    lines.append("Stability data incomplete")
            except Exception as e:
                lines.append(f"ERROR computing stability: {e}")

        lines.append("\n" + "=" * 70)
        report = "\n".join(lines)

        # Ensure directory exists
        report_dir.mkdir(parents=True, exist_ok=True)

        with open(report_dir / "final_report.txt", "w") as f:
            f.write(report)

        logger.info("Final report generated successfully")
        return report

    except Exception as e:
        error_msg = f"CRITICAL ERROR in generate_final_report:\n{traceback.format_exc()}"
        logger.error(error_msg)

        # Try to write error report
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            with open(report_dir / "final_report.txt", "w") as f:
                f.write(f"ERROR GENERATING REPORT\n\n{error_msg}\n")
        except:
            pass

        # Re-raise to ensure error is visible
        raise RuntimeError(f"Report generation failed: {e}") from e
