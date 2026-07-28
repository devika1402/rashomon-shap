#!/usr/bin/env python3
"""
Benchmark Testing Script for Rashomon × SHAP Pipeline

Tests the pipeline on standard forecasting benchmarks to validate:
1. Pipeline works on different data structures
2. Results are consistent across datasets
3. Performance is competitive with baselines

Usage (run from project root):
    # Test single benchmark
    python scripts/run_benchmarks.py --dataset M4_Monthly

    # Test multiple benchmarks
    python scripts/run_benchmarks.py --dataset M4_Monthly Electricity ETTh1

    # Quick test mode (reduced settings)
    python scripts/run_benchmarks.py --dataset M4_Monthly --quick

    # Run all available benchmarks
    python scripts/run_benchmarks.py --all
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

# Add src to path (scripts/ is one level below project root)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark_datasets import get_benchmark_dataset, validate_benchmark_panel


# Configure basic logging. Proper logging will replace this later.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


AVAILABLE_BENCHMARKS = [
    'M4_Monthly',
    'M4_Quarterly',
    'M4_Yearly',
    'Electricity',
    'ETTh1',
    'ETTh2',
    'ETTm1',
]


def create_benchmark_config(
    dataset_name: str,
    panel: pd.DataFrame,
    panel_file: Path,
    quick: bool = False
) -> Dict:
    """Create configuration for benchmark testing.

    Args:
        dataset_name: Name of the benchmark dataset
        panel: Loaded panel data
        panel_file: Path to saved panel parquet file
        quick: If True, use reduced settings for fast testing

    Returns:
        Configuration dictionary matching run_all.py expected format
    """
    n_series = panel['series_id'].nunique()

    # Use MINIMUM series length for unbalanced panels (M4 has varying lengths)
    series_lengths = panel.groupby('series_id').size()
    n_timepoints_min = series_lengths.min()
    n_timepoints_max = series_lengths.max()

    # Log if panel is unbalanced
    if n_timepoints_max > n_timepoints_min * 1.5:
        logger.info(f"Unbalanced panel detected: {n_timepoints_min} to {n_timepoints_max} points")
        logger.info(f"Using conservative settings based on minimum length: {n_timepoints_min}")
        n_timepoints = n_timepoints_min
    else:
        n_timepoints = n_timepoints_max

    # IMPORTANT: Account for data loss from lag creation
    # After tabularisation with target_lags, we lose ~6-7 time points
    # Use conservative estimate based on expected lags
    expected_lag_loss = min(7, n_timepoints // 4)
    n_timepoints_after_lags = max(10, n_timepoints - expected_lag_loss)
    logger.info(f"Accounting for lag creation: {n_timepoints} points → ~{n_timepoints_after_lags} points after lags")
    n_timepoints = n_timepoints_after_lags

    # Conservative split sizes for benchmark data
    # For unbalanced panels, we need to ensure splits work for ALL series
    # Start with desired number of splits
    n_splits = 2 if quick else 3

    # Calculate the safe split sizes. This makes sure there is enough data.
    # Formula: min_train + (val + test) * n_splits <= n_timepoints
    # Conservative: use 40% train, 15% val, 15% test (leaving 30% buffer)
    safe_train = max(10, int(n_timepoints * 0.4))
    safe_val = max(6, int(n_timepoints * 0.15))
    safe_test = max(6, int(n_timepoints * 0.15))

    # Verify this will work with the desired n_splits
    required = safe_train + (safe_val + safe_test) * n_splits

    if required > n_timepoints:
        # Try reducing to 2 splits
        logger.warning(f"Series too short for {n_splits} splits (need {required}, have {n_timepoints})")
        n_splits = 2
        required = safe_train + (safe_val + safe_test) * n_splits

        if required > n_timepoints:
            # Still too much, reduce split sizes
            safe_train = max(10, int(n_timepoints * 0.5))
            safe_val = max(6, int(n_timepoints * 0.15))
            safe_test = max(6, int(n_timepoints * 0.15))
            required = safe_train + (safe_val + safe_test) * n_splits

            if required > n_timepoints:
                # Last resort: use minimal settings with 2 splits
                safe_train = max(8, int(n_timepoints * 0.4))
                safe_val = max(4, int(n_timepoints * 0.15))
                safe_test = max(4, int(n_timepoints * 0.15))
                required = safe_train + (safe_val + safe_test) * n_splits

                if required > n_timepoints:
                    # Ultra-minimal: reduce to absolute minimum (1 split only)
                    n_splits = 1
                    safe_train = max(8, int(n_timepoints * 0.5))
                    safe_val = max(3, int(n_timepoints * 0.2))
                    safe_test = max(3, int(n_timepoints * 0.2))
                    logger.warning(f"Series extremely short - using single split: train={safe_train}, val={safe_val}, test={safe_test}")
                else:
                    logger.warning(f"Using minimal split settings: train={safe_train}, val={safe_val}, test={safe_test}, n_splits={n_splits}")

    # Create config matching the expected structure
    # Note: outdir should be the PARENT directory. run_all.py creates outdir/run_name.
    config = {
        'project': {
            'outdir': 'results',
            'run_name': f'benchmark_{dataset_name.lower()}',
            'random_seed': 42
        },

        'data': {
            'source': 'real',
            'real': {
                'panel_path': str(panel_file),
                'metadata_path': None,  # Benchmarks don't have metadata
                'target_col': 'target',
                'id_col': 'series_id',
                'time_col': 'time_idx'
            },
            'horizon': 1,
            'target_lags': min(6, max(1, n_timepoints // 4)),
            'cov_lag': 1,
            'freq': 'D'  # Default to daily
        },

        'splits': {
            'n_splits': n_splits,
            'min_train_time': safe_train,
            'val_size': safe_val,
            'test_size': safe_test
        },

        'automl': {
            'time_limit_s': 300 if quick else 600,
            'presets': 'medium_quality',
            'seeds': [0] if quick else [0, 1, 2],
            'eval_metric': 'mae',
            'max_models_per_rashomon': 10 if quick else 15,
            'num_gpus': 0,
            'num_cpus': 'auto',
            'fit_strategy': 'sequential',
            'verbosity': 1
        },

        'rashomon': {
            'eps_list': [0.02, 0.05, 0.10]
        },

        'shap': {
            'background_size': min(100, len(panel) // 10),
            'explain_size': min(50, len(panel) // 20),
            'max_evals': 500,
            'batch_size': 50,
            'quantiles': [0.1, 0.5, 0.9],
            'prefer_tree': True,
            'chunk_size': 50,
            'show_progress': True
        },

        'report': {
            'topk_features': 10
        }
    }

    return config


def convert_to_python_types(obj):
    """Recursively convert numpy types to Python native types for YAML serialisation.

    Args:
        obj: Object to convert (dict, list, numpy type, or primitive)

    Returns:
        Object with all numpy types converted to Python native types
    """
    import numpy as np

    if isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_benchmark_config(config: Dict, output_file: Path):
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_file: Path to save config
    """
    # Convert numpy types to Python native types
    config_clean = convert_to_python_types(config)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        yaml.dump(config_clean, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {output_file}")


def run_single_benchmark(
    dataset_name: str,
    quick: bool = False,
    n_series: Optional[int] = None
) -> bool:
    """Run pipeline on a single benchmark dataset.

    Args:
        dataset_name: Name of benchmark (e.g., 'M4_Monthly')
        quick: If True, use reduced settings
        n_series: Number of series to load (None = use default)

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running benchmark: {dataset_name}")
        logger.info(f"{'='*60}\n")

        # Load dataset
        kwargs = {}
        if n_series is not None:
            kwargs['n_series'] = n_series

        panel, metadata = get_benchmark_dataset(dataset_name, **kwargs)
        validate_benchmark_panel(panel)

        # Print dataset info
        logger.info(f"Dataset: {metadata['name']}")
        logger.info(f"Description: {metadata['description']}")
        logger.info(f"Source: {metadata['source']}")
        logger.info(f"Series: {panel['series_id'].nunique()}")
        logger.info(f"Time points: {panel.groupby('series_id').size().max()}")

        # Extract feature names (avoid nested brackets in f-string)
        feature_cols = [c for c in panel.columns if c.startswith('cov_') or c not in ['series_id', 'time_idx', 'target']]
        logger.info(f"Features: {', '.join(feature_cols)}\n")

        # Create organised output directories (match run_all layout)
        run_name = f"benchmark_{dataset_name.lower()}"
        run_dir = Path("results") / run_name

        # Clean up existing results to allow overwrite
        if run_dir.exists():
            import shutil
            logger.info(f"Removing existing results at {run_dir}")
            shutil.rmtree(run_dir)

        # Create fresh directory structure
        run_dir.mkdir(parents=True, exist_ok=True)
        data_dir = run_dir / "01_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save panel data inside 01_data so downstream artifacts stay contained
        panel_file = data_dir / "panel.parquet"
        panel.to_parquet(panel_file, index=False)
        logger.info(f"Saved panel data to {panel_file}")

        # Create configuration
        config = create_benchmark_config(dataset_name, panel, panel_file, quick)

        # Save config
        config_file = run_dir / "benchmark_config.yaml"
        save_benchmark_config(config, config_file)

        # Run pipeline by importing and calling directly
        logger.info(f"\nStarting Rashomon × SHAP pipeline...")
        logger.info(f"Results will be saved to: {run_dir}\n")

        # Import run_all module and execute
        from run_all import main as run_pipeline_main

        # Save and restore sys.argv
        old_argv = sys.argv
        sys.argv = ['run_all.py', '--config', str(config_file)]

        try:
            run_pipeline_main()
            logger.info(f"\n✓ Benchmark {dataset_name} completed successfully!")
            logger.info(f"Results saved to: {run_dir}")
            return True
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            traceback.print_exc()
            return False
        finally:
            sys.argv = old_argv

    except Exception as e:
        logger.error(f"\n✗ Benchmark {dataset_name} failed: {e}")
        traceback.print_exc()
        return False


def run_all_benchmarks(quick: bool = False) -> Dict[str, bool]:
    """Run pipeline on all available benchmarks.

    Args:
        quick: If True, use reduced settings

    Returns:
        Dictionary mapping dataset name to success status
    """
    results = {}

    logger.info(f"\n{'='*60}")
    logger.info(f"Running ALL benchmarks ({len(AVAILABLE_BENCHMARKS)} datasets)")
    logger.info(f"Mode: {'Quick' if quick else 'Full'}")
    logger.info(f"{'='*60}\n")

    for dataset_name in AVAILABLE_BENCHMARKS:
        # Use smaller subsets for quick mode
        n_series = 10 if quick else None
        success = run_single_benchmark(dataset_name, quick, n_series)
        results[dataset_name] = success

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"{'='*60}")

    for dataset_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{dataset_name:20s} {status}")

    n_passed = sum(results.values())
    n_total = len(results)
    logger.info(f"\nTotal: {n_passed}/{n_total} passed")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Rashomon × SHAP pipeline on benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available benchmarks:
  {', '.join(AVAILABLE_BENCHMARKS)}

Examples:
  # Test single benchmark
  python run_benchmarks.py --dataset M4_Monthly

  # Test multiple benchmarks
  python run_benchmarks.py --dataset M4_Monthly Electricity

  # Quick test mode
  python run_benchmarks.py --dataset M4_Monthly --quick

  # Run all benchmarks
  python run_benchmarks.py --all
        """
    )

    parser.add_argument(
        '--dataset',
        nargs='+',
        choices=AVAILABLE_BENCHMARKS,
        help='Benchmark dataset(s) to run'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all available benchmarks'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use reduced settings for faster testing'
    )

    parser.add_argument(
        '--n-series',
        type=int,
        help='Number of series to load (default: dataset-specific)'
    )

    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    args = parser.parse_args()

    # Set logging level
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)

    # Validate arguments
    if not args.all and not args.dataset:
        parser.error("Must specify either --dataset or --all")

    # Run benchmarks
    if args.all:
        results = run_all_benchmarks(args.quick)
        sys.exit(0 if all(results.values()) else 1)
    else:
        all_passed = True
        for dataset_name in args.dataset:
            success = run_single_benchmark(dataset_name, args.quick, args.n_series)
            all_passed = all_passed and success

        sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
