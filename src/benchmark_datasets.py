#!/usr/bin/env python3
"""
Benchmark dataset loaders.

This module loads the public forecasting benchmarks used in this study:
- M4 Competition (Yearly, Quarterly, Monthly)
- Electricity (UCI LD2011-2014, 20-series subset)
- ETT (Electricity Transformer Temperature): ETTh1, ETTh2, ETTm1

Each loader downloads the data on first use. It then caches the data to data/benchmark_cache/.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkDataset:
    """Base class for benchmark dataset loaders."""

    def __init__(self, cache_dir: Path = Path("data/benchmark_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        """Load and return panel data in standard format.

        Returns:
            DataFrame with columns: [series_id, time_idx, target, cov_1, cov_2, ...]
        """
        raise NotImplementedError

    def get_metadata(self) -> Dict:
        """Return metadata about the dataset."""
        raise NotImplementedError


class M4Dataset(BenchmarkDataset):
    """M4 Competition dataset loader.

    The M4 competition included 100,000 time series across different frequencies.
    We focus on a subset for computational feasibility.

    Reference: https://github.com/Mcompetitions/M4-methods
    """

    def __init__(
        self,
        frequency: str = "Monthly",
        n_series: int = 50,
        cache_dir: Path = Path("data/benchmark_cache")
    ):
        """Initialize M4 dataset loader.

        Args:
            frequency: One of ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
            n_series: Number of series to load (for computational feasibility)
            cache_dir: Directory to cache downloaded data
        """
        super().__init__(cache_dir)
        self.frequency = frequency
        self.n_series = n_series

    def load(self) -> pd.DataFrame:
        """Load M4 data and convert to panel format."""
        logger.info(f"Loading M4 {self.frequency} dataset (first {self.n_series} series)...")

        # Try to load from cache first
        cache_file = self.cache_dir / f"m4_{self.frequency.lower()}_{self.n_series}.parquet"
        if cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            return pd.read_parquet(cache_file)

        try:
            # Download using requests if not cached
            import requests
            import io

            base_url = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
            train_url = f"{base_url}/Train/{self.frequency}-train.csv"

            logger.info(f"Downloading from {train_url}...")
            response = requests.get(train_url, timeout=60)
            response.raise_for_status()

            # Parse the CSV. The first column is the series ID. The other columns are time points.
            df = pd.read_csv(io.StringIO(response.text))

            # Take only first n_series
            df = df.head(self.n_series)

            # Convert wide format to long panel format
            panel_data = []
            for idx, row in df.iterrows():
                series_id = row.iloc[0]
                values = row.iloc[1:].dropna().values

                for time_idx, value in enumerate(values):
                    panel_data.append({
                        'series_id': series_id,
                        'time_idx': time_idx,
                        'target': float(value)
                    })

            panel = pd.DataFrame(panel_data)

            # This loader handles the Monthly subset only. The calendar cycle
            # is therefore fixed at 12. Other M4 frequencies would need their
            # own period value. We create the lag features later, during tabularisation.
            panel['month'] = (panel['time_idx'] % 12) + 1
            panel['trend'] = panel['time_idx']

            # Cache the result
            panel.to_parquet(cache_file, index=False)
            logger.info(f"Cached to {cache_file}")

            return panel

        except Exception as e:
            logger.error(f"Failed to load M4 dataset: {e}")
            raise

    def get_metadata(self) -> Dict:
        return {
            'name': f'M4_{self.frequency}',
            'frequency': self.frequency,
            'n_series': self.n_series,
            'source': 'https://github.com/Mcompetitions/M4-methods',
            'description': 'M4 Forecasting Competition dataset'
        }


class ElectricityDataset(BenchmarkDataset):
    """Electricity demand dataset.

    This dataset is a common benchmark for multivariate time series forecasting.
    It contains hourly electricity consumption data.
    """

    def __init__(
        self,
        n_series: int = 20,
        cache_dir: Path = Path("data/benchmark_cache")
    ):
        super().__init__(cache_dir)
        self.n_series = n_series

    def load(self) -> pd.DataFrame:
        """Load electricity data and convert to panel format."""
        logger.info(f"Loading Electricity dataset (first {self.n_series} series)...")

        cache_file = self.cache_dir / f"electricity_{self.n_series}_v2.parquet"
        if cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            return pd.read_parquet(cache_file)

        try:
            # UCI Electricity Load dataset
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"

            import requests
            import zipfile
            import io

            logger.info(f"Downloading from {url}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Extract and read
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open('LD2011_2014.txt') as f:
                    df = pd.read_csv(f, sep=';', decimal=',', parse_dates=[0], index_col=0)

            # Take first n_series clients by CSV column order (MT_001…MT_N, deterministic).
            # This is standard in the literature (cf. Informer, iTransformer). We use no random sampling.
            df = df.iloc[:, :self.n_series]

            # Skip initial zero-padding period (data collection started ~2012)
            # Find first row where any column has non-zero values
            first_valid_idx = (df.sum(axis=1) > 0).idxmax()
            df = df.loc[first_valid_idx:]
            logger.info(f"Skipped zero-padding, data starts at {first_valid_idx}")

            # Convert to panel format
            panel_data = []
            for client_id, col in enumerate(df.columns):
                series = df[col].dropna()
                for time_idx, (timestamp, value) in enumerate(series.items()):
                    panel_data.append({
                        'series_id': f'client_{client_id}',
                        'time_idx': time_idx,
                        'target': float(value),
                        'hour': timestamp.hour,
                        'day_of_week': timestamp.dayofweek,
                        'month': timestamp.month,
                        'trend': time_idx
                    })

            panel = pd.DataFrame(panel_data)

            # Verify we have non-zero data
            if panel['target'].sum() == 0:
                raise ValueError("All target values are zero after processing")

            # Cache the result
            panel.to_parquet(cache_file, index=False)
            logger.info(f"Cached to {cache_file}")

            return panel

        except Exception as e:
            logger.error(f"Failed to load Electricity dataset: {e}")
            raise

    def get_metadata(self) -> Dict:
        return {
            'name': 'Electricity',
            'n_series': self.n_series,
            'source': 'UCI Machine Learning Repository',
            'description': 'Electricity consumption of 370 clients (hourly)'
        }


class ETTDataset(BenchmarkDataset):
    """Electricity Transformer Temperature (ETT) dataset.

    This dataset is a standard benchmark for long-term time series forecasting.
    It contains oil temperature and other sensor readings from transformers.
    """

    def __init__(
        self,
        variant: str = "ETTh1",
        cache_dir: Path = Path("data/benchmark_cache")
    ):
        """Initialize ETT dataset loader.

        Args:
            variant: One of ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
                    h = hourly, m = 15-minute intervals
            cache_dir: Directory to cache downloaded data
        """
        super().__init__(cache_dir)
        self.variant = variant

    def load(self) -> pd.DataFrame:
        """Load ETT data and convert to panel format."""
        logger.info(f"Loading ETT dataset ({self.variant})...")

        cache_file = self.cache_dir / f"ett_{self.variant.lower()}.parquet"
        if cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            return pd.read_parquet(cache_file)

        try:
            # Download from GitHub repository (ETT-small subdirectory)
            url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{self.variant}.csv"

            logger.info(f"Downloading from {url}...")
            df = pd.read_csv(url, parse_dates=['date'])

            # ETT has multiple features. We treat each feature as a separate series.
            feature_cols = [col for col in df.columns if col != 'date']

            panel_data = []
            for series_id, col in enumerate(feature_cols):
                for time_idx, (_, row) in enumerate(df.iterrows()):
                    # Use the first feature (OT) as the target. Use the other features as covariates.
                    if col == feature_cols[0]:
                        target_val = row[col]
                        record = {
                            'series_id': 'transformer_1',
                            'time_idx': time_idx,
                            'target': float(target_val),
                            'date': row['date']
                        }
                        # Add other features as covariates
                        for other_col in feature_cols[1:]:
                            record[f'cov_{other_col}'] = float(row[other_col])

                        panel_data.append(record)

            panel = pd.DataFrame(panel_data)

            # Add temporal features
            panel['hour'] = pd.to_datetime(panel['date']).dt.hour
            panel['day_of_week'] = pd.to_datetime(panel['date']).dt.dayofweek
            panel['month'] = pd.to_datetime(panel['date']).dt.month
            panel.drop(columns=['date'], inplace=True)

            # Cache the result
            panel.to_parquet(cache_file, index=False)
            logger.info(f"Cached to {cache_file}")

            return panel

        except Exception as e:
            logger.error(f"Failed to load ETT dataset: {e}")
            raise

    def get_metadata(self) -> Dict:
        return {
            'name': f'ETT_{self.variant}',
            'variant': self.variant,
            'source': 'https://github.com/zhouhaoyi/ETDataset',
            'description': 'Electricity Transformer Temperature dataset'
        }


def get_benchmark_dataset(
    dataset_name: str,
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """Load a benchmark dataset by name.

    Args:
        dataset_name: One of ['M4_Monthly', 'M4_Quarterly', 'Electricity',
                              'ETTh1', 'ETTm1']
        **kwargs: Additional arguments passed to dataset loader

    Returns:
        Tuple of (panel DataFrame, metadata dict)
    """
    dataset_map = {
        'M4_Monthly': lambda: M4Dataset(frequency='Monthly', **kwargs),
        'M4_Quarterly': lambda: M4Dataset(frequency='Quarterly', **kwargs),
        'M4_Yearly': lambda: M4Dataset(frequency='Yearly', **kwargs),
        'Electricity': lambda: ElectricityDataset(**kwargs),
        'ETTh1': lambda: ETTDataset(variant='ETTh1', **kwargs),
        'ETTh2': lambda: ETTDataset(variant='ETTh2', **kwargs),
        'ETTm1': lambda: ETTDataset(variant='ETTm1', **kwargs),
    }

    if dataset_name not in dataset_map:
        available = ', '.join(dataset_map.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    dataset = dataset_map[dataset_name]()
    panel = dataset.load()
    metadata = dataset.get_metadata()

    logger.info(f"Loaded {dataset_name}: {len(panel)} observations, "
                f"{panel['series_id'].nunique()} series")

    return panel, metadata


def validate_benchmark_panel(panel: pd.DataFrame) -> bool:
    """Validate that panel data has the required structure.

    Args:
        panel: DataFrame to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_cols = ['series_id', 'time_idx', 'target']
    missing = [col for col in required_cols if col not in panel.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check for nulls in required columns
    for col in required_cols:
        n_null = panel[col].isna().sum()
        if n_null > 0:
            logger.warning(f"Column '{col}' has {n_null} null values")

    # Check series are balanced (same length)
    series_lengths = panel.groupby('series_id').size()
    if series_lengths.nunique() > 1:
        logger.warning(f"Unbalanced panel: series lengths vary from "
                      f"{series_lengths.min()} to {series_lengths.max()}")

    logger.info(f"Panel validation passed: {len(panel)} rows, "
                f"{panel['series_id'].nunique()} series, "
                f"{panel['time_idx'].nunique()} time points")

    return True
