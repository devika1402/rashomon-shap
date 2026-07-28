"""
Rolling-origin cross-validation for time series.

This implements the expanding-window partitions of the thesis (ch. 3.4,
eq:train_set / eq:val_set / eq:test_set). For fold k with origin T_k:
    train = { t : t <= T_k }
    val   = { T_k + 1, ..., T_k + v }
    test  = { T_k + v + 1, ..., T_k + v + h }
Training data always precedes the validation and test data. No future
information therefore leaks into model fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class SplitDef:
    """Definition of a single train/val/test split."""
    split_id: int
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    def __repr__(self) -> str:
        return (
            f"Split({self.split_id}: "
            f"train≤{self.train_end.date()}, "
            f"val=[{self.val_start.date()}, {self.val_end.date()}], "
            f"test=[{self.test_start.date()}, {self.test_end.date()}])"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialisation."""
        return {
            "split_id": self.split_id,
            "train_end": str(self.train_end),
            "val_start": str(self.val_start),
            "val_end": str(self.val_end),
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
        }


def build_splits(
    df: pd.DataFrame,
    n_splits: int,
    min_train_time: int,
    val_size: int,
    test_size: int,
) -> List[SplitDef]:
    """
    Build rolling-origin splits from panel data.

    This uses only the timestamps that exist for ALL series. This keeps the
    training, validation, and test sets consistent across the panel.

    IMPORTANT: If label_timestamp exists, the function bases the splits on the
    label time (t+horizon) rather than the origin time (t). This prevents
    horizon leakage. Horizon leakage is when training rows near the split
    boundary have labels from the future.

    The function creates the splits with:
    - Expanding training window (grows with each split)
    - Fixed validation and test window sizes
    - Non-overlapping test windows

    Timeline:
    ```
    Split 0: [====TRAIN====][VAL][TEST]
    Split 1: [=====TRAIN=====][VAL][TEST]
    Split 2: [======TRAIN======][VAL][TEST]
    ```

    Args:
        df: Panel DataFrame with item_id and timestamp columns
        n_splits: Number of temporal splits to create
        min_train_time: Minimum number of time steps for first training set
        val_size: Number of time steps for validation
        test_size: Number of time steps for test

    Returns:
        List of SplitDef objects

    Raises:
        ValueError: If not enough timestamps for requested configuration

    Example:
        >>> splits = build_splits(df, n_splits=5, min_train_time=36,
        ...                       val_size=3, test_size=3)
        >>> len(splits)
        5
    """
    id_col = "item_id"
    # Use label_timestamp if available (prevents horizon leakage)
    # Otherwise fall back to timestamp for backward compatibility
    time_col = "label_timestamp" if "label_timestamp" in df.columns else "timestamp"
    
    # Find timestamps present in ALL series
    n_series = df[id_col].nunique()
    counts = df.groupby(time_col)[id_col].nunique().sort_index()
    common_times = counts[counts == n_series].index.to_list()
    
    # Validate we have enough time points
    # Each split needs: train_end + val + test
    # Split i has train_end at min_train_time + i*test_size - 1
    needed = min_train_time + val_size + (n_splits * test_size)
    have = len(common_times)
    
    if have < needed:
        raise ValueError(
            f"Insufficient common timestamps for {n_splits} splits. "
            f"Have {have}, need {needed} "
            f"(min_train={min_train_time}, val={val_size}, "
            f"test={test_size} × {n_splits} splits)"
        )
    
    splits: List[SplitDef] = []
    
    for s in range(n_splits):
        # Calculate indices into common_times
        # Test window slides forward with each split
        test_start_idx = min_train_time + val_size + s * test_size
        test_end_idx = test_start_idx + test_size - 1
        
        # Validation immediately before test
        val_end_idx = test_start_idx - 1
        val_start_idx = val_end_idx - val_size + 1
        
        # Training ends before validation
        train_end_idx = val_start_idx - 1
        
        splits.append(
            SplitDef(
                split_id=s,
                train_end=common_times[train_end_idx],
                val_start=common_times[val_start_idx],
                val_end=common_times[val_end_idx],
                test_start=common_times[test_start_idx],
                test_end=common_times[test_end_idx],
            )
        )
    
    return splits


def slice_split(
    df: pd.DataFrame,
    split: SplitDef
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Slice DataFrame according to the split definition.

    IMPORTANT: This uses label_timestamp if available. This prevents horizon
    leakage. The split boundaries correspond to when the labels occur. They do
    not correspond to when the predictions are made.

    Args:
        df: Full panel DataFrame
        split: Split definition with time boundaries

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Use label_timestamp if available (prevents horizon leakage)
    time_col = "label_timestamp" if "label_timestamp" in df.columns else "timestamp"
    t = df[time_col]

    train = df[t <= split.train_end].copy()
    val = df[(t >= split.val_start) & (t <= split.val_end)].copy()
    test = df[(t >= split.test_start) & (t <= split.test_end)].copy()

    return train, val, test


def get_split_sizes(
    df: pd.DataFrame,
    splits: List[SplitDef]
) -> pd.DataFrame:
    """
    Calculate the sizes for each split's train/val/test sets.
    
    This helps to validate the split construction.
    """
    rows = []
    for split in splits:
        train, val, test = slice_split(df, split)
        rows.append({
            "split_id": split.split_id,
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            "train_series": train["item_id"].nunique() if "item_id" in train.columns else 1,
            "val_series": val["item_id"].nunique() if "item_id" in val.columns else 1,
            "test_series": test["item_id"].nunique() if "item_id" in test.columns else 1,
        })
    return pd.DataFrame(rows)
