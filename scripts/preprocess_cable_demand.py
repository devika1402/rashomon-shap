import numpy as np
import pandas as pd
from pathlib import Path
from autogluon.timeseries import TimeSeriesDataFrame
import warnings

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
META_DIR = DATA_DIR / "metadata"
PROC_DIR = DATA_DIR / "processed"

TARGET_KPI = "KPI_0001"  # Cables tons.

# thresholds for sparse/noisy filtering
MIN_COVERAGE = 0.5
MIN_ITEMS = 3

# threshold for deciding when to log-transform a feature
SKEW_THRESHOLD = 1.0  # abs(skew) >= 1 => log1p
# columns that must never be scaled
EXCLUDE_FROM_SCALING = {"item_id", "timestamp", "target"}

def excel_serial_to_month_end(s):
    # s is Excel serial date (e.g. 44957)
    ts = pd.to_datetime(s, unit="D", origin="1899-12-30")
    return ts.dt.to_period("M").dt.to_timestamp("M")

def safe_div(num, den):
    return np.where(den == 0, np.nan, num / den)

# add comprehensive validation
def validate_panel_structure(df: pd.DataFrame) -> None:
    """Validate panel before expensive preprocessing."""
    required = ["item_id", "timestamp", TARGET_KPI]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check for duplicate timestamps per series
    dups = df.groupby("item_id")["timestamp"].apply(
        lambda x: x.duplicated().sum()
    )
    if dups.sum() > 0:
        raise ValueError(f"Duplicate timestamps found in series: {dups[dups > 0].to_dict()}")
    
    # Warn about extreme imbalance
    lengths = df.groupby("item_id").size()
    if lengths.max() / lengths.min() > 2:
        warnings.warn(f"Unbalanced panel: {lengths.min()}-{lengths.max()} obs per series")


def main():
    # 1. Read raw and metadata
    df = pd.read_csv(RAW_DIR / "data_10plants.csv")
    meta = pd.read_csv(META_DIR / "kpi_meta.csv")

    # Ensure numeric dtypes for metadata where needed
    meta["Numeric_Id"] = pd.to_numeric(meta["Numeric_Id"], errors="coerce")
    meta["factor_to_base"] = pd.to_numeric(meta["factor_to_base"], errors="coerce").fillna(1.0)

    # 2. Join metadata
    df = df.merge(meta, left_on="kpi_code", right_on="KPI_Code", how="left")

    missing_meta = df[df["Numeric_Id"].isna()]["kpi_code"].unique()
    if len(missing_meta) > 0:
        raise ValueError(f"KPI codes without metadata: {missing_meta}. "
                         f"Please add them to kpi_meta.csv before proceeding.")

    # 3. Convert the date to month-end. Set item_id from the plant.
    df["timestamp"] = excel_serial_to_month_end(df["date"])
    df["item_id"] = df["plant_id"].astype(str)

    # 4. Convert to base units
    df["value_base"] = df["value"] * df["factor_to_base"]

    # 5. Pivot to wide (base units)
    wide = (
        df.pivot_table(
            index=["item_id", "timestamp"],
            columns="kpi_code",
            values="value_base",
            aggfunc="sum"  # Monthly total. Change this if needed.
        )
        .sort_index()
        .reset_index()
    )

    # 6. Target column
    if TARGET_KPI not in wide.columns:
        raise ValueError(f"Target KPI {TARGET_KPI} not found in wide panel.")
    wide = wide.rename(columns={TARGET_KPI: "target"})

    # 7. Drop degenerate KPI columns
    kpi_cols = [c for c in wide.columns if c.startswith("KPI_") and c != TARGET_KPI]
    drop_cols = []
    for c in kpi_cols:
        col = wide[c]
        if col.isna().all() or col.nunique(dropna=True) <= 1:
            drop_cols.append(c)
    if drop_cols:
        print("[INFO] Dropping all-NaN/constant KPIs:", drop_cols)
        wide = wide.drop(columns=drop_cols)

    # 8. Add calendar features (known covariates)
    wide["cal_month"] = wide["timestamp"].dt.month.astype("int8")
    wide["cal_year"] = wide["timestamp"].dt.year.astype("int16")
    wide["cal_quarter"] = wide["timestamp"].dt.quarter.astype("int8")

    # 9. Add simple ratio / aggregate features
    cols = wide.columns

    # Overtime / workable hours
    if {"KPI_0060", "KPI_0061"}.issubset(cols):
        wide["feat_overtime_ratio"] = safe_div(wide["KPI_0060"], wide["KPI_0061"])

    # Absenteeism / workable hours
    if {"KPI_0057", "KPI_0061"}.issubset(cols):
        wide["feat_absenteeism_ratio"] = safe_div(wide["KPI_0057"], wide["KPI_0061"])

    # Total material tons = strategic + other
    if {"KPI_0065", "KPI_0066"}.issubset(cols):
        wide["feat_total_material_tons"] = (
            wide["KPI_0065"].fillna(0) + wide["KPI_0066"].fillna(0)
        )

    # Total scrap per ton of cables
    # Note: we renamed KPI_0001 to "target" above. We use "target" here.
    if "KPI_0073" in cols and "target" in wide.columns:
        wide["feat_scrap_rate_total"] = safe_div(wide["KPI_0073"], wide["target"])

    # Plant scrap per ton of cables
    # Note: we renamed KPI_0001 to "target" above. We use "target" here.
    if "KPI_0075" in cols and "target" in wide.columns:
        wide["feat_plant_scrap_rate"] = safe_div(wide["KPI_0075"], wide["target"])

    # 10. Filter very sparse / noisy features (RAW stats)
    df_all = wide.copy()
    features = [
        c for c in df_all.columns
        if c not in ["item_id", "timestamp", "target"]
    ]

    quality_rows = []
    for col in features:
        s = df_all[col]
        coverage = s.notna().mean()
        n_items = df_all.loc[s.notna(), "item_id"].nunique()
        var = s.var(skipna=True)
        mean = s.mean(skipna=True)
        std = s.std(skipna=True)
        vmin = s.min(skipna=True)
        vmax = s.max(skipna=True)

        # === negative values & skewness (raw) ===
        non_nan = s.dropna()
        has_negative = (non_nan < 0).any()
        neg_fraction = float((non_nan < 0).mean()) if len(non_nan) > 0 else 0.0
        skew = s.skew(skipna=True)

        keep = (
            (coverage >= MIN_COVERAGE) and
            (n_items >= MIN_ITEMS) and
            (var is not None and var > 0)
        )

        quality_rows.append({
            "feature_name": col,
            "coverage": coverage,
            "n_items": n_items,
            "variance": var,
            "mean": mean,
            "std": std,
            "min": vmin,
            "max": vmax,
            "has_negative": has_negative,
            "neg_fraction": neg_fraction,
            "skew": skew,
            "keep": keep,
        })

    feature_quality_raw = pd.DataFrame(quality_rows)

    keep_features = feature_quality_raw.loc[feature_quality_raw["keep"], "feature_name"].tolist()
    print(f"[INFO] Keeping {len(keep_features)} features out of {len(features)}")

    # === NEW: save the initial raw feature quality for inspection ===
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    feature_quality_raw.to_parquet(PROC_DIR / "feature_quality_raw.parquet", index=False)

    # Base final frame before further preprocessing
    df_final = df_all[["item_id", "timestamp", "target"] + keep_features].copy()

    # === NEW 10a. Remove duplicate features (identical columns) ===
    dup_pairs = []
    to_drop_dups = set()
    # Only compare among features actually kept
    for i, f1 in enumerate(keep_features):
        if f1 in to_drop_dups:
            continue
        for f2 in keep_features[i + 1:]:
            if f2 in to_drop_dups:
                continue
            s1 = df_final[f1]
            s2 = df_final[f2]
            # equals() treats NaNs at same positions as equal
            if s1.equals(s2):
                dup_pairs.append((f1, f2))
                to_drop_dups.add(f2)

    if dup_pairs:
        print("[INFO] Dropping duplicate features (identical series):")
        for f1, f2 in dup_pairs:
            print(f"       {f2} (duplicate of {f1})")
        df_final = df_final.drop(columns=list(to_drop_dups))
        keep_features = [f for f in keep_features if f not in to_drop_dups]

    # === NEW 10b. Log-transform skewed positive features (in-place) ===
    log_transformed = set()
    for f in keep_features:
        s = df_final[f]
        non_nan = s.dropna()
        if non_nan.empty:
            continue
        vmin = non_nan.min()
        skew = non_nan.skew()
        if (vmin > 0) and (skew is not None) and (abs(skew) >= SKEW_THRESHOLD):
            # use log1p for stability
            df_final[f] = np.log1p(s)
            log_transformed.add(f)

    if log_transformed:
        print(f"[INFO] Log-transformed (log1p) {len(log_transformed)} skewed positive features.")
        print("       Examples:", list(log_transformed)[:10])

    # === NEW 10c. Standardise / scale features (z-score) ===
    scaler_rows = []
    for f in keep_features:
        if f in EXCLUDE_FROM_SCALING:
            continue
        s = df_final[f]
        mean = s.mean(skipna=True)
        std = s.std(skipna=True)
        if std is not None and std > 0:
            df_final[f] = (s - mean) / std
        else:
            # If std=0, leave the feature as is.
            # The feature is constant after the previous filters.
            # This case should almost never happen.
            pass
        scaler_rows.append({
            "feature_name": f,
            "mean": mean,
            "std": std,
        })

    feature_scaler = pd.DataFrame(scaler_rows)
    feature_scaler.to_parquet(PROC_DIR / "feature_scaler.parquet", index=False)
    print("[INFO] Saved feature_scaler.parquet with scaling parameters.")

    # === NEW 10d. Recompute feature_quality AFTER transforms for downstream inspection ===
    quality_rows_final = []
    for col in keep_features:
        s = df_final[col]
        coverage = s.notna().mean()
        n_items = df_final.loc[s.notna(), "item_id"].nunique()
        var = s.var(skipna=True)
        mean = s.mean(skipna=True)
        std = s.std(skipna=True)
        vmin = s.min(skipna=True)
        vmax = s.max(skipna=True)
        non_nan = s.dropna()
        has_negative = (non_nan < 0).any()
        neg_fraction = float((non_nan < 0).mean()) if len(non_nan) > 0 else 0.0
        skew = s.skew(skipna=True)

        quality_rows_final.append({
            "feature_name": col,
            "coverage": coverage,
            "n_items": n_items,
            "variance": var,
            "mean": mean,
            "std": std,
            "min": vmin,
            "max": vmax,
            "has_negative": has_negative,
            "neg_fraction": neg_fraction,
            "skew": skew,
            "keep": True,  # we keep all features at this stage
        })

    feature_quality = pd.DataFrame(quality_rows_final)
    feature_quality.to_parquet(PROC_DIR / "feature_quality.parquet", index=False)
    print("[INFO] Saved feature_quality.parquet (post-transform stats).")

    # 11. Build TimeSeriesDataFrame and enforce monthly frequency
    ts = TimeSeriesDataFrame.from_data_frame(
        df_final,
        id_column="item_id",
        timestamp_column="timestamp"
    )
    freq = ts.freq or ts.infer_frequency()
    if str(freq) not in ("M", "ME"):
        ts = ts.convert_frequency("ME")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    ts.to_parquet(PROC_DIR / "panel_final.parquet")

    # 12. Feature metadata for XAI (raw + derived + calendar + transform info)
    meta_rows = []

    # For convenience, convert feature sets to lookups
    log_transformed_set = set(log_transformed)
    keep_feature_set = set(keep_features)

    # raw KPI features
    for f in keep_features:
        if f.startswith("KPI_"):
            row = meta.loc[meta["KPI_Code"] == f]
            if row.empty:
                # The earlier check should prevent this. We keep this guard just in case.
                continue
            row = row.iloc[0]
            meta_rows.append({
                "feature_name": f,
                "source_type": "raw_kpi",
                "Description": row.get("Description", None),
                "Role": row.get("Role", None),
                "base_unit": row.get("base_unit", None),
                "formula": f,
                "transform": "log1p" if f in log_transformed_set else "none",
                "scaled": True,  # we scale all numeric features in df_final (except target)
            })

    # derived & calendar features
    derived_info = {
        "feat_overtime_ratio": (
            "Overtime / Workable hours", "ratio", "KPI_0060 / KPI_0061"
        ),
        "feat_absenteeism_ratio": (
            "Absenteeism / Workable hours", "ratio", "KPI_0057 / KPI_0061"
        ),
        "feat_total_material_tons": (
            "Total material consumption (tons)", "tons", "KPI_0065 + KPI_0066"
        ),
        "feat_scrap_rate_total": (
            "Total scrap per ton of cables", "ratio", "KPI_0073 / KPI_0001"
        ),
        "feat_plant_scrap_rate": (
            "Plant scrap per ton of cables", "ratio", "KPI_0075 / KPI_0001"
        ),
        # calendar features
        "cal_month": ("Calendar month (1-12)", "month_index", "month(timestamp)"),
        "cal_year": ("Calendar year", "year", "year(timestamp)"),
        "cal_quarter": ("Calendar quarter (1-4)", "quarter_index", "quarter(timestamp)"),
    }

    for f in keep_features:
        if f in derived_info:
            desc, unit, formula = derived_info[f]
            meta_rows.append({
                "feature_name": f,
                "source_type": "derived",
                "Description": desc,
                "Role": None,
                "base_unit": unit,
                "formula": formula,
                "transform": "log1p" if f in log_transformed_set else "none",
                "scaled": True,
            })

    feature_meta = pd.DataFrame(meta_rows)
    feature_meta.to_parquet(PROC_DIR / "feature_meta.parquet", index=False)

    print("[INFO] Saved panel_final.parquet, feature_meta.parquet, feature_quality.parquet, feature_quality_raw.parquet")

if __name__ == "__main__":
    main()
