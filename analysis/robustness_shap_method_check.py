#!/usr/bin/env python3
"""
analysis/robustness_shap_method_check.py
─────────────────────────────────────────────────────────────────────────────
Robustness check: does mixing permutation-SHAP and TreeSHAP explainers
inflate or deflate the measured stability metrics?

Pre-analysis of the SHAP npz caches established the following:
    All AutoGluon bagged models (_BAG_L1, _BAG_L2) use permutation SHAP.
    The module-check in _try_tree_explainer fails for bagging wrappers. They
    are not in the supported tree-module list ("xgboost", "lightgbm",
    "catboost", "sklearn.ensemble", "sklearn.tree"). There is therefore no
    explainer-type confound in AutoGluon results. All stability metrics reflect
    permutation SHAP uniformly across every model family.

    For AutoGluon, this script runs a family-based sensitivity check instead:
    full_set   : all models in the Rashomon set
    gbm_family : LightGBM*, XGBoost*, CatBoost*  (would be tree-exact if not bagged)
    forest     : WeightedEnsemble*, RandomForestMSE*, ExtraTreesMSE*

H2O:
    H2O uses predict_contributions() for all tree models. This is exact
    Shapley for GBM/XGBoost and Saabas-approximate for DRF/XRT. H2O uses
    permutation SHAP for GLM. This script evaluates three comparison groups at
    each (split, seed, eps):
      full_set    : all model families (GBM + XGBoost + DRF + XRT + GLM)
      tree_approx : GBM_ + XGBoost_ + DRF_ + XRT_  (drops GLM permutation)
      tree_exact  : GBM_ + XGBoost_ only            (exact Shapley only)

Meaningful shift threshold: |Δ mean Spearman ρ| > 0.05 at ε = 0.05.
A group with < 2 models is recorded as "insufficient_models". It is excluded
from the mean ρ computation. It is still counted in the report.

Reads from : results/bq_*/05_rashomon/raw_importance.csv
             results/h2o_bq_*/05_rashomon/raw_importance.csv
Writes to  : analysis/robustness_tree_only/
"""

from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
OUT_DIR      = PROJECT_ROOT / "analysis" / "robustness_tree_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset registry ──────────────────────────────────────────────────────────
AG_RUNS: dict[str, str] = {
    "ETTh1":        "bq_etth1",
    "ETTh2":        "bq_etth2",
    "ETTm1":        "bq_ettm1",
    "Electricity":  "bq_electricity",
    "M4 Monthly":   "bq_m4_monthly",
    "Cable Demand": "bq_cable_demand",
}

H2O_RUNS: dict[str, str] = {
    "ETTh1":        "h2o_bq_etth1",
    "ETTh2":        "h2o_bq_etth2",
    "ETTm1":        "h2o_bq_ettm1",
    "Electricity":  "h2o_bq_electricity",
    "M4 Monthly":   "h2o_bq_m4_monthly",
    "Cable Demand": "h2o_bq_cable_demand",
}

# ── Family prefixes ───────────────────────────────────────────────────────────
AG_GBM_PREFIXES    = ("LightGBM", "XGBoost", "CatBoost")
AG_FOREST_PREFIXES = ("WeightedEnsemble", "RandomForestMSE", "ExtraTreesMSE")

# H2O model names start with e.g. "GBM_1_AutoML_...", "DRF_1_AutoML_..."
H2O_EXACT_PREFIXES  = ("GBM_", "XGBoost_")
H2O_APPROX_PREFIXES = ("GBM_", "XGBoost_", "DRF_", "XRT_")
# GLM_ uses permutation. It is excluded from both the exact and approx groups.

PRIMARY_EPS = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_importance(run_dir: Path) -> pd.DataFrame | None:
    # Pipeline writes raw_importance.csv to 03_importance/, not 05_rashomon/
    csv = run_dir / "03_importance" / "raw_importance.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    # item_id is a series identifier. AutoGluon SHAP emitted it as a
    # pseudo-feature. This code excludes it from the rankings. The reason is
    # the same as in src/importance_aggregation.py (NON_FEATURE_COLUMNS).
    return df[df["feature"] != "item_id"]


def pairwise_spearman(group: pd.DataFrame) -> list[float]:
    """
    Pivot a (model, feature, global_importance) group into a feature × model
    matrix. Return all pairwise Spearman ρ values.
    """
    models = group["model"].unique()
    if len(models) < 2:
        return []

    pivot = (
        group
        .pivot_table(index="feature", columns="model",
                     values="global_importance", aggfunc="first")
        .fillna(0.0)
    )

    rhos = []
    for m1, m2 in combinations(pivot.columns, 2):
        r, _ = spearmanr(pivot[m1].values, pivot[m2].values)
        if not np.isnan(r):
            rhos.append(float(r))
    return rhos


def compute_stability(
    df: pd.DataFrame,
    model_filter: tuple[str, ...] | None,
    label: str,
) -> pd.DataFrame:
    """
    Process each (split_id, seed, eps) group. Optionally filter the models by
    prefix. Compute the pairwise Spearman ρ. Return a per-group detail
    DataFrame.

    Columns: split_id, seed, eps, label, n_models, n_pairs, mean_spearman, status
    """
    rows: list[dict] = []

    for (split_id, seed, eps), grp in df.groupby(["split_id", "seed", "eps"]):
        if model_filter is not None:
            mask = grp["model"].apply(
                lambda m: any(m.startswith(p) for p in model_filter)
            )
            grp = grp[mask]

        n_models = int(grp["model"].nunique())

        if n_models < 2:
            rows.append({
                "split_id": split_id, "seed": seed, "eps": eps,
                "label": label, "n_models": n_models,
                "n_pairs": 0, "mean_spearman": np.nan,
                "status": "insufficient_models",
            })
            continue

        rhos = pairwise_spearman(grp)
        rows.append({
            "split_id": split_id, "seed": seed, "eps": eps,
            "label": label, "n_models": n_models,
            "n_pairs": len(rhos),
            "mean_spearman": float(np.mean(rhos)) if rhos else np.nan,
            "status": "ok",
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["split_id", "seed", "eps", "label", "n_models",
                 "n_pairs", "mean_spearman", "status"]
    )


def summarise_at_eps(detail: pd.DataFrame, eps: float) -> dict:
    """Aggregate stability at a fixed ε."""
    sub = detail[np.abs(detail["eps"] - eps) < 1e-9]
    ok    = sub[sub["status"] == "ok"]
    insuf = sub[sub["status"] == "insufficient_models"]
    return {
        "mean_spearman":  float(ok["mean_spearman"].mean()) if len(ok) else np.nan,
        "std_spearman":   float(ok["mean_spearman"].std())  if len(ok) else np.nan,
        "n_ok":           int(len(ok)),
        "n_insufficient": int(len(insuf)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AutoGluon
# ─────────────────────────────────────────────────────────────────────────────

def run_ag_analysis() -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("AUTOGLUON FAMILY-BASED SENSITIVITY CHECK")
    print("=" * 72)
    print("""
All AutoGluon bagged models use permutation SHAP.
The _try_tree_explainer module check fails for _BAG_L1/_BAG_L2 wrappers
because bagging wrappers are not in the supported tree-module list.
Every model family (LightGBM, XGBoost, CatBoost, WeightedEnsemble, ...) uses
shap.Explainer(..., algorithm="permutation") uniformly. There is no
explainer-type confound in AutoGluon stability results.

Sensitivity groups (family-based, not explainer-based):
  full_set   : all Rashomon set models
  gbm_family : LightGBM*, XGBoost*, CatBoost*
  forest     : WeightedEnsemble*, RandomForestMSE*, ExtraTreesMSE*
""")

    summary_rows: list[dict] = []

    for dataset, run_name in AG_RUNS.items():
        run_dir = RESULTS_DIR / run_name
        df = load_raw_importance(run_dir)
        if df is None:
            print(f"  [SKIP] {dataset}: raw_importance.csv not found in {run_dir}")
            continue

        print(f"\n  Dataset: {dataset}  ({run_name})")
        print(f"    Rows: {len(df):,}   Unique models: {df['model'].nunique()}")

        family_counts = (
            df.groupby("model")["feature"].count()
            .reset_index()
            .rename(columns={"feature": "n_rows"})
        )
        family_counts["family"] = family_counts["model"].apply(
            lambda m: next(
                (p for p in list(AG_GBM_PREFIXES) + list(AG_FOREST_PREFIXES)
                 if m.startswith(p)),
                "other",
            )
        )
        fam_summary = (
            family_counts.groupby("family")["model"]
            .count()
            .sort_values(ascending=False)
        )
        print("    Families in Rashomon set:")
        for fam, cnt in fam_summary.items():
            print(f"      {fam}: {cnt} model(s)")

        groups = [
            ("full_set",   None),
            ("gbm_family", AG_GBM_PREFIXES),
            ("forest",     AG_FOREST_PREFIXES),
        ]

        for label, filt in groups:
            det = compute_stability(df, model_filter=filt, label=label)
            det.to_csv(OUT_DIR / f"ag_{run_name}_{label}_detail.csv", index=False)
            s = summarise_at_eps(det, PRIMARY_EPS)
            summary_rows.append({
                "framework": "AutoGluon",
                "dataset": dataset,
                "run": run_name,
                "label": label,
                **s,
            })

    cols = ["framework", "dataset", "run", "label",
            "mean_spearman", "std_spearman", "n_ok", "n_insufficient"]
    return pd.DataFrame(summary_rows, columns=cols) if summary_rows else pd.DataFrame(columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# H2O
# ─────────────────────────────────────────────────────────────────────────────

def run_h2o_analysis() -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("H2O AUTOML EXPLAINER-TYPE SENSITIVITY CHECK")
    print("=" * 72)
    print("""
H2O explainer assignment by model name prefix:
  GBM_, XGBoost_  → predict_contributions()  = exact Shapley   → tree_exact
  DRF_, XRT_      → predict_contributions()  = Saabas approx   → tree_approx only
  GLM_            → permutation SHAP                            → full_set only

Comparison groups at each (split, seed, eps):
  full_set    : GBM + XGBoost + DRF + XRT + GLM
  tree_approx : GBM + XGBoost + DRF + XRT          (drop GLM permutation)
  tree_exact  : GBM + XGBoost only                  (exact Shapley only)
""")

    summary_rows: list[dict] = []

    for dataset, run_name in H2O_RUNS.items():
        run_dir = RESULTS_DIR / run_name
        df = load_raw_importance(run_dir)
        if df is None:
            print(f"  [SKIP] {dataset}: raw_importance.csv not found in {run_dir}")
            continue

        print(f"\n  Dataset: {dataset}  ({run_name})")
        print(f"    Rows: {len(df):,}   Unique models: {df['model'].nunique()}")

        # Classify each model by prefix
        def h2o_family(m: str) -> str:
            for p in ("GBM_", "XGBoost_", "DRF_", "XRT_", "GLM_",
                      "StackedEnsemble_"):
                if m.startswith(p):
                    return p.rstrip("_")
            return "other"

        models_df = pd.DataFrame({"model": df["model"].unique()})
        models_df["family"] = models_df["model"].apply(h2o_family)
        fam_summary = models_df["family"].value_counts()
        print("    Families in Rashomon set:")
        for fam, cnt in fam_summary.items():
            print(f"      {fam}: {cnt} model(s)")

        groups = [
            ("full_set",    None),
            ("tree_approx", H2O_APPROX_PREFIXES),
            ("tree_exact",  H2O_EXACT_PREFIXES),
        ]

        for label, filt in groups:
            det = compute_stability(df, model_filter=filt, label=label)
            det.to_csv(OUT_DIR / f"h2o_{run_name}_{label}_detail.csv", index=False)
            s = summarise_at_eps(det, PRIMARY_EPS)
            summary_rows.append({
                "framework": "H2O",
                "dataset": dataset,
                "run": run_name,
                "label": label,
                **s,
            })

    cols = ["framework", "dataset", "run", "label",
            "mean_spearman", "std_spearman", "n_ok", "n_insufficient"]
    return pd.DataFrame(summary_rows, columns=cols) if summary_rows else pd.DataFrame(columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison tables
# ─────────────────────────────────────────────────────────────────────────────

def comparison_table(summary: pd.DataFrame, framework: str) -> pd.DataFrame:
    if summary.empty or "framework" not in summary.columns:
        print(f"\n  No results for {framework}.")
        return pd.DataFrame()
    sub = summary[summary["framework"] == framework].copy()
    if sub.empty:
        print(f"\n  No results for {framework}.")
        return pd.DataFrame()

    datasets = sub["dataset"].unique()

    print(f"\n{'─' * 72}")
    print(f"  {framework}  |  Mean Spearman ρ at ε = {PRIMARY_EPS}  (Δ vs full_set)")
    print(f"{'─' * 72}")
    header = f"  {'Dataset':<16} {'Group':<14} {'ρ':>7}  {'Δ':>7}  {'OK':>4}  {'Insuf':>5}"
    print(header)
    print(f"  {'─'*16} {'─'*14} {'─'*7}  {'─'*7}  {'─'*4}  {'─'*5}")

    rows_out: list[dict] = []

    for ds in datasets:
        ds_sub = sub[sub["dataset"] == ds].set_index("label")
        full_rho = ds_sub.loc["full_set", "mean_spearman"] \
            if "full_set" in ds_sub.index else np.nan

        for lbl in sub["label"].unique():
            if lbl not in ds_sub.index:
                continue
            row = ds_sub.loc[lbl]
            rho = row["mean_spearman"]
            delta = (rho - full_rho) if lbl != "full_set" else np.nan
            flagged = (not np.isnan(delta)) and (abs(delta) > 0.05)

            rho_s   = f"{rho:.3f}"   if not np.isnan(rho)   else "  nan"
            delta_s = f"{delta:+.3f}" if not np.isnan(delta) else "    -"
            flag    = " ***" if flagged else "    "
            insuf_c = int(row["n_insufficient"])
            ok_c    = int(row["n_ok"])

            print(f"  {ds:<16} {lbl:<14} {rho_s:>7}  {delta_s}{flag}  "
                  f"{ok_c:>4}  {insuf_c:>5}")

            rows_out.append({
                "framework": framework,
                "dataset": ds,
                "label": lbl,
                "mean_spearman": rho,
                "delta_vs_full": delta,
                "flagged": flagged,
                "n_ok": ok_c,
                "n_insufficient": insuf_c,
            })

    print()
    return pd.DataFrame(rows_out)


# ─────────────────────────────────────────────────────────────────────────────
# Final verdict
# ─────────────────────────────────────────────────────────────────────────────

def final_verdict(ag_cmp: pd.DataFrame, h2o_cmp: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("FINAL VERDICT")
    print("=" * 72)
    print(f"Threshold: |Δ mean Spearman ρ| > 0.05 at ε = {PRIMARY_EPS}\n")

    # ── AutoGluon ──────────────────────────────────────────────────────────
    print("AutoGluon:")
    print("  Finding: all models use permutation SHAP, so there is no explainer-type confound.")

    if ag_cmp.empty:
        print("  No AG results to evaluate.\n")
    else:
        gbm_vs_full = ag_cmp[
            (ag_cmp["label"] == "gbm_family") & ag_cmp["flagged"]
        ]["dataset"].tolist()
        if gbm_vs_full:
            print(f"  GBM-family vs full_set: |Δ| > 0.05 on {gbm_vs_full}")
            print("  → Family composition influences measured stability.")
            print("    Report sensitivity check alongside main results.")
        else:
            print("  GBM-family vs full_set: |Δ| ≤ 0.05 on all datasets.")
            print("  → Stability is robust to family composition.")

    # ── H2O ───────────────────────────────────────────────────────────────
    print("\nH2O:")
    if h2o_cmp.empty:
        print("  No H2O results to evaluate.\n")
        return

    exact_vs_full = h2o_cmp[
        (h2o_cmp["label"] == "tree_exact") & h2o_cmp["flagged"]
    ]["dataset"].tolist()
    approx_vs_exact = (
        h2o_cmp[h2o_cmp["label"] == "tree_approx"]
        .merge(
            h2o_cmp[h2o_cmp["label"] == "tree_exact"][["dataset", "mean_spearman"]]
            .rename(columns={"mean_spearman": "exact_rho"}),
            on="dataset", how="left",
        )
        .assign(delta_approx_exact=lambda d: d["mean_spearman"] - d["exact_rho"])
    )
    drf_drives = approx_vs_exact[
        approx_vs_exact["delta_approx_exact"].abs() > 0.05
    ]["dataset"].tolist()

    if exact_vs_full:
        print(f"  tree_exact vs full_set: |Δ| > 0.05 on {exact_vs_full}")
        print("  → Removing GLM and/or Saabas-approximate models changes stability.")
        print("    Instability on affected datasets is partially attributable to")
        print("    approximation artefacts as well as architectural diversity.")
    else:
        print("  tree_exact vs full_set: |Δ| ≤ 0.05 on all datasets.")
        print("  → Stability findings are robust to the explainer-type confound;")
        print("    the instability is attributable to architectural diversity.")

    if drf_drives:
        print(f"\n  Adding DRF/XRT (Saabas) changes ρ by > 0.05 on: {drf_drives}")
        print("  → DRF/XRT Saabas approximation inflates attribution disagreement")
        print("    beyond what exact-Shapley models show; instability magnitude")
        print("    on these datasets is partially an approximation artefact.")
    else:
        print("\n  Adding DRF/XRT (Saabas) does NOT change ρ by > 0.05 on any dataset.")
        print("  → Saabas approximation does not materially inflate instability.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("robustness_shap_method_check.py")
    print(f"Output directory : {OUT_DIR}")
    print(f"Primary ε        : {PRIMARY_EPS}")

    ag_summary  = run_ag_analysis()
    h2o_summary = run_h2o_analysis()

    print("\n\n" + "=" * 72)
    print("COMPARISON TABLES  (within-set Spearman ρ, pairwise mean)")
    print("=" * 72)
    print("*** = |Δ vs full_set| > 0.05  (meaningful shift)")

    ag_cmp  = comparison_table(ag_summary,  "AutoGluon")
    h2o_cmp = comparison_table(h2o_summary, "H2O")

    # Persist
    ag_summary.to_csv(OUT_DIR / "ag_summary.csv", index=False)
    h2o_summary.to_csv(OUT_DIR / "h2o_summary.csv", index=False)
    if not ag_cmp.empty:
        ag_cmp.to_csv(OUT_DIR / "ag_comparison.csv", index=False)
    if not h2o_cmp.empty:
        h2o_cmp.to_csv(OUT_DIR / "h2o_comparison.csv", index=False)

    final_verdict(ag_cmp, h2o_cmp)

    print(f"Detail and summary CSVs written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
