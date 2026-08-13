# results/

Result files behind the figures and tables in the top-level [README](../README.md). These are the small analysis-ready outputs of the pipeline.

## Layout

Each run has its own directory, named `<framework>_<dataset>`:

- `bq_*` are AutoGluon `best_quality` runs.
- `h2o_bq_*` are H2O AutoML runs.
- `h2o_perm_*` are the dual-explainer control runs for RQ1 (Electricity, ETTm1, ETTh1), where every H2O model is explained twice, once with its native explainer and once with permutation.

Inside each run:

- `03_importance/raw_importance.csv` is the per-model SHAP importance for every feature, seed, split, and ε.
- `03_importance/feature_ranks.csv` and `aggregated_summary.csv` are the ranked and aggregated importances.
- `04_stability/stability_summary.csv` and `epsilon_sensitivity*.csv` are the Spearman ρ, Kendall τ, and Jaccard numbers, by ε.
- `05_rashomon/rashomon_models.csv` and `model_metrics.csv` record which models enter each set and their validation MAE.

The `h2o_perm_*` runs have `raw_importance.csv` (permutation for every model) and `raw_importance_alt.csv` (native explainer per family), so both arms see identical models.
