# Glossary

Plain definitions of the terms used in this repository, grouped by where they appear in the pipeline. [Back to the README](README.md).

## 1. AutoML and the two frameworks

**AutoML (automated machine learning)** is a system that trains and tunes many candidate models on its own and keeps the ones that predict best, so a person does not have to hand-pick and tune a single model.

**Framework** here means one AutoML system. This study uses two and compares them.

**AutoGluon** is an open-source AutoML toolkit from Amazon. Its `best_quality` preset builds each model by bagging and stacking (both defined below). On these datasets one bagged LightGBM variant is far enough ahead of the rest that its near-optimal sets stay small and contain a single model family. It is the primary framework here.

**H2O AutoML** is a second toolkit, from H2O.ai. It runs a random grid search across five model families (GBM, XGBoost, DRF, XRT, GLM), so its near-optimal sets are larger and mix families. It needs Java to run and is the secondary framework.

**Why two frameworks.** They search in opposite ways. AutoGluon concentrates on one strong family, and H2O spreads across many. Running both tests whether the make-up of the near-optimal set, its size and which families it contains, changes how stable the SHAP explanations are.

## 2. Turning a time series into a table (the inputs)

**Time series forecasting** predicts the next value of a quantity measured over time, such as electricity load or product demand.

**Tabularisation (lag-based features).** A time series has no fixed input columns, so each row is built from a sliding window of past values. Those past values become the input columns (the lags), and the next value becomes the target. This lets ordinary table-based models forecast.

**Target lag** is a past value of the series being forecast, for example its value one step ago or two steps ago. "Autoregressive" means a series predicts itself from its own past.

**Covariate lag** is a past value of a different input series measured alongside the target.

**Calendar features** are columns read from the timestamp, such as hour of day or month, encoded so the model can use seasonality.

**item_id** is a label naming which series a row belongs to. It identifies the series, so it is excluded from every importance ranking.

## 3. Models

**Model family** is a type of learning algorithm. A near-optimal set can contain one family or several.

**LightGBM, GBM, and XGBoost** are tree-based gradient boosting methods. They build many small decision trees in sequence, each one correcting the errors of the last. GBM is H2O's own gradient boosting, and XGBoost and LightGBM are widely used versions of the same idea.

**DRF (distributed random forest)** trains many decision trees on random subsets of the data and averages them.

**XRT (extremely randomised trees)** is a random-forest variant that also chooses split points at random, for extra diversity.

**GLM (generalised linear model)** is a linear model, such as linear or logistic regression. It is the one non-tree family here.

**Bagging** trains a model on many resampled copies of the data and averages them, which lowers variance.

**Stacking** trains a further model to combine the predictions of several base models.

**Meta-ensemble** is the further model that stacking trains, one whose inputs are the predictions of the other models rather than the original features. AutoGluon's `best_quality` preset builds a weighted meta-ensemble on top of its bagged base learners.


**Grid search** tries many combinations of settings. H2O's is random and budget-limited.

## 4. Explaining a model with SHAP

**SHAP (SHapley Additive exPlanations)** splits a single prediction into one number per input feature, showing how much each feature pushed the prediction up or down. It builds on Shapley values from cooperative game theory.

**Attribution** is the number SHAP gives one feature for one prediction. Averaging the size of these across the test rows (the mean absolute SHAP) scores how important a feature is overall.

**Feature-importance ranking** orders the features from most to least important by their mean absolute SHAP. How stable this order stays across the near-optimal models is what the study measures.

**Explainers.** SHAP is computed differently for different model types: exact TreeSHAP for GBM and XGBoost, Saabas for DRF and XRT, and permutation for GLM and ensembles. Saabas is an older tree method whose values are not on the same numeric scale as TreeSHAP's.

**rank-then-mean.** Because those explainers put attributions on different scales, the study ranks the features inside each model first and then averages the ranks across models. Ranking is scale-invariant, so the average does not depend on any single large-magnitude model.

## 5. Near-optimal models and the Rashomon set

**MAE (mean absolute error)** is the average size of the gap between forecast and actual value. Lower is better. It scores the models and defines which one is best.

**Validation split and m\*.** Models are scored on a validation slice kept separate from the training data. m\* is the model with the lowest validation MAE.

**Rashomon set** is every model whose validation MAE falls within a tolerance of m\*'s. Formally, `R(ε) = { m : MAE(m) ≤ MAE(m*) × (1 + ε) }`. The name comes from the Rashomon effect, where several equally good accounts can still disagree.

**ε (epsilon)** is that tolerance. At ε = 0.05 the set admits every model within 5% of the best MAE, and widening ε only ever adds models.

**Empirical Rashomon set** is the set the AutoML search produced, as opposed to every near-optimal model that could exist.

## 6. Measuring stability

**Spearman ρ (rho)** is the rank correlation between two importance rankings. A value of 1 means the same order and 0 means no relation. This is the headline stability number.

**Kendall τ_b (tau-b)** is a second rank-agreement measure. It counts the feature pairs ordered the same way by both rankings and corrects for ties.

**Jaccard** measures the overlap between the top-k features of two rankings, as the shared count over the combined count. Here k is 30% of the feature count.

**SHAP-CV** is the coefficient of variation, the spread over the mean, of a feature's mean absolute SHAP across the set. The study uses it only where the explainer scales are comparable.

**Seeds** are the random starting points. Each run repeats over three seeds, and the results are averaged over them.

## 7. Datasets and the table columns

The Datasets table uses the column names `Series`, `Frequency`, and `Splits`. The configuration files use the short forms `n_series`, `freq`, and `n_splits`.

**Series** is how many separate time series a dataset contains. Electricity has 20 (one per meter), each ETT set has one, M4 Monthly has 50, and Cable Demand has 10.

**Frequency (`freq`)** is the spacing between observations: 15-min, hourly, or monthly.

**Splits (`n_splits`)** is how many rolling-origin train and test folds the data is cut into. More history allows more splits.

**Rolling-origin (expanding-window) split** is a time-respecting form of cross-validation. The model trains on the earliest block and is tested on the next, then the training window extends forward and the step repeats. The test period always comes after the training period, so no future information leaks in.

The six datasets, and why each is included:

**Electricity** (UCI LD2011-2014) is 20 households at 15-min spacing, a public multi-series load benchmark.

**ETTh1 and ETTh2** are Electricity Transformer Temperature, hourly, one series each. Both are standard forecasting benchmarks.

**ETTm1** is the 15-min version of the same transformer data.

**M4 Monthly** is a 50-series subset of the M4 forecasting competition, monthly, with many short series.

**Cable Demand** is a proprietary monthly demand dataset. It is the one production business series among the six.

**Why this mix.** Three domains (energy, transformer sensors, and product demand) and two resolutions (sub-hourly and monthly), single-series and multi-series, mostly public with one proprietary set, so stability is tested across varied conditions.
