# Churn Harness Experiment

E-commerce churn prediction experiment for a D&B academic report. The project uses UCI Online Retail transactions as the main dataset, IBM Telco Customer Churn as a benchmark, and includes model evaluation, SHAP interpretation, Neslin-style campaign profit simulation, synthetic uplift demo, and a harness validation loop for governed campaign decisions.

## What Is Included

- Online Retail preprocessing and RFM feature engineering
- 90-day inactivity churn labeling and BG/NBD `P(alive)` labeling analysis
- Logistic Regression, Random Forest, XGBoost, and LightGBM model comparisons
- Class imbalance comparison with `class_weight` and SMOTE
- SHAP summary and top-feature plots
- Neslin et al. campaign profit simulation and EMP-style approximation
- Synthetic uplift/Qini demo for methodology illustration
- Harness validation loop with confidence, profit, and rate-limit gates
- Leakage correction experiment removing `Recency`
- BG/NBD threshold recalibration experiment
- Neslin parameter sensitivity analysis
- Telco churn benchmark using the same modeling/SHAP pattern

## Repository Structure

```text
.
|-- data/
|   `-- raw/
|       |-- Online Retail.xlsx
|       |-- online_retail.zip
|       `-- Telco-Customer-Churn.csv
|-- outputs/
|   |-- RESULTS.md
|   |-- online_model_performance.csv
|   |-- online_model_performance_no_recency.csv
|   |-- online_model_performance_bgnbd_label.csv
|   |-- online_leakage_comparison.csv
|   |-- online_neslin_sensitivity.csv
|   `-- ... plots, summaries, and benchmark outputs
`-- scripts/
    |-- run_churn_pipeline.py
    |-- run_leakage_sensitivity_experiments.py
    `-- harness_validation.py
```

## Data Sources

- UCI Online Retail: <https://archive.ics.uci.edu/dataset/352/online+retail>
- IBM Telco Customer Churn public CSV mirror

The raw files are committed under `data/raw/` so the current results can be inspected without re-downloading data.

## Environment Setup

The experiment was run with Python 3.11.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install pandas numpy scikit-learn matplotlib seaborn shap xgboost lightgbm imbalanced-learn lifetimes openpyxl requests scipy kagglehub
```

## Run The Experiments

Full baseline pipeline:

```powershell
.venv\Scripts\python.exe scripts\run_churn_pipeline.py
```

Additional leakage correction, BG/NBD recalibration, and Neslin sensitivity experiments:

```powershell
.venv\Scripts\python.exe scripts\run_leakage_sensitivity_experiments.py
```

Harness validation loop only:

```powershell
.venv\Scripts\python.exe scripts\harness_validation.py
```

## Key Results

Detailed report-ready findings are in [`outputs/RESULTS.md`](outputs/RESULTS.md).

Selected highlights:

- Online Retail clean rows: `397,884 / 541,909`
- CustomerID missing ratio in raw Online Retail: `24.93%`
- Cancel invoice ratio: `1.71%`
- 90-day inactivity churn rate: `33.40%`
- Original Online Retail model with `Recency` leaked the label and reached AUC-ROC `1.000`
- No-Recency leakage-corrected best model: XGBoost + class_weight, AUC-ROC `0.804`, PR-AUC `0.575`
- BG/NBD recalibrated-label best model: XGBoost + class_weight, AUC-ROC `0.864`, PR-AUC `0.848`
- Telco benchmark best model: XGBoost + class_weight, AUC-ROC `0.837`, PR-AUC `0.653`
- Neslin sensitivity scenarios produced interior optimal target ratios around `0.78` to `0.80`

## Important Notes

- The initial perfect Online Retail score is intentionally retained as a leakage example. The corrected no-Recency experiment should be used when discussing non-leaky predictive performance.
- BG/NBD `P(alive)` was highly concentrated near `1.0`; the recalibrated churn label therefore uses a rank-based lower-tail selection rather than a plain fixed `0.5` threshold.
- The uplift experiment is synthetic because Online Retail does not include observed treatment/control campaign assignment. It is a method demo, not evidence of real campaign lift.
- Neslin parameters are scenario assumptions, not observed business costs.

## Main Output Files

- [`outputs/RESULTS.md`](outputs/RESULTS.md): report-ready summary
- [`outputs/online_leakage_comparison.csv`](outputs/online_leakage_comparison.csv): leaky vs corrected model comparison
- [`outputs/online_labeling_summary_v2.csv`](outputs/online_labeling_summary_v2.csv): BG/NBD recalibration results
- [`outputs/online_neslin_sensitivity.csv`](outputs/online_neslin_sensitivity.csv): campaign profit sensitivity table
- [`outputs/harness_validation_log.json`](outputs/harness_validation_log.json): sample governed decision audit log
