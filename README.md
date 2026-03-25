# featurewiz-pro

**A comprehensive 7-phase feature engineering toolkit for tabular machine learning.**

Handles regression and classification tasks with any target dtype — numeric, boolean, or string (`'Yes'`/`'No'`, `'High'`/`'Low'`, etc.). Wraps pandas, scikit-learn, statsmodels, scipy, and seaborn into a single opinionated workflow so you spend time interpreting findings, not wiring libraries together.

---

## Installation

```bash
pip install featurewiz-pro
```

Install all optional extras (SHAP, pygam, missingno, category_encoders, lightgbm):

```bash
pip install "featurewiz-pro[full]"
```

---

## Quick start

### Option A — low-level class (full control)

```python
import pandas as pd
from featurewiz_pro import FeatureEngineer

df  = pd.read_csv("train.csv")
df  = df.sample(frac=0.15, random_state=42)     # dont use the whole data , just sample
fe  = FeatureEngineer(df, target="Churn", task="classification")

fe.run_all()                     # runs all 7 phases
report   = fe.report()           # dict of all findings
clean_df = fe.get_clean_df()     # fully engineered DataFrame
```

### Option B — master pipeline (recommended)

```python
from featurewiz_pro import FeatureEngineeringPipeline, PipelineConfig

cfg = PipelineConfig(
    target     = "Churn",
    task       = "classification",
    phases     = [1, 2, 3, 4, 5, 6, 7],   # run all phases
    skip_steps = ["shap_dependence",        # skip expensive optional steps
                  "gam_smooth_terms"],
    save_report = "report.json",            # write findings to JSON
    save_df     = "clean.csv",             # write engineered data to CSV
    show_plots  = True,
)

pipeline = FeatureEngineeringPipeline(df, cfg)
result   = pipeline.run()

result.summary()                    # coloured console report + recommendations
clean_df = result.df                # engineered DataFrame
selected = result.selected_features # RFECV-selected features
```

---

## The 7-phase workflow

| Phase | Focus | Key methods |
|-------|-------|-------------|
| **1 · Audit** | Schema, missing values, cardinality, target profile | `audit`, `missing_map`, `cardinality_check`, `target_profile`, `drop_useless` |
| **2 · Univariate** | Distributions, outliers, datetime / text extraction | `distribution_plots`, `detect_outliers`, `extract_datetime_features`, `extract_text_features` |
| **3 · Bivariate** | Feature → target relationships | `bivariate_scatter`, `correlation_analysis`, `mutual_information_scores`, `categorical_vs_target` |
| **4 · Linearity** | Detect non-linear signals | `residual_diagnostics`, `reset_test`, `partial_dependence_plots`, `shap_dependence`, `gam_smooth_terms` |
| **5 · Collinearity** | Multicollinearity, interactions | `correlation_matrix`, `vif_analysis`, `cramers_v_matrix`, `interaction_screening`, `feature_clustering` |
| **6 · Transform** | Encoding and scaling | `apply_transforms`, `polynomial_spline_features`, `target_encode`, `cyclic_encode`, `scale_features` |
| **7 · Selection** | Rank and select features | `mutual_info_ranking`, `rfecv_selection`, `lasso_importance`, `permutation_importance_analysis`, `leakage_audit` |

---

## Incremental execution

Run phases one at a time, or re-run a single failed step:

```python
pipeline = FeatureEngineeringPipeline(df, cfg)

# run phases incrementally
pipeline.run_phases([1, 2])
pipeline.run_phases([3, 4])

# re-run a single step by name
pipeline.run_step("rfecv")

# retry all failed steps
pipeline.retry_failed()

# timing breakdown
print(pipeline.timing_report())
```

---

## PipelineConfig — every knob

```python
PipelineConfig(
    # core
    target            = "price",        # required
    task              = "regression",   # or "classification"
    phases            = [1,2,3,4,5,6,7],
    skip_steps        = [],             # list of step names to skip

    # phase 1
    missing_thresh    = 95.0,           # drop cols with >x% missing
    variance_thresh   = 0.0,            # drop zero-variance cols
    drop_duplicates   = True,

    # phase 2
    outlier_method    = "iqr",          # or "zscore"
    z_thresh          = 3.0,

    # phase 4
    pdp_max_cols      = 6,

    # phase 5
    corr_threshold    = 0.85,           # |r| for collinear pair flag
    vif_threshold     = 10.0,
    interaction_top_k = 5,

    # phase 6
    transform_method  = "yeo-johnson",  # "box-cox" | "log1p" | "sqrt"
    skew_thresh       = 0.75,
    poly_degree       = 2,
    use_spline        = True,
    scale_method      = "standard",     # "robust" | "minmax" | None

    # phase 7
    lasso_alpha       = 0.01,
    perm_n_repeats    = 10,
    min_features      = 5,

    # output
    save_report       = "report.json",  # None to skip
    save_df           = "clean.csv",    # or "clean.parquet"
    show_plots        = True,           # False for CI / batch mode
)
```

---

## PipelineResult — what you get back

```python
result.df                   # pd.DataFrame — fully engineered
result.report               # dict — all findings
result.selected_features    # List[str] — RFECV winners
result.nonlinear_features   # List[str] — PDP-flagged non-linear cols
result.collinear_pairs      # List[tuple] — (feat_a, feat_b, r)
result.leakage_suspects     # List[str] — columns to investigate
result.transformations      # dict — {col: transform_applied}
result.summary()            # pretty-print with auto-recommendations
result.save_report_json("report.json")
result.save_dataframe("clean.parquet", drop_originals=True)
```

---

## Target dtype support

The library auto-encodes any target dtype before passing it to sklearn / statsmodels / scipy:

| Target dtype | Example | Encoding |
|---|---|---|
| `float` / `int` | `205.0`, `1` | used as-is |
| `bool` | `True` / `False` | cast to `int` |
| `object` / `str` | `"Yes"` / `"No"` | label-encoded (alphabetical order) |
| `category` | `pd.Categorical(...)` | label-encoded |

The mapping is stored in `fe._target_label_map` for reference.

---

## Optional dependencies

| Package | Enables |
|---|---|
| `missingno` | Visual missing-value heatmap (Phase 1) |
| `category_encoders` | Laplace-smoothed TargetEncoder (Phase 6) |
| `pygam` | GAM smooth-term EDF analysis (Phase 4) |
| `shap` | SHAP dependence plots (Phase 4) |
| `lightgbm` | Faster gradient boosting (Phases 4, 5, 7) |

All optional — methods gracefully print an install hint and continue if the package is absent.

---

## Step names (for `skip_steps` / `run_step`)

```
audit                missing_map           cardinality_check     target_profile
drop_useless         distribution_plots    detect_outliers       categorical_frequencies
extract_datetime     extract_text          bivariate_scatter     correlation_analysis
mutual_info_scores   categorical_vs_target chi_squared_test      partial_regression
residual_diagnostics reset_test            pdp                   shap_dependence
gam_smooth_terms     correlation_matrix    vif_analysis          cramers_v_matrix
interaction_screen   feature_clustering    apply_transforms      poly_spline
target_encode        cyclic_encode         scale_features        mi_ranking
rfecv                lasso_importance      perm_importance       leakage_audit
```

---

## License

MIT © 2025 Your Name
