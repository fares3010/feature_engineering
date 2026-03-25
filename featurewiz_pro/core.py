"""
FeatureEngineer
===============
A self-contained class that walks through all 7 phases of the
feature engineering procedure:

  Phase 1 – Data profiling & initial audit
  Phase 2 – Univariate feature analysis
  Phase 3 – Bivariate: feature → target relationship
  Phase 4 – Linearity vs non-linearity diagnosis
  Phase 5 – Feature–feature multicollinearity & interaction
  Phase 6 – Transformations & encoding decisions
  Phase 7 – Feature selection & validation

Dependencies
------------
  Core  : pandas, numpy, scipy, statsmodels
  Visual: matplotlib, seaborn
  ML    : scikit-learn
  Optional (gracefully skipped if absent):
    missingno, category_encoders, pygam, shap, lightgbm

Quick start
-----------
  from feature_engineer import FeatureEngineer
  fe = FeatureEngineer(df, target="price", task="regression")
  fe.run_all()          # full pipeline, prints + plots everything
  report = fe.report()  # dict summarising findings
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

# ── matplotlib / seaborn ────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── sklearn ─────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.feature_selection import (
    mutual_info_regression,
    mutual_info_classif,
    RFECV,
    SelectFromModel,
)
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    PolynomialFeatures, SplineTransformer, PowerTransformer,
)

# ── statsmodels ─────────────────────────────────────────────────────────────
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import linear_reset

# ── optional imports (soft) ─────────────────────────────────────────────────
try:
    import missingno as msno
    _HAS_MSNO = True
except ImportError:
    _HAS_MSNO = False

try:
    import category_encoders as ce
    _HAS_CE = True
except ImportError:
    _HAS_CE = False

try:
    from pygam import LinearGAM, LogisticGAM, s
    _HAS_GAM = True
except ImportError:
    _HAS_GAM = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    bar = "─" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V association between two categorical series."""
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


# ─────────────────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Raw input data (features + target column).
    target : str
        Name of the target column inside *df*.
    task : {'regression', 'classification'}
        Determines which correlation / MI / model variants are used.
    cat_threshold : int
        Max unique values for a numeric column to be treated as categorical.
    figsize_base : tuple
        Default figure size multiplier.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        task: str = "regression",
        cat_threshold: int = 15,
        figsize_base: tuple = (14, 5),
    ) -> None:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'.")

        self.df_raw = df.copy()
        self.df = df.copy()
        self.target = target
        self.task = task
        self.cat_threshold = cat_threshold
        self.figsize_base = figsize_base

        # internal state populated by the methods
        self._report: dict = {}
        self._numeric_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._datetime_cols: List[str] = []
        self._high_card_cols: List[str] = []
        self._low_card_cols: List[str] = []
        self._outlier_cols: List[str] = []
        self._nonlinear_cols: List[str] = []
        self._collinear_pairs: List[tuple] = []
        self._selected_features: List[str] = []
        self._transformations_applied: dict = {}

        self._classify_columns()

    # =========================================================================
    #  INTERNAL UTILITIES
    # =========================================================================

    def _classify_columns(self) -> None:
        """Auto-detect numeric, categorical, and datetime columns (excludes target)."""
        features = [c for c in self.df.columns if c != self.target]
        self._numeric_cols = [
            c for c in features
            if pd.api.types.is_numeric_dtype(self.df[c])
            and self.df[c].nunique() > self.cat_threshold
        ]
        self._cat_cols = [
            c for c in features
            if not pd.api.types.is_numeric_dtype(self.df[c])
            or self.df[c].nunique() <= self.cat_threshold
        ]
        self._datetime_cols = [
            c for c in features
            if pd.api.types.is_datetime64_any_dtype(self.df[c])
        ]

    def _features(self) -> List[str]:
        return [c for c in self.df.columns if c != self.target]

    def _X(self) -> pd.DataFrame:
        return self.df[self._features()]

    def _y(self) -> pd.Series:
        return self.df[self.target]

    def _encode_target(self) -> pd.Series:
        """
        Return the target as a numeric Series.

        • Already numeric  → returned as-is.
        • Boolean          → cast to int (True=1, False=0).
        • String / object  → label-encoded (sorted unique values → 0,1,2,…).
          The mapping is stored in self._target_label_map for reference.

        This is the safe replacement for self._y() in any context that
        feeds the target into sklearn / statsmodels / scipy.
        """
        y = self.df[self.target]

        if pd.api.types.is_numeric_dtype(y):
            return y.copy()

        if pd.api.types.is_bool_dtype(y):
            return y.astype(int)

        # string / object / category → integer label encoding
        uniq = sorted(y.dropna().unique().astype(str))
        label_map = {v: i for i, v in enumerate(uniq)}
        self._target_label_map = label_map
        encoded = y.astype(str).map(label_map)
        return encoded

    def _mi_func(self):
        return mutual_info_regression if self.task == "regression" else mutual_info_classif

    def _base_model(self):
        if self.task == "regression":
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        return GradientBoostingClassifier(n_estimators=100, random_state=42)

    def _cv(self):
        if self.task == "classification":
            return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return KFold(n_splits=5, shuffle=True, random_state=42)

    # =========================================================================
    #  PHASE 1 – Data profiling & initial audit
    # =========================================================================

    def audit(self, verbose: bool = True) -> pd.DataFrame:
        """
        Phase 1 — Full schema audit.

        Returns a DataFrame with dtype, nunique, missing %, skew, and range
        for every column. Prints a readable summary when verbose=True.
        """
        _section("Phase 1 · Data profiling & initial audit")
        rows = []
        for col in self.df.columns:
            s = self.df[col]
            is_num = pd.api.types.is_numeric_dtype(s)
            rows.append({
                "column":       col,
                "dtype":        str(s.dtype),
                "nunique":      s.nunique(),
                "missing_%":    round(s.isna().mean() * 100, 2),
                "skew":         round(float(s.skew()), 3) if is_num else None,
                "min":          s.min() if is_num else None,
                "max":          s.max() if is_num else None,
                "role":         "target" if col == self.target
                                else ("numeric" if col in self._numeric_cols
                                else ("datetime" if col in self._datetime_cols
                                else "categorical")),
            })
        audit_df = pd.DataFrame(rows).set_index("column")

        if verbose:
            print(f"\n  Rows: {self.df.shape[0]:,}   Cols: {self.df.shape[1]}")
            print(f"  Numeric features   : {len(self._numeric_cols)}")
            print(f"  Categorical features: {len(self._cat_cols)}")
            print(f"  Datetime features  : {len(self._datetime_cols)}")
            print(f"  Target             : {self.target} ({self.task})")
            print(f"\n{audit_df.to_string()}")

        self._report["audit"] = audit_df
        return audit_df

    def missing_map(self, show_plot: bool = True) -> pd.Series:
        """
        Phase 1 — Compute and visualise missing-value percentages.

        Returns a Series of (column → % missing), sorted descending.
        Uses missingno heatmap when the library is available.
        """
        _section("Phase 1 · Missing values map")
        miss = (self.df.isna().mean() * 100).sort_values(ascending=False)
        miss = miss[miss > 0]

        if miss.empty:
            print("  No missing values found.")
            self._report["missing"] = miss
            return miss

        print(f"  Columns with missing data ({len(miss)}):")
        print(miss.to_string())

        if show_plot:
            if _HAS_MSNO:
                msno.heatmap(self.df, figsize=self.figsize_base)
                plt.title("Missing-value correlation heatmap")
                plt.tight_layout()
                plt.show()
            else:
                fig, ax = plt.subplots(figsize=(10, max(3, len(miss) * 0.4)))
                miss.plot.barh(ax=ax, color="#378ADD")
                ax.set_xlabel("% missing")
                ax.set_title("Missing values per column")
                plt.tight_layout()
                plt.show()

        self._report["missing"] = miss
        return miss

    def cardinality_check(self, verbose: bool = True) -> pd.DataFrame:
        """
        Phase 1 — Cardinality analysis for categorical columns.

        Classifies each categorical as:
          • binary  (2 unique)
          • low-cardinality  (≤ 20) → one-hot recommended
          • high-cardinality (> 20) → target / embedding recommended
        """
        _section("Phase 1 · Cardinality check")
        rows = []
        for col in self._cat_cols:
            n = self.df[col].nunique()
            label = "binary" if n == 2 else ("low-card" if n <= 20 else "high-card")
            rows.append({"column": col, "nunique": n, "classification": label,
                         "recommendation": ("0/1 flag" if n == 2
                                            else ("one-hot" if n <= 20
                                                  else "target-encode / embed"))})
            if label == "high-card":
                self._high_card_cols.append(col)
            else:
                self._low_card_cols.append(col)

        df_card = pd.DataFrame(rows).set_index("column")
        if verbose:
            print(df_card.to_string())
        self._report["cardinality"] = df_card
        return df_card

    def target_profile(self, show_plot: bool = True) -> dict:
        """
        Phase 1 — Deep-dive into the target variable distribution.

        Returns a dict with skew, kurtosis, class balance (classification),
        and outlier fraction (regression).
        """
        _section("Phase 1 · Target variable profile")
        y = self._y()
        y_enc = self._encode_target()
        profile = {
            "dtype":    str(y.dtype),
            "nunique":  y.nunique(),
            "missing_%": round(y.isna().mean() * 100, 2),
        }
        if hasattr(self, "_target_label_map"):
            profile["label_encoding"] = self._target_label_map
            print(f"  Target encoded as: {self._target_label_map}")
        if pd.api.types.is_numeric_dtype(y_enc):
            profile.update({
                "mean":     round(float(y_enc.mean()), 4),
                "median":   round(float(y_enc.median()), 4),
                "std":      round(float(y_enc.std()), 4),
                "skew":     round(float(y_enc.skew()), 4),
                "kurtosis": round(float(y_enc.kurt()), 4),
            })
            q1, q3 = y_enc.quantile(0.25), y_enc.quantile(0.75)
            iqr = q3 - q1
            outliers = ((y_enc < q1 - 1.5 * iqr) | (y_enc > q3 + 1.5 * iqr)).mean()
            profile["outlier_fraction_%"] = round(float(outliers) * 100, 2)

            if show_plot:
                fig, axes = plt.subplots(1, 2, figsize=self.figsize_base)
                sns.histplot(y_enc, kde=True, ax=axes[0], color="#378ADD")
                axes[0].set_title(f"Target distribution  (skew={profile['skew']:.2f})")
                sns.boxplot(y=y_enc, ax=axes[1], color="#378ADD")
                axes[1].set_title("Box-and-whisker")
                plt.tight_layout()
                plt.show()

        if self.task == "classification":
            vc = y.value_counts(normalize=True).round(4)
            profile["class_balance"] = vc.to_dict()
            print(f"  Class balance:\n{vc.to_string()}")
            if show_plot:
                vc.plot.bar(color="#378ADD")
                plt.title("Class balance")
                plt.ylabel("proportion")
                plt.tight_layout()
                plt.show()

        for k, v in profile.items():
            print(f"  {k:<25} {v}")

        self._report["target_profile"] = profile
        return profile

    def drop_useless(
        self,
        missing_thresh: float = 95.0,
        variance_thresh: float = 0.0,
        drop_duplicates: bool = True,
        inplace: bool = True,
    ) -> List[str]:
        """
        Phase 1 — Remove zero-variance, near-constant, and over-missing columns,
        plus duplicate rows.

        Parameters
        ----------
        missing_thresh : float
            Drop columns with > this % missing (default 95).
        variance_thresh : float
            Drop numeric columns with variance ≤ this value (default 0 = constant).
        inplace : bool
            If True, modifies self.df; otherwise returns a copy.

        Returns a list of dropped column names.
        """
        _section("Phase 1 · Drop useless columns & duplicate rows")
        dropped = []

        # duplicate rows
        if drop_duplicates:
            before = len(self.df)
            self.df.drop_duplicates(inplace=True)
            n_dup = before - len(self.df)
            if n_dup:
                print(f"  Dropped {n_dup:,} duplicate rows.")

        # high-missing columns
        miss_pct = self.df.drop(columns=[self.target]).isna().mean() * 100
        high_miss = miss_pct[miss_pct > missing_thresh].index.tolist()
        if high_miss:
            print(f"  High-missing (>{missing_thresh}%) dropped: {high_miss}")
            dropped.extend(high_miss)

        # zero / near-zero variance (numeric columns)
        num_df = self.df[self._numeric_cols].select_dtypes(include=np.number)
        low_var = num_df.columns[num_df.var() <= variance_thresh].tolist()
        if low_var:
            print(f"  Zero-variance numeric dropped: {low_var}")
            dropped.extend(low_var)

        # constant columns of any dtype (nunique == 1) not already caught above
        features = [c for c in self.df.columns if c != self.target]
        constant_any = [
            c for c in features
            if c not in low_var and self.df[c].nunique(dropna=True) <= 1
        ]
        if constant_any:
            print(f"  Constant columns (any dtype) dropped: {constant_any}")
            dropped.extend(constant_any)

        all_drop = list(set(dropped))
        if inplace:
            self.df.drop(columns=all_drop, errors="ignore", inplace=True)
            self._classify_columns()
        print(f"  Total columns dropped: {len(all_drop)}")
        self._report["dropped_columns"] = all_drop
        return all_drop

    # =========================================================================
    #  PHASE 2 – Univariate feature analysis
    # =========================================================================

    def distribution_plots(
        self,
        cols: Optional[List[str]] = None,
        max_cols: int = 20,
    ) -> None:
        """
        Phase 2 — Histogram + KDE for numeric features.

        Parameters
        ----------
        cols : list, optional
            Specific columns to plot. Defaults to self._numeric_cols.
        max_cols : int
            Cap on columns plotted in a single call (default 20).
        """
        _section("Phase 2 · Distribution plots (numeric)")
        cols = cols or self._numeric_cols[:max_cols]
        n = len(cols)
        if n == 0:
            print("  No numeric columns to plot.")
            return

        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes = np.array(axes).flatten()

        for i, col in enumerate(cols):
            skew_val = self.df[col].skew()
            sns.histplot(self.df[col].dropna(), kde=True, ax=axes[i], color="#378ADD", bins=30)
            axes[i].set_title(f"{col}\nskew={skew_val:.2f}", fontsize=9)
            axes[i].tick_params(labelsize=7)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Numeric feature distributions", y=1.01)
        plt.tight_layout()
        plt.show()

    def detect_outliers(
        self,
        method: str = "iqr",
        z_thresh: float = 3.0,
        cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Phase 2 — Outlier detection for numeric columns.

        Parameters
        ----------
        method : {'iqr', 'zscore'}
        z_thresh : float
            Z-score threshold (used when method='zscore').

        Returns a DataFrame with outlier counts and fractions per column.
        """
        _section("Phase 2 · Outlier detection")
        cols = cols or self._numeric_cols
        rows = []
        for col in cols:
            s = self.df[col].dropna()
            if method == "iqr":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
            else:
                z = (s - s.mean()) / s.std()
                mask = z.abs() > z_thresh
            n_out = int(mask.sum())
            rows.append({"column": col, "outliers": n_out,
                         "outlier_%": round(n_out / len(s) * 100, 2)})
            if n_out > 0:
                self._outlier_cols.append(col)

        df_out = pd.DataFrame(rows).set_index("column").sort_values("outlier_%", ascending=False)
        print(df_out[df_out["outlier_%"] > 0].to_string())
        self._report["outliers"] = df_out
        return df_out

    def categorical_frequencies(
        self,
        cols: Optional[List[str]] = None,
        rare_thresh: float = 1.0,
        show_plot: bool = True,
    ) -> dict:
        """
        Phase 2 — Value-count frequency tables for categorical columns.

        Parameters
        ----------
        rare_thresh : float
            Percentage below which a category is flagged as 'rare'.

        Returns dict of {column: value_counts Series}.
        """
        _section("Phase 2 · Categorical frequencies")
        cols = cols or self._cat_cols
        results = {}
        for col in cols:
            vc = self.df[col].value_counts(normalize=True).mul(100).round(2)
            rare = vc[vc < rare_thresh].index.tolist()
            results[col] = vc
            print(f"\n  {col}  (nunique={self.df[col].nunique()}, "
                  f"rare (<{rare_thresh}%): {len(rare)} categories)")
            print(f"  {vc.head(10).to_string()}")

        if show_plot and cols:
            ncols = min(3, len(cols))
            nrows = int(np.ceil(len(cols) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3))
            axes = np.array(axes).flatten()
            for i, col in enumerate(cols):
                results[col].head(15).plot.bar(ax=axes[i], color="#1D9E75")
                axes[i].set_title(col, fontsize=9)
                axes[i].tick_params(axis="x", rotation=45, labelsize=7)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.suptitle("Categorical frequencies (top 15)")
            plt.tight_layout()
            plt.show()

        return results

    def extract_datetime_features(
        self,
        cols: Optional[List[str]] = None,
        drop_original: bool = False,
    ) -> pd.DataFrame:
        """
        Phase 2 — Extract year, month, day-of-week, hour, is_weekend,
        and cyclic sin/cos encodings from datetime columns.

        Returns the updated DataFrame.
        """
        _section("Phase 2 · Datetime feature extraction")
        cols = cols or self._datetime_cols
        if not cols:
            print("  No datetime columns detected.")
            return self.df

        for col in cols:
            dt = pd.to_datetime(self.df[col])
            self.df[f"{col}_year"]     = dt.dt.year
            self.df[f"{col}_month"]    = dt.dt.month
            self.df[f"{col}_day"]      = dt.dt.day
            self.df[f"{col}_dayofweek"]= dt.dt.dayofweek
            self.df[f"{col}_hour"]     = dt.dt.hour
            self.df[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
            # cyclic encodings
            self.df[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
            self.df[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
            self.df[f"{col}_hour_sin"]  = np.sin(2 * np.pi * dt.dt.hour / 24)
            self.df[f"{col}_hour_cos"]  = np.cos(2 * np.pi * dt.dt.hour / 24)
            new_cols = [c for c in self.df.columns if c.startswith(f"{col}_")]
            print(f"  {col} → {new_cols}")
            if drop_original:
                self.df.drop(columns=[col], inplace=True)

        self._classify_columns()
        return self.df

    def extract_text_features(
        self,
        text_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Phase 2 — Basic text feature extraction: token count, char count,
        average word length, and unique word ratio.
        """
        _section("Phase 2 · Text feature extraction")
        if not text_cols:
            text_cols = [c for c in self.df.select_dtypes(include=["object", "string"]).columns
                         if c != self.target and self.df[c].str.split().str.len().mean() > 3]

        if not text_cols:
            print("  No free-text columns detected.")
            return self.df

        for col in text_cols:
            self.df[f"{col}_token_count"]     = self.df[col].str.split().str.len()
            self.df[f"{col}_char_count"]      = self.df[col].str.len()
            self.df[f"{col}_avg_word_len"]    = (
                self.df[col].str.split().apply(
                    lambda ws: np.mean([len(w) for w in ws]) if ws else 0
                )
            )
            self.df[f"{col}_unique_word_ratio"] = (
                self.df[col].str.split().apply(
                    lambda ws: len(set(ws)) / len(ws) if ws else 0
                )
            )
            print(f"  {col} → 4 text features extracted.")

        self._classify_columns()
        return self.df

    # =========================================================================
    #  PHASE 3 – Bivariate: feature → target
    # =========================================================================

    def bivariate_scatter(
        self,
        cols: Optional[List[str]] = None,
        lowess: bool = True,
        max_cols: int = 12,
    ) -> None:
        """
        Phase 3 — Scatter plot + optional LOWESS smoother for each numeric
        feature vs. the target. Curvature in LOWESS → non-linear relationship.
        """
        _section("Phase 3 · Scatter + LOWESS (numeric vs target)")
        cols = cols or [c for c in self._numeric_cols if c != self.target][:max_cols]
        y = self._encode_target()          # always numeric — safe for LOWESS & scatter
        n = len(cols)
        if n == 0:
            print("  No numeric features to plot.")
            return

        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
        axes = np.array(axes).flatten()

        for i, col in enumerate(cols):
            mask = self.df[col].notna() & y.notna()
            x_vals = self.df.loc[mask, col]
            y_vals = y[mask]
            axes[i].scatter(x_vals, y_vals, alpha=0.25, s=8, color="#378ADD")
            if lowess and len(x_vals) > 30:
                from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
                order = np.argsort(x_vals)
                lw = sm_lowess(y_vals.iloc[order], x_vals.iloc[order], frac=0.4)
                axes[i].plot(lw[:, 0], lw[:, 1], color="#E8593C", lw=2)
            axes[i].set_xlabel(col, fontsize=8)
            axes[i].set_ylabel(self.target, fontsize=8)
            axes[i].set_title(col, fontsize=9)
            axes[i].tick_params(labelsize=7)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"Feature vs {self.target}  (red = LOWESS smoother)", y=1.01)
        plt.tight_layout()
        plt.show()

    def correlation_analysis(
        self,
        cols: Optional[List[str]] = None,
        show_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Phase 3 — Pearson AND Spearman correlations with the target.

        Large Spearman >> Pearson → monotonic non-linear relationship.
        Both ≈ 0 → complex / no relationship.

        Returns a DataFrame sorted by |Spearman|.
        """
        _section("Phase 3 · Pearson vs Spearman correlation with target")
        cols = cols or [c for c in self._numeric_cols if c != self.target]
        y = self._encode_target()          # numeric regardless of original dtype
        rows = []
        for col in cols:
            mask = self.df[col].notna() & y.notna()
            x = self.df.loc[mask, col]
            yy = y[mask]
            pearson, p_p  = stats.pearsonr(x, yy)
            spearman, p_s = stats.spearmanr(x, yy)
            delta = abs(spearman) - abs(pearson)
            rows.append({
                "feature":    col,
                "pearson_r":  round(float(pearson), 4),
                "pearson_p":  round(float(p_p), 4),
                "spearman_r": round(float(spearman), 4),
                "spearman_p": round(float(p_s), 4),
                "|spearman|-|pearson|": round(float(delta), 4),
                "signal":     "non-linear mono." if delta > 0.1
                              else ("linear" if abs(pearson) > 0.3 else "weak / none"),
            })

        df_corr = pd.DataFrame(rows).set_index("feature") \
                                    .sort_values("|spearman|-|pearson|", ascending=False)
        print(df_corr.to_string())

        if show_plot:
            fig, axes = plt.subplots(1, 2, figsize=self.figsize_base)
            top = df_corr.sort_values("pearson_r", key=abs, ascending=False).head(20)
            top["pearson_r"].sort_values().plot.barh(ax=axes[0], color="#185FA5")
            axes[0].set_title("Pearson r")
            top["spearman_r"].sort_values().plot.barh(ax=axes[1], color="#0F6E56")
            axes[1].set_title("Spearman r")
            plt.suptitle(f"Correlation with {self.target}")
            plt.tight_layout()
            plt.show()

        self._report["correlation"] = df_corr
        return df_corr

    def mutual_information_scores(
        self,
        cols: Optional[List[str]] = None,
        show_plot: bool = True,
    ) -> pd.Series:
        """
        Phase 3 — Mutual Information scores for all features vs target.

        MI > 0 with near-zero Pearson → non-linear predictive signal.
        Uses sklearn mutual_info_regression / _classif.

        Returns a Series sorted descending.
        """
        _section("Phase 3 · Mutual Information scores")
        X = self._X().select_dtypes(include=np.number).dropna()
        y = self._encode_target()[X.index]  # encoded → always numeric
        mi = self._mi_func()(X, y, random_state=42)
        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

        print(mi_series.to_string())

        if show_plot:
            mi_series.head(25).sort_values().plot.barh(
                figsize=(10, max(4, len(mi_series.head(25)) * 0.35)),
                color="#533AB7"
            )
            plt.title("Mutual Information scores (top 25)")
            plt.xlabel("MI score")
            plt.tight_layout()
            plt.show()

        self._report["mutual_info"] = mi_series
        return mi_series

    def categorical_vs_target(
        self,
        cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Phase 3 — Box/violin plots + ANOVA F-test (regression) or chi-squared
        (classification) for each categorical feature vs the target.

        Returns a DataFrame with test statistics and p-values.
        """
        _section("Phase 3 · Categorical feature vs target")
        cols = cols or self._cat_cols
        # use encoded target for statistical tests (kruskal/ANOVA need floats)
        y = self._encode_target()
        rows = []

        for col in cols:
            groups = [y[self.df[col] == val].dropna()
                      for val in self.df[col].dropna().unique()]
            groups = [g for g in groups if len(g) > 1]
            if not groups:
                continue

            if self.task == "regression":
                stat, p = stats.f_oneway(*groups)
                test = "ANOVA F"
            else:
                stat, p = stats.kruskal(*groups)
                test = "Kruskal-Wallis H"

            rows.append({"feature": col, "test": test,
                         "statistic": round(float(stat), 4),
                         "p_value": round(float(p), 6)})

        df_cat = pd.DataFrame(rows).set_index("feature").sort_values("p_value") if rows else pd.DataFrame()
        if not df_cat.empty:
            print(df_cat.to_string())

        # visualise top features — use encoded target column for ordering/boxplot
        if not df_cat.empty:
            sig = df_cat[df_cat["p_value"] < 0.05].head(6).index.tolist()
            if sig:
                tmp = self.df.copy()
                tmp["__target_enc__"] = y.values
                fig, axes = plt.subplots(1, len(sig), figsize=(4 * len(sig), 4))
                axes = [axes] if len(sig) == 1 else axes
                for i, col in enumerate(sig):
                    order = (tmp.groupby(col)["__target_enc__"].median()
                                .sort_values().index.tolist())
                    sns.boxplot(data=tmp, x=col, y="__target_enc__",
                                order=order, ax=axes[i], color="#1D9E75")
                    axes[i].set_title(f"{col}\np={df_cat.loc[col,'p_value']:.4f}", fontsize=9)
                    axes[i].set_ylabel(self.target)
                    axes[i].tick_params(axis="x", rotation=30, labelsize=7)
                plt.suptitle(f"Significant categoricals vs {self.target} (p<0.05)")
                plt.tight_layout()
                plt.show()

        self._report["categorical_vs_target"] = df_cat
        return df_cat

    def chi_squared_test(
        self,
        cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Phase 3 — Chi-squared test + Cramér's V for categorical features
        vs a categorical target (classification tasks).
        """
        _section("Phase 3 · Chi-squared & Cramér's V (categorical target)")
        if self.task != "classification":
            print("  Skipped — only applicable for classification tasks.")
            return pd.DataFrame()

        cols = cols or self._cat_cols
        y = self._y().astype(str)
        rows = []
        for col in cols:
            ct = pd.crosstab(self.df[col].astype(str), y)
            chi2, p, _, _ = stats.chi2_contingency(ct)
            v = _cramers_v(self.df[col].astype(str), y)
            rows.append({"feature": col, "chi2": round(float(chi2), 3),
                         "p_value": round(float(p), 6),
                         "cramers_v": round(v, 4)})

        df_chi = pd.DataFrame(rows).set_index("feature").sort_values("cramers_v", ascending=False)
        print(df_chi.to_string())
        self._report["chi_squared"] = df_chi
        return df_chi

    def partial_regression_plots(
        self,
        cols: Optional[List[str]] = None,
        max_cols: int = 6,
    ) -> None:
        """
        Phase 3 — Partial regression (added-variable) plots using statsmodels.

        Exposes the true marginal relationship between each feature and the
        target after controlling for all other variables (FWL theorem).
        """
        _section("Phase 3 · Partial regression plots (FWL)")
        if self.task != "regression":
            print("  Partial regression is for regression tasks only.")
            return

        num_feats = [c for c in self._numeric_cols if c != self.target][:max_cols]
        if not num_feats:
            print("  Not enough numeric features.")
            return

        sub = self.df[[self.target] + num_feats].dropna()
        X = sm.add_constant(sub[num_feats])
        y = sub[self.target]
        model = sm.OLS(y, X).fit()

        fig = plt.figure(figsize=(14, 3 * int(np.ceil(len(num_feats) / 3))))
        sm.graphics.plot_partregress_grid(model, fig=fig)
        plt.suptitle("Partial regression plots (marginal effect per feature)", y=1.01)
        plt.tight_layout()
        plt.show()

    # =========================================================================
    #  PHASE 4 – Linearity vs non-linearity diagnosis
    # =========================================================================

    def residual_diagnostics(self, show_plot: bool = True) -> dict:
        """
        Phase 4 — Fit OLS / Logistic on numeric features; inspect residuals.

        Plots:
          • Residuals vs Fitted (pattern → non-linearity)
          • Scale-Location         (funnel → heteroscedasticity)
          • Q-Q plot               (normality of residuals)

        Returns dict with model summary stats.
        """
        _section("Phase 4 · Linear model residual diagnostics")
        num_feats = [c for c in self._numeric_cols if c != self.target]
        if not num_feats:
            print("  No numeric features for OLS.")
            return {}

        sub_idx = self.df[num_feats].dropna().index
        X_sub   = sm.add_constant(self.df.loc[sub_idx, num_feats])
        y_sub   = self._encode_target()[sub_idx]   # always float-safe

        if self.task == "regression":
            model = sm.OLS(y_sub, X_sub).fit()
            fitted = model.fittedvalues
            residuals = model.resid
            result = {"r2": round(model.rsquared, 4), "aic": round(model.aic, 2)}
        else:
            model = sm.Logit(y_sub.astype(float), X_sub).fit(disp=0)
            fitted = model.fittedvalues
            residuals = model.resid_response
            result = {"pseudo_r2": round(model.prsquared, 4), "aic": round(model.aic, 2)}

        if show_plot and self.task == "regression":
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].scatter(fitted, residuals, alpha=0.3, s=8, color="#378ADD")
            axes[0].axhline(0, color="#E8593C", lw=1)
            axes[0].set_xlabel("Fitted values"); axes[0].set_ylabel("Residuals")
            axes[0].set_title("Residuals vs Fitted")

            sqrt_abs_res = np.sqrt(np.abs(residuals))
            axes[1].scatter(fitted, sqrt_abs_res, alpha=0.3, s=8, color="#1D9E75")
            axes[1].set_xlabel("Fitted values"); axes[1].set_ylabel("√|Residuals|")
            axes[1].set_title("Scale-Location")

            stats.probplot(residuals, dist="norm", plot=axes[2])
            axes[2].set_title("Q-Q plot (residuals)")

            plt.suptitle("OLS residual diagnostics")
            plt.tight_layout()
            plt.show()

        print(f"  Model summary: {result}")
        self._report["residual_diagnostics"] = result
        return result

    def reset_test(
        self,
        cols: Optional[List[str]] = None,
    ) -> dict:
        """
        Phase 4 — Ramsey RESET test for functional form misspecification.

        A significant p-value (< 0.05) indicates that higher-order (non-linear)
        terms of the features improve the model → non-linearity present.
        """
        _section("Phase 4 · Ramsey RESET test")
        if self.task != "regression":
            print("  RESET test is for regression tasks only.")
            return {}

        cols = cols or [c for c in self._numeric_cols if c != self.target]
        sub_idx = self.df[cols].dropna().index
        X = sm.add_constant(self.df.loc[sub_idx, cols])
        y = self._encode_target()[sub_idx]
        model = sm.OLS(y, X).fit()
        reset = linear_reset(model, power=3, use_f=True)
        result = {
            "F_statistic": round(float(reset.statistic), 4),
            "p_value":     round(float(reset.pvalue), 6),
            "verdict":     "non-linearity detected" if reset.pvalue < 0.05
                           else "no significant non-linearity",
        }
        for k, v in result.items():
            print(f"  {k:<25} {v}")

        self._report["reset_test"] = result
        return result

    def partial_dependence_plots(
        self,
        cols: Optional[List[str]] = None,
        max_cols: int = 6,
    ) -> None:
        """
        Phase 4 — Partial Dependence Plots (PDP) using a gradient-boosted model.

        A straight PDP → near-linear contribution.
        A curved / humped PDP → non-linear; consider explicit encoding.
        """
        _section("Phase 4 · Partial Dependence Plots (PDP)")
        cols = cols or [c for c in self._numeric_cols if c != self.target][:max_cols]
        sub_idx = self.df[cols].dropna().index
        X = self.df.loc[sub_idx, cols].astype(float)   # cast → avoids sklearn integer warning
        y = self._encode_target()[sub_idx]

        model = self._base_model()
        model.fit(X, y)

        fig, ax = plt.subplots(figsize=self.figsize_base)
        disp = PartialDependenceDisplay.from_estimator(
            model, X, features=list(range(len(cols))),
            feature_names=cols, ax=ax,
            grid_resolution=50,
        )
        plt.suptitle("Partial Dependence Plots — curvature = non-linear", y=1.02)
        plt.tight_layout()
        plt.show()

        # flag non-linear cols by std of PDP gradient
        pdp_results = {}
        for i, col in enumerate(cols):
            pd_values = disp.pd_results[i].average[0]
            grad = np.diff(pd_values)
            std_grad = float(np.std(grad))
            label = "non-linear" if std_grad > np.mean(np.abs(grad)) * 0.5 else "linear"
            pdp_results[col] = {"std_grad": round(std_grad, 4), "signal": label}
            if label == "non-linear":
                self._nonlinear_cols.append(col)

        print(pd.DataFrame(pdp_results).T.to_string())
        self._report["pdp"] = pdp_results

    def shap_dependence(
        self,
        cols: Optional[List[str]] = None,
        max_cols: int = 6,
    ) -> None:
        """
        Phase 4 — SHAP dependence plots.

        Straight SHAP vs feature → linear contribution.
        Curved SHAP → non-linear. Requires the 'shap' package.
        """
        _section("Phase 4 · SHAP dependence plots")
        if not _HAS_SHAP:
            print("  Install 'shap' to use this method: pip install shap")
            return

        cols = cols or [c for c in self._numeric_cols if c != self.target][:max_cols]
        sub_idx = self.df[cols].dropna().index
        X = self.df.loc[sub_idx, cols]
        y = self._encode_target()[sub_idx]

        model = self._base_model()
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        for i, col in enumerate(cols):
            shap.dependence_plot(
                col, shap_values, X,
                show=False, title=f"SHAP dependence: {col}"
            )
            plt.tight_layout()
            plt.show()

    def gam_smooth_terms(
        self,
        cols: Optional[List[str]] = None,
        max_cols: int = 8,
    ) -> pd.DataFrame:
        """
        Phase 4 — Generalised Additive Model smooth terms via pygam.

        EDF (effective degrees of freedom) ≈ 1 → linear spline.
        EDF > 2 → strong non-linearity present.

        Requires 'pygam': pip install pygam.
        """
        _section("Phase 4 · GAM smooth terms (EDF analysis)")
        if not _HAS_GAM:
            print("  Install 'pygam' to use this method: pip install pygam")
            return pd.DataFrame()

        cols = cols or [c for c in self._numeric_cols if c != self.target][:max_cols]
        sub_idx = self.df[cols].dropna().index
        X = self.df.loc[sub_idx, cols].values
        y = self._encode_target()[sub_idx].values

        terms = sum([s(i) for i in range(X.shape[1])])
        gam = (LinearGAM if self.task == "regression" else LogisticGAM)(terms)
        gam.gridsearch(X, y, progress=False)

        rows = []
        for i, col in enumerate(cols):
            edf = round(float(gam.statistics_["edof_per_coef"][i + 1]), 3)
            rows.append({"feature": col, "EDF": edf,
                         "signal": "linear" if edf < 1.5 else
                                   ("mild non-linear" if edf < 3 else "strong non-linear")})

        df_gam = pd.DataFrame(rows).set_index("feature").sort_values("EDF", ascending=False)
        print(df_gam.to_string())

        fig, axes = plt.subplots(int(np.ceil(len(cols) / 4)), 4,
                                 figsize=(16, 3 * int(np.ceil(len(cols) / 4))))
        axes = np.array(axes).flatten()
        for i, col in enumerate(cols):
            XX = gam.generate_X_grid(term=i)
            axes[i].plot(*gam.partial_dependence(term=i, X=XX, width=0.95),
                         color="#185FA5")
            axes[i].set_title(f"{col} (EDF={df_gam.loc[col,'EDF']})", fontsize=9)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("GAM smooth terms — EDF > 2 = non-linear", y=1.01)
        plt.tight_layout()
        plt.show()

        self._report["gam"] = df_gam
        return df_gam

    # =========================================================================
    #  PHASE 5 – Multicollinearity & interactions
    # =========================================================================

    def correlation_matrix(
        self,
        threshold: float = 0.85,
        show_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Phase 5 — Pearson correlation matrix with high-correlation pair detection.

        Parameters
        ----------
        threshold : float
            |r| above this value flags a collinear pair (default 0.85).

        Returns the full correlation matrix.
        """
        _section("Phase 5 · Pearson correlation matrix")
        num_df = self.df[self._numeric_cols].dropna()
        corr = num_df.corr()

        if show_plot:
            mask = np.triu(np.ones_like(corr, dtype=bool))
            fig, ax = plt.subplots(figsize=(min(16, len(corr) * 0.6 + 2),
                                            min(14, len(corr) * 0.6 + 2)))
            sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                        annot=len(corr) <= 20, fmt=".2f", linewidths=0.5, ax=ax,
                        square=True, cbar_kws={"shrink": 0.7})
            ax.set_title("Pearson correlation matrix")
            plt.tight_layout()
            plt.show()

        # report high-correlation pairs
        self._collinear_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) >= threshold:
                    pair = (corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 4))
                    self._collinear_pairs.append(pair)

        if self._collinear_pairs:
            print(f"\n  Collinear pairs (|r| ≥ {threshold}):")
            for a, b, r in self._collinear_pairs:
                print(f"    {a}  ↔  {b}  (r={r})")
        else:
            print(f"  No collinear pairs found above threshold {threshold}.")

        self._report["correlation_matrix"] = corr
        self._report["collinear_pairs"] = self._collinear_pairs
        return corr

    def vif_analysis(
        self,
        cols: Optional[List[str]] = None,
        threshold: float = 10.0,
    ) -> pd.DataFrame:
        """
        Phase 5 — Variance Inflation Factor for multicollinearity.

        VIF > 10 → severe; VIF 5–10 → moderate; VIF < 5 → acceptable.

        Returns a DataFrame of VIF values sorted descending.
        """
        _section("Phase 5 · Variance Inflation Factor (VIF)")
        cols = cols or self._numeric_cols
        sub = self.df[cols].dropna()
        X = sm.add_constant(sub)
        vif_data = pd.DataFrame({
            "feature": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        }).set_index("feature").drop(index="const", errors="ignore")
        vif_data["VIF"] = vif_data["VIF"].round(2)
        vif_data["verdict"] = vif_data["VIF"].apply(
            lambda v: "severe" if v > 10 else ("moderate" if v > 5 else "ok")
        )
        vif_data = vif_data.sort_values("VIF", ascending=False)
        print(vif_data.to_string())
        high_vif = vif_data[vif_data["VIF"] > threshold].index.tolist()
        if high_vif:
            print(f"\n  High VIF (>{threshold}) features: {high_vif}")

        self._report["vif"] = vif_data
        return vif_data

    def cramers_v_matrix(
        self,
        cols: Optional[List[str]] = None,
        show_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Phase 5 — Cramér's V association matrix for categorical columns.

        Returns the symmetric Cramér's V matrix.
        """
        _section("Phase 5 · Cramér's V matrix (categoricals)")
        cols = cols or self._cat_cols
        if len(cols) < 2:
            print("  Need ≥ 2 categorical columns.")
            return pd.DataFrame()

        n = len(cols)
        mat = np.zeros((n, n))
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                mat[i, j] = 1.0 if i == j else _cramers_v(
                    self.df[c1].astype(str), self.df[c2].astype(str)
                )

        df_v = pd.DataFrame(mat, index=cols, columns=cols).round(4)
        if show_plot:
            fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.7)))
            sns.heatmap(df_v, annot=n <= 15, fmt=".2f", cmap="YlOrRd",
                        vmin=0, vmax=1, ax=ax, linewidths=0.5)
            ax.set_title("Cramér's V — categorical association matrix")
            plt.tight_layout()
            plt.show()

        self._report["cramers_v"] = df_v
        return df_v

    def interaction_screening(
        self,
        top_k: int = 5,
        show_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Phase 5 — Screen pairwise interaction terms (products) of the top-K
        most important numeric features using permutation importance.

        Returns a DataFrame of interaction terms ranked by importance gain.
        """
        _section("Phase 5 · Interaction term screening")
        mi = self._report.get("mutual_info")
        if mi is None:
            mi = self.mutual_information_scores(show_plot=False)

        top_feats = mi.head(top_k).index.tolist()
        sub_idx = self.df[top_feats].dropna().index
        X_base = self.df.loc[sub_idx, top_feats]
        y = self._encode_target()[sub_idx]

        model = self._base_model()
        base_score = cross_val_score(model, X_base, y, cv=3,
                                     scoring=("r2" if self.task == "regression"
                                              else "roc_auc")).mean()

        rows = []
        for i in range(len(top_feats)):
            for j in range(i + 1, len(top_feats)):
                a, b = top_feats[i], top_feats[j]
                X_int = X_base.copy()
                X_int[f"{a}_x_{b}"] = X_base[a] * X_base[b]
                score = cross_val_score(model, X_int, y, cv=3,
                                        scoring=("r2" if self.task == "regression"
                                                 else "roc_auc")).mean()
                rows.append({"interaction": f"{a} × {b}",
                             "base_score": round(float(base_score), 4),
                             "interaction_score": round(float(score), 4),
                             "gain": round(float(score - base_score), 4)})

        df_int = pd.DataFrame(rows).set_index("interaction").sort_values("gain", ascending=False)
        print(df_int.to_string())

        if show_plot and not df_int.empty:
            df_int["gain"].sort_values().plot.barh(
                figsize=(10, max(4, len(df_int) * 0.35)), color="#854F0B"
            )
            plt.title("Interaction term gain")
            plt.xlabel("Score gain over baseline")
            plt.tight_layout()
            plt.show()

        self._report["interactions"] = df_int
        return df_int

    def feature_clustering(
        self,
        n_clusters: int = 5,
        show_plot: bool = True,
    ) -> dict:
        """
        Phase 5 — Hierarchical clustering of features by Spearman correlation.

        Each cluster → pick the feature with the highest MI as representative.
        Returns dict {cluster_id: [features]}.
        """
        _section("Phase 5 · Feature clustering (Spearman dendrogram)")
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform

        num_df = self.df[self._numeric_cols].dropna()
        if num_df.shape[1] < 3:
            print("  Need ≥ 3 numeric features for clustering.")
            return {}

        corr_mat = num_df.corr(method="spearman").abs()
        dist_mat = 1 - corr_mat
        condensed = squareform(dist_mat.values, checks=False)
        linkage = hierarchy.ward(condensed)

        if show_plot:
            fig, ax = plt.subplots(figsize=(14, 5))
            hierarchy.dendrogram(linkage, labels=num_df.columns.tolist(),
                                 leaf_rotation=45, ax=ax)
            ax.set_title("Feature clustering dendrogram (Spearman distance)")
            plt.tight_layout()
            plt.show()

        labels = hierarchy.fcluster(linkage, n_clusters, criterion="maxclust")
        clusters: dict = {}
        for col, lab in zip(num_df.columns, labels):
            clusters.setdefault(int(lab), []).append(col)

        mi = self._report.get("mutual_info", pd.Series(dtype=float))
        print("\n  Clusters & recommended representative:")
        for cid, members in sorted(clusters.items()):
            if mi.empty:
                rep = members[0]
            else:
                rep = mi[mi.index.isin(members)].idxmax() if any(m in mi.index for m in members) else members[0]
            print(f"  Cluster {cid}: {members}  →  keep: {rep}")

        self._report["feature_clusters"] = clusters
        return clusters

    # =========================================================================
    #  PHASE 6 – Transformations & encoding
    # =========================================================================

    def apply_transforms(
        self,
        cols: Optional[List[str]] = None,
        method: str = "yeo-johnson",
        skew_thresh: float = 0.75,
        inplace: bool = True,
    ) -> pd.DataFrame:
        """
        Phase 6 — Apply power transforms to skewed numeric columns.

        Parameters
        ----------
        method : {'yeo-johnson', 'box-cox', 'log1p', 'sqrt'}
            'box-cox' requires strictly positive data.
        skew_thresh : float
            Only transform columns with |skew| > this value.
        """
        _section("Phase 6 · Power transforms for skewed features")
        cols = cols or self._numeric_cols
        target_df = self.df if inplace else self.df.copy()

        for col in cols:
            if col == self.target:
                continue
            skew = float(target_df[col].skew())
            if abs(skew) <= skew_thresh:
                continue

            orig = target_df[col].dropna().values.reshape(-1, 1)

            if method == "log1p":
                if (orig < 0).any():
                    print(f"  Skipping {col} (log1p requires non-negative values).")
                    continue
                target_df[f"{col}_log1p"] = np.log1p(target_df[col])
                new_skew = float(target_df[f"{col}_log1p"].skew())
                self._transformations_applied[col] = "log1p"

            elif method == "sqrt":
                if (orig < 0).any():
                    print(f"  Skipping {col} (sqrt requires non-negative values).")
                    continue
                target_df[f"{col}_sqrt"] = np.sqrt(target_df[col])
                new_skew = float(target_df[f"{col}_sqrt"].skew())
                self._transformations_applied[col] = "sqrt"

            else:
                try:
                    pt = PowerTransformer(method=method, standardize=False)
                    transformed = pt.fit_transform(orig).flatten()
                    target_df[f"{col}_{method.replace('-','_')}"] = np.where(
                        target_df[col].notna(), np.nan, np.nan
                    )
                    idx = target_df[col].dropna().index
                    target_df.loc[idx, f"{col}_{method.replace('-','_')}"] = transformed
                    new_skew = float(target_df[f"{col}_{method.replace('-','_')}"].skew())
                    self._transformations_applied[col] = method
                except Exception as e:
                    print(f"  {col}: transform failed — {e}")
                    continue

            print(f"  {col}: skew {skew:.3f} → {new_skew:.3f}  [{method}]")

        return target_df

    def polynomial_spline_features(
        self,
        cols: Optional[List[str]] = None,
        degree: int = 2,
        use_spline: bool = True,
        n_knots: int = 5,
        inplace: bool = True,
    ) -> pd.DataFrame:
        """
        Phase 6 — Add polynomial or spline basis expansions for non-linear features.

        Only applied to columns flagged as non-linear in Phase 4 (self._nonlinear_cols),
        or to the explicitly provided list.

        Parameters
        ----------
        degree : int
            Polynomial degree (default 2).
        use_spline : bool
            If True uses SplineTransformer; otherwise PolynomialFeatures.
        n_knots : int
            Number of spline knots (quantile-placed).
        """
        _section("Phase 6 · Polynomial / spline feature expansion")
        cols = cols or self._nonlinear_cols or self._numeric_cols[:5]
        target_df = self.df if inplace else self.df.copy()

        for col in cols:
            if col == self.target:
                continue
            vals = target_df[[col]].dropna()

            if use_spline:
                transformer = SplineTransformer(
                    n_knots=n_knots, degree=degree,
                    knots="quantile", include_bias=False
                )
                name_prefix = f"{col}_spline"
            else:
                transformer = PolynomialFeatures(
                    degree=degree, include_bias=False
                )
                name_prefix = f"{col}_poly"

            try:
                new_vals = transformer.fit_transform(vals)
                new_cols = [f"{name_prefix}_{i}" for i in range(new_vals.shape[1])]
                for k, nc in enumerate(new_cols):
                    target_df.loc[vals.index, nc] = new_vals[:, k]
                print(f"  {col} → {new_cols}")
            except Exception as e:
                print(f"  {col}: failed — {e}")

        self._classify_columns()
        return target_df

    def target_encode(
        self,
        cols: Optional[List[str]] = None,
        smoothing: int = 10,
        inplace: bool = True,
    ) -> pd.DataFrame:
        """
        Phase 6 — Target encoding for high-cardinality categorical columns.

        Uses category_encoders.TargetEncoder with Laplace smoothing.
        Falls back to simple mean encoding if category_encoders is absent.
        """
        _section("Phase 6 · Target encoding (high-cardinality categoricals)")
        cols = cols or self._high_card_cols
        target_df = self.df if inplace else self.df.copy()
        # always use numeric-encoded target so groupby .mean() and TargetEncoder work
        y_enc = self._encode_target()
        target_df["__y_enc__"] = y_enc.values

        for col in cols:
            if _HAS_CE:
                enc = ce.TargetEncoder(cols=[col], smoothing=smoothing)
                target_df[f"{col}_te"] = enc.fit_transform(
                    target_df[[col]], target_df["__y_enc__"]
                )[col]
            else:
                means = target_df.groupby(col)["__y_enc__"].mean()
                global_mean = float(y_enc.mean())
                counts = target_df.groupby(col)["__y_enc__"].count()
                smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
                target_df[f"{col}_te"] = target_df[col].map(smoothed).fillna(global_mean)

            print(f"  {col} → {col}_te (target-encoded)")

        target_df.drop(columns=["__y_enc__"], inplace=True)

        return target_df

    def cyclic_encode(
        self,
        col_period: Optional[dict] = None,
        inplace: bool = True,
    ) -> pd.DataFrame:
        """
        Phase 6 — Cyclic sin/cos encoding for periodic features.

        Parameters
        ----------
        col_period : dict
            {column_name: period}. E.g. {"hour": 24, "month": 12, "day_of_week": 7}.
            If None, auto-detects columns named *hour*, *month*, *day*.
        """
        _section("Phase 6 · Cyclic encoding (periodic features)")
        target_df = self.df if inplace else self.df.copy()

        if col_period is None:
            col_period = {}
            for col in target_df.columns:
                lc = col.lower()
                if "hour" in lc:
                    col_period[col] = 24
                elif "month" in lc:
                    col_period[col] = 12
                elif "day" in lc and "week" in lc:
                    col_period[col] = 7
                elif "minute" in lc:
                    col_period[col] = 60

        for col, period in col_period.items():
            if col not in target_df.columns:
                continue
            target_df[f"{col}_sin"] = np.sin(2 * np.pi * target_df[col] / period)
            target_df[f"{col}_cos"] = np.cos(2 * np.pi * target_df[col] / period)
            print(f"  {col} (period={period}) → {col}_sin, {col}_cos")

        return target_df

    def scale_features(
        self,
        cols: Optional[List[str]] = None,
        method: str = "standard",
        inplace: bool = True,
    ) -> pd.DataFrame:
        """
        Phase 6 — Scale numeric features.

        Parameters
        ----------
        method : {'standard', 'robust', 'minmax'}
            standard → zero mean, unit variance (linear / SVM / KNN).
            robust   → median / IQR (handles outliers).
            minmax   → [0, 1] range (neural networks).
        """
        _section("Phase 6 · Feature scaling")
        cols = cols or [c for c in self._numeric_cols if c != self.target]
        target_df = self.df if inplace else self.df.copy()
        scalers = {"standard": StandardScaler(),
                   "robust":   RobustScaler(),
                   "minmax":   MinMaxScaler()}
        scaler = scalers.get(method, StandardScaler())
        sub = target_df[cols].dropna()
        scaled = scaler.fit_transform(sub)
        scaled_cols = [f"{c}_scaled" for c in cols]
        for i, sc in enumerate(scaled_cols):
            target_df.loc[sub.index, sc] = scaled[:, i]
        print(f"  Scaled {len(cols)} features using {method}Scaler → suffix '_scaled'")
        return target_df

    # =========================================================================
    #  PHASE 7 – Feature selection & validation
    # =========================================================================

    def mutual_info_ranking(
        self,
        top_n: int = 20,
        show_plot: bool = True,
    ) -> pd.Series:
        """
        Phase 7 — Rank all features by Mutual Information score.

        Returns top-N features above the 'elbow' in the sorted MI curve.
        """
        _section("Phase 7 · Mutual Information ranking (filter)")
        mi = self.mutual_information_scores(show_plot=False)
        top = mi.head(top_n)

        if show_plot:
            fig, axes = plt.subplots(1, 2, figsize=self.figsize_base)
            mi.sort_values(ascending=False).head(top_n).plot.bar(ax=axes[0], color="#533AB7")
            axes[0].set_title(f"MI scores (top {top_n})")
            axes[0].tick_params(axis="x", rotation=45, labelsize=7)
            axes[1].plot(range(len(mi)), mi.sort_values(ascending=False).values,
                         marker="o", ms=4, color="#533AB7")
            axes[1].set_title("Elbow plot — select above the kink")
            axes[1].set_xlabel("Feature rank")
            axes[1].set_ylabel("MI score")
            plt.tight_layout()
            plt.show()

        return top

    def rfecv_selection(
        self,
        estimator=None,
        min_features: int = 5,
        show_plot: bool = True,
    ) -> List[str]:
        """
        Phase 7 — Recursive Feature Elimination with Cross-Validation (RFECV).

        Returns a list of selected feature names.
        """
        _section("Phase 7 · RFECV feature selection (wrapper)")
        num_feats = [c for c in self._numeric_cols if c != self.target]
        sub_idx = self.df[num_feats].dropna().index
        X = self.df.loc[sub_idx, num_feats]
        y = self._encode_target()[sub_idx]

        if estimator is None:
            if self.task == "regression":
                from sklearn.linear_model import Ridge
                estimator = Ridge()
            else:
                from sklearn.linear_model import LogisticRegression
                estimator = LogisticRegression(max_iter=500)

        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=self._cv(),
            min_features_to_select=min_features,
            scoring=("r2" if self.task == "regression" else "roc_auc"),
            n_jobs=-1,
        )
        selector.fit(X, y)
        selected = [f for f, s in zip(num_feats, selector.support_) if s]
        print(f"  Optimal features: {selector.n_features_}  →  {selected}")

        if show_plot:
            plt.figure(figsize=(10, 4))
            plt.plot(range(1, len(selector.cv_results_["mean_test_score"]) + 1),
                     selector.cv_results_["mean_test_score"], color="#185FA5", marker="o", ms=4)
            plt.axvline(selector.n_features_, color="#E8593C", ls="--", label="optimal")
            plt.xlabel("Number of features")
            plt.ylabel("CV score")
            plt.title("RFECV — optimal feature count")
            plt.legend()
            plt.tight_layout()
            plt.show()

        self._selected_features = selected
        self._report["rfecv_selected"] = selected
        return selected

    def lasso_importance(
        self,
        alpha: float = 0.01,
        show_plot: bool = True,
    ) -> pd.Series:
        """
        Phase 7 — LASSO (L1) regularisation to identify zero-coefficient features.

        Non-zero coefficients → predictive. Zero → irrelevant / collinear.
        Returns a Series of |coefficients| sorted descending.
        """
        _section("Phase 7 · LASSO / L1 embedded importance")
        num_feats = [c for c in self._numeric_cols if c != self.target]
        sub_idx = self.df[num_feats].dropna().index
        X = StandardScaler().fit_transform(self.df.loc[sub_idx, num_feats])
        y = self._encode_target()[sub_idx].values

        if self.task == "regression":
            model = Lasso(alpha=alpha, max_iter=5000)
        else:
            model = LogisticRegression(solver="saga", C=1 / alpha, max_iter=2000, l1_ratio=1.0)

        model.fit(X, y)
        coefs = model.coef_.flatten() if hasattr(model, "coef_") else np.zeros(len(num_feats))
        imp = pd.Series(np.abs(coefs), index=num_feats).sort_values(ascending=False)
        print(f"  Features selected (coef > 0): {(imp > 0).sum()}")
        print(imp[imp > 0].to_string())

        if show_plot:
            imp.head(20).sort_values().plot.barh(figsize=(10, 6), color="#A32D2D")
            plt.title(f"LASSO |coefficient| (α={alpha})")
            plt.xlabel("|coefficient|")
            plt.tight_layout()
            plt.show()

        self._report["lasso_importance"] = imp
        return imp

    def permutation_importance_analysis(
        self,
        n_repeats: int = 10,
        show_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Phase 7 — Permutation importance using a gradient-boosted model.

        More reliable than impurity-based importance; catches correlated features.
        Returns a DataFrame of importances with mean ± std.
        """
        _section("Phase 7 · Permutation importance")
        num_feats = [c for c in self._numeric_cols if c != self.target]
        sub_idx = self.df[num_feats].dropna().index
        X = self.df.loc[sub_idx, num_feats]
        y = self._encode_target()[sub_idx]

        model = self._base_model()
        model.fit(X, y)

        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=42,
            scoring=("r2" if self.task == "regression" else "roc_auc"),
        )
        df_perm = pd.DataFrame({
            "feature": num_feats,
            "mean_importance": result.importances_mean.round(4),
            "std":             result.importances_std.round(4),
        }).set_index("feature").sort_values("mean_importance", ascending=False)
        print(df_perm.to_string())

        if show_plot:
            fig, ax = plt.subplots(figsize=(10, max(5, len(df_perm) * 0.35)))
            df_perm["mean_importance"].sort_values().plot.barh(ax=ax, color="#3B6D11",
                                                               xerr=df_perm["std"].reindex(
                                                                   df_perm["mean_importance"].sort_values().index))
            ax.set_title("Permutation importance (mean ± std)")
            ax.set_xlabel("Mean score decrease")
            plt.tight_layout()
            plt.show()

        self._report["permutation_importance"] = df_perm
        return df_perm

    def leakage_audit(
        self,
        correlation_thresh: float = 0.99,
        verbose: bool = True,
    ) -> List[str]:
        """
        Phase 7 — Data leakage audit.

        Flags:
          • Features with near-perfect correlation with the target (> threshold).
          • Columns whose names suggest they are derived from the target
            (e.g. contain the target name as a substring).
          • Datetime columns that might encode future information.

        Returns a list of suspicious column names.
        """
        _section("Phase 7 · Leakage audit")
        suspicious = []
        y = self._encode_target()          # always numeric — safe for pearsonr

        # near-perfect correlation with target
        for col in self._numeric_cols:
            if col == self.target:
                continue
            mask = self.df[col].notna() & y.notna()
            if mask.sum() < 10:
                continue
            r, _ = stats.pearsonr(self.df.loc[mask, col], y[mask])
            if abs(r) >= correlation_thresh:
                suspicious.append(col)
                if verbose:
                    print(f"  [HIGH CORR WITH TARGET] {col}  r={r:.4f}")

        # name-based heuristic
        target_lower = self.target.lower()
        for col in self._features():
            if col == self.target:
                continue
            if target_lower in col.lower() and col != self.target:
                suspicious.append(col)
                if verbose:
                    print(f"  [NAME OVERLAP WITH TARGET] {col}")

        if not suspicious:
            print("  No obvious leakage detected.")
        else:
            print(f"\n  Suspicious columns: {suspicious}")
            print("  Review carefully before training!")

        self._report["leakage_suspicious"] = suspicious
        return suspicious

    # =========================================================================
    #  ORCHESTRATION
    # =========================================================================

    def run_phase(self, phase: int) -> None:
        """Run a single phase (1–7) with its core methods."""
        phases = {
            1: self._run_phase1,
            2: self._run_phase2,
            3: self._run_phase3,
            4: self._run_phase4,
            5: self._run_phase5,
            6: self._run_phase6,
            7: self._run_phase7,
        }
        if phase not in phases:
            raise ValueError("phase must be an integer 1–7.")
        phases[phase]()

    def run_all(self, phases: Optional[List[int]] = None) -> dict:
        """
        Execute all (or selected) phases sequentially and return the final report.

        Parameters
        ----------
        phases : list of int, optional
            Subset of phases to run (default: all 1–7).
        """
        phases = phases or [1, 2, 3, 4, 5, 6, 7]
        for p in phases:
            self.run_phase(p)
        return self.report()

    def _run_phase1(self):
        self.audit()
        self.missing_map()
        self.cardinality_check()
        self.target_profile()
        self.drop_useless()

    def _run_phase2(self):
        self.distribution_plots()
        self.detect_outliers()
        self.categorical_frequencies()
        if self._datetime_cols:
            self.extract_datetime_features()

    def _run_phase3(self):
        self.bivariate_scatter()
        self.correlation_analysis()
        self.mutual_information_scores()
        self.categorical_vs_target()
        if self.task == "classification":
            self.chi_squared_test()

    def _run_phase4(self):
        self.residual_diagnostics()
        if self.task == "regression":
            self.reset_test()
        self.partial_dependence_plots()

    def _run_phase5(self):
        self.correlation_matrix()
        self.vif_analysis()
        if self._cat_cols:
            self.cramers_v_matrix()
        self.interaction_screening()
        self.feature_clustering()

    def _run_phase6(self):
        self.apply_transforms()
        self.polynomial_spline_features()
        if self._high_card_cols:
            self.target_encode()
        self.cyclic_encode()

    def _run_phase7(self):
        self.mutual_info_ranking()
        self.rfecv_selection()
        self.lasso_importance()
        self.permutation_importance_analysis()
        self.leakage_audit()

    def report(self) -> dict:
        """Return a dict summarising all findings accumulated so far."""
        return {
            "task":                 self.task,
            "shape":                self.df.shape,
            "numeric_features":     self._numeric_cols,
            "categorical_features": self._cat_cols,
            "datetime_features":    self._datetime_cols,
            "nonlinear_features":   list(set(self._nonlinear_cols)),
            "collinear_pairs":      self._collinear_pairs,
            "outlier_columns":      list(set(self._outlier_cols)),
            "selected_features":    self._selected_features,
            "transformations":      self._transformations_applied,
            **self._report,
        }

    def get_clean_df(self, drop_originals: bool = False) -> pd.DataFrame:
        """
        Return the processed DataFrame with all engineered features.

        Parameters
        ----------
        drop_originals : bool
            If True, remove original columns that have been transformed/encoded.
        """
        df_out = self.df.copy()
        if drop_originals:
            to_drop = list(self._transformations_applied.keys())
            to_drop += [c for c in self._high_card_cols if f"{c}_te" in df_out.columns]
            df_out.drop(columns=[c for c in to_drop if c in df_out.columns], inplace=True)
        return df_out
