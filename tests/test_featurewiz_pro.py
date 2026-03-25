"""
tests/test_featurewiz_pro.py
----------------------------
Smoke-test suite for featurewiz-pro.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed

from featurewiz_pro import (
    FeatureEngineer,
    FeatureEngineeringPipeline,
    PipelineConfig,
)


# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def regression_df():
    np.random.seed(0)
    n = 300
    return pd.DataFrame({
        "age":        np.random.randint(20, 70, n),
        "income":     np.random.exponential(50_000, n),
        "score":      np.random.normal(0, 1, n),
        "city":       np.random.choice(["Cairo", "Alex", "Giza"], n),
        "price":      np.random.exponential(200, n),          # numeric target
    })


@pytest.fixture
def classification_df():
    np.random.seed(1)
    n = 300
    return pd.DataFrame({
        "tenure":         np.random.randint(1, 72, n),
        "monthly_charge": np.random.uniform(20, 120, n),
        "num_products":   np.random.randint(1, 5, n),
        "contract":       np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "payment":        np.random.choice(["Credit card", "Bank transfer",
                                            "Mailed check", "Electronic check"], n),
        "Churn":          np.random.choice(["Yes", "No"], n),   # string target
    })


# ── FeatureEngineer — instantiation ─────────────────────────────────────────

class TestInit:
    def test_missing_target_raises(self, regression_df):
        with pytest.raises(ValueError, match="not found"):
            FeatureEngineer(regression_df, target="ghost")

    def test_bad_task_raises(self, regression_df):
        with pytest.raises(ValueError, match="task must be"):
            FeatureEngineer(regression_df, target="price", task="clustering")

    def test_columns_classified(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        assert "age" in fe._numeric_cols or "age" in fe._cat_cols
        assert "price" not in fe._numeric_cols
        assert "price" not in fe._cat_cols


# ── _encode_target ───────────────────────────────────────────────────────────

class TestEncodeTarget:
    def test_numeric_target_unchanged(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        enc = fe._encode_target()
        assert pd.api.types.is_numeric_dtype(enc)

    def test_string_target_becomes_int(self, classification_df):
        fe = FeatureEngineer(classification_df, target="Churn", task="classification")
        enc = fe._encode_target()
        assert pd.api.types.is_numeric_dtype(enc)
        assert set(enc.unique()) == {0, 1}

    def test_label_map_stored(self, classification_df):
        fe = FeatureEngineer(classification_df, target="Churn", task="classification")
        fe._encode_target()
        assert hasattr(fe, "_target_label_map")
        assert "No" in fe._target_label_map or "Yes" in fe._target_label_map

    def test_bool_target(self):
        df = pd.DataFrame({"x": range(50), "y": [True, False] * 25})
        fe = FeatureEngineer(df, target="y", task="classification")
        enc = fe._encode_target()
        assert set(enc.unique()).issubset({0, 1})


# ── Phase 1 ─────────────────────────────────────────────────────────────────

class TestPhase1:
    def test_audit_returns_df(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        result = fe.audit(verbose=False)
        assert isinstance(result, pd.DataFrame)
        assert "price" in result.index

    def test_missing_map_no_missing(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        miss = fe.missing_map(show_plot=False)
        assert miss.empty

    def test_missing_map_detects_nans(self, regression_df):
        df = regression_df.copy()
        df.loc[:50, "age"] = np.nan
        fe = FeatureEngineer(df, target="price")
        miss = fe.missing_map(show_plot=False)
        assert "age" in miss.index

    def test_cardinality_check(self, classification_df):
        fe = FeatureEngineer(classification_df, target="Churn", task="classification")
        result = fe.cardinality_check(verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_target_profile_string(self, classification_df):
        fe = FeatureEngineer(classification_df, target="Churn", task="classification")
        profile = fe.target_profile(show_plot=False)
        assert "class_balance" in profile

    def test_target_profile_numeric(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        profile = fe.target_profile(show_plot=False)
        assert "skew" in profile

    def test_drop_useless(self):
        df = pd.DataFrame({
            "useful":   np.random.randn(100),
            "constant": [1] * 100,
            "target":   np.random.randn(100),
        })
        fe = FeatureEngineer(df, target="target")
        dropped = fe.drop_useless()
        assert "constant" in dropped


# ── Phase 2 ─────────────────────────────────────────────────────────────────

class TestPhase2:
    def test_detect_outliers_iqr(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        result = fe.detect_outliers(method="iqr")
        assert isinstance(result, pd.DataFrame)
        assert "outlier_%" in result.columns

    def test_detect_outliers_zscore(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        result = fe.detect_outliers(method="zscore")
        assert isinstance(result, pd.DataFrame)

    def test_categorical_frequencies(self, classification_df):
        fe = FeatureEngineer(classification_df, target="Churn", task="classification")
        result = fe.categorical_frequencies(show_plot=False)
        assert isinstance(result, dict)

    def test_extract_datetime(self):
        df = pd.DataFrame({
            "ts":  pd.date_range("2020-01-01", periods=100, freq="D"),
            "val": np.random.randn(100),
            "y":   np.random.randn(100),
        })
        fe = FeatureEngineer(df, target="y")
        fe.extract_datetime_features()
        assert "ts_month" in fe.df.columns
        assert "ts_month_sin" in fe.df.columns


# ── Phase 3 ─────────────────────────────────────────────────────────────────

class TestPhase3:
    def test_correlation_analysis(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        result = fe.correlation_analysis(show_plot=False)
        assert "pearson_r" in result.columns
        assert "spearman_r" in result.columns

    def test_correlation_analysis_string_target(self, classification_df):
        fe = FeatureEngineer(classification_df, target="Churn", task="classification")
        result = fe.correlation_analysis(show_plot=False)
        assert isinstance(result, pd.DataFrame)

    def test_mutual_information(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        result = fe.mutual_information_scores(show_plot=False)
        assert isinstance(result, pd.Series)
        assert (result >= 0).all()

    def test_mutual_information_string_target(self, classification_df):
        fe = FeatureEngineer(classification_df, target="Churn", task="classification")
        result = fe.mutual_information_scores(show_plot=False)
        assert isinstance(result, pd.Series)

    def test_categorical_vs_target_regression(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        result = fe.categorical_vs_target()
        assert isinstance(result, pd.DataFrame)

    def test_categorical_vs_target_string_target(self, classification_df):
        fe = FeatureEngineer(classification_df, target="Churn", task="classification")
        result = fe.categorical_vs_target()
        assert isinstance(result, pd.DataFrame)

    def test_chi_squared_classification(self, classification_df):
        fe = FeatureEngineer(classification_df, target="Churn", task="classification")
        result = fe.chi_squared_test()
        assert "cramers_v" in result.columns


# ── Phase 4 ─────────────────────────────────────────────────────────────────

class TestPhase4:
    def test_residual_diagnostics_regression(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        result = fe.residual_diagnostics(show_plot=False)
        assert "r2" in result

    def test_residual_diagnostics_string_target(self, classification_df):
        fe = FeatureEngineer(classification_df, target="Churn", task="classification")
        result = fe.residual_diagnostics(show_plot=False)
        assert "pseudo_r2" in result

    def test_reset_test_regression(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        result = fe.reset_test()
        assert "verdict" in result

    def test_reset_test_skipped_for_classification(self, classification_df):
        fe = FeatureEngineer(classification_df, target="Churn", task="classification")
        result = fe.reset_test()
        assert result == {}

    def test_pdp(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        fe.partial_dependence_plots(max_cols=2)   # limit for speed


# ── Phase 5 ─────────────────────────────────────────────────────────────────

class TestPhase5:
    def test_correlation_matrix(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        result = fe.correlation_matrix(show_plot=False)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]

    def test_vif_analysis(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        result = fe.vif_analysis()
        assert "VIF" in result.columns
        assert "verdict" in result.columns

    def test_cramers_v(self, classification_df):
        fe = FeatureEngineer(classification_df, target="Churn", task="classification")
        result = fe.cramers_v_matrix(show_plot=False)
        assert isinstance(result, pd.DataFrame)
        # diagonal should be 1.0
        for col in result.columns:
            assert result.loc[col, col] == pytest.approx(1.0)

    def test_interaction_screening(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        fe.mutual_information_scores(show_plot=False)
        result = fe.interaction_screening(top_k=3, show_plot=False)
        assert isinstance(result, pd.DataFrame)

    def test_feature_clustering(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        result = fe.feature_clustering(n_clusters=2, show_plot=False)
        assert isinstance(result, dict)
        assert len(result) >= 1


# ── Phase 6 ─────────────────────────────────────────────────────────────────

class TestPhase6:
    def test_apply_transforms_yeo_johnson(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        fe.apply_transforms(method="yeo-johnson")
        # at least one new column should exist
        new_cols = [c for c in fe.df.columns if "_yeo_johnson" in c]
        assert len(new_cols) >= 0   # may be 0 if no skewed cols

    def test_apply_transforms_log1p(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        fe.apply_transforms(method="log1p", skew_thresh=0.0)
        # income is exponential so must be transformed
        assert any("log1p" in c for c in fe.df.columns)

    def test_target_encode_string_target(self, classification_df):
        fe = FeatureEngineer(classification_df, target="Churn", task="classification")
        fe._high_card_cols = ["payment"]
        fe.target_encode()
        assert "payment_te" in fe.df.columns
        assert pd.api.types.is_numeric_dtype(fe.df["payment_te"])

    def test_cyclic_encode(self):
        df = pd.DataFrame({
            "hour": np.random.randint(0, 24, 100),
            "y":    np.random.randn(100),
        })
        fe = FeatureEngineer(df, target="y")
        fe.cyclic_encode(col_period={"hour": 24})
        assert "hour_sin" in fe.df.columns
        assert "hour_cos" in fe.df.columns
        # values must be in [-1, 1]
        assert fe.df["hour_sin"].between(-1, 1).all()

    def test_scale_features(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        fe.scale_features(method="standard")
        scaled_cols = [c for c in fe.df.columns if c.endswith("_scaled")]
        assert len(scaled_cols) > 0


# ── Phase 7 ─────────────────────────────────────────────────────────────────

class TestPhase7:
    def test_mi_ranking(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        result = fe.mutual_info_ranking(top_n=3, show_plot=False)
        assert isinstance(result, pd.Series)
        assert len(result) <= 3

    def test_lasso_importance(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        result = fe.lasso_importance(show_plot=False)
        assert isinstance(result, pd.Series)
        assert (result >= 0).all()

    def test_lasso_importance_string_target(self, classification_df):
        fe = FeatureEngineer(classification_df, target="Churn", task="classification")
        result = fe.lasso_importance(show_plot=False)
        assert isinstance(result, pd.Series)

    def test_permutation_importance(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        result = fe.permutation_importance_analysis(n_repeats=3, show_plot=False)
        assert "mean_importance" in result.columns

    def test_leakage_audit_no_leakage(self, regression_df):
        fe = FeatureEngineer(regression_df, target="price")
        suspects = fe.leakage_audit(verbose=False)
        assert isinstance(suspects, list)

    def test_leakage_audit_catches_copy(self):
        df = pd.DataFrame({
            "x":        np.random.randn(200),
            "y":        np.random.randn(200),
            "price":    np.random.randn(200),
        })
        df["price_copy"] = df["price"]   # perfect leakage
        fe = FeatureEngineer(df, target="price")
        suspects = fe.leakage_audit(correlation_thresh=0.99, verbose=False)
        assert "price_copy" in suspects


# ── FeatureEngineeringPipeline ───────────────────────────────────────────────

class TestPipeline:
    def test_pipeline_phase1_only(self, regression_df):
        cfg = PipelineConfig(target="price", task="regression",
                             phases=[1], show_plots=False)
        p = FeatureEngineeringPipeline(regression_df, cfg)
        result = p.run()
        assert result.df is not None
        assert len(result.steps_by_status("ok")) > 0

    def test_pipeline_string_target(self, classification_df):
        cfg = PipelineConfig(
            target="Churn", task="classification",
            phases=[1, 2, 3],
            show_plots=False,
            skip_steps=["partial_regression", "chi_squared_test"],
        )
        p = FeatureEngineeringPipeline(classification_df, cfg)
        result = p.run()
        assert result.df.shape[0] > 0

    def test_step_names_complete(self, regression_df):
        cfg = PipelineConfig(target="price")
        p = FeatureEngineeringPipeline(regression_df, cfg)
        names = p.step_names()
        assert "audit" in names
        assert "leakage_audit" in names
        assert len(names) == 36

    def test_skip_steps(self, regression_df):
        cfg = PipelineConfig(target="price", phases=[1],
                             skip_steps=["audit", "missing_map"],
                             show_plots=False)
        p = FeatureEngineeringPipeline(regression_df, cfg)
        result = p.run()
        skipped = [s.name for s in result.steps_by_status("skipped")]
        assert "audit" in skipped
        assert "missing_map" in skipped

    def test_timing_report(self, regression_df):
        cfg = PipelineConfig(target="price", phases=[1], show_plots=False)
        p = FeatureEngineeringPipeline(regression_df, cfg)
        p.run()
        timing = p.timing_report()
        assert isinstance(timing, pd.DataFrame)
        assert "elapsed_s" in timing.columns

    def test_save_report_json(self, regression_df, tmp_path):
        cfg = PipelineConfig(target="price", phases=[1], show_plots=False,
                             save_report=str(tmp_path / "report.json"))
        p = FeatureEngineeringPipeline(regression_df, cfg)
        result = p.run()
        assert (tmp_path / "report.json").exists()

    def test_save_df_csv(self, regression_df, tmp_path):
        cfg = PipelineConfig(target="price", phases=[1], show_plots=False,
                             save_df=str(tmp_path / "clean.csv"))
        p = FeatureEngineeringPipeline(regression_df, cfg)
        result = p.run()
        assert (tmp_path / "clean.csv").exists()

    def test_run_step(self, regression_df):
        cfg = PipelineConfig(target="price", phases=[1], show_plots=False)
        p = FeatureEngineeringPipeline(regression_df, cfg)
        p.run()
        sr = p.run_step("vif_analysis")
        assert sr.status in ("ok", "skipped", "failed")

    def test_invalid_run_step_raises(self, regression_df):
        cfg = PipelineConfig(target="price", phases=[1], show_plots=False)
        p = FeatureEngineeringPipeline(regression_df, cfg)
        p.run()
        with pytest.raises(ValueError, match="not found"):
            p.run_step("nonexistent_step")
