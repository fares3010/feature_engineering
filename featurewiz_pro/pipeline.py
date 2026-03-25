"""
FeatureEngineeringPipeline
==========================
Master orchestrator for the full 7-phase feature engineering workflow.
Wraps FeatureEngineer with:

  • PipelineConfig  — single dataclass controlling every knob
  • Step-level execution with individual try/except (one failure never
    kills the whole run)
  • Per-step and per-phase wall-clock timing
  • Rich console progress with ✓ / ✗ / ⚠ status per step
  • Structured PipelineResult with findings, recommendations, and timings
  • JSON report export
  • Cleaned DataFrame export to CSV / parquet

Quickstart
----------
  from feature_pipeline import FeatureEngineeringPipeline, PipelineConfig

  cfg = PipelineConfig(
      target      = "price",
      task        = "regression",
      skip_steps  = ["shap_dependence", "gam_smooth_terms"],  # optional heavy steps
      save_report = "report.json",
      save_df     = "clean_data.csv",
  )
  pipeline = FeatureEngineeringPipeline(df, config=cfg)
  result   = pipeline.run()

  result.summary()            # print human-readable findings
  clean_df = result.df        # fully engineered DataFrame
  report   = result.report    # dict of all findings
"""

from __future__ import annotations

import json
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from featurewiz_pro.core import FeatureEngineer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Terminal colour helpers (no external deps)
# ─────────────────────────────────────────────────────────────────────────────

_C = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "blue":   "\033[94m",
    "white":  "\033[97m",
    "dim":    "\033[2m",
}

def _c(text: str, *codes: str) -> str:
    return "".join(_C[c] for c in codes) + str(text) + _C["reset"]

def _banner(text: str) -> None:
    width = 72
    pad   = max(0, (width - len(text) - 2) // 2)
    print()
    print(_c("╔" + "═" * width + "╗", "cyan", "bold"))
    print(_c("║" + " " * pad + f" {text} " + " " * (width - pad - len(text) - 1) + "║", "cyan", "bold"))
    print(_c("╚" + "═" * width + "╝", "cyan", "bold"))

def _phase_header(phase: int, title: str) -> None:
    icons = {1: "🔍", 2: "📊", 3: "🔗", 4: "〰", 5: "🕸", 6: "⚙", 7: "✂"}
    icon  = icons.get(phase, "▶")
    bar   = "─" * 68
    print(f"\n{_c(bar, 'blue')}")
    print(f"  {_c(f'Phase {phase}', 'bold', 'blue')}  {icon}  {_c(title, 'white', 'bold')}")
    print(f"{_c(bar, 'blue')}")

def _step_ok(name: str, elapsed: float) -> None:
    print(f"  {_c('✓', 'green', 'bold')}  {name:<42} {_c(f'{elapsed:.2f}s', 'dim')}")

def _step_skip(name: str, reason: str) -> None:
    print(f"  {_c('⊘', 'yellow')}  {name:<42} {_c(reason, 'dim')}")

def _step_fail(name: str, elapsed: float, exc: Exception) -> None:
    print(f"  {_c('✗', 'red', 'bold')}  {name:<42} {_c(f'{elapsed:.2f}s', 'dim')}")
    print(f"     {_c(str(exc)[:80], 'red')}")


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    Single source of truth for every pipeline knob.

    Core
    ----
    target        : str   — target column name (required)
    task          : str   — 'regression' or 'classification'
    phases        : list  — which phases to run (default: all 1–7)
    skip_steps    : list  — individual step names to skip (see STEP_REGISTRY)

    Phase 1 — Audit
    ---------------
    missing_thresh   : float  — drop cols with > x% missing (default 95)
    variance_thresh  : float  — drop cols with variance ≤ x (default 0)
    drop_duplicates  : bool

    Phase 2 — Univariate
    --------------------
    outlier_method   : str   — 'iqr' or 'zscore'
    z_thresh         : float — Z-score threshold (default 3.0)
    max_plot_cols    : int   — cap on distribution plots (default 20)

    Phase 3 — Bivariate
    --------------------
    lowess           : bool  — add LOWESS smoother to scatter plots

    Phase 4 — Linearity
    --------------------
    pdp_max_cols     : int   — max features in PDP (default 6)

    Phase 5 — Collinearity
    ----------------------
    corr_threshold   : float — |r| above which pair is flagged (default 0.85)
    vif_threshold    : float — VIF threshold for flagging (default 10)
    interaction_top_k: int   — top-K features screened for interactions

    Phase 6 — Transforms
    ---------------------
    transform_method : str   — 'yeo-johnson' | 'box-cox' | 'log1p' | 'sqrt'
    skew_thresh      : float — only transform if |skew| > x (default 0.75)
    poly_degree      : int   — degree for polynomial / spline (default 2)
    use_spline       : bool  — True = splines, False = poly features
    n_knots          : int   — spline knots (default 5)
    scale_method     : str   — 'standard' | 'robust' | 'minmax' | None
    target_encode_smoothing : int

    Phase 7 — Selection
    --------------------
    mi_top_n         : int   — top-N features for MI ranking display
    lasso_alpha      : float — L1 penalty strength (default 0.01)
    perm_n_repeats   : int   — permutation importance repeats (default 10)
    min_features     : int   — minimum features for RFECV (default 5)
    leakage_corr_thresh : float

    Output
    ------
    save_report      : str | None — path to write JSON report (None = skip)
    save_df          : str | None — path to write clean CSV/parquet (None = skip)
    show_plots       : bool       — set False to suppress all plots (CI / batch)
    verbose          : bool
    """

    # core
    target:           str   = "target"
    task:             str   = "regression"
    phases:           List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    skip_steps:       List[str] = field(default_factory=list)

    # phase 1
    missing_thresh:   float = 95.0
    variance_thresh:  float = 0.0
    drop_duplicates:  bool  = True

    # phase 2
    outlier_method:   str   = "iqr"
    z_thresh:         float = 3.0
    max_plot_cols:    int   = 20

    # phase 3
    lowess:           bool  = True

    # phase 4
    pdp_max_cols:     int   = 6

    # phase 5
    corr_threshold:   float = 0.85
    vif_threshold:    float = 10.0
    interaction_top_k:int   = 5

    # phase 6
    transform_method: str   = "yeo-johnson"
    skew_thresh:      float = 0.75
    poly_degree:      int   = 2
    use_spline:       bool  = True
    n_knots:          int   = 5
    scale_method:     Optional[str] = "standard"
    target_encode_smoothing: int = 10

    # phase 7
    mi_top_n:         int   = 20
    lasso_alpha:      float = 0.01
    perm_n_repeats:   int   = 10
    min_features:     int   = 5
    leakage_corr_thresh: float = 0.99

    # output
    save_report:      Optional[str] = None
    save_df:          Optional[str] = None
    show_plots:       bool  = True
    verbose:          bool  = True


# ─────────────────────────────────────────────────────────────────────────────
#  Step result & pipeline result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    phase:    int
    name:     str
    status:   str          # 'ok' | 'skipped' | 'failed'
    elapsed:  float = 0.0
    output:   Any   = None
    error:    str   = ""


@dataclass
class PipelineResult:
    df:           pd.DataFrame
    report:       Dict[str, Any]
    step_results: List[StepResult]
    elapsed_total:float
    config:       PipelineConfig

    # ── convenience accessors ─────────────────────────────────────────────────
    @property
    def selected_features(self) -> List[str]:
        return self.report.get("selected_features", [])

    @property
    def nonlinear_features(self) -> List[str]:
        return self.report.get("nonlinear_features", [])

    @property
    def collinear_pairs(self) -> list:
        return self.report.get("collinear_pairs", [])

    @property
    def leakage_suspects(self) -> List[str]:
        return self.report.get("leakage_suspicious", [])

    @property
    def transformations(self) -> dict:
        return self.report.get("transformations", {})

    def steps_by_status(self, status: str) -> List[StepResult]:
        return [s for s in self.step_results if s.status == status]

    # ── summary ───────────────────────────────────────────────────────────────
    def summary(self) -> None:
        ok      = self.steps_by_status("ok")
        failed  = self.steps_by_status("failed")
        skipped = self.steps_by_status("skipped")

        _banner("Pipeline Result Summary")

        print(f"\n  {_c('Run time', 'bold')}   : {self.elapsed_total:.1f}s")
        print(f"  {_c('Steps', 'bold')}       : "
              f"{_c(f'{len(ok)} ok', 'green')}  "
              f"{_c(f'{len(failed)} failed', 'red')}  "
              f"{_c(f'{len(skipped)} skipped', 'yellow')}")
        print(f"  {_c('Output shape', 'bold')}: {self.df.shape[0]:,} rows × "
              f"{self.df.shape[1]} cols")

        # key findings
        print(f"\n  {_c('─── Key findings ───────────────────────────────────────', 'dim')}")

        nl = self.nonlinear_features
        if nl:
            print(f"  {_c('Non-linear features', 'yellow')}   : {nl}")

        cp = self.collinear_pairs
        if cp:
            print(f"  {_c('Collinear pairs', 'yellow')}       : "
                  f"{[(a, b) for a, b, _ in cp]}")

        sf = self.selected_features
        if sf:
            print(f"  {_c('Selected features', 'green')}     : {sf}")

        ls = self.leakage_suspects
        if ls:
            print(f"  {_c('⚠  Leakage suspects', 'red', 'bold')}  : {ls}")

        tr = self.transformations
        if tr:
            print(f"  {_c('Transforms applied', 'cyan')}    : {tr}")

        # failed steps
        if failed:
            print(f"\n  {_c('─── Failed steps ────────────────────────────────────────', 'dim')}")
            for s in failed:
                print(f"  {_c('✗', 'red')} Phase {s.phase} · {s.name}")
                print(f"    {_c(s.error[:100], 'dim')}")

        # auto-recommendations
        recs = self._recommendations()
        if recs:
            print(f"\n  {_c('─── Recommendations ─────────────────────────────────────', 'dim')}")
            for i, r in enumerate(recs, 1):
                print(f"  {_c(str(i) + '.', 'cyan')} {r}")

        print()

    def _recommendations(self) -> List[str]:
        recs = []
        nl = self.nonlinear_features
        if nl:
            recs.append(
                f"Apply spline/polynomial features to non-linear cols: {nl[:4]}"
            )
        cp = self.collinear_pairs
        if cp:
            recs.append(
                f"Drop or merge one column from each collinear pair: "
                f"{[(a, b) for a, b, _ in cp[:3]]}"
            )
        ls = self.leakage_suspects
        if ls:
            recs.append(f"Investigate potential leakage in: {ls}")
        mi = self.report.get("mutual_info")
        if mi is not None:
            zero_mi = [f for f, v in mi.items() if v == 0]
            if zero_mi:
                recs.append(f"Zero MI features (consider dropping): {zero_mi[:5]}")
        vif = self.report.get("vif")
        if vif is not None:
            severe = vif[vif["VIF"] > 10].index.tolist() if "VIF" in vif.columns else []
            if severe:
                recs.append(f"VIF > 10 — severe multicollinearity: {severe[:4]}")
        return recs

    # ── export helpers ────────────────────────────────────────────────────────
    def save_report_json(self, path: str) -> None:
        """Serialise the findings report to JSON (numpy types auto-converted)."""
        def _json_safe(obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            if isinstance(obj, pd.DataFrame):   return obj.to_dict()
            if isinstance(obj, pd.Series):      return obj.to_dict()
            return str(obj)

        with open(path, "w") as fh:
            json.dump(self.report, fh, default=_json_safe, indent=2)
        print(f"  Report saved → {path}")

    def save_dataframe(self, path: str, drop_originals: bool = False) -> None:
        """Save the engineered DataFrame to CSV or Parquet (auto-detected by extension)."""
        df_out = self.df.copy()
        if drop_originals:
            tr = list(self.transformations.keys())
            df_out.drop(columns=[c for c in tr if c in df_out.columns], inplace=True)
        if path.endswith(".parquet"):
            df_out.to_parquet(path, index=False)
        else:
            df_out.to_csv(path, index=False)
        print(f"  DataFrame saved → {path}  ({df_out.shape[0]:,} rows × {df_out.shape[1]} cols)")


# ─────────────────────────────────────────────────────────────────────────────
#  Master Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineeringPipeline:
    """
    Master orchestrator for the full 7-phase feature engineering workflow.

    Usage
    -----
    pipeline = FeatureEngineeringPipeline(df, PipelineConfig(target="price"))
    result   = pipeline.run()
    result.summary()
    """

    # Registry: phase → ordered list of (step_name, callable_attr, condition_fn)
    # condition_fn(fe, cfg) → bool: True = run, False = skip
    STEP_REGISTRY: Dict[int, List[tuple]] = {
        1: [
            ("audit",             "audit",             lambda fe, cfg: True),
            ("missing_map",       "missing_map",       lambda fe, cfg: True),
            ("cardinality_check", "cardinality_check", lambda fe, cfg: bool(fe._cat_cols)),
            ("target_profile",    "target_profile",    lambda fe, cfg: True),
            ("drop_useless",      "drop_useless",      lambda fe, cfg: True),
        ],
        2: [
            ("distribution_plots",      "distribution_plots",      lambda fe, cfg: bool(fe._numeric_cols)),
            ("detect_outliers",         "detect_outliers",         lambda fe, cfg: bool(fe._numeric_cols)),
            ("categorical_frequencies", "categorical_frequencies", lambda fe, cfg: bool(fe._cat_cols)),
            ("extract_datetime",        "extract_datetime_features", lambda fe, cfg: bool(fe._datetime_cols)),
            ("extract_text",            "extract_text_features",   lambda fe, cfg: True),
        ],
        3: [
            ("bivariate_scatter",      "bivariate_scatter",      lambda fe, cfg: bool(fe._numeric_cols)),
            ("correlation_analysis",   "correlation_analysis",   lambda fe, cfg: bool(fe._numeric_cols)),
            ("mutual_info_scores",     "mutual_information_scores", lambda fe, cfg: bool(fe._numeric_cols)),
            ("categorical_vs_target",  "categorical_vs_target",  lambda fe, cfg: bool(fe._cat_cols)),
            ("chi_squared_test",       "chi_squared_test",       lambda fe, cfg: cfg.task == "classification" and bool(fe._cat_cols)),
            ("partial_regression",     "partial_regression_plots", lambda fe, cfg: cfg.task == "regression" and len(fe._numeric_cols) >= 2),
        ],
        4: [
            ("residual_diagnostics",   "residual_diagnostics",   lambda fe, cfg: bool(fe._numeric_cols)),
            ("reset_test",             "reset_test",             lambda fe, cfg: cfg.task == "regression" and bool(fe._numeric_cols)),
            ("pdp",                    "partial_dependence_plots", lambda fe, cfg: bool(fe._numeric_cols)),
            ("shap_dependence",        "shap_dependence",        lambda fe, cfg: bool(fe._numeric_cols)),
            ("gam_smooth_terms",       "gam_smooth_terms",       lambda fe, cfg: bool(fe._numeric_cols)),
        ],
        5: [
            ("correlation_matrix",  "correlation_matrix",  lambda fe, cfg: len(fe._numeric_cols) >= 2),
            ("vif_analysis",        "vif_analysis",        lambda fe, cfg: len(fe._numeric_cols) >= 2),
            ("cramers_v_matrix",    "cramers_v_matrix",    lambda fe, cfg: len(fe._cat_cols) >= 2),
            ("interaction_screen",  "interaction_screening", lambda fe, cfg: len(fe._numeric_cols) >= 2),
            ("feature_clustering",  "feature_clustering",  lambda fe, cfg: len(fe._numeric_cols) >= 3),
        ],
        6: [
            ("apply_transforms",   "apply_transforms",   lambda fe, cfg: bool(fe._numeric_cols)),
            ("poly_spline",        "polynomial_spline_features", lambda fe, cfg: bool(fe._numeric_cols)),
            ("target_encode",      "target_encode",      lambda fe, cfg: bool(fe._high_card_cols)),
            ("cyclic_encode",      "cyclic_encode",      lambda fe, cfg: True),
            ("scale_features",     "scale_features",     lambda fe, cfg: cfg.scale_method is not None and bool(fe._numeric_cols)),
        ],
        7: [
            ("mi_ranking",         "mutual_info_ranking",          lambda fe, cfg: bool(fe._numeric_cols)),
            ("rfecv",              "rfecv_selection",              lambda fe, cfg: len(fe._numeric_cols) >= cfg.min_features),
            ("lasso_importance",   "lasso_importance",             lambda fe, cfg: bool(fe._numeric_cols)),
            ("perm_importance",    "permutation_importance_analysis", lambda fe, cfg: bool(fe._numeric_cols)),
            ("leakage_audit",      "leakage_audit",                lambda fe, cfg: True),
        ],
    }

    PHASE_TITLES = {
        1: "Data profiling & initial audit",
        2: "Univariate feature analysis",
        3: "Bivariate: feature → target relationships",
        4: "Linearity vs non-linearity diagnosis",
        5: "Multicollinearity & interaction screening",
        6: "Transformations & encoding",
        7: "Feature selection & validation",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.df     = df.copy()
        self.cfg    = config or PipelineConfig()
        self._fe: Optional[FeatureEngineer] = None
        self._step_results: List[StepResult] = []

    # ── argument builders — map config → method kwargs ─────────────────────

    def _step_kwargs(self, step_name: str) -> dict:
        """Return the correct keyword arguments for each step method."""
        cfg = self.cfg
        sp  = cfg.show_plots
        return {
            # phase 1
            "audit":                  dict(verbose=cfg.verbose),
            "missing_map":            dict(show_plot=sp),
            "cardinality_check":      dict(verbose=cfg.verbose),
            "target_profile":         dict(show_plot=sp),
            "drop_useless":           dict(missing_thresh=cfg.missing_thresh,
                                           variance_thresh=cfg.variance_thresh,
                                           drop_duplicates=cfg.drop_duplicates),
            # phase 2
            "distribution_plots":     dict(max_cols=cfg.max_plot_cols),
            "detect_outliers":        dict(method=cfg.outlier_method,
                                           z_thresh=cfg.z_thresh),
            "categorical_frequencies":dict(show_plot=sp),
            "extract_datetime_features": dict(),
            "extract_text_features":  dict(),
            # phase 3
            "bivariate_scatter":      dict(lowess=cfg.lowess),
            "correlation_analysis":   dict(show_plot=sp),
            "mutual_information_scores": dict(show_plot=sp),
            "categorical_vs_target":  dict(),
            "chi_squared_test":       dict(),
            "partial_regression_plots": dict(),
            # phase 4
            "residual_diagnostics":   dict(show_plot=sp),
            "reset_test":             dict(),
            "partial_dependence_plots": dict(max_cols=cfg.pdp_max_cols),
            "shap_dependence":        dict(max_cols=cfg.pdp_max_cols),
            "gam_smooth_terms":       dict(),
            # phase 5
            "correlation_matrix":     dict(threshold=cfg.corr_threshold, show_plot=sp),
            "vif_analysis":           dict(threshold=cfg.vif_threshold),
            "cramers_v_matrix":       dict(show_plot=sp),
            "interaction_screening":  dict(top_k=cfg.interaction_top_k, show_plot=sp),
            "feature_clustering":     dict(show_plot=sp),
            # phase 6
            "apply_transforms":       dict(method=cfg.transform_method,
                                           skew_thresh=cfg.skew_thresh),
            "polynomial_spline_features": dict(degree=cfg.poly_degree,
                                               use_spline=cfg.use_spline,
                                               n_knots=cfg.n_knots),
            "target_encode":          dict(smoothing=cfg.target_encode_smoothing),
            "cyclic_encode":          dict(),
            "scale_features":         dict(method=cfg.scale_method or "standard"),
            # phase 7
            "mutual_info_ranking":    dict(top_n=cfg.mi_top_n, show_plot=sp),
            "rfecv_selection":        dict(min_features=cfg.min_features, show_plot=sp),
            "lasso_importance":       dict(alpha=cfg.lasso_alpha, show_plot=sp),
            "permutation_importance_analysis": dict(n_repeats=cfg.perm_n_repeats, show_plot=sp),
            "leakage_audit":          dict(correlation_thresh=cfg.leakage_corr_thresh),
        }.get(step_name, {})

    # ── core executor ─────────────────────────────────────────────────────────

    def _run_step(
        self,
        phase:     int,
        step_name: str,
        attr_name: str,
        condition: Any,
    ) -> StepResult:
        """
        Execute a single pipeline step with full error isolation.

        Returns a StepResult regardless of outcome.
        """
        cfg = self.cfg

        # user-requested skip
        if step_name in cfg.skip_steps:
            _step_skip(step_name, "user skip_steps")
            return StepResult(phase=phase, name=step_name, status="skipped",
                              error="user-requested")

        # condition check
        try:
            should_run = condition(self._fe, cfg)
        except Exception:
            should_run = False

        if not should_run:
            _step_skip(step_name, "condition not met")
            return StepResult(phase=phase, name=step_name, status="skipped",
                              error="condition not met")

        # execute
        method   = getattr(self._fe, attr_name)
        kwargs   = self._step_kwargs(attr_name)
        t0       = time.perf_counter()
        output   = None
        try:
            output  = method(**kwargs)
            elapsed = time.perf_counter() - t0
            _step_ok(step_name, elapsed)
            return StepResult(phase=phase, name=step_name, status="ok",
                              elapsed=elapsed, output=output)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            _step_fail(step_name, elapsed, exc)
            if cfg.verbose:
                traceback.print_exc()
            return StepResult(phase=phase, name=step_name, status="failed",
                              elapsed=elapsed, error=str(exc))

    # ── public API ────────────────────────────────────────────────────────────

    def run(self) -> PipelineResult:
        """
        Execute the full configured pipeline.

        Returns a PipelineResult with the engineered DataFrame, full report,
        per-step results, and timing.
        """
        cfg = self.cfg
        t_total = time.perf_counter()

        _banner(f"Feature Engineering Pipeline  ·  {cfg.task.upper()}")
        print(f"  target={_c(cfg.target, 'cyan')}   "
              f"phases={_c(str(cfg.phases), 'cyan')}   "
              f"skip={_c(str(cfg.skip_steps or '—'), 'yellow')}\n")

        # initialise FeatureEngineer
        self._fe = FeatureEngineer(
            self.df,
            target=cfg.target,
            task=cfg.task,
        )

        # phase loop
        for phase in cfg.phases:
            steps = self.STEP_REGISTRY.get(phase, [])
            _phase_header(phase, self.PHASE_TITLES[phase])
            t_phase = time.perf_counter()

            for step_name, attr_name, condition in steps:
                sr = self._run_step(phase, step_name, attr_name, condition)
                self._step_results.append(sr)

            phase_elapsed = time.perf_counter() - t_phase
            print(f"\n  {_c(f'Phase {phase} done', 'dim')}  "
                  f"{_c(f'{phase_elapsed:.1f}s', 'dim')}")

        # collect outputs
        elapsed_total = time.perf_counter() - t_total
        report        = self._fe.report()
        df_out        = self._fe.get_clean_df()

        result = PipelineResult(
            df=df_out,
            report=report,
            step_results=self._step_results,
            elapsed_total=elapsed_total,
            config=cfg,
        )

        # optional exports
        if cfg.save_report:
            result.save_report_json(cfg.save_report)
        if cfg.save_df:
            result.save_dataframe(cfg.save_df)

        result.summary()
        return result

    # ── incremental / diagnostic helpers ────────────────────────────────────

    def run_phases(self, phases: List[int]) -> "FeatureEngineeringPipeline":
        """
        Run a subset of phases in-place (mutates internal FeatureEngineer state).
        Allows incremental execution:

            pipeline.run_phases([1, 2])
            pipeline.run_phases([3, 4])
        """
        if self._fe is None:
            self._fe = FeatureEngineer(self.df, target=self.cfg.target, task=self.cfg.task)

        for phase in phases:
            steps = self.STEP_REGISTRY.get(phase, [])
            _phase_header(phase, self.PHASE_TITLES[phase])
            for step_name, attr_name, condition in steps:
                sr = self._run_step(phase, step_name, attr_name, condition)
                self._step_results.append(sr)

        return self

    def run_step(self, step_name: str) -> StepResult:
        """
        Execute a single named step on demand.

        Useful for re-running a failed step after fixing data:
            pipeline.run_step("rfecv")
        """
        if self._fe is None:
            raise RuntimeError("Call run() or run_phases() before run_step().")

        for phase, steps in self.STEP_REGISTRY.items():
            for sname, attr_name, condition in steps:
                if sname == step_name:
                    return self._run_step(phase, step_name, attr_name, condition)

        raise ValueError(
            f"Step '{step_name}' not found. Available: "
            + ", ".join(s for steps in self.STEP_REGISTRY.values() for s, _, _ in steps)
        )

    def step_names(self) -> List[str]:
        """Return every registered step name for reference."""
        return [s for steps in self.STEP_REGISTRY.values() for s, _, _ in steps]

    def timing_report(self) -> pd.DataFrame:
        """Return a DataFrame of per-step timings, sorted by elapsed time."""
        rows = [
            {"phase": s.phase, "step": s.name,
             "status": s.status, "elapsed_s": round(s.elapsed, 3)}
            for s in self._step_results
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("elapsed_s", ascending=False).reset_index(drop=True)
        return df

    def failed_steps(self) -> List[StepResult]:
        """Return all steps that raised exceptions."""
        return [s for s in self._step_results if s.status == "failed"]

    def retry_failed(self) -> List[StepResult]:
        """Re-run every failed step and return updated StepResults."""
        failed = self.failed_steps()
        if not failed:
            print("  No failed steps to retry.")
            return []

        new_results = []
        print(f"\n  Retrying {len(failed)} failed step(s)…")
        for s in failed:
            # update the original entry
            self._step_results = [r for r in self._step_results if r.name != s.name]
            new_sr = self.run_step(s.name)
            self._step_results.append(new_sr)
            new_results.append(new_sr)

        return new_results
