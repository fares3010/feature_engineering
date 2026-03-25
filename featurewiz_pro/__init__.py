"""
featurewiz-pro
==============
A comprehensive 7-phase feature engineering toolkit for tabular data.

Handles regression and classification tasks with string, numeric, boolean,
and datetime targets. All 35 analytical methods and a master pipeline
orchestrator included.

Quick start
-----------
>>> import pandas as pd
>>> from featurewiz_pro import FeatureEngineer, FeatureEngineeringPipeline, PipelineConfig
>>>
>>> fe = FeatureEngineer(df, target="price", task="regression")
>>> fe.run_all()
>>>
>>> # Or use the master pipeline with full config control
>>> cfg = PipelineConfig(target="Churn", task="classification",
...                      skip_steps=["shap_dependence", "gam_smooth_terms"],
...                      save_report="report.json")
>>> result = FeatureEngineeringPipeline(df, cfg).run()
>>> result.summary()
"""

from featurewiz_pro.core import FeatureEngineer
from featurewiz_pro.pipeline import (
    FeatureEngineeringPipeline,
    PipelineConfig,
    PipelineResult,
    StepResult,
)

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "you@example.com"
__license__ = "MIT"

__all__ = [
    "FeatureEngineer",
    "FeatureEngineeringPipeline",
    "PipelineConfig",
    "PipelineResult",
    "StepResult",
]
