"""Experiment orchestration utilities for the option pricing engine.

The modules under :mod:`multi_option.experiments` aim to provide reproducible
training and evaluation pipelines that build on the core library components.
"""

from .pipeline import load_pipeline_config, run_pipeline
from .types import PipelineSpec, StageOutcome, StageSpec

__all__ = [
    "PipelineSpec",
    "StageOutcome",
    "StageSpec",
    "load_pipeline_config",
    "run_pipeline",
]
