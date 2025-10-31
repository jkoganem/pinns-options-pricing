"""Registry of pipeline stage handlers."""

from __future__ import annotations

from typing import Callable

from multi_option.experiments.types import PipelineContext, StageOutcome, StageSpec

from .tests import run_stage as run_tests_stage
from .european import run_stage as run_european_stage
from .pinn import run_stage as run_pinn_stage

StageHandler = Callable[[StageSpec, PipelineContext, int], StageOutcome]

STAGE_HANDLERS: dict[str, StageHandler] = {
    "tests": run_tests_stage,
    "european_benchmark": run_european_stage,
    "pinn_training": run_pinn_stage,
}

__all__ = ["STAGE_HANDLERS"]
