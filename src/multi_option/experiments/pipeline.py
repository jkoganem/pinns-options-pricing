"""Stage-based experiment pipeline orchestration."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np

try:  # Torch may be unavailable in some environments when only planning.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from multi_option.experiments.stages import STAGE_HANDLERS
from multi_option.experiments.types import (
    PipelineContext,
    PipelineSpec,
    StageOutcome,
    StageSpec,
)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_pipeline_config(config_path: Path) -> PipelineSpec:
    """Load a pipeline configuration from TOML."""
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    pipeline_section = data.get("pipeline", {})
    seed = int(pipeline_section.get("seed", 42))
    output_root = pipeline_section.get("output_root", "runs/staged_pipeline")
    fail_fast = bool(pipeline_section.get("fail_fast", True))
    python_executable = pipeline_section.get("python_executable")

    stage_entries: Iterable[dict] = data.get("stages", [])
    stages: List[StageSpec] = []
    for entry in stage_entries:
        name = entry.get("name")
        stage_type = entry.get("type")
        if not name or not stage_type:
            raise ValueError("Each stage must define both 'name' and 'type'")
        config = {k: v for k, v in entry.items() if k not in {"name", "type"}}
        stages.append(StageSpec(name=name, type=stage_type, config=config))

    return PipelineSpec(
        seed=seed,
        output_root=Path(output_root),
        fail_fast=fail_fast,
        python_executable=python_executable,
        stages=stages,
    )


def run_pipeline(spec: PipelineSpec, workspace: Path | None = None) -> List[StageOutcome]:
    """Execute all stages defined in the pipeline specification."""
    workspace = Path(workspace) if workspace is not None else Path.cwd()
    python_executable = spec.python_executable or sys.executable

    output_root = spec.output_root
    if not output_root.is_absolute():
        output_root = workspace / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    context = PipelineContext(workspace=workspace, output_root=output_root, python=python_executable)

    outcomes: List[StageOutcome] = []

    for index, stage in enumerate(spec.stages):
        handler = STAGE_HANDLERS.get(stage.type)
        if handler is None:
            raise ValueError(f"Unknown stage type '{stage.type}'")

        stage_seed = spec.seed + index
        _set_global_seed(stage_seed)

        try:
            outcome = handler(stage, context, stage_seed)
        except Exception as exc:  # pragma: no cover - defensive path
            outcome = StageOutcome(
                name=stage.name,
                status="failed",
                details={"error": str(exc)},
                artifact_dir=context.output_root / stage.name,
            )
            if spec.fail_fast:
                raise

        outcome_dir = outcome.artifact_dir
        outcome_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "name": outcome.name,
            "status": outcome.status,
            "details": outcome.details,
        }
        (outcome_dir / "stage_outcome.json").write_text(
            json.dumps(summary_payload, indent=2),
            encoding="utf-8",
        )
        outcomes.append(outcome)

    pipeline_summary = {
        "seed": spec.seed,
        "fail_fast": spec.fail_fast,
        "stages": [
            {"name": outcome.name, "status": outcome.status, "details": outcome.details}
            for outcome in outcomes
        ],
    }
    (context.output_root / "pipeline_summary.json").write_text(
        json.dumps(pipeline_summary, indent=2),
        encoding="utf-8",
    )

    return outcomes


__all__ = ["load_pipeline_config", "run_pipeline"]
