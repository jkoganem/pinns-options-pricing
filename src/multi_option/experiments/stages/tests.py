"""Test execution stage."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Sequence

from multi_option.experiments.types import PipelineContext, StageOutcome, StageSpec


def run_stage(stage: StageSpec, context: PipelineContext, stage_seed: int) -> StageOutcome:
    """Execute pytest for the repository."""
    stage_dir = context.output_root / stage.name
    stage_dir.mkdir(parents=True, exist_ok=True)

    pytest_args: Sequence[str] = stage.config.get("pytest_args", ["-q"])
    env = os.environ.copy()
    pythonpath_entries = [str(context.workspace / "src")]
    if existing := env.get("PYTHONPATH"):
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["PYTHONHASHSEED"] = str(stage_seed)

    cmd = [context.python, "-m", "pytest", *pytest_args]
    result = subprocess.run(
        cmd,
        cwd=context.workspace,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    (stage_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (stage_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")

    details = {
        "command": cmd,
        "returncode": result.returncode,
        "stdout_log": str(stage_dir / "stdout.log"),
        "stderr_log": str(stage_dir / "stderr.log"),
    }
    (stage_dir / "stage_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")

    status = "success" if result.returncode == 0 else "failed"
    return StageOutcome(name=stage.name, status=status, details=details, artifact_dir=stage_dir)
