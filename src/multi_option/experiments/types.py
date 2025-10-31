"""Shared dataclasses for experiment pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class StageSpec:
    """Single pipeline stage specification."""

    name: str
    type: str
    config: Mapping[str, Any]


@dataclass(frozen=True)
class PipelineSpec:
    """Top-level pipeline specification."""

    seed: int
    output_root: Path
    fail_fast: bool
    python_executable: Optional[str]
    stages: List[StageSpec]


@dataclass(frozen=True)
class PipelineContext:
    """Execution context shared across stages."""

    workspace: Path
    output_root: Path
    python: str


@dataclass
class StageOutcome:
    """Result emitted by a pipeline stage."""

    name: str
    status: str
    details: Dict[str, Any]
    artifact_dir: Path
