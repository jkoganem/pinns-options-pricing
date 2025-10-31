"""Training stages for European PINNs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from multi_option.bs_closed_form import bs_price
from multi_option.experiments.types import PipelineContext, StageOutcome, StageSpec
from multi_option.pinns.model import make_pinn
from multi_option.pinns.train import train_pinn


REQUIRED_PROBLEM_KEYS = {"r", "q", "sigma", "K", "T", "s_max"}


def _load_model(model_path: Path, hidden: int, use_fourier: bool = False) -> torch.nn.Module:
    model = make_pinn(hidden, use_fourier=use_fourier)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_stage(stage: StageSpec, context: PipelineContext, stage_seed: int) -> StageOutcome:
    stage_dir = context.output_root / stage.name
    stage_dir.mkdir(parents=True, exist_ok=True)

    problem: Dict[str, float] = dict(stage.config.get("problem", {}))
    missing = REQUIRED_PROBLEM_KEYS - problem.keys()
    if missing:
        raise ValueError(f"Missing problem parameters for stage '{stage.name}': {sorted(missing)}")

    experiments: Iterable[Dict[str, object]] = stage.config.get("experiments", [])
    experiments = list(experiments)
    if not experiments:
        raise ValueError(f"No experiments specified for stage '{stage.name}'")

    evaluation_cfg = stage.config.get("evaluation", {})
    evaluation_spots: List[float] = list(evaluation_cfg.get("spots", [80.0, 100.0, 120.0]))
    evaluation_tau: float = float(evaluation_cfg.get("tau", problem["T"]))

    aggregated: Dict[str, Dict[str, object]] = {}

    for idx, experiment in enumerate(experiments):
        label = str(experiment.get("label", f"experiment_{idx}"))
        option_type = str(experiment.get("option_type", "call")).lower()
        is_call = option_type == "call"
        hidden = int(experiment.get("hidden", 64))
        epochs = int(experiment.get("epochs", 3000))
        lr = float(experiment.get("lr", 1e-3))
        batch_size = int(experiment.get("batch_size", 1000))
        save_weights = bool(experiment.get("save_weights", True))
        use_fourier = bool(experiment.get("use_fourier", False))  # Default: simple PINN
        seed = int(experiment.get("seed_override", stage_seed + idx))

        exp_dir = stage_dir / label
        exp_dir.mkdir(parents=True, exist_ok=True)

        metrics = train_pinn(
            r=float(problem["r"]),
            q=float(problem["q"]),
            sigma=float(problem["sigma"]),
            K=float(problem["K"]),
            T=float(problem["T"]),
            s_max=float(problem["s_max"]),
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            seed=seed,
            call=is_call,
            batch_size=batch_size,
            save_dir=str(exp_dir) if save_weights else None,
            use_fourier=use_fourier,
        )
        metrics["seedused"] = seed
        metrics_path = exp_dir / "training_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        evaluation_records: List[Dict[str, float]] = []
        model_path = metrics.get("model_path")
        if model_path is not None:
            model = _load_model(Path(model_path), hidden, use_fourier=use_fourier)
            with torch.no_grad():
                for spot in evaluation_spots:
                    s_tensor = torch.tensor([float(spot)], dtype=torch.float32)
                    t_tensor = torch.tensor([evaluation_tau], dtype=torch.float32)
                    pinn_value = float(model(s_tensor, t_tensor).item())
                    reference_value = float(
                        bs_price(
                            float(spot),
                            float(problem["K"]),
                            float(problem["r"]),
                            float(problem["q"]),
                            float(problem["sigma"]),
                            float(problem["T"]),
                            is_call,
                        )
                    )
                    if reference_value:
                        rel_error = (pinn_value - reference_value) / reference_value
                    else:
                        rel_error = None
                    evaluation_records.append(
                        {
                            "spot": float(spot),
                            "tau": evaluation_tau,
                            "pinn_price": pinn_value,
                            "reference_price": reference_value,
                            "relative_error": rel_error,
                        }
                    )

        evaluation_path = exp_dir / "evaluation.json"
        evaluation_path.write_text(json.dumps(evaluation_records, indent=2), encoding="utf-8")

        aggregated[label] = {
            "metrics_file": str(metrics_path),
            "evaluation_file": str(evaluation_path),
            **metrics,
        }

    summary_path = stage_dir / "pinn_summary.json"
    summary_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")

    details = {
        "experiments": list(aggregated.keys()),
        "summary_file": str(summary_path),
    }
    return StageOutcome(name=stage.name, status="success", details=details, artifact_dir=stage_dir)
