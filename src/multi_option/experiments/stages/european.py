"""Pricing benchmarks for European-style options."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Mapping

from multi_option.config import parse_config
from multi_option.experiments.types import PipelineContext, StageOutcome, StageSpec
from multi_option.products.american_put import price_american_put_methods
from multi_option.products.barrierup_out_call import price_barrier_methods
from multi_option.products.european import price_european_all_methods


DEFAULT_SCENARIO: Dict[str, object] = {
    "s0": 100.0,
    "K": 100.0,
    "T": 1.0,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.2,
    "s_max": 400.0,
    "ns": 401,
    "nt": 400,
    "mc_paths": 50000,
    "mc_steps": 252,
    "barrier": 120.0,
    "pinn_epochs": 300,
    "pinn_lr": 1e-3,
    "pinn_hidden": 64,
    "seed": 42,
    "chain_csv": None,
    "use_mid": True,
}


def run_stage(stage: StageSpec, context: PipelineContext, stage_seed: int) -> StageOutcome:
    stage_dir = context.output_root / stage.name
    stage_dir.mkdir(parents=True, exist_ok=True)

    scenario = dict(DEFAULT_SCENARIO)
    scenario.update(stage.config.get("scenario", {}))
    scenario.setdefault("seed", stage_seed)
    scenario["out_dir"] = str(stage_dir)
    scenario["plots"] = bool(stage.config.get("generate_plots", False))
    scenario["ui_json"] = bool(stage.config.get("generateui_bundle", False))

    cfg = parse_config(scenario)
    products: Iterable[str] = stage.config.get(
        "products",
        ["european_call", "european_put", "american_put", "barrierup_out_call"],
    )
    reference_method = stage.config.get("reference_method", "bs")

    results: Dict[str, Dict[str, Mapping[str, object]]] = {}

    for product in products:
        if product not in {
            "european_call",
            "european_put",
            "american_put",
            "barrierup_out_call",
        }:
            raise ValueError(f"Unsupported product '{product}' in stage '{stage.name}'")

        if product in {"european_call", "european_put"}:
            is_call = product == "european_call"
            pricing = price_european_all_methods(cfg, is_call)
        elif product == "american_put":
            pricing = price_american_put_methods(cfg)
        else:
            if cfg.barrier is None:
                raise ValueError("Barrier value required for barrier option pricing stage")
            pricing = price_barrier_methods(cfg)

        product_summary: Dict[str, Dict[str, object]] = {}
        reference_price = None
        if reference_method in pricing:
            reference_price = pricing[reference_method].price

        for method, result in pricing.items():
            record = asdict(result)
            if reference_price is not None:
                delta = result.price - reference_price
                record["delta_vs_reference"] = delta
                record["relative_error_vs_reference"] = (
                    delta / reference_price if reference_price else None
                )
            product_summary[method] = record

        results[product] = product_summary

    summary_path = stage_dir / "pricing_results.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    details = {
        "products": list(results.keys()),
        "reference_method": reference_method,
        "result_file": str(summary_path),
    }
    return StageOutcome(name=stage.name, status="success", details=details, artifact_dir=stage_dir)
