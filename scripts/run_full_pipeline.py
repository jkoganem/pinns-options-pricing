"""Execute the staged research pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from multi_option.experiments import load_pipeline_config, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staged option pricing pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline.toml"),
        help="Path to the pipeline TOML configuration",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Project workspace directory (defaults to current working directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_pipeline_config(args.config)
    outcomes = run_pipeline(spec, workspace=args.workspace)

    print("Pipeline execution completed.")
    for outcome in outcomes:
        print(f"- {outcome.name}: {outcome.status}")
        if outcome.details:
            print(f"  Details: {outcome.details}")


if __name__ == "__main__":
    main()
