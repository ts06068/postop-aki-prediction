"""Evaluate frozen or newly trained postoperative AKI models."""

from __future__ import annotations

import argparse
from pathlib import Path

from postop_aki.config import load_experiment_config
from postop_aki.pipeline import evaluate_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Private completed dataset")
    parser.add_argument("--config", required=True, help="Outcome YAML configuration")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional root containing newly trained models",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2_000,
        help="Bootstrap replicates (manuscript default: 2000)",
    )
    parser.add_argument(
        "--output",
        default="outputs/metrics.csv",
        help="CSV destination for the evaluation table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    results = evaluate_experiment(
        args.data,
        config,
        model_dir=args.model_dir,
        n_bootstrap=args.bootstrap,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Export aggregate metrics only; never write patient-level predictions.
    results.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
