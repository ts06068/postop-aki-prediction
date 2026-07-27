"""Train the three XGBoost variants for one AKI outcome."""

from __future__ import annotations

import argparse
from pathlib import Path

from postop_aki.config import load_experiment_config
from postop_aki.pipeline import train_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Private completed dataset")
    parser.add_argument("--config", required=True, help="Outcome YAML configuration")
    parser.add_argument(
        "--output-dir",
        default="outputs/models",
        help="Directory for newly trained models",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Repeat five-fold tuning instead of using frozen parameters",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    # The default output path leaves frozen models untouched.
    artifacts = train_experiment(
        args.data,
        config,
        Path(args.output_dir),
        tune=args.tune,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
