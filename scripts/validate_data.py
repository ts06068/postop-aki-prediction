"""Validate dataset schema without fitting a model."""

from __future__ import annotations

import argparse

from postop_aki.data import load_modeling_data, split_cohorts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to CSV or Excel input")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_modeling_data(args.data)
    cohorts = split_cohorts(data)
    # Do not print patient-level data.
    print(f"columns={data.shape[1]}")
    print(f"internal_rows={len(cohorts.internal)}")
    print(f"external_rows={len(cohorts.external)}")


if __name__ == "__main__":
    main()
