"""High-level training and evaluation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from postop_aki.config import ExperimentConfig
from postop_aki.data import (
    Cohorts,
    InternalSplit,
    build_model_matrix,
    load_modeling_data,
    make_target,
    split_cohorts,
    split_internal,
)
from postop_aki.evaluation import (
    bootstrap_confidence_intervals,
    discrimination_metrics,
    expected_calibration_error,
    threshold_metrics,
    youden_threshold,
)
from postop_aki.models import (
    load_frozen_model,
    predict_probabilities,
    save_model,
)
from postop_aki.training import fit_classifier, tune_parameters


@dataclass(frozen=True)
class PreparedExperiment:
    """Cohorts and outcomes shared across all model variants."""

    cohorts: Cohorts
    internal_split: InternalSplit
    external_outcome: pd.Series


def prepare_experiment(
    data_path: str | Path,
    config: ExperimentConfig,
) -> PreparedExperiment:
    """Load data once and create the fixed development/validation split."""
    cohorts = split_cohorts(load_modeling_data(data_path))
    internal = split_internal(cohorts.internal, config.target)
    external_outcome = make_target(cohorts.external, config.target)
    return PreparedExperiment(cohorts, internal, external_outcome)


def train_experiment(
    data_path: str | Path,
    config: ExperimentConfig,
    output_dir: str | Path,
    *,
    tune: bool = False,
) -> dict[str, Path]:
    """Train full, preoperative, and reduced models for one outcome."""
    prepared = prepare_experiment(data_path, config)
    output_root = Path(output_dir)
    artifacts: dict[str, Path] = {}

    for model_config in config.models:
        features = build_model_matrix(
            prepared.internal_split.train,
            model_config.feature_set,
        )
        # Use recorded parameters unless tuning is explicitly requested.
        parameters = (
            tune_parameters(features, prepared.internal_split.y_train)
            if tune
            else model_config.parameters
        )
        model = fit_classifier(features, prepared.internal_split.y_train, parameters)
        artifact = output_root / config.name / f"{model_config.name}.model"
        artifacts[model_config.name] = save_model(model, artifact)

    return artifacts


def _resolve_model_path(
    config: ExperimentConfig,
    model_name: str,
    frozen_path: Path,
    model_dir: str | Path | None,
) -> Path:
    if model_dir is None:
        return config.project_root / frozen_path
    return Path(model_dir) / config.name / f"{model_name}.model"


def evaluate_experiment(
    data_path: str | Path,
    config: ExperimentConfig,
    *,
    model_dir: str | Path | None = None,
    n_bootstrap: int = 2_000,
) -> pd.DataFrame:
    """Evaluate each model on internal hold-out and external cohorts."""
    prepared = prepare_experiment(data_path, config)
    validation_sets = {
        "internal_holdout": (
            prepared.internal_split.test,
            prepared.internal_split.y_test,
        ),
        "external": (prepared.cohorts.external, prepared.external_outcome),
    }
    rows: list[dict[str, object]] = []

    for model_config in config.models:
        model_path = _resolve_model_path(
            config,
            model_config.name,
            model_config.artifact,
            model_dir,
        )
        # Only frozen artifacts have reference checksums.
        expected_hash = model_config.sha256 if model_dir is None else None
        model = load_frozen_model(model_path, expected_sha256=expected_hash)

        for validation_name, (frame, outcome) in validation_sets.items():
            features = build_model_matrix(frame, model_config.feature_set)
            probability = predict_probabilities(model, features)
            # Select a separate Youden threshold for each cohort.
            threshold = youden_threshold(outcome, probability)
            point = discrimination_metrics(outcome, probability)
            point.update(threshold_metrics(outcome, probability, threshold))
            intervals = bootstrap_confidence_intervals(
                outcome,
                probability,
                threshold,
                n_bootstrap=n_bootstrap,
            )

            row: dict[str, object] = {
                "outcome": config.target,
                "model": model_config.name,
                "validation": validation_name,
                "n": len(outcome),
                "threshold": threshold,
                "ece": expected_calibration_error(outcome, probability),
                **point,
            }
            for metric in intervals.index:
                row[f"{metric}_ci_lower"] = intervals.loc[metric, "ci_lower"]
                row[f"{metric}_ci_upper"] = intervals.loc[metric, "ci_upper"]
            rows.append(row)

    return pd.DataFrame(rows)


__all__ = [
    "PreparedExperiment",
    "evaluate_experiment",
    "prepare_experiment",
    "train_experiment",
]
