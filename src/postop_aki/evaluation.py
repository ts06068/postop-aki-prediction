"""Performance evaluation for postoperative AKI predictions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_RANDOM_SEED = 42
DEFAULT_CALIBRATION_BINS = 50


def _binary_inputs(
    y_true: ArrayLike, y_prob: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated one-dimensional outcome and probability arrays."""
    try:
        outcome = np.asarray(y_true, dtype=float)
        probability = np.asarray(y_prob, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true and y_prob must be numeric.") from exc

    if outcome.ndim != 1 or probability.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional.")
    if outcome.size == 0:
        raise ValueError("y_true and y_prob must not be empty.")
    if outcome.size != probability.size:
        raise ValueError("y_true and y_prob must have the same length.")
    if not np.isfinite(outcome).all() or not np.isfinite(probability).all():
        raise ValueError(
            "y_true and y_prob must not contain missing or infinite values."
        )
    if not np.isin(outcome, (0, 1)).all():
        raise ValueError("y_true must contain only binary values 0 and 1.")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("y_prob must contain probabilities between 0 and 1.")

    return outcome.astype(int), probability


def _require_two_classes(outcome: np.ndarray) -> None:
    if np.unique(outcome).size != 2:
        raise ValueError("Both outcome classes are required for this calculation.")


def _check_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    return threshold


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _threshold_metrics(
    outcome: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(outcome, predicted, labels=[0, 1]).ravel()

    return {
        "accuracy": (tp + tn) / outcome.size,
        "sensitivity": _safe_ratio(tp, tp + fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "ppv": _safe_ratio(tp, tp + fp),
        "npv": _safe_ratio(tn, tn + fn),
        "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
    }


def _discrimination_metrics(
    outcome: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(outcome, probability)),
        "auprc": float(average_precision_score(outcome, probability)),
    }


def youden_threshold(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Return the probability threshold that maximizes the Youden index."""
    outcome, probability = _binary_inputs(y_true, y_prob)
    _require_two_classes(outcome)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        outcome, probability
    )
    finite = np.flatnonzero(np.isfinite(thresholds))
    best = finite[np.argmax((true_positive_rate - false_positive_rate)[finite])]
    return float(thresholds[best])


def threshold_metrics(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    threshold: float,
) -> dict[str, float]:
    """Calculate classification metrics at a fixed probability threshold."""
    outcome, probability = _binary_inputs(y_true, y_prob)
    return _threshold_metrics(outcome, probability, _check_threshold(threshold))


def discrimination_metrics(y_true: ArrayLike, y_prob: ArrayLike) -> dict[str, float]:
    """Calculate AUROC and area under the precision-recall curve (AUPRC)."""
    outcome, probability = _binary_inputs(y_true, y_prob)
    _require_two_classes(outcome)
    return _discrimination_metrics(outcome, probability)


def bootstrap_confidence_intervals(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    threshold: float | None = None,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Estimate percentile bootstrap confidence intervals for performance metrics."""
    outcome, probability = _binary_inputs(y_true, y_prob)
    _require_two_classes(outcome)
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    point_estimates = _discrimination_metrics(outcome, probability)
    if threshold is not None:
        threshold = _check_threshold(threshold)
        point_estimates.update(_threshold_metrics(outcome, probability, threshold))

    rng = np.random.default_rng(seed)
    bootstrap_values = {metric: [] for metric in point_estimates}

    for _ in range(n_bootstrap):
        # Reject one-class resamples; AUROC is undefined for them.
        for _ in range(1_000):
            sample = rng.integers(0, outcome.size, size=outcome.size)
            if np.unique(outcome[sample]).size == 2:
                break
        else:
            raise RuntimeError(
                "Unable to draw a bootstrap sample containing both classes."
            )

        sampled_outcome = outcome[sample]
        sampled_probability = probability[sample]
        estimates = _discrimination_metrics(sampled_outcome, sampled_probability)
        if threshold is not None:
            estimates.update(
                _threshold_metrics(sampled_outcome, sampled_probability, threshold)
            )
        for metric, value in estimates.items():
            bootstrap_values[metric].append(value)

    alpha = (1 - confidence_level) / 2
    rows = []
    for metric, estimate in point_estimates.items():
        lower, upper = np.quantile(bootstrap_values[metric], [alpha, 1 - alpha])
        rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
            }
        )

    return pd.DataFrame(rows).set_index("metric")


def _quantile_bin_ids(probability: np.ndarray, n_bins: int) -> np.ndarray:
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    # Calibration and risk summaries use the same quantile rule.
    edges = np.quantile(probability, np.linspace(0, 1, n_bins + 1))
    return np.searchsorted(edges[1:-1], probability)


def calibration_table(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> pd.DataFrame:
    """Summarize observed and predicted risk in equal-frequency quantile bins."""
    outcome, probability = _binary_inputs(y_true, y_prob)
    bin_ids = _quantile_bin_ids(probability, n_bins)
    frame = pd.DataFrame(
        {"bin_id": bin_ids, "outcome": outcome, "probability": probability}
    )

    table = (
        frame.groupby("bin_id", sort=True, observed=True)
        .agg(
            n=("outcome", "size"),
            mean_predicted_risk=("probability", "mean"),
            observed_event_rate=("outcome", "mean"),
        )
        .reset_index()
    )
    table["calibration_error"] = (
        table["observed_event_rate"] - table["mean_predicted_risk"]
    ).abs()
    table["bin"] = table.pop("bin_id") + 1
    return table[
        [
            "bin",
            "n",
            "mean_predicted_risk",
            "observed_event_rate",
            "calibration_error",
        ]
    ]


def expected_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> float:
    """Calculate sample-weighted ECE using equal-frequency quantile bins."""
    table = calibration_table(y_true, y_prob, n_bins=n_bins)
    return float(np.average(table["calibration_error"], weights=table["n"]))


def decision_curve_net_benefit(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    thresholds: ArrayLike | None = None,
) -> pd.DataFrame:
    """Calculate model, treat-all, and treat-none decision-curve net benefit."""
    outcome, probability = _binary_inputs(y_true, y_prob)
    if thresholds is None:
        threshold_values = np.arange(0.01, 1.0, 0.01)
    else:
        threshold_values = np.asarray(thresholds, dtype=float)

    if threshold_values.ndim != 1 or threshold_values.size == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional sequence.")
    if not np.isfinite(threshold_values).all():
        raise ValueError("thresholds must contain only finite values.")
    if ((threshold_values <= 0) | (threshold_values >= 1)).any():
        raise ValueError("decision-curve thresholds must lie strictly between 0 and 1.")

    prevalence = outcome.mean()
    rows = []
    for threshold in threshold_values:
        predicted = probability >= threshold
        true_positives = np.count_nonzero(predicted & (outcome == 1))
        false_positives = np.count_nonzero(predicted & (outcome == 0))
        threshold_odds = threshold / (1 - threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "model_net_benefit": (
                    true_positives / outcome.size
                    - false_positives / outcome.size * threshold_odds
                ),
                "treat_all_net_benefit": (
                    prevalence - (1 - prevalence) * threshold_odds
                ),
                "treat_none_net_benefit": 0.0,
            }
        )

    return pd.DataFrame(rows)


def evaluate_reduced_model_subgroups(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    subgroups: pd.DataFrame,
    threshold: float,
    *,
    subgroup_columns: Sequence[str] | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Evaluate fixed reduced-model predictions within prespecified subgroups."""
    outcome, probability = _binary_inputs(y_true, y_prob)
    threshold = _check_threshold(threshold)
    if len(subgroups) != outcome.size:
        raise ValueError("subgroups must have the same number of rows as y_true.")

    columns = (
        list(subgroup_columns) if subgroup_columns is not None else list(subgroups)
    )
    missing = [column for column in columns if column not in subgroups]
    if missing:
        raise KeyError(f"Unknown subgroup columns: {missing}")
    if not columns:
        raise ValueError("At least one subgroup column is required.")

    rows: list[dict[str, object]] = []
    for column in columns:
        values = subgroups[column].reset_index(drop=True)
        for level in pd.unique(values.dropna()):
            # Keep predictions and the threshold fixed across subgroup slices.
            mask = values.eq(level).fillna(False).to_numpy(dtype=bool)
            group_outcome = outcome[mask]
            group_probability = probability[mask]
            row: dict[str, object] = {
                "subgroup": column,
                "level": level,
                "n": int(mask.sum()),
                "events": int(group_outcome.sum()),
                "event_rate": float(group_outcome.mean()),
                "threshold": threshold,
                **_threshold_metrics(group_outcome, group_probability, threshold),
            }

            if np.unique(group_outcome).size == 2:
                intervals = bootstrap_confidence_intervals(
                    group_outcome,
                    group_probability,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                for metric in ("auroc", "auprc"):
                    row[metric] = intervals.loc[metric, "estimate"]
                    row[f"{metric}_ci_lower"] = intervals.loc[metric, "ci_lower"]
                    row[f"{metric}_ci_upper"] = intervals.loc[metric, "ci_upper"]
            else:
                for metric in ("auroc", "auprc"):
                    row[metric] = np.nan
                    row[f"{metric}_ci_lower"] = np.nan
                    row[f"{metric}_ci_upper"] = np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def predicted_risk_deciles(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Summarize predicted and observed risk across probability deciles."""
    outcome, probability = _binary_inputs(y_true, y_prob)
    bin_ids = _quantile_bin_ids(probability, n_deciles)
    frame = pd.DataFrame(
        {"risk_decile": bin_ids + 1, "outcome": outcome, "probability": probability}
    )

    return (
        frame.groupby("risk_decile", sort=True, observed=True)
        .agg(
            n=("outcome", "size"),
            events=("outcome", "sum"),
            mean_predicted_risk=("probability", "mean"),
            min_predicted_risk=("probability", "min"),
            max_predicted_risk=("probability", "max"),
            observed_event_rate=("outcome", "mean"),
        )
        .reset_index()
    )


__all__ = [
    "bootstrap_confidence_intervals",
    "calibration_table",
    "decision_curve_net_benefit",
    "discrimination_metrics",
    "evaluate_reduced_model_subgroups",
    "expected_calibration_error",
    "predicted_risk_deciles",
    "threshold_metrics",
    "youden_threshold",
]
