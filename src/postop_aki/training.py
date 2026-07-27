"""Shared fitting and five-fold hyperparameter tuning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from postop_aki.models import build_classifier


CV_FOLDS = 5
CV_SEED = 42


def make_stratified_folds(
    *,
    n_splits: int = CV_FOLDS,
    seed: int = CV_SEED,
) -> StratifiedKFold:
    """Create the shuffled five-fold splitter used during model development."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cross_validated_auc(
    model: XGBClassifier,
    features: pd.DataFrame,
    outcome: pd.Series,
) -> float:
    """Return mean five-fold AUROC for a candidate classifier."""
    scores = cross_val_score(
        model,
        features,
        outcome,
        scoring="roc_auc",
        cv=make_stratified_folds(),
        n_jobs=1,
    )
    return float(np.mean(scores))


def fit_classifier(
    features: pd.DataFrame,
    outcome: pd.Series,
    parameters: Mapping[str, Any],
) -> XGBClassifier:
    """Fit one final model on the complete internal training split."""
    model = build_classifier(dict(parameters))
    model.fit(features, outcome)
    return model


def tune_parameters(
    features: pd.DataFrame,
    outcome: pd.Series,
    *,
    n_trials: int = 30,
    seed: int = CV_SEED,
) -> dict[str, Any]:
    """Tune XGBoost parameters with seeded Optuna and five-fold AUROC."""
    folds = make_stratified_folds(seed=seed)

    def objective(trial: optuna.Trial) -> float:
        parameters = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 700),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 9),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 9.0),
        }
        fold_scores = []
        for train_index, validation_index in folds.split(features, outcome):
            # Fit inside each fold; validation rows remain unseen.
            model = build_classifier(parameters)
            model.fit(features.iloc[train_index], outcome.iloc[train_index])
            probability = model.predict_proba(features.iloc[validation_index])[:, 1]
            fold_scores.append(
                roc_auc_score(outcome.iloc[validation_index], probability)
            )
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_trial.params)


__all__ = [
    "cross_validated_auc",
    "fit_classifier",
    "make_stratified_folds",
    "tune_parameters",
]
