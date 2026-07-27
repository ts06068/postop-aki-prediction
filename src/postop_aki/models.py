"""XGBoost construction, artifact integrity, and prediction helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


MODEL_SEED = 0  # Make XGBoost's implicit seed explicit.


def build_classifier(parameters: dict[str, Any]) -> XGBClassifier:
    """Build the binary XGBoost classifier used by all six variants."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        enable_categorical=True,
        random_state=MODEL_SEED,
        n_jobs=-1,
        **parameters,
    )


def file_sha256(path: str | Path) -> str:
    """Calculate a model artifact checksum without loading the model."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        # Hash in 1 MiB chunks to bound memory use.
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_model(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> XGBClassifier:
    """Load a frozen model after optional integrity verification."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if expected_sha256 and file_sha256(model_path) != expected_sha256.lower():
        raise ValueError(f"Checksum mismatch for {model_path}")

    model = XGBClassifier()
    model.load_model(str(model_path))
    return model


def save_model(model: XGBClassifier, path: str | Path) -> Path:
    """Save a newly trained model to an explicit output directory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    return output_path


def predict_probabilities(model: XGBClassifier, features: pd.DataFrame) -> np.ndarray:
    """Predict event probabilities after checking frozen feature order."""
    trained_features = model.get_booster().feature_names
    if trained_features and list(features.columns) != trained_features:
        raise ValueError("Prediction columns do not match the model feature order.")
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


__all__ = [
    "build_classifier",
    "file_sha256",
    "load_frozen_model",
    "predict_probabilities",
    "save_model",
]
