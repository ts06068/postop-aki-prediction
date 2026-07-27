"""Dataset loading, preprocessing, and cohort splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from postop_aki.features import FEATURE_SETS, get_feature_names


INTERNAL_HOSPITAL_CODE = 10
EXTERNAL_HOSPITAL_CODE = 20
SPLIT_SEED = 42


@dataclass(frozen=True)
class Cohorts:
    """Internal development and independent external cohorts."""

    internal: pd.DataFrame
    external: pd.DataFrame


@dataclass(frozen=True)
class InternalSplit:
    """Stratified internal training and hold-out test data."""

    train: pd.DataFrame
    test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        # The completed research dataset is pipe-delimited and stores its index first.
        return pd.read_csv(path, sep="|", index_col=0)
    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path)
        unnamed = [column for column in frame if str(column).startswith("Unnamed:")]
        return frame.drop(columns=unnamed)
    raise ValueError("Supported data formats are .csv and .xlsx.")


def _clean_column_name(column: object) -> str:
    return re.sub(r"[<\[\]]", "_", str(column))


def _required_model_columns() -> set[str]:
    return {feature for names in FEATURE_SETS.values() for feature in names}


def load_modeling_data(path: str | Path) -> pd.DataFrame:
    """Load a private completed dataset or the synthetic schema sample."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    return prepare_modeling_data(_read_dataset(data_path))


def prepare_modeling_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Create fixed blood-pressure indicators and numeric model columns."""
    data = frame.copy()
    data.columns = [_clean_column_name(column) for column in data.columns]

    required_base = {"HospitalCode", "AKIStage", "AKI_binary"}
    missing_base = sorted(required_base.difference(data.columns))
    if missing_base:
        raise KeyError(f"Missing required columns: {missing_base}")

    if "BP_cat_hypertension" not in data or "BP_cat_hypotension" not in data:
        if "BP_cat" not in data:
            raise KeyError("BP_cat is required to create blood-pressure indicators.")
        # Normal blood pressure remains the reference category, as in the notebooks.
        blood_pressure = data["BP_cat"].astype("string").str.strip().str.lower()
        data["BP_cat_hypertension"] = blood_pressure.eq("hypertension").astype(int)
        data["BP_cat_hypotension"] = blood_pressure.eq("hypotension").astype(int)

    required_features = _required_model_columns()
    missing_features = sorted(required_features.difference(data.columns))
    if missing_features:
        raise KeyError(f"Missing model features: {missing_features}")

    # Explicit numeric conversion catches schema drift before a long training run.
    numeric_columns = sorted(required_features | required_base)
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    return data


def split_cohorts(data: pd.DataFrame) -> Cohorts:
    """Split hospital 10 for development and hospital 20 for validation."""
    internal = data.loc[data["HospitalCode"] == INTERNAL_HOSPITAL_CODE].copy()
    external = data.loc[data["HospitalCode"] == EXTERNAL_HOSPITAL_CODE].copy()
    if internal.empty or external.empty:
        raise ValueError("Both internal (10) and external (20) cohorts are required.")
    return Cohorts(internal=internal, external=external)


def make_target(data: pd.DataFrame, target: str) -> pd.Series:
    """Create manuscript-aligned binary outcomes."""
    if target == "any_aki":
        outcome = data["AKI_binary"]
    elif target == "severe_aki":
        # Severe AKI is KDIGO stage 2 or 3.
        outcome = data["AKIStage"].gt(1).astype(int)
    else:
        raise ValueError("target must be 'any_aki' or 'severe_aki'.")
    return outcome.astype(int).rename(target)


def split_internal(
    internal: pd.DataFrame,
    target: str,
    *,
    test_size: float = 0.2,
    seed: int = SPLIT_SEED,
) -> InternalSplit:
    """Create the outcome-stratified 8:2 internal split."""
    outcome = make_target(internal, target)
    train, test, y_train, y_test = train_test_split(
        internal,
        outcome,
        test_size=test_size,
        stratify=outcome,
        random_state=seed,
    )
    return InternalSplit(train=train, test=test, y_train=y_train, y_test=y_test)


def build_model_matrix(data: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Select model columns in their frozen training order."""
    features = get_feature_names(feature_set)
    return data.loc[:, list(features)].copy()


__all__ = [
    "Cohorts",
    "InternalSplit",
    "build_model_matrix",
    "load_modeling_data",
    "make_target",
    "prepare_modeling_data",
    "split_cohorts",
    "split_internal",
]
