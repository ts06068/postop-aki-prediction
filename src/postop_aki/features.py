"""Frozen feature sets and SHAP-based feature ranking."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


# Feature order is frozen because XGBoost uses column order during prediction.
ANY_AKI_FULL = (
    "BUN_post",
    "Albumin",
    "eGFR",
    "ASA_cat",
    "BUN",
    "UricAcid_post",
    "Hct",
    "Sex",
    "InorganicP_post",
    "ProteinAlbuminRandomUrine_post",
    "RDW",
    "InorganicP",
    "CardiacSurgery",
    "aPTT_post",
    "UricAcid",
    "K_post",
    "PT",
    "Hemoglobin",
    "PLTCount",
    "aPTT",
    "DM",
    "Na",
    "K",
    "Albumin_post",
    "BP_cat_hypertension",
    "Cl_post",
    "Age",
    "Hemoglobin_post",
    "TBilirubin",
    "WBCCount_post",
    "TotalProtein",
    "AlkPhos",
    "Calcium_post",
    "PLTCount_post",
    "Duration",
    "TBilirubin_post",
    "Na_post",
    "Calcium",
    "NSAIDS",
    "ARB",
    "BMI",
    "Cl",
    "ProteinAlbuminRandomUrine",
    "TotalProtein_post",
    "WBCCount",
    "AlkPhos_post",
    "AST",
    "ALT",
    "PT_post",
    "Asthma",
    "PulTbc",
    "CVA",
    "Hepatitis",
    "NeurologicDisease",
    "BP_cat_hypotension",
)

ANY_AKI_PREOPERATIVE = (
    "eGFR",
    "Albumin",
    "ASA_cat",
    "Sex",
    "Hct",
    "RDW",
    "CardiacSurgery",
    "aPTT",
    "PLTCount",
    "PT",
    "ProteinAlbuminRandomUrine",
    "Na",
    "Hemoglobin",
    "BP_cat_hypertension",
    "DM",
    "TBilirubin",
    "InorganicP",
    "WBCCount",
    "K",
    "TotalProtein",
    "AlkPhos",
    "Age",
    "UricAcid",
    "Calcium",
    "ARB",
    "Cl",
    "BUN",
    "NSAIDS",
    "AST",
    "BMI",
    "CVA",
    "ALT",
    "Hepatitis",
    "Asthma",
    "NeurologicDisease",
    "PulTbc",
    "BP_cat_hypotension",
)

ANY_AKI_REDUCED = (
    "eGFR",
    "Albumin",
    "ASA_cat",
    "Sex",
    "Hct",
    "RDW",
    "CardiacSurgery",
    "aPTT",
)

SEVERE_AKI_FULL = (
    "BUN_post",
    "ASA_cat",
    "eGFR",
    "RDW",
    "Hct",
    "Albumin",
    "UricAcid_post",
    "ProteinAlbuminRandomUrine_post",
    "BP_cat_hypertension",
    "InorganicP_post",
    "BUN",
    "AlkPhos_post",
    "aPTT_post",
    "Hemoglobin_post",
    "InorganicP",
    "Albumin_post",
    "K_post",
    "Hemoglobin",
    "Calcium_post",
    "PT_post",
    "Na",
    "aPTT",
    "Cl_post",
    "TBilirubin_post",
    "Age",
    "UricAcid",
    "Sex",
    "TotalProtein_post",
    "ProteinAlbuminRandomUrine",
    "PLTCount_post",
    "DM",
    "PT",
    "TotalProtein",
    "WBCCount",
    "PLTCount",
    "AlkPhos",
    "K",
    "Calcium",
    "WBCCount_post",
    "Cl",
    "Na_post",
    "Duration",
    "BMI",
    "CardiacSurgery",
    "ARB",
    "AST",
    "ALT",
    "TBilirubin",
    "NSAIDS",
    "PulTbc",
    "NeurologicDisease",
    "Hepatitis",
    "CVA",
    "Asthma",
    "BP_cat_hypotension",
)

SEVERE_AKI_PREOPERATIVE = (
    "eGFR",
    "ASA_cat",
    "Albumin",
    "RDW",
    "Hct",
    "BP_cat_hypertension",
    "ProteinAlbuminRandomUrine",
    "BUN",
    "Sex",
    "aPTT",
    "AlkPhos",
    "InorganicP",
    "Calcium",
    "PT",
    "Hemoglobin",
    "Na",
    "PLTCount",
    "WBCCount",
    "K",
    "TBilirubin",
    "Cl",
    "Age",
    "DM",
    "TotalProtein",
    "UricAcid",
    "CardiacSurgery",
    "AST",
    "BMI",
    "ALT",
    "CVA",
    "ARB",
    "NSAIDS",
    "BP_cat_hypotension",
    "Hepatitis",
    "NeurologicDisease",
    "Asthma",
    "PulTbc",
)

SEVERE_AKI_REDUCED = (
    "eGFR",
    "ASA_cat",
    "Albumin",
    "RDW",
    "Hct",
    "BP_cat_hypertension",
    "ProteinAlbuminRandomUrine",
    "BUN",
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "any_aki.full": ANY_AKI_FULL,
    "any_aki.preoperative": ANY_AKI_PREOPERATIVE,
    "any_aki.reduced": ANY_AKI_REDUCED,
    "severe_aki.full": SEVERE_AKI_FULL,
    "severe_aki.preoperative": SEVERE_AKI_PREOPERATIVE,
    "severe_aki.reduced": SEVERE_AKI_REDUCED,
}


def get_feature_names(name: str) -> tuple[str, ...]:
    """Return a frozen feature list by configuration key."""
    try:
        return FEATURE_SETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown feature set: {name}") from exc


def rank_features_by_mean_shap(
    model: object,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Rank columns by mean absolute SHAP value on the training set."""
    import shap

    # TreeExplainer matches the selection procedure used in the original notebooks.
    values = np.asarray(shap.TreeExplainer(model).shap_values(features))
    importance = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"feature": features.columns, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def select_ranked_features(
    ranking: pd.DataFrame,
    count: int = 8,
) -> Sequence[str]:
    """Select the leading features from a SHAP ranking table."""
    if count < 1 or count > len(ranking):
        raise ValueError("count must be between 1 and the number of ranked features.")
    return ranking.loc[: count - 1, "feature"].tolist()


__all__ = [
    "FEATURE_SETS",
    "get_feature_names",
    "rank_features_by_mean_shap",
    "select_ranked_features",
]
