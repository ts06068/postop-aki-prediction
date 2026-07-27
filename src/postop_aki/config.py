"""Load the human-readable experiment configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for one model variant."""

    name: str
    feature_set: str
    artifact: Path
    sha256: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by the three variants for one AKI outcome."""

    name: str
    target: str
    models: tuple[ModelConfig, ...]
    project_root: Path


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Read and validate an outcome-specific YAML configuration."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    if not isinstance(raw, dict) or not isinstance(raw.get("models"), dict):
        raise ValueError(f"Invalid experiment configuration: {config_path}")

    models = []
    for name, values in raw["models"].items():
        # Resolve artifact paths from the repository root, not the caller's cwd.
        models.append(
            ModelConfig(
                name=name,
                feature_set=str(values["feature_set"]),
                artifact=Path(values["artifact"]),
                sha256=str(values["sha256"]).lower(),
                parameters=dict(values["parameters"]),
            )
        )

    target = str(raw.get("target", ""))
    if target not in {"any_aki", "severe_aki"}:
        raise ValueError("target must be 'any_aki' or 'severe_aki'.")

    return ExperimentConfig(
        name=str(raw.get("name", target)),
        target=target,
        models=tuple(models),
        project_root=config_path.parent.parent,
    )


__all__ = ["ExperimentConfig", "ModelConfig", "load_experiment_config"]
