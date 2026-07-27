# Postoperative AKI Prediction

## Study Objective

Develop and externally validate XGBoost models for any-stage and severe postoperative AKI using full, preoperative, and reduced preoperative feature sets.

## Folder structure

```text
.github/workflows/    GitHub Actions CI
configs/              Outcome-specific model settings
data/sample.xlsx      Fully synthetic 10-row schema sample
models/               Checksum-verified frozen model artifacts
notebooks/modeling.ipynb
scripts/              Validation, training, and evaluation entry points
src/postop_aki/       Reusable pipeline modules
tests/                Metric and project-integrity tests
```

## Code structure

- `config.py` and `features.py` define experiment settings and frozen feature order.
- `data.py` validates input, derives outcomes, and creates the internal and external cohorts.
- `models.py` and `training.py` verify, load, fit, and save XGBoost models.
- `evaluation.py` calculates performance, calibration, bootstrap, decision-curve, subgroup, and risk-decile results.
- `pipeline.py` connects the modules; the scripts and notebook are thin entry points.
- `ci.yml` runs code-quality checks, sample validation, and unit tests on GitHub Actions.

## Environment

- Python 3.10
- XGBoost 1.7.6
- Package versions are pinned in `requirements.txt`.

```powershell
conda env create -f environment.yml
conda activate postop-aki
```

## How to run

Run commands from the repository root. Validate the synthetic sample without training:

```powershell
python scripts/validate_data.py --data data/sample.xlsx
```

Evaluate the frozen models with the private completed dataset:

```powershell
$dataPath = "D:\path\to\outcome_full_ml.csv"
python scripts/evaluate.py --data $dataPath --config configs/any_aki.yaml --output outputs/any_aki_metrics.csv
python scripts/evaluate.py --data $dataPath --config configs/severe_aki.yaml --output outputs/severe_aki_metrics.csv
```

Optionally retrain with the stored hyperparameters and evaluate the new artifacts:

```powershell
python scripts/train.py --data $dataPath --config configs/any_aki.yaml --output-dir outputs/models
python scripts/evaluate.py --data $dataPath --config configs/any_aki.yaml --model-dir outputs/models --output outputs/any_aki_retrained_metrics.csv
```

Run the lightweight checks:

```powershell
ruff check src scripts tests
ruff format --check src scripts tests
python -m unittest discover -s tests -v
```

`sample.xlsx` is for schema validation only. Published estimates require the private completed dataset; `--tune` repeats the computationally expensive five-fold search and is not part of the standard run.
