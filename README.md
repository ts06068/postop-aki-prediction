# Postoperative AKI Prediction

## Study Objective

Develop and externally validate XGBoost models for any-stage and severe postoperative AKI using full, preoperative, and reduced preoperative feature sets.

## Folder structure

```text
configs/              Experiment settings for the two AKI outcomes
data/sample.xlsx      Fully synthetic schema example
models/               Frozen models used for prediction parity
notebooks/modeling.ipynb
scripts/              Command-line entry points
src/postop_aki/       Reusable data, training, and evaluation modules
tests/                 Lightweight unit tests
```

## Code structure

`data.py` loads and splits cohorts, `features.py` preserves the final feature order, `models.py` builds or loads XGBoost models, and `training.py` handles fitting and optional tuning. `pipeline.py` connects those steps, while `evaluation.py` calculates the manuscript-specified metrics. The notebook only orchestrates these modules.

## Environment

Python 3.10 and XGBoost 1.7.6 are used to match the manuscript environment.

```bash
conda env create -f environment.yml
conda activate postop-aki
```

## How to run

Validate the synthetic sample without training:

```bash
python scripts/validate_data.py --data data/sample.xlsx
```

Keep the private completed dataset outside Git and pass its local path explicitly:

```bash
python scripts/evaluate.py --data outcome_full_ml.csv --config configs/any_aki.yaml
python scripts/train.py --data outcome_full_ml.csv --config configs/severe_aki.yaml
```

`sample.xlsx` is for schema checks only; published estimates require the original completed dataset.
