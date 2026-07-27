# Postoperative AKI Prediction

## Study Objective

Train and externally validate XGBoost classifiers for any-stage and severe postoperative AKI. Each outcome has full, preoperative, and reduced feature variants.

## Folder structure

```text
.github/workflows/    GitHub Actions configuration
configs/              Outcomes, parameters, artifact paths, and hashes
data/sample.xlsx      Synthetic schema sample with 10 rows
models/               Frozen XGBoost model artifacts
notebooks/modeling.ipynb
scripts/              Command-line entry points
src/postop_aki/       Data, training, and evaluation code
tests/                Metric and repository-integrity tests
```

## Code structure

- `config.py` reads the experiment definitions. `features.py` fixes the feature order for each model.
- `data.py` loads the dataset, checks its schema, creates outcomes, and splits hospitals 10 and 20.
- `models.py` builds classifiers and verifies model artifacts. `training.py` owns fitting and tuning.
- `evaluation.py` computes the reported metrics. `pipeline.py` connects the training and evaluation paths.
- The scripts and notebook pass inputs to the pipeline. They do not duplicate pipeline logic.
- `ci.yml` checks formatting, lint, the synthetic sample, and unit tests. It does not train models.

## Environment

Use Python 3.10 and XGBoost 1.7.6. `requirements.txt` pins the remaining dependencies.

```powershell
conda env create -f environment.yml
conda activate postop-aki
```

## How to run

Run every command from the repository root.

Check the input schema:

```powershell
python scripts/validate_data.py --data data/sample.xlsx
```

Evaluate the frozen models:

```powershell
$dataPath = "D:\path\to\outcome_full_ml.csv"
python scripts/evaluate.py --data $dataPath --config configs/any_aki.yaml --output outputs/any_aki_metrics.csv
python scripts/evaluate.py --data $dataPath --config configs/severe_aki.yaml --output outputs/severe_aki_metrics.csv
```

Train new models with the stored parameters:

```powershell
python scripts/train.py --data $dataPath --config configs/any_aki.yaml --output-dir outputs/models
python scripts/evaluate.py --data $dataPath --config configs/any_aki.yaml --model-dir outputs/models --output outputs/any_aki_retrained_metrics.csv
```

Run the repository checks:

```powershell
ruff check src scripts tests
ruff format --check src scripts tests
python -m unittest discover -s tests -v
```

`sample.xlsx` checks the schema. It cannot reproduce the study results. Use the private completed dataset for evaluation.

`--tune` runs the five-fold parameter search. It is not part of the standard run.
