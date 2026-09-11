# LLM-GCL

LLM-GCL is a standalone cascaded ensemble for imbalanced binary classification on small clinical datasets. This repository contains the complete training and inference implementation and does not import code or data from other project directories.

## Current model

Version 2.0 implements the final 4:4:1:1 model:

- mutually exclusive training, tuning, validation, and test sets;
- validation-only guardrail selection of base members;
- tuning-only rule discovery, Bayesian optimization, threshold selection, and stage gating;
- protected test and optional external evaluation after the configuration is fixed;
- one XGBoost member and one logistic-regression member in the final upstream ensemble;
- Round 9 error-pattern correction as the terminal stage after Round 8 and Round 10 candidates were rejected by the gate.

## Install

```powershell
python -m pip install -r requirements.txt
```

The command-line entry point can also be installed from this directory:

```powershell
python -m pip install .
```

## Train

```powershell
python run.py train `
  --data "D:\path\to\train.csv" `
  --external-data "D:\path\to\external.csv" `
  --output-dir "D:\path\to\model_output"
```

The internal CSV must contain the binary target column `Left-sided label`. External validation is optional. Training writes:

- `llm_gcl_model.joblib`: selected fitted members and the final inference configuration;
- `llm_gcl_config.json`: the frozen 4:4:1:1 configuration, data roles, and stage decisions;
- `training_summary.json`: tuning, validation, test, and optional external metrics.

Use `--strict-signature` to require exact reproduction of the frozen 4:4:1:1 members, rule, weights, and thresholds.

## Predict

```powershell
python run.py predict `
  --model "D:\path\to\model_output\llm_gcl_model.joblib" `
  --data "D:\path\to\new_data.csv" `
  --output "D:\path\to\predictions.csv"
```

Prediction requires only the installed `llm_gcl` package and the saved model file. It does not read any path outside this repository.

## Results

`Result/result_4411/` contains the frozen final run. `Result/paper_experiment_results/` contains the supporting experiment outputs used for the paper. Raw source datasets and media files are excluded.
