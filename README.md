# LLM-GCL

LLM-GCL is a cascaded ensemble for imbalanced binary classification on small clinical datasets.

## Method

The training pipeline fits candidate XGBoost, logistic regression, and CatBoost models. A multi-metric guardrail selects the base members, then Bayesian optimization sets their ensemble weights.

The later stages calibrate probabilities using disagreement between logistic regression and XGBoost, inspect validation errors, and tune local correction rules and the final decision threshold.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Train

```powershell
python run.py train `
  --data "D:\path\to\train.csv" `
  --external-data "D:\path\to\external.csv" `
  --output-dir "D:\path\to\model_output"
```

The training CSV must contain a binary target column named `左旋标签`. External validation is optional, so you can omit `--external-data`.

Training writes three files to the output directory: the fitted model, its configuration, and a training summary. Add `--strict-signature` if you need an exact check against the paper configuration.

## Predict

```powershell
python run.py predict `
  --model "D:\path\to\model_output\llm_gcl_model.joblib" `
  --data "D:\path\to\new_data.csv" `
  --output "D:\path\to\predictions.csv"
```
