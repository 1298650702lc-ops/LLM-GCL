# LLM-GCL

## Project Overview

LLM-GCL is a standalone cascaded ensemble learning model designed for imbalanced binary classification tasks on small clinical datasets.

This repository provides complete code for model training, parameter optimization, stage gating, inference, and result persistence. The code runs independently and does not rely on source code or data from other project directories.

The current version is **2.0.0**, implementing the final **4:4:1:1 data split and cascaded model configuration** used in the paper.

## Key Features

- Splits the internal dataset into mutually exclusive training, tuning, validation, and test sets;
- Uses the validation set to select base model members;
- Uses the tuning set for rule discovery, Bayesian optimization, classification-threshold selection, and stage gating;
- Evaluates performance on the test set and the optional external validation set only after the configuration has been frozen;
- Uses one XGBoost model and one logistic regression model in the final upstream ensemble;
- Uses Round 9 error-pattern correction as the final cascade stage;
- Supports strict reproduction of the frozen 4:4:1:1 model configuration;
- Supports saving trained models and independently generating predictions for new CSV data.

## Dataset Information

### Data Format

Training data must be provided as a CSV file and meet the following requirements:

| Item | Requirement |
|---|---|
| Target column | Must contain a column named `左旋标签` |
| Target values | Must contain only `0` and `1`, with both classes represented |
| ID column | A column named `id` is excluded from the model features |
| Feature types | Both numerical and categorical features are supported |
| File encoding | `UTF-8 BOM`, `UTF-8`, `GB18030`, and `GBK` are supported |
| Missing values | Handled uniformly by the model's data preprocessing pipeline |
| External validation set | Optional, but it must contain the same target column and compatible features |

### 4:4:1:1 Data Split

The current frozen experiment contains 844 internal samples, divided as follows:

| Data Subset | Number of Samples | Primary Purpose |
|---|---:|---|
| Training set | 337 | Train the base models |
| Tuning set | 337 | Rule discovery, Bayesian optimization, threshold selection, and stage gating |
| Validation set | 85 | Base model member selection and generalization constraints |
| Test set | 85 | Final internal evaluation after configuration freezing |

All subsets are mutually exclusive. Stratified random splitting is used to preserve the class distribution across subsets as closely as possible.

## Code Information

### Project Structure

| Path | Description |
|---|---|
| `run.py` | Local command-line entry point |
| `src/llm_gcl/cli.py` | Training and prediction command definitions |
| `src/llm_gcl/config.py` | Model parameters, random seeds, and frozen configuration |
| `src/llm_gcl/data.py` | Data loading, type conversion, and dataset splitting |
| `src/llm_gcl/base_models.py` | Base classifier implementations |
| `src/llm_gcl/selection.py` | Base model member selection |
| `src/llm_gcl/bayes.py` | Bayesian optimization implementation |
| `src/llm_gcl/rules.py` | Error-pattern and cascade rules |
| `src/llm_gcl/training.py` | Complete training and stage-gating pipeline |
| `src/llm_gcl/model.py` | Model saving, loading, and inference |
| `src/llm_gcl/metrics.py` | Classification metric computation |
| `Result/result_4411/` | Frozen results from the final 4:4:1:1 experiment |
| `Result/paper_experiment_results/` | Supporting experimental results used in the paper |

## Requirements

- Python 3.11 or later;
- An isolated virtual environment is recommended;
- Sufficient memory and computing resources are required for training.

The main dependencies are listed below:

| Dependency | Version |
|---|---:|
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.6.1 |
| XGBoost | 3.1.3 |
| CatBoost | 1.2.10 |
| joblib | 1.4.2 |

## Installation

After cloning the repository and entering the project directory, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

You can also install the project as a local Python package:

```powershell
python -m pip install .
```

After installation, you can use the `llm-gcl` command or run the project directly with `python run.py`.

## Usage

### Training a Model

```powershell
python run.py train `
  --data "D:\path\to\train.csv" `
  --external-data "D:\path\to\external.csv" `
  --output-dir "D:\path\to\model_output"
```

Arguments:

| Argument | Required | Description |
|---|---|---|
| `--data` | Yes | Internal training data containing the `左旋标签` target column |
| `--output-dir` | Yes | Output directory for the model and training results |
| `--external-data` | No | Labeled CSV file used for final external validation |
| `--strict-signature` | No | Requires the model members, rules, weights, and thresholds to strictly reproduce the frozen configuration |

The output directory should be located outside the source directory to keep generated model artifacts separate from the code.

Training generates the following files:

| File | Description |
|---|---|
| `llm_gcl_model.joblib` | Fitted base members and the final inference configuration |
| `llm_gcl_config.json` | Frozen 4:4:1:1 configuration, data roles, and stage decisions |
| `training_summary.json` | Metrics for the tuning, validation, test, and optional external validation sets |

To strictly reproduce the frozen model, add `--strict-signature`:

```powershell
python run.py train `
  --data "D:\path\to\train.csv" `
  --output-dir "D:\path\to\model_output" `
  --strict-signature
```

The training process terminates if the generated model members, rules, or continuous parameters do not match the frozen 4:4:1:1 configuration.

### Generating Predictions

```powershell
python run.py predict `
  --model "D:\path\to\model_output\llm_gcl_model.joblib" `
  --data "D:\path\to\new_data.csv" `
  --output "D:\path\to\predictions.csv"
```

Prediction data does not need to contain the target column, but it should provide features consistent with those used during training whenever possible. Missing feature columns are added and filled with missing values.

The output file contains the following fields:

| Field | Description |
|---|---|
| `y_prob` | Predicted probability of the positive class |
| `y_pred` | Binary prediction based on the frozen threshold |
| `threshold` | Classification threshold used for the prediction |

Prediction requires only the installed `llm_gcl` package and the specified model file. It does not read from any other project path outside this repository.

## Methodology

The model training and evaluation workflow is as follows:

1. **Data Loading and Validation**  
   The CSV file is loaded, the `左旋标签` column is verified as a binary target containing both `0` and `1`, and numerical and categorical features are identified.

2. **Stratified Data Splitting**  
   The internal data is divided into training, tuning, validation, and test sets using a 4:4:1:1 ratio. Each subset has a fixed role, and the subsets are never mixed.

3. **Base Model Training**  
   Multiple XGBoost, logistic regression, and CatBoost candidate members are trained using different random seeds and hyperparameter configurations.

4. **Base Member Selection**  
   The validation set is used to enforce generalization constraints and select base model members. The final frozen upstream ensemble contains one XGBoost member and one logistic regression member.

5. **Parameter and Threshold Optimization**  
   The tuning set alone is used for Bayesian optimization, ensemble-weight search, classification-threshold selection, and error-pattern rule discovery.

6. **Cascade Stage Gating**  
   Candidate stages are evaluated using multiple random seeds and bootstrap resampling. A stage is included in the final model only if it satisfies the required objective gain, improvement probability, random-seed support, and per-metric degradation constraints.

7. **Protected Final Evaluation**  
   The test set and optional external dataset are used for final evaluation only after the entire model configuration has been frozen, reducing the risk of test-set information leakage.

Model performance is evaluated using metrics including ROC AUC, average precision (AP), F2 score, Matthews correlation coefficient (MCC), and recall.

## Experimental Results

`Result/result_4411/` contains the frozen final experimental results.

`Result/paper_experiment_results/` contains the supporting experimental outputs used in the paper.

To protect clinical data, the results directories do not contain raw datasets or media files.
