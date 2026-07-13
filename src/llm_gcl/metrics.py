from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

from .config import AUC_RANKING, EPS, THRESHOLDS


METRIC_NAMES = ["ROC AUC", "AP", "Accuracy", "Balanced Acc", "Precision", "Recall", "Specificity", "F1", "F2", "MCC"]


def clip_probability(probability: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)


def safe_roc_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=int)
    return float("nan") if len(np.unique(labels)) < 2 else float(roc_auc_score(labels, clip_probability(probability)))


def safe_average_precision(y_true: np.ndarray, probability: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=int)
    return float("nan") if len(np.unique(labels)) < 2 else float(average_precision_score(labels, clip_probability(probability)))


def evaluate_predictions(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=int)
    probabilities = clip_probability(probability)
    predictions = (probabilities >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = float(precision_score(labels, predictions, zero_division=0))
    recall = float(recall_score(labels, predictions, zero_division=0))
    f2_denominator = 4.0 * precision + recall
    return {
        "threshold": float(threshold),
        "ROC AUC": safe_roc_auc(labels, probabilities),
        "AP": safe_average_precision(labels, probabilities),
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Balanced Acc": float(balanced_accuracy_score(labels, predictions)),
        "Precision": precision,
        "Recall": recall,
        "Specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "F2": float(5.0 * precision * recall / f2_denominator) if f2_denominator else 0.0,
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def threshold_scan(y_true: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    rows = [summarize_metrics(evaluate_predictions(y_true, probability, threshold)) for threshold in THRESHOLDS]
    return pd.DataFrame(rows).sort_values(by=["F2", "Recall", "MCC", "AP"], ascending=False).reset_index(drop=True)


def best_scanned_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {key: float(value) for key, value in threshold_scan(y_true, probability).iloc[0].to_dict().items()}


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    summary = {name: float(metrics[name]) for name in METRIC_NAMES if name in metrics}
    if "threshold" in metrics:
        summary["threshold"] = float(metrics["threshold"])
    return summary


def rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(row[name]) for name in AUC_RANKING)


def metric_objective(metrics: dict[str, Any]) -> float:
    return float(metrics["ROC AUC"]) + 0.20 * float(metrics["AP"]) + 0.10 * float(metrics["F2"]) + 0.05 * float(metrics["MCC"])
