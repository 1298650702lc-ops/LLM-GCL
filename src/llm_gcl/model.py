from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .base_models import predict_bundle
from .data import prepare_features
from .metrics import evaluate_predictions, summarize_metrics
from .rules import apply_rule, segmented_probability, three_group_probability


@dataclass
class LLMGCLModel:
    feature_columns: list[str]
    members: dict[str, dict[str, Any]]
    config: dict[str, Any]

    def _component_probabilities(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        aligned = prepare_features(frame, self.feature_columns)
        xgb = predict_bundle(self.members["xgb_member_03"], aligned)
        lr = predict_bundle(self.members["lr_member_03"], aligned)
        cat = predict_bundle(self.members["cat_member_03"], aligned)
        return xgb, lr, cat

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        aligned = prepare_features(frame, self.feature_columns)
        xgb, lr, cat = self._component_probabilities(aligned)
        weights = self.config["upstream"]["weights"]
        upstream = weights["xgb_weight"] * xgb + weights["lr_weight"] * lr + weights["cat_weight"] * cat
        round8 = self.config["round8"]
        regional, _ = segmented_probability(upstream, lr - xgb, float(round8["delta"]), float(round8["t_lr"]), float(round8["t_mid"]), float(round8["t_xgb"]))
        round10 = self.config["round10"]
        return three_group_probability(
            regional,
            apply_rule(frame, round10["primary_rule"]),
            apply_rule(frame, round10["aux_rule"]),
            float(round10["threshold_primary_aux"]),
            float(round10["threshold_primary_base"]),
            float(round10["threshold_other"]),
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame) >= float(self.config["round10"]["threshold"])).astype(int)

    def evaluate(self, frame: pd.DataFrame, y_true: np.ndarray) -> dict[str, Any]:
        return summarize_metrics(evaluate_predictions(y_true, self.predict_proba(frame), float(self.config["round10"]["threshold"])))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> "LLMGCLModel":
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError("文件不是 LLMGCLModel 模型")
        return model
