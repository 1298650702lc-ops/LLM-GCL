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
from .rules import apply_rule, group_probability


@dataclass
class LLMGCLModel:
    feature_columns: list[str]
    members: dict[str, dict[str, Any]]
    config: dict[str, Any]

    def _family_probability(self, frame: pd.DataFrame, family: str) -> np.ndarray:
        member_ids = list(self.config["selected_member_ids"].get(family, []))
        if not member_ids:
            return np.zeros(len(frame), dtype=float)
        return np.mean(
            np.column_stack([predict_bundle(self.members[member_id], frame) for member_id in member_ids]),
            axis=1,
        )

    def predict_stages(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        aligned = prepare_features(frame, self.feature_columns)
        weights = self.config["upstream"]["weights"]
        upstream = (
            float(weights["xgb"]) * self._family_probability(aligned, "xgb")
            + float(weights["lr"]) * self._family_probability(aligned, "lr")
            + float(weights["cat"]) * self._family_probability(aligned, "cat")
        )
        round9 = self.config["round9"]
        corrected = group_probability(
            upstream,
            apply_rule(frame, round9["rule"]),
            float(round9["threshold_group"]),
            float(round9["threshold_other"]),
        )
        return {"upstream": upstream, "round9": corrected}

    @property
    def threshold(self) -> float:
        return float(self.config["round9"]["threshold"])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_stages(frame)["round9"]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame) >= self.threshold).astype(int)

    def evaluate(self, frame: pd.DataFrame, y_true: np.ndarray) -> dict[str, Any]:
        return summarize_metrics(evaluate_predictions(y_true, self.predict_proba(frame), self.threshold))

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
