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
from .rules import apply_rule, group_probability, segmented_probability, three_group_probability


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
        terminal = self.terminal_stage
        aligned = prepare_features(frame, self.feature_columns)
        weights = self.config["upstream"]["weights"]
        xgb = self._family_probability(aligned, "xgb")
        lr = self._family_probability(aligned, "lr")
        upstream = (
            float(weights["xgb"]) * xgb
            + float(weights["lr"]) * lr
            + float(weights["cat"]) * self._family_probability(aligned, "cat")
        )
        stages = {"upstream": upstream}
        if terminal == "upstream":
            return stages

        regional = self.config["round8"]
        if regional["method"] == "identity_no_round8":
            round8 = upstream
        elif regional["method"] == "bayesian_optimization":
            round8, _ = segmented_probability(
                upstream, lr - xgb,
                **{name: float(regional[name]) for name in ("delta", "t_lr", "t_mid", "t_xgb")},
            )
        else:
            raise ValueError(f"不支持的 Round8 方法：{regional['method']}")
        stages["round8"] = round8
        if terminal == "round8":
            return stages

        if terminal == "round9":
            config = self.config["round9"]
            stages["round9"] = group_probability(
                round8, apply_rule(frame, config["rule"]),
                float(config["threshold_group"]), float(config["threshold_other"]),
            )
        else:
            config = self.config["round10_candidate"]
            # Training fits Round10 on the retained Round8 output, not on Round9.
            stages["round10"] = three_group_probability(
                round8,
                apply_rule(frame, config["primary_rule"]),
                apply_rule(frame, config["aux_rule"]),
                float(config["threshold_primary_aux"]),
                float(config["threshold_primary_base"]),
                float(config["threshold_other"]),
            )
        return stages

    @property
    def terminal_stage(self) -> str:
        terminal = self.config["evaluation_terminal_stage"]
        if terminal not in {"upstream", "round8", "round9", "round10"}:
            raise ValueError(f"不支持的终止阶段：{terminal}")
        return terminal

    @property
    def threshold(self) -> float:
        terminal = self.terminal_stage
        key = "round10_candidate" if terminal == "round10" else terminal
        return float(self.config[key]["threshold"])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_stages(frame)[self.terminal_stage]

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
