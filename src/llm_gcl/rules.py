from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import EPS, TARGET_COLUMN
from .metrics import clip_probability


def shift_probability(probability: np.ndarray, threshold: float) -> np.ndarray:
    values = clip_probability(probability)
    threshold = float(np.clip(threshold, EPS, 1.0 - EPS))
    shifted = np.log(values / (1.0 - values)) - np.log(threshold / (1.0 - threshold))
    return 1.0 / (1.0 + np.exp(-shifted))


def segmented_probability(probability: np.ndarray, bias: np.ndarray, delta: float, t_lr: float, t_mid: float, t_xgb: float) -> tuple[np.ndarray, np.ndarray]:
    adjusted = np.zeros(len(probability), dtype=float)
    region = np.full(len(probability), "mid", dtype=object)
    lr_mask = bias >= delta
    xgb_mask = bias <= -delta
    mid_mask = ~(lr_mask | xgb_mask)
    adjusted[lr_mask] = shift_probability(probability[lr_mask], t_lr)
    adjusted[mid_mask] = shift_probability(probability[mid_mask], t_mid)
    adjusted[xgb_mask] = shift_probability(probability[xgb_mask], t_xgb)
    region[lr_mask] = "lr"
    region[xgb_mask] = "xgb"
    return adjusted, region


def group_probability(base_probability: np.ndarray, mask: np.ndarray, group_threshold: float, other_threshold: float) -> np.ndarray:
    adjusted = np.zeros(len(base_probability), dtype=float)
    adjusted[mask] = shift_probability(base_probability[mask], group_threshold)
    adjusted[~mask] = shift_probability(base_probability[~mask], other_threshold)
    return adjusted


def three_group_probability(base_probability: np.ndarray, primary_mask: np.ndarray, aux_mask: np.ndarray, primary_aux_threshold: float, primary_base_threshold: float, other_threshold: float) -> np.ndarray:
    adjusted = np.zeros(len(base_probability), dtype=float)
    primary_aux = primary_mask & aux_mask
    primary_only = primary_mask & ~aux_mask
    outside = ~primary_mask
    adjusted[primary_aux] = shift_probability(base_probability[primary_aux], primary_aux_threshold)
    adjusted[primary_only] = shift_probability(base_probability[primary_only], primary_base_threshold)
    adjusted[outside] = shift_probability(base_probability[outside], other_threshold)
    return adjusted


def prediction_frame(dataset: str, y_true: np.ndarray, probability: np.ndarray, threshold: float) -> pd.DataFrame:
    frame = pd.DataFrame({"dataset": dataset, "y_true": np.asarray(y_true, dtype=int), "y_prob": clip_probability(probability), "selected_threshold": float(threshold)})
    frame["y_pred_threshold_opt"] = (frame["y_prob"] >= float(threshold)).astype(int)
    return frame


def attach_predictions(meta: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    result = meta.copy().reset_index(drop=True)
    aligned = predictions.reset_index(drop=True)
    for column in aligned.columns:
        if column != "dataset":
            result[f"pred_{column}"] = aligned[column].to_numpy()
    result["error_type"] = "TN"
    predicted = result["pred_y_pred_threshold_opt"]
    result.loc[(result[TARGET_COLUMN] == 1) & (predicted == 1), "error_type"] = "TP"
    result.loc[(result[TARGET_COLUMN] == 0) & (predicted == 1), "error_type"] = "FP"
    result.loc[(result[TARGET_COLUMN] == 1) & (predicted == 0), "error_type"] = "FN"
    return result


def build_rule_label(feature: str, rule_type: str, value: Any) -> str:
    if rule_type == "numeric_gt":
        return f"{feature} > {float(value):.4f}"
    if rule_type == "numeric_le":
        return f"{feature} <= {float(value):.4f}"
    if rule_type == "categorical_eq":
        return f"{feature} == {value}"
    if rule_type == "categorical_ne":
        return f"{feature} != {value}"
    raise ValueError(f"不支持的规则类型：{rule_type}")


def apply_rule(frame: pd.DataFrame, rule: dict[str, Any]) -> np.ndarray:
    feature = str(rule["feature"])
    rule_type = str(rule["rule_type"])
    if rule_type == "numeric_gt":
        return (pd.to_numeric(frame[feature], errors="coerce") > float(rule["split_value"])).to_numpy(dtype=bool)
    if rule_type == "numeric_le":
        return (pd.to_numeric(frame[feature], errors="coerce") <= float(rule["split_value"])).to_numpy(dtype=bool)
    values = frame[feature].astype("string").fillna("missing")
    if rule_type == "categorical_eq":
        return (values == str(rule["split_value"])).to_numpy(dtype=bool)
    if rule_type == "categorical_ne":
        return (values != str(rule["split_value"])).to_numpy(dtype=bool)
    raise ValueError(f"不支持的规则类型：{rule_type}")


def _numeric_error_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in frame.columns:
        if feature == TARGET_COLUMN or feature.startswith("pred_") or feature == "error_type" or not pd.api.types.is_numeric_dtype(frame[feature]):
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        standard_deviation = float(values.std(ddof=0)) if values.notna().sum() > 1 else 0.0
        if standard_deviation == 0.0:
            continue
        for comparison, group_a_name, group_b_name in (("FN_vs_TP", "FN", "TP"), ("FP_vs_TN", "FP", "TN")):
            group_a = values[frame["error_type"] == group_a_name].dropna()
            group_b = values[frame["error_type"] == group_b_name].dropna()
            if len(group_a) >= 3 and len(group_b) >= 3:
                difference = float(group_a.mean() - group_b.mean())
                rows.append({"comparison": comparison, "feature": feature, "mean_diff": difference, "abs_std_diff": abs(difference) / standard_deviation})
    if not rows:
        return pd.DataFrame(columns=["comparison", "feature", "mean_diff", "abs_std_diff"])
    return pd.DataFrame(rows).sort_values(by=["comparison", "abs_std_diff"], ascending=[True, False]).reset_index(drop=True)


def _categorical_error_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in frame.columns:
        if feature == TARGET_COLUMN or feature.startswith("pred_") or feature == "error_type" or pd.api.types.is_numeric_dtype(frame[feature]):
            continue
        values = frame[feature].astype("string").fillna("missing")
        if values.nunique() == 0 or values.nunique() > 10:
            continue
        for comparison, group_a_name, group_b_name in (("FN_vs_TP", "FN", "TP"), ("FP_vs_TN", "FP", "TN")):
            group_a = values[frame["error_type"] == group_a_name]
            group_b = values[frame["error_type"] == group_b_name]
            if len(group_a) < 3 or len(group_b) < 3:
                continue
            share_a = group_a.value_counts(normalize=True)
            share_b = group_b.value_counts(normalize=True)
            for category in sorted(set(share_a.index) | set(share_b.index)):
                a_value = float(share_a.get(category, 0.0))
                b_value = float(share_b.get(category, 0.0))
                rows.append({"comparison": comparison, "feature": feature, "category": category, "group_a_share": a_value, "group_b_share": b_value, "abs_share_diff": abs(a_value - b_value)})
    if not rows:
        return pd.DataFrame(columns=["comparison", "feature", "category", "group_a_share", "group_b_share", "abs_share_diff"])
    return pd.DataFrame(rows).sort_values(by=["comparison", "abs_share_diff"], ascending=[True, False]).reset_index(drop=True)


def candidate_rules(validation_frame: pd.DataFrame) -> list[dict[str, Any]]:
    numeric = _numeric_error_summary(validation_frame)
    categorical = _categorical_error_summary(validation_frame)
    rules: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for comparison in ("FN_vs_TP", "FP_vs_TN"):
        numeric_subset = numeric.loc[numeric["comparison"] == comparison]
        selected_features: list[tuple[str, float]] = []
        for _, row in numeric_subset.iterrows():
            feature = str(row["feature"])
            if feature.startswith("pred_") or feature not in validation_frame.columns or any(item[0] == feature for item in selected_features):
                continue
            selected_features.append((feature, float(row["mean_diff"])))
            if len(selected_features) >= 6:
                break
        for feature, difference in selected_features:
            values = pd.to_numeric(validation_frame[feature], errors="coerce")
            thresholds = sorted({float(values.quantile(quantile)) for quantile in (0.2, 0.35, 0.5, 0.65, 0.8) if values.notna().sum() > 0})
            rule_type = "numeric_gt" if difference >= 0.0 else "numeric_le"
            for value in thresholds:
                key = (comparison, feature, rule_type, value)
                if np.isnan(value) or key in seen:
                    continue
                seen.add(key)
                rules.append({"feature": feature, "rule_type": rule_type, "split_value": value, "rule_label": build_rule_label(feature, rule_type, value), "source_comparison": comparison})
        categorical_subset = categorical.loc[categorical["comparison"] == comparison]
        selected_pairs: list[tuple[str, str, float, float]] = []
        for _, row in categorical_subset.iterrows():
            pair = (str(row["feature"]), str(row["category"]))
            if pair[0].startswith("pred_") or pair[0] not in validation_frame.columns or any(item[:2] == pair for item in selected_pairs):
                continue
            selected_pairs.append((pair[0], pair[1], float(row["group_a_share"]), float(row["group_b_share"])))
            if len(selected_pairs) >= 6:
                break
        for feature, category, share_a, share_b in selected_pairs:
            rule_type = "categorical_eq" if share_a >= share_b else "categorical_ne"
            key = (comparison, feature, rule_type, category)
            if key not in seen:
                seen.add(key)
                rules.append({"feature": feature, "rule_type": rule_type, "split_value": category, "rule_label": build_rule_label(feature, rule_type, category), "source_comparison": comparison})
    return rules


def local_rule_variants(frame: pd.DataFrame, rule: dict[str, Any], max_variants: int = 9) -> list[dict[str, Any]]:
    if not str(rule["rule_type"]).startswith("numeric_"):
        return [rule]
    feature = str(rule["feature"])
    values = pd.to_numeric(frame[feature], errors="coerce").dropna().sort_values().unique()
    if len(values) == 0:
        return [rule]
    index = int(np.searchsorted(values, float(rule["split_value"])))
    radius = max(1, max_variants // 2)
    variants = []
    for value in values[max(0, index - radius):min(len(values), index + radius + 1)]:
        local = dict(rule)
        local["split_value"] = float(value)
        local["rule_label"] = build_rule_label(feature, str(rule["rule_type"]), float(value))
        variants.append(local)
    return variants or [rule]
