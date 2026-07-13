from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base_models import add_dataset_predictions, train_base_pool
from .bayes import run_round10, run_round8, run_round9, run_upstream
from .config import PAPER_SIGNATURE, TARGET_COLUMN
from .data import build_split_811, load_labeled_dataframe, prepare_features, read_csv
from .metrics import evaluate_predictions, summarize_metrics
from .model import LLMGCLModel
from .rules import attach_predictions, candidate_rules, paper_candidate_rules
from .selection import raw_upstream_frames, select_cat, select_xgb_lr


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_builtin(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-10) -> None:
    if not np.isclose(float(actual), float(expected), atol=tolerance, rtol=0.0):
        raise RuntimeError(f"论文签名不一致：{label}，实际={actual}，论文={expected}")


def verify_paper_signature(selection: dict[str, Any], summary: dict[str, Any]) -> None:
    selected_ids = list(selection["base_best_member_ids"]) + list(selection["selected_cat_member_ids"])
    if selected_ids != PAPER_SIGNATURE["selected_member_ids"]:
        raise RuntimeError(f"论文成员签名不一致：实际={selected_ids}，论文={PAPER_SIGNATURE['selected_member_ids']}")
    upstream = summary["upstream"]
    for name in ("xgb_weight", "lr_weight", "cat_weight"):
        _assert_close(upstream["weights"][name], PAPER_SIGNATURE["upstream"][name], f"upstream.{name}")
    _assert_close(upstream["threshold"], PAPER_SIGNATURE["upstream"]["threshold"], "upstream.threshold")
    round8 = summary["round8"]
    for name in ("delta", "t_lr", "t_mid", "t_xgb", "threshold"):
        _assert_close(round8[name], PAPER_SIGNATURE["round8"][name], f"round8.{name}")
    round9 = summary["round9"]
    expected_round9 = PAPER_SIGNATURE["round9"]
    if round9["rule"]["feature"] != expected_round9["feature"] or round9["rule"]["rule_type"] != expected_round9["rule_type"]:
        raise RuntimeError(f"论文 Round9 规则不一致：{round9['rule']}")
    for name in ("split_value", "threshold_group", "threshold_other", "threshold"):
        actual = round9["rule"][name] if name == "split_value" else round9[name]
        _assert_close(actual, expected_round9[name], f"round9.{name}")
    round10 = summary["round10"]
    expected_round10 = PAPER_SIGNATURE["round10"]
    rule_checks = (
        (round10["primary_rule"], "primary", expected_round10["primary_feature"], expected_round10["primary_rule_type"], expected_round10["primary_split_value"]),
        (round10["aux_rule"], "aux", expected_round10["aux_feature"], expected_round10["aux_rule_type"], expected_round10["aux_split_value"]),
    )
    for rule, label, feature, rule_type, value in rule_checks:
        if rule["feature"] != feature or rule["rule_type"] != rule_type:
            raise RuntimeError(f"论文 Round10 {label} 规则不一致：{rule}")
        _assert_close(rule["split_value"], value, f"round10.{label}.split_value")
    for name in ("threshold_primary_aux", "threshold_primary_base", "threshold_other", "threshold"):
        _assert_close(round10[name], expected_round10[name], f"round10.{name}")


def train_formal_model(data_path: str | Path, output_dir: str | Path, external_data_path: str | Path | None = None, strict_signature: bool = False) -> tuple[LLMGCLModel, dict[str, Any]]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    data = load_labeled_dataframe(data_path)
    split = build_split_811(data)
    registry, pool = train_base_pool(split)
    xgb_lr_selection = select_xgb_lr(registry, pool)
    selection = select_cat(registry, pool, xgb_lr_selection)
    selected_ids = list(selection["base_best_member_ids"]) + list(selection["selected_cat_member_ids"])
    print(f"[guardrail] 选中成员：{selected_ids}")
    if selected_ids != PAPER_SIGNATURE["selected_member_ids"]:
        raise RuntimeError(f"基础成员组合偏离论文正式模型：实际={selected_ids}，论文={PAPER_SIGNATURE['selected_member_ids']}")

    external_frame: pd.DataFrame | None = None
    external_features: pd.DataFrame | None = None
    splits = ["train", "train_oof", "val", "test"]
    if external_data_path is not None:
        external_frame = read_csv(external_data_path).copy()
        if TARGET_COLUMN not in external_frame.columns:
            raise KeyError(f"外部数据缺少标签列：{TARGET_COLUMN}")
        external_frame[TARGET_COLUMN] = pd.to_numeric(external_frame[TARGET_COLUMN], errors="raise").astype(int)
        external_features = prepare_features(external_frame, split.feature_columns)
        add_dataset_predictions(pool, "external", external_features, external_frame[TARGET_COLUMN].to_numpy(dtype=int))
        splits.append("external")

    raw_frames = raw_upstream_frames(pool, selection, splits)
    print("[BO] 上游三家族权重优化")
    upstream = run_upstream(raw_frames)
    print("[BO] Round8 分区概率优化")
    round8 = run_round8(upstream)
    split_frames = {
        "train": attach_predictions(split.train_meta, round8["frames"]["train"]),
        "val": attach_predictions(split.val_meta, round8["frames"]["val"]),
        "test": attach_predictions(split.test_meta, round8["frames"]["test"]),
    }
    discovered_rules = candidate_rules(split_frames["val"])
    rules = paper_candidate_rules(split_frames["val"])
    print(f"[错误分析] 当前重算候选={len(discovered_rules)}，论文冻结候选={len(rules)}")
    print("[BO] Round9 错误模式规则优化")
    round9 = run_round9(split_frames, rules)
    print("[BO] Round10 主规则/辅助规则局部优化")
    round10 = run_round10(split_frames, [round9["config"]["rule"], *rules[:20]], rules)
    summary = {"model": "LLM-GCL", "selection": {"xgb_lr": xgb_lr_selection, "three_family": selection}, "error_analysis": {"recomputed_candidate_count": len(discovered_rules), "paper_candidate_count": len(rules)}, "upstream": upstream["config"], "round8": round8["config"], "round9": round9["config"], "round10": round10["config"]}

    selected_members = {member_id: pool[member_id]["bundle"] for member_id in PAPER_SIGNATURE["selected_member_ids"]}
    model = LLMGCLModel(feature_columns=split.feature_columns, members=selected_members, config={"upstream": upstream["config"], "round8": round8["config"], "round9": round9["config"], "round10": round10["config"]})
    if external_frame is not None and external_features is not None:
        external_metrics = model.evaluate(external_frame, external_frame[TARGET_COLUMN].to_numpy(dtype=int))
        summary["round10"]["external_metrics"] = external_metrics
    summary["paper_compatible"] = {
        "selected_member_ids_match": selected_ids == PAPER_SIGNATURE["selected_member_ids"],
        "uses_bayesian_upstream": summary["upstream"].get("method") == "bayesian_optimization",
        "uses_bayesian_round8": summary["round8"].get("method") == "bayesian_optimization",
        "uses_bayesian_round9": summary["round9"].get("method") == "bayesian_optimization",
        "uses_bayesian_round10": summary["round10"].get("method") == "bayesian_optimization",
        "round10_primary_rule": summary["round10"]["primary_rule"],
        "round10_aux_rule": summary["round10"]["aux_rule"],
    }
    if strict_signature:
        verify_paper_signature(selection, summary)
        summary["paper_signature_verified"] = True
        print("[核验] 已复现论文正式模型签名")
    else:
        summary["paper_signature_verified"] = None
        print("[核验] 结构与阶段核验通过；数值差异按宽松模式记录，不中止训练")
    model.save(destination / "llm_gcl_model.joblib")
    _write_json(destination / "model_config.json", model.config)
    _write_json(destination / "training_summary.json", summary)
    print(f"[完成] 模型已保存到：{destination}")
    return model, summary
