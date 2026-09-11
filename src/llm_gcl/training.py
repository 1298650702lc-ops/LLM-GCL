from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base_models import train_base_pool
from .bayes import identity_round8, run_round10, run_round8, run_round9, run_upstream
from .config import (
    BO_RANDOM_STATE,
    FROZEN_4411_SIGNATURE,
    STAGE_GATE_BOOTSTRAPS,
    STAGE_GATE_METRIC_DROP_MARGIN,
    STAGE_GATE_METRIC_DROP_PROBABILITY,
    STAGE_GATE_MIN_IMPROVEMENT_PROBABILITY,
    STAGE_GATE_MIN_OBJECTIVE_GAIN,
    STAGE_GATE_MIN_SEED_SUPPORT,
    STAGE_GATE_SEED_OFFSETS,
    TARGET_COLUMN,
)
from .data import SplitBundle, build_split_4411, load_labeled_dataframe, read_csv
from .metrics import evaluate_predictions, metric_objective, summarize_metrics
from .model import LLMGCLModel
from .rules import attach_predictions, candidate_rules
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
        raise RuntimeError(f"4411 签名不一致：{label}，实际={actual}，冻结值={expected}")


def verify_4411_signature(config: dict[str, Any]) -> None:
    selected = config["selected_member_ids"]
    member_ids = list(selected["xgb"]) + list(selected["lr"]) + list(selected["cat"])
    if member_ids != FROZEN_4411_SIGNATURE["selected_member_ids"]:
        raise RuntimeError(f"4411 成员签名不一致：实际={member_ids}，冻结值={FROZEN_4411_SIGNATURE['selected_member_ids']}")
    if config["evaluation_terminal_stage"] != FROZEN_4411_SIGNATURE["evaluation_terminal_stage"]:
        raise RuntimeError(
            f"4411 终止阶段不一致：实际={config['evaluation_terminal_stage']}，"
            f"冻结值={FROZEN_4411_SIGNATURE['evaluation_terminal_stage']}"
        )
    for name in ("xgb", "lr", "cat"):
        _assert_close(config["upstream"]["weights"][name], FROZEN_4411_SIGNATURE["upstream"][name], f"upstream.{name}")
    _assert_close(config["upstream"]["threshold"], FROZEN_4411_SIGNATURE["upstream"]["threshold"], "upstream.threshold")
    if config["round8"]["method"] != FROZEN_4411_SIGNATURE["round8"]["method"]:
        raise RuntimeError(f"4411 Round8 决策不一致：{config['round8']['method']}")
    expected = FROZEN_4411_SIGNATURE["round9"]
    actual = config["round9"]
    for name in ("feature", "rule_type"):
        if actual["rule"][name] != expected[name]:
            raise RuntimeError(f"4411 Round9 规则不一致：{actual['rule']}")
    for name in ("split_value", "threshold_group", "threshold_other", "threshold"):
        value = actual["rule"][name] if name == "split_value" else actual[name]
        _assert_close(value, expected[name], f"round9.{name}")


def _split_sizes(split: SplitBundle) -> dict[str, int]:
    return {
        "train": len(split.train_meta),
        "tuning": len(split.tuning_meta),
        "validation": len(split.validation_meta),
        "test": len(split.test_meta),
    }


def _stage_probability(stage: dict[str, Any], role: str) -> np.ndarray:
    frame = stage["frames"][role]
    column = "y_prob" if "y_prob" in frame.columns else "ensemble_prob"
    return frame[column].to_numpy(dtype=float)


def _stage_metrics(stage: dict[str, Any], role: str = "tuning") -> dict[str, Any]:
    frame = stage["frames"][role]
    return summarize_metrics(
        evaluate_predictions(
            frame["y_true"].to_numpy(dtype=int),
            _stage_probability(stage, role),
            float(stage["config"]["threshold"]),
        )
    )


def _evaluate_stage_gate(
    stage_name: str,
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
    bootstrap_seed: int,
) -> dict[str, Any]:
    baseline_frame = baseline["frames"]["tuning"]
    candidate_frame = candidates[0]["frames"]["tuning"]
    labels = baseline_frame["y_true"].to_numpy(dtype=int)
    if not np.array_equal(labels, candidate_frame["y_true"].to_numpy(dtype=int)):
        raise AssertionError(f"{stage_name} 阶段门控数据未对齐")
    baseline_metrics = _stage_metrics(baseline)
    candidate_metrics = _stage_metrics(candidates[0])
    baseline_objective = metric_objective(baseline_metrics)
    candidate_objective = metric_objective(candidate_metrics)
    objective_gain = candidate_objective - baseline_objective
    seed_rows = []
    for index, candidate in enumerate(candidates):
        metrics = _stage_metrics(candidate)
        objective = metric_objective(metrics)
        seed_rows.append(
            {
                "seed_index": index,
                "objective": objective,
                "objective_gain": objective - baseline_objective,
                **{name: metrics[name] for name in ("ROC AUC", "AP", "F2", "MCC", "Recall")},
            }
        )
    seed_support = sum(float(row["objective_gain"]) > 0.0 for row in seed_rows)

    rng = np.random.default_rng(bootstrap_seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    baseline_probability = _stage_probability(baseline, "tuning")
    candidate_probability = _stage_probability(candidates[0], "tuning")
    metric_names = ("ROC AUC", "AP", "F2", "MCC")
    deltas: dict[str, list[float]] = {"objective": [], **{name: [] for name in metric_names}}
    for _ in range(STAGE_GATE_BOOTSTRAPS):
        indices = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        rng.shuffle(indices)
        sample_y = labels[indices]
        baseline_sample = summarize_metrics(
            evaluate_predictions(sample_y, baseline_probability[indices], float(baseline["config"]["threshold"]))
        )
        candidate_sample = summarize_metrics(
            evaluate_predictions(sample_y, candidate_probability[indices], float(candidates[0]["config"]["threshold"]))
        )
        deltas["objective"].append(metric_objective(candidate_sample) - metric_objective(baseline_sample))
        for metric_name in metric_names:
            deltas[metric_name].append(float(candidate_sample[metric_name]) - float(baseline_sample[metric_name]))

    objective_deltas = np.asarray(deltas["objective"], dtype=float)
    improvement_probability = float(np.mean(objective_deltas > 0.0))
    metric_checks: dict[str, Any] = {}
    significant_degradation = False
    for metric_name in metric_names:
        values = np.asarray(deltas[metric_name], dtype=float)
        median_delta = float(np.median(values))
        degradation_probability = float(np.mean(values < 0.0))
        degraded = median_delta < -STAGE_GATE_METRIC_DROP_MARGIN and degradation_probability >= STAGE_GATE_METRIC_DROP_PROBABILITY
        significant_degradation = significant_degradation or degraded
        metric_checks[metric_name] = {
            "median_delta": median_delta,
            "degradation_probability": degradation_probability,
            "significant_degradation": degraded,
        }

    reasons: list[str] = []
    if objective_gain < STAGE_GATE_MIN_OBJECTIVE_GAIN:
        reasons.append("objective_gain_below_complexity_margin")
    if improvement_probability < STAGE_GATE_MIN_IMPROVEMENT_PROBABILITY:
        reasons.append("bootstrap_support_too_low")
    if seed_support < STAGE_GATE_MIN_SEED_SUPPORT:
        reasons.append("bo_seed_support_too_low")
    if significant_degradation:
        reasons.append("core_metric_degradation")
    accepted = not reasons
    return {
        "stage": stage_name,
        "accepted": accepted,
        "decision": "continue" if accepted else "reject_and_keep_simpler_stage",
        "reasons": reasons or ["all_pre_registered_gates_passed"],
        "baseline_objective": baseline_objective,
        "candidate_objective": candidate_objective,
        "objective_gain": objective_gain,
        "bootstrap_improvement_probability": improvement_probability,
        "bootstrap_objective_median_delta": float(np.median(objective_deltas)),
        "seed_support": seed_support,
        "seed_count": len(seed_rows),
        "seed_results": seed_rows,
        "metric_checks": metric_checks,
        "criteria": {
            "minimum_objective_gain": STAGE_GATE_MIN_OBJECTIVE_GAIN,
            "minimum_improvement_probability": STAGE_GATE_MIN_IMPROVEMENT_PROBABILITY,
            "minimum_seed_support": STAGE_GATE_MIN_SEED_SUPPORT,
            "metric_drop_margin": STAGE_GATE_METRIC_DROP_MARGIN,
            "metric_drop_probability": STAGE_GATE_METRIC_DROP_PROBABILITY,
            "bootstrap_replicates": STAGE_GATE_BOOTSTRAPS,
        },
    }


def _attached_frames(split: SplitBundle, stage: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {
        "train": attach_predictions(split.train_meta, stage["frames"]["train"]),
        "tuning": attach_predictions(split.tuning_meta, stage["frames"]["tuning"]),
    }


def train_formal_model(
    data_path: str | Path,
    output_dir: str | Path,
    external_data_path: str | Path | None = None,
    strict_signature: bool = False,
) -> tuple[LLMGCLModel, dict[str, Any]]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    data = load_labeled_dataframe(data_path)
    split = build_split_4411(data)
    sizes = _split_sizes(split)
    print(f"[split] 4:4:1:1 {sizes}")

    registry, pool = train_base_pool(split)
    xgb_lr_selection = select_xgb_lr(registry, pool)
    selection = select_cat(registry, pool, xgb_lr_selection)
    selected_ids = {
        "xgb": list(xgb_lr_selection["selected_xgb_member_ids"]),
        "lr": list(xgb_lr_selection["selected_lr_member_ids"]),
        "cat": list(selection["selected_cat_member_ids"]),
    }
    print(f"[guardrail] 验证集选中成员：{selected_ids}")

    selected_flat = selected_ids["xgb"] + selected_ids["lr"] + selected_ids["cat"]
    selection_for_frames = {
        "base_best_member_ids": selected_ids["xgb"] + selected_ids["lr"],
        "selected_cat_member_ids": selected_ids["cat"],
    }
    family_frames = raw_upstream_frames(pool, selection_for_frames, ["train", "tuning"])
    print("[BO] 调参集优化上游家族权重")
    upstream = run_upstream(family_frames)

    print("[gate] 评估 Round8 区域概率修正")
    round8_candidates = [run_round8(upstream, BO_RANDOM_STATE + 2 + offset) for offset in STAGE_GATE_SEED_OFFSETS]
    round8_decision = _evaluate_stage_gate("round8", upstream, round8_candidates, BO_RANDOM_STATE + 2002)
    round8 = round8_candidates[0] if round8_decision["accepted"] else identity_round8(upstream)

    split_frames = _attached_frames(split, round8)
    analysis_frame = split_frames["tuning"].drop(columns=["id"], errors="ignore")
    rules = [rule for rule in candidate_rules(analysis_frame) if rule["feature"] in split.feature_columns]
    if not rules:
        raise RuntimeError("调参集错误分析未产生候选规则")
    print(f"[gate] 评估 Round9 错误模式修正，候选规则={len(rules)}")
    round9_candidates = [run_round9(split_frames, rules, BO_RANDOM_STATE + 3 + offset) for offset in STAGE_GATE_SEED_OFFSETS]
    round9 = round9_candidates[0]
    round9_decision = _evaluate_stage_gate("round9", round8, round9_candidates, BO_RANDOM_STATE + 2003)

    primary_rules = ([round9["config"]["rule"]] if round9_decision["accepted"] else []) + rules[:20]
    print("[gate] 评估 Round10 局部概率修正")
    round10_candidates = [run_round10(split_frames, primary_rules, rules, BO_RANDOM_STATE + 4 + offset) for offset in STAGE_GATE_SEED_OFFSETS]
    round10 = round10_candidates[0]
    round10_baseline = round9 if round9_decision["accepted"] else round8
    round10_decision = _evaluate_stage_gate("round10", round10_baseline, round10_candidates, BO_RANDOM_STATE + 2004)

    terminal_stage = "round8" if round8_decision["accepted"] else "upstream"
    if round9_decision["accepted"]:
        terminal_stage = "round9"
    if round10_decision["accepted"]:
        terminal_stage = "round10"
    if terminal_stage != "round9":
        raise RuntimeError(f"当前训练未得到 4411 最终结构，终止阶段={terminal_stage}")

    config = {
        "model": "LLM-GCL",
        "version": "2.0.0-4411",
        "experiment": "4411",
        "data_roles": {
            "base_fit": "train",
            "guardrail_member_selection": "validation",
            "rule_candidate_discovery": "tuning",
            "all_bayesian_search": "tuning",
            "protected_evaluation": ["test", "external"],
        },
        "split_sizes": sizes,
        "selected_member_ids": selected_ids,
        "evaluation_terminal_stage": terminal_stage,
        "stage_decisions": [round8_decision, round9_decision, round10_decision],
        "upstream": upstream["config"],
        "round8": round8["config"],
        "round9": round9["config"],
        "round10_candidate": round10["config"],
    }
    if strict_signature:
        verify_4411_signature(config)

    model = LLMGCLModel(
        feature_columns=split.feature_columns,
        members={member_id: pool[member_id]["bundle"] for member_id in selected_flat},
        config=config,
    )
    model_path = destination / "llm_gcl_model.joblib"
    config_path = destination / "llm_gcl_config.json"
    model.save(model_path)
    _write_json(config_path, config)

    summary: dict[str, Any] = {
        "model": "LLM-GCL",
        "version": "2.0.0-4411",
        "selected_member_ids": selected_ids,
        "evaluation_terminal_stage": terminal_stage,
        "split_sizes": sizes,
        "candidate_rule_count": len(rules),
        "stage_decisions": config["stage_decisions"],
        "tuning_metrics": round9["config"]["tuning_metrics"],
        "validation_metrics": model.evaluate(split.validation_meta, split.y_validation.to_numpy(dtype=int)),
        "test_metrics": model.evaluate(split.test_meta, split.y_test.to_numpy(dtype=int)),
        "strict_signature_checked": strict_signature,
    }
    if external_data_path is not None:
        external = read_csv(external_data_path).copy()
        if TARGET_COLUMN not in external.columns:
            raise KeyError(f"外部数据缺少标签列：{TARGET_COLUMN}")
        external[TARGET_COLUMN] = pd.to_numeric(external[TARGET_COLUMN], errors="raise").astype(int)
        summary["external_metrics"] = model.evaluate(external, external[TARGET_COLUMN].to_numpy(dtype=int))
    _write_json(destination / "training_summary.json", summary)
    print(f"[完成] 4411 模型已保存到：{model_path}")
    return model, summary
