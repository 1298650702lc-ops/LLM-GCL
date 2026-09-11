from __future__ import annotations

from typing import Any


TARGET_COLUMN = "左旋标签"
ID_COLUMNS = {"id"}
RANDOM_STATE = 20260417
SPLIT_RANDOM_STATE = 20260419
BO_RANDOM_STATE = 20260623
N_SPLITS = 5
EARLY_STOPPING_VALID_SIZE = 0.15
EARLY_STOPPING_ROUNDS = 50
BASE_ITERATIONS = 1000
BASE_N_ESTIMATORS = 1000
THRESHOLDS = [round(index / 100, 2) for index in range(5, 96)]
EPS = 1e-6
BO_INIT_POINTS = 12
BO_ITERATIONS = 48
BO_CANDIDATES = 512
MIN_GROUP_SIZE = 5

STAGE_GATE_SEED_OFFSETS = (0, 100, 200)
STAGE_GATE_BOOTSTRAPS = 1000
STAGE_GATE_MIN_OBJECTIVE_GAIN = 0.002
STAGE_GATE_MIN_IMPROVEMENT_PROBABILITY = 0.75
STAGE_GATE_MIN_SEED_SUPPORT = 2
STAGE_GATE_METRIC_DROP_MARGIN = 0.02
STAGE_GATE_METRIC_DROP_PROBABILITY = 0.80

AUC_RANKING = ["ROC AUC", "AP", "F2", "MCC", "Recall"]

FROZEN_4411_SIGNATURE: dict[str, Any] = {
    "experiment": "4411",
    "split_sizes": {"train": 337, "tuning": 337, "validation": 85, "test": 85},
    "selected_member_ids": ["xgb_member_04", "lr_member_01"],
    "evaluation_terminal_stage": "round9",
    "upstream": {
        "xgb": 0.5536180767491298,
        "lr": 0.4463819232508702,
        "cat": 0.0,
        "threshold": 0.17,
    },
    "round8": {
        "method": "identity_no_round8",
        "threshold": 0.17,
    },
    "round9": {
        "feature": "体重",
        "rule_type": "numeric_gt",
        "split_value": 58.0,
        "threshold_group": 0.3228927810673956,
        "threshold_other": 0.2094046412245567,
        "threshold": 0.3,
    },
}


def xgb_member_configs() -> list[dict[str, Any]]:
    base = {
        "colsample_bytree": 0.7,
        "gamma": 1.0,
        "max_depth": 2,
        "min_child_weight": 5,
        "reg_alpha": 0.5,
        "reg_lambda": 8.0,
        "subsample": 0.7,
    }
    ratio = 0.25
    return [
        {"member_id": "xgb_member_01", "display_name": "xgb_auc_seed42_base", "family": "xgb", "seed": 42, "params": {**base, "sampling_ratio": ratio}},
        {"member_id": "xgb_member_02", "display_name": "xgb_auc_seed52_low_child", "family": "xgb", "seed": 52, "params": {**base, "min_child_weight": 3, "reg_alpha": 0.0, "sampling_ratio": ratio}},
        {"member_id": "xgb_member_03", "display_name": "xgb_auc_seed62_more_sampling", "family": "xgb", "seed": 62, "params": {**base, "subsample": 0.75, "colsample_bytree": 0.75, "sampling_ratio": ratio}},
        {"member_id": "xgb_member_04", "display_name": "xgb_auc_seed72_strong_reg", "family": "xgb", "seed": 72, "params": {**base, "reg_lambda": 10.0, "gamma": 1.5, "sampling_ratio": ratio}},
        {"member_id": "xgb_member_05", "display_name": "xgb_auc_seed82_conservative", "family": "xgb", "seed": 82, "params": {**base, "min_child_weight": 7, "subsample": 0.65, "sampling_ratio": ratio}},
    ]


def lr_member_configs() -> list[dict[str, Any]]:
    return [
        {"member_id": "lr_member_01", "display_name": "lr_auc_seed42_base", "family": "lr", "seed": 42, "solver": "liblinear", "c_value": 0.3, "penalty": "l1", "fit_intercept": False, "class_weight": None, "sampling_ratio": 0.25},
        {"member_id": "lr_member_02", "display_name": "lr_auc_seed52_balanced", "family": "lr", "seed": 52, "solver": "liblinear", "c_value": 0.15, "penalty": "l1", "fit_intercept": False, "class_weight": "balanced", "sampling_ratio": 0.25},
        {"member_id": "lr_member_03", "display_name": "lr_auc_seed62_intercept", "family": "lr", "seed": 62, "solver": "liblinear", "c_value": 0.6, "penalty": "l1", "fit_intercept": True, "class_weight": None, "sampling_ratio": 0.25},
        {"member_id": "lr_member_04", "display_name": "lr_auc_seed72_l2", "family": "lr", "seed": 72, "solver": "liblinear", "c_value": 0.3, "penalty": "l2", "fit_intercept": False, "class_weight": None, "sampling_ratio": 0.25},
    ]


def cat_member_configs() -> list[dict[str, Any]]:
    return [
        {"member_id": "cat_member_01", "family": "cat", "seed": 42, "sampling_ratio": 1.0, "params": {"depth": 3, "learning_rate": 0.03, "l2_leaf_reg": 12.0, "random_strength": 2.0, "bagging_temperature": 1.0}},
        {"member_id": "cat_member_02", "family": "cat", "seed": 52, "sampling_ratio": 1.0, "params": {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 10.0, "random_strength": 1.5, "bagging_temperature": 1.0}},
        {"member_id": "cat_member_03", "family": "cat", "seed": 62, "sampling_ratio": 1.0, "params": {"depth": 3, "learning_rate": 0.05, "l2_leaf_reg": 8.0, "random_strength": 2.0, "bagging_temperature": 0.5}},
        {"member_id": "cat_member_04", "family": "cat", "seed": 72, "sampling_ratio": 1.0, "params": {"depth": 4, "learning_rate": 0.02, "l2_leaf_reg": 18.0, "random_strength": 3.0, "bagging_temperature": 1.5}},
    ]
