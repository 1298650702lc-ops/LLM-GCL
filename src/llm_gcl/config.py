from __future__ import annotations

from typing import Any


TARGET_COLUMN = "左旋标签"
ID_COLUMNS = {"id"}
RANDOM_STATE = 20260417
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

AUC_RANKING = ["ROC AUC", "AP", "F2", "MCC", "Recall"]

PAPER_SIGNATURE: dict[str, Any] = {
    "selected_member_ids": ["xgb_member_03", "lr_member_03", "cat_member_03"],
    "upstream": {
        "xgb_weight": 0.3364496794519848,
        "lr_weight": 0.2639327543151499,
        "cat_weight": 0.3996175662328653,
        "threshold": 0.35,
    },
    "round8": {
        "delta": 0.05362436970744623,
        "t_lr": 0.3531320954277771,
        "t_mid": 0.3981631211903832,
        "t_xgb": 0.4049510114761756,
        "threshold": 0.44,
    },
    "round9": {
        "feature": "体重",
        "rule_type": "numeric_le",
        "split_value": 64.0,
        "threshold_group": 0.2855254503160047,
        "threshold_other": 0.3251799886369514,
        "threshold": 0.67,
    },
    "round10": {
        "primary_feature": "体重",
        "primary_rule_type": "numeric_le",
        "primary_split_value": 65.0,
        "aux_feature": "甲状腺疾病",
        "aux_rule_type": "numeric_le",
        "aux_split_value": 0.0,
        "threshold_primary_aux": 0.22446507100301277,
        "threshold_primary_base": 0.28048412949716106,
        "threshold_other": 0.3133202625837724,
        "threshold": 0.6559624358478064,
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
        {"member_id": "xgb_member_01", "family": "xgb", "seed": 42, "params": {**base, "sampling_ratio": ratio}},
        {"member_id": "xgb_member_02", "family": "xgb", "seed": 52, "params": {**base, "min_child_weight": 3, "reg_alpha": 0.0, "sampling_ratio": ratio}},
        {"member_id": "xgb_member_03", "family": "xgb", "seed": 62, "params": {**base, "subsample": 0.75, "colsample_bytree": 0.75, "sampling_ratio": ratio}},
        {"member_id": "xgb_member_04", "family": "xgb", "seed": 72, "params": {**base, "reg_lambda": 10.0, "gamma": 1.5, "sampling_ratio": ratio}},
        {"member_id": "xgb_member_05", "family": "xgb", "seed": 82, "params": {**base, "min_child_weight": 7, "subsample": 0.65, "sampling_ratio": ratio}},
    ]


def lr_member_configs() -> list[dict[str, Any]]:
    return [
        {"member_id": "lr_member_01", "family": "lr", "seed": 42, "solver": "liblinear", "c_value": 0.3, "penalty": "l1", "fit_intercept": False, "class_weight": None, "sampling_ratio": 0.25},
        {"member_id": "lr_member_02", "family": "lr", "seed": 52, "solver": "liblinear", "c_value": 0.15, "penalty": "l1", "fit_intercept": False, "class_weight": "balanced", "sampling_ratio": 0.25},
        {"member_id": "lr_member_03", "family": "lr", "seed": 62, "solver": "liblinear", "c_value": 0.6, "penalty": "l1", "fit_intercept": True, "class_weight": None, "sampling_ratio": 0.25},
        {"member_id": "lr_member_04", "family": "lr", "seed": 72, "solver": "liblinear", "c_value": 0.3, "penalty": "l2", "fit_intercept": False, "class_weight": None, "sampling_ratio": 0.25},
    ]


def cat_member_configs() -> list[dict[str, Any]]:
    return [
        {"member_id": "cat_member_01", "family": "cat", "seed": 42, "sampling_ratio": 1.0, "params": {"depth": 3, "learning_rate": 0.03, "l2_leaf_reg": 12.0, "random_strength": 2.0, "bagging_temperature": 1.0}},
        {"member_id": "cat_member_02", "family": "cat", "seed": 52, "sampling_ratio": 1.0, "params": {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 10.0, "random_strength": 1.5, "bagging_temperature": 1.0}},
        {"member_id": "cat_member_03", "family": "cat", "seed": 62, "sampling_ratio": 1.0, "params": {"depth": 3, "learning_rate": 0.05, "l2_leaf_reg": 8.0, "random_strength": 2.0, "bagging_temperature": 0.5}},
        {"member_id": "cat_member_04", "family": "cat", "seed": 72, "sampling_ratio": 1.0, "params": {"depth": 4, "learning_rate": 0.02, "l2_leaf_reg": 18.0, "random_strength": 3.0, "bagging_temperature": 1.5}},
    ]
