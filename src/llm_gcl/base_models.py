from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import BASE_ITERATIONS, BASE_N_ESTIMATORS, EARLY_STOPPING_ROUNDS, EARLY_STOPPING_VALID_SIZE, N_SPLITS, cat_member_configs, lr_member_configs, xgb_member_configs
from .data import SplitBundle, prepare_features
from .metrics import best_scanned_metrics, evaluate_predictions, summarize_metrics


def oversample_minority(x: pd.DataFrame, y: pd.Series, ratio: float | None, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    if ratio is None:
        return x.copy().reset_index(drop=True), y.copy().reset_index(drop=True)
    positive = y == 1
    negative = y == 0
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    target_positive = int(np.ceil(negative_count * ratio))
    if positive_count == 0 or negative_count == 0 or positive_count >= target_positive:
        return x.copy().reset_index(drop=True), y.copy().reset_index(drop=True)
    sampled_indices = y[positive].sample(n=target_positive - positive_count, replace=True, random_state=random_state).index
    x_resampled = pd.concat([x, x.loc[sampled_indices]], ignore_index=True)
    y_resampled = pd.concat([y, y.loc[sampled_indices]], ignore_index=True)
    shuffled = y_resampled.sample(frac=1.0, random_state=random_state).index
    return x_resampled.iloc[shuffled].reset_index(drop=True), y_resampled.iloc[shuffled].reset_index(drop=True)


def _column_preprocessor(x: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in x.columns if column not in numeric]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def _xgb_model(params: dict[str, Any], n_estimators: int, random_state: int, early_stopping_rounds: int | None) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=int(params["max_depth"]),
        learning_rate=0.05,
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_weight=float(params["min_child_weight"]),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        gamma=float(params["gamma"]),
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )


def _fit_xgb_fold(x: pd.DataFrame, y: pd.Series, config: dict[str, Any], random_state: int) -> tuple[dict[str, Any], int]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=EARLY_STOPPING_VALID_SIZE, random_state=random_state)
    fit_indices, stop_indices = next(splitter.split(x, y))
    x_fit, y_fit = x.iloc[fit_indices].copy(), y.iloc[fit_indices].copy()
    x_stop, y_stop = x.iloc[stop_indices].copy(), y.iloc[stop_indices].copy()
    x_fit, y_fit = oversample_minority(x_fit, y_fit, config["params"]["sampling_ratio"], random_state)
    preprocessor = _column_preprocessor(x_fit, scale_numeric=False)
    transformed_fit = preprocessor.fit_transform(x_fit)
    transformed_stop = preprocessor.transform(x_stop)
    params = {key: value for key, value in config["params"].items() if key != "sampling_ratio"}
    model = _xgb_model(params, BASE_N_ESTIMATORS, random_state, EARLY_STOPPING_ROUNDS)
    model.fit(transformed_fit, y_fit, eval_set=[(transformed_stop, y_stop)], verbose=False)
    best_iteration = getattr(model, "best_iteration", BASE_N_ESTIMATORS - 1)
    return {"family": "xgb", "preprocessor": preprocessor, "model": model, "feature_columns": list(x.columns)}, int(best_iteration)


def _fit_xgb(config: dict[str, Any], split: SplitBundle) -> tuple[dict[str, Any], np.ndarray]:
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=int(config["seed"]))
    oof = np.zeros(len(split.x_train), dtype=float)
    iterations: list[int] = []
    for fold_index, (train_indices, val_indices) in enumerate(folds.split(split.x_train, split.y_train), start=1):
        bundle, best_iteration = _fit_xgb_fold(split.x_train.iloc[train_indices], split.y_train.iloc[train_indices], config, int(config["seed"]) + fold_index)
        oof[val_indices] = predict_bundle(bundle, split.x_train.iloc[val_indices])
        iterations.append(best_iteration)
    selected_iteration = int(round(float(np.mean(iterations))))
    x_train, y_train = oversample_minority(split.x_train, split.y_train, config["params"]["sampling_ratio"], int(config["seed"]))
    preprocessor = _column_preprocessor(x_train, scale_numeric=False)
    transformed = preprocessor.fit_transform(x_train)
    params = {key: value for key, value in config["params"].items() if key != "sampling_ratio"}
    model = _xgb_model(params, max(10, selected_iteration + 1), int(config["seed"]), None)
    model.fit(transformed, y_train, verbose=False)
    return {"family": "xgb", "preprocessor": preprocessor, "model": model, "feature_columns": split.feature_columns}, oof


def _lr_pipeline(x: pd.DataFrame, config: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _column_preprocessor(x, scale_numeric=True)),
            (
                "classifier",
                LogisticRegression(
                    C=float(config["c_value"]),
                    penalty=str(config["penalty"]),
                    fit_intercept=bool(config["fit_intercept"]),
                    class_weight=config["class_weight"],
                    max_iter=4000,
                    solver=str(config["solver"]),
                    random_state=int(config["seed"]),
                ),
            ),
        ]
    )


def _fit_lr(config: dict[str, Any], split: SplitBundle) -> tuple[dict[str, Any], np.ndarray]:
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=int(config["seed"]))
    oof = np.zeros(len(split.x_train), dtype=float)
    for fold_index, (train_indices, val_indices) in enumerate(folds.split(split.x_train, split.y_train), start=1):
        x_train, y_train = oversample_minority(split.x_train.iloc[train_indices], split.y_train.iloc[train_indices], config["sampling_ratio"], int(config["seed"]) + fold_index)
        model = _lr_pipeline(x_train, config)
        model.fit(x_train, y_train)
        oof[val_indices] = model.predict_proba(split.x_train.iloc[val_indices])[:, 1]
    x_train, y_train = oversample_minority(split.x_train, split.y_train, config["sampling_ratio"], int(config["seed"]) + 1000)
    model = _lr_pipeline(x_train, config)
    model.fit(x_train, y_train)
    return {"family": "lr", "model": model, "feature_columns": split.feature_columns}, oof


def _cat_ready(x: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    ready = x.copy()
    numeric = ready.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in ready.columns if column not in numeric]
    for column in categorical:
        ready[column] = ready[column].fillna("missing").astype(str)
    return ready, categorical


def _cat_model(iterations: int, seed: int, params: dict[str, Any]) -> CatBoostClassifier:
    return CatBoostClassifier(iterations=iterations, depth=int(params["depth"]), learning_rate=float(params["learning_rate"]), l2_leaf_reg=float(params["l2_leaf_reg"]), random_strength=float(params["random_strength"]), bagging_temperature=float(params["bagging_temperature"]), loss_function="Logloss", eval_metric="AUC", random_seed=seed, verbose=False)


def _fit_cat_fold(x: pd.DataFrame, y: pd.Series, config: dict[str, Any], random_state: int) -> tuple[dict[str, Any], int]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=EARLY_STOPPING_VALID_SIZE, random_state=random_state)
    fit_indices, stop_indices = next(splitter.split(x, y))
    x_fit, y_fit = oversample_minority(x.iloc[fit_indices], y.iloc[fit_indices], config["sampling_ratio"], random_state)
    x_stop = x.iloc[stop_indices].copy()
    y_stop = y.iloc[stop_indices].copy()
    x_fit, categorical = _cat_ready(x_fit)
    x_stop, _ = _cat_ready(x_stop)
    model = _cat_model(BASE_ITERATIONS, int(config["seed"]), config["params"])
    model.fit(x_fit, y_fit, cat_features=categorical, eval_set=(x_stop, y_stop), use_best_model=True, early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
    best_iteration = model.get_best_iteration()
    return {"family": "cat", "model": model, "feature_columns": list(x.columns)}, int(BASE_ITERATIONS - 1 if best_iteration is None or best_iteration < 0 else best_iteration)


def _fit_cat(config: dict[str, Any], split: SplitBundle) -> tuple[dict[str, Any], np.ndarray]:
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=int(config["seed"]))
    oof = np.zeros(len(split.x_train), dtype=float)
    iterations: list[int] = []
    for fold_index, (train_indices, val_indices) in enumerate(folds.split(split.x_train, split.y_train), start=1):
        bundle, best_iteration = _fit_cat_fold(split.x_train.iloc[train_indices], split.y_train.iloc[train_indices], config, int(config["seed"]) + fold_index)
        oof[val_indices] = predict_bundle(bundle, split.x_train.iloc[val_indices])
        iterations.append(best_iteration)
    x_train, y_train = oversample_minority(split.x_train, split.y_train, config["sampling_ratio"], int(config["seed"]) + 2000)
    x_train, categorical = _cat_ready(x_train)
    model = _cat_model(max(20, int(round(float(np.mean(iterations)))) + 1), int(config["seed"]), config["params"])
    model.fit(x_train, y_train, cat_features=categorical, verbose=False)
    return {"family": "cat", "model": model, "feature_columns": split.feature_columns}, oof


def predict_bundle(bundle: dict[str, Any], x: pd.DataFrame) -> np.ndarray:
    aligned = prepare_features(x, bundle["feature_columns"])
    if bundle["family"] == "xgb":
        return bundle["model"].predict_proba(bundle["preprocessor"].transform(aligned))[:, 1]
    if bundle["family"] == "lr":
        return bundle["model"].predict_proba(aligned)[:, 1]
    ready, _ = _cat_ready(aligned)
    return bundle["model"].predict_proba(ready)[:, 1]


def train_base_pool(split: SplitBundle) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    pool: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for family, configs in (("xgb", xgb_member_configs()), ("lr", lr_member_configs()), ("cat", cat_member_configs())):
        for config in configs:
            member_id = str(config["member_id"])
            print(f"[基础模型] 训练 {member_id}")
            if family == "xgb":
                bundle, oof = _fit_xgb(config, split)
            elif family == "lr":
                bundle, oof = _fit_lr(config, split)
            else:
                bundle, oof = _fit_cat(config, split)
            train_prob = predict_bundle(bundle, split.x_train)
            val_prob = predict_bundle(bundle, split.x_val)
            test_prob = predict_bundle(bundle, split.x_test)
            val_best = best_scanned_metrics(split.y_val.to_numpy(), val_prob)
            payload = {
                "member_id": member_id,
                "family": family,
                "bundle": bundle,
                "train_prob": train_prob,
                "oof_prob": oof,
                "val_prob": val_prob,
                "test_prob": test_prob,
                "train_true": split.y_train.to_numpy(dtype=int),
                "oof_true": split.y_train.to_numpy(dtype=int),
                "val_true": split.y_val.to_numpy(dtype=int),
                "test_true": split.y_test.to_numpy(dtype=int),
                "val_metrics": summarize_metrics(evaluate_predictions(split.y_val.to_numpy(), val_prob, val_best["threshold"])),
                "test_metrics": summarize_metrics(evaluate_predictions(split.y_test.to_numpy(), test_prob, val_best["threshold"])),
            }
            pool[member_id] = payload
            rows.append({"member_id": member_id, "family": family, **{f"val_{key}": value for key, value in payload["val_metrics"].items()}})
    return pd.DataFrame(rows), pool


def add_dataset_predictions(pool: dict[str, dict[str, Any]], name: str, x: pd.DataFrame, y: np.ndarray | None = None) -> None:
    for payload in pool.values():
        payload[f"{name}_prob"] = predict_bundle(payload["bundle"], x)
        if y is not None:
            payload[f"{name}_true"] = np.asarray(y, dtype=int)
