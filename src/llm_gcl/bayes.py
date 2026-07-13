from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from .config import BO_CANDIDATES, BO_INIT_POINTS, BO_ITERATIONS, BO_RANDOM_STATE, MIN_GROUP_SIZE, TARGET_COLUMN
from .metrics import best_scanned_metrics, evaluate_predictions, metric_objective, summarize_metrics
from .rules import apply_rule, group_probability, local_rule_variants, prediction_frame, segmented_probability, three_group_probability


warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class SearchSpace:
    bounds: np.ndarray
    integer_indices: set[int]
    sampler: Callable[[np.random.Generator, int], np.ndarray]
    decoder: Callable[[np.ndarray], dict[str, Any]]


def _unique_rows(rows: list[np.ndarray], integer_indices: set[int]) -> list[np.ndarray]:
    unique = []
    seen = set()
    for row in rows:
        local = row.copy()
        for index in integer_indices:
            local[index] = round(float(local[index]))
        key = tuple(round(float(value), 8) for value in local)
        if key not in seen:
            seen.add(key)
            unique.append(local)
    return unique


def _expected_improvement(mean: np.ndarray, standard_deviation: np.ndarray, best: float) -> np.ndarray:
    standard_deviation = np.maximum(standard_deviation, 1e-12)
    improvement = mean - best
    z_value = improvement / standard_deviation
    return improvement * norm.cdf(z_value) + standard_deviation * norm.pdf(z_value)


def bayesian_optimize(space: SearchSpace, objective: Callable[[dict[str, Any]], dict[str, Any] | None], random_state: int, iterations: int = BO_ITERATIONS, candidate_count: int = BO_CANDIDATES) -> tuple[dict[str, Any], pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    evaluated = set()

    def evaluate_vector(vector: np.ndarray, source: str, trial: int) -> None:
        local = vector.copy()
        for index in space.integer_indices:
            local[index] = round(float(local[index]))
        local = np.clip(local, space.bounds[:, 0], space.bounds[:, 1])
        key = tuple(round(float(value), 8) for value in local)
        if key in evaluated:
            return
        evaluated.add(key)
        parameters = space.decoder(local)
        metrics = objective(parameters)
        if metrics is None:
            return
        score = metric_objective(metrics)
        x_values.append(local)
        y_values.append(score)
        rows.append({"trial": trial, "source": source, "objective": score, **parameters, **metrics})

    trial = 0
    for vector in _unique_rows(list(space.sampler(rng, BO_INIT_POINTS)), space.integer_indices):
        trial += 1
        evaluate_vector(vector, "initial", trial)
    for _ in range(iterations):
        candidates = _unique_rows(list(space.sampler(rng, candidate_count)), space.integer_indices)
        if len(x_values) < 3:
            vector = candidates[0]
        else:
            lower, upper = space.bounds[:, 0], space.bounds[:, 1]
            x_array = (np.vstack(x_values) - lower) / np.maximum(upper - lower, 1e-12)
            kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
            surrogate = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=random_state)
            surrogate.fit(x_array, np.asarray(y_values, dtype=float))
            candidate_array = (np.vstack(candidates) - lower) / np.maximum(upper - lower, 1e-12)
            mean, standard_deviation = surrogate.predict(candidate_array, return_std=True)
            order = np.argsort(_expected_improvement(mean, standard_deviation, float(np.max(y_values))))[::-1]
            vector = candidates[int(order[0])]
            for index in order:
                candidate = candidates[int(index)]
                key = tuple(round(float(value), 8) for value in candidate)
                if key not in evaluated:
                    vector = candidate
                    break
        trial += 1
        evaluate_vector(vector, "bayes_ei", trial)
    trials = pd.DataFrame(rows).sort_values(by=["objective", "ROC AUC", "AP", "F2", "MCC", "Recall"], ascending=False).reset_index(drop=True)
    if trials.empty:
        raise RuntimeError("贝叶斯优化没有产生有效候选")
    return trials.iloc[0].to_dict(), trials


def run_upstream(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    validation = frames["val"]
    y_val = validation["y_true"].to_numpy(dtype=int)

    def sampler(rng: np.random.Generator, count: int) -> np.ndarray:
        weights = rng.dirichlet([4.0, 2.5, 1.8], size=count)
        weights[:, 2] = np.minimum(weights[:, 2], 0.35)
        weights /= weights.sum(axis=1, keepdims=True)
        return np.column_stack([weights[:, 0], weights[:, 1]])

    def decoder(vector: np.ndarray) -> dict[str, float]:
        wx = float(np.clip(vector[0], 0.05, 0.90))
        wl = float(np.clip(vector[1], 0.05, 0.90))
        if wx + wl > 0.98:
            total = wx + wl
            wx, wl = wx / total * 0.98, wl / total * 0.98
        wc = max(0.02, 1.0 - wx - wl)
        total = wx + wl + wc
        return {"xgb_weight": wx / total, "lr_weight": wl / total, "cat_weight": wc / total}

    def probability(frame: pd.DataFrame, parameters: dict[str, float]) -> np.ndarray:
        return parameters["xgb_weight"] * frame["xgb_mean_prob"].to_numpy(dtype=float) + parameters["lr_weight"] * frame["lr_mean_prob"].to_numpy(dtype=float) + parameters["cat_weight"] * frame["cat_mean_prob"].to_numpy(dtype=float)

    space = SearchSpace(np.asarray([[0.05, 0.90], [0.05, 0.90]], dtype=float), set(), sampler, decoder)
    best, trials = bayesian_optimize(space, lambda params: best_scanned_metrics(y_val, probability(validation, params)), BO_RANDOM_STATE + 1)
    weights = {name: float(best[name]) for name in ("xgb_weight", "lr_weight", "cat_weight")}
    threshold = float(best["threshold"])
    stage_frames = {}
    metrics = {}
    for name, frame in frames.items():
        values = probability(frame, weights)
        y_true = frame["y_true"].to_numpy(dtype=int)
        stage_frames[name] = pd.DataFrame({"dataset": name, "y_true": y_true, "xgb_mean_prob": frame["xgb_mean_prob"], "lr_mean_prob": frame["lr_mean_prob"], "cat_mean_prob": frame["cat_mean_prob"], "ensemble_prob": values, "bias": frame["lr_mean_prob"].to_numpy(dtype=float) - frame["xgb_mean_prob"].to_numpy(dtype=float)})
        metrics[f"{name}_metrics"] = summarize_metrics(evaluate_predictions(y_true, values, threshold))
    return {"config": {"method": "bayesian_optimization", "weights": weights, "threshold": threshold, **metrics}, "frames": stage_frames, "trials": trials}


def run_round8(upstream: dict[str, Any]) -> dict[str, Any]:
    validation = upstream["frames"]["val"]
    y_val = validation["y_true"].to_numpy(dtype=int)
    probability = validation["ensemble_prob"].to_numpy(dtype=float)
    bias = validation["bias"].to_numpy(dtype=float)

    def sampler(rng: np.random.Generator, count: int) -> np.ndarray:
        return np.column_stack([rng.uniform(0.02, 0.18, count), rng.uniform(0.12, 0.38, count), rng.uniform(0.18, 0.48, count), rng.uniform(0.24, 0.58, count)])

    def decoder(vector: np.ndarray) -> dict[str, float]:
        thresholds = sorted([float(vector[1]), float(vector[2]), float(vector[3])])
        return {"delta": float(vector[0]), "t_lr": thresholds[0], "t_mid": thresholds[1], "t_xgb": thresholds[2]}

    def objective(params: dict[str, float]) -> dict[str, Any]:
        adjusted, _ = segmented_probability(probability, bias, **params)
        return best_scanned_metrics(y_val, adjusted)

    bounds = np.asarray([[0.02, 0.18], [0.12, 0.38], [0.18, 0.48], [0.24, 0.58]], dtype=float)
    best, trials = bayesian_optimize(SearchSpace(bounds, set(), sampler, decoder), objective, BO_RANDOM_STATE + 2)
    parameters = {name: float(best[name]) for name in ("delta", "t_lr", "t_mid", "t_xgb")}
    threshold = float(best["threshold"])
    frames, metrics = {}, {}
    for name, source in upstream["frames"].items():
        adjusted, _ = segmented_probability(source["ensemble_prob"].to_numpy(dtype=float), source["bias"].to_numpy(dtype=float), **parameters)
        y_true = source["y_true"].to_numpy(dtype=int)
        frames[name] = prediction_frame(name, y_true, adjusted, threshold)
        metrics[f"{name}_metrics"] = summarize_metrics(evaluate_predictions(y_true, adjusted, threshold))
    return {"config": {"method": "bayesian_optimization", **parameters, "threshold": threshold, **metrics}, "frames": frames, "trials": trials}


def run_round9(split_frames: dict[str, pd.DataFrame], rules: list[dict[str, Any]]) -> dict[str, Any]:
    validation = split_frames["val"]
    y_val = validation[TARGET_COLUMN].to_numpy(dtype=int)
    probability = validation["pred_y_prob"].to_numpy(dtype=float)
    valid_rules = [rule for rule in rules if rule["feature"] in validation.columns and apply_rule(validation, rule).sum() >= MIN_GROUP_SIZE and (~apply_rule(validation, rule)).sum() >= MIN_GROUP_SIZE]
    if not valid_rules:
        raise RuntimeError("错误模式分析未产生有效规则")

    def sampler(rng: np.random.Generator, count: int) -> np.ndarray:
        return np.column_stack([rng.integers(0, len(valid_rules), count), rng.uniform(0.12, 0.50, count), rng.uniform(0.12, 0.50, count)])

    def decoder(vector: np.ndarray) -> dict[str, Any]:
        return {"rule_index": int(np.clip(round(float(vector[0])), 0, len(valid_rules) - 1)), "threshold_group": float(vector[1]), "threshold_other": float(vector[2])}

    def objective(params: dict[str, Any]) -> dict[str, Any]:
        adjusted = group_probability(probability, apply_rule(validation, valid_rules[params["rule_index"]]), params["threshold_group"], params["threshold_other"])
        return best_scanned_metrics(y_val, adjusted)

    bounds = np.asarray([[0, len(valid_rules) - 1], [0.12, 0.50], [0.12, 0.50]], dtype=float)
    best, trials = bayesian_optimize(SearchSpace(bounds, {0}, sampler, decoder), objective, BO_RANDOM_STATE + 3)
    rule = valid_rules[int(best["rule_index"])]
    threshold = float(best["threshold"])
    frames, metrics = {}, {}
    for name in ("train", "val", "test"):
        frame = split_frames[name]
        adjusted = group_probability(frame["pred_y_prob"].to_numpy(dtype=float), apply_rule(frame, rule), float(best["threshold_group"]), float(best["threshold_other"]))
        y_true = frame[TARGET_COLUMN].to_numpy(dtype=int)
        frames[name] = prediction_frame(name, y_true, adjusted, threshold)
        metrics[f"{name}_metrics"] = summarize_metrics(evaluate_predictions(y_true, adjusted, threshold))
    config = {"method": "bayesian_optimization", "candidate_pool_size": len(valid_rules), "rule": rule, "threshold_group": float(best["threshold_group"]), "threshold_other": float(best["threshold_other"]), "threshold": threshold, **metrics}
    return {"config": config, "frames": frames, "rules": valid_rules, "trials": trials}


def run_round10(split_frames: dict[str, pd.DataFrame], primary_rules: list[dict[str, Any]], aux_rules: list[dict[str, Any]]) -> dict[str, Any]:
    validation = split_frames["val"]
    y_val = validation[TARGET_COLUMN].to_numpy(dtype=int)
    probability = validation["pred_y_prob"].to_numpy(dtype=float)
    valid_primary = []
    for rule in primary_rules:
        for local in local_rule_variants(validation, rule):
            mask = apply_rule(validation, local)
            if mask.sum() >= MIN_GROUP_SIZE and (~mask).sum() >= MIN_GROUP_SIZE:
                valid_primary.append(local)
    valid_aux = []
    for rule in aux_rules:
        mask = apply_rule(validation, rule)
        if mask.sum() >= 3:
            valid_aux.append(rule)
    if not valid_primary or not valid_aux:
        raise RuntimeError("Round10 没有有效主规则或辅助规则")

    def sampler(rng: np.random.Generator, count: int) -> np.ndarray:
        return np.column_stack([rng.integers(0, len(valid_primary), count), rng.integers(0, len(valid_aux), count), rng.uniform(0.12, 0.45, count), rng.uniform(0.12, 0.45, count), rng.uniform(0.12, 0.45, count), rng.uniform(0.56, 0.70, count)])

    def decoder(vector: np.ndarray) -> dict[str, Any]:
        return {"primary_index": int(np.clip(round(float(vector[0])), 0, len(valid_primary) - 1)), "aux_index": int(np.clip(round(float(vector[1])), 0, len(valid_aux) - 1)), "threshold_primary_aux": float(vector[2]), "threshold_primary_base": float(vector[3]), "threshold_other": float(vector[4]), "report_threshold": float(vector[5])}

    def objective(params: dict[str, Any]) -> dict[str, Any] | None:
        primary = valid_primary[params["primary_index"]]
        aux = valid_aux[params["aux_index"]]
        if primary["feature"] == aux["feature"]:
            return None
        primary_mask = apply_rule(validation, primary)
        aux_mask = apply_rule(validation, aux)
        if (primary_mask & aux_mask).sum() < 3:
            return None
        adjusted = three_group_probability(probability, primary_mask, aux_mask, params["threshold_primary_aux"], params["threshold_primary_base"], params["threshold_other"])
        return summarize_metrics(evaluate_predictions(y_val, adjusted, params["report_threshold"]))

    bounds = np.asarray([[0, len(valid_primary) - 1], [0, len(valid_aux) - 1], [0.12, 0.45], [0.12, 0.45], [0.12, 0.45], [0.56, 0.70]], dtype=float)
    best, trials = bayesian_optimize(SearchSpace(bounds, {0, 1}, sampler, decoder), objective, BO_RANDOM_STATE + 4, iterations=72, candidate_count=768)
    primary = valid_primary[int(best["primary_index"])]
    aux = valid_aux[int(best["aux_index"])]
    threshold = float(best["report_threshold"])
    frames, metrics = {}, {}
    for name in ("train", "val", "test"):
        frame = split_frames[name]
        adjusted = three_group_probability(frame["pred_y_prob"].to_numpy(dtype=float), apply_rule(frame, primary), apply_rule(frame, aux), float(best["threshold_primary_aux"]), float(best["threshold_primary_base"]), float(best["threshold_other"]))
        y_true = frame[TARGET_COLUMN].to_numpy(dtype=int)
        frames[name] = prediction_frame(name, y_true, adjusted, threshold)
        metrics[f"{name}_metrics"] = summarize_metrics(evaluate_predictions(y_true, adjusted, threshold))
    config = {"method": "bayesian_optimization", "primary_pool_size": len(valid_primary), "aux_pool_size": len(valid_aux), "primary_rule": primary, "aux_rule": aux, "threshold_primary_aux": float(best["threshold_primary_aux"]), "threshold_primary_base": float(best["threshold_primary_base"]), "threshold_other": float(best["threshold_other"]), "threshold": threshold, **metrics}
    return {"config": config, "frames": frames, "trials": trials}
