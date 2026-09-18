from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .config import AUC_RANKING
from .metrics import best_scanned_metrics, rank_key


def _ids(registry: pd.DataFrame, family: str) -> list[str]:
    return registry.loc[registry["family"] == family, "member_id"].tolist()


def _mean(pool: dict[str, dict[str, Any]], member_ids: list[str], split: str, length: int | None = None) -> np.ndarray:
    if not member_ids:
        if length is None:
            raise ValueError("length is required for an empty model family")
        return np.zeros(length, dtype=float)
    return np.mean(np.column_stack([pool[member_id][f"{split}_prob"] for member_id in member_ids]), axis=1)


def _blend_xgb_lr(pool: dict[str, dict[str, Any]], member_ids: list[str], split: str, xgb_weight: float) -> np.ndarray:
    xgb_ids = [member_id for member_id in member_ids if pool[member_id]["family"] == "xgb"]
    lr_ids = [member_id for member_id in member_ids if pool[member_id]["family"] == "lr"]
    if xgb_ids and lr_ids:
        return xgb_weight * _mean(pool, xgb_ids, split) + (1.0 - xgb_weight) * _mean(pool, lr_ids, split)
    return _mean(pool, xgb_ids or lr_ids, split)


def select_xgb_lr(registry: pd.DataFrame, pool: dict[str, dict[str, Any]]) -> dict[str, Any]:
    y_validation = next(iter(pool.values()))["validation_true"]

    def best_for(member_ids: list[str]) -> dict[str, Any]:
        families = {pool[member_id]["family"] for member_id in member_ids}
        weights = [1.0] if families == {"xgb"} else [0.0] if families == {"lr"} else [round(value, 2) for value in np.arange(0.1, 0.91, 0.1)]
        rows = []
        for weight in weights:
            metrics = best_scanned_metrics(y_validation, _blend_xgb_lr(pool, member_ids, "validation", weight))
            rows.append({"selected_member_ids": member_ids, "xgb_weight": weight, "lr_weight": 1.0 - weight, **metrics})
        return sorted(rows, key=rank_key, reverse=True)[0]

    seeds = [best_for([xgb_id, lr_id]) for xgb_id in _ids(registry, "xgb") for lr_id in _ids(registry, "lr")]
    current = sorted(seeds, key=rank_key, reverse=True)[0]
    selected = list(current["selected_member_ids"])
    remaining = [member_id for member_id in registry.loc[registry["family"].isin(["xgb", "lr"]), "member_id"].tolist() if member_id not in selected]
    while remaining and len(selected) < 7:
        feasible = []
        for candidate in remaining:
            trial = best_for(selected + [candidate])
            if trial["ROC AUC"] - current["ROC AUC"] <= 0.002:
                continue
            if trial["F2"] - current["F2"] < -0.005 or trial["MCC"] - current["MCC"] < -0.010 or trial["AP"] - current["AP"] < -0.002:
                continue
            feasible.append((candidate, trial))
        if not feasible:
            break
        candidate, current = max(feasible, key=lambda item: rank_key(item[1]))
        selected = list(current["selected_member_ids"])
        remaining.remove(candidate)
    return {
        "selection_dataset": "validation",
        "selected_member_ids": selected,
        "selected_xgb_member_ids": [member_id for member_id in selected if pool[member_id]["family"] == "xgb"],
        "selected_lr_member_ids": [member_id for member_id in selected if pool[member_id]["family"] == "lr"],
        "xgb_weight": float(current["xgb_weight"]),
        "lr_weight": float(current["lr_weight"]),
        "threshold": float(current["threshold"]),
        "validation_metrics": {name: float(current[name]) for name in AUC_RANKING},
    }


def select_cat(registry: pd.DataFrame, pool: dict[str, dict[str, Any]], base: dict[str, Any]) -> dict[str, Any]:
    xgb_ids = list(base["selected_xgb_member_ids"])
    lr_ids = list(base["selected_lr_member_ids"])
    cat_ids = _ids(registry, "cat")
    y_validation = next(iter(pool.values()))["validation_true"]
    baseline = best_scanned_metrics(y_validation, _blend_xgb_lr(pool, xgb_ids + lr_ids, "validation", float(base["xgb_weight"])))
    candidates: list[dict[str, Any]] = []
    values = np.round(np.arange(0.05, 1.0, 0.05), 2)
    for size in (1, 2):
        for subset in combinations(cat_ids, size):
            for wx in values:
                for wl in values:
                    wc = round(1.0 - float(wx) - float(wl), 2)
                    if wc < 0.05 or wc > 0.25:
                        continue
                    probability = wx * _mean(pool, xgb_ids, "validation") + wl * _mean(pool, lr_ids, "validation") + wc * _mean(pool, list(subset), "validation")
                    metrics = best_scanned_metrics(y_validation, probability)
                    if metrics["ROC AUC"] - baseline["ROC AUC"] < 0.002:
                        continue
                    if metrics["F2"] - baseline["F2"] < -0.005 or metrics["MCC"] - baseline["MCC"] < -0.010 or metrics["AP"] - baseline["AP"] < -0.002:
                        continue
                    candidates.append({"selected_cat_member_ids": list(subset), "xgb_weight": float(wx), "lr_weight": float(wl), "cat_weight": float(wc), **metrics})
    if not candidates:
        return {"selection_dataset": "validation", "base_best_member_ids": xgb_ids + lr_ids, "selected_cat_member_ids": [], "xgb_weight": base["xgb_weight"], "lr_weight": base["lr_weight"], "cat_weight": 0.0, "threshold": base["threshold"]}
    selected = sorted(candidates, key=rank_key, reverse=True)[0]
    return {"selection_dataset": "validation", "base_best_member_ids": xgb_ids + lr_ids, **selected}


def raw_upstream_frames(pool: dict[str, dict[str, Any]], selection: dict[str, Any], splits: list[str]) -> dict[str, pd.DataFrame]:
    base_ids = list(selection["base_best_member_ids"])
    cat_ids = list(selection["selected_cat_member_ids"])
    xgb_ids = [member_id for member_id in base_ids if member_id.startswith("xgb_")]
    lr_ids = [member_id for member_id in base_ids if member_id.startswith("lr_")]
    frames = {}
    for split in splits:
        lookup = "oof" if split == "train_oof" else split
        y_true = pool[base_ids[0]][f"{lookup}_true"]
        xgb_mean = _mean(pool, xgb_ids, lookup, len(y_true))
        lr_mean = _mean(pool, lr_ids, lookup, len(y_true))
        cat_mean = _mean(pool, cat_ids, lookup, len(y_true))
        frames[split] = pd.DataFrame({"dataset": split, "y_true": y_true, "xgb_mean_prob": xgb_mean, "lr_mean_prob": lr_mean, "cat_mean_prob": cat_mean})
    return frames
