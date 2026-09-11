from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .config import ID_COLUMNS, RANDOM_STATE, SPLIT_RANDOM_STATE, TARGET_COLUMN


@dataclass
class SplitBundle:
    train_meta: pd.DataFrame
    tuning_meta: pd.DataFrame
    validation_meta: pd.DataFrame
    test_meta: pd.DataFrame
    x_train: pd.DataFrame
    x_tuning: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_tuning: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    feature_columns: list[str]

def read_csv(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    return pd.read_csv(source)


def coerce_feature_types(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.copy()
    for column in converted.columns:
        numeric = pd.to_numeric(converted[column], errors="coerce")
        if converted[column].notna().sum() > 0 and numeric.notna().sum() == converted[column].notna().sum():
            converted[column] = numeric
        else:
            converted[column] = converted[column].astype("string").str.strip()
    return converted


def load_labeled_dataframe(path: str | Path) -> pd.DataFrame:
    frame = read_csv(path).copy()
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(f"数据缺少标签列：{TARGET_COLUMN}")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").astype(int)
    labels = set(frame[TARGET_COLUMN].dropna().unique().tolist())
    if not labels.issubset({0, 1}) or len(labels) < 2:
        raise ValueError(f"{TARGET_COLUMN} 必须同时包含 0 和 1")
    return frame.reset_index(drop=True)


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in ID_COLUMNS | {TARGET_COLUMN}]


def prepare_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    local = frame.copy()
    for column in feature_columns:
        if column not in local.columns:
            local[column] = pd.NA
    return coerce_feature_types(local[feature_columns].copy())


def _stratified_take(frame: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(splitter.split(frame, frame[TARGET_COLUMN].to_numpy(dtype=int)))
    return frame.iloc[train_indices].copy().reset_index(drop=True), frame.iloc[test_indices].copy().reset_index(drop=True)


def build_split_4411(frame: pd.DataFrame) -> SplitBundle:
    train_val, test = _stratified_take(frame, test_size=0.1, random_state=RANDOM_STATE)
    original_train, validation = _stratified_take(train_val, test_size=1.0 / 9.0, random_state=RANDOM_STATE + 1)
    train, tuning = _stratified_take(original_train, test_size=0.5, random_state=SPLIT_RANDOM_STATE)
    columns = select_feature_columns(frame)
    return SplitBundle(
        train_meta=train,
        tuning_meta=tuning,
        validation_meta=validation,
        test_meta=test,
        x_train=prepare_features(train, columns),
        x_tuning=prepare_features(tuning, columns),
        x_validation=prepare_features(validation, columns),
        x_test=prepare_features(test, columns),
        y_train=train[TARGET_COLUMN].copy().reset_index(drop=True),
        y_tuning=tuning[TARGET_COLUMN].copy().reset_index(drop=True),
        y_validation=validation[TARGET_COLUMN].copy().reset_index(drop=True),
        y_test=test[TARGET_COLUMN].copy().reset_index(drop=True),
        feature_columns=columns,
    )
