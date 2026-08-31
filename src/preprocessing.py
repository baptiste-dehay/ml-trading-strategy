from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.config import TRAIN_RATIO
from src.features import FEATURE_COLUMNS


def validate_dataset(
    dataset: pd.DataFrame,
) -> None:
    if dataset.empty:
        raise ValueError(
            "The dataset is empty."
        )

    required_columns = FEATURE_COLUMNS + ["Label"]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if dataset[required_columns].isna().any().any():
        raise ValueError(
            "The dataset contains missing values."
        )

    if not dataset.index.is_monotonic_increasing:
        raise ValueError(
            "The dataset must be sorted chronologically."
        )

    unique_labels = set(
        dataset["Label"].unique()
    )

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            "Label must contain only 0 and 1."
        )


def split_features_target(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    validate_dataset(dataset)

    X = dataset[
        FEATURE_COLUMNS
    ].copy()

    y = dataset[
        "Label"
    ].astype(int).copy()

    return X, y


def chronological_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must have the same number of observations."
        )

    if len(X) < 2:
        raise ValueError(
            "The dataset must contain at least two observations."
        )

    if not X.index.equals(y.index):
        raise ValueError(
            "X and y must have identical indices."
        )

    if not X.index.is_monotonic_increasing:
        raise ValueError(
            "X must be sorted chronologically."
        )

    split_index = int(
        len(X) * train_ratio
    )

    if split_index == 0 or split_index >= len(X):
        raise ValueError(
            "Invalid train/test split."
        )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    y_test = y.iloc[
        split_index:
    ].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    StandardScaler,
]:
    if X_train.empty:
        raise ValueError(
            "X_train cannot be empty."
        )

    if X_test.empty:
        raise ValueError(
            "X_test cannot be empty."
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "X_train and X_test must contain the same columns."
        )

    if X_train.isna().any().any():
        raise ValueError(
            "X_train contains missing values."
        )

    if X_test.isna().any().any():
        raise ValueError(
            "X_test contains missing values."
        )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        X_train_scaled,
        X_test_scaled,
        scaler,
    )


def prepare_data(
    dataset: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> dict:
    X, y = split_features_target(
        dataset
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = chronological_train_test_split(
        X=X,
        y=y,
        train_ratio=train_ratio,
    )

    (
        X_train_scaled,
        X_test_scaled,
        scaler,
    ) = scale_features(
        X_train=X_train,
        X_test=X_test,
    )

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }