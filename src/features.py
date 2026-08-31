from __future__ import annotations

import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, CCIIndicator


FEATURE_COLUMNS = [
    "Return_1D",
    "Return_8D",
    "ADX_20",
    "RSI_20",
    "Stochastic_14_3",
    "CCI_20",
    "Volatility_10D",
]


REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
]


def validate_ohlc_data(
    data: pd.DataFrame,
) -> None:
    if data.empty:
        raise ValueError("The dataset is empty.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data[REQUIRED_COLUMNS].isna().all().any():
        raise ValueError(
            "One or more OHLC columns contain only missing values."
        )

    if (
        data[REQUIRED_COLUMNS] <= 0
    ).any().any():
        raise ValueError(
            "OHLC prices must be strictly positive."
        )


def compute_return_1d(
    data: pd.DataFrame,
) -> pd.Series:
    return data["Close"].pct_change(
        periods=1,
        fill_method=None,
    )


def compute_return_8d(
    data: pd.DataFrame,
) -> pd.Series:
    return data["Close"].pct_change(
        periods=8,
        fill_method=None,
    )


def compute_adx(
    data: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    indicator = ADXIndicator(
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        window=period,
        fillna=False,
    )

    return indicator.adx()


def compute_rsi(
    data: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    indicator = RSIIndicator(
        close=data["Close"],
        window=period,
        fillna=False,
    )

    return indicator.rsi()


def compute_stochastic(
    data: pd.DataFrame,
    fastk_period: int = 14,
    slowk_period: int = 3,
) -> pd.Series:
    lowest_low = data["Low"].rolling(
        window=fastk_period,
        min_periods=fastk_period,
    ).min()

    highest_high = data["High"].rolling(
        window=fastk_period,
        min_periods=fastk_period,
    ).max()

    denominator = highest_high - lowest_low

    denominator = denominator.replace(
        0,
        np.nan,
    )

    fast_k = (
        100
        * (data["Close"] - lowest_low)
        / denominator
    )

    slow_k = fast_k.rolling(
        window=slowk_period,
        min_periods=slowk_period,
    ).mean()

    return slow_k


def compute_cci(
    data: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    indicator = CCIIndicator(
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        window=period,
        constant=0.015,
        fillna=False,
    )

    return indicator.cci()


def compute_volatility(
    data: pd.DataFrame,
    window: int = 10,
) -> pd.Series:
    daily_returns = data["Close"].pct_change(
        fill_method=None
    )

    return daily_returns.rolling(
        window=window,
        min_periods=window,
    ).std()


def build_target(
    data: pd.DataFrame,
) -> pd.Series:
    future_increase = (
        data["Close"].shift(-1)
        > data["Close"]
    )

    positive_return_8d = (
        data["Return_8D"] > 0
    )

    target = (
        future_increase
        & positive_return_8d
    ).astype(int)

    target = target.astype("float64")

    target.iloc[-1] = np.nan

    return target


def build_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    validate_ohlc_data(data)

    features = data.copy()

    features["Return_1D"] = compute_return_1d(
        features
    )

    features["Return_8D"] = compute_return_8d(
        features
    )

    features["ADX_20"] = compute_adx(
        features,
        period=20,
    )

    features["RSI_20"] = compute_rsi(
        features,
        period=20,
    )

    features["Stochastic_14_3"] = compute_stochastic(
        features,
        fastk_period=14,
        slowk_period=3,
    )

    features["CCI_20"] = compute_cci(
        features,
        period=20,
    )

    features["Volatility_10D"] = compute_volatility(
        features,
        window=10,
    )

    return features


def build_dataset(
    data: pd.DataFrame,
    dropna: bool = True,
) -> pd.DataFrame:
    dataset = build_features(data)

    dataset["Label"] = build_target(
        dataset
    )

    if dropna:
        dataset = dataset.dropna(
            subset=FEATURE_COLUMNS + ["Label"]
        ).copy()

    dataset["Label"] = (
        dataset["Label"]
        .astype(int)
    )

    return dataset


def get_feature_matrix(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    return dataset[
        FEATURE_COLUMNS
    ].copy()


def get_target_vector(
    dataset: pd.DataFrame,
) -> pd.Series:
    if "Label" not in dataset.columns:
        raise ValueError(
            "The dataset does not contain a Label column."
        )

    return dataset[
        "Label"
    ].astype(int).copy()