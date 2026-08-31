from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cac40_2017_2022.csv"

REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def download_data(
    ticker: str = "^FCHI",
    start_date: str = "2017-01-01",
    end_date: str = "2022-01-01",
) -> pd.DataFrame:

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No data were downloaded for ticker {ticker}."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.loc[:, ~data.columns.duplicated()].copy()

    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"

    data = data.sort_index()

    data = data.loc[
        ~data.index.duplicated(keep="first")
    ]

    validate_data(data)

    return data


def validate_data(
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

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(
            "The DataFrame index must be a DatetimeIndex."
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "The dataset must be sorted chronologically."
        )

    if data.index.duplicated().any():
        raise ValueError(
            "Duplicate dates were detected."
        )

    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "Missing values were detected in OHLCV data."
        )

    if (data["High"] < data["Low"]).any():
        raise ValueError(
            "Invalid OHLC observations: High < Low."
        )

    if (
        (data["High"] < data["Open"])
        | (data["High"] < data["Close"])
    ).any():
        raise ValueError(
            "Invalid OHLC observations detected for High."
        )

    if (
        (data["Low"] > data["Open"])
        | (data["Low"] > data["Close"])
    ).any():
        raise ValueError(
            "Invalid OHLC observations detected for Low."
        )

    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError(
            "OHLC prices must be strictly positive."
        )


def save_data(
    data: pd.DataFrame,
    path: str | Path = DEFAULT_DATA_PATH,
) -> Path:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        path,
        index=True,
    )

    return path


def load_data(
    path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}"
        )

    data = pd.read_csv(
        path,
        index_col="Date",
        parse_dates=["Date"],
    )

    data = data.sort_index()

    validate_data(data)

    return data


def get_data(
    ticker: str = "^FCHI",
    start_date: str = "2017-01-01",
    end_date: str = "2022-01-01",
    path: str | Path = DEFAULT_DATA_PATH,
    force_download: bool = False,
) -> pd.DataFrame:

    path = Path(path)

    if path.exists() and not force_download:
        return load_data(path)

    data = download_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    save_data(
        data=data,
        path=path,
    )

    return data