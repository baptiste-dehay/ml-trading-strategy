from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    INITIAL_CAPITAL,
    PREDICTION_THRESHOLD,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    TRANSACTION_COST_BPS,
)


def generate_trading_signals(
    predictions,
    threshold: float = PREDICTION_THRESHOLD,
) -> pd.Series:
    predictions = pd.Series(
        np.asarray(predictions).reshape(-1),
        dtype=float,
    )

    if predictions.empty:
        raise ValueError(
            "predictions cannot be empty."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if predictions.isna().any():
        raise ValueError(
            "predictions contain missing values."
        )

    if (
        (predictions < 0)
        | (predictions > 1)
    ).any():
        raise ValueError(
            "predictions must be between 0 and 1."
        )

    return (
        predictions >= threshold
    ).astype(int)


def compute_forward_returns(
    prices: pd.Series,
) -> pd.Series:
    prices = pd.Series(
        prices,
        index=prices.index,
        dtype=float,
    )

    if prices.empty:
        raise ValueError(
            "prices cannot be empty."
        )

    if prices.isna().any():
        raise ValueError(
            "prices contain missing values."
        )

    if (prices <= 0).any():
        raise ValueError(
            "prices must be strictly positive."
        )

    return (
        prices.shift(-1)
        / prices
        - 1
    )


def compute_max_drawdown(
    equity_curve: pd.Series,
) -> float:
    equity_curve = pd.Series(
        equity_curve,
        dtype=float,
    ).dropna()

    if equity_curve.empty:
        return np.nan

    running_max = equity_curve.cummax()

    drawdown = (
        equity_curve
        / running_max
        - 1
    )

    return float(
        drawdown.min()
    )


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> float:
    returns = pd.Series(
        returns,
        dtype=float,
    ).dropna()

    if returns.empty:
        return np.nan

    if annualization_factor <= 0:
        raise ValueError(
            "annualization_factor must be strictly positive."
        )

    if risk_free_rate <= -1:
        raise ValueError(
            "risk_free_rate must be greater than -1."
        )

    daily_rf = (
        (1 + risk_free_rate)
        ** (1 / annualization_factor)
        - 1
    )

    excess_returns = (
        returns - daily_rf
    )

    volatility = excess_returns.std(
        ddof=1
    )

    if (
        volatility == 0
        or np.isnan(volatility)
    ):
        return np.nan

    return float(
        excess_returns.mean()
        / volatility
        * np.sqrt(
            annualization_factor
        )
    )


def run_backtest(
    prices: pd.Series,
    predictions,
    threshold: float = PREDICTION_THRESHOLD,
    initial_capital: float = INITIAL_CAPITAL,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
    risk_free_rate: float = RISK_FREE_RATE,
) -> tuple[pd.DataFrame, dict]:
    if initial_capital <= 0:
        raise ValueError(
            "initial_capital must be strictly positive."
        )

    if transaction_cost_bps < 0:
        raise ValueError(
            "transaction_cost_bps cannot be negative."
        )

    prices = pd.Series(
        prices,
        index=prices.index,
        dtype=float,
    ).copy()

    predictions = np.asarray(
        predictions
    ).reshape(-1)

    if prices.empty:
        raise ValueError(
            "prices cannot be empty."
        )

    if len(prices) != len(predictions):
        raise ValueError(
            "prices and predictions must contain the same number "
            "of observations."
        )

    signals = generate_trading_signals(
        predictions=predictions,
        threshold=threshold,
    )

    signals.index = prices.index

    backtest = pd.DataFrame(
        index=prices.index
    )

    backtest["Close"] = prices

    backtest["Prediction"] = (
        predictions
    )

    backtest["Signal"] = (
        signals
    )

    backtest["Market_Return"] = (
        compute_forward_returns(
            prices
        )
    )

    backtest["Strategy_Return_Gross"] = (
        backtest["Signal"]
        * backtest["Market_Return"]
    )

    transaction_cost = (
        transaction_cost_bps
        / 10_000
    )

    backtest["Transaction_Cost"] = (
        backtest["Signal"]
        * transaction_cost
    )

    backtest["Strategy_Return"] = (
        backtest["Strategy_Return_Gross"]
        - backtest["Transaction_Cost"]
    )

    backtest = backtest.dropna(
        subset=["Market_Return"]
    ).copy()

    if backtest.empty:
        raise ValueError(
            "No observations are available for the backtest."
        )

    backtest[
        "Strategy_Cumulative_Return"
    ] = (
        1
        + backtest["Strategy_Return"]
    ).cumprod() - 1

    backtest[
        "Market_Cumulative_Return"
    ] = (
        1
        + backtest["Market_Return"]
    ).cumprod() - 1

    backtest["Strategy_Equity"] = (
        initial_capital
        * (
            1
            + backtest["Strategy_Return"]
        ).cumprod()
    )

    backtest["Market_Equity"] = (
        initial_capital
        * (
            1
            + backtest["Market_Return"]
        ).cumprod()
    )

    trade_returns = backtest.loc[
        backtest["Signal"] == 1,
        "Strategy_Return",
    ]

    total_trades = int(
        len(trade_returns)
    )

    winning_trades = int(
        (trade_returns > 0).sum()
    )

    losing_trades = int(
        (trade_returns < 0).sum()
    )

    flat_trades = int(
        (trade_returns == 0).sum()
    )

    win_rate = (
        winning_trades
        / total_trades
        if total_trades > 0
        else np.nan
    )

    final_capital = float(
        backtest[
            "Strategy_Equity"
        ].iloc[-1]
    )

    pnl = (
        final_capital
        - initial_capital
    )

    cumulative_return = (
        final_capital
        / initial_capital
        - 1
    )

    strategy_volatility = float(
        backtest[
            "Strategy_Return"
        ].std(ddof=1)
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )

    sharpe_ratio = (
        compute_sharpe_ratio(
            returns=backtest[
                "Strategy_Return"
            ],
            risk_free_rate=risk_free_rate,
            annualization_factor=TRADING_DAYS_PER_YEAR,
        )
    )

    max_drawdown = (
        compute_max_drawdown(
            backtest[
                "Strategy_Equity"
            ]
        )
    )

    if total_trades > 0:
        average_trade_return = float(
            trade_returns.mean()
        )

        best_trade = float(
            trade_returns.max()
        )

        worst_trade = float(
            trade_returns.min()
        )

    else:
        average_trade_return = np.nan
        best_trade = np.nan
        worst_trade = np.nan

    exposure = float(
        backtest["Signal"].mean()
    )

    benchmark_return = float(
        backtest[
            "Market_Equity"
        ].iloc[-1]
        / initial_capital
        - 1
    )

    statistics = {
        "Initial Capital": initial_capital,
        "Final Capital": final_capital,
        "P&L": pnl,
        "Cumulative Return": cumulative_return,
        "Total Trades": total_trades,
        "Winning Trades": winning_trades,
        "Losing Trades": losing_trades,
        "Flat Trades": flat_trades,
        "Win Rate": win_rate,
        "Average Trade Return": average_trade_return,
        "Best Trade": best_trade,
        "Worst Trade": worst_trade,
        "Annualized Volatility": strategy_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
        "Market Exposure": exposure,
        "Benchmark Return": benchmark_return,
        "Transaction Cost (bps)": transaction_cost_bps,
    }

    return (
        backtest,
        statistics,
    )


def statistics_to_dataframe(
    statistics: dict,
) -> pd.DataFrame:
    return pd.DataFrame(
        statistics.items(),
        columns=[
            "Metric",
            "Value",
        ],
    )