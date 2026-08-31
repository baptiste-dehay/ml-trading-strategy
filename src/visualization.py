from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(
    path: str | Path | None,
) -> None:
    if path is None:
        return

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        path,
        bbox_inches="tight",
        dpi=300,
    )


def plot_price(
    data: pd.DataFrame,
    title: str = "CAC 40 Closing Price",
    save_path: str | Path | None = None,
) -> None:
    if "Close" not in data.columns:
        raise ValueError(
            "The DataFrame must contain a Close column."
        )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        data.index,
        data["Close"],
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("CAC 40")
    plt.grid(alpha=0.3)

    save_figure(save_path)

    plt.show()


def plot_training_history(
    history,
    save_path: str | Path | None = None,
) -> None:
    history_data = (
        history.history
        if hasattr(history, "history")
        else history
    )

    if "loss" not in history_data:
        raise ValueError(
            "Training history does not contain loss."
        )

    epochs = range(
        1,
        len(history_data["loss"]) + 1,
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        epochs,
        history_data["loss"],
        label="Training Loss",
    )

    if "val_loss" in history_data:
        plt.plot(
            epochs,
            history_data["val_loss"],
            label="Validation Loss",
        )

    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path is not None:
        path = Path(save_path)
        loss_path = (
            path.parent
            / f"{path.stem}_loss{path.suffix}"
        )
    else:
        loss_path = None

    save_figure(loss_path)

    plt.show()

    if "accuracy" in history_data:
        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            epochs,
            history_data["accuracy"],
            label="Training Accuracy",
        )

        if "val_accuracy" in history_data:
            plt.plot(
                epochs,
                history_data["val_accuracy"],
                label="Validation Accuracy",
            )

        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(alpha=0.3)

        if save_path is not None:
            path = Path(save_path)
            accuracy_path = (
                path.parent
                / f"{path.stem}_accuracy{path.suffix}"
            )
        else:
            accuracy_path = None

        save_figure(accuracy_path)

        plt.show()


def plot_confusion_matrix(
    matrix,
    class_names: tuple[str, str] = (
        "No Increase",
        "Increase",
    ),
    save_path: str | Path | None = None,
) -> None:
    matrix = np.asarray(matrix)

    if matrix.shape != (2, 2):
        raise ValueError(
            "Confusion matrix must have shape (2, 2)."
        )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(matrix)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    positions = np.arange(2)

    plt.xticks(
        positions,
        class_names,
    )

    plt.yticks(
        positions,
        class_names,
    )

    threshold = (
        matrix.max() / 2
        if matrix.size
        else 0
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
            )

    plt.tight_layout()

    save_figure(save_path)

    plt.show()


def plot_prediction_distribution(
    probabilities,
    threshold: float = 0.5,
    bins: int = 30,
    save_path: str | Path | None = None,
) -> None:
    probabilities = np.asarray(
        probabilities
    ).reshape(-1)

    if probabilities.size == 0:
        raise ValueError(
            "probabilities cannot be empty."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    plt.figure(
        figsize=(10, 5)
    )

    plt.hist(
        probabilities,
        bins=bins,
        alpha=0.8,
    )

    plt.axvline(
        threshold,
        linestyle="--",
        label=f"Threshold = {threshold:.2f}",
    )

    plt.title(
        "Distribution of Predicted Probabilities"
    )

    plt.xlabel(
        "Predicted Probability"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    save_figure(save_path)

    plt.show()


def plot_cumulative_returns(
    backtest: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    required_columns = [
        "Strategy_Cumulative_Return",
        "Market_Cumulative_Return",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in backtest.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        backtest.index,
        backtest[
            "Strategy_Cumulative_Return"
        ],
        label="ML Strategy",
    )

    plt.plot(
        backtest.index,
        backtest[
            "Market_Cumulative_Return"
        ],
        label="CAC 40 Buy & Hold",
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.title(
        "Cumulative Returns"
    )

    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(alpha=0.3)

    save_figure(save_path)

    plt.show()


def plot_equity_curve(
    backtest: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    required_columns = [
        "Strategy_Equity",
        "Market_Equity",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in backtest.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        backtest.index,
        backtest["Strategy_Equity"],
        label="ML Strategy",
    )

    plt.plot(
        backtest.index,
        backtest["Market_Equity"],
        label="CAC 40 Buy & Hold",
    )

    plt.title(
        "Portfolio Equity Curve"
    )

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(alpha=0.3)

    save_figure(save_path)

    plt.show()


def plot_drawdown(
    backtest: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    if "Strategy_Equity" not in backtest.columns:
        raise ValueError(
            "The DataFrame must contain Strategy_Equity."
        )

    equity = backtest[
        "Strategy_Equity"
    ]

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
        - 1
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        backtest.index,
        drawdown,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.title(
        "Strategy Drawdown"
    )

    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.3)

    save_figure(save_path)

    plt.show()