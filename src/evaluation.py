from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def validate_binary_targets(
    y_true,
) -> np.ndarray:
    y_true = np.asarray(y_true).reshape(-1)

    if y_true.size == 0:
        raise ValueError("y_true cannot be empty.")

    if np.isnan(y_true.astype(float)).any():
        raise ValueError("y_true contains missing values.")

    unique_values = set(np.unique(y_true))

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            "y_true must contain only binary values 0 and 1."
        )

    return y_true.astype(int)


def probabilities_to_classes(
    probabilities,
    threshold: float = 0.5,
) -> np.ndarray:
    probabilities = np.asarray(probabilities).reshape(-1).astype(float)

    if probabilities.size == 0:
        raise ValueError("probabilities cannot be empty.")

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if np.isnan(probabilities).any():
        raise ValueError(
            "probabilities contain missing values."
        )

    if (
        (probabilities < 0)
        | (probabilities > 1)
    ).any():
        raise ValueError(
            "probabilities must be between 0 and 1."
        )

    return (probabilities >= threshold).astype(int)


def compute_confusion_matrix(
    y_true,
    y_pred,
) -> np.ndarray:
    y_true = validate_binary_targets(y_true)
    y_pred = validate_binary_targets(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    return confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )


def compute_metrics(
    y_true,
    y_pred,
) -> dict:
    y_true = validate_binary_targets(y_true)
    y_pred = validate_binary_targets(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


def get_classification_report(
    y_true,
    y_pred,
) -> str:
    y_true = validate_binary_targets(y_true)
    y_pred = validate_binary_targets(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "No Increase",
            "Increase",
        ],
        zero_division=0,
    )


def classification_report_to_dataframe(
    y_true,
    y_pred,
) -> pd.DataFrame:
    y_true = validate_binary_targets(y_true)
    y_pred = validate_binary_targets(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "No Increase",
            "Increase",
        ],
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report).transpose()


def confusion_matrix_to_dataframe(
    y_true,
    y_pred,
) -> pd.DataFrame:
    matrix = compute_confusion_matrix(
        y_true,
        y_pred,
    )

    return pd.DataFrame(
        matrix,
        index=[
            "Actual 0",
            "Actual 1",
        ],
        columns=[
            "Predicted 0",
            "Predicted 1",
        ],
    )


def evaluate_predictions(
    y_true,
    probabilities,
    threshold: float = 0.5,
) -> dict:
    y_true = validate_binary_targets(y_true)

    probabilities = np.asarray(
        probabilities
    ).reshape(-1).astype(float)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    y_pred = probabilities_to_classes(
        probabilities,
        threshold=threshold,
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    matrix = compute_confusion_matrix(
        y_true,
        y_pred,
    )

    report = get_classification_report(
        y_true,
        y_pred,
    )

    return {
        "predictions": y_pred,
        "probabilities": probabilities,
        "metrics": metrics,
        "confusion_matrix": matrix,
        "classification_report": report,
    }


def evaluate_model(
    model,
    X_test,
    y_test,
    threshold: float = 0.5,
) -> dict:
    probabilities = model.predict(
        X_test,
        verbose=0,
    ).reshape(-1)

    return evaluate_predictions(
        y_true=y_test,
        probabilities=probabilities,
        threshold=threshold,
    )


def metrics_to_dataframe(
    metrics: dict,
) -> pd.DataFrame:
    return pd.DataFrame(
        metrics.items(),
        columns=[
            "Metric",
            "Value",
        ],
    )