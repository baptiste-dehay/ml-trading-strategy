from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

import keras

from keras import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Input,
)
from keras.models import load_model

from src.config import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EPOCHS,
    MODEL_PATH,
    PREDICTION_THRESHOLD,
    RANDOM_SEED,
)


def set_random_seed(
    seed: int = RANDOM_SEED,
) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(
    input_dim: int,
    dropout_rate: float = DROPOUT_RATE,
) -> Sequential:
    if input_dim <= 0:
        raise ValueError(
            "input_dim must be strictly positive."
        )

    if not 0 <= dropout_rate < 1:
        raise ValueError(
            "dropout_rate must be between 0 and 1."
        )

    model = Sequential(
        [
            Input(shape=(input_dim,)),

            Dense(
                512,
                activation="elu",
            ),
            Dropout(dropout_rate),

            Dense(
                256,
                activation="relu",
            ),
            Dropout(dropout_rate),

            Dense(
                128,
                activation="elu",
            ),
            Dropout(dropout_rate),

            Dense(
                32,
                activation="relu",
            ),
            Dropout(dropout_rate),

            Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )

    return model


def compile_model(
    model: Sequential,
) -> Sequential:
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def create_model(
    input_dim: int,
    dropout_rate: float = DROPOUT_RATE,
    random_seed: int = RANDOM_SEED,
) -> Sequential:
    set_random_seed(random_seed)

    model = build_model(
        input_dim=input_dim,
        dropout_rate=dropout_rate,
    )

    model = compile_model(model)

    return model


def train_model(
    model: Sequential,
    X_train,
    y_train,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    validation_data=None,
    verbose: int = 1,
):
    if epochs <= 0:
        raise ValueError(
            "epochs must be strictly positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be strictly positive."
        )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=False,
    )

    return history


def predict_probabilities(
    model: Sequential,
    X,
) -> np.ndarray:
    probabilities = model.predict(
        X,
        verbose=0,
    )

    return probabilities.reshape(-1)


def predict_classes(
    model: Sequential,
    X,
    threshold: float = PREDICTION_THRESHOLD,
) -> np.ndarray:
    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities = predict_probabilities(
        model,
        X,
    )

    return (
        probabilities >= threshold
    ).astype(int)


def save_trained_model(
    model: Sequential,
    path: str | Path = MODEL_PATH,
) -> Path:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(path)

    return path


def load_trained_model(
    path: str | Path = MODEL_PATH,
) -> Sequential:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    return load_model(path)