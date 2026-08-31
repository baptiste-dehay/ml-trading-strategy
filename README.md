# CAC 40 Machine Learning Trading Strategy

Machine Learning project developed as part of the
"Machine Learning sous Python" course.

The objective is to build and backtest a trading strategy
on the CAC 40 index using a Feedforward Neural Network.

## Objective

The model aims to predict an upward movement of the CAC 40
at J+1 using historical market data and technical indicators.

Financial instrument:

- CAC 40 Index
- Yahoo Finance ticker: `^FCHI`
- Period: 2017-01-01 to 2022-01-01

## Features

The model uses:

- Close return versus J-1
- Close return versus J-8
- ADX (20)
- RSI (20)
- Stochastic Oscillator (14, 3)
- CCI (20)
- 10-day volatility

## Model

Feedforward Neural Network using TensorFlow/Keras.

Architecture:

- Hidden Layer 1: 512 neurons, ELU
- Hidden Layer 2: 256 neurons, ReLU
- Hidden Layer 3: 128 neurons, ELU
- Hidden Layer 4: 32 neurons, ReLU
- Output Layer: 1 neuron, Sigmoid
- Dropout: 15%

## Repository structure

```text
data/       Raw and processed market data
models/     Trained model artifacts
notebooks/  Jupyter notebooks
results/    Metrics, figures and backtest outputs
src/        Reusable Python source code