# ML-Based Equity Strategy with Regime-Aware Leverage

This repository contains a research-oriented implementation of an equity trading strategy for SPY, based on a Random Forest classifier combined with a drawdown-aware leverage overlay.

The goal is not short-term price prediction, but to study whether simple machine-learning signals can be used to **modulate exposure** in a statistically controlled way under realistic, out-of-sample conditions.

---

## Overview

The strategy follows three basic principles:

- Remain fully invested (1.0× exposure) by default.
- Use a probabilistic ML model to estimate short-horizon directionality.
- Increase exposure selectively during drawdowns when a rebound appears likely.

All results are evaluated using strict walk-forward validation to avoid look-ahead bias.

---

## Data

- Instrument: SPY (daily OHLCV)
- Source: Yahoo Finance
- Time span configurable via `START_DATE` / `END_DATE`

Prices are transformed into returns and normalized features to reduce non-stationarity.

---

## Features

The model uses the following inputs:

- Momentum indicators (RSI(14), ROC(10), ROC(30)), compressed into a single PCA factor
- 20-day realized volatility, z-scored per training window
- Distance from the 50-day moving average, z-scored
- Drawdown from a rolling 1-year peak (used as a regime indicator)

All feature transformations are fit on training data only within each walk-forward split.

---

## Model

- Random Forest classifier (scikit-learn)
- Conservative depth and leaf-size constraints
- Target: sign of next-day log return

The model output is interpreted as a probability, not a trading signal by itself.

---

## Exposure Logic

- Base exposure is always 1.0× (fully invested).
- Additional leverage is applied only when:
  - The market is in a drawdown, and
  - The model assigns a sufficiently high probability to an upward move.

Exposure is capped and transaction costs are applied when exposure changes.

---

## Validation

- Walk-forward cross-validation using `TimeSeriesSplit`
- Metrics reported per fold:
  - Train and test accuracy
  - Sharpe ratio
  - Maximum drawdown
  - Average exposure
  - Fraction of leveraged days

An aggregated out-of-sample equity curve is constructed across all folds.

---

## Output

The script prints fold-level diagnostics and a summary across folds, including feature importance averaged over time. This allows inspection of both predictive stability and regime dependence.

---

## Dependencies

```bash
pip install yfinance scikit-learn pandas numpy matplotlib
