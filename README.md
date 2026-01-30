# Trading SPY Using Machine Learning and Leverage

This repo contains a small research project where I test a machine-learning–based trading strategy on SPY.  
Instead of predicting prices directly, the model is used to **adjust market exposure** in a controlled way.

The emphasis is on **clean data flow, reproducibility, and out-of-sample validation**, rather than maximizing backtest performance.

---

## Idea

The strategy is built around three simple ideas:

- Stay fully invested most of the time (1.0× exposure).
- Use an ML model to estimate the probability of the market going up the next day.
- Add leverage only during drawdowns, when rebounds are historically more likely.

All decisions are evaluated using walk-forward testing to avoid look-ahead bias.

---

## Example Results

**Out-of-sample equity curve**  
![Equity Curve](figures/equity_curve_oos.png)

**Model-driven exposure over time**  
![Exposure](figures/exposure_oos.png)

---

## Data & Pipeline

- Asset: SPY (daily data)
- Source: Yahoo Finance
- Date range controlled via script arguments

The project follows a small **ETL-style batch pipeline**:

- **Extract**: download market data from Yahoo Finance
- **Transform**: compute returns, indicators, normalization, and PCA-based factors
- **Load**: persist processed features, metrics, and plots as artifacts

Raw price data is cached on disk to avoid repeated downloads, while the authoritative dataset is the transformed, ML-ready feature set.

---

## Features

The model uses a compact set of intuitive signals:

- **Momentum**: RSI(14), ROC(10), ROC(30), compressed into a single PCA-based *alpha factor*
- **Volatility**: 20-day realized volatility (z-scored per training window)
- **Trend**: distance from the 50-day moving average
- **Drawdown**: distance from the rolling 1-year peak

All transformations are fit **only on training data** in each walk-forward split to prevent leakage.

---

## Model

- Random Forest classifier (scikit-learn)
- Conservative hyperparameters to limit overfitting
- Target: whether the next day’s return is positive

Predictions are treated as **probabilities**, not binary signals, and are used to modulate exposure rather than trigger frequent trades.

---

## Exposure and Leverage Logic

- Base exposure is always 1.0× (fully invested).
- Additional leverage is applied only when:
  - The market is in a drawdown, and
  - The model assigns a sufficiently high probability to an upward move.

Total exposure is capped, and simple transaction costs are applied when exposure changes.

---

## Validation & Overfitting Control

- Walk-forward cross-validation using `TimeSeriesSplit`
- Train and test metrics are tracked separately per fold

This makes it easy to:
- Compare **training vs out-of-sample accuracy**
- Detect overfitting when training performance diverges from test performance
- See how performance varies across different market regimes

Out-of-sample equity curves are stitched together across folds to reflect realistic strategy behavior.

---

## Outputs

The pipeline produces:

- Fold-level metrics (accuracy, Sharpe, drawdown, exposure)
- Mean summary across folds
- Feature importance averages
- Out-of-sample equity and exposure plots

Results are written as artifacts rather than hard-coded into the code or notebooks.

---

## Reproducibility & Infrastructure

- **Docker**: the entire pipeline is containerized for reproducible execution
- **Kubernetes**: the backtest can be run as a batch job using a Kubernetes Job manifest
- **CI**: GitHub Actions runs a short sanity backtest on every push to validate integration and prevent silent breakage

The goal is to treat the strategy as a small **production-style research pipeline**, not a one-off notebook.

---

## Dependencies

```bash
pip install yfinance scikit-learn pandas numpy matplotlib
