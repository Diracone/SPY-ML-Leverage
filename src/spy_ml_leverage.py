# src/spy_ml_leverage.py
# Main entry point for the SPY ML + leverage backtest.
# The goal is to keep data, features, modeling, and outputs clearly separated
# so experiments are reproducible and easy to rerun.

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


# ============================================================
# DATA INGESTION + CACHING
# ============================================================
def fetch_prices(ticker: str, start: str, end: str, raw_dir: Path) -> pd.DataFrame:
    """
    Download market data once and cache it locally.
    If the file already exists, reuse it instead of hitting the API again.
    This makes runs faster and ensures reproducibility.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"{ticker}_{start}_{end}.parquet"

    if cache_path.exists():
        print(f"Using cached data: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Downloading data from yfinance...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    df = df.dropna()
    df.to_parquet(cache_path)
    print(f"Saved cache: {cache_path}")
    return df


# ============================================================
# ARGUMENT PARSING
# ============================================================
def parse_args():
    """
    Keep configuration outside the code so the same pipeline
    can be reused for different tickers or time ranges.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2020-01-01")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    args = parse_args()

    # Fix randomness so results are comparable across runs
    np.random.seed(args.seed)

    # --------------------------------------------------------
    # Folder layout (data engineering hygiene)
    # --------------------------------------------------------
    # Raw data, processed features, and outputs are kept separate
    # to avoid accidental leakage or overwriting.
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    plots_dir = out_dir / "plots"
    metrics_dir = out_dir / "metrics"

    processed_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("bmh")

    # --------------------------------------------------------
    # Strategy and model configuration
    # --------------------------------------------------------
    # These are kept explicit so it’s clear what assumptions
    # the strategy is making.
    N_SPLITS = 5
    COST_PER_TRADE = 0.0002  # simple transaction cost proxy

    BASE_EXPOSURE = 1.0
    MAX_LEVERAGE_ADD = 1.0

    # Drawdown thresholds used to decide when to add leverage
    DD_MED = -0.08
    DD_STRONG = -0.15

    # Probability thresholds from the classifier
    PROB_MED = 0.55
    PROB_STRONG = 0.62

    LEV_ADD_MED = 0.5
    LEV_ADD_STRONG = 1.0

    # Conservative RF settings to reduce overfitting
    RF_PARAMS = dict(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=50,
        max_features="sqrt",
        random_state=args.seed,
        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Load data (from cache if available)
    # --------------------------------------------------------
    df = fetch_prices(args.ticker, args.start, args.end, raw_dir)

    # yfinance sometimes returns multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Only keep fields actually used downstream
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # --------------------------------------------------------
    # CI / sanity-check guard (CORRECT PLACE)
    # --------------------------------------------------------
    MIN_ROWS = 300  # due to 252-day rolling features

    if len(df) < MIN_ROWS:
        print(
            f"Not enough data for walk-forward backtest "
            f"({len(df)} rows < {MIN_ROWS}). "
            "Exiting early (CI sanity check)."
        )
        return
    # exits main() cleanly
    
    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================
    # We work mostly with returns and relative measures,
    # since prices themselves are non-stationary.
    df["fwd_ret_1"] = np.log(df["Close"].shift(-1) / df["Close"])
    df["log_ret_1"] = np.log(df["Close"] / df["Close"].shift(1))

    # Volatility and trend distance
    df["vol_20"] = df["log_ret_1"].rolling(20).std()
    df["dist_sma50"] = df["Close"] / df["Close"].rolling(50).mean() - 1

    # RSI as a bounded momentum indicator
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Short and medium-term momentum
    df["roc_10"] = df["Close"].pct_change(10)
    df["roc_30"] = df["Close"].pct_change(30)

    # Drawdown from rolling 1-year peak
    df["roll_peak_252"] = df["Close"].rolling(252).max()
    df["dd_252"] = df["Close"] / df["roll_peak_252"] - 1

    # Binary prediction target: next-day direction
    df["target"] = (df["fwd_ret_1"] > 0).astype(int)

    # Drop rows where rolling features are not yet defined
    df = df.dropna()

    # Persist processed features so modeling is deterministic
    df.to_parquet(
        processed_dir / f"{args.ticker}_{args.start}_{args.end}_features.parquet"
    )

    momentum_cols = ["rsi_14", "roc_10", "roc_30"]

    # ========================================================
    # WALK-FORWARD MODELING
    # ========================================================
    # TimeSeriesSplit avoids look-ahead bias by respecting time order
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    fold_rows = []
    oos_frames = []
    fold_imps = []

    def zscore_by_train(train_s, test_s):
        """
        Normalize using training statistics only
        to avoid leaking information from the future.
        """
        mu = train_s.mean()
        sd = train_s.std(ddof=0) or 1.0
        return (train_s - mu) / sd, (test_s - mu) / sd

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        # Reduce correlated momentum features to one factor
        scaler = StandardScaler()
        mom_train = scaler.fit_transform(train[momentum_cols])
        mom_test = scaler.transform(test[momentum_cols])

        pca = PCA(n_components=1, random_state=args.seed)
        train["alpha_factor"] = pca.fit_transform(mom_train).ravel()
        test["alpha_factor"] = pca.transform(mom_test).ravel()

        # Z-score regime features using training window only
        train["vol_z"], test["vol_z"] = zscore_by_train(train["vol_20"], test["vol_20"])
        train["dist_z"], test["dist_z"] = zscore_by_train(
            train["dist_sma50"], test["dist_sma50"]
        )

        features = ["alpha_factor", "vol_z", "dist_z", "dd_252"]

        # Train classifier on past data only
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(train[features], train["target"])

        # Predict probability of an up move
        proba_up = rf.predict_proba(test[features])[:, 1]

        # Leverage logic: add exposure only in drawdowns
        # and only when model confidence is high
        dd = test["dd_252"].values
        lev_add = np.zeros_like(proba_up)
        lev_add[(dd <= DD_MED) & (proba_up >= PROB_MED)] = LEV_ADD_MED
        lev_add[(dd <= DD_STRONG) & (proba_up >= PROB_STRONG)] = LEV_ADD_STRONG

        exposure = np.clip(
            BASE_EXPOSURE + lev_add, 0, BASE_EXPOSURE + MAX_LEVERAGE_ADD
        )

        # Simple transaction cost based on exposure changes
        exposure_change = np.abs(np.diff(np.r_[exposure[0], exposure]))

        strat_ret = (
            exposure * test["fwd_ret_1"].values
            - COST_PER_TRADE * exposure_change
        )
        mkt_ret = test["fwd_ret_1"].values

        sharpe = (
            strat_ret.mean() * 252
            / (strat_ret.std() * np.sqrt(252))
            if strat_ret.std() > 0
            else 0
        )

        fold_rows.append({
            "fold": fold,
            "test_acc": accuracy_score(
                test["target"], (proba_up >= 0.5).astype(int)
            ),
            "sharpe": sharpe,
            "avg_exposure": exposure.mean(),
        })

        fold_imps.append(rf.feature_importances_)

        # Store out-of-sample returns for later stitching
        oos_frames.append(
            pd.DataFrame({
                "Date": test.index,
                "mkt_ret": mkt_ret,
                "strat_ret": strat_ret,
                "exposure": exposure,
            }).set_index("Date")
        )

    # ========================================================
    # SAVE METRICS
    # ========================================================
    metrics_df = pd.DataFrame(fold_rows)
    metrics_df.to_csv(metrics_dir / "walkforward_metrics.csv", index=False)

    summary = metrics_df.mean().to_dict()
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Human-readable summary for quick inspection
    with open(metrics_dir / "summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k:<15}: {v:.4f}\n")

    # ========================================================
    # SAVE PLOTS
    # ========================================================
    oos = pd.concat(oos_frames).sort_index()
    oos["cum_mkt"] = np.exp(oos["mkt_ret"].cumsum())
    oos["cum_strat"] = np.exp(oos["strat_ret"].cumsum())

    plt.figure(figsize=(12, 5))
    plt.plot(oos["cum_mkt"], "--", label="Buy & Hold")
    plt.plot(oos["cum_strat"], label="Strategy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "equity_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
