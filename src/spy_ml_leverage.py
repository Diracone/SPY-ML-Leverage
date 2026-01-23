# src/spy_ml_leverage.py
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


# -----------------------------
# data cache / ingestion
# -----------------------------
def fetch_prices(ticker: str, start: str, end: str, raw_dir: Path) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"{ticker}_{start}_{end}.parquet"

    if cache_path.exists():
        print(f"Using cached: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Downloading from yfinance...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    df = df.dropna()
    df.to_parquet(cache_path)
    print(f"Saved cache: {cache_path}")
    return df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # reproducibility
    np.random.seed(args.seed)

    # folders
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    plots_dir = out_dir / "plots"
    metrics_dir = out_dir / "metrics"

    processed_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # plotting style
    plt.style.use("bmh")

    # -----------------------------
    # config (same as your notebook)
    # -----------------------------
    TICKER = args.ticker
    START_DATE = args.start
    END_DATE = args.end

    N_SPLITS = 5
    COST_PER_TRADE = 0.0002

    BASE_EXPOSURE = 1.0
    MAX_LEVERAGE_ADD = 1.0

    DD_MED = -0.08
    DD_STRONG = -0.15

    PROB_MED = 0.55
    PROB_STRONG = 0.62

    LEV_ADD_WEAK = 0.0
    LEV_ADD_MED = 0.5
    LEV_ADD_STRONG = 1.0

    RF_PARAMS = dict(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=50,
        max_features="sqrt",
        random_state=args.seed,
        n_jobs=-1
    )

    # -----------------------------
    # data (use cached fetch instead of downloading again)
    # -----------------------------
    print(f"Loading {TICKER} data ({START_DATE} → {END_DATE})...")
    df = fetch_prices(TICKER, START_DATE, END_DATE, raw_dir)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # -----------------------------
    # features (your notebook code unchanged)
    # -----------------------------
    df["fwd_ret_1"] = np.log(df["Close"].shift(-1) / df["Close"])
    df["log_ret_1"] = np.log(df["Close"] / df["Close"].shift(1))

    df["vol_20"] = df["log_ret_1"].rolling(20).std()
    df["dist_sma50"] = (df["Close"] / df["Close"].rolling(50).mean()) - 1

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["roc_10"] = df["Close"].pct_change(10)
    df["roc_30"] = df["Close"].pct_change(30)

    df["roll_peak_252"] = df["Close"].rolling(252).max()
    df["dd_252"] = df["Close"] / df["roll_peak_252"] - 1

    df["target"] = (df["fwd_ret_1"] > 0).astype(int)

    df.dropna(inplace=True)

    momentum_cols = ["rsi_14", "roc_10", "roc_30"]

    # Optionally persist a processed snapshot (nice “data engineering” touch)
    processed_path = processed_dir / f"{TICKER}_{START_DATE}_{END_DATE}_features.parquet"
    df.to_parquet(processed_path)

    # -----------------------------
    # walk-forward (your notebook code, only small edits for saving)
    # -----------------------------
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    fold_rows = []
    oos_frames = []
    fold_imps = []

    def zscore_by_train(train_s, test_s):
        mu = train_s.mean()
        sd = train_s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        return (train_s - mu) / sd, (test_s - mu) / sd

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        scaler = StandardScaler()
        mom_train = scaler.fit_transform(train[momentum_cols])
        mom_test = scaler.transform(test[momentum_cols])

        pca = PCA(n_components=1, random_state=args.seed)
        train["alpha_factor"] = pca.fit_transform(mom_train).ravel()
        test["alpha_factor"] = pca.transform(mom_test).ravel()

        train["vol_z"], test["vol_z"] = zscore_by_train(train["vol_20"], test["vol_20"])
        train["dist_z"], test["dist_z"] = zscore_by_train(train["dist_sma50"], test["dist_sma50"])

        features = ["alpha_factor", "vol_z", "dist_z", "dd_252"]

        X_train = train[features]
        y_train = train["target"].astype(int)
        X_test = test[features]
        y_test = test["target"].astype(int)

        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train, y_train)

        proba_up = rf.predict_proba(X_test)[:, 1]
        pred = (proba_up >= 0.5).astype(int)

        dd = test["dd_252"].values
        lev_add = np.zeros_like(proba_up)

        lev_add[(dd <= DD_MED) & (proba_up >= PROB_MED)] = LEV_ADD_MED
        lev_add[(dd <= DD_STRONG) & (proba_up >= PROB_STRONG)] = LEV_ADD_STRONG

        exposure = np.clip(BASE_EXPOSURE + lev_add, 0.0, BASE_EXPOSURE + MAX_LEVERAGE_ADD)

        exposure_change = np.abs(np.diff(np.r_[exposure[0], exposure]))
        strat_ret = exposure * test["fwd_ret_1"].values - COST_PER_TRADE * exposure_change
        mkt_ret = test["fwd_ret_1"].values

        baseline_acc = y_test.mean()
        train_acc = accuracy_score(y_train, rf.predict(X_train))
        test_acc = accuracy_score(y_test, pred)

        mu = strat_ret.mean() * 252
        sd = strat_ret.std() * np.sqrt(252)
        sharpe = mu / sd if sd > 0 else 0.0

        cum = np.exp(np.cumsum(strat_ret))
        peak = np.maximum.accumulate(cum)
        max_dd = ((cum - peak) / peak).min()

        fold_rows.append({
            "fold": fold,
            "n_train": len(train),
            "n_test": len(test),
            "baseline_acc": float(baseline_acc),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "sharpe": float(sharpe),
            "max_dd": float(max_dd),
            "avg_prob_up": float(proba_up.mean()),
            "avg_exposure": float(exposure.mean()),
            "pct_leveraged_days": float(np.mean(exposure > 1.0))
        })

        fold_imps.append(rf.feature_importances_)

        oos_frames.append(pd.DataFrame({
            "Date": test.index,
            "mkt_ret": mkt_ret,
            "strat_ret": strat_ret,
            "exposure": exposure,
            "proba_up": proba_up,
            "dd_252": test["dd_252"].values
        }).set_index("Date"))

    # -----------------------------
    # reporting (save + print)
    # -----------------------------
    metrics_df = pd.DataFrame(fold_rows)

    print("\n================ WALK-FORWARD DIAGNOSTICS ================\n")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    metrics_csv = metrics_dir / "walkforward_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    summary = metrics_df[
        ["baseline_acc","train_acc","test_acc","sharpe","max_dd",
         "avg_prob_up","avg_exposure","pct_leveraged_days"]
    ].mean()

    summary_dict = {k: float(v) for k, v in summary.to_dict().items()}
    summary_txt = metrics_dir / "summary.txt"
    with open(summary_txt, "w") as f:
        f.write("SPY ML Leverage Strategy — Summary\n")
        f.write("----------------------------------\n\n")
        for k, v in summary_dict.items():
            f.write(f"{k:<20}: {v:.4f}\n")
    summary_json = metrics_dir / "summary_mean_across_folds.json"
    summary_json.write_text(json.dumps(summary_dict, indent=2))

    avg_imp = np.mean(np.vstack(fold_imps), axis=0)
    imp = pd.Series(avg_imp, index=["alpha_factor","vol_z","dist_z","dd_252"]).sort_values(ascending=False)

    imp_df = imp.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_csv = metrics_dir / "feature_importance_avg.csv"
    imp_df.to_csv(imp_csv, index=False)

    print("\n---------------- Summary (mean across folds) ----------------\n")
    for k, v in summary_dict.items():
        print(f"{k:<18} = {v:.4f}")

    print("\n=== Feature Importance (avg across folds) ===")
    meaning_map = {
        "vol_z": "volatility regime",
        "dd_252": "drawdown from 1Y peak",
        "alpha_factor": "momentum (PCA factor)",
        "dist_z": "distance from trend",
    }
    for k, v in imp.items():
        print(f"{k:<15} {v:.6f}  ({meaning_map.get(k, '')})")

    # -----------------------------
    # equity curves (save plots + stitched OOS)
    # -----------------------------
    oos = pd.concat(oos_frames).sort_index()
    oos["cum_mkt"] = np.exp(oos["mkt_ret"].cumsum())
    oos["cum_strat"] = np.exp(oos["strat_ret"].cumsum())

    oos_csv = metrics_dir / "oos_timeseries.csv"
    oos.to_csv(oos_csv)

    # plot 1: equity curve
    plt.figure(figsize=(12, 5))
    plt.plot(oos.index, oos["cum_mkt"], linestyle="--", alpha=0.6, label="Buy & Hold (OOS)")
    plt.plot(oos.index, oos["cum_strat"], label="RF + Leverage-on-Dips (OOS)")
    plt.legend()
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    eq_path = plots_dir / "equity_curve_oos.png"
    plt.savefig(eq_path, dpi=160)
    plt.close()

    # plot 2: exposure
    plt.figure(figsize=(12, 3))
    plt.plot(oos.index, oos["exposure"])
    plt.ylabel("Exposure")
    plt.xlabel("Date")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    exp_path = plots_dir / "exposure_oos.png"
    plt.savefig(exp_path, dpi=160)
    plt.close()

    print("\nSaved:")
    print(f"  - {metrics_csv}")
    print(f"  - {summary_json}")
    print(f"  - {imp_csv}")
    print(f"  - {oos_csv}")
    print(f"  - {eq_path}")
    print(f"  - {exp_path}")


if __name__ == "__main__":
    main()
