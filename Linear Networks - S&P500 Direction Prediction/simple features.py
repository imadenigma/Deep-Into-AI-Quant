import torch
import yfinance as yf
import pandas as pd 
import numpy as np

TICKERS = ["SPY", "AAPL", "MSFT"]   # change selon ton besoin
START   = "2015-01-01"
END     = "2024-12-31"
 
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True)
 
# On bosse sur le Close (MultiIndex si plusieurs tickers)
prices = raw["Close"]               # shape : (n_days, n_tickers)

print(f"Prix téléchargés : {prices.shape[0]} jours, {prices.shape[1]} tickers")

def make_feature(prices_series: pd.Series, lags = (1, 2, 5), windows = (5,10,20)):
    df = pd.DataFrame(index=prices_series.index)
    df["close"] = prices_series
    df["ret_1d"] = df["close"].pct_change()
    for lag in lags:
        df[f"ret_lag_{lag}"] = df["ret_1d"].shift(lag)
    for w in windows:
        df[f"roll_mean_{w}d"] = df["ret_1d"].rolling(w).mean()
    for w in windows:
        df[f"roll_std_{w}d"] = df["ret_1d"].rolling(w).std()
    df["target_ret_1d"] = df["ret_1d"].shift(-1)
    df.dropna(inplace=True)

    return df
features_dict = {}
for ticker in TICKERS:
    feat = make_feature(prices[ticker])
    features_dict[ticker] = feat
    print(f"[✓] {ticker} — {feat.shape[0]} lignes, {feat.shape[1]} colonnes")

# ── Exemple : afficher les premières lignes pour SPY
print("\n── SPY features (head) ──")
print(features_dict["SPY"].head())

print("\n── Colonnes disponibles ──")
print(features_dict["SPY"].columns.tolist())


# ─────────────────────────────────────────
# BONUS — Concat tous les tickers (panel)
# ─────────────────────────────────────────

panel = pd.concat(features_dict, names=["ticker", "date"])
print(f"\n[✓] Panel complet : {panel.shape}")