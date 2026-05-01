import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from yapftests import yapf_test

TICKERS = ["SPY", "AAPL", "MSFT"]  # change selon ton besoin
START = "2015-01-01"
END = "2024-12-31"

raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True)

# On bosse sur le Close (MultiIndex si plusieurs tickers)
prices = raw["Close"]  # shape : (n_days, n_tickers)

print(f"Prix téléchargés : {prices.shape[0]} jours, {prices.shape[1]} tickers")


def make_feature(prices_series: pd.Series, lags=(1, 2, 5), windows=(5, 10, 20)):
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
    df["target_clf"] = (df["target_ret_1d"] > 0).astype(int)  # ← ajoute ça ici

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

df = features_dict["SPY"]

feature_cols = [c for c in df.columns if c not in ("target_ret_1d", "target_clf")]
print(f"\n[✓] Features utilisées : {feature_cols}")

# ── Split temporel 80/20 — jamais de shuffle sur du time series
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[feature_cols].values
y_train = train_df["target_clf"].values
X_test = test_df[feature_cols].values
y_test = test_df["target_clf"].values

print(f"Train on : {X_train.shape[0]} days")
print(f"[✓] Test on  : {X_test.shape[0]} days")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class FinanceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


train_loader = DataLoader(FinanceDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(FinanceDataset(X_test, y_test))

class LinearClassifier(nn.Module):
    def __init__(self, input_size: int, n_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_size, n_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

n_features = X_train.shape[1]
model = LinearClassifier(n_features)

print(f"\n[✓] Modèle : {model}")
print(f"[✓] Paramètres : {sum(p.numel() for p in model.parameters())}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Training LOOP

N_EPOCHS = 50
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    for X_batch, y_batch in train_loader:
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item() * len(y_batch)
        epoch_correct += (logits.argmax(1) == y_batch).sum().item()
        epoch_total += len(y_batch)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1:3d}/{N_EPOCHS} | "
              f"Loss: {epoch_loss / epoch_total:.4f} | "
              f"Acc: {epoch_correct / epoch_total:.3f}")

# EVALUATION

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch).argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y_batch.numpy())
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print(f"\n[✓] Accuracy test : {accuracy_score(all_labels, all_preds):.3f}")
print(classification_report(all_labels, all_preds,
                             target_names=["Baisse (0)", "Hausse (1)"]))

weights = model.linear.weight.detach().numpy()  # (2, n_features)
importance = pd.Series(
    np.abs(weights[1] - weights[0]),
    index=feature_cols
).sort_values(ascending=False)

print("Importance des features :")
print(importance.round(4).to_string())
















