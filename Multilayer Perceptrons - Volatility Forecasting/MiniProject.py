from xml.sax.handler import all_features

import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
TICKERS = [
    "AAPL",  # Tech
    "MSFT",  # Tech
    "JPM",   # Finance
    "GS",    # Finance
    "XOM",   # Énergie
    "JNJ",   # Santé
    "GOOG",  # Tech
    "TSLA",  # Auto/Tech
    "NVDA",   # Commodités (Gold ETF)
    "SPY",   # Indice (S&P500 ETF)
]

START = "2022-01-01"
END   = "2024-12-31"

def download_data(tickers, start, end):
    raw = yf.download(tickers, start, end, auto_adjust=True)
    close = raw["Close"]
    volume = raw["Volume"]
    return close, volume

close, volume = download_data(TICKERS, START, END)

def compute_realized_vol(close, window=5):
    log_ret = np.log(close / close.shift(1))
    realized_vol = log_ret.rolling(window).std() * np.sqrt(252)
    target_vol = realized_vol.shift(window)
    return log_ret, target_vol

def compute_features(close, volume):
    log_ret = np.log(close / close.shift(1))
    all_features = {}
    for ticker in close.columns:
        ret = log_ret[ticker]
        vol = volume[ticker]
        f = pd.DataFrame(index=close.index)
        # lagged returns from j = 1 a j = 5
        for lag in range(1, 5 + 1):
            f[f'ret_log{lag}'] = ret.shift(lag)
        f["rolling_vol_5"] = ret.rolling(window=5).std() * np.sqrt(252)
        f["rolling_vol_21"] = ret.rolling(window=21).std() * np.sqrt(252)

        # == RSI on 14jrs == #
        delta = ret.copy()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        f['rsi_14'] = 100 - (100/(1 + rs))
        f["volume_change"] = vol / (vol.rolling(5).mean() + 1e-9)
        all_features[ticker] = f
        return all_features, log_ret

def build_dataset(close, volume, vol_window=5):
    log_ret, target = compute_realized_vol(close, vol_window)
    features_all, _ = compute_features(close, volume)

    dfs = []
    for ticker in close.columns:
        df = features_all[ticker].copy()
        df["target"] = target[ticker]
        df["ticker"] = ticker
        dfs.append(df)

    data = pd.concat(dfs)
    data = data.dropna()  # Vire les NaN dus aux rolling/lags

    print(f"Dataset shape: {data.shape}")
    print(f"Features: {[c for c in data.columns if c not in ['target','ticker']]}")
    return data


def prepare_tensors(data):
    features_col = [c for c in data.columns if c not in ['target','ticker']]
    X = data[features_col].values
    y = data['target'].values.reshape(-1, 1)

    # Split temporel
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Normalisation — fit sur train seulement, transform sur les deux
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    # Conversion en tenseurs
    X_train_t = torch.FloatTensor(X_train)
    X_test_t = torch.FloatTensor(X_test)
    y_train_t = torch.FloatTensor(y_train)
    y_test_t = torch.FloatTensor(y_test)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    return train_loader, test_loader, scaler_y

data = build_dataset(close, volume)

train_loader, test_loader, scaler_y = prepare_tensors(data)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))  # Output : 1 valeur de vol
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


input_dim = len([c for c in data.columns if c not in ["target", "ticker"]])
model = MLP(input_dim=input_dim)
print(model)


def train_model(model, train_loader, test_loader, epochs=100, lr=1e-3, weight_decay=1e-4):
    # Weight decay = L2 regularisation directement dans l'optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Learning rate scheduler — réduit le lr si le loss stagne
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        # -- Phase train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # -- Phase eval
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                pred = model(X_batch)
                test_loss += criterion(pred, y_batch).item()

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        scheduler.step(test_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    return train_losses, test_losses


train_losses, test_losses = train_model(model, train_loader, test_loader)


def evaluate_models(model, test_loader, scaler_y, data):
    feature_cols = [c for c in data.columns if c not in ["target", "ticker"]]
    split = int(len(data) * 0.8)

    X = data[feature_cols].values
    y = data["target"].values.reshape(-1, 1)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # -- Normalisation (même scaler que pour le MLP)
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # ---- BASELINE : Ridge Regression (L2 comme le MLP)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)
    y_pred_ridge = ridge.predict(X_test_s)  # déjà dans l'espace original

    # ---- MLP predictions
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds.append(model(X_batch).numpy())

    y_pred_mlp_scaled = np.concatenate(preds)
    y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_scaled)  # on remet à l'échelle

    return y_test, y_pred_ridge, y_pred_mlp


def print_metrics(y_test, y_pred_ridge, y_pred_mlp):
    def metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        # R² = part de variance expliquée par le modèle
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        return rmse, mae, r2

    rmse_r, mae_r, r2_r = metrics(y_test, y_pred_ridge)
    rmse_m, mae_m, r2_m = metrics(y_test, y_pred_mlp)

    print(f"{'':20} {'Ridge':>10} {'MLP':>10}")
    print(f"{'RMSE':20} {rmse_r:>10.4f} {rmse_m:>10.4f}")
    print(f"{'MAE':20} {mae_r:>10.4f} {mae_m:>10.4f}")
    print(f"{'R²':20} {r2_r:>10.4f} {r2_m:>10.4f}")


y_test, y_pred_ridge, y_pred_mlp = evaluate_models(model, test_loader, scaler_y, data)
print_metrics(y_test, y_pred_ridge, y_pred_mlp)


def plot_results(y_test, y_pred_ridge, y_pred_mlp, train_losses, test_losses):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -- Plot 1 : Prédictions vs Réel
    ax = axes[0]
    ax.plot(y_test[:100], label="Réel", color="black", linewidth=1.5)
    ax.plot(y_pred_ridge[:100], label="Ridge", color="blue", linestyle="--")
    ax.plot(y_pred_mlp[:100], label="MLP", color="red", linestyle="--")
    ax.set_title("Volatilité réalisée : Réel vs Prédictions")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Volatilité annualisée")
    ax.legend()

    # -- Plot 2 : Scatter plot MLP
    ax = axes[1]
    ax.scatter(y_test, y_pred_mlp, alpha=0.3, color="red", s=10)
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], "k--")  # diagonale parfaite
    ax.set_title("MLP : Prédit vs Réel")
    ax.set_xlabel("Réel")
    ax.set_ylabel("Prédit")

    # -- Plot 3 : Learning curves
    ax = axes[2]
    ax.plot(train_losses, label="Train loss", color="blue")
    ax.plot(test_losses, label="Test loss", color="orange")
    ax.set_title("Learning curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    plt.show()


plot_results(y_test, y_pred_ridge, y_pred_mlp, train_losses, test_losses)