# Week 2 — Multilayer Perceptrons (Ch. 4-5)

## Objective

Go beyond linear models. Understand overfitting in financial data (it's extreme), add regularization, and build a custom financial feature layer. By the end of the week you'll have an MLP that predicts realized volatility.

## D2L Chapters

| Day | Theory | Practice |
|-----|--------|----------|
| Mon | Ch. 4.1-4.2 — MLP, activations | Implement MLP from scratch, compare ReLU vs Tanh on financial data |
| Tue | Ch. 4.3-4.5 — Overfitting, weight decay, dropout | Add regularization to your Week 1 model — observe the overfit (very real in finance) |
| Wed | Ch. 4.6-4.7 — Forward/backward prop, numerical stability | Debug: why do gradients explode on raw returns? (hint: normalize) |
| Thu | Ch. 5 — Computation, GPU, custom layers | Build a custom `FinancialFeatures` layer (rolling stats, RSI) |
| Fri | Synthesis | Weekend mini-project |

## Weekend Mini-Project

MLP that predicts **5-day realized volatility** on 10 stocks:
1. Download 2+ years of data for 10 tickers (mix of sectors)
2. Engineer features: lagged returns, rolling vol, RSI, volume change
3. Train MLP with dropout and weight decay
4. Compare against linear regression baseline

## Deliverable

Notebook: **"MLP vs Linear Regression for Volatility Prediction"**

- Clean comparison with proper metrics
- Sharpe ratio, MAE, directional accuracy
- Overfitting analysis: training vs validation curves

## Key Concepts

- Universal approximation theorem and why MLPs can fit anything (including noise)
- Overfitting in finance: low signal-to-noise ratio makes it severe
- Dropout as a regularization technique
- Numerical stability: gradient clipping, input normalization
- Custom `nn.Module` layers

## Custom FinancialFeatures Layer

```python
class FinancialFeatures(nn.Module):
    def __init__(self, window=14):
        super().__init__()
        self.window = window

    def forward(self, x):
        # x: (batch, time, features)
        # compute rolling stats, RSI, etc.
        ...
```

## Data Sources

- `yfinance`: 10 tickers of your choice (e.g. AAPL, MSFT, JPM, GS, XOM, ...)
- Realized volatility = std of daily returns over 5-day window

## Tips

- Always normalize inputs before feeding into an MLP
- Plot training vs validation loss — overfitting in finance appears very quickly
- Weight decay (L2) often works better than dropout alone on tabular financial data
- Keep a time-ordered split: train on years 1-3, validate on year 4, test on year 5
