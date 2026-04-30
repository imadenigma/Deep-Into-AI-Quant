# Week 1 — Preliminaries & Linear Networks (Ch. 2-3)

## Objective

Build strong foundations in PyTorch and apply linear models to real financial data. By the end of the week you'll have a full pipeline from raw market data to a trained model with a naive backtest.

## D2L Chapters

| Day | Theory | Practice |
|-----|--------|----------|
| Mon | Ch. 2.1-2.3 — Data manipulation, calculus, autograd (speedrun) | Re-implement linear regression from scratch in pure PyTorch (no sklearn) |
| Tue | Ch. 3.1-3.3 — Linear regression | Predict S&P500 next-day closing price using lagged return features |
| Wed | Ch. 3.4-3.6 — Softmax regression | Classify daily returns as {up, flat, down} on 5 years of AAPL data |
| Thu | Ch. 3.7 — Concise implementation | Refactor all code using `nn.Module` cleanly |
| Fri | Review + reading | Weekend mini-project: full pipeline |

## Weekend Mini-Project

Build a complete pipeline:
1. Download historical data with `yfinance`
2. Engineer simple features (lagged returns, rolling mean, rolling std)
3. Train a linear classifier
4. Run a naive backtest (long if predicted up, flat otherwise)
5. Report accuracy and naive Sharpe ratio

## Deliverable

Notebook: **"Linear models for stock direction prediction"**

- Clean `nn.Module` implementation
- Data download → feature engineering → training → evaluation
- Metrics: accuracy, confusion matrix, naive Sharpe

## Key Concepts

- Tensors and autograd in PyTorch
- Linear regression: MSE loss, gradient descent
- Softmax regression: cross-entropy loss, multi-class classification
- Why financial returns are hard to predict linearly (signal-to-noise)

## Data Sources

- `yfinance`: S&P500 (`^GSPC`), AAPL
- Features: lagged returns (t-1, t-2, t-5), rolling mean, rolling std

## Tips

- Normalize your features — raw prices will break gradient descent
- Use a train/validation split that respects time order (no shuffling)
- A random baseline on 3-class classification is ~33% accuracy; beating it matters
