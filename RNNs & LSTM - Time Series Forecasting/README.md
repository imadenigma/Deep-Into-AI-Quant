# Week 4 — RNNs (Ch. 8-9) ⭐ The Crucial Week

## Objective

This is the most important week for quant finance. Time series IS sequential data, and RNNs/LSTMs are the natural fit. You'll implement everything from scratch, understand BPTT (asked in quant interviews), and build a serious LSTM forecaster benchmarked against ARIMA.

## D2L Chapters

| Day | Theory | Practice |
|-----|--------|----------|
| Mon | Ch. 8.1-8.3 — Sequences, preprocessing, language models | Read with "financial time series" in mind throughout |
| Tue | Ch. 8.4-8.6 — RNN from scratch + concise | RNN from scratch predicting next-day returns |
| Wed | Ch. 8.7 — BPTT (Backprop Through Time) | Work through the math carefully — this is an interview question |
| Thu | Ch. 9.1-9.2 — GRU, LSTM | LSTM on multi-asset return sequences |
| Fri | Ch. 9.3-9.7 — Deep RNN, Bi-RNN, encoder-decoder | Project: LSTM multi-step forecasting on BTC or EUR/USD |

## Weekend Mini-Project

**LSTM vs ARIMA — complete benchmark:**
1. Pick a liquid asset (BTC/USD or EUR/USD for non-stationarity fun)
2. Fit ARIMA with proper order selection (AIC/BIC)
3. Train LSTM with sequence length tuning
4. Multi-step forecasting: predict 1, 3, 5 days ahead
5. Report: MAE, RMSE, directional accuracy, Sharpe of each signal

## Deliverable

Notebook: **"LSTM vs ARIMA — Complete Benchmark"**

This is a **classic quant interview question**: "How does deep learning compare to classical time series models on financial data?"

Expected honest answer: ARIMA often wins on short horizons; LSTM wins when there are nonlinear dependencies and enough data.

## Key Concepts

- Hidden state as a memory mechanism
- Vanishing gradients in long sequences — why LSTM was invented
- BPTT: gradient flows backwards through time steps
- GRU vs LSTM: GRU is simpler, often similar performance
- Multi-step forecasting: direct vs recursive strategies

## BPTT Interview Question

Be able to explain:
- Why gradients vanish in vanilla RNNs on long sequences
- How LSTM gates (forget, input, output) solve this
- The trade-off between sequence length and training stability

## Architecture Example

```python
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

## Data Sources

- `yfinance`: BTC-USD, EURUSD=X, or any liquid asset
- `statsmodels`: for ARIMA baseline
- Consider: use log-returns, not prices (stationarity)

## Tips

- **Stationarity matters**: model log-returns, not raw prices
- Sequence length is a key hyperparameter — try 10, 20, 60 days
- Use walk-forward validation, not a simple train/test split
- Gradient clipping is essential (`torch.nn.utils.clip_grad_norm_`)
- ARIMA is a strong baseline — don't be surprised if it ties or wins
