# Week 8 — Capstone Project

## Objective

No more chapters. You build a serious, portfolio-grade project that combines everything from weeks 1-7. This is what you show in interviews and on your GitHub. It must be clean, well-documented, and honestly evaluated.

## Schedule

| Day | Activity |
|-----|----------|
| Mon-Tue | Project conception + data collection + architecture design |
| Wed-Thu | Implementation + training |
| Fri | Backtest + write professional README |

## Choose One Capstone

### Option A — Deep RL Portfolio Optimization
**Multi-asset portfolio optimization with Deep RL (PPO on 10-20 stocks)**

Build a PPO agent that allocates capital across 10-20 assets:
- **State**: recent returns, volatilities, correlations, current weights
- **Actions**: portfolio weight vector (continuous, softmax output)
- **Reward**: risk-adjusted return (Sharpe-like) minus transaction costs
- **Constraint**: weights sum to 1, no leverage

Compare against:
- Equal-weight portfolio
- Markowitz mean-variance optimization
- Momentum strategy

### Option B — Transformer Factor Model
**Transformer-based factor model: predict returns from fundamental + technical features**

Build a Transformer encoder that ingests a panel of features for N stocks:
- **Features**: P/E, P/B, momentum, volatility, analyst revisions, volume
- **Output**: expected return ranking (cross-sectional)
- **Training signal**: next-month return z-score
- **Evaluation**: Information Coefficient (IC), rank IC, long-short portfolio Sharpe

This is the most "quant research" flavored option — closest to what a systematic fund does.

### Option C — Volatility Surface Forecasting
**Forecast implied volatility surface with LSTM/Transformer (very in demand for quant options roles)**

Model the full IV surface (strike × maturity) and forecast it forward:
- **Data**: options data from CBOE or approximated from VIX term structure
- **Architecture**: Transformer or LSTM over time, with strike/maturity as cross-section
- **Output**: forecast IV for each (strike, maturity) cell
- **Evaluation**: RMSE on IV, P&L of delta-hedged straddle strategy

## Deliverable

A production-quality GitHub repository with:

### Required Files

```
week8/
├── README.md          ← this file, but also a professional project README
├── data/
│   └── download.py    ← reproducible data download script
├── src/
│   ├── model.py       ← clean model implementation
│   ├── env.py         ← environment (if RL)
│   ├── train.py       ← training script with config
│   └── backtest.py    ← evaluation and backtest
├── notebooks/
│   └── analysis.ipynb ← results, plots, analysis
└── results/
    └── metrics.json   ← final numbers
```

### Professional README Must Include

1. **Problem statement**: what are you predicting/optimizing and why it matters
2. **Data**: source, time period, preprocessing steps
3. **Architecture**: diagram or clear description
4. **Results**: tables with honest numbers, compared to baselines
5. **Limitations**: what doesn't work and why (shows intellectual honesty)
6. **How to run**: clear instructions, `requirements.txt`

## Evaluation Metrics

Depending on your project, report as many of these as relevant:

| Metric | Use case |
|--------|----------|
| Sharpe ratio | Any trading strategy |
| Max drawdown | Risk assessment |
| Information Coefficient (IC) | Factor models |
| Directional accuracy | Return prediction |
| MAE / RMSE | Regression tasks |
| Cumulative returns chart | Visual, always include |

## What Makes a Great Capstone

- **Honest evaluation**: don't cherry-pick — show the strategy on an out-of-sample period
- **Strong baselines**: beat something real (not just random)
- **Clean code**: someone should be able to clone and run it
- **Visual results**: plots of cumulative returns, loss curves, attention weights
- **Written analysis**: what did you learn? what surprised you?

## Interview Talking Points

You should be able to answer:
- "Walk me through your capstone project in 5 minutes"
- "What were the biggest challenges?"
- "How did you avoid look-ahead bias?"
- "How does it perform out-of-sample?"
- "What would you do with 3 more months?"

## Congratulations

After 8 weeks you have:
- [x] Linear models, MLPs, CNNs, RNNs, Transformers applied to finance
- [x] A full understanding of optimization and training efficiency
- [x] A working RL trading agent
- [x] A portfolio-grade capstone project
- [x] A GitHub repo that demonstrates all of the above

This is a serious foundation for a quant research or quant developer internship.
