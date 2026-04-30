# Week 3 — CNNs (Ch. 6-7) — Condensed

## Objective

Understand CNNs conceptually and apply them to a finance-specific use case: treating candlestick charts as images and predicting next-day direction. Not the most common quant tool, but it appears in pattern recognition at funds like Two Sigma and makes for a strong CV project.

## D2L Chapters

| Day | Theory | Practice |
|-----|--------|----------|
| Mon | Ch. 6.1-6.4 — Convolutions, padding, channels | Implement conv2D from scratch on a mini example |
| Tue | Ch. 6.5-6.6 — Pooling, LeNet | Train LeNet on Fashion-MNIST (mandatory warm-up) |
| Wed | Ch. 7.1-7.3 — AlexNet, VGG, NiN | Quick read, focus on architectural concepts |
| Thu | Ch. 7.4-7.6 — BatchNorm, ResNet (skip GoogLeNet) | Implement BatchNorm + ResNet in depth |
| Fri | Quant application | Weekend project |

## Weekend Mini-Project

**CNN on candlestick patterns:**
1. Download 5 years of daily OHLCV data for 5-10 stocks
2. Convert each 30-day window into a candlestick chart image (use `mplfinance`)
3. Label each image: did the stock go up or down the next day?
4. Train a small CNN (LeNet-style or ResNet-style) on these images
5. Evaluate: accuracy, Sharpe of signal

## Deliverable

Notebook: **"CNN on Candlestick Patterns"**

- Image generation pipeline (OHLCV → PNG)
- CNN training with BatchNorm
- Honest evaluation: does it beat random? By how much?

## Key Concepts

- Convolution as a local pattern detector — translates to "chart pattern" detection
- BatchNorm: why it matters for training stability
- ResNet residual connections: solves vanishing gradients in deep nets
- Transfer learning: why pretrained ImageNet weights don't transfer well to financial charts

## Generating Candlestick Images

```python
import mplfinance as mpf
import matplotlib.pyplot as plt

def ohlcv_to_image(df_window, save_path):
    mpf.plot(df_window, type='candle', style='charles',
             savefig=save_path, figsize=(2.24, 2.24))
```

## Data Sources

- `yfinance`: any liquid stocks or indices
- `mplfinance`: for rendering candlestick charts

## Tips

- Keep images small (64x64 or 32x32) to keep training fast
- This approach is **controversial** in academic finance — be ready to defend it in interviews
- The real value here is the pipeline, not necessarily the alpha
- Compare against a simple momentum baseline (if yesterday was up, predict up)
- Mention class imbalance: markets go up more than down on average
