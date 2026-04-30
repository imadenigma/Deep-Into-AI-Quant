# Week 5 — Attention & Transformers (Ch. 10-11) ⭐⭐ The Chapter

## Objective

Transformers have taken over finance: from alpha factor models to volatility forecasting to LLM-based sentiment. This week you'll understand attention from first principles, implement it from scratch, and build a Transformer encoder for multi-asset return prediction.

## D2L Chapters

| Day | Theory | Practice |
|-----|--------|----------|
| Mon | Ch. 10.1-10.3 — Attention mechanisms, scoring functions | Really understand QKV — draw diagrams by hand |
| Tue | Ch. 10.4-10.5 — Bahdanau attention, multi-head attention | Implement multi-head attention from scratch |
| Wed | Ch. 10.6-10.7 — Positional encoding, full Transformer | Code a mini Transformer encoder |
| Thu | Ch. 11.1-11.3 — BERT-style models, large-scale pretraining | High-level reading and conceptual understanding |
| Fri | Quant application | Project: Transformer for multi-asset prediction |

## Weekend Mini-Project

**Transformer encoder for multi-asset return prediction** (inspired by Temporal Fusion Transformer):
1. Download daily returns for 20 assets (stocks, ETFs, indices)
2. Use a Transformer encoder to model cross-asset attention (which assets attend to which?)
3. Predict next-day returns for all assets simultaneously
4. Analyze the attention weights: do they reveal economically meaningful relationships?

## Deliverable

Notebook + Blog post: **"Why Transformers for Finance"**

Write a short GitHub blog post (500-800 words) explaining:
- What attention is and why it's useful for time series
- How it differs from RNNs
- What you found in the attention weights
- Honest limitations (data hunger, no inductive bias for time)

This is **gold in interviews** — it shows you can think and communicate, not just code.

## Key Concepts

- **QKV attention**: Query, Key, Value — the core operation
- **Scaled dot-product attention**: why divide by sqrt(d_k)?
- **Multi-head attention**: learn different relationship types simultaneously
- **Positional encoding**: Transformers have no built-in notion of order
- **Self-attention for time series**: each time step attends to all others

## Multi-Head Attention from Scratch

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights
```

## Finance-Specific Considerations

- **Causal masking**: in live trading, you can't attend to future data — use a causal mask
- **Cross-asset attention**: use assets as the sequence dimension (not time) to model correlations
- **Temporal Fusion Transformer (TFT)**: a production-grade architecture for financial forecasting — read the paper

## Data Sources

- `yfinance`: 20 diversified tickers (large caps, sectors, ETFs)
- Use daily log-returns as input features

## Tips

- Start with a small model (2 heads, 2 layers, d_model=64) — financial data is scarce
- Transformer needs more data than LSTM to shine — use a long history (10+ years)
- Visualize attention weights with a heatmap — it's the most impressive part of your notebook
- The positional encoding is critical for time series — experiment with learned vs sinusoidal
- Read the original "Attention is All You Need" paper — you'll be asked about it
