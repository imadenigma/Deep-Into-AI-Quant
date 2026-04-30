# Week 6 — Optimization & Computational Performance (Ch. 12-13)

## Objective

Understanding optimization is what separates quants who tinker from quants who ship. This week you'll master optimizers, learning rate scheduling, and profiling — then use it to make your Week 5 Transformer train 3x faster.

## D2L Chapters

| Day | Theory | Practice |
|-----|--------|----------|
| Mon | Ch. 12.1-12.4 — Gradient descent, SGD, mini-batch | Compare convergence on your Week 5 model |
| Tue | Ch. 12.5-12.7 — Momentum, Adagrad, RMSProp | Tune optimizers and observe the effect on training curves |
| Wed | Ch. 12.8-12.10 — Adam, LR scheduling | Apply LR scheduling to your LSTM — usually a big boost |
| Thu | Ch. 13.1-13.4 — Compilation, GPU, parallelism | Profile your code and find bottlenecks |
| Fri | Synthesis | Project: optimize Week 5 Transformer training by 3x |

## Weekend Mini-Project

**Optimize the Week 5 Transformer training by 3x:**
1. Profile the current training loop (use `torch.profiler`)
2. Identify bottlenecks: data loading, forward pass, backward pass
3. Apply: mixed-precision training (`torch.cuda.amp`), better data loading, compiled model (`torch.compile`)
4. Benchmark before vs after — report wall time and convergence

## Deliverable

Report: **"Optimization diary"** — a section in your notebook showing:
- Profiling output
- Changes made
- Before/after timing comparison
- Final model performance (should not regress)

## Key Concepts

- **SGD vs Adam**: Adam adapts learning rates per parameter — almost always preferred in deep learning
- **Momentum**: smooths gradient updates, helps escape flat regions
- **Learning rate scheduling**: cosine annealing, warmup + decay — critical for Transformers
- **Mixed precision (fp16/bf16)**: 2x memory reduction, often 2x speed on modern GPUs
- **`torch.compile`**: PyTorch 2.0+ graph compilation — easy 10-30% speedup

## LR Scheduling for Transformers

```python
# Warmup + cosine decay (standard for Transformers)
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

## Profiling with PyTorch

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    train_one_epoch(model, loader)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Optimizer Comparison

| Optimizer | Pros | Cons | Best for |
|-----------|------|------|----------|
| SGD+momentum | Generalizes well | Needs careful LR tuning | CNNs, vision |
| Adam | Fast convergence | Can overfit, higher memory | Transformers, NLP |
| AdamW | Adam + weight decay fix | Slightly more memory | Modern default |
| RMSProp | Good for RNNs | Less used now | RNNs, RL |

## Tips

- Always use `AdamW` over `Adam` (correct weight decay implementation)
- LR warmup is **mandatory** for Transformers — skip it and training will diverge
- Mixed precision is free on any modern GPU (`torch.cuda.amp.autocast()`)
- Data loading is often the bottleneck: use `num_workers > 0` in `DataLoader`
- `torch.compile` (PyTorch 2.0+) is often the easiest speedup with zero code change
