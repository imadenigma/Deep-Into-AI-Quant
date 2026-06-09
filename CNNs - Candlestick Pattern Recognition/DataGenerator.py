import numpy as np

#Doji (1 candle) — open ≈ close (a tiny body), with wicks above and below. Signals indecision. We make open and close nearly equal and add symmetric shadows.
#Hammer (1 candle) — small body sitting near the top of the range, a long lower wick (≥ 2× the body), almost no upper wick. We give it a small body and a long lower shadow.
#Bullish engulfing (2 candles) — a small bearish candle (close < open), then a larger bullish candle (close > open) whose body fully engulfs the first: it opens below the first's close and closes above the first's open.

W = 20 # Candles per Window
N_PER_CLASS = 2000
rng = np.random.default_rng(0)

def background(n, start = 100.0):
    o,h,l,c =  (np.empty(n) for _ in range(4))
    prev = start
    for t in range(n):
        op = prev + rng.normal(0,0.3)
        cl = op + rng.normal(0,1)
        hi = max(op,cl) + abs(rng.normal(0,0.4))
        lo = min(op,cl) - abs(rng.normal(0,0.4))
        o[t], h[t], l[t], c[t] = op, hi, lo, cl
        prev = cl
    return o,h,l,c

def inject_doji(o,h,l,c):
    p = c[-2]
    o[-1] = p
    c[-1] = p + rng.normal(0,0.05)
    wick = abs(rng.normal(1.0,0.3))
    h[-1] = max(o[-1],c[-1]) + wick
    l[-1] = min(o[-1], c[-1]) - wick

def inject_hammer(o,h,l,c):
    p = c[-2]
    body = abs(rng.normal(0.4,0.1))
    o[-1] = p
    c[-1] = p + body
    h[-1] = max(o[-1], c[-1]) + abs(rng.normal(0.05,0.02))
    l[-1] = min(o[-1], c[-1]) - body * rng.uniform(2.0,3.0)


def inject_bullish_engulfing(o, h, l, c):
    p = c[-3]
    o[-2] = p                                 # candle 1: small bearish
    c[-2] = p - abs(rng.normal(0.5, 0.1))
    h[-2] = o[-2] + abs(rng.normal(0.1, 0.05))
    l[-2] = c[-2] - abs(rng.normal(0.1, 0.05))
    o[-1] = c[-2] - abs(rng.normal(0.3, 0.1)) # candle 2: opens below c1's close
    c[-1] = o[-2] + abs(rng.normal(0.6, 0.2)) # ...closes above c1's open -> engulfs
    h[-1] = c[-1] + abs(rng.normal(0.1, 0.05))
    l[-1] = o[-1] - abs(rng.normal(0.1, 0.05))

makers = {0: inject_doji, 1: inject_hammer, 2: inject_bullish_engulfing}

X, y = [], []
for label, fn in makers.items():
    for _ in range(N_PER_CLASS):
        o,h,l,c = background(W)
        fn(o,h,l,c)
        X.append(np.stack([o,h,l,c], axis=0))
        y.append(label)

X = np.asarray(X)
y = np.asarray(y)
perm = rng.permutation(len(y))  # shuffle so classes aren't grouped
X, y = X[perm], y[perm]

X.tofile("X.bin")
y.tofile("y.bin")
print("X:", X.shape, X.dtype, " y:", y.shape, y.dtype)

