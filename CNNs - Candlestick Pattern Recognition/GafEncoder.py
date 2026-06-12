import numpy as np

W = 20
X = np.fromfile("X.bin", dtype=np.float32).reshape(-1, 4, W)
N = X.shape[0]

def gasf(series):
    s = np.sqrt(np.clip(1.0 -series ** 2, 0.0, 1.0))
    return np.outer(series, series) - np.outer(s, s)

X_gaf = np.empty((N, 4, W, W), dtype=np.float32)

for i in range(N):
    window = X[i]
    l, h = window.min(), window.max()
    rng = h - l if h > l else 1.0
    norm = 2.0 * (window - l) / rng - 1.0
    for ch in range(4):
        X_gaf[i, ch] = gasf(norm[ch])
X_gaf.tofile("X_gaf.bin")
print(X_gaf)