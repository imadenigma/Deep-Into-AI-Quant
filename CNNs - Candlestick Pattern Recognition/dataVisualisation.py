import numpy as np
import matplotlib.pyplot as plt

W = 20
X     = np.fromfile("X.bin",     dtype=np.float32).reshape(-1, 4, W)
X_gaf = np.fromfile("X_gaf.bin", dtype=np.float32).reshape(-1, 4, W, W)
y     = np.fromfile("y.bin",     dtype=np.int64)

names    = {0: "doji", 1: "hammer", 2: "bullish_engulfing"}
channels = ["Open", "High", "Low", "Close"]

def show_gaf(idx):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ch in range(4):
        im = axes[ch].imshow(X_gaf[idx, ch], cmap="rainbow", vmin=-1, vmax=1)
        axes[ch].set_title(channels[ch])
    fig.suptitle(f"#{idx} — {names[y[idx]]}")
    fig.colorbar(im, ax=axes, shrink=0.8)
    plt.show()

for lbl in (0, 1, 2):
    show_gaf(np.where(y == lbl)[0][0])