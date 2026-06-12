# Each module should:
1. Ingest input data as arguments to its forward propagation method.

2. Generate an output by having the forward propagation method return a value. Note that the output may have a different shape from the input. For example, the first fully connected layer in our model above ingests an input of arbitrary dimension but returns an output of dimension 256.

3. Calculate the gradient of its output with respect to its input, which can be accessed via its backpropagation method. Typically this happens automatically.

4. Store and provide access to those parameters necessary for executing the forward propagation computation.

5. Initialize model parameters as needed.
# In C++:
### torch::NoGradGuard ng; (this optimise the forward when we don't need to calculate the backward as it turns Grad Tracking Off and re On on destruction)

# CNN:
### Translation Invariance
$$[\mathbf{H}]_{i, j} = u + \sum_a \sum_b [\mathbf{V}]_{a, b} [\mathbf{X}]_{i+a, j+b}.$$
### Locality
$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b} [\mathbf{X}]_{i+a, j+b}.$$
the output size is given by the input size  minus the size of the convolution kernel  via
 <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
<mo stretchy="false">(</mo>
<msub>
<mi>n</mi>
<mtext>h</mtext>
</msub>
<mo>&#x2212;</mo>
<msub>
<mi>k</mi>
<mtext>h</mtext>
</msub>
<mo>+</mo>
<mn>1</mn>
<mo stretchy="false">)</mo>
<mo>&#xD7;</mo>
<mo stretchy="false">(</mo>
<msub>
<mi>n</mi>
<mtext>w</mtext>
</msub>
<mo>&#x2212;</mo>
<msub>
<mi>k</mi>
<mtext>w</mtext>
</msub>
<mo>+</mo>
<mn>1</mn>
<mo stretchy="false">)</mo>
<mo>.</mo>
</math>
The two parameters of a convolutional layer are the kernel and the scalar bias.

# What is Gramian Angular Field
The goal is to turn a 1D time series into a 2D image _without_ just plotting it — an image whose pixels encode the **relationship between every pair of timesteps**. Three sub-steps.

**1. Normalize to [−1, 1].** GAF requires every value to sit in [−1, 1], because the next step feeds them into `arccos`, whose domain is exactly [−1, 1]. So we rescale.

**2. Reinterpret each value as an angle.** Take `φᵢ = arccos(xᵢ)`. Since each `xᵢ ∈ [−1, 1]`, each angle `φᵢ ∈ [0, π]`. We've moved the series from "values on a line" to "angles on a circle" — the _polar_ view. This is the "Angular" in GAF.

**3. Build the Gramian field.** Form a W×W matrix where each cell compares two timesteps by **summing their angles** (the "Gramian Angular Summation Field", GASF): GASF[i, j] = xᵢ·xⱼ − √(1 − xᵢ²)·√(1 − xⱼ²)
#### The one real design decision: shared vs. per-channel normalization

This matters for _candlesticks specifically_, so it's worth a deliberate choice rather than a default.

The textbook GAF normalizes each series independently. But our four channels aren't independent series — they're O/H/L/C of the _same_ candles, and a candlestick pattern is **defined by the relationships between them** (a hammer = close sits high in the range, long lower wick). If I normalize each channel to fill [−1, 1] on its own, I stretch each one separately and _destroy_ the fact that High sits above Close. That would erase the very signal we want.

So I normalize all four channels together using a **shared** min/max across the whole window. High stays above Close after scaling; the candle geometry survives. This is the faithful choice for OHLC.

# Padding and Stride

## Padding

**What:** Adding extra pixels (usually zeros) around the boundary of the input image.

**Why:**
- Convolutions shrink output — kernels >1 lose pixels each layer, adding up across many layers (e.g. ten 5×5 convs turn 240×240 into 200×200, losing ~30%)
- Corner/edge pixels are barely used — padding makes all pixels used more equally
- Lets you control output size and keep input/output dimensions the same

**How:**
- Add p_h rows + p_w columns → output = (n_h − k_h + p_h + 1) × (n_w − k_w + p_w + 1)
- Set p_h = k_h − 1 to preserve input size
- Use **odd kernels** (1, 3, 5, 7) so padding is symmetric on both sides
- In code: `padding=1` on a 3×3 kernel keeps an 8×8 input at 8×8

## Stride

**What:** The number of rows/columns the kernel moves per slide (default = 1).

**Why:**
- Computational efficiency
- Downsampling — reduce resolution drastically when input is unwieldy (e.g. stride 2 halves height & width)

**How:**
- With strides s_h, s_w → output = ⌊(n_h − k_h + p_h + s_h)/s_h⌋ × ⌊(n_w − k_w + p_w + s_w)/s_w⌋
- If input is divisible by stride → output = (n_h/s_h) × (n_w/s_w)
- In code: `stride=2` on an 8×8 input → 4×4 output

## Key takeaways
- Defaults: padding = 0, stride = 1
- Padding ↑ output size (avoids shrinkage, equalizes pixel use); stride ↓ output size (downsamples)
- Zero-padding is cheap and lets CNNs encode implicit position info

