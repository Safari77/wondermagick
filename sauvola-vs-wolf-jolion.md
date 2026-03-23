# Sauvola vs Wolf-Jolion: Algorithm Comparison

## Threshold formulas

### Sauvola

```
T(x,y) = mean * (1 + k * (std_dev / R - 1))
```

Three parameters: `window_size`, `k` (default 0.2), `R` (default 128).

`R` is a **hardcoded constant** representing half the dynamic range of an 8-bit image.
The formula is purely local — only the pixels inside the window determine the threshold.

### Wolf-Jolion

```
T(x,y) = (1 - k) * mean + k * M + k * (std_dev / Rmax) * (mean - M)

       = mean - k * (mean - M) * (1 - std_dev / Rmax)
```

Two parameters: `window_size`, `k` (default 0.5).

`M` and `Rmax` are **derived from the image** before thresholding begins:
- `M` = global minimum gray value (the darkest pixel in the entire image).
- `Rmax` = maximum local standard deviation across all windows.

## What the differences mean in practice

### Dynamic range normalization

Sauvola divides the local standard deviation by a fixed `R=128`, which assumes the
document uses the full 8-bit range. Wolf-Jolion divides by `Rmax`, the actual peak
contrast present in the document. On a low-contrast scan (e.g. faded ink on yellowed
paper) the local std_dev is small, so `std_dev / 128` in Sauvola is tiny and the
`k*(…-1)` term aggressively lowers the threshold. Wolf-Jolion's `std_dev / Rmax`
stays proportional to the document's own contrast, keeping the threshold stable.

### Stain and shadow handling

Sauvola's threshold is `mean * (1 + …)`, which is anchored entirely to the local mean.
A dark stain or shadow pulls the local mean down, which drags the threshold down with
it — the stain gets binarized as foreground (black), producing noise.

Wolf-Jolion's `(mean - M)` term compares the local mean against the darkest text in
the whole document. In a stained region the local mean drops, but since M (the true
ink darkness) is even lower, the gap `(mean - M)` shrinks and the threshold stays
close to `mean`. The stain is correctly classified as background.

### Uniform background regions

In areas with no text, `std_dev ≈ 0`.

- **Sauvola:** `T = mean * (1 + k*(0/R - 1)) = mean * (1 - k)`. The threshold sits
  at `(1-k)*mean`, a fixed fraction below the local mean. With the default k=0.2 this
  is 80% of the local mean, which usually works but can misfire on low-contrast areas.

- **Wolf-Jolion:** `T = mean - k*(mean - M)*(1 - 0) = mean - k*(mean - M)`.
  The threshold drops by `k` times the full gap between the local brightness and the
  global ink darkness. On a bright background where `mean ≈ 230` and `M ≈ 10`, this
  gives `T ≈ 230 - 0.5*220 = 120`, well below any background pixel, so the entire
  region is correctly white.

### Number of parameters

Sauvola takes three parameters (`window_size, k, R`), and `R` often requires manual
tuning per-document. Wolf-Jolion takes two (`window_size, k`) and computes its scaling
factor from the image. This makes Wolf-Jolion easier to use as a fire-and-forget
operation across varied scans.

## Computational cost

Both use the same integral-image approach for fast local statistics in O(1) per pixel.
Wolf-Jolion adds one extra pass over all windows to find `Rmax` (the global minimum `M`
is found for free during integral image construction). In practice the overhead is small
because the Rmax pass reuses the same integral images and involves no pixel reads — only
table lookups and arithmetic.

| Step                     | Sauvola | Wolf-Jolion |
|--------------------------|---------|-------------|
| Build integral images    | 1 pass  | 1 pass (also finds M) |
| Find Rmax                | —       | 1 pass over integral images |
| Threshold per pixel      | 1 pass  | 1 pass |
| **Total pixel-level passes** | **2** | **3** |

## When to use which

- **Sauvola** is a good starting point and the more widely documented algorithm.
  It works well on clean, high-contrast documents. If results are off, tune `R` to
  match the document's contrast range.

- **Wolf-Jolion** is better suited for degraded documents: uneven lighting, shadows,
  stains, faded ink, low-contrast scans. It adapts to the document automatically and
  requires less parameter tuning.
