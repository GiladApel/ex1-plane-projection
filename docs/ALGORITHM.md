# Algorithm

Decision numbers (D1..D16) refer to [SRS.md](SRS.md).

## 1. Plane to rails

A plane becomes two rails of equal length `L`, each a start point and a unit direction:

```
Triangle (a, b, c):            A1 = a,   B1 = b,   A2 = a,   B2 = c
Trapezoid (a', b, a'', c):     A1 = a',  B1 = b,   A2 = a'', B2 = c

L  = |B1 - A1|                 (equals |B2 - A2| within tolerance, D9)
v1 = (B1 - A1) / L
v2 = (B2 - A2) / L
H  = floor(L) + 1                                   (D3, D4)
Wtop    = |A2 - A1|            (0 for a triangle)
Wbottom = |B2 - B1|
W  = floor(max(Wtop, Wbottom)) + 1                  (D4, D10)
```

Because of this, the projector never distinguishes triangle from trapezoid.

## 2. Projection (nearest neighbor)

```
Input:  volume M (sx, sy, sz); rails A1, v1, A2, v2; sizes H, W
Output: image E of shape (H, W), dtype of M, initialised to 0

for each row y in 0 .. H-1:                          # independent rows, parallel
    a1  = A1 + y * v1                                # D3, D7
    a2  = A2 + y * v2
    d12 = |a2 - a1|
    dp  = (a2 - a1) / d12   if d12 > 0 else (0,0,0)
    off = W/2 - d12/2                                # centers the row
    for x1 in 0 .. floor(d12):
        p = a1 + x1 * dp                             # closed form, D7
        x = floor(off + x1 + 0.5)                    # D5, D8
        if x < 0 or x >= W: continue
        (i, j, k) = floor(p + 0.5)                   # D8
        if 0 <= i < sx and 0 <= j < sy and 0 <= k < sz:
            E[y, x] = M[i, j, k]                     # D6
        # else E[y, x] stays 0                       # FR5
return E
```

Differences from the assignment's pseudo-code: D3 (row indexing), D4 (sizes), D5 (rounding of `x`), D6 (index order), D7 (closed form instead of accumulation).

## 3. Trilinear variant

Steps `(i, j, k) = floor(p + 0.5)` and the lookup are replaced by:

```
(i0, j0, k0) = floor(p)
(fx, fy, fz) = p - (i0, j0, k0)
v = sum over the 8 corners (di, dj, dk in {0,1}) of
        w(di,fx) * w(dj,fy) * w(dk,fz) * M_or_0[i0+di, j0+dj, k0+dk]
    where w(0,f) = 1 - f, w(1,f) = f, and M_or_0 is 0 outside the volume
E[y, x] = floor(v + 0.5)
```

## 4. Backends

| Backend | Strategy |
|---|---|
| `reference` | Literal nested loops from section 2. Ground truth for tests. |
| `numpy` | For all `(y, x1)` at once, broadcast `a1[y] + x1*dp[y]`, mask, round, gather from the flattened volume. Preallocated buffers. |
| `numba` | Section 2 compiled with `@njit(parallel=True)`, `prange` over `y`, no intermediate arrays, writes into a caller-owned `out`. One kernel per sampler. |

All three compute positions with the same float64 expressions and the same rounding rule, so nearest-sampling output is identical.

## 5. Frame to video

```
window = (p1, p99) percentiles of the volume, computed once          (D12)
Hc, Wc = max H and W over all planes, rounded up to even             (D13)
for each plane (CSV order):
    E     = project(M, plane)
    img   = clip((E - lo) / (hi - lo), 0, 1) * 255  as uint8
    frame = zeros(Hc, Wc); paste img centered
    write frame
```

## 6. Complexity
Per frame: `O(H * W)` samples, one memory read each. Rows are independent, so the work parallelizes over `y`. Memory: the output frame only; the Numba backend allocates nothing else per frame.
