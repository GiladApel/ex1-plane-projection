"""Numba kernels. Every floating-point expression mirrors ReferenceProjector in the same order,
so results are bit-identical to the reference."""
from __future__ import annotations

import math

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def project_nearest(data, a1, v1, a2, v2, height, width, out):
    """Pseudo-code (docs/ALGORITHM.md section 2), rows run in parallel:

    for y in 0 .. H-1:
        a1 = A1 + y*v1;  a2 = A2 + y*v2;  d12 = |a2 - a1|;  dp = (a2 - a1)/d12
        off = (W-1)/2 - d12/2
        for x1 in 0 .. floor(d12):
            x = floor(off + x1 + 0.5);  skip if x outside [0, W)
            (i, j, k) = floor(a1 + x1*dp + 0.5);  out[y][x] = M[i][j][k] if inside M
    """
    sx, sy, sz = data.shape
    for y in nb.prange(height):
        a1x = a1[0] + y * v1[0]
        a1y = a1[1] + y * v1[1]
        a1z = a1[2] + y * v1[2]
        dx = (a2[0] + y * v2[0]) - a1x
        dy = (a2[1] + y * v2[1]) - a1y
        dz = (a2[2] + y * v2[2]) - a1z
        d12 = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d12 > 0:
            dx = dx / d12
            dy = dy / d12
            dz = dz / d12
        off = (width - 1) / 2.0 - d12 / 2.0
        for x1 in range(int(np.floor(d12)) + 1):
            xf = np.floor(off + x1 + 0.5)
            if xf < 0 or xf >= width:
                continue
            i = np.floor(a1x + x1 * dx + 0.5)
            j = np.floor(a1y + x1 * dy + 0.5)
            k = np.floor(a1z + x1 * dz + 0.5)
            if 0 <= i < sx and 0 <= j < sy and 0 <= k < sz:
                out[y, int(xf)] = data[int(i), int(j), int(k)]


@nb.njit(parallel=True, cache=True)
def project_trilinear(data, a1, v1, a2, v2, height, width, out):
    """Same row/column walk as project_nearest; each sample is the trilinear blend of the 8 corners
    around p (corners outside the volume count as 0), summed with di outermost and dk innermost."""
    sx, sy, sz = data.shape
    for y in nb.prange(height):
        a1x = a1[0] + y * v1[0]
        a1y = a1[1] + y * v1[1]
        a1z = a1[2] + y * v1[2]
        dx = (a2[0] + y * v2[0]) - a1x
        dy = (a2[1] + y * v2[1]) - a1y
        dz = (a2[2] + y * v2[2]) - a1z
        d12 = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d12 > 0:
            dx = dx / d12
            dy = dy / d12
            dz = dz / d12
        off = (width - 1) / 2.0 - d12 / 2.0
        for x1 in range(int(np.floor(d12)) + 1):
            xf = np.floor(off + x1 + 0.5)
            if xf < 0 or xf >= width:
                continue
            px = a1x + x1 * dx
            py = a1y + x1 * dy
            pz = a1z + x1 * dz
            i0 = np.floor(px)
            j0 = np.floor(py)
            k0 = np.floor(pz)
            fx = px - i0
            fy = py - j0
            fz = pz - k0
            acc = 0.0
            for di in range(2):
                i = i0 + di
                wx = fx if di == 1 else 1.0 - fx
                for dj in range(2):
                    j = j0 + dj
                    wy = fy if dj == 1 else 1.0 - fy
                    for dk in range(2):
                        k = k0 + dk
                        wz = fz if dk == 1 else 1.0 - fz
                        if 0 <= i < sx and 0 <= j < sy and 0 <= k < sz:
                            acc += wx * wy * wz * np.float64(data[int(i), int(j), int(k)])
            out[y, int(xf)] = int(np.floor(acc + 0.5))
