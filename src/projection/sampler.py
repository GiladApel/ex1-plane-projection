from __future__ import annotations

import math
from enum import Enum

import numpy as np


class Interpolation(Enum):
    NEAREST = "nearest"
    TRILINEAR = "trilinear"


def sample_nearest(data: np.ndarray, px: float, py: float, pz: float):
    """Value of the voxel nearest to (px, py, pz), or 0 outside the volume.

    Pseudo-code:
        1. (i, j, k) = floor(p + 0.5) per axis
        2. return data[i][j][k] if all three are inside the volume, else 0
    """
    sx, sy, sz = data.shape
    i = math.floor(px + 0.5)
    j = math.floor(py + 0.5)
    k = math.floor(pz + 0.5)
    if 0 <= i < sx and 0 <= j < sy and 0 <= k < sz:
        return data[i, j, k]
    return 0


def sample_trilinear(data: np.ndarray, px: float, py: float, pz: float) -> int:
    """Trilinear interpolation at (px, py, pz); voxels outside the volume count as 0.

    Pseudo-code:
        1. (i0, j0, k0) = floor(p), (fx, fy, fz) = p - (i0, j0, k0)
        2. v = 0
           for di, dj, dk in {0, 1}:                       # x outermost, z innermost
               w = wx(di) * wy(dj) * wz(dk)                # w(0) = 1 - f, w(1) = f
               if the corner is inside the volume: v += w * data[i0+di][j0+dj][k0+dk]
        3. return floor(v + 0.5)
    """
    sx, sy, sz = data.shape
    i0, j0, k0 = math.floor(px), math.floor(py), math.floor(pz)
    fx, fy, fz = px - i0, py - j0, pz - k0
    acc = 0.0
    for di in (0, 1):
        i = i0 + di
        wx = fx if di else 1.0 - fx
        for dj in (0, 1):
            j = j0 + dj
            wy = fy if dj else 1.0 - fy
            for dk in (0, 1):
                k = k0 + dk
                wz = fz if dk else 1.0 - fz
                if 0 <= i < sx and 0 <= j < sy and 0 <= k < sz:
                    acc += wx * wy * wz * float(data[i, j, k])
    return math.floor(acc + 0.5)
