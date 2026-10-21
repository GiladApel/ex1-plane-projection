from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from .errors import InvalidInputError
from .geometry import Plane
from .sampler import Interpolation, sample_nearest, sample_trilinear
from .volume import Volume


class Projector(ABC):
    """Projects a Volume onto a Plane; returns an (H, W) image of the volume's dtype."""

    @abstractmethod
    def project(self, volume: Volume, plane: Plane, out: np.ndarray | None = None) -> np.ndarray:
        """Return the projection. If out is given it must have shape plane.shape and the volume dtype;
        it is cleared and filled in place and returned."""

    @staticmethod
    def _prepare_out(volume: Volume, plane: Plane, out: np.ndarray | None) -> np.ndarray:
        if out is None:
            return np.zeros(plane.shape, dtype=volume.dtype)
        if out.shape != plane.shape or out.dtype != volume.dtype:
            raise InvalidInputError(
                f"out must have shape {plane.shape} and dtype {volume.dtype}, got {out.shape} {out.dtype}"
            )
        out.fill(0)
        return out


class ReferenceProjector(Projector):
    """Literal, slow implementation of the algorithm; the ground truth for the fast backends."""

    def __init__(self, interpolation: Interpolation = Interpolation.NEAREST) -> None:
        self._sample = sample_nearest if interpolation is Interpolation.NEAREST else sample_trilinear

    def project(self, volume: Volume, plane: Plane, out: np.ndarray | None = None) -> np.ndarray:
        """Pseudo-code (see docs/ALGORITHM.md section 2):

        E = zeros(H, W)
        for y in 0 .. H-1:
            a1  = A1 + y * v1
            a2  = A2 + y * v2
            d12 = |a2 - a1|
            dp  = (a2 - a1) / d12 if d12 > 0 else 0
            off = (W - 1) / 2 - d12 / 2
            for x1 in 0 .. floor(d12):
                p = a1 + x1 * dp
                x = floor(off + x1 + 0.5)
                if 0 <= x < W: E[y][x] = sample(M, p)
        return E
        """
        rails = plane.rails()
        image = self._prepare_out(volume, plane, out)
        data = volume.data
        a1x0, a1y0, a1z0 = (float(c) for c in rails.A1)
        a2x0, a2y0, a2z0 = (float(c) for c in rails.A2)
        v1x, v1y, v1z = (float(c) for c in rails.v1)
        v2x, v2y, v2z = (float(c) for c in rails.v2)
        height, width = rails.H, rails.W

        for y in range(height):
            a1x, a1y, a1z = a1x0 + y * v1x, a1y0 + y * v1y, a1z0 + y * v1z
            dx, dy, dz = (a2x0 + y * v2x) - a1x, (a2y0 + y * v2y) - a1y, (a2z0 + y * v2z) - a1z
            d12 = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d12 > 0:
                dx, dy, dz = dx / d12, dy / d12, dz / d12
            off = (width - 1) / 2.0 - d12 / 2.0
            for x1 in range(math.floor(d12) + 1):
                x = math.floor(off + x1 + 0.5)
                if 0 <= x < width:
                    image[y, x] = self._sample(data, a1x + x1 * dx, a1y + x1 * dy, a1z + x1 * dz)
        return image
