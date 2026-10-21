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
        """Return the projection. If out is given it must be C-contiguous with shape plane.shape and the volume dtype;
        it is cleared and filled in place and returned."""

    @staticmethod
    def _prepare_out(volume: Volume, plane: Plane, out: np.ndarray | None) -> np.ndarray:
        if out is None:
            return np.zeros(plane.shape, dtype=volume.dtype)
        if out.shape != plane.shape or out.dtype != volume.dtype or not out.flags.c_contiguous:
            raise InvalidInputError(
                f"out must be C-contiguous with shape {plane.shape} and dtype {volume.dtype}, "
                f"got {out.shape} {out.dtype}"
            )
        out.fill(0)
        return out


class NumpyProjector(Projector):
    """Vectorized implementation: the whole (H, N) sample grid is computed with broadcasting.

    Large temporaries are cached between calls (one slot per name), so repeated calls on same-sized
    planes allocate almost nothing. An instance must not be used from several threads at once.
    """

    def __init__(self, interpolation: Interpolation = Interpolation.NEAREST) -> None:
        self._interpolation = interpolation
        self._buffers: dict[str, np.ndarray] = {}

    def project(self, volume: Volume, plane: Plane, out: np.ndarray | None = None) -> np.ndarray:
        """Pseudo-code (same arithmetic, in the same order, as ReferenceProjector):

        1. y = 0 .. H-1 as a column; a1 = A1 + y*v1; a2 = A2 + y*v2; d12 = |a2 - a1|; dp = (a2 - a1)/d12
        2. x1 = 0 .. N-1 as a row, N = floor(max(d12)) + 1
        3. xf = floor((W-1)/2 - d12/2 + x1 + 0.5); keep[y, x1] = x1 <= floor(d12) and 0 <= xf < W
        4. for the whole (H, N) grid: p = a1[y] + x1*dp[y]; value = sample(M, p)
           (nearest sampling also clears keep where p is outside the volume)
        5. image[y, xf] = value where keep
        """
        rails = plane.rails()
        image = self._prepare_out(volume, plane, out)
        height, width = rails.H, rails.W

        y = np.arange(height, dtype=np.float64)[:, None]
        a1 = rails.A1 + y * rails.v1
        diff = (rails.A2 + y * rails.v2) - a1
        d12 = np.sqrt(diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2])
        dp = diff / np.where(d12 > 0, d12, 1.0)[:, None]
        count = np.floor(d12) + 1
        x1 = np.arange(int(count.max()), dtype=np.float64)[None, :]
        grid = (height, x1.shape[1])

        xf = self._buffer("xf", grid, np.float64)
        np.add(((width - 1) / 2.0 - d12 / 2.0)[:, None], x1, out=xf)
        xf += 0.5
        np.floor(xf, out=xf)
        keep = (x1 < count[:, None]) & (xf >= 0) & (xf < width)

        if self._interpolation is Interpolation.NEAREST:
            values = self._nearest(volume, a1, dp, x1, keep)
        else:
            positions = [a1[:, axis, None] + x1 * dp[:, axis, None] for axis in range(3)]
            values = self._trilinear(volume, positions)

        xf += y * width
        image.reshape(-1)[xf[keep].astype(np.intp)] = values[keep]
        return image

    def _buffer(self, name: str, shape: tuple[int, ...], dtype) -> np.ndarray:
        buf = self._buffers.get(name)
        if buf is None or buf.shape != shape or buf.dtype != dtype:
            buf = self._buffers[name] = np.empty(shape, dtype=dtype)
        return buf

    def _nearest(self, volume: Volume, a1, dp, x1, keep: np.ndarray) -> np.ndarray:
        """Gather the nearest voxel for the whole grid; samples outside the volume are cleared in keep."""
        grid = keep.shape
        coord = self._buffer("coord", grid, np.float64)
        flat = self._buffer("flat", grid, np.float64)
        index = self._buffer("index", grid, np.intp)
        for axis, size in enumerate(volume.shape):
            np.multiply(x1, dp[:, axis, None], out=coord)
            coord += a1[:, axis, None]
            coord += 0.5
            np.floor(coord, out=coord)
            keep &= coord >= 0
            keep &= coord < size
            if axis == 0:
                np.copyto(flat, coord)
            else:
                flat *= size
                flat += coord
        flat *= keep
        np.copyto(index, flat, casting="unsafe")
        return volume.flat.take(index)

    @staticmethod
    def _trilinear(volume: Volume, positions: list[np.ndarray]) -> np.ndarray:
        sx, sy, sz = volume.shape
        base = [np.floor(p) for p in positions]
        frac = [p - b for p, b in zip(positions, base)]
        acc = np.zeros(positions[0].shape, dtype=np.float64)
        for di in (0, 1):
            i = base[0] + di
            wx = frac[0] if di else 1.0 - frac[0]
            for dj in (0, 1):
                j = base[1] + dj
                wy = frac[1] if dj else 1.0 - frac[1]
                for dk in (0, 1):
                    k = base[2] + dk
                    wz = frac[2] if dk else 1.0 - frac[2]
                    inside = (i >= 0) & (i < sx) & (j >= 0) & (j < sy) & (k >= 0) & (k < sz)
                    flat = np.where(inside, (i * sy + j) * sz + k, 0).astype(np.intp)
                    acc += wx * wy * wz * np.where(inside, volume.flat[flat], 0).astype(np.float64)
        return np.floor(acc + 0.5).astype(volume.dtype)


class NumbaProjector(Projector):
    """JIT-compiled, multi-threaded implementation. The first call compiles (about a second)."""

    def __init__(self, interpolation: Interpolation = Interpolation.NEAREST) -> None:
        from . import kernels_numba

        self._kernel = (
            kernels_numba.project_nearest
            if interpolation is Interpolation.NEAREST
            else kernels_numba.project_trilinear
        )

    def project(self, volume: Volume, plane: Plane, out: np.ndarray | None = None) -> np.ndarray:
        """Pseudo-code: validate/clear out, then run the compiled kernel (see kernels_numba.py)."""
        rails = plane.rails()
        image = self._prepare_out(volume, plane, out)
        self._kernel(volume.data, rails.A1, rails.v1, rails.A2, rails.v2, rails.H, rails.W, image)
        return image


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


BACKENDS = ("auto", "numba", "numpy", "reference")


def make_projector(backend: str = "auto", interpolation: Interpolation = Interpolation.NEAREST) -> Projector:
    """Create a projector. "auto" picks numba when it can be imported and numpy otherwise."""
    if backend not in BACKENDS:
        raise InvalidInputError(f"unknown backend {backend!r}; choose from {', '.join(BACKENDS)}")
    if backend in ("auto", "numba"):
        try:
            return NumbaProjector(interpolation)
        except ImportError as exc:
            if backend == "numba":
                raise InvalidInputError("numba backend requested but numba is not installed") from exc
            backend = "numpy"
    if backend == "numpy":
        return NumpyProjector(interpolation)
    return ReferenceProjector(interpolation)
