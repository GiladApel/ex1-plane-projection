from __future__ import annotations

from pathlib import Path

import nrrd
import numpy as np

from .geometry import Trapezoid

DTYPES = (np.uint8, np.uint16, np.uint32)


def gradient(shape: tuple[int, int, int], dtype=np.uint8) -> np.ndarray:
    """M[x][y][z] = x. Known answer for axis-aligned planes."""
    _require_fits(shape[0] - 1, dtype)
    return np.broadcast_to(np.arange(shape[0], dtype=dtype)[:, None, None], shape).copy()


def sphere(shape: tuple[int, int, int], dtype=np.uint8) -> np.ndarray:
    """Filled sphere (value = max of dtype) centered in the volume, radius = a third of the smallest side."""
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    center = [(s - 1) / 2 for s in shape]
    radius = min(shape) / 3
    dist2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    return np.where(dist2 <= radius**2, np.iinfo(dtype).max, 0).astype(dtype)


def checkerboard(shape: tuple[int, int, int], dtype=np.uint8, cell: int = 8) -> np.ndarray:
    """Alternating 0 / max cells of the given edge length."""
    x, y, z = np.ogrid[tuple(slice(0, s) for s in shape)]
    parity = ((x // cell) + (y // cell) + (z // cell)) % 2
    return (parity * np.iinfo(dtype).max).astype(dtype)


def random_volume(shape: tuple[int, int, int], dtype=np.uint8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype, endpoint=True)


def random_plane(
    rng: np.random.Generator, center, length: float, top: float, bottom: float
) -> Trapezoid:
    """Isosceles trapezoid with a random orientation, centered on `center`.

    Pseudo-code:
        1. u, w = two random orthonormal vectors (u = rail direction, w = width direction)
        2. m = center - (length / 2) * u                       (middle of the top edge)
        3. a' = m - top/2 * w,  a'' = m + top/2 * w
           b  = m + length*u - bottom/2 * w,  c = m + length*u + bottom/2 * w
    """
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    u, w = q[:, 0], q[:, 1]
    m = np.asarray(center, dtype=np.float64) - (length / 2.0) * u
    far = m + length * u
    return Trapezoid(m - top / 2 * w, far - bottom / 2 * w, m + top / 2 * w, far + bottom / 2 * w)


def oblique_rectangles(size: int, count: int, seed: int = 0) -> list[Trapezoid]:
    """Random oblique rectangles through the center of a size^3 volume, all with the same output shape."""
    rng = np.random.default_rng(seed)
    side = size - 0.5
    center = [(size - 1) / 2] * 3
    return [random_plane(rng, center, side, side, side) for _ in range(count)]


def save_nrrd(path: str | Path, data: np.ndarray) -> None:
    nrrd.write(str(path), data, index_order="F")


def _require_fits(value: int, dtype) -> None:
    if value > np.iinfo(dtype).max:
        raise ValueError(f"value {value} does not fit in {np.dtype(dtype)}")
