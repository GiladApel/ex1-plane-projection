"""Generators of plane sequences, returned as (4, 3) arrays of points in the order a', b, a'', c."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

AXES = {"x": 0, "y": 1, "z": 2}


def axial_sweep(shape: Sequence[int], axis: int, frames: int | None = None) -> list[np.ndarray]:
    """Rectangles covering the whole slice perpendicular to `axis`, moving along that axis.

    Pseudo-code:
        1. r, t = the two other axes in ascending order
        2. positions = every slice index, or `frames` evenly spaced values over 0 .. shape[axis]-1
        3. for each position k: rectangle with rails along r (length shape[r]-1), top along t
           (length shape[t]-1), at coordinate k of `axis`; with integer k its image is exactly the slice
    """
    r, t = [a for a in range(3) if a != axis]
    positions = np.arange(shape[axis], dtype=np.float64) if frames is None else np.linspace(0, shape[axis] - 1, frames)
    sets = []
    for k in positions:
        corners = []
        for rv, tv in ((0, 0), (shape[r] - 1, 0), (0, shape[t] - 1), (shape[r] - 1, shape[t] - 1)):
            p = [0.0, 0.0, 0.0]
            p[r], p[t], p[axis] = rv, tv, k
            corners.append(p)
        sets.append(np.array(corners))
    return sets


def rotation(shape: Sequence[int], axis: int, frames: int, side: float | None = None) -> list[np.ndarray]:
    """A square plane through the volume center that contains `axis` and turns a full 360 degrees around it.

    Pseudo-code:
        1. c = volume center; u = unit vector of `axis`; p, q = the other two unit axes
        2. side defaults to the volume diagonal rounded up, plus 0.5 so floor(side) cannot flip between
           frames from floating-point noise (that would make the image size jitter by one pixel)
        3. for i in 0 .. frames-1: theta = 2*pi*i/frames; w = cos(theta)*p + sin(theta)*q
           m = c - side/2 * u (middle of the top edge); far = m + side*u
           points = (m - side/2*w, far - side/2*w, m + side/2*w, far + side/2*w)
    """
    center = (np.asarray(shape, dtype=np.float64) - 1) / 2
    side = math.ceil(float(np.linalg.norm(shape))) + 0.5 if side is None else side
    u = np.eye(3)[axis]
    p, q = [np.eye(3)[a] for a in range(3) if a != axis]
    sets = []
    for i in range(frames):
        theta = 2 * math.pi * i / frames
        w = math.cos(theta) * p + math.sin(theta) * q
        m = center - side / 2 * u
        far = m + side * u
        sets.append(np.array([m - side / 2 * w, far - side / 2 * w, m + side / 2 * w, far + side / 2 * w]))
    return sets


def write_planes_csv(path: str | Path, point_sets: Sequence[np.ndarray]) -> None:
    """One row per plane: the flattened points with 6 decimals."""
    lines = [",".join(f"{v:.6f}" for v in points.reshape(-1)) for points in point_sets]
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
