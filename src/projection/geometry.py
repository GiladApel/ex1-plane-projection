from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .errors import InvalidInputError

REL_TOL = 1e-3
COLLINEAR_TOL = 1e-6
TRAPEZOID_ORDER = "expected point order a', b, a'', c, where a'b and a''c are the equal legs"


@dataclass(frozen=True, eq=False)
class Rails:
    """Two equal-step rails over a plane: row y spans a1 = A1 + y*v1 to a2 = A2 + y*v2."""

    A1: np.ndarray
    v1: np.ndarray
    A2: np.ndarray
    v2: np.ndarray
    H: int
    W: int


class Plane:
    """A planar region described by two rails; Triangle and Trapezoid are the concrete shapes."""

    def __init__(self, a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray) -> None:
        self._rails = _build_rails(a1, b1, a2, b2)

    def rails(self) -> Rails:
        return self._rails

    @property
    def shape(self) -> tuple[int, int]:
        """Output image size (H, W)."""
        return self._rails.H, self._rails.W


class Triangle(Plane):
    """Isosceles triangle (a, b, c) with |ab| = |ac|; the apex is a."""

    def __init__(self, a, b, c, force: bool = False) -> None:
        a, b, c = (_as_point(p, n) for p, n in zip((a, b, c), "abc", strict=True))
        ab, ac = b - a, c - a
        dab, dac = np.linalg.norm(ab), np.linalg.norm(ac)
        _check_edge(dab, "ab")
        _check_edge(dac, "ac")
        if np.linalg.norm(np.cross(ab, ac)) / (dab * dac) < COLLINEAR_TOL:
            raise InvalidInputError("triangle points are collinear")
        _check_isosceles(dab, dac, force)
        super().__init__(a, b, a, c)


class Trapezoid(Plane):
    """Isosceles trapezoid (a1, b, a2, c): rail 1 is a1 -> b, rail 2 is a2 -> c, |a1 b| = |a2 c|."""

    def __init__(self, a1, b, a2, c, force: bool = False) -> None:
        a1, b, a2, c = (_as_point(p, n) for p, n in zip((a1, b, a2, c), ("a'", "b", "a''", "c"), strict=True))
        rail1, rail2 = b - a1, c - a2
        d1, d2 = np.linalg.norm(rail1), np.linalg.norm(rail2)
        _check_edge(d1, "a'b")
        _check_edge(d2, "a''c")
        _check_isosceles(d1, d2, force, hint=TRAPEZOID_ORDER)
        top, bottom = a2 - a1, c - b
        top_len, bottom_len = np.linalg.norm(top), np.linalg.norm(bottom)
        if top_len == 0 and bottom_len == 0:
            raise InvalidInputError("trapezoid is degenerate: both parallel sides have zero length")
        if np.linalg.norm(np.cross(rail1, rail2)) / (d1 * d2) < COLLINEAR_TOL and top_len == 0:
            raise InvalidInputError("trapezoid points are collinear")
        triple = abs(np.dot(top, np.cross(rail1, c - a1)))
        if triple > REL_TOL * top_len * d1 * np.linalg.norm(c - a1):
            raise InvalidInputError("trapezoid points are not coplanar")
        if top_len > 0 and bottom_len > 0 and np.linalg.norm(np.cross(top, bottom)) > REL_TOL * top_len * bottom_len:
            raise InvalidInputError(f"trapezoid sides a'a'' and bc are not parallel ({TRAPEZOID_ORDER})")
        super().__init__(a1, b, a2, c)


def plane_from_points(points: Sequence[Sequence[float]] | np.ndarray, force: bool = False) -> Plane:
    """Build a Triangle from 3 points or a Trapezoid from 4 points (order a', b, a'', c)."""
    if len(points) == 3:
        a, b, c = points
        return Triangle(a, b, c, force=force)
    if len(points) == 4:
        a1, b, a2, c = points
        return Trapezoid(a1, b, a2, c, force=force)
    raise InvalidInputError(f"a plane needs 3 or 4 points, got {len(points)}")


def _build_rails(a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray) -> Rails:
    """Turn rail endpoints into stepping data.

    Pseudo-code:
        1. L  = |b1 - a1|
        2. v1 = (b1 - a1) / L, v2 = (b2 - a2) / L     (both rails advance in the same H steps)
        3. H  = floor(L) + 1
        4. W  = floor(max(|a2 - a1|, |b2 - b1|)) + 1  (widest row)
    """
    length = float(np.linalg.norm(b1 - a1))
    v1 = (b1 - a1) / length
    v2 = (b2 - a2) / length
    height = math.floor(length) + 1
    width = math.floor(max(np.linalg.norm(a2 - a1), np.linalg.norm(b2 - b1))) + 1
    return Rails(a1, v1, a2, v2, height, width)


def _as_point(value, name: str) -> np.ndarray:
    try:
        p = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"point {name} is not numeric") from exc
    if p.shape != (3,):
        raise InvalidInputError(f"point {name} must have 3 coordinates, got shape {p.shape}")
    if not np.all(np.isfinite(p)):
        raise InvalidInputError(f"point {name} has a non-finite coordinate")
    return p


def _check_edge(length: float, name: str) -> None:
    if length < 1.0:
        raise InvalidInputError(f"edge {name} is shorter than one voxel ({length:.3g})")


def _check_isosceles(l1: float, l2: float, force: bool, hint: str = "") -> None:
    if not force and abs(l1 - l2) > REL_TOL * max(l1, l2):
        suffix = f" ({hint})" if hint else ""
        raise InvalidInputError(f"plane is not isosceles: equal edges are {l1:.6g} and {l2:.6g}{suffix}")
