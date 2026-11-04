from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .errors import InvalidInputError, NotIsoscelesError
from .sampler import SNAP

REL_TOL = 1e-3
COLLINEAR_TOL = 1e-6
SAME_POINT_TOL = 1e-9
TRAPEZOID_ORDER = "expected point order a', b, a'', c, where a'b and a''c are the equal legs"
# H.264 encoders and players reject larger frames, and one oversized plane would enlarge every frame of the video.
MAX_IMAGE_SIDE = 4096


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
        self._corners = (a1, b1, a2, b2)
        self._rails = _build_rails(a1, b1, a2, b2)
        if max(self._rails.H, self._rails.W) > MAX_IMAGE_SIDE:
            raise InvalidInputError(
                f"the image would be {self._rails.H} x {self._rails.W} pixels, more than {MAX_IMAGE_SIDE} on a side"
            )

    def rails(self) -> Rails:
        return self._rails

    @property
    def corners(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The rail end points (A1, B1, A2, B2) as given: (a', b, a'', c), or (a, b, a, c) for a triangle."""
        return self._corners

    @property
    def shape(self) -> tuple[int, int]:
        """Output image size (H, W)."""
        return self._rails.H, self._rails.W


class Triangle(Plane):
    """Isosceles triangle (a, b, c) with |ab| = |ac|; the apex is a."""

    def __init__(self, a, b, c, force: bool = False) -> None:
        a, b, c = (_as_point(p, n) for p, n in zip((a, b, c), "abc", strict=True))
        _check_distinct({"a": a, "b": b, "c": c})
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
        # a' = a'' or b = c alone is a parallel side of length zero, which is allowed; both at once is not.
        _check_distinct({"a'": a1, "b": b, "a''": a2, "c": c}, allowed=(("a'", "a''"), ("b", "c")))
        rail1, rail2 = b - a1, c - a2
        d1, d2 = np.linalg.norm(rail1), np.linalg.norm(rail2)
        _check_edge(d1, "a'b")
        _check_edge(d2, "a''c")
        top, bottom = a2 - a1, c - b
        top_len, bottom_len = np.linalg.norm(top), np.linalg.norm(bottom)
        if top_len == 0 and bottom_len == 0:
            raise InvalidInputError("trapezoid is degenerate: a' and a'' are the same point, and so are b and c")
        if np.linalg.norm(np.cross(rail1, rail2)) / (d1 * d2) < COLLINEAR_TOL and top_len == 0:
            raise InvalidInputError("trapezoid points are collinear")
        triple = abs(np.dot(top, np.cross(rail1, c - a1)))
        if triple > REL_TOL * top_len * d1 * np.linalg.norm(c - a1):
            raise InvalidInputError("trapezoid points are not coplanar")
        if top_len > 0 and bottom_len > 0 and np.linalg.norm(np.cross(top, bottom)) > REL_TOL * top_len * bottom_len:
            raise InvalidInputError(f"trapezoid sides a'a'' and bc are not parallel ({TRAPEZOID_ORDER})")
        if top_len > 0 and bottom_len > 0 and np.dot(top, bottom) < 0:
            raise InvalidInputError(f"trapezoid legs a'b and a''c cross each other ({TRAPEZOID_ORDER})")
        _check_isosceles(d1, d2, force, hint=TRAPEZOID_ORDER)
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
        3. H  = floor(L + SNAP) + 1                       (SNAP: see sampler.py)
        4. W  = floor(max(|a2 - a1|, |b2 - b1|) + SNAP) + 1  (widest row)
    """
    length = float(np.linalg.norm(b1 - a1))
    v1 = (b1 - a1) / length
    v2 = (b2 - a2) / length
    height = math.floor(length + SNAP) + 1
    width = math.floor(max(np.linalg.norm(a2 - a1), np.linalg.norm(b2 - b1)) + SNAP) + 1
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


def _check_distinct(points: dict[str, np.ndarray], allowed: tuple[tuple[str, str], ...] = ()) -> None:
    """Name the first pair of points that coincide, other than the allowed pairs, with its coordinates."""
    names = list(points)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if (first, second) not in allowed and np.linalg.norm(points[first] - points[second]) <= SAME_POINT_TOL:
                where = ", ".join(f"{v:g}" for v in points[first])
                raise InvalidInputError(f"points {first} and {second} are the same point ({where})")


def _check_edge(length: float, name: str) -> None:
    if length < 1.0:
        raise InvalidInputError(f"edge {name} is shorter than one voxel ({length:.3g})")


def _check_isosceles(l1: float, l2: float, force: bool, hint: str = "") -> None:
    if not force and abs(l1 - l2) > REL_TOL * max(l1, l2):
        suffix = f" ({hint})" if hint else ""
        raise NotIsoscelesError(f"plane is not isosceles: equal edges are {l1:.6g} and {l2:.6g}{suffix}")
