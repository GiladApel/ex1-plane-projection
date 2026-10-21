from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from .errors import InvalidInputError
from .geometry import Plane, plane_from_points

POINT_COUNTS = {9: 3, 12: 4}


def parse_planes(
    path: str | Path,
    to_voxel: Callable[[np.ndarray], np.ndarray] | None = None,
    force: bool = False,
) -> list[Plane]:
    """Read one plane per CSV row.

    Pseudo-code:
        1. read all non-blank rows
        2. if the first row has a non-numeric cell: treat it as a header and drop it
        3. for each remaining row (keeping its file line number):
             a. parse every cell as a float
             b. require 9 cells (triangle) or 12 cells (trapezoid), the same for all rows
             c. reshape to points, apply to_voxel if given
             d. build and validate the plane
           any failure raises InvalidInputError prefixed with "<path>:<line>:"
        4. require at least one plane
    """
    path = Path(path)
    if not path.is_file():
        raise InvalidInputError(f"CSV file not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [(n, row) for n, row in enumerate(csv.reader(handle), start=1) if any(c.strip() for c in row)]

    if rows and not _is_numeric(rows[0][1]):
        rows = rows[1:]
    if not rows:
        raise InvalidInputError(f"{path}: no planes found")

    planes: list[Plane] = []
    expected: int | None = None
    for line, row in rows:
        try:
            values = _parse_row(row)
            if expected is None:
                expected = len(values)
            elif len(values) != expected:
                raise InvalidInputError(f"expected {expected} values like the first row, got {len(values)}")
            points = values.reshape(-1, 3)
            if to_voxel is not None:
                points = to_voxel(points)
            planes.append(plane_from_points(points, force=force))
        except InvalidInputError as exc:
            raise InvalidInputError(f"{path}:{line}: {exc}") from exc
    return planes


def _is_numeric(row: list[str]) -> bool:
    try:
        [float(c) for c in row]
    except ValueError:
        return False
    return True


def _parse_row(row: list[str]) -> np.ndarray:
    try:
        values = np.array([float(c) for c in row], dtype=np.float64)
    except ValueError as exc:
        raise InvalidInputError("non-numeric value") from exc
    if len(values) not in POINT_COUNTS:
        raise InvalidInputError(f"expected 9 or 12 values, got {len(values)}")
    return values
