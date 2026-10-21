from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .errors import InvalidInputError
from .geometry import Plane, plane_from_points

POINT_COUNTS = {9: 3, 12: 4}


def parse_planes(
    path: str | Path,
    to_voxel: Callable[[np.ndarray], np.ndarray] | None = None,
    force: bool = False,
) -> list[Plane]:
    """Read the planes of a CSV file. Two layouts are accepted:

    plane per row:  every row has 9 numbers (triangle) or 12 numbers (trapezoid); one plane per row
    point per row:  3 or 4 rows of exactly 3 numbers (x, y, z); together they are one plane

    Pseudo-code:
        1. read all non-blank rows
        2. if the first row has a non-numeric cell: treat it as a header and drop it
        3. if the first row has 3 cells: point-per-row layout
               require 3 or 4 rows of 3 numbers, build one plane from them
           else: plane-per-row layout
               for each row (keeping its file line number):
                   parse every cell as a float; require 9 or 12 cells, the same for all rows
                   reshape to points, apply to_voxel if given, build and validate the plane
        4. any failure raises InvalidInputError prefixed with "<path>:<line>:"
        5. require at least one plane
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

    if len(rows[0][1]) == 3:
        return [_plane_from_point_rows(path, rows, to_voxel, force)]
    return _planes_from_plane_rows(path, rows, to_voxel, force)


def _planes_from_plane_rows(path: Path, rows: list, to_voxel, force: bool) -> list[Plane]:
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


def _plane_from_point_rows(path: Path, rows: list, to_voxel, force: bool) -> Plane:
    first_line = rows[0][0]
    if len(rows) not in (3, 4):
        raise InvalidInputError(f"{path}:{first_line}: point-per-row layout needs 3 or 4 rows, got {len(rows)}")
    points = []
    for line, row in rows:
        try:
            values = np.array([float(c) for c in row], dtype=np.float64)
        except ValueError as exc:
            raise InvalidInputError(f"{path}:{line}: non-numeric value") from exc
        if len(values) != 3:
            raise InvalidInputError(f"{path}:{line}: expected 3 values (x, y, z) per point, got {len(values)}")
        points.append(values)
    corners = np.array(points)
    try:
        if to_voxel is not None:
            corners = to_voxel(corners)
        return plane_from_points(corners, force=force)
    except InvalidInputError as exc:
        raise InvalidInputError(f"{path}:{first_line}: {exc}") from exc


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
