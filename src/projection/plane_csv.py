from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .errors import InvalidInputError, NotIsoscelesError
from .geometry import Plane, plane_from_points

POINT_COUNTS = {9: 3, 12: 4}
Warn = Callable[[str], None]


def parse_planes(
    path: str | Path,
    to_voxel: Callable[[np.ndarray], np.ndarray] | None = None,
    force: bool = False,
    warn: Warn | None = None,
) -> list[Plane]:
    """Read the planes of a CSV file. Three layouts are accepted:

    counted:        the lecturer's example. Every row is the number of points (3 or 4) followed by their
                    9 or 12 coordinates; a triangle leaves the fourth point's cells empty. Triangles and
                    trapezoids may be mixed. A header such as "Number of Points (3 or 4),x1,..." is skipped.
    plane per row:  every row has 9 numbers (triangle) or 12 numbers (trapezoid); one plane per row
    point per row:  every row has 3 numbers (x, y, z); 3 rows form a triangle, 4 rows a trapezoid.
                    Planes are separated by blank lines; without blank lines the row count decides
                    (see _group_point_rows).

    Pseudo-code:
        1. read all rows, drop empty cells at the end of a row (a trailing comma),
           keep the non-blank rows with their file line number
        2. if the first row has a non-empty, non-numeric cell: treat it as a header and drop it
        3. if the first row has 3 cells: point-per-row layout
               split the rows into groups of 3 or 4, build one plane per group
           else, for each row (keeping its file line number), parse every cell as a float:
               counted layout (the first row is n followed by 3n numbers, n = 3 or 4):
                   every row must be n followed by exactly 3n coordinates
               plane-per-row layout: require 9 or 12 numbers, the same for all rows
               reshape to points, apply to_voxel if given, build and validate the plane
        4. a plane that is valid except that it is not isosceles:
               force set      -> keep it
               warn given     -> report "<path>:<line>: ..." through warn and keep it
               otherwise      -> error
        5. any other failure raises InvalidInputError prefixed with "<path>:<line>:"
        6. require at least one plane
    """
    path = Path(path)
    if not path.is_file():
        raise InvalidInputError(f"CSV file not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [(n, cells) for n, row in enumerate(csv.reader(handle), start=1) if (cells := _drop_trailing_empty(row))]

    if rows and _is_header(rows[0][1]):
        rows = rows[1:]
    if not rows:
        raise InvalidInputError(f"{path}: no planes found")

    if len(rows[0][1]) == 3:
        groups = _group_point_rows(path, rows)
        return [_plane_from_point_rows(path, group, to_voxel, force, warn) for group in groups]
    return _planes_from_plane_rows(path, rows, to_voxel, force, warn)


def _planes_from_plane_rows(path: Path, rows: list, to_voxel, force: bool, warn: Warn | None) -> list[Plane]:
    planes: list[Plane] = []
    counted = _is_counted(rows[0][1])
    expected: int | None = None
    for line, row in rows:
        try:
            values = _parse_row(row, counted)
            if not counted:
                if expected is None:
                    expected = len(values)
                elif len(values) != expected:
                    raise InvalidInputError(f"expected {expected} values like the first row, got {len(values)}")
            points = values.reshape(-1, 3)
            if to_voxel is not None:
                points = to_voxel(points)
            planes.append(_build_plane(points, force, warn, f"{path}:{line}"))
        except InvalidInputError as exc:
            raise InvalidInputError(f"{path}:{line}: {exc}") from exc
    return planes


def _group_point_rows(path: Path, rows: list) -> list[list]:
    """Split point rows into planes.

    Pseudo-code:
        1. blocks = runs of rows with no blank line between them
        2. if there are several blocks: every block is one plane and must have 3 or 4 rows
        3. otherwise, with n rows:
               n = 3 or 4           -> one plane
               n a multiple of 12   -> ambiguous (triangles or trapezoids): reject, ask for blank lines
               n a multiple of 3    -> triangles, 3 rows each
               n a multiple of 4    -> trapezoids, 4 rows each
               else                 -> reject
    """
    blocks = [[rows[0]]]
    for previous, row in zip(rows, rows[1:], strict=False):
        if row[0] > previous[0] + 1:
            blocks.append([])
        blocks[-1].append(row)
    if len(blocks) > 1:
        for block in blocks:
            if len(block) not in (3, 4):
                raise InvalidInputError(
                    f"{path}:{block[0][0]}: a plane needs 3 or 4 point rows, this block has {len(block)}"
                )
        return blocks

    n = len(rows)
    if n in (3, 4):
        size = n
    elif n % 12 == 0:
        raise InvalidInputError(
            f"{path}:{rows[0][0]}: {n} point rows could be triangles or trapezoids; "
            "separate the planes with blank lines"
        )
    elif n % 3 == 0:
        size = 3
    elif n % 4 == 0:
        size = 4
    else:
        raise InvalidInputError(
            f"{path}:{rows[0][0]}: point-per-row layout needs 3 rows per triangle or 4 per trapezoid, got {n} rows"
        )
    return [rows[i:i + size] for i in range(0, n, size)]


def _plane_from_point_rows(path: Path, rows: list, to_voxel, force: bool, warn: Warn | None) -> Plane:
    first_line = rows[0][0]
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
        return _build_plane(corners, force, warn, f"{path}:{first_line}")
    except InvalidInputError as exc:
        raise InvalidInputError(f"{path}:{first_line}: {exc}") from exc


def _build_plane(points: np.ndarray, force: bool, warn: Warn | None, where: str) -> Plane:
    try:
        return plane_from_points(points, force=force)
    except NotIsoscelesError as exc:
        if warn is None:
            raise
        warn(f"{where}: {exc}; projected as given")
        return plane_from_points(points, force=True)


def _drop_trailing_empty(row: list[str]) -> list[str]:
    end = len(row)
    while end and not row[end - 1].strip():
        end -= 1
    return row[:end]


def _is_header(row: list[str]) -> bool:
    for cell in row:
        try:
            float(cell)
        except ValueError:
            if cell.strip():
                return True
    return False


def _is_counted(row: list[str]) -> bool:
    """True for a row of the counted layout: n (3 or 4) followed by 3n numbers."""
    try:
        count = float(row[0])
    except ValueError:
        return False
    return count in (3, 4) and len(row) == 1 + 3 * int(count)


def _parse_row(row: list[str], counted: bool) -> np.ndarray:
    """The coordinates of one plane row; in the counted layout the leading point count is checked and dropped."""
    try:
        values = np.array([float(c) for c in row], dtype=np.float64)
    except ValueError as exc:
        raise InvalidInputError("non-numeric value") from exc
    if counted:
        count = values[0]
        if count not in (3, 4):
            raise InvalidInputError(f"the number of points must be 3 or 4, got {count:g}")
        if len(values) - 1 != 3 * int(count):
            raise InvalidInputError(f"{int(count)} points need {3 * int(count)} coordinates, got {len(values) - 1}")
        return values[1:]
    if len(values) not in POINT_COUNTS:
        raise InvalidInputError(f"expected 9 or 12 values, got {len(values)}")
    return values
