from __future__ import annotations

from pathlib import Path

import nrrd
import numpy as np

from .errors import InvalidInputError
from .volume import Volume

_SIGNED_TO_UNSIGNED = {
    np.dtype(np.int8): np.uint8,
    np.dtype(np.int16): np.uint16,
    np.dtype(np.int32): np.uint32,
}


def load_nrrd(path: str | Path) -> Volume:
    """Load a 3D NRRD file into a Volume.

    Pseudo-code:
        1. read the file with index order F, so data[x][y][z] matches the header axis order
        2. reject anything that is not exactly 3D (this also rejects multi-component data)
        3. if dtype is unsigned 8/16/32: keep it
           elif dtype is signed 8/16/32: subtract min when min < 0, cast to the unsigned type of
                the same width, and remember the subtracted amount as value_offset
           else: reject
        4. read optional 'space origin' and 'space directions' for world coordinates
    """
    path = Path(path)
    if not path.is_file():
        raise InvalidInputError(f"NRRD file not found: {path}")
    try:
        data, header = nrrd.read(str(path), index_order="F")
    except (nrrd.NRRDError, OSError, ValueError) as exc:
        raise InvalidInputError(f"cannot read NRRD file {path}: {exc}") from exc

    if data.ndim != 3:
        raise InvalidInputError(f"{path}: expected a 3D volume, got {data.ndim}D data")

    data, offset = _to_unsigned(data, path)
    origin, directions = _spatial_metadata(header)
    return Volume(data, origin, directions, offset)


def _to_unsigned(data: np.ndarray, path: Path) -> tuple[np.ndarray, int]:
    if data.dtype.kind == "u" and data.dtype.itemsize in (1, 2, 4):
        return data, 0
    unsigned = _SIGNED_TO_UNSIGNED.get(data.dtype)
    if unsigned is None:
        raise InvalidInputError(f"{path}: unsupported dtype {data.dtype}; expected integer 8/16/32 bit")
    low = int(data.min())
    offset = low if low < 0 else 0
    return (data.astype(np.int64) - offset).astype(unsigned), offset


def _spatial_metadata(header: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    origin = header.get("space origin")
    directions = header.get("space directions")
    if origin is None or directions is None:
        return None, None
    origin = np.asarray(origin, dtype=np.float64)
    try:
        directions = np.asarray(directions, dtype=np.float64)
    except (TypeError, ValueError):
        return None, None
    if origin.shape != (3,) or directions.shape != (3, 3):
        return None, None
    return origin, directions
