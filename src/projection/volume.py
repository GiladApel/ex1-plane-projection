from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import InvalidInputError

SUPPORTED_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.uint32))
MAX_DIM = 1 << 10


@dataclass(frozen=True, eq=False)
class Volume:
    """Immutable 3D unsigned-integer volume indexed data[x][y][z].

    origin and directions (one row per axis) place voxel indices in world space:
    world = origin + directions.T @ index. Both are None when the file has no spatial metadata.
    value_offset is the amount that was subtracted from the original values to make them unsigned.
    """

    data: np.ndarray
    origin: np.ndarray | None = None
    directions: np.ndarray | None = None
    value_offset: int = 0

    def __post_init__(self) -> None:
        data = np.ascontiguousarray(self.data).view()
        if data.ndim != 3:
            raise InvalidInputError(f"volume must be 3D, got {data.ndim}D")
        if data.dtype not in SUPPORTED_DTYPES:
            raise InvalidInputError(f"unsupported volume dtype {data.dtype}; expected uint8/16/32")
        if not all(1 <= s < MAX_DIM for s in data.shape):
            raise InvalidInputError(f"each volume dimension must be in [1, {MAX_DIM}), got {data.shape}")
        data.flags.writeable = False
        object.__setattr__(self, "data", data)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def flat(self) -> np.ndarray:
        """1D view of the data in C order; index = (x * sy + y) * sz + z."""
        return self.data.reshape(-1)

    def world_to_voxel(self, points: np.ndarray) -> np.ndarray:
        """Convert (n, 3) world coordinates to voxel indices.

        Pseudo-code:
            1. require origin and directions
            2. M = directions.T          (columns are the axis vectors)
            3. index = inverse(M) @ (world - origin) for every point
        """
        if self.origin is None or self.directions is None:
            raise InvalidInputError("volume has no spatial metadata; cannot use world coordinates")
        m = np.asarray(self.directions, dtype=np.float64).T
        try:
            inv = np.linalg.inv(m)
        except np.linalg.LinAlgError as exc:
            raise InvalidInputError("volume space directions are singular") from exc
        return (np.asarray(points, dtype=np.float64) - self.origin) @ inv.T
