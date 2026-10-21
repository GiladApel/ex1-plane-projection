from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


class Mp4Writer:
    """Context manager writing 8-bit gray frames of one fixed even size to an H.264 mp4."""

    def __init__(self, path: str | Path, fps: float, height: int, width: int) -> None:
        if height % 2 or width % 2:
            raise ValueError(f"H.264 needs even frame sizes, got {height}x{width}")
        self._path = str(path)
        self._fps = fps
        self._shape = (height, width)
        self._writer: Any = None
        self.frames = 0

    def __enter__(self) -> Mp4Writer:
        self._writer = imageio.get_writer(
            self._path, fps=self._fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1, quality=8
        )
        return self

    def append(self, frame: np.ndarray) -> None:
        if frame.shape != self._shape or frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 frame of shape {self._shape}, got {frame.dtype} {frame.shape}")
        self._writer.append_data(np.repeat(frame[:, :, None], 3, axis=2))
        self.frames += 1

    def __exit__(self, *exc_info) -> None:
        self._writer.close()
