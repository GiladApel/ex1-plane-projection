from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import Plane
from .volume import Volume

_MAX_WINDOW_SAMPLES = 2_000_000


@dataclass(frozen=True)
class IntensityWindow:
    """Linear map from volume values to 8-bit gray: lo -> 0, hi -> 255, clipped outside."""

    lo: float
    hi: float

    @classmethod
    def from_volume(cls, volume: Volume, low_percentile: float = 1.0, high_percentile: float = 99.0) -> IntensityWindow:
        """Pseudo-code:
            1. take every step-th voxel so that at most ~2 million voxels remain
            2. lo, hi = the low and high percentiles of that sample
            3. if hi <= lo: hi = lo + 1
        """
        flat = volume.flat
        step = max(1, flat.size // _MAX_WINDOW_SAMPLES)
        lo, hi = np.percentile(flat[::step], [low_percentile, high_percentile])
        return cls(float(lo), float(max(hi, lo + 1.0)))

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Pseudo-code: gray = floor(clip((value - lo) / (hi - lo), 0, 1) * 255 + 0.5) as uint8."""
        scaled = np.clip((image.astype(np.float64) - self.lo) / (self.hi - self.lo), 0.0, 1.0)
        return np.floor(scaled * 255.0 + 0.5).astype(np.uint8)


class FrameComposer:
    """Places variable-size gray images centered on one fixed-size black canvas."""

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self._canvas = np.zeros((height, width), dtype=np.uint8)

    @classmethod
    def for_planes(cls, planes: Sequence[Plane]) -> FrameComposer:
        """Canvas = the largest H and W over all planes, each rounded up to an even number (H.264)."""
        height = max(p.shape[0] for p in planes)
        width = max(p.shape[1] for p in planes)
        return cls(height + height % 2, width + width % 2)

    def compose(self, image: np.ndarray) -> np.ndarray:
        """Return the canvas with image pasted at its center. The returned array is reused by the next call."""
        h, w = image.shape
        if h > self.height or w > self.width:
            raise ValueError(f"image {image.shape} does not fit the {self.height}x{self.width} canvas")
        top, left = (self.height - h) // 2, (self.width - w) // 2
        self._canvas.fill(0)
        self._canvas[top:top + h, left:left + w] = image
        return self._canvas
