from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import Plane
from .volume import Volume

_MAX_WINDOW_SAMPLES = 2_000_000
_BACKGROUND_SHARE = 0.05
EMPTY_CANVAS = (64, 64)
ERROR_BACKGROUND = 64
ERROR_CROSS = 96
ERROR_TEXT = 255
_SMALLEST_FONT = 7


@dataclass(frozen=True)
class IntensityWindow:
    """Linear map from volume values to 8-bit gray: lo -> 0, hi -> 255, clipped outside."""

    lo: float
    hi: float

    @classmethod
    def from_volume(cls, volume: Volume, low_percentile: float = 1.0, high_percentile: float = 99.0) -> IntensityWindow:
        """Pseudo-code:
            1. take every step-th voxel so that at most ~2 million voxels remain
            2. background = voxels equal to the sample minimum; if they are at least 5% of the sample and
               other voxels remain, drop them (padding or air around the object, see SRS D12)
            3. lo, hi = the low and high percentiles of what is left;
               if hi <= lo there (for example a binary mask), use the whole sample instead
            4. if hi <= lo: hi = lo + 1
        """
        flat = volume.flat
        step = max(1, flat.size // _MAX_WINDOW_SAMPLES)
        sample = flat[::step]
        candidates = [sample]
        foreground = sample[sample > sample.min()]
        if 0 < foreground.size <= (1.0 - _BACKGROUND_SHARE) * sample.size:
            candidates.insert(0, foreground)
        for values in candidates:
            lo, hi = np.percentile(values, [low_percentile, high_percentile])
            if hi > lo:
                break
        return cls(float(lo), float(max(hi, lo + 1.0)))

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Pseudo-code:
            gray(value) = floor(clip((value - lo) / (hi - lo), 0, 1) * 255 + 0.5) as uint8
            uint8 and uint16 images: look gray up in a table of every possible value, built once per dtype
            uint32 images: compute gray per pixel (a table of 2^32 entries is too large)
        """
        if image.dtype.itemsize <= 2:
            return self._table(image.dtype)[image]
        return self._gray(image)

    def _table(self, dtype: np.dtype) -> np.ndarray:
        tables = self.__dict__.setdefault("_tables", {})
        if dtype not in tables:
            tables[dtype] = self._gray(np.arange(np.iinfo(dtype).max + 1))
        return tables[dtype]

    def _gray(self, values: np.ndarray) -> np.ndarray:
        scaled = np.clip((values.astype(np.float64) - self.lo) / (self.hi - self.lo), 0.0, 1.0)
        return np.floor(scaled * 255.0 + 0.5).astype(np.uint8)


class FrameComposer:
    """Places variable-size gray images centered on one fixed-size black canvas."""

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self._canvas = np.zeros((height, width), dtype=np.uint8)

    @classmethod
    def for_planes(cls, planes: Sequence[Plane]) -> FrameComposer:
        """Canvas = the largest H and W over all planes, each rounded up to an even number (H.264);
        EMPTY_CANVAS when there is no plane (every row of the file was bad)."""
        if not planes:
            return cls(*EMPTY_CANVAS)
        height = max(p.shape[0] for p in planes)
        width = max(p.shape[1] for p in planes)
        return cls(height + height % 2, width + width % 2)

    def error_frame(self, frame: int, line: int, cause: str) -> np.ndarray:
        """The frame of a plane that could not be projected; reused like compose(). It cannot be mistaken for
        the black frame of a plane outside the volume.

        Pseudo-code:
            1. fill dark gray, draw a white border and a faint X
            2. text blocks: "ERROR" (large), "frame N, line L", the cause wrapped to the width inside the border
            3. for a font size from large to small: if all blocks fit inside the border, draw them centered, stop
            4. if none fits even at the smallest size: drop the cause, then the frame and line; at worst only
               the border and the X remain
        """
        h, w = self.height, self.width
        canvas = self._canvas
        canvas.fill(ERROR_BACKGROUND)
        border = max(1, min(h, w) // 64)
        rows = np.arange(h)
        cols = np.floor(rows * (w - 1) / max(h - 1, 1) + 0.5).astype(np.intp)
        for shift in range(-border // 2, border - border // 2):
            shifted = np.clip(cols + shift, 0, w - 1)
            canvas[rows, shifted] = ERROR_CROSS
            canvas[rows, w - 1 - shifted] = ERROR_CROSS
        for edge in (canvas[:border], canvas[h - border:], canvas[:, :border], canvas[:, w - border:]):
            edge.fill(ERROR_TEXT)

        margin = 3 * border + 2
        box_w, box_h = w - 2 * margin, h - 2 * margin
        for blocks in (("ERROR", f"frame {frame}, line {line}", cause), ("ERROR", f"frame {frame}, line {line}"),
                       ("ERROR",)):
            layout = _fit_text(blocks, box_w, box_h)
            if layout is not None:
                _draw_lines(canvas, layout, h)
                break
        return canvas

    def compose(self, image: np.ndarray) -> np.ndarray:
        """Return the canvas with image pasted at its center. The returned array is reused by the next call."""
        h, w = image.shape
        if h > self.height or w > self.width:
            raise ValueError(f"image {image.shape} does not fit the {self.height}x{self.width} canvas")
        top, left = (self.height - h) // 2, (self.width - w) // 2
        self._canvas.fill(0)
        self._canvas[top:top + h, left:left + w] = image
        return self._canvas


@lru_cache(maxsize=64)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Pillow's bundled font at this size: no font file to ship, the same glyphs on every machine."""
    return ImageFont.load_default(size=size)


def _text_size(text: str, size: int) -> tuple[int, int]:
    left, top, right, bottom = _font(size).getbbox(text)
    return int(right - left), int(bottom - top)


def _wrap(text: str, size: int, width: int) -> list[str] | None:
    """Split text at spaces into lines no wider than width; None if one word alone is wider."""
    lines: list[str] = []
    for word in text.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and _text_size(candidate, size)[0] <= width:
            lines[-1] = candidate
        elif _text_size(word, size)[0] <= width:
            lines.append(word)
        else:
            return None
    return lines


def _fit_text(blocks: Sequence[str], width: int, height: int) -> list[tuple[str, int]] | None:
    """(line, font size) for every line of the blocks at the largest size that fits width x height:
    the first block at twice the size of the others, the others wrapped at spaces."""
    for size in range(max(_SMALLEST_FONT, height // 6), _SMALLEST_FONT - 1, -1):
        wrapped = [_wrap(block, size, width) for block in blocks[1:]]
        if any(block is None for block in wrapped):
            continue
        lines = [(blocks[0], 2 * size)] + [(text, size) for block in wrapped if block for text in block]
        if all(_text_size(text, s)[0] <= width for text, s in lines) and _block_height(lines) <= height:
            return lines
    return None


def _block_height(lines: Sequence[tuple[str, int]]) -> int:
    return sum(round(1.3 * size) for _, size in lines)


def _draw_lines(canvas: np.ndarray, lines: Sequence[tuple[str, int]], height: int) -> None:
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    top = (height - _block_height(lines)) // 2
    for text, size in lines:
        draw.text((canvas.shape[1] // 2, top), text, fill=ERROR_TEXT, font=_font(size), anchor="mt")
        top += round(1.3 * size)
    canvas[:] = np.asarray(image)
