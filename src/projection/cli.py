from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .errors import InvalidInputError
from .io_nrrd import load_nrrd
from .plane_csv import parse_planes
from .projector import BACKENDS, make_projector
from .render import FrameComposer, IntensityWindow
from .sampler import Interpolation
from .video import Mp4Writer

EXIT_OK, EXIT_INTERNAL, EXIT_INVALID = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Ex0.py",
        description="Project a 3D NRRD volume onto the planes listed in a CSV and write one video frame per plane.",
    )
    parser.add_argument("nrrd", type=Path, help="3D volume in NRRD format")
    parser.add_argument("csv", type=Path, help="one plane per row: 9 numbers (triangle) or 12 numbers (trapezoid)")
    parser.add_argument("mp4", type=Path, help="output video")
    parser.add_argument("--fps", type=float, default=15.0, help="video frame rate (default 15)")
    parser.add_argument("--interp", choices=[m.value for m in Interpolation], default="nearest")
    parser.add_argument("--backend", choices=BACKENDS, default="auto")
    parser.add_argument("--world", action="store_true", help="CSV points are NRRD world coordinates, not voxel indices")
    parser.add_argument("--force", action="store_true", help="skip the isosceles check")
    return parser


def run(args: argparse.Namespace) -> int:
    """Pseudo-code:
        1. load the volume; parse and validate every plane (fail fast, nothing is written on error)
        2. build the projector, the intensity window and the fixed-size canvas
        3. project the first plane once, untimed (JIT warm-up)
        4. for each plane in CSV order: project (timed) -> map to 8-bit -> center on canvas -> write frame
        5. print frames written, canvas size, projection fps and end-to-end fps
    """
    if args.fps <= 0:
        raise InvalidInputError("--fps must be positive")
    if not args.mp4.parent.is_dir():
        raise InvalidInputError(f"output directory does not exist: {args.mp4.parent}")

    volume = load_nrrd(args.nrrd)
    to_voxel = volume.world_to_voxel if args.world else None
    planes = parse_planes(args.csv, to_voxel=to_voxel, force=args.force)

    projector = make_projector(args.backend, Interpolation(args.interp))
    window = IntensityWindow.from_volume(volume)
    composer = FrameComposer.for_planes(planes)
    buffers: dict[tuple[int, int], np.ndarray] = {}

    def project(plane):
        out = buffers.get(plane.shape)
        if out is None:
            out = buffers[plane.shape] = np.zeros(plane.shape, dtype=volume.dtype)
        return projector.project(volume, plane, out=out)

    project(planes[0])
    projection_time = 0.0
    started = time.perf_counter()
    with Mp4Writer(args.mp4, args.fps, composer.height, composer.width) as writer:
        for plane in planes:
            t0 = time.perf_counter()
            image = project(plane)
            projection_time += time.perf_counter() - t0
            writer.append(composer.compose(window.apply(image)))
    total = time.perf_counter() - started

    print(
        f"wrote {writer.frames} frames of {composer.height}x{composer.width} to {args.mp4} "
        f"[{type(projector).__name__}, {args.interp}] | projection {writer.frames / projection_time:.0f} fps, "
        f"end-to-end {writer.frames / total:.0f} fps"
    )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except InvalidInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except Exception as exc:
        print(f"internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INTERNAL
