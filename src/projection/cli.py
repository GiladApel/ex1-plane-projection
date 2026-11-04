from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .errors import InvalidInputError
from .io_nrrd import load_nrrd
from .plane_csv import read_planes
from .projector import BACKENDS, NumpyProjector, make_projector
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
    parser.add_argument(
        "csv",
        type=Path,
        help="planes, one frame each. One plane per row: the number of points (3 or 4) and then the coordinates "
        "(the lecturer's format), or just 9 numbers (triangle a,b,c) or 12 (trapezoid a',b,a'',c). "
        "Or one point per row: 3 rows = triangle, 4 = trapezoid, planes separated by blank lines",
    )
    parser.add_argument("mp4", type=Path, help="output video")
    parser.add_argument("--fps", type=float, default=15.0, help="video frame rate (default 15)")
    parser.add_argument("--interp", choices=[m.value for m in Interpolation], default="nearest")
    parser.add_argument("--backend", choices=BACKENDS, default="auto")
    parser.add_argument("--world", action="store_true", help="CSV points are NRRD world coordinates, not voxel indices")
    parser.add_argument("--force", action="store_true", help="accept non-isosceles planes without a warning")
    return parser


def run(args: argparse.Namespace) -> int:
    """Pseudo-code:
        1. check the options and the output folder; load the volume; read every plane of the CSV
           (a problem with a whole file stops here, before anything is written)
        2. in plane order, warn about every plane accepted although not isosceles, and every plane that
           could not be built (it will be an error frame)
        3. build the projector, the intensity window and the fixed-size canvas
        4. project the first good plane once, untimed (JIT warm-up)
        5. for each plane in CSV order:
               good plane: project (timed) -> map to 8-bit -> center on canvas -> write frame
               bad plane, or projection failed: write an error frame
        6. print frames written, canvas size, projection fps and end-to-end fps,
           then every error frame with its CSV line and cause
        7. exit code 0, or 1 if a projection failed (a bug, not bad input)
    """
    if args.fps <= 0:
        raise InvalidInputError("--fps must be positive")
    if not args.mp4.parent.is_dir():
        raise InvalidInputError(f"output directory does not exist: {args.mp4.parent}")

    volume = load_nrrd(args.nrrd)
    if args.world and volume.origin is None:
        raise InvalidInputError(f"{args.nrrd}: --world needs 'space origin' and 'space directions' in the file")
    to_voxel = volume.world_to_voxel if args.world else None
    rows = read_planes(args.csv, to_voxel=to_voxel, force=args.force, warn=_accept)
    errors: dict[int, str] = {}
    for frame, row in enumerate(rows, start=1):
        if row.warning is not None:
            _warn(f"{args.csv}:{row.line}: {row.warning}")
        if row.error is not None:
            errors[frame] = f"line {row.line}: {row.error}"
            _warn(f"{args.csv}:{row.line}: {row.error}; frame {frame} is an error frame")
    planes = [row.plane for row in rows if row.plane is not None]

    projector = make_projector(args.backend, Interpolation(args.interp))
    if isinstance(projector, NumpyProjector) and args.interp == Interpolation.TRILINEAR.value:
        print(
            "warning: trilinear sampling on the NumPy backend is slow (about 20 fps on a 500^3 volume); "
            "install numba or use --interp nearest",
            file=sys.stderr,
        )
    window = IntensityWindow.from_volume(volume)
    composer = FrameComposer.for_planes(planes)
    buffers: dict[tuple[int, int], np.ndarray] = {}

    def project(plane):
        out = buffers.get(plane.shape)
        if out is None:
            out = buffers[plane.shape] = np.zeros(plane.shape, dtype=volume.dtype)
        return projector.project(volume, plane, out=out)

    if planes:
        project(planes[0])
    projection_time = 0.0
    projected = 0
    failed = False
    started = time.perf_counter()
    with projector, Mp4Writer(args.mp4, args.fps, composer.height, composer.width) as writer:
        for frame, row in enumerate(rows, start=1):
            if row.plane is None:
                writer.append(composer.error_frame())
                continue
            t0 = time.perf_counter()
            try:
                image = project(row.plane)
            except Exception as exc:
                failed = True
                errors[frame] = f"line {row.line}: internal error while projecting ({type(exc).__name__}: {exc})"
                _warn(f"{args.csv}:{row.line}: {errors[frame]}; frame {frame} is an error frame")
                writer.append(composer.error_frame())
                continue
            projection_time += time.perf_counter() - t0
            projected += 1
            writer.append(composer.compose(window.apply(image)))
    total = time.perf_counter() - started

    speed = f"projection {projected / projection_time:.0f} fps" if projected else "no plane projected"
    print(
        f"wrote {writer.frames} frames of {composer.height}x{composer.width} to {args.mp4} "
        f"[{type(projector).__name__}, {args.interp}] | {speed}, end-to-end {writer.frames / total:.0f} fps"
    )
    if errors:
        print(f"{len(errors)} of {writer.frames} frames are error frames:")
        for frame in sorted(errors):
            print(f"  frame {frame}, {errors[frame]}")
    return EXIT_INTERNAL if failed else EXIT_OK


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _accept(message: str) -> None:
    """Accept non-isosceles planes while reading; run() prints their warnings in plane order."""


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
