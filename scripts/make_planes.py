"""Write a plane CSV for a volume.

python scripts/make_planes.py sweep  VOLUME.nrrd OUT.csv --axis y [--frames N]
python scripts/make_planes.py rotate VOLUME.nrrd OUT.csv --axis y --frames 180

sweep:  one rectangle per slice perpendicular to the axis (or N evenly spaced ones).
rotate: a plane through the volume center containing the axis, turning 360 degrees around it.
Coordinates are voxel indices. The volume shape is read from the NRRD header.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import nrrd  # noqa: E402

from projection import plane_gen  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["sweep", "rotate"])
    parser.add_argument("nrrd", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--axis", choices=list(plane_gen.AXES), default="z")
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    shape = [int(s) for s in nrrd.read_header(str(args.nrrd))["sizes"]]
    axis = plane_gen.AXES[args.axis]
    if args.mode == "sweep":
        sets = plane_gen.axial_sweep(shape, axis, args.frames)
    elif args.frames:
        sets = plane_gen.rotation(shape, axis, args.frames)
    else:
        parser.error("rotate needs --frames")
    plane_gen.write_planes_csv(args.out, sets)
    print(f"wrote {len(sets)} planes for volume {shape} to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
