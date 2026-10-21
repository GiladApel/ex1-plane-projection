"""Projection throughput benchmark.

python scripts/bench.py [--size 500] [--dtype uint8 ...] [--planes oblique axis] [--backend numba numpy]
                        [--interp nearest ...] [--frames 200] [--seed 0] [--csv]

Measures only project(volume, plane, out) per frame: volume creation, plane construction, JIT
compilation (one untimed warm-up call) and video encoding are excluded.
oblique: random oblique squares through the volume center. axis: random axis-aligned full slices.
Set NUMBA_NUM_THREADS to limit Numba's thread count.
"""
import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from projection import plane_gen, synthetic  # noqa: E402
from projection.geometry import plane_from_points  # noqa: E402
from projection.projector import make_projector  # noqa: E402
from projection.sampler import Interpolation  # noqa: E402
from projection.volume import Volume  # noqa: E402

COLUMNS = ["dtype", "planes", "backend", "interp", "median_fps", "mean_fps", "p95_ms", "min_fps"]


def make_planes(kind: str, size: int, count: int, seed: int):
    if kind == "oblique":
        return synthetic.oblique_rectangles(size, count, seed)
    rng = np.random.default_rng(seed)
    sweeps = [plane_gen.axial_sweep((size,) * 3, axis) for axis in range(3)]
    picks = [(rng.integers(3), rng.integers(size)) for _ in range(count)]
    return [plane_from_points(sweeps[axis][k]) for axis, k in picks]


def measure(projector, volume: Volume, planes) -> np.ndarray:
    out = np.zeros(planes[0].shape, dtype=volume.dtype)
    projector.project(volume, planes[0], out=out)
    times = []
    for plane in planes:
        start = time.perf_counter()
        projector.project(volume, plane, out=out)
        times.append(time.perf_counter() - start)
    return np.array(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--dtype", nargs="+", choices=["uint8", "uint16", "uint32"], default=["uint8"])
    parser.add_argument("--planes", nargs="+", choices=["oblique", "axis"], default=["oblique"])
    parser.add_argument("--backend", nargs="+", default=["numba", "numpy"])
    parser.add_argument("--interp", nargs="+", default=["nearest"], choices=[m.value for m in Interpolation])
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    rows = []
    for dtype in args.dtype:
        volume = Volume(synthetic.random_volume((args.size,) * 3, getattr(np, dtype), seed=args.seed))
        for kind in args.planes:
            planes = make_planes(kind, args.size, args.frames, args.seed)
            for backend in args.backend:
                for interp in args.interp:
                    times = measure(make_projector(backend, Interpolation(interp)), volume, planes)
                    rows.append([dtype, kind, backend, interp, f"{1 / np.median(times):.0f}",
                                 f"{1 / times.mean():.0f}", f"{np.percentile(times, 95) * 1e3:.2f}",
                                 f"{1 / times.max():.0f}"])
    if args.csv:
        print(",".join(COLUMNS))
        print("\n".join(",".join(row) for row in rows))
        return
    print(f"{platform.processor()} | python {platform.python_version()} | volume {args.size}^3 | {args.frames} planes")
    print(f"{'dtype':<8}{'planes':<9}{'backend':<9}{'interp':<11}{'median fps':>11}{'mean fps':>10}{'p95 ms':>9}{'min fps':>9}")
    for r in rows:
        print(f"{r[0]:<8}{r[1]:<9}{r[2]:<9}{r[3]:<11}{r[4]:>11}{r[5]:>10}{r[6]:>9}{r[7]:>9}")


if __name__ == "__main__":
    main()
