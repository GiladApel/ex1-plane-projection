"""Projection throughput benchmark.

python scripts/bench.py [--size 500] [--dtype uint8] [--frames 200] [--backend numba numpy]
                        [--interp nearest trilinear] [--seed 0]

Measures only project(volume, plane, out) per frame: volume loading, plane construction, JIT
compilation (one untimed warm-up call) and video encoding are excluded.
"""
import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from projection import synthetic  # noqa: E402
from projection.projector import make_projector  # noqa: E402
from projection.sampler import Interpolation  # noqa: E402
from projection.volume import Volume  # noqa: E402


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
    parser.add_argument("--dtype", choices=["uint8", "uint16", "uint32"], default="uint8")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--backend", nargs="+", default=["numba", "numpy"])
    parser.add_argument("--interp", nargs="+", default=["nearest"], choices=[m.value for m in Interpolation])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    volume = Volume(synthetic.random_volume((args.size,) * 3, getattr(np, args.dtype), seed=args.seed))
    planes = synthetic.oblique_rectangles(args.size, args.frames, args.seed)
    print(f"{platform.processor()} | python {platform.python_version()} | volume {args.size}^3 {args.dtype} "
          f"| frame {planes[0].shape[0]}x{planes[0].shape[1]} | {args.frames} oblique planes")
    print(f"{'backend':<10}{'interp':<11}{'median fps':>12}{'mean fps':>10}{'p95 ms':>9}{'min fps':>9}")
    for backend in args.backend:
        for interp in args.interp:
            times = measure(make_projector(backend, Interpolation(interp)), volume, planes)
            print(f"{backend:<10}{interp:<11}{1 / np.median(times):>12.0f}{1 / times.mean():>10.0f}"
                  f"{np.percentile(times, 95) * 1e3:>9.2f}{1 / times.max():>9.0f}")


if __name__ == "__main__":
    main()
