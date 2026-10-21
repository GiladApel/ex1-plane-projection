"""Write synthetic NRRD volumes: python scripts/make_synthetic.py OUT_DIR [--size 500]."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from projection import synthetic  # noqa: E402

KINDS = {
    "gradient": synthetic.gradient,
    "sphere": synthetic.sphere,
    "checkerboard": synthetic.checkerboard,
    "random": synthetic.random_volume,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shape = (args.size,) * 3
    for name, make in KINDS.items():
        for dtype in synthetic.DTYPES:
            if name == "gradient" and args.size - 1 > np.iinfo(dtype).max:
                continue
            path = args.out_dir / f"{name}_{np.dtype(dtype).name}_{args.size}.nrrd"
            synthetic.save_nrrd(path, make(shape, dtype))
            print(path)


if __name__ == "__main__":
    main()
