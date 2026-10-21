"""Download the 3D Slicer sample volumes listed in data/MANIFEST.md into data/.

python scripts/download_data.py [NAME ...]

Files that already exist with the right SHA-256 are skipped. A file with a different checksum is
never overwritten; the script reports it and moves on.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASETS = {
    "MRHead": ("MRHead.nrrd", "cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93"),
    "MRBrainTumor1": ("MRBrainTumor1.nrrd", "998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95"),
    "CTChest": ("CTChest.nrrd", "4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e"),
    "CTACardio": ("CTACardio.nrrd", "3b0d4eb1a7d8ebb0c5a89cc0504640f76a030b4e869e33ff34c564c3d3b88ad2"),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str) -> bool:
    filename, checksum = DATASETS[name]
    target = DATA_DIR / filename
    if target.exists():
        if sha256_of(target) == checksum:
            print(f"{name}: already present")
            return True
        print(f"{name}: {target} exists with a different checksum; not overwriting", file=sys.stderr)
        return False
    partial = target.with_suffix(".part")
    print(f"{name}: downloading")
    urllib.request.urlretrieve(BASE_URL + checksum, partial)
    if sha256_of(partial) != checksum:
        partial.unlink()
        print(f"{name}: checksum mismatch after download", file=sys.stderr)
        return False
    partial.rename(target)
    return True


def main() -> int:
    names = sys.argv[1:] or list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        print(f"unknown dataset(s): {', '.join(unknown)}; choose from {', '.join(DATASETS)}", file=sys.stderr)
        return 2
    DATA_DIR.mkdir(exist_ok=True)
    results = [fetch(n) for n in names]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
