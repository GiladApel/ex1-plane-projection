"""Demo entry point: python Ex0.py file.nrrd file.csv file.mp4 [options]"""
import sys
from pathlib import Path

# The package lives in src/; add it so the script runs without installing anything.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from projection.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
