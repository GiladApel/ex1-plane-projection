# Plane Projection of a 3D Volume

Given a 3D raster volume `M` (NRRD, uint8/16/32) and a plane defined by 3 points (isosceles triangle) or 4 points (isosceles trapezoid), compute the 2D image of the volume projected onto that plane. A CSV file lists many planes; the demo renders one video frame per plane.

Course: Introduction to Software Engineering, Ariel University. Assignment: Ex1.

## Problem

`M[x][y][z]` is an unsigned integer volume of size `sx*sy*sz`, with `1 <= sx,sy,sz < 2^10`. A plane is given by points `a, b, c` with `dist(a,b) = dist(a,c)`. The output image `E` has one pixel per unit step along the plane. Pixels whose sample position falls outside `M` are 0.

The assignment's pseudo-code is not fully defined (off-by-one at the apex, non-integer sizes, float coordinates). Every such gap is resolved explicitly in [docs/SRS.md](docs/SRS.md) and the exact algorithm is in [docs/ALGORITHM.md](docs/ALGORITHM.md).

## Scope

In scope:
- NRRD loading (3D, uint8/16/32, no multi-component data).
- Triangle and trapezoid planes, in voxel-index coordinates (optionally world coordinates).
- Nearest-neighbor sampling (default) and trilinear sampling.
- Three interchangeable projector backends: reference (literal loops), NumPy, Numba.
- CSV of planes in, mp4 video out, one frame per plane.
- Performance: at least 100 fps projection on a 500^3 volume (hard minimum 30 fps).

Out of scope: GUI, volume rendering, non-uniform voxel spacing in voxel mode, multi-channel volumes.

## Usage

```
python Ex0.py file.nrrd file.csv file.mp4 [--fps 15] [--interp nearest|trilinear]
                                          [--backend auto|numba|numpy|reference]
                                          [--world] [--force]
```

Setup:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

Tests: `pytest` (fast tests), `pytest -m slow` (performance).

## Documents

| Assignment item | Where |
|---|---|
| Problem and scope | this README |
| SRS | [docs/SRS.md](docs/SRS.md) |
| Main algorithm | [docs/ALGORITHM.md](docs/ALGORITHM.md) |
| NRRD data set | `data/MANIFEST.md` (Phase 6) |
| Testing strategy | `docs/TESTING.md` (Phase 7) |
| Demo | `Ex0.py` (Phase 5) |
| Experiment | `docs/EXPERIMENT.md` (Phase 6) |

## Status

Phase 1 (specification) complete. Implementation follows in gated phases.
