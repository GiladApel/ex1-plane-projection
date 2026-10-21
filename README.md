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

CSV formats (both give frames in row order, see SRS FR2):
```
# plane per row: 9 numbers = triangle (a, b, c), 12 numbers = trapezoid (a', b, a'', c); one frame per row
128,5,60, 28,250,60, 228,250,60

# point per row: 3 or 4 rows of x,y,z = one plane, one frame
128,5,60
28,250,60
228,250,60
```
Coordinates are voxel indices; `--world` reads them as NRRD world coordinates.

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
| Design (classes, decisions) | [docs/DESIGN.md](docs/DESIGN.md) |
| Main algorithm | [docs/ALGORITHM.md](docs/ALGORITHM.md) |
| NRRD data set | [data/MANIFEST.md](data/MANIFEST.md) |
| Testing strategy | `docs/TESTING.md` (Phase 7) |
| Demo | [Ex0.py](Ex0.py) |
| Experiment | [docs/EXPERIMENT.md](docs/EXPERIMENT.md) |

## Assumptions to confirm

The assignment leaves three points open; the choices below are documented in [docs/SRS.md](docs/SRS.md) section 6 and are easy to change.

1. CSV coordinates are voxel indices (`--world` switches to NRRD world coordinates).
2. A CSV is either one plane per row (9 or 12 numbers) or 3 or 4 rows of 3 numbers forming one plane. Both are accepted.
3. Nearest-neighbor sampling is the default reading of `v = M[p]`; trilinear is available with `--interp trilinear`.

## Status

Specification, implementation, demo, data set and experiments are complete. Testing strategy document and CI are the remaining items.
