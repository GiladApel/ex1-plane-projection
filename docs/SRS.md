# Software Requirements Specification

## 1. Purpose
Project a 3D volume onto arbitrary planes efficiently and render the sequence of projections as a video.

## 2. Definitions
- **Volume `M`**: 3D array indexed `M[x][y][z]`, dtype uint8/16/32.
- **Plane**: 3 points (isosceles triangle `a,b,c`) or 4 points (isosceles trapezoid `a',b,a'',c`).
- **Rails**: the two equal-length edges the algorithm walks: rail 1 `A1 -> B1`, rail 2 `A2 -> B2`.
- **Frame**: the 2D matrix `E` produced for one plane.
- **Voxel space**: coordinates are array indices.

## 3. Functional requirements

| ID | Requirement |
|---|---|
| FR1 | Load a 3D NRRD file into a `Volume`. Reject non-3D, multi-component and unsupported dtypes with a clear error. |
| FR2 | Parse a CSV: each row is one plane (9 numbers = triangle, 12 numbers = trapezoid). Optional header row. All rows have the same point count. Errors report the line number. |
| FR3 | Validate each plane: finite values, non-degenerate, isosceles within relative tolerance 1e-3 (`--force` skips the equal-length check), trapezoid coplanar with `a'a'' || bc`. |
| FR4 | Project the volume onto each plane (see ALGORITHM.md). Output dtype equals volume dtype. |
| FR5 | Positions outside the volume produce 0. A valid plane entirely outside the volume gives an all-zero frame. |
| FR6 | Sampling modes: nearest neighbor (default), trilinear. |
| FR7 | Backends: `reference`, `numpy`, `numba`, `auto` (numba if importable, else numpy). All produce identical output for nearest sampling. |
| FR8 | Write an mp4 with exactly one frame per CSV row, in row order, constant frame size. |
| FR9 | CLI: `python Ex0.py file.nrrd file.csv file.mp4` plus options. Exit code 0 on success, 2 on invalid input, 1 on internal error. |
| FR10 | Optional `--world`: CSV points are in NRRD world coordinates and are converted to voxel space using `space origin` and `space directions`. |

## 4. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR1 | Performance: projection of a 500^3 volume at >= 100 fps (hard minimum 30 fps), measured as plane in, matrix out, excluding IO, normalization, encoding and JIT warm-up. |
| NFR2 | Correctness: the fast backends equal the reference backend exactly (nearest sampling). |
| NFR3 | Robustness: invalid input never crashes with a traceback; it produces a one-line error and exit code 2. |
| NFR4 | Maintainability: each backend and sampler sits behind an interface; core functions carry English pseudo-code in their docstrings. |
| NFR5 | Portability: runs on Python 3.10+; works without Numba (slower NumPy fallback). |
| NFR6 | Determinism: same inputs produce the same output on every run and backend. |

## 5. Design decisions

The assignment pseudo-code is under-specified. These are the resolutions.

| # | Issue in the assignment | Decision |
|---|---|---|
| D1 | Coordinate space of the CSV points | Voxel indices by default. `--world` converts from NRRD world space. |
| D2 | `M[p]` with a float `p` | Nearest neighbor by default. Trilinear selectable. |
| D3 | `a1` is incremented before first use, so the apex is never sampled and the last row overshoots the base | `a1 = A1 + y*v1` for `y = 0..floor(Dab)`. The image covers apex through base. Deviation from the assignment. |
| D4 | Output size `Dab*Dbc` is not an integer, and `x1 <= d12` reaches `x = Dbc` | `H = floor(Dab) + 1`, `W = floor(Dbc) + 1`. |
| D5 | `x = Dbc/2 - d12/2 + x1` is a float | `x = floor(off + x1 + 0.5)`. Consecutive `x1` map to consecutive `x`, so rows have no holes. |
| D6 | `E[x][y]` | Stored as `E[y][x]` (row-major image). |
| D7 | `p = p + dp` accumulates floating-point error | Closed form `p = a1 + x1*dp` in float64 in every backend. Deviation from the assignment, needed for exact cross-backend equality. |
| D8 | Rounding rule | `floor(v + 0.5)` everywhere. Not banker's rounding. |
| D9 | Isosceles check with floats | Relative tolerance 1e-3 on `|Dab - Dac| / max(Dab, Dac)`. |
| D10 | Trapezoid point order | `a', b, a'', c`. Rail 1 is `a'->b`, rail 2 is `a''->c`. `W = floor(max(|a'a''|, |bc|)) + 1`. A triangle is a trapezoid with `a' = a''`. |
| D11 | Degenerate input (a=b, collinear, Dab < 1, NaN, inf) | Validation error. |
| D12 | uint8/16/32 data vs 8-bit video | One global intensity window per volume (1st to 99th percentile), computed once, so brightness is consistent across frames. |
| D13 | Frame size varies per plane, video size must be fixed | Canvas = max `H` and `W` over all planes, rounded up to even numbers (H.264). Frames are centered and zero-padded. |
| D14 | NRRD axis order | pynrrd with `index_order='F'` gives `[x][y][z]`. The array is made C-contiguous once. |
| D15 | Anisotropic voxel spacing | Ignored in voxel space. Only `--world` takes spacing into account. |
| D16 | Meaning of "fps" | Projection only, as in NFR1. |
| D17 | Real data is often signed (MRHead is int16) but the assignment specifies unsigned values | Signed 8/16/32-bit data is cast to the unsigned type of the same width. If it has negative values (CT), the minimum is subtracted first and kept as `value_offset`. Floats and 64-bit types are rejected. |
| D18 | Both rails must advance in the same number of steps when the plane is slightly non-isosceles (within tolerance or `--force`) | `v1 = (B1-A1)/L` and `v2 = (B2-A2)/L` with `L = |B1-A1|`. For an exact isosceles plane both are unit vectors. |

## 6. Assumptions to confirm with the lecturer
- A1. CSV points are voxel indices (D1).
- A2. One CSV row is one plane and one frame.
- A3. Nearest neighbor satisfies "`v = M[p]`".

## 7. Early feasibility result
A throwaway Numba spike on a random 500^3 uint8 volume with oblique planes (560x501 frames) measured a median of 0.47 ms per frame (about 2100 fps) on an i7-13620H with 16 threads. Real figures will come from `scripts/bench.py`. An H.264 mp4 was written successfully via imageio-ffmpeg.

## 8. Traceability
Each FR and NFR maps to tests in `docs/TESTING.md` (Phase 7).
