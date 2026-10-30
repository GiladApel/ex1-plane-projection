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
| FR2 | Parse a CSV in one of two layouts. Plane per row: each row is one plane (9 numbers = triangle, 12 numbers = trapezoid), all rows with the same count; one frame per row. Point per row: rows of 3 numbers (x, y, z), 3 rows per triangle and 4 per trapezoid; one frame per plane. Planes are separated by blank lines; without blank lines the row count decides (multiple of 3: triangles, multiple of 4: trapezoids, multiple of 12: rejected as ambiguous). The layout is detected from the number of columns in the first row. Optional header row; empty cells at the end of a row (trailing commas) are ignored. Errors report the line number. |
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
| NFR1 | Performance: with nearest sampling, projection of a 500^3 volume at >= 100 fps median on both the Numba and the NumPy backend (hard minimum 30 fps), measured on the 16-thread reference machine; the NumPy backend uses up to 4 threads, so machines with fewer cores will see less (section 7) as plane in, matrix out, excluding IO, normalization, encoding and JIT warm-up. Trilinear sampling is optional and only held to >= 100 fps on Numba. |
| NFR2 | Correctness: the fast backends equal the reference backend exactly (nearest sampling). |
| NFR3 | Robustness: invalid input never crashes with a traceback; it produces a one-line error and exit code 2. |
| NFR4 | Maintainability: each backend and sampler sits behind an interface; core functions carry English pseudo-code in their docstrings. |
| NFR5 | Portability: runs on Python 3.10 to 3.14 (CI passes on every version in that range, with and without Numba); works without Numba, using the slower NumPy fallback. |
| NFR6 | Determinism: same inputs produce the same output on every run and backend. |

## 5. Design decisions

The assignment pseudo-code is under-specified. These are the resolutions.

| # | Issue in the assignment | Decision |
|---|---|---|
| D1 | Coordinate space of the CSV points | Voxel indices by default. `--world` converts from NRRD world space. |
| D2 | `M[p]` with a float `p` | Nearest neighbor by default. Trilinear selectable. |
| D3 | `a1` is incremented before first use, so the apex is never sampled and the last row overshoots the base | `a1 = A1 + y*v1` for `y = 0..floor(Dab)`. The image covers apex through base. Deviation from the assignment. |
| D4 | Output size `Dab*Dbc` is not an integer, and `x1 <= d12` reaches `x = Dbc` | `H = floor(Dab) + 1`, `W = floor(Dbc) + 1`. |
| D5 | `x = Dbc/2 - d12/2 + x1` is a float, and `Dbc` is the assignment's (non-integer) width, not the integer number of columns we allocate | `off = (W-1)/2 - d12/2`, `x = floor(off + x1 + 0.5)`. With `W = floor(Dbc)+1` the widest row maps `x1` to `x` exactly, consecutive `x1` map to consecutive `x` (no holes), and the center equals the assignment's `Dbc/2` when `Dbc` is an integer (less than one pixel apart otherwise). |
| D6 | `E[x][y]` | Stored as `E[y][x]` (row-major image). |
| D7 | `p = p + dp` accumulates floating-point error | Closed form `p = a1 + x1*dp` in float64 in every backend. Deviation from the assignment, needed for exact cross-backend equality. |
| D8 | Rounding rule | `floor(v + 0.5)` everywhere. Not banker's rounding. |
| D9 | Isosceles check with floats | Relative tolerance 1e-3 on `|Dab - Dac| / max(Dab, Dac)`. |
| D10 | Trapezoid point order | `a', b, a'', c`. Rail 1 is `a'->b`, rail 2 is `a''->c`. `W = floor(max(|a'a''|, |bc|)) + 1`. A triangle is a trapezoid with `a' = a''`. The order is not guessed: for a rectangle both orders are valid planes (the second one turned by 90 degrees), so reordering cannot be done safely. Points in another order fail the isosceles or parallel check, and the error names the expected order. |
| D11 | Degenerate input (a=b, collinear, Dab < 1, NaN, inf) | Validation error. |
| D12 | uint8/16/32 data vs 8-bit video | One global intensity window per volume (1st to 99th percentile), computed once, so brightness is consistent across frames. Voxels at the volume's minimum value are left out when they cover at least 5% of it (padding or air around the object, for example CT padding at -3024 HU), otherwise they would pull the 1st percentile down and wash out the image. If the remaining values give an empty range (a binary mask), the whole volume is used. |
| D13 | Frame size varies per plane, video size must be fixed | Canvas = max `H` and `W` over all planes, rounded up to even numbers (H.264). Frames are centered and zero-padded. |
| D14 | The assignment says the CSV is "a list of 3 or 4 points" and wants a frame per plane in list order, which allows two readings | Both are supported (FR2). Plane per row is the primary reading. Point per row groups 3 or 4 rows into a plane; blank lines separate planes and make the grouping explicit, so a file can mix triangles and trapezoids. A contiguous block whose row count fits both groupings (a multiple of 12) or neither is rejected with a clear message rather than guessed. |
| D15 | Anisotropic voxel spacing | Ignored in voxel space. Only `--world` takes spacing into account. |
| D16 | Meaning of "fps" | Projection only, as in NFR1. |
| D17 | Real data is often signed (MRHead is int16) but the assignment specifies unsigned values | Signed 8/16/32-bit data is cast to the unsigned type of the same width. If it has negative values (CT), the minimum is subtracted first and kept as `value_offset`. Floats and 64-bit types are rejected. |
| D18 | Both rails must advance in the same number of steps when the plane is slightly non-isosceles (within tolerance or `--force`) | `v1 = (B1-A1)/L` and `v2 = (B2-A2)/L` with `L = |B1-A1|`. For an exact isosceles plane both are unit vectors. |

## 6. Assumptions to confirm with the lecturer
- A1. CSV points are voxel indices (D1).
- A2. A CSV row with 9 or 12 numbers is one plane and one frame; a file of 3 or 4 rows with 3 numbers each is one plane. Both layouts are supported, so this assumption only matters if the lecturer uses a third layout.
- A3. Nearest neighbor satisfies "`v = M[p]`".

## 7. Measured performance
Full matrix (dtype, sampling, plane orientation, thread scaling): [experiment/benchmark.md](experiment/benchmark.md), produced by `python scripts/bench_matrix.py`. Summary for a random 500^3 uint8 volume, 500x500 frames, oblique planes through the center, Intel i7-13620H (16 threads), Python 3.14:

| Backend | Sampling | Median fps | Slowest frame fps |
|---|---|---|---|
| numba | nearest | about 840 (1950 for axis-aligned planes) | about 330 |
| numba | trilinear | about 410 | about 240 |
| numpy | nearest | about 150 (140 to 240 across dtypes and plane orientations) | about 90 |
| numpy | trilinear | about 22 | 17 |

- Numba meets the 100 fps target by a factor of 8 or more and stays above it even on a single thread (169 fps).
- NumPy nearest meets the 100 fps target on median on this machine, with its slowest frames at 50 to 160 fps. Run-to-run variation is large (about 20 percent), so values are approximate.
- NumPy trilinear is correct but does not reach the target (16 to 35 fps); NFR1 only requires trilinear on Numba.
- NumPy history: a first version gathered only the kept samples with fancy indexing and ran at 27 fps; computing the dense grid with cached buffers gave about 70 fps; splitting the frame into row chunks on 4 threads gave about 150 fps. Without chunking the result is identical, only slower; chunking never changes the output (tested for 1, 2, 3 and 7 threads).

## 8. Traceability
Each FR and NFR maps to tests in `docs/TESTING.md` (Phase 7).
