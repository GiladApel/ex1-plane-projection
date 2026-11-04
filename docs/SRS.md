# Software Requirements Specification

Ex1: plane projection of a 3D volume. Introduction to Software Engineering, Ariel University.

## 1. Introduction

### 1.1 Purpose
This document states what the program must do, how well, and how each requirement is verified. It is written for the lecturer and graders, who run and assess the program, and for developers, who change it.

### 1.2 Scope
The program reads a 3D volume and a list of planes and writes a video with one frame per plane, each frame being the image of the volume on that plane. It is a command-line program (`python Ex0.py file.nrrd file.csv file.mp4`) built on a Python package (`src/projection/`) that can also be used as a library.

| In scope | Out of scope |
|---|---|
| NRRD volumes of 8, 16 or 32-bit integers, each dimension below 1024 | Other volume formats, floating-point or 64-bit data, multi-channel data |
| Isosceles triangles and trapezoids from a CSV file, in voxel or world coordinates | Other shapes; a graphical interface |
| Nearest-voxel sampling (default) and trilinear sampling (option) | Volume rendering, other interpolation methods |
| An H.264 mp4 video with one frame per plane | Other output formats; tissue-specific brightness windows |
| The speed target, a data set, tests, an experiment and these documents | Non-uniform voxel spacing in voxel coordinates |

### 1.3 Definitions
- **Volume `M`**: 3D array indexed `M[x][y][z]`, values uint8/16/32.
- **Plane**: 3 points (isosceles triangle `a, b, c`, apex `a`) or 4 points (isosceles trapezoid `a', b, a'', c`, legs `a' -> b` and `a'' -> c`).
- **Rails**: the two edges the algorithm walks: rail 1 `A1 -> B1`, rail 2 `A2 -> B2` (for a triangle both start at the apex).
- **Image `E`**: the 2D matrix computed for one plane, `H` rows by `W` columns, in the volume's value type.
- **Frame**: one image of the video: `E` mapped to 8-bit gray and centered on a canvas of fixed size.
- **Voxel coordinates**: positions given as array indices. **World coordinates**: positions in the NRRD file's physical space (`space origin`, `space directions`).
- **Counted layout**: the CSV format of the lecturer's example sheet: the number of points, then their coordinates.
- **Warning**: a `warning:` line on standard error; the run continues. **Error**: an `error:` line on standard error; the run stops before any output is written.
- **Error frame**: the frame written for a plane that could not be built or projected: dark gray with a white border and a faint X, and the text "ERROR", the frame number, the CSV line and the cause, so it is never confused with the black frame of a plane outside the volume and can be read without the terminal output.
- **fps**: planes projected per second, counting the projection only (D16).
- **Backend**: one of three interchangeable implementations of the projection: reference, NumPy, Numba.

### 1.4 References
- The assignment text (Ex1, Ariel University) and the lecture in which it was presented.
- The lecturer's [example planes sheet](https://docs.google.com/spreadsheets/d/1Tfy1yMYhAJjRBA8ZpRp_07pZXqGtxWhWoP02yZW1FQE/edit).
- [DEVIATIONS.md](DEVIATIONS.md): every difference from the assignment text, with its reason.
- [ALGORITHM.md](ALGORITHM.md), [DESIGN.md](DESIGN.md), [TESTING.md](TESTING.md), [EXPERIMENT.md](EXPERIMENT.md).
- The NRRD file format: <https://teem.sourceforge.net/nrrd/format.html>.

### 1.5 How requirements are written
Each requirement is one statement with an ID. Groups keep the IDs `FR1` to `FR10` and `NFR1` to `NFR6`, which other documents and the code use; single statements are numbered inside a group (`FR2.1`). Priorities:
- **Must**: stated by the assignment or needed for its demo to work.
- **Should**: not stated, but needed for the lecturer to run the program on his own files without trouble.
- **Could**: an addition that does not change the default behavior.

## 2. Overall description

### 2.1 Product perspective
A single pipeline: read the volume, read and validate every plane, then for each plane project, map to gray, center on the canvas and encode a frame. A problem with a whole file (volume, planes file, output folder) is found before the video file is opened, so it leaves no partial output. A problem with one plane does not stop the run: that plane becomes an error frame, and the summary lists it (D19). The projection is behind one interface (`Projector`) with three implementations; the default picks Numba when installed and NumPy otherwise.

### 2.2 Users
| User | Needs |
|---|---|
| The lecturer and graders | Run the demo on their own NRRD and CSV files; get a video, a speed figure, and clear messages when an input is wrong |
| Developers | Change or extend the program without breaking it: documented algorithm and design, a test suite that names the failing part |

### 2.3 Operating environment
Python 3.10 to 3.14 on Windows, Linux or macOS (developed on Windows 11; CI on Ubuntu). Required packages: NumPy, pynrrd, imageio with imageio-ffmpeg (which brings its own FFmpeg), and Pillow (for the text of error frames; imageio needs it anyway). Numba is optional and gives the full speed.

### 2.4 Constraints
- The demo command is fixed by the assignment: `python Ex0.py file.nrrd file.csv file.mp4`.
- Each volume dimension is below `2^10`.
- At least 30 planes per second on a 500^3 volume, about 100 expected.

### 2.5 Assumptions and dependencies
The open points of the assignment are listed in [section 6](#6-assumptions-to-confirm-with-the-lecturer). The program depends on pynrrd for reading NRRD and on FFmpeg (through imageio-ffmpeg) for writing H.264.

## 3. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| **FR1** | **Volume input** | |
| FR1.1 | The program shall read a 3D NRRD file of unsigned 8, 16 or 32-bit integers into a `Volume` indexed `[x][y][z]` in the order of the file's axes. | Must |
| FR1.2 | The program shall accept signed 8, 16 and 32-bit data, shifting it to the unsigned type of the same width when it has negative values (D17). | Should |
| FR1.3 | The program shall reject, with an error naming the file, a missing or unreadable file, data that is not 3D (including multi-component data), floating-point or 64-bit data, and any dimension outside 1 to 1023. | Must |
| **FR2** | **Planes file** | |
| FR2.1 | The program shall read the counted layout of the lecturer's example sheet: each row is the number of points (3 or 4) followed by their 9 or 12 coordinates, a triangle may leave the fourth point's cells empty, and triangles and trapezoids may be mixed. | Must |
| FR2.2 | The program shall also read rows of 9 numbers (triangle) or 12 numbers (trapezoid) without a point count, all rows of one file having the same count. | Could |
| FR2.3 | The program shall also read one point per row (3 numbers), grouping 3 rows into a triangle and 4 into a trapezoid; blank lines separate planes, and without blank lines a row count that is a multiple of 12 is rejected as ambiguous. | Could |
| FR2.4 | The program shall skip a first row that holds text (a header), lines starting with `#` (comments), blank lines, a UTF-8 byte order mark and empty cells at the end of a row. | Should |
| FR2.8 | The program shall choose the layout from the header when it starts with "Number of Points" (the lecturer's sheet), and otherwise from the layout that most rows fit, so that one bad row cannot change how the other rows are read. | Should |
| FR2.5 | The program shall report every unreadable row with the file name and line number. | Must |
| FR2.7 | A problem with the planes file as a whole (missing, unreadable, no rows, point rows that cannot be grouped into planes) shall be an error. | Must |
| FR2.6 | The program shall reject a file with no planes. | Must |
| **FR3** | **Plane validation** | |
| FR3.1 | The program shall reject a plane with a non-finite coordinate, an edge shorter than one voxel, collinear points, or (for a trapezoid) points not in one plane, a top side not parallel to the base, or legs that cross (top and base pointing opposite ways). These checks come before the isosceles check, so a plane is reported for its real fault. | Must |
| FR3.7 | The program shall reject a plane whose image would be larger than 4096 pixels on a side (D20). | Should |
| FR3.2 | The program shall project a plane whose legs differ by more than the relative tolerance 1e-3 as given, and print a warning naming its CSV line (D9). | Should |
| FR3.3 | With `--force`, the program shall not print the warning of FR3.2. | Could |
| FR3.4 | When a trapezoid fails the isosceles or parallel check, the message shall state the expected point order `a', b, a'', c` (D10). | Could |
| FR3.5 | The program shall check every plane before writing any output, and warn about each bad one with its line, its frame number and the cause. | Must |
| FR3.6 | A row or plane that cannot be read or fails FR3.1 shall not stop the run: its frame shall be an error frame (D19) showing "ERROR", the frame number, the CSV line and the cause, at the largest text size that fits the frame; what does not fit is left out, cause first. | Should |
| **FR4** | **Projection** | |
| FR4.1 | The program shall compute the image of each plane by the algorithm of [ALGORITHM.md](ALGORITHM.md), section 2: two rails, one row per unit step along them, samples one unit apart along each row, centered (D3 to D8). | Must |
| FR4.2 | A trapezoid shall be projected by the same procedure, with rails `a' -> b` and `a'' -> c` (D10). | Must |
| FR4.3 | The image shall have `floor(Dab) + 1` rows and `floor(max(|a'a''|, |bc|)) + 1` columns (D4). | Must |
| FR4.4 | The image shall have the value type of the volume. | Must |
| **FR5** | **Outside the volume** | |
| FR5.1 | A sample whose position falls outside the volume shall be 0. | Must |
| FR5.2 | A valid plane entirely outside the volume shall give an all-zero image, not an error. | Must |
| **FR6** | **Sampling** | |
| FR6.1 | By default a sample shall take the value of the nearest voxel, rounding `.5` up on each axis (D2, D8). | Must |
| FR6.2 | With `--interp trilinear`, a sample shall be the trilinear blend of the 8 surrounding voxels, rounded to the value type. | Could |
| **FR7** | **Backends** | |
| FR7.1 | The program shall offer the backends `reference`, `numpy` and `numba`, selected with `--backend`. | Could |
| FR7.2 | The default `auto` shall use Numba when it can be imported and NumPy otherwise, without an error. | Should |
| **FR8** | **Video output** | |
| FR8.1 | The program shall write an mp4 file (H.264, yuv420p) with exactly one frame per plane, in the order of the planes file, error frames included. | Must |
| FR8.2 | All frames shall have one size: the largest image height and width over the planes, each rounded up to an even number; each image shall be centered on a black canvas (D13). | Must |
| FR8.3 | Values shall be mapped to gray by one window for the whole video, from the 1st to the 99th percentile of the volume, leaving out background padding (D12). | Should |
| FR8.4 | The frame rate shall be set with `--fps` (default 15). | Could |
| **FR9** | **Command line** | |
| FR9.1 | The program shall run as `python Ex0.py file.nrrd file.csv file.mp4 [options]` from the repository, without installation. | Must |
| FR9.2 | The program shall exit with code 0 when it writes the video (error frames for bad planes included), 2 on a problem with a whole input (with a one-line `error:` message), and 1 on an internal error, including a projection that failed for one plane. | Should |
| FR9.3 | The program shall reject an output path whose folder does not exist, before reading the inputs. | Should |
| FR9.4 | After a run the program shall print the number of frames, the frame size, the backend, the projection fps and the end-to-end fps, and then every error frame with its CSV line and cause. | Should |
| **FR10** | **World coordinates** | |
| FR10.1 | With `--world`, the program shall read the CSV points as NRRD world coordinates and convert them to voxel coordinates with `space origin` and `space directions`. | Could |
| FR10.2 | With `--world`, a volume without spatial metadata shall be an error, before any plane is read. | Could |

## 4. Non-functional requirements

| ID | Requirement | Priority |
|---|---|---|
| **NFR1** | **Performance**, measured as planes projected per second (median), excluding file reading, gray mapping, encoding and JIT warm-up (D16) | |
| NFR1.1 | On a 500^3 uint8 volume with nearest sampling, the default backend shall project at least 100 planes per second on the reference machine (section 7), and never below 30. | Must |
| NFR1.2 | Without Numba, the NumPy backend shall reach 100 planes per second with nearest sampling on the reference machine. | Should |
| NFR1.3 | With Numba, trilinear sampling shall reach 100 planes per second on the reference machine. | Could |
| NFR1.4 | A loss of speed in the Numba backend (lost parallelism, a slower kernel) shall be detected automatically on every push. | Should |
| **NFR2** | **Correctness**: with nearest sampling, the NumPy and Numba backends shall produce images identical, pixel for pixel, to the reference backend. | Must |
| **NFR3** | **Robustness**: no input file shall make the program end with a Python traceback. A bad whole input gives a one-line error and exit code 2; a bad plane gives a warning and an error frame, and the run goes on (FR3.6, FR9.2). | Must |
| **NFR4** | **Maintainability**: each backend and sampler shall sit behind one interface; core functions shall carry English pseudo-code in their docstrings; the code shall pass `ruff` and `mypy`. | Should |
| NFR4.1 | Each projection engine shall release the resources it holds (threads, buffers) through `close()` or at the end of a `with` block, and shall stay usable after closing. | Should |
| **NFR5** | **Portability**: the program shall run on Python 3.10 to 3.14, with and without Numba installed. | Should |
| **NFR6** | **Determinism**: the same inputs shall give the same images on every run, backend and thread count. | Should |

## 5. Design decisions

The assignment pseudo-code is under-specified. These are the resolutions. Where a decision departs from the assignment text, [DEVIATIONS.md](DEVIATIONS.md) gives the assignment's version, the reason and the effect on the image.

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
| D9 | Isosceles check with floats, and what to do with a plane that fails it | Relative tolerance 1e-3 on `|Dab - Dac| / max(Dab, Dac)`. A plane outside the tolerance gets a warning and is projected as given (D18 handles unequal legs), instead of stopping the run: the lecturer's own example sheet has a triangle with legs 497.4 and 476.9. |
| D10 | Trapezoid point order | `a', b, a'', c`. Rail 1 is `a'->b`, rail 2 is `a''->c`. `W = floor(max(|a'a''|, |bc|)) + 1`. A triangle is a trapezoid with `a' = a''`. The order is not guessed: for a rectangle both orders are valid planes (the second one turned by 90 degrees), so reordering cannot be done safely. Points in another order fail the isosceles or parallel check, and the error names the expected order. |
| D11 | Degenerate input (a=b, collinear, Dab < 1, NaN, inf) | Validation error. |
| D12 | uint8/16/32 data vs 8-bit video | One global intensity window per volume (1st to 99th percentile), computed once, so brightness is consistent across frames. Voxels at the volume's minimum value are left out when they cover at least 5% of it (padding or air around the object, for example CT padding at -3024 HU), otherwise they would pull the 1st percentile down and wash out the image. If the remaining values give an empty range (a binary mask), the whole volume is used. |
| D13 | Frame size varies per plane, video size must be fixed | Canvas = max `H` and `W` over all planes, rounded up to even numbers (H.264). Frames are centered and zero-padded. |
| D14 | The assignment says the CSV is "a list of 3 or 4 points" and wants a frame per plane in list order, which allows several readings | All are supported (FR2). The counted layout is the lecturer's own format (his example sheet) and the primary one; it is recognized by its header ("Number of Points ...") or, without that header, by most of its rows having the shape `n` then `3n` numbers, so it cannot be confused with plane per row (9 or 12 numbers). Deciding by the majority of rows, not by the first row, means a first row that lost a number is reported as one bad row instead of changing how every other row is read. Lines starting with `#` are comments. Point per row groups 3 or 4 rows into a plane; blank lines separate planes and make the grouping explicit, so a file can mix triangles and trapezoids. A contiguous block whose row count fits both groupings (a multiple of 12) or neither is rejected with a clear message rather than guessed. |
| D15 | Anisotropic voxel spacing | Ignored in voxel space. Only `--world` takes spacing into account. |
| D16 | Meaning of "fps" | Projection only, as in NFR1. |
| D17 | Real data is often signed (MRHead is int16) but the assignment specifies unsigned values | Signed 8/16/32-bit data is cast to the unsigned type of the same width. If it has negative values (CT), the minimum is subtracted first and kept as `value_offset`. Floats and 64-bit types are rejected. |
| D20 | A plane can be arbitrarily large, but H.264 encoders and players reject frames beyond about 4096 pixels on a side, and one large plane would enlarge every frame of the video (D13) | A plane whose image would exceed 4096 pixels on a side is a bad plane (an error frame, D19). Volumes are below 1024 voxels on a side, so such a plane lies mostly outside the volume anyway. |
| D19 | What to do with one bad plane among many (a typo in one row, a degenerate triangle) | The run goes on: the plane gets a warning with its line, frame number and cause, its frame is an error frame, and the summary lists all of them. The frame count still equals the number of planes, so frame `n` is always plane `n`. A problem with a whole file still stops the run before any output, since no frame could be trusted. A projection that raises (a bug, not bad input) also gives an error frame, but the exit code becomes 1 so it is not missed. |
| D18 | Both rails must advance in the same number of steps when the plane is not exactly isosceles (within tolerance, or accepted with a warning) | `v1 = (B1-A1)/L` and `v2 = (B2-A2)/L` with `L = |B1-A1|`. For an exact isosceles plane both are unit vectors. |

## 6. Assumptions to confirm with the lecturer
- A1. CSV points are voxel indices (D1).
- A2. Resolved: the lecturer's example sheet uses the counted layout (FR2.1). The plane-per-row and point-per-row layouts stay supported.
- A3. Nearest neighbor satisfies "`v = M[p]`". This is the one open point that changes the image noticeably: reading `M[p]` by truncation instead would move about half of the pixels ([DEVIATIONS.md](DEVIATIONS.md#how-far-the-deviations-move-the-output)).

## 7. Measured performance
Full matrix (dtype, sampling, plane orientation, thread scaling): [experiment/benchmark.md](experiment/benchmark.md), produced by `python scripts/bench_matrix.py`. Summary for a random 500^3 uint8 volume, 500x500 frames, oblique planes through the center, on the reference machine (Intel i7-13620H, 16 threads, on mains power), Python 3.14:

| Backend | Sampling | Median fps | Slowest frame fps |
|---|---|---|---|
| numba | nearest | about 840 (1950 for axis-aligned planes) | about 330 |
| numba | trilinear | about 410 | about 240 |
| numpy | nearest | about 150 (140 to 240 across dtypes and plane orientations) | about 90 |
| numpy | trilinear | about 22 | 17 |

- Numba meets the 100 fps target by a factor of 8 or more and stays above it even on a single thread (169 fps).
- NumPy nearest meets the 100 fps target on median on this machine, with its slowest frames at 50 to 160 fps. Run-to-run variation is large (about 20 percent), so values are approximate.
- NumPy trilinear is correct but does not reach the target (16 to 35 fps); NFR1.3 only requires trilinear on Numba.
- NumPy history: a first version gathered only the kept samples with fancy indexing and ran at 27 fps; computing the dense grid with cached buffers gave about 70 fps; splitting the frame into row chunks on 4 threads gave about 150 fps. Without chunking the result is identical, only slower; chunking never changes the output (tested for 1, 2, 3 and 7 threads).

## 8. Use cases

| ID | Actor | Main flow | Alternatives |
|---|---|---|---|
| UC-01 Make a video | Lecturer or grader | Runs `python Ex0.py file.nrrd file.csv file.mp4`. The program reads and checks the volume and every plane, projects each plane, writes one frame per plane, prints the summary (FR9.4) and exits with 0. | Bad volume, bad planes file or missing output folder: one `error:` line, no video, exit 2 (FR1.3, FR2.7, FR9.3). A bad row or plane: a `warning:` line naming its line and frame, an error frame, and the run goes on; the summary lists the error frames (FR3.6, FR9.4). A non-isosceles plane: a `warning:` line naming its row, and the plane is drawn (FR3.2). A plane outside the volume: a black frame (FR5.2). |
| UC-02 Use the lecturer's planes sheet | Lecturer | Exports his sheet to CSV (header, point count, empty cells for triangles) and runs UC-01 with it. | Rows of other layouts are accepted too (FR2.2, FR2.3). |
| UC-03 Give planes in world coordinates | User with a scan in millimeters | Adds `--world`; points are converted with the file's origin and directions (FR10.1). | The file has no spatial metadata: error, exit 2 (FR10.2). |
| UC-04 Run without Numba | User on a machine without Numba | Runs UC-01; `auto` falls back to NumPy without an error (FR7.2). | Asks for `--backend numba`: error naming the missing package, exit 2. |
| UC-05 Choose quality or speed | User | Adds `--interp trilinear` for smoother images, or `--backend` to pick an implementation (FR6.2, FR7.1). | Trilinear on NumPy: a warning that it is slow. |
| UC-06 Make planes for an experiment | Developer | Runs `scripts/make_planes.py sweep` or `rotate` to write a CSV for a volume, then UC-01. | |
| UC-07 Check a change | Developer | Runs `pytest` (correctness, about 15 s), `pytest -m slow` (speed) and `ruff check .` / `mypy`; CI runs the same on every push (NFR1.4, NFR5). | A failing test names the backend, plane or row that differs. |

## 9. Traceability

Each requirement, where it is implemented, and the tests that verify it. Test files are in `tests/`; `test_x::name` is a test function, and parametrized tests run once per case.

| Requirement | Design (module, decision) | Verified by |
|---|---|---|
| FR1.1 | `io_nrrd.load_nrrd`, `volume.Volume` | `test_volume_io::test_roundtrip_keeps_axis_order_and_dtype`, `::test_axis_order_distinguishes_axes`, `::test_loads_mrhead` |
| FR1.2 | `io_nrrd._to_unsigned`, D17 | `test_volume_io::test_signed_non_negative_is_cast_without_offset`, `::test_signed_negative_is_shifted` |
| FR1.3 | `io_nrrd.load_nrrd`, `volume.Volume.__post_init__` | `test_volume_io::test_rejects_4d`, `::test_rejects_float`, `::test_rejects_missing_file`, `::test_rejects_garbage_file`, `::test_rejects_oversize_dimension`, `::test_volume_rejects_bad_dtype_and_rank`; `test_cli::test_non_3d_nrrd_is_rejected`, `::test_missing_nrrd_is_reported` |
| FR2.1 | `plane_csv._is_counted`, `._parse_row`, D14 | `test_plane_csv::test_counted_layout_*` (6 tests); `test_cli::test_lecturer_csv_format` |
| FR2.2 | `plane_csv._planes_from_plane_rows` | `test_plane_csv::test_triangles`, `::test_trapezoids`, `::test_mixed_point_counts_rejected`, `::test_wrong_column_count_reports_line_number` |
| FR2.3 | `plane_csv._group_point_rows` | `test_plane_csv::test_point_per_row_*` (14 tests); `test_cli::test_point_per_row_csv_*` |
| FR2.4 | `plane_csv.read_planes`, `._is_header`, `._drop_trailing_empty` | `test_plane_csv::test_header_is_skipped`, `::test_utf8_bom`, `::test_blank_lines_and_spaces_are_ignored`, `::test_trailing_commas_are_ignored`, `::test_empty_cell_inside_a_row_is_rejected`, `::test_comment_lines_are_skipped`, `::test_comments_between_point_blocks_separate_planes` |
| FR2.8 | `plane_csv._layout`, D14 | `test_plane_csv::test_the_lecturers_header_selects_the_counted_layout`, `::test_the_layout_most_rows_fit_wins`, `::test_a_bad_first_row_does_not_change_how_the_others_are_read` |
| FR2.5 | `plane_csv.read_planes`, `PlaneRow` (line and cause per plane) | `test_plane_csv::test_bad_value_reports_line_number`, `::test_invalid_plane_reports_line_number`, `::test_point_per_row_bad_row_reports_line`; `test_cli::test_bad_rows_become_error_frames_and_the_run_goes_on` |
| FR2.6 | `plane_csv.parse_planes` | `test_plane_csv::test_header_only_file_rejected`, `::test_empty_file_rejected`, `::test_missing_file` |
| FR3.1 | `geometry.Triangle`, `geometry.Trapezoid`, D11 | `test_geometry::test_triangle_rejects_collinear`, `::test_triangle_rejects_coincident_points`, `::test_rejects_edge_shorter_than_one_voxel`, `::test_rejects_non_finite`, `::test_trapezoid_rejects_non_coplanar`, `::test_trapezoid_rejects_non_parallel_sides`, `::test_trapezoid_rejects_fully_degenerate`, `::test_trapezoid_with_crossing_legs_is_rejected`, `::test_structural_errors_come_before_the_isosceles_check`; `test_plane_csv::test_warning_does_not_hide_other_plane_errors` |
| FR3.7 | `geometry.Plane`, `MAX_IMAGE_SIDE`, D20 | `test_geometry::test_image_larger_than_the_video_limit_is_rejected` |
| FR3.2 | `errors.NotIsoscelesError`, `plane_csv._build_plane`, `cli._warn`, D9, D18 | `test_plane_csv::test_non_isosceles_plane_is_kept_with_a_warning`; `test_cli::test_non_isosceles_plane_is_projected_with_a_warning`, `::test_lecturer_csv_format`, `::test_warnings_come_in_plane_order`; `test_plane_csv::test_an_accepted_non_isosceles_plane_carries_its_warning`; `test_geometry::test_triangle_rejects_non_isosceles`, `::test_triangle_within_tolerance_is_accepted` |
| FR3.3 | `cli` (`--force`) | `test_plane_csv::test_force_silences_the_warning`, `::test_force_skips_isosceles_check`; `test_cli::test_non_isosceles_plane_is_projected_with_a_warning` |
| FR3.4 | `geometry.TRAPEZOID_ORDER`, D10 | `test_geometry::test_trapezoid_in_wrong_point_order_names_the_expected_order`, `::test_triangle_isosceles_error_has_no_order_hint` |
| FR2.7 | `plane_csv.read_planes`, `._group_point_rows` | `test_plane_csv::test_point_per_row_wrong_row_count`, `::test_empty_file_rejected`; `test_cli::test_problems_with_the_whole_file_still_stop_the_run` |
| FR3.5 | `cli.run` (read all planes before opening the writer) | `test_cli::test_bad_rows_become_error_frames_and_the_run_goes_on` |
| FR3.6 | `plane_csv.read_planes`, `render.FrameComposer.error_frame` (Pillow's bundled font), `cli.run`, D19 | `test_cli::test_bad_rows_become_error_frames_and_the_run_goes_on`, `::test_a_file_of_bad_rows_still_gives_one_frame_per_row`, `::test_a_failing_projection_gives_an_error_frame_and_exit_code_1`; `test_plane_csv::test_read_planes_*` (2 tests); `test_render::test_error_frame_on_a_tiny_canvas_keeps_border_and_x`, `::test_error_frame_shows_text_that_depends_on_frame_line_and_cause`, `::test_long_cause_is_wrapped_inside_the_border`, `::test_canvas_for_no_planes` |
| FR4.1 | `projector.ReferenceProjector`, ALGORITHM section 2, D3 to D8 | `test_reference_projector::test_axis_aligned_plane_equals_numpy_slice`, `::test_triangle_on_gradient_has_constant_value_and_exact_pixel_count`, `::test_triangle_apex_row_is_one_centered_pixel`, `::test_oblique_plane_on_gradient_matches_closed_form`, `::test_non_integer_edges_have_no_holes_in_rows` |
| FR4.2 | `geometry._build_rails`, D10 | `test_geometry::test_trapezoid_rails_and_size`, `::test_trapezoid_with_equal_top_is_triangle`; `test_cli::test_trapezoid_rows` |
| FR4.3 | `geometry._build_rails`, D4 | `test_geometry::test_isosceles_triangle_size`, `::test_trapezoid_rails_and_size` |
| FR4.4 | `Projector._prepare_out` | `test_reference_projector::test_axis_aligned_plane_equals_numpy_slice` (3 dtypes), `::test_out_buffer_is_cleared_reused_and_validated` |
| FR5.1 | `sampler.sample_nearest`, `sampler.sample_trilinear` | `test_reference_projector::test_plane_partially_outside_is_zero_only_outside`, `::test_sample_nearest_rounds_half_up_and_clips` |
| FR5.2 | all backends | `test_reference_projector::test_plane_fully_outside_volume_is_all_zero`; `test_backend_equivalence::test_plane_outside_volume_is_zero`; `test_cli::test_demo_writes_one_frame_per_row_with_constant_size` |
| FR6.1 | `sampler.sample_nearest`, D2, D8 | `test_reference_projector::test_sample_nearest_rounds_half_up_and_clips`; `test_backend_equivalence::test_half_integer_coordinates_lock_the_rounding_rule` |
| FR6.2 | `sampler.sample_trilinear`, `kernels_numba.project_trilinear` | `test_reference_projector::test_trilinear_*` (3 tests), `::test_sample_trilinear_weights`; `test_cli::test_trilinear_runs` |
| FR7.1 | `projector.make_projector`, `BACKENDS` | `test_backend_equivalence::test_make_projector_selection`; `test_cli::test_backends_produce_identical_videos` |
| FR7.2 | `projector.make_projector` | `test_backend_equivalence::test_make_projector_falls_back_without_numba`, `::test_package_import_exposes_the_api_without_loading_numba`; CI jobs without Numba |
| FR8.1 | `video.Mp4Writer`, `cli.run` | `test_cli::test_demo_writes_one_frame_per_row_with_constant_size`, `::test_script_runs_as_documented` |
| FR8.2 | `render.FrameComposer`, D13 | `test_render::test_canvas_is_largest_plane_rounded_up_to_even`, `::test_compose_centers_and_clears_between_calls`, `::test_compose_rejects_oversize_image`, `::test_writer_rejects_odd_size_and_bad_frames` |
| FR8.3 | `render.IntensityWindow`, D12 | `test_render::test_window_*` (8 tests) |
| FR8.4 | `cli` (`--fps`) | `test_cli::test_bad_arguments_exit_with_code_2` |
| FR9.1 | `Ex0.py`, `cli.main` | `test_cli::test_script_runs_as_documented` (subprocess, other working directory) |
| FR9.2 | `cli.main`, `errors.InvalidInputError` | `test_cli::test_bad_arguments_exit_with_code_2`, `::test_missing_nrrd_is_reported`, `::test_problems_with_the_whole_file_still_stop_the_run`, `::test_a_failing_projection_gives_an_error_frame_and_exit_code_1` |
| FR9.3 | `cli.run` | `test_cli::test_missing_output_directory` |
| FR9.4 | `cli.run` | `test_cli::test_demo_writes_one_frame_per_row_with_constant_size`; `test_performance::test_demo_end_to_end_speed` |
| FR10.1 | `volume.Volume.world_to_voxel`, D1, D15 | `test_volume_io::test_world_to_voxel_roundtrip`; `test_cli::test_world_coordinates_are_converted`; `test_plane_csv::test_to_voxel_is_applied` |
| FR10.2 | `cli.run`, `volume.Volume.world_to_voxel` | `test_volume_io::test_world_to_voxel_requires_metadata`; `test_cli::test_world_coordinates_without_metadata_stop_the_run` |
| NFR1.1 | `kernels_numba`, `projector.NumbaProjector` | `test_performance::test_projection_speed_on_500_cubed` (Numba, nearest); `scripts/bench.py`, `scripts/bench_matrix.py` |
| NFR1.2 | `projector.NumpyProjector` | `test_performance::test_projection_speed_on_500_cubed` (NumPy, nearest) |
| NFR1.3 | `kernels_numba.project_trilinear` | `test_performance::test_projection_speed_on_500_cubed` (Numba, trilinear) |
| NFR1.4 | CI job "speed ratios" | `test_performance::test_numba_scales_with_threads`, `::test_numba_is_much_faster_than_numpy` (marked `portable`, run in CI) |
| NFR2 | same arithmetic in all backends (ALGORITHM section 4) | `test_backend_equivalence::test_random_planes_match_reference` (Hypothesis), `::test_triangle_matches_reference`, `::test_half_integer_coordinates_lock_the_rounding_rule` |
| NFR3 | `cli.main` (catches `InvalidInputError`), `cli.run` (error frames) | the error tests of `test_cli` (exit code and one `error:` line, or warnings and error frames) |
| NFR4 | `Projector` interface, `Interpolation` | `ruff check .` and `mypy` in CI; review of [DESIGN.md](DESIGN.md) |
| NFR4.1 | `Projector.close`, `__enter__`, `__exit__`; `NumpyProjector.close`; `cli.run` | `test_backend_equivalence::test_every_backend_is_a_context_manager`, `::test_closing_the_numpy_engine_stops_its_threads_and_it_stays_usable` |
| NFR5 | `pyproject.toml` (Numba optional) | CI matrix: Python 3.10 to 3.14, each with and without Numba |
| NFR6 | closed-form positions (D7), fixed summation order | `test_backend_equivalence::test_numpy_result_does_not_depend_on_thread_count`; seeded synthetic data |
