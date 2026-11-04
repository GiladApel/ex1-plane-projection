# Testing strategy

## Principles

1. **The reference implementation is the oracle.** `ReferenceProjector` is a literal translation of the algorithm. It is itself checked against answers computed independently (NumPy slices, closed-form formulas), and the NumPy and Numba backends are then checked for exact equality with it.
2. **Known-answer data.** Synthetic volumes (gradient, sphere, checkerboard, seeded noise) make the expected output computable without running the code under test.
3. **Edge cases are enumerated, then randomized.** Hand-written tests cover the cases that are easy to name (outside the volume, rounding on .5, degenerate planes). Hypothesis then searches the space of random planes, volumes and dtypes for what was not thought of.
4. **The demo is tested as a user runs it.** End-to-end tests call the CLI, and one test runs `python Ex0.py ...` as a subprocess.
5. **Performance is tested separately and marked slow**, so the normal run stays under 15 seconds. Speed tests that compare two measurements on the same machine are also marked `portable` and run in CI; absolute frame rates depend on the machine and are checked locally.

## Levels

| Level | What | Files |
|---|---|---|
| Unit: inputs | Plane validation (isosceles, collinear, degenerate, non-finite, trapezoid coplanarity), CSV parsing for both layouts with line-numbered errors, NRRD loading (dtypes, axis order, signed conversion, 4D rejection, garbage files), world-to-voxel conversion | `test_geometry.py`, `test_plane_csv.py`, `test_volume_io.py`, `test_synthetic.py` |
| Unit: plane generators | Sweep planes reproduce each slice exactly; rotation planes are valid, centered and contain the axis; CSV round trip; constant frame size | `test_plane_gen.py` |
| Correctness | Reference projector against independent answers: axis-aligned planes equal NumPy slices for all 6 axis permutations and 3 dtypes; exact pixel counts on a gradient; oblique plane against a closed form; apex row; plane fully and partly outside; no holes in rows; trilinear weights and uint32 overflow safety; `out` buffer handling | `test_reference_projector.py` |
| Known answers from geometry | Cuts through a cube: an axis-aligned cut equals the face, the cut through the center perpendicular to the diagonal is a regular hexagon of area 3√3/4·s², corner cuts are equilateral triangles of area √3/2·t²; cuts through a sphere at any orientation are disks of area π(r²−d²); all within one outline of pixels. Points exactly halfway between voxels along 3-4-5, 5-12-13, 8-15-17 and 20-21-29 directions, with integer row widths, compared on every backend with the voxel computed in exact fractions | `test_known_answers.py` |
| Equivalence | NumPy (single and forced multi-chunk) and Numba equal the reference exactly: Hypothesis over random and axis-aligned planes, all dtypes, both samplers; half-integer coordinates that lock the rounding rule; thread-count independence (1, 2, 3, 7); backend selection and fallback when Numba is missing | `test_backend_equivalence.py` |
| Real data | On the MRHead scan (256 x 256 x 130 MRI, downloaded and checksum-verified): voxel-grid planes along x, y and z equal the array slices on every backend and sampler; the 256 frames of the published sweep are the y slices; every frame of the published rotation, nearest and trilinear, matches a CRC-32 recorded in `tests/data/mrhead_golden.json` on every backend; the rotation planes written in world coordinates (anisotropic spacing, axes swapped and flipped) give the same images; `Ex0.py` on the rotation and the three committed MRHead videos match the exact frames within video-codec noise | `test_real_data.py` |
| Rendering | Intensity window, canvas centering and even sizing, video writer checks | `test_render.py` |
| End to end | CLI on synthetic data: one frame per row, constant even frame size, black frame for an outside plane, both backends give identical decoded video, trilinear, trapezoids, the lecturer's example sheet (counted layout, mixed shapes, one non-isosceles triangle that gives a warning), point-per-row CSV with one or several planes, `--world`, `--force`, bad rows that become error frames while the run goes on, whole-file errors that stop it, a failing projection (exit code 1), error exit codes, output directory check, subprocess run of `Ex0.py` | `test_cli.py` |
| Interactive demo | The step-by-step explanation of the "How it works" tab rebuilds every pixel of 80 random triangles and trapezoids exactly as the reference backend does, nearest and trilinear; both tabs render without an exception (Streamlit's `AppTest`), including a pixel chosen by a click, pixels in the first, middle and last rows of both shapes and samplers, and a chosen pixel that no longer exists after the plane shrinks | `test_demo.py` (needs `demo/requirements.txt`; CI job "demo") |
| Performance | 500^3 uint8, 40 random oblique planes, best of 3 medians. Absolute floors: Numba nearest 500 fps, Numba trilinear 250, NumPy nearest 100 (about half of the measured speed). Portable: Numba on all threads at least 3x its 1-thread speed (1.5x on 4-thread machines), Numba at least 1.4x NumPy (measured 4x on the reference machine, 1.9x on CI runners). End to end: the demo on 180 frames of 446x446 at 100+ fps projection and 30+ fps including encoding. Run with `pytest -m slow` | `test_performance.py`, `scripts/bench.py`, `scripts/bench_matrix.py` |

326 tests: 320 in the default run and 6 slow ones (2 of them portable, also run in CI). The 9 demo tests skip when Streamlit or Plotly is not installed; CI runs them in a job of their own.

## Failure reports

A failing test should say where the fault is, not only that there is one. Two parts of `tests/conftest.py` do this:
- `assert_images_equal(actual, expected)` replaces bare `np.array_equal` checks of images. On failure it gives the number of differing pixels, the box that contains all of them (one row, one column edge, or scattered), and the first five with both values, for example `60 of 60 pixels differ, within [0..9, 0..5]: (0, 0): got 0, expected 125; (0, 1): got 125, expected 33; ...`, which shows a one-column shift at a glance.
- After a failing run, a section "failures by location" lists every failure as `test file:line -> src/projection file:line`, the second being the deepest line of the package that was running when an exception was raised, and the first line of the message.

Both were checked by injecting faults into a copy of the code: a centering error was reported as a one-column shift in every pixel, and an exception in the sampler was traced to `src/projection/sampler.py:22`.

## Demo testing

The assignment's demo is `python Ex0.py file.nrrd file.csv file.mp4`. It is covered at three levels:
- in process, through `projection.cli.main`, for speed and to capture messages and exit codes;
- as a subprocess of the real script, in a different working directory, to prove it needs no installation;
- automatically on real data: `test_real_data.py` runs the published MRHead experiments and compares every frame with recorded values and the committed videos;
- by eye on real data (MRHead, CTChest) to judge the picture itself; the results are in [EXPERIMENT.md](EXPERIMENT.md).

Video content is checked by decoding the mp4 and comparing frame counts, sizes and brightness, and on MRHead by checking that every decoded frame is closer to its exact frame than to that frame moved by one pixel or to its neighboring frames. A fixed limit on the difference did not work: the codec noise is up to 1.6 gray levels on Windows and 2.05 on macOS, whose decoder converts colors slightly differently, while a one-pixel shift costs 2.8 or more. The comparison calibrates itself and keeps a margin of more than 2x even with simulated extra decoder noise of ±2 gray levels. Exact pixel equality is asserted between backends and against the recorded CRC-32 values, never on decoded video, because H.264 is lossy.

## Real data

`tests/test_real_data.py` needs `data/MRHead.nrrd` (`python scripts/download_data.py MRHead`, 6.6 MB). Without it the tests skip locally. CI downloads it once per cache and sets `REAL_DATA_REQUIRED=1`, which turns a missing file into a failure, so these tests cannot silently stop running there.

The recorded frames are CRC-32 values, one per frame, in `tests/data/mrhead_golden.json` (5 KB instead of 50 MB of images). A failure lists the frames that changed. After an intended change of the output, regenerate the file with `UPDATE_GOLDEN=1 pytest tests/test_real_data.py -k "golden and numpy"` and re-make the videos with the commands in EXPERIMENT.md. The golden values catch any change of a single pixel; the video comparison only catches visible ones (a frame moved by a pixel, frames out of order), so re-making the videos is a step to remember, not one the tests enforce.

These tests found a real fault: after a change of the rounding constants in `sampler.py`, the Numba backend kept the old values, because Numba's on-disk cache does not notice changed globals imported from another module. The kernels now take the constants as arguments.

## Traceability

The full matrix, from each requirement (FR1.1 to NFR6) to the module that implements it and the tests that verify it, is in [SRS.md section 9](SRS.md#9-traceability). Every functional requirement has at least one automated test; NFR4 (maintainability) is checked by `ruff`, `mypy` and review, and NFR5 (portability) by the CI matrix.

## Continuous integration

`.github/workflows/ci.yml` runs the default test suite, real-data tests included, on Python 3.10 to 3.14, each with and without Numba installed, on Ubuntu, and on Windows and macOS for 3.12 and 3.14. The "without Numba" runs prove the fallback path; the equivalence tests then skip the Numba parts automatically. A separate job runs `ruff check .` (style, bugbear and simplification rules) and `mypy` (type check of `src/`); both are clean and are configured in `pyproject.toml`.

## Coverage and its limits

Statement coverage of the package is 92 percent (`pytest --cov=projection`). Two caveats:
- `kernels_numba.py` shows 11 percent because compiled code is invisible to the coverage tool. It is verified by the equivalence tests instead, which compare its output with the reference.
- The remaining uncovered lines are error branches that need unusual files (for example a singular direction matrix).

## Not tested

- Visual quality of the video, beyond brightness checks. This is judged by eye (EXPERIMENT.md).
- Absolute speed on other hardware. The absolute floors were set on one machine (on mains power; on battery the laptop throttles and they can fail) and are not run in CI, because shared CI runners are too noisy. CI runs only the portable ratio tests, which catch lost parallelism or a slowed-down Numba kernel on any machine. Verified by injecting regressions: removing `parallel=True` from the Numba kernel fails both portable tests, and a 30 ms delay in the intensity window fails the end-to-end test.
- Extreme plane coordinates (beyond about 1e9); the tests use planes within a few volume sizes of the volume.

## Running

```
pytest                 # 320 tests, about 45 seconds (32 on MRHead, 9 on the demo)
pytest -m slow         # all performance tests, about 20 seconds and about 150 MB of memory (run on mains power)
pytest -m portable     # only the machine-independent speed tests, as in CI
pytest --cov=projection --cov-report=term-missing
ruff check .           # lint
mypy                   # type check
python scripts/bench.py --size 500 --backend numba numpy --interp nearest trilinear
```
