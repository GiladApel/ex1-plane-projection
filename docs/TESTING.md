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
| Equivalence | NumPy (single and forced multi-chunk) and Numba equal the reference exactly: Hypothesis over random and axis-aligned planes, all dtypes, both samplers; half-integer coordinates that lock the rounding rule; thread-count independence (1, 2, 3, 7); backend selection and fallback when Numba is missing | `test_backend_equivalence.py` |
| Rendering | Intensity window, canvas centering and even sizing, video writer checks | `test_render.py` |
| End to end | CLI on synthetic data: one frame per row, constant even frame size, black frame for an outside plane, both backends give identical decoded video, trilinear, trapezoids, the lecturer's example sheet (counted layout, mixed shapes, one non-isosceles triangle that gives a warning), point-per-row CSV with one or several planes, `--world`, `--force`, bad rows that become error frames while the run goes on, whole-file errors that stop it, a failing projection (exit code 1), error exit codes, output directory check, subprocess run of `Ex0.py` | `test_cli.py` |
| Performance | 500^3 uint8, 40 random oblique planes, best of 3 medians. Absolute floors: Numba nearest 500 fps, Numba trilinear 250, NumPy nearest 100 (about half of the measured speed). Portable: Numba on all threads at least 3x its 1-thread speed (1.5x on 4-thread machines), Numba at least 1.4x NumPy (measured 4x on the reference machine, 1.9x on CI runners). End to end: the demo on 180 frames of 446x446 at 100+ fps projection and 30+ fps including encoding. Run with `pytest -m slow` | `test_performance.py`, `scripts/bench.py`, `scripts/bench_matrix.py` |

249 tests: 243 in the default run and 6 slow ones (2 of them portable, also run in CI).

## Failure reports

A failing test should say where the fault is, not only that there is one. Two parts of `tests/conftest.py` do this:
- `assert_images_equal(actual, expected)` replaces bare `np.array_equal` checks of images. On failure it gives the number of differing pixels, the box that contains all of them (one row, one column edge, or scattered), and the first five with both values, for example `60 of 60 pixels differ, within [0..9, 0..5]: (0, 0): got 0, expected 125; (0, 1): got 125, expected 33; ...`, which shows a one-column shift at a glance.
- After a failing run, a section "failures by location" lists every failure as `test file:line -> src/projection file:line`, the second being the deepest line of the package that was running when an exception was raised, and the first line of the message.

Both were checked by injecting faults into a copy of the code: a centering error was reported as a one-column shift in every pixel, and an exception in the sampler was traced to `src/projection/sampler.py:22`.

## Demo testing

The assignment's demo is `python Ex0.py file.nrrd file.csv file.mp4`. It is covered at three levels:
- in process, through `projection.cli.main`, for speed and to capture messages and exit codes;
- as a subprocess of the real script, in a different working directory, to prove it needs no installation;
- manually on real data (MRHead, CTChest) to check the picture itself; the results are in [EXPERIMENT.md](EXPERIMENT.md).

Video content is checked by decoding the mp4 and comparing frame counts, sizes and brightness. Exact pixel equality is only asserted between backends, because H.264 is lossy.

## Traceability

The full matrix, from each requirement (FR1.1 to NFR6) to the module that implements it and the tests that verify it, is in [SRS.md section 9](SRS.md#9-traceability). Every functional requirement has at least one automated test; NFR4 (maintainability) is checked by `ruff`, `mypy` and review, and NFR5 (portability) by the CI matrix.

## Continuous integration

`.github/workflows/ci.yml` runs the default test suite on Python 3.10 to 3.14, each with and without Numba installed. The "without Numba" runs prove the fallback path; the equivalence tests then skip the Numba parts automatically. A separate job runs `ruff check .` (style, bugbear and simplification rules) and `mypy` (type check of `src/`); both are clean and are configured in `pyproject.toml`.

## Coverage and its limits

Statement coverage of the package is 90 percent (`pytest --cov=projection`). Two caveats:
- `kernels_numba.py` shows 11 percent because compiled code is invisible to the coverage tool. It is verified by the equivalence tests instead, which compare its output with the reference.
- The remaining uncovered lines are error branches that need unusual files (for example a singular direction matrix).

## Not tested

- Visual quality of the video, beyond brightness checks. This is judged by eye (EXPERIMENT.md).
- Absolute speed on other hardware. The absolute floors were set on one machine (on mains power; on battery the laptop throttles and they can fail) and are not run in CI, because shared CI runners are too noisy. CI runs only the portable ratio tests, which catch lost parallelism or a slowed-down Numba kernel on any machine. Verified by injecting regressions: removing `parallel=True` from the Numba kernel fails both portable tests, and a 30 ms delay in the intensity window fails the end-to-end test.
- Extreme plane coordinates (beyond about 1e9); the tests use planes within a few volume sizes of the volume.

## Running

```
pytest                 # 243 tests, about 15 seconds
pytest -m slow         # all performance tests, about 20 seconds and about 150 MB of memory (run on mains power)
pytest -m portable     # only the machine-independent speed tests, as in CI
pytest --cov=projection --cov-report=term-missing
ruff check .           # lint
mypy                   # type check
python scripts/bench.py --size 500 --backend numba numpy --interp nearest trilinear
```
