# Testing strategy

## Principles

1. **The reference implementation is the oracle.** `ReferenceProjector` is a literal translation of the algorithm. It is itself checked against answers computed independently (NumPy slices, closed-form formulas), and the NumPy and Numba backends are then checked for exact equality with it.
2. **Known-answer data.** Synthetic volumes (gradient, sphere, checkerboard, seeded noise) make the expected output computable without running the code under test.
3. **Edge cases are enumerated, then randomized.** Hand-written tests cover the cases that are easy to name (outside the volume, rounding on .5, degenerate planes). Hypothesis then searches the space of random planes, volumes and dtypes for what was not thought of.
4. **The demo is tested as a user runs it.** End-to-end tests call the CLI, and one test runs `python Ex0.py ...` as a subprocess.
5. **Performance is tested separately and marked slow**, so the normal run stays under 15 seconds.

## Levels

| Level | What | Files |
|---|---|---|
| Unit: inputs | Plane validation (isosceles, collinear, degenerate, non-finite, trapezoid coplanarity), CSV parsing for both layouts with line-numbered errors, NRRD loading (dtypes, axis order, signed conversion, 4D rejection, garbage files), world-to-voxel conversion | `test_geometry.py`, `test_plane_csv.py`, `test_volume_io.py`, `test_synthetic.py` |
| Unit: plane generators | Sweep planes reproduce each slice exactly; rotation planes are valid, centered and contain the axis; CSV round trip; constant frame size | `test_plane_gen.py` |
| Correctness | Reference projector against independent answers: axis-aligned planes equal NumPy slices for all 6 axis permutations and 3 dtypes; exact pixel counts on a gradient; oblique plane against a closed form; apex row; plane fully and partly outside; no holes in rows; trilinear weights and uint32 overflow safety; `out` buffer handling | `test_reference_projector.py` |
| Equivalence | NumPy (single and forced multi-chunk) and Numba equal the reference exactly: Hypothesis over random and axis-aligned planes, all dtypes, both samplers; half-integer coordinates that lock the rounding rule; thread-count independence (1, 2, 3, 7); backend selection and fallback when Numba is missing | `test_backend_equivalence.py` |
| Rendering | Intensity window, canvas centering and even sizing, video writer checks | `test_render.py` |
| End to end | CLI on synthetic data: one frame per row, constant even frame size, black frame for an outside plane, both backends give identical decoded video, trilinear, trapezoids, point-per-row CSV, `--world`, `--force`, error exit codes, output directory check, subprocess run of `Ex0.py` | `test_cli.py` |
| Performance | 500^3 uint8, 100 random oblique planes: Numba and NumPy each at least 100 fps median (run with `pytest -m slow`) | `test_performance.py`, `scripts/bench.py`, `scripts/bench_matrix.py` |

189 tests: 187 in the default run and 2 slow ones.

## Demo testing

The assignment's demo is `python Ex0.py file.nrrd file.csv file.mp4`. It is covered at three levels:
- in process, through `projection.cli.main`, for speed and to capture messages and exit codes;
- as a subprocess of the real script, in a different working directory, to prove it needs no installation;
- manually on real data (MRHead, CTChest) to check the picture itself; the results are in [EXPERIMENT.md](EXPERIMENT.md).

Video content is checked by decoding the mp4 and comparing frame counts, sizes and brightness. Exact pixel equality is only asserted between backends, because H.264 is lossy.

## Traceability

| Requirement | Verified by |
|---|---|
| FR1 NRRD loading | `test_volume_io.py`, `test_cli.py` (non-3D file) |
| FR2 CSV parsing | `test_plane_csv.py`, `test_cli.py` (bad CSV, point-per-row) |
| FR3 plane validation | `test_geometry.py`, `test_plane_csv.py` (line numbers), `test_cli.py` (`--force`) |
| FR4 projection | `test_reference_projector.py`, `test_backend_equivalence.py` |
| FR5 outside the volume | `test_reference_projector.py`, `test_backend_equivalence.py`, `test_cli.py` |
| FR6 sampling modes | `test_reference_projector.py`, `test_backend_equivalence.py` |
| FR7 backends | `test_backend_equivalence.py` |
| FR8 video | `test_cli.py`, `test_render.py` |
| FR9 CLI and exit codes | `test_cli.py` |
| FR10 world coordinates | `test_volume_io.py`, `test_cli.py` |
| NFR1 speed | `test_performance.py`, `scripts/bench.py` |
| NFR2 exact equality | `test_backend_equivalence.py` |
| NFR3 no tracebacks on bad input | `test_cli.py` error tests |
| NFR4 maintainability | review of [DESIGN.md](DESIGN.md); not machine-tested |
| NFR5 Python versions and no-Numba | CI matrix (below) |
| NFR6 determinism | `test_backend_equivalence.py` (thread-count independence, seeded data) |

## Continuous integration

`.github/workflows/ci.yml` runs the default test suite on Python 3.10 to 3.14, each with and without Numba installed. The "without Numba" runs prove the fallback path; the equivalence tests then skip the Numba parts automatically. A separate job runs `ruff check .` (style, bugbear and simplification rules) and `mypy` (type check of `src/`); both are clean and are configured in `pyproject.toml`.

## Coverage and its limits

Statement coverage of the package is 89 percent (`pytest --cov=projection`). Two caveats:
- `kernels_numba.py` shows 11 percent because compiled code is invisible to the coverage tool. It is verified by the equivalence tests instead, which compare its output with the reference.
- The remaining uncovered lines are error branches that need unusual files (for example a singular direction matrix).

## Not tested

- Visual quality of the video, beyond brightness checks. This is judged by eye (EXPERIMENT.md).
- Speed on other hardware. The performance tests encode thresholds measured on one machine; they are marked slow and not run in CI.
- Extreme plane coordinates (beyond about 1e9); the tests use planes within a few volume sizes of the volume.

## Running

```
pytest                 # 187 tests, about 15 seconds
pytest -m slow         # performance tests, needs a few seconds and about 150 MB of memory
pytest --cov=projection --cov-report=term-missing
ruff check .           # lint
mypy                   # type check
python scripts/bench.py --size 500 --backend numba numpy --interp nearest trilinear
```
