"""Speed tests, all marked slow (run with `pytest -m slow`).

Absolute floors hold on the reference machine (SRS section 7) with a margin of about 2x.
Tests also marked `portable` compare two measurements taken on the same machine (thread counts,
backends), so they hold on any hardware; CI runs them with `pytest -m portable`.
Every assert message names the backend, mode and measured fps, so a failure says what slowed down.
"""
import re
import time

import numpy as np
import pytest

from projection import synthetic
from projection.cli import EXIT_OK, main
from projection.plane_gen import rotation, write_planes_csv
from projection.projector import NumbaProjector, NumpyProjector
from projection.sampler import Interpolation
from projection.volume import Volume

FRAMES = 40
REPEATS = 3
NEAREST, TRILINEAR = Interpolation.NEAREST, Interpolation.TRILINEAR


def best_fps(projector, volume, planes) -> float:
    """Best of REPEATS median frame rates; the first call (JIT compile, buffer allocation) is not timed."""
    out = np.zeros(planes[0].shape, dtype=volume.dtype)
    projector.project(volume, planes[0], out=out)
    best = 0.0
    for _ in range(REPEATS):
        times = []
        for plane in planes:
            start = time.perf_counter()
            projector.project(volume, plane, out=out)
            times.append(time.perf_counter() - start)
        best = max(best, 1.0 / float(np.median(times)))
    return best


def build(backend: str, mode: Interpolation):
    if backend == "numba":
        pytest.importorskip("numba")
        return NumbaProjector(mode)
    return NumpyProjector(mode)


@pytest.fixture(scope="module")
def big_volume():
    return Volume(synthetic.random_volume((500, 500, 500), np.uint8, seed=0))


@pytest.fixture(scope="module")
def planes():
    return synthetic.oblique_rectangles(500, FRAMES)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("backend", "mode", "floor"),
    [("numba", NEAREST, 500), ("numba", TRILINEAR, 250), ("numpy", NEAREST, 100)],
)
def test_projection_speed_on_500_cubed(big_volume, planes, backend, mode, floor):
    fps = best_fps(build(backend, mode), big_volume, planes)
    assert fps >= floor, f"{backend} {mode.value}: {fps:.0f} fps, floor {floor} fps (500^3 uint8, oblique planes)"


@pytest.mark.slow
@pytest.mark.portable
def test_numba_scales_with_threads(big_volume, planes):
    numba = pytest.importorskip("numba")
    threads = numba.config.NUMBA_NUM_THREADS
    if threads < 2:
        pytest.skip("needs at least 2 CPU threads")
    # 16 threads measured x6.5 over 1 thread; a 4-thread CI runner gives less, so the bar follows the thread count.
    required = min(3.0, max(1.5, threads / 4))
    projector = NumbaProjector()
    numba.set_num_threads(1)
    try:
        single = best_fps(projector, big_volume, planes)
    finally:
        numba.set_num_threads(threads)
    multi = best_fps(projector, big_volume, planes)
    assert multi >= required * single, (
        f"numba nearest: {multi:.0f} fps on {threads} threads vs {single:.0f} fps on 1 thread "
        f"(x{multi / single:.1f}, need x{required:.1f}); is project_nearest still parallel?"
    )


@pytest.mark.slow
@pytest.mark.portable
def test_numba_is_much_faster_than_numpy(big_volume, planes):
    numba_fps = best_fps(build("numba", NEAREST), big_volume, planes)
    numpy_fps = best_fps(build("numpy", NEAREST), big_volume, planes)
    assert numba_fps >= 2 * numpy_fps, (
        f"numba nearest {numba_fps:.0f} fps vs numpy nearest {numpy_fps:.0f} fps "
        f"(x{numba_fps / numpy_fps:.1f}, need x2); the Numba kernel lost its advantage"
    )


@pytest.mark.slow
def test_demo_end_to_end_speed(tmp_path, capsys):
    """The full demo (load, project, intensity window, compose, H.264 encode) on 445x445 frames."""
    shape = (256, 256, 256)
    nrrd_path, csv_path = tmp_path / "volume.nrrd", tmp_path / "planes.csv"
    synthetic.save_nrrd(nrrd_path, synthetic.checkerboard(shape, np.uint16))
    write_planes_csv(csv_path, rotation(shape, axis=1, frames=180))

    assert main([str(nrrd_path), str(csv_path), str(tmp_path / "out.mp4")]) == EXIT_OK
    summary = capsys.readouterr().out
    match = re.search(r"\[(\w+), \w+\] \| projection (\d+) fps, end-to-end (\d+) fps", summary)
    assert match, summary
    backend, projection_fps, end_to_end_fps = match[1], int(match[2]), int(match[3])
    assert projection_fps >= 100, f"{backend}: projection {projection_fps} fps, floor 100 ({summary.strip()})"
    assert end_to_end_fps >= 30, f"{backend}: end-to-end {end_to_end_fps} fps, floor 30 ({summary.strip()})"
