import numpy as np
import pytest

from projection import synthetic
from projection.projector import NumbaProjector, NumpyProjector
from projection.volume import Volume

MIN_FPS = {"numba": 100, "numpy": 30}


def median_fps(projector, volume, planes):
    import time

    out = np.zeros(planes[0].shape, dtype=volume.dtype)
    projector.project(volume, planes[0], out=out)
    times = []
    for plane in planes:
        start = time.perf_counter()
        projector.project(volume, plane, out=out)
        times.append(time.perf_counter() - start)
    return 1 / np.median(times)


@pytest.fixture(scope="module")
def big_volume():
    return Volume(synthetic.random_volume((500, 500, 500), np.uint8, seed=0))


@pytest.mark.slow
def test_numba_meets_100_fps_on_500_cubed(big_volume):
    pytest.importorskip("numba")
    fps = median_fps(NumbaProjector(), big_volume, synthetic.oblique_rectangles(500, 100))
    assert fps >= MIN_FPS["numba"], f"{fps:.0f} fps"


@pytest.mark.slow
def test_numpy_meets_30_fps_on_500_cubed(big_volume):
    fps = median_fps(NumpyProjector(), big_volume, synthetic.oblique_rectangles(500, 50))
    assert fps >= MIN_FPS["numpy"], f"{fps:.0f} fps"
