"""NumpyProjector and NumbaProjector must reproduce ReferenceProjector exactly."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from projection import synthetic
from projection.errors import InvalidInputError
from projection.geometry import Trapezoid, Triangle
from projection.projector import (
    NumbaProjector,
    NumpyProjector,
    ReferenceProjector,
    make_projector,
)
from projection.sampler import Interpolation
from projection.volume import Volume

try:
    NumbaProjector()
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

def threaded_numpy(mode=Interpolation.NEAREST):
    """NumPy backend forced onto the multi-chunk path even for tiny frames."""
    return NumpyProjector(mode, threads=3, min_parallel_cells=0)


BACKENDS = [NumpyProjector, threaded_numpy] + ([NumbaProjector] if HAVE_NUMBA else [])
MODES = list(Interpolation)
SETTINGS = settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def grid(lo, hi):
    return st.integers(int(lo * 4), int(hi * 4)).map(lambda v: v / 4)


@st.composite
def volumes(draw):
    shape = tuple(draw(st.integers(3, 14)) for _ in range(3))
    dtype = draw(st.sampled_from(synthetic.DTYPES))
    return Volume(synthetic.random_volume(shape, dtype, seed=draw(st.integers(0, 2**16))))


@st.composite
def planes(draw, shape):
    """Isosceles trapezoids (top may be 0, which gives a triangle) in random or axis-aligned poses."""
    length = draw(st.one_of(grid(1, 25), st.integers(1, 25).map(float)))
    top = draw(st.one_of(st.just(0.0), grid(1, 20), st.integers(1, 20).map(float)))
    bottom = draw(st.one_of(grid(1, 20), st.integers(1, 20).map(float)))
    center = np.array([draw(grid(-3, s + 3)) for s in shape])
    if draw(st.booleans()):
        rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        u, w = q[:, 0], q[:, 1]
    else:
        axes = draw(st.permutations(range(3)))
        signs = draw(st.lists(st.sampled_from([-1.0, 1.0]), min_size=2, max_size=2))
        u, w = np.zeros(3), np.zeros(3)
        u[axes[0]], w[axes[1]] = signs
    m = center - length / 2 * u
    far = m + length * u
    try:
        return Trapezoid(m - top / 2 * w, far - bottom / 2 * w, m + top / 2 * w, far + bottom / 2 * w)
    except InvalidInputError:
        assume(False)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", MODES)
@SETTINGS
@given(data=st.data())
def test_random_planes_match_reference(backend, mode, data):
    volume = data.draw(volumes())
    plane = data.draw(planes(volume.shape))
    expected = ReferenceProjector(mode).project(volume, plane)
    actual = backend(mode).project(volume, plane)
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("offset", [0.0, 0.5, 0.25, 0.49999999, 0.5000001])
def test_half_integer_coordinates_lock_the_rounding_rule(backend, mode, offset):
    volume = Volume(synthetic.random_volume((10, 10, 10), np.uint16, seed=7))
    plane = Trapezoid(
        (1 + offset, 1 + offset, 4 + offset), (1 + offset, 8 + offset, 4 + offset),
        (6 + offset, 1 + offset, 4 + offset), (6 + offset, 8 + offset, 4 + offset),
    )
    assert np.array_equal(backend(mode).project(volume, plane), ReferenceProjector(mode).project(volume, plane))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", MODES)
def test_triangle_matches_reference(backend, mode):
    volume = Volume(synthetic.random_volume((16, 16, 16), np.uint8, seed=8))
    plane = Triangle((8, 2.5, 3), (3, 14, 12), (13, 14, 12))
    assert np.array_equal(backend(mode).project(volume, plane), ReferenceProjector(mode).project(volume, plane))


@pytest.mark.parametrize("backend", BACKENDS)
def test_plane_outside_volume_is_zero(backend):
    volume = Volume(synthetic.random_volume((8, 8, 8), seed=9))
    plane = Triangle((100, 100, 100), (90, 120, 100), (110, 120, 100))
    assert not backend().project(volume, plane).any()


@pytest.mark.parametrize("backend", BACKENDS)
def test_out_buffer_is_used_and_cleared(backend):
    volume = Volume(synthetic.random_volume((8, 8, 8), seed=10))
    plane = Triangle((4, 1, 3), (1, 7, 3), (7, 7, 3))
    out = np.full(plane.shape, 200, dtype=np.uint8)
    result = backend().project(volume, plane, out=out)
    assert result is out
    assert np.array_equal(out, ReferenceProjector().project(volume, plane))
    with pytest.raises(InvalidInputError):
        backend().project(volume, plane, out=np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(InvalidInputError, match="C-contiguous"):
        backend().project(volume, plane, out=np.zeros(plane.shape[::-1], dtype=np.uint8).T)


def test_make_projector_selection():
    assert isinstance(make_projector("reference"), ReferenceProjector)
    assert isinstance(make_projector("numpy"), NumpyProjector)
    assert isinstance(make_projector("auto"), NumbaProjector if HAVE_NUMBA else NumpyProjector)
    with pytest.raises(InvalidInputError, match="unknown backend"):
        make_projector("gpu")


def test_make_projector_falls_back_without_numba(monkeypatch):
    import projection

    monkeypatch.setitem(sys.modules, "projection.kernels_numba", None)
    monkeypatch.delattr(projection, "kernels_numba", raising=False)
    assert isinstance(make_projector("auto"), NumpyProjector)
    with pytest.raises(InvalidInputError, match="not installed"):
        make_projector("numba")


def test_package_import_exposes_the_api_without_loading_numba():
    code = (
        "import sys, projection; assert set(projection.__all__) <= set(dir(projection)); "
        "assert 'projection.kernels_numba' not in sys.modules and 'numba' not in sys.modules"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("threads", [1, 2, 3, 7])
def test_numpy_result_does_not_depend_on_thread_count(mode, threads):
    volume = Volume(synthetic.random_volume((40, 40, 40), np.uint16, seed=11))
    plane = synthetic.random_plane(np.random.default_rng(5), (19.5, 19.5, 19.5), 41.5, 30.0, 12.0)
    expected = NumpyProjector(mode, threads=1).project(volume, plane)
    actual = NumpyProjector(mode, threads=threads, min_parallel_cells=0).project(volume, plane)
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual, ReferenceProjector(mode).project(volume, plane))


def test_numpy_rejects_zero_threads():
    with pytest.raises(ValueError, match="threads"):
        NumpyProjector(threads=0)
