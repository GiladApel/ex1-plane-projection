import itertools
import math

import numpy as np
import pytest

from projection import synthetic
from projection.errors import InvalidInputError
from projection.geometry import Trapezoid, Triangle
from projection.projector import ReferenceProjector
from projection.sampler import Interpolation, sample_nearest, sample_trilinear
from projection.volume import Volume

DTYPES = synthetic.DTYPES
NEAREST = ReferenceProjector(Interpolation.NEAREST)
TRILINEAR = ReferenceProjector(Interpolation.TRILINEAR)


def axis_aligned_trapezoid(r, t, f, r0, t0, f0, length=9, top=5):
    """Rails run along axis r, the top edge along axis t, the plane sits at coordinate f0 of axis f."""
    def pt(rv, tv):
        p = [0.0, 0.0, 0.0]
        p[r], p[t], p[f] = rv, tv, f0
        return p
    return Trapezoid(pt(r0, t0), pt(r0 + length, t0), pt(r0, t0 + top), pt(r0 + length, t0 + top))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axes", list(itertools.permutations(range(3))))
def test_axis_aligned_plane_equals_numpy_slice(dtype, axes):
    r, t, f = axes
    volume = Volume(synthetic.random_volume((11, 13, 15), dtype, seed=1))
    r0, t0, f0 = 1, 2, 4
    plane = axis_aligned_trapezoid(r, t, f, r0, t0, f0)
    image = NEAREST.project(volume, plane)

    index = [slice(None)] * 3
    index[r], index[t], index[f] = slice(r0, r0 + 10), slice(t0, t0 + 6), f0
    expected = volume.data[tuple(index)]
    if r > t:
        expected = expected.T
    assert image.shape == (10, 6)
    assert image.dtype == np.dtype(dtype)
    assert np.array_equal(image, expected)


def test_triangle_on_gradient_has_constant_value_and_exact_pixel_count():
    volume = Volume(synthetic.gradient((12, 12, 12)))
    plane = Triangle((3, 5, 0), (3, 0, 10), (3, 10, 10))
    image = NEAREST.project(volume, plane)

    rails = plane.rails()
    assert set(np.unique(image)) == {0, 3}
    row_width = lambda y: 10 * y / math.sqrt(125)
    assert np.count_nonzero(image) == sum(math.floor(row_width(y)) + 1 for y in range(rails.H))


def test_triangle_apex_row_is_one_centered_pixel():
    data = synthetic.random_volume((12, 12, 12), seed=2)
    plane = Triangle((3, 5, 4), (3, 0, 10), (3, 10, 10))
    image = NEAREST.project(Volume(data), plane)
    centre = math.floor((plane.shape[1] - 1) / 2 + 0.5)
    expected_row = np.zeros(plane.shape[1], dtype=data.dtype)
    expected_row[centre] = data[3, 5, 4]
    assert np.array_equal(image[0], expected_row)


def test_plane_fully_outside_volume_is_all_zero():
    volume = Volume(synthetic.random_volume((12, 12, 12), seed=3))
    plane = axis_aligned_trapezoid(1, 2, 0, 50, 50, 50)
    image = NEAREST.project(volume, plane)
    assert image.shape == (10, 6) and not image.any()


def test_plane_partially_outside_is_zero_only_outside():
    data = synthetic.random_volume((12, 12, 12), seed=4)
    image = NEAREST.project(Volume(data), axis_aligned_trapezoid(1, 0, 2, -5, 2, 6))
    assert not image[:5].any()
    assert np.array_equal(image[5:], data[2:8, 0:5, 6].T)


def test_oblique_plane_on_gradient_matches_closed_form():
    volume = Volume(synthetic.gradient((12, 12, 12)))
    plane = Trapezoid((2, 3, 1), (8, 11, 1), (2, 3, 6), (8, 11, 6))
    image = NEAREST.project(volume, plane)
    assert image.shape == (11, 6)
    for y in range(11):
        assert np.all(image[y] == math.floor(2 + y * 0.6 + 0.5))


def test_out_buffer_is_cleared_reused_and_validated():
    volume = Volume(synthetic.random_volume((12, 12, 12), seed=5))
    plane = axis_aligned_trapezoid(0, 1, 2, 0, 0, 3)
    out = np.full(plane.shape, 77, dtype=np.uint8)
    result = NEAREST.project(volume, plane, out=out)
    assert result is out
    assert np.array_equal(out, NEAREST.project(volume, plane))
    with pytest.raises(InvalidInputError):
        NEAREST.project(volume, plane, out=np.zeros((3, 3), dtype=np.uint8))
    with pytest.raises(InvalidInputError):
        NEAREST.project(volume, plane, out=np.zeros(plane.shape, dtype=np.uint16))


def test_non_integer_edges_have_no_holes_in_rows():
    volume = Volume(np.full((20, 20, 20), 9, dtype=np.uint8))
    plane = Triangle((10, 2, 10), (4, 15, 10), (16, 15, 10))
    image = NEAREST.project(volume, plane)
    for row in image:
        filled = np.flatnonzero(row)
        if len(filled):
            assert np.all(row[filled[0]:filled[-1] + 1] == 9)


@pytest.mark.parametrize("dtype", DTYPES)
def test_trilinear_on_linear_gradient_equals_rounded_coordinate(dtype):
    volume = Volume(synthetic.gradient((12, 12, 12), dtype))
    plane = Trapezoid((2, 3, 1), (8, 11, 1), (2, 3, 6), (8, 11, 6))
    assert np.array_equal(TRILINEAR.project(volume, plane), NEAREST.project(volume, plane))


def test_trilinear_equals_nearest_on_integer_positions():
    volume = Volume(synthetic.random_volume((11, 13, 15), seed=6))
    plane = axis_aligned_trapezoid(0, 1, 2, 1, 2, 4)
    assert np.array_equal(TRILINEAR.project(volume, plane), NEAREST.project(volume, plane))


def test_sample_nearest_rounds_half_up_and_clips():
    data = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    assert sample_nearest(data, 0.49, 0, 0) == data[0, 0, 0]
    assert sample_nearest(data, 0.5, 0, 0) == data[1, 0, 0]
    assert sample_nearest(data, -0.5, 0, 0) == data[0, 0, 0]
    assert sample_nearest(data, -0.51, 0, 0) == 0
    assert sample_nearest(data, 1.49, 1.49, 1.49) == data[1, 1, 1]
    assert sample_nearest(data, 1.5, 0, 0) == 0


def test_sample_trilinear_weights():
    data = np.zeros((2, 2, 2), dtype=np.uint8)
    data[0, 0, 0], data[1, 1, 1] = 80, 160
    assert sample_trilinear(data, 0.5, 0.5, 0.5) == round((80 + 160) / 8)
    assert sample_trilinear(data, 0, 0, 0) == 80
    full = np.full((2, 2, 2), 100, dtype=np.uint8)
    assert sample_trilinear(full, -0.5, 0, 0) == 50
    assert sample_trilinear(full, 5, 5, 5) == 0


def test_trilinear_never_exceeds_dtype_maximum():
    top = np.iinfo(np.uint32).max
    data = np.full((3, 3, 3), top, dtype=np.uint32)
    assert sample_trilinear(data, 1.3, 1.7, 0.9) == top
