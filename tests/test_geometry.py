import numpy as np
import pytest

from projection.errors import InvalidInputError
from projection.geometry import Trapezoid, Triangle, plane_from_points


def test_isosceles_triangle_size():
    plane = Triangle((5, 0, 0), (0, 10, 0), (10, 10, 0))
    r = plane.rails()
    dab = np.hypot(5, 10)
    assert r.H == int(dab) + 1
    assert r.W == 11
    np.testing.assert_allclose(r.A1, r.A2)
    np.testing.assert_allclose(np.linalg.norm(r.v1), 1.0, atol=1e-3)
    assert plane.shape == (r.H, r.W)


def test_triangle_rejects_non_isosceles():
    with pytest.raises(InvalidInputError, match="isosceles"):
        Triangle((0, 0, 0), (10, 0, 0), (0, 6, 0))


def test_triangle_force_allows_non_isosceles():
    plane = Triangle((0, 0, 0), (10, 0, 0), (0, 6, 0), force=True)
    assert plane.rails().H == 11


def test_triangle_within_tolerance_is_accepted():
    Triangle((0, 0, 0), (10, 0, 0), (0, 10.005, 0))


def test_triangle_rejects_collinear():
    with pytest.raises(InvalidInputError, match="collinear"):
        Triangle((0, 0, 0), (5, 0, 0), (5, 0, 0))
    with pytest.raises(InvalidInputError, match="collinear"):
        Triangle((0, 0, 0), (5, 0, 0), (-5, 0, 0))


def test_triangle_rejects_coincident_points():
    with pytest.raises(InvalidInputError, match="shorter than one voxel"):
        Triangle((1, 1, 1), (1, 1, 1), (4, 5, 6), force=True)


def test_rejects_edge_shorter_than_one_voxel():
    with pytest.raises(InvalidInputError, match="shorter than one voxel"):
        Triangle((0, 0, 0), (0.5, 0, 0), (0, 0.5, 0))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite(bad):
    with pytest.raises(InvalidInputError, match="non-finite"):
        Triangle((0, 0, 0), (10, 0, 0), (0, bad, 0))


def test_rejects_wrong_coordinate_count():
    with pytest.raises(InvalidInputError, match="3 coordinates"):
        Triangle((0, 0), (10, 0, 0), (0, 10, 0))


def test_trapezoid_rails_and_size():
    plane = Trapezoid((2, 0, 0), (0, 10, 0), (8, 0, 0), (10, 10, 0))
    r = plane.rails()
    np.testing.assert_allclose(r.A1, (2, 0, 0))
    np.testing.assert_allclose(r.A2, (8, 0, 0))
    assert r.W == 11  # max(top=6, bottom=10) + 1
    assert r.H == int(np.hypot(2, 10)) + 1


def test_trapezoid_with_equal_top_is_triangle():
    tri = Triangle((5, 0, 0), (0, 10, 0), (10, 10, 0)).rails()
    trap = Trapezoid((5, 0, 0), (0, 10, 0), (5, 0, 0), (10, 10, 0)).rails()
    assert (tri.H, tri.W) == (trap.H, trap.W)
    np.testing.assert_allclose(tri.v1, trap.v1)


def test_trapezoid_rejects_non_coplanar():
    with pytest.raises(InvalidInputError, match="coplanar"):
        Trapezoid((2, 0, 0), (0, 10, 0), (8, 0, 0), (10, 8, 6))


def test_trapezoid_rejects_non_parallel_sides():
    with pytest.raises(InvalidInputError, match="parallel"):
        Trapezoid((2, 0, 0), (0, 10, 0), (8, 1, 0), (10, 10, 0), force=True)


def test_trapezoid_rejects_non_isosceles():
    with pytest.raises(InvalidInputError, match="isosceles"):
        Trapezoid((2, 0, 0), (0, 10, 0), (8, 0, 0), (10, 20, 0))


def test_trapezoid_rejects_fully_degenerate():
    with pytest.raises(InvalidInputError, match="degenerate"):
        Trapezoid((0, 0, 0), (5, 0, 0), (0, 0, 0), (5, 0, 0))


def test_plane_from_points_dispatch():
    assert isinstance(plane_from_points([(5, 0, 0), (0, 10, 0), (10, 10, 0)]), Triangle)
    assert isinstance(plane_from_points([(2, 0, 0), (0, 10, 0), (8, 0, 0), (10, 10, 0)]), Trapezoid)
    with pytest.raises(InvalidInputError, match="3 or 4 points"):
        plane_from_points([(0, 0, 0), (1, 1, 1)])
