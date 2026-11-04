import numpy as np
import pytest

from projection.errors import InvalidInputError, NotIsoscelesError
from projection.geometry import MAX_IMAGE_SIDE, Trapezoid, Triangle, plane_from_points


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
        Triangle((0, 0, 0), (5, 0, 0), (10, 0, 0), force=True)
    with pytest.raises(InvalidInputError, match="collinear"):
        Triangle((0, 0, 0), (5, 0, 0), (-5, 0, 0))


@pytest.mark.parametrize(("points", "message"), [
    (((1, 1, 1), (1, 1, 1), (4, 5, 6)), "points a and b are the same point (1, 1, 1)"),
    (((1, 1, 1), (4, 5, 6), (1, 1, 1)), "points a and c are the same point (1, 1, 1)"),
    (((0, 0, 0), (5, 0, 0), (5, 0, 0)), "points b and c are the same point (5, 0, 0)"),
    (((0, 0, 0), (2.5, -1, 0), (2.5, -1, 0)), "points b and c are the same point (2.5, -1, 0)"),
])
def test_triangle_names_the_coincident_points(points, message):
    with pytest.raises(InvalidInputError) as error:
        Triangle(*points, force=True)
    assert str(error.value) == message


@pytest.mark.parametrize(("points", "pair"), [
    (((0, 0, 0), (0, 0, 0), (8, 0, 0), (10, 10, 0)), "a' and b"),
    (((2, 0, 0), (0, 10, 0), (0, 10, 0), (10, 10, 0)), "b and a''"),
    (((2, 0, 0), (0, 10, 0), (2, 0, 0), (2, 0, 0)), "a' and c"),
    (((2, 0, 0), (0, 10, 0), (8, 0, 0), (8, 0, 0)), "a'' and c"),
])
def test_trapezoid_names_the_coincident_points(points, pair):
    with pytest.raises(InvalidInputError, match=f"^points {pair} are the same point"):
        Trapezoid(*points, force=True)


def test_trapezoid_with_one_side_of_zero_length_is_allowed():
    assert Trapezoid((5, 0, 0), (0, 10, 0), (5, 0, 0), (10, 10, 0)).shape == (12, 11)
    assert Trapezoid((0, 0, 0), (5, 10, 0), (10, 0, 0), (5, 10, 0)).shape == (12, 11)


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
    with pytest.raises(NotIsoscelesError, match="isosceles.*expected point order a', b, a'', c"):
        Trapezoid((2, 0, 0), (0, 10, 0), (8, 0, 0), (12, 10, 0))


def test_trapezoid_in_wrong_point_order_names_the_expected_order():
    # a', a'', b, c instead of a', b, a'', c
    for force in (False, True):
        with pytest.raises(InvalidInputError, match="not parallel.*expected point order a', b, a'', c"):
            Trapezoid((2, 0, 0), (8, 0, 0), (0, 10, 0), (10, 10, 0), force=force)


def test_trapezoid_with_crossing_legs_is_rejected():
    # a' and a'' swapped: top and base are parallel but point opposite ways, so the legs cross
    with pytest.raises(InvalidInputError, match="legs a'b and a''c cross"):
        Trapezoid((8, 0, 0), (0, 10, 0), (2, 0, 0), (10, 10, 0))


def test_structural_errors_come_before_the_isosceles_check():
    # not isosceles and not coplanar: the structural error is reported, not the isosceles one
    with pytest.raises(InvalidInputError, match="not coplanar"):
        Trapezoid((2, 0, 0), (0, 10, 0), (8, 0, 3), (12, 10, 0))


def test_image_larger_than_the_video_limit_is_rejected():
    with pytest.raises(InvalidInputError, match=r"would be 7072 x 10001 pixels, more than 4096"):
        Triangle((0, 0, 0), (-5000, 5000, 0), (5000, 5000, 0), force=True)
    assert max(Triangle((0, 0, 0), (-1000, 2000, 0), (1000, 2000, 0)).shape) <= MAX_IMAGE_SIDE


def test_triangle_isosceles_error_has_no_order_hint():
    with pytest.raises(InvalidInputError, match=r"isosceles: equal edges are 10 and 6$"):
        Triangle((0, 0, 0), (10, 0, 0), (0, 6, 0))


def test_trapezoid_rejects_fully_degenerate():
    with pytest.raises(InvalidInputError, match="degenerate"):
        Trapezoid((0, 0, 0), (5, 0, 0), (0, 0, 0), (5, 0, 0))


def test_corners_are_the_points_as_given():
    triangle = Triangle((5, 0, 0), (0, 10, 0), (10, 10, 0))
    assert [p.tolist() for p in triangle.corners] == [[5, 0, 0], [0, 10, 0], [5, 0, 0], [10, 10, 0]]
    trapezoid = Trapezoid((2, 0, 0), (0, 10, 0), (8, 0, 0), (10, 10, 0))
    assert [p.tolist() for p in trapezoid.corners] == [[2, 0, 0], [0, 10, 0], [8, 0, 0], [10, 10, 0]]


def test_plane_from_points_dispatch():
    assert isinstance(plane_from_points([(5, 0, 0), (0, 10, 0), (10, 10, 0)]), Triangle)
    assert isinstance(plane_from_points([(2, 0, 0), (0, 10, 0), (8, 0, 0), (10, 10, 0)]), Trapezoid)
    with pytest.raises(InvalidInputError, match="3 or 4 points"):
        plane_from_points([(0, 0, 0), (1, 1, 1)])
