"""Known answers from geometry: areas of cuts through a cube and a sphere, and exact rounding at half-voxel points.

A rectangle whose legs are perpendicular to its top samples an orthonormal grid on the plane: pixel (row, col) is at
corner + col * u + row * v, one voxel apart in both directions. The number of pixels inside a solid is then the area of
the cut, up to pixels along its outline, so every expected value below comes from a formula, not from the program.
"""
import math
from fractions import Fraction

import numpy as np
import pytest
from conftest import assert_images_equal

from projection import synthetic
from projection.geometry import Trapezoid
from projection.projector import NumbaProjector, NumpyProjector, ReferenceProjector
from projection.volume import Volume

try:
    NumbaProjector()
    BACKENDS = [ReferenceProjector, NumpyProjector, NumbaProjector]
except ImportError:
    BACKENDS = [ReferenceProjector, NumpyProjector]

SIDE, LOW = 60, 10
CUBE = Volume(synthetic.cube((80, 80, 80), LOW, SIDE))
CUBE_CENTER = np.full(3, LOW - 0.5 + SIDE / 2)              # center of the continuous box


def rectangle(center, u, v, size: int) -> Trapezoid:
    """A size x size square with orthonormal edge directions u (columns) and v (rows), centered on center."""
    center, u, v = (np.asarray(x, dtype=np.float64) for x in (center, u, v))
    corner = center - size / 2 * u - size / 2 * v
    return Trapezoid(corner, corner + size * v, corner + size * u, corner + size * u + size * v)


def inside_count(volume: Volume, plane: Trapezoid) -> int:
    return int(np.count_nonzero(ReferenceProjector().project(volume, plane)))


def basis(normal):
    """Two orthonormal vectors spanning the plane perpendicular to normal."""
    n = np.asarray(normal, dtype=np.float64) / np.linalg.norm(normal)
    u = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def test_axis_aligned_cut_of_a_cube_is_its_face():
    plane = rectangle((40, 40, LOW + 20), (1, 0, 0), (0, 1, 0), 78)
    assert inside_count(CUBE, plane) == SIDE * SIDE


def test_cut_through_the_center_across_the_diagonal_is_a_regular_hexagon():
    # perpendicular to (1, 1, 1) through the center: a regular hexagon with sides SIDE / sqrt(2)
    u, v = basis((1, 1, 1))
    area = 3 * math.sqrt(3) / 4 * SIDE**2
    perimeter = 6 * SIDE / math.sqrt(2)
    count = inside_count(CUBE, rectangle(CUBE_CENTER, u, v, 100))
    assert abs(count - area) <= perimeter, (count, area)


@pytest.mark.parametrize("t", [12.0, 25.0, 40.0])
def test_cut_near_a_corner_is_an_equilateral_triangle(t):
    # the plane x' + y' + z' = t, measured from the cube's corner, cuts off a triangle with sides t * sqrt(2)
    corner = np.full(3, LOW - 0.5)
    u, v = basis((1, 1, 1))
    center = corner + t / 3                                  # the triangle's center, on the plane
    area = math.sqrt(3) / 2 * t**2
    perimeter = 3 * t * math.sqrt(2)
    count = inside_count(CUBE, rectangle(center, u, v, int(2 * t) + 4))
    assert abs(count - area) <= perimeter, (count, area)


SPHERE_SHAPE = (90, 90, 90)
SPHERE = Volume(synthetic.sphere(SPHERE_SHAPE))
SPHERE_CENTER = (np.array(SPHERE_SHAPE) - 1) / 2
RADIUS = min(SPHERE_SHAPE) / 3


@pytest.mark.parametrize("distance", [0.0, 10.0, 20.0, 27.0])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_any_cut_of_a_sphere_is_a_disk_of_the_right_area(distance, seed):
    normal = np.random.default_rng(seed).normal(size=3)
    normal /= np.linalg.norm(normal)
    u, v = basis(normal)
    disk = math.sqrt(RADIUS**2 - distance**2)
    count = inside_count(SPHERE, rectangle(SPHERE_CENTER + distance * normal, u, v, int(2 * disk) + 6))
    assert abs(count - math.pi * disk**2) <= 2 * math.pi * disk, (count, math.pi * disk**2)


HALF_SHAPE = (120, 120, 12)
INDEX = np.indices(HALF_SHAPE, dtype=np.uint32)
INDEX_DATA = np.ascontiguousarray(INDEX[0] + 1000 * INDEX[1] + 1000000 * INDEX[2] + 1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("p", "q", "n"), [(3, 4, 5), (5, 12, 13), (8, 15, 17), (20, 21, 29)])
def test_points_exactly_between_voxels_round_up_on_every_backend(backend, p, q, n):
    """Along a p-q-n direction many sample points fall exactly halfway between two voxels, and an integer row
    width ends exactly on its last sample. Floats land a hair off both; the expected voxel of every pixel is
    computed here with exact fractions, so the image must follow the rule "a half rounds up" (SRS D8) exactly."""
    u = (Fraction(p, n), Fraction(q, n), Fraction(0))
    v = (Fraction(-q, n), Fraction(p, n), Fraction(0))
    origin = (Fraction(121, 2), Fraction(41, 2), Fraction(11, 2))
    size = 2 * n
    corners = [
        origin,
        tuple(origin[k] + size * v[k] for k in range(3)),
        tuple(origin[k] + size * u[k] for k in range(3)),
        tuple(origin[k] + size * (u[k] + v[k]) for k in range(3)),
    ]
    plane = Trapezoid(*[[float(c) for c in point] for point in corners])
    expected = np.zeros((size + 1, size + 1), dtype=INDEX_DATA.dtype)
    halves = 0
    for row in range(size + 1):
        for col in range(size + 1):
            point = [origin[k] + col * u[k] + row * v[k] for k in range(3)]
            halves += any((x + Fraction(1, 2)).denominator == 1 for x in point)
            voxel = tuple(math.floor(x + Fraction(1, 2)) for x in point)
            if all(0 <= voxel[k] < HALF_SHAPE[k] for k in range(3)):
                expected[row, col] = INDEX_DATA[voxel]
    assert halves > 0
    assert_images_equal(backend().project(Volume(INDEX_DATA), plane), expected)
