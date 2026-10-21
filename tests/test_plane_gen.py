import numpy as np
import pytest

from projection import synthetic
from projection.geometry import plane_from_points
from projection.plane_csv import parse_planes
from projection.plane_gen import axial_sweep, rotation, write_planes_csv
from projection.projector import ReferenceProjector
from projection.volume import Volume


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_sweep_reproduces_every_slice(axis):
    data = synthetic.random_volume((7, 9, 11), np.uint16, seed=axis)
    volume = Volume(data)
    sets = axial_sweep(data.shape, axis)
    assert len(sets) == data.shape[axis]
    for k, points in enumerate(sets):
        image = ReferenceProjector().project(volume, plane_from_points(points))
        assert np.array_equal(image, np.take(data, k, axis=axis))


def test_sweep_with_frame_count_spans_the_volume():
    sets = axial_sweep((10, 12, 14), 2, frames=5)
    assert len(sets) == 5
    heights = [s[0, 2] for s in sets]
    assert heights[0] == 0 and heights[-1] == 13


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rotation_planes_are_valid_centered_and_contain_the_axis(axis):
    shape = (20, 24, 28)
    center = (np.array(shape) - 1) / 2
    sets = rotation(shape, axis, frames=12)
    assert len(sets) == 12
    for points in sets:
        plane_from_points(points)
        assert np.allclose(points.mean(axis=0), center)
        rail = points[1] - points[0]
        assert np.allclose(rail / np.linalg.norm(rail), np.eye(3)[axis])
    first, quarter = sets[0], sets[3]
    assert not np.allclose(first, quarter)


def test_rotation_by_a_quarter_turn_swaps_the_in_plane_axis():
    shape = (21, 21, 21)
    sets = rotation(shape, axis=2, frames=4, side=20)
    width_dirs = [(s[2] - s[0]) / np.linalg.norm(s[2] - s[0]) for s in sets]
    assert np.allclose(width_dirs[0], [1, 0, 0]) and np.allclose(width_dirs[1], [0, 1, 0], atol=1e-12)


def test_csv_roundtrip(tmp_path):
    sets = rotation((30, 30, 30), 1, frames=8)
    path = tmp_path / "rot.csv"
    write_planes_csv(path, sets)
    planes = parse_planes(path)
    assert len(planes) == 8
    assert all(p.shape == planes[0].shape for p in planes)
