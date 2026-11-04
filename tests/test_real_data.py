"""Tests on a real scan: MRHead (256 x 256 x 130, int16 MRI), the volume the assignment links to.

`python scripts/download_data.py MRHead` fetches it. Locally the tests skip when it is missing. CI downloads it and
sets REAL_DATA_REQUIRED=1, which turns the skip into a failure, so these tests cannot silently stop running there.

Known answers:
- a plane on a voxel grid plane is exactly that slice of the array (no oracle code involved);
- the planes of the published experiments (docs/experiment/*.csv) still give the published images: a CRC-32
  per frame in tests/data/mrhead_golden.json, and the committed MP4 files up to video-codec noise;
- world coordinates (--world) give the same images as the equivalent voxel coordinates.
After an intended change to the output, regenerate the golden file with UPDATE_GOLDEN=1 and re-make the videos.
"""
import json
import os
import zlib
from pathlib import Path

import numpy as np
import pytest
from conftest import assert_images_equal
from test_cli import read_frames

from projection import Trapezoid, load_nrrd, make_projector, parse_planes
from projection.cli import EXIT_OK, main
from projection.projector import NumbaProjector
from projection.render import FrameComposer, IntensityWindow
from projection.sampler import Interpolation

ROOT = Path(__file__).resolve().parent.parent
MRHEAD = ROOT / "data" / "MRHead.nrrd"
EXPERIMENT = ROOT / "docs" / "experiment"
GOLDEN = Path(__file__).resolve().parent / "data" / "mrhead_golden.json"

if not MRHEAD.is_file():
    if os.environ.get("REAL_DATA_REQUIRED"):
        pytest.fail(f"{MRHEAD} is missing; run python scripts/download_data.py MRHead", pytrace=False)
    pytest.skip("data/MRHead.nrrd not downloaded (python scripts/download_data.py MRHead)", allow_module_level=True)

try:
    NumbaProjector()
    FAST = ["numpy", "numba"]
except ImportError:
    FAST = ["numpy"]
MODES = list(Interpolation)


@pytest.fixture(scope="module")
def volume():
    return load_nrrd(MRHEAD)


@pytest.fixture(scope="module")
def rotate():
    return parse_planes(EXPERIMENT / "mrhead_rotate.csv")


def axis_plane(axis: int, k: int, shape) -> tuple[Trapezoid, tuple]:
    """The rectangle covering voxel plane `k` across `axis`, and the index of the matching array slice.
    Rows run along the first remaining axis (a' to b), columns along the second (a' to a'')."""
    rows, cols = (a for a in range(3) if a != axis)
    a1 = np.zeros(3)
    a1[axis] = k
    b, a2 = a1.copy(), a1.copy()
    b[rows] = shape[rows] - 1
    a2[cols] = shape[cols] - 1
    index = [slice(None)] * 3
    index[axis] = k
    return Trapezoid(a1, b, a2, b + a2 - a1), tuple(index)


def test_loads_mrhead(volume):
    assert volume.shape == (256, 256, 130)
    assert volume.dtype == np.dtype(np.uint16)
    assert volume.value_offset == 0
    assert volume.origin is not None and volume.directions.shape == (3, 3)
    np.testing.assert_allclose(volume.world_to_voxel(volume.origin[None, :]), [[0, 0, 0]], atol=1e-9)


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
@pytest.mark.parametrize("backend", FAST + ["reference"])
@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
def test_axis_planes_are_the_array_slices(volume, axis, backend, mode):
    projector = make_projector(backend, mode)
    last = volume.shape[axis] - 1
    for k in (0, 1, last // 2, last) if backend != "reference" else (last // 2,):
        plane, index = axis_plane(axis, k, volume.shape)
        assert_images_equal(projector.project(volume, plane), volume.data[index], f"slice {k} pixels")


def test_sweep_experiment_is_the_y_slices(volume):
    """Experiment 1: frame k of the published sweep is slice y = k of the scan."""
    planes = parse_planes(EXPERIMENT / "mrhead_sweep.csv")
    projector = make_projector("auto")
    assert len(planes) == volume.shape[1]
    for k, plane in enumerate(planes):
        assert_images_equal(projector.project(volume, plane), volume.data[:, k, :], f"frame {k + 1} pixels")


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
@pytest.mark.parametrize("backend", FAST + ["reference"])
def test_rotate_experiment_reproduces_the_golden_frames(volume, rotate, backend, mode):
    """Experiments 2 and 3: every frame matches its stored CRC-32 (the reference backend checks every 15th)."""
    projector = make_projector(backend, mode)
    frames = range(0, len(rotate), 15 if backend == "reference" else 1)
    key = f"rotate_{mode.value}"
    actual = {i: zlib.crc32(np.ascontiguousarray(projector.project(volume, rotate[i]))) for i in frames}
    if os.environ.get("UPDATE_GOLDEN") and backend == "numpy":
        stored = json.loads(GOLDEN.read_text()) if GOLDEN.is_file() else {}
        stored[key] = [actual[i] for i in frames]
        GOLDEN.write_text(json.dumps(stored, indent=0) + "\n")
    golden = json.loads(GOLDEN.read_text())[key]
    assert len(golden) == len(rotate)
    changed = [i + 1 for i in frames if actual[i] != golden[i]]
    assert not changed, f"{len(changed)} of {len(frames)} {key} frames changed: frames {changed[:10]}"


def test_world_coordinates_give_the_voxel_images(volume, rotate, tmp_path):
    """The rotation planes written in NRRD world coordinates (anisotropic, axes swapped and flipped)."""
    points = np.loadtxt(EXPERIMENT / "mrhead_rotate.csv", delimiter=",").reshape(-1, 4, 3)
    world = volume.origin + points @ volume.directions
    path = tmp_path / "world.csv"
    np.savetxt(path, world.reshape(len(world), 12), delimiter=",", fmt="%.17g")
    converted = parse_planes(path, to_voxel=volume.world_to_voxel)
    projector = make_projector("auto")
    for i, (plane, expected) in enumerate(zip(converted, rotate, strict=True)):
        assert plane.shape == expected.shape, f"frame {i + 1}"
        assert_images_equal(projector.project(volume, plane), projector.project(volume, expected),
                            f"frame {i + 1} pixels")


def expected_frames(volume, planes, mode):
    """The frames Ex0.py writes before video encoding."""
    projector = make_projector("auto", mode)
    window = IntensityWindow.from_volume(volume)
    composer = FrameComposer.for_planes(planes)
    return np.stack([composer.compose(window.apply(projector.project(volume, p))).copy() for p in planes])


def assert_video_matches(path, expected):
    """Every decoded frame is closer (mean |difference|) to its exact frame than to that frame moved by one pixel
    or to the frames before and after it. Codec noise differs between platforms' decoders (up to 1.6 gray levels
    here, 2.05 on macOS), so a fixed limit would not hold; this comparison calibrates itself."""
    decoded = read_frames(path)[..., 0].astype(np.float64)
    expected = expected.astype(np.float64)
    assert decoded.shape == expected.shape, f"{path.name}: {decoded.shape}, expected {expected.shape}"
    own = np.abs(decoded - expected).mean(axis=(1, 2))
    rival = np.min([np.abs(decoded - np.roll(expected, step, axis)).mean(axis=(1, 2))
                    for step in (-1, 1) for axis in (0, 1, 2)], axis=0)
    wrong = np.nonzero(rival < own)[0]
    assert not len(wrong), (f"{path.name}: {len(wrong)} frames look moved by a pixel or a frame, "
                            f"first frame {wrong[0] + 1} ({own[wrong[0]]:.2f} from its own, {rival[wrong[0]]:.2f})")


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
def test_cli_renders_the_rotation_experiment(volume, rotate, tmp_path, capsys, mode):
    out = tmp_path / "rotate.mp4"
    csv = EXPERIMENT / "mrhead_rotate.csv"
    assert main([str(MRHEAD), str(csv), str(out), "--fps", "30", "--interp", mode.value]) == EXIT_OK
    assert f"wrote {len(rotate)} frames" in capsys.readouterr().out
    assert_video_matches(out, expected_frames(volume, rotate, mode))


@pytest.mark.parametrize(("video", "csv", "mode"), [
    ("mrhead_sweep.mp4", "mrhead_sweep.csv", Interpolation.NEAREST),
    ("mrhead_rotate.mp4", "mrhead_rotate.csv", Interpolation.NEAREST),
    ("mrhead_rotate_trilinear.mp4", "mrhead_rotate.csv", Interpolation.TRILINEAR),
])
def test_published_videos_match_the_program(volume, video, csv, mode):
    """The MP4 files in docs/experiment are what the current code produces (re-make them after a change)."""
    assert_video_matches(EXPERIMENT / video, expected_frames(volume, parse_planes(EXPERIMENT / csv), mode))
