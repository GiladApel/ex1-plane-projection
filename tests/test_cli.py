import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from projection import synthetic
from projection.cli import EXIT_INVALID, EXIT_OK, main
from projection.projector import NumbaProjector

try:
    NumbaProjector()
    BACKENDS = ["numpy", "numba"]
except ImportError:
    BACKENDS = ["numpy"]

ROOT = Path(__file__).resolve().parent.parent
CENTER_PLANE = "16,6,16, 6,26,16, 26,26,16"
SHIFTED_PLANE = "16,6,10, 6,26,10, 26,26,10"
OUTSIDE_PLANE = "216,206,16, 206,226,16, 226,226,16"
LARGE_TRIANGLE = "16,2,20, 4,30,20, 28,30,20"
TRAPEZOID = "11,6,20, 6,26,20, 21,6,20, 26,26,20"


@pytest.fixture
def inputs(tmp_path):
    nrrd_path = tmp_path / "sphere.nrrd"
    synthetic.save_nrrd(nrrd_path, synthetic.sphere((32, 32, 32), np.uint8))
    csv_path = tmp_path / "planes.csv"
    csv_path.write_text("\n".join([CENTER_PLANE, SHIFTED_PLANE, OUTSIDE_PLANE, LARGE_TRIANGLE]) + "\n")
    return nrrd_path, csv_path, tmp_path / "out.mp4"


def read_frames(path):
    reader = imageio.get_reader(str(path))
    try:
        return np.stack([frame for frame in reader])
    finally:
        reader.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_demo_writes_one_frame_per_row_with_constant_size(inputs, capsys, backend):
    nrrd_path, csv_path, mp4_path = inputs
    assert main([str(nrrd_path), str(csv_path), str(mp4_path), "--backend", backend]) == EXIT_OK
    assert "wrote 4 frames" in capsys.readouterr().out

    frames = read_frames(mp4_path)
    assert frames.shape[0] == 4
    assert frames.shape[1] % 2 == 0 and frames.shape[2] % 2 == 0
    brightness = frames.mean(axis=(1, 2, 3))
    assert brightness[0] > 20           # plane through the sphere's center
    assert brightness[2] < 1            # plane entirely outside the volume: black frame


def test_backends_produce_identical_videos(inputs, tmp_path):
    if len(BACKENDS) < 2:
        pytest.skip("numba not installed")
    nrrd_path, csv_path, _ = inputs
    decoded = []
    for backend in BACKENDS:
        out = tmp_path / f"{backend}.mp4"
        assert main([str(nrrd_path), str(csv_path), str(out), "--backend", backend]) == EXIT_OK
        decoded.append(read_frames(out))
    assert np.array_equal(decoded[0], decoded[1])


def test_trapezoid_rows(inputs):
    nrrd_path, csv_path, mp4_path = inputs
    csv_path.write_text(f"{TRAPEZOID}\n{TRAPEZOID}\n")
    assert main([str(nrrd_path), str(csv_path), str(mp4_path), "--backend", "numpy"]) == EXIT_OK
    assert read_frames(mp4_path).shape[0] == 2


def test_trilinear_runs(inputs):
    nrrd_path, csv_path, mp4_path = inputs
    assert main([str(nrrd_path), str(csv_path), str(mp4_path), "--interp", "trilinear", "--backend", "numpy"]) == EXIT_OK
    assert read_frames(mp4_path).shape[0] == 4


def test_world_coordinates_are_converted(tmp_path):
    nrrd_path = tmp_path / "v.nrrd"
    header = {"space": "left-posterior-superior", "space origin": [100.0, 0.0, 0.0],
              "space directions": [[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]]}
    import nrrd
    nrrd.write(str(nrrd_path), synthetic.sphere((32, 32, 32), np.uint8), header, index_order="F")
    csv_path = tmp_path / "w.csv"
    csv_path.write_text("132,12,32, 112,52,32, 152,52,32\n")
    out = tmp_path / "w.mp4"
    assert main([str(nrrd_path), str(csv_path), str(out), "--world", "--backend", "numpy"]) == EXIT_OK
    assert read_frames(out).mean() > 20


def test_missing_nrrd_is_reported(inputs, capsys):
    _, csv_path, mp4_path = inputs
    assert main(["nope.nrrd", str(csv_path), str(mp4_path)]) == EXIT_INVALID
    assert "error: NRRD file not found" in capsys.readouterr().err
    assert not mp4_path.exists()


def test_bad_csv_reports_line_and_writes_nothing(inputs, capsys):
    nrrd_path, csv_path, mp4_path = inputs
    csv_path.write_text(CENTER_PLANE + "\n1,2,3\n")
    assert main([str(nrrd_path), str(csv_path), str(mp4_path)]) == EXIT_INVALID
    assert ":2:" in capsys.readouterr().err
    assert not mp4_path.exists()


def test_non_3d_nrrd_is_rejected(tmp_path, inputs, capsys):
    _, csv_path, mp4_path = inputs
    flat = tmp_path / "flat.nrrd"
    synthetic.save_nrrd(flat, np.zeros((4, 5, 6, 3), dtype=np.uint8))
    assert main([str(flat), str(csv_path), str(mp4_path)]) == EXIT_INVALID
    assert "3D" in capsys.readouterr().err


def test_missing_output_directory(inputs, tmp_path, capsys):
    nrrd_path, csv_path, _ = inputs
    assert main([str(nrrd_path), str(csv_path), str(tmp_path / "no" / "out.mp4")]) == EXIT_INVALID
    assert "output directory" in capsys.readouterr().err


def test_non_isosceles_plane_needs_force(inputs, capsys):
    nrrd_path, csv_path, mp4_path = inputs
    csv_path.write_text("16,6,16, 6,26,16, 26,20,16\n")
    assert main([str(nrrd_path), str(csv_path), str(mp4_path)]) == EXIT_INVALID
    assert "isosceles" in capsys.readouterr().err
    assert main([str(nrrd_path), str(csv_path), str(mp4_path), "--force", "--backend", "numpy"]) == EXIT_OK


def test_bad_arguments_exit_with_code_2(inputs):
    nrrd_path, csv_path, mp4_path = inputs
    with pytest.raises(SystemExit) as exc:
        main([str(nrrd_path), str(csv_path), str(mp4_path), "--backend", "gpu"])
    assert exc.value.code == 2
    assert main([str(nrrd_path), str(csv_path), str(mp4_path), "--fps", "0"]) == EXIT_INVALID


def test_script_runs_as_documented(inputs):
    nrrd_path, csv_path, mp4_path = inputs
    result = subprocess.run(
        [sys.executable, str(ROOT / "Ex0.py"), str(nrrd_path), str(csv_path), str(mp4_path)],
        capture_output=True, text=True, cwd=mp4_path.parent,
    )
    assert result.returncode == 0, result.stderr
    assert read_frames(mp4_path).shape[0] == 4
