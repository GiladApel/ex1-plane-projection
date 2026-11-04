import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from projection import synthetic
from projection.cli import EXIT_INTERNAL, EXIT_INVALID, EXIT_OK, main
from projection.projector import NumbaProjector, NumpyProjector

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
        # list(reader) would call len(reader), which imageio reports as infinite
        return np.stack([frame for frame in reader])  # noqa: C416
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


def test_point_per_row_csv_gives_a_single_frame(inputs):
    nrrd_path, csv_path, mp4_path = inputs
    csv_path.write_text("16,6,16\n6,26,16\n26,26,16\n")
    assert main([str(nrrd_path), str(csv_path), str(mp4_path), "--backend", "numpy"]) == EXIT_OK
    assert read_frames(mp4_path).shape[0] == 1


def test_point_per_row_csv_with_several_planes_gives_a_frame_per_plane(inputs):
    nrrd_path, csv_path, mp4_path = inputs
    rows = [",".join(CENTER_PLANE.split(",")[i:i + 3]) for i in (0, 3, 6)]
    trapezoid = [",".join(TRAPEZOID.split(",")[i:i + 3]) for i in (0, 3, 6, 9)]
    csv_path.write_text("\n".join(rows + [""] + trapezoid + [""] + rows) + "\n")
    assert main([str(nrrd_path), str(csv_path), str(mp4_path), "--backend", "numpy"]) == EXIT_OK
    assert read_frames(mp4_path).shape[0] == 3


def test_trilinear_on_numpy_warns_but_nearest_does_not(inputs, capsys):
    nrrd_path, csv_path, mp4_path = inputs
    args = [str(nrrd_path), str(csv_path), str(mp4_path), "--interp", "trilinear", "--backend", "numpy"]
    assert main(args) == EXIT_OK
    assert "warning: trilinear sampling on the NumPy backend is slow" in capsys.readouterr().err
    assert main([str(nrrd_path), str(csv_path), str(mp4_path), "--backend", "numpy"]) == EXIT_OK
    assert "warning" not in capsys.readouterr().err


def test_trilinear_runs(inputs):
    nrrd_path, csv_path, mp4_path = inputs
    args = [str(nrrd_path), str(csv_path), str(mp4_path), "--interp", "trilinear", "--backend", "numpy"]
    assert main(args) == EXIT_OK
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


def is_error_frame(frame):
    gray = frame[:, :, 0].astype(int)
    return 40 < np.median(gray) < 90 and gray.max() > 200


def test_bad_rows_become_error_frames_and_the_run_goes_on(inputs, capsys):
    nrrd_path, csv_path, mp4_path = inputs
    collinear = "0,0,0, 5,5,5, 10,10,10"
    csv_path.write_text("\n".join([CENTER_PLANE, "16,6,x, 6,26,16, 26,26,16", collinear, SHIFTED_PLANE]) + "\n")
    assert main([str(nrrd_path), str(csv_path), str(mp4_path), "--backend", "numpy"]) == EXIT_OK
    out, err = capsys.readouterr()
    assert "planes.csv:2: non-numeric value; frame 2 is an error frame" in err
    assert "planes.csv:3:" in err and "collinear" in err
    assert "2 of 4 frames are error frames:" in out
    assert "frame 2, line 2: non-numeric value" in out and "frame 3, line 3:" in out
    frames = read_frames(mp4_path)
    assert frames.shape[0] == 4
    assert [is_error_frame(f) for f in frames] == [False, True, True, False]
    assert frames[0].mean() > 20 and frames[3].mean() > 20


def test_a_file_of_bad_rows_still_gives_one_frame_per_row(inputs, capsys):
    nrrd_path, csv_path, mp4_path = inputs
    csv_path.write_text("1,2,3,4,5,6,7,8,9,10\n0,0,0,5,5,5,10,10,10\n")
    assert main([str(nrrd_path), str(csv_path), str(mp4_path), "--backend", "numpy"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "no plane projected" in out and "2 of 2 frames are error frames" in out
    frames = read_frames(mp4_path)
    assert frames.shape[:3] == (2, 64, 64) and all(is_error_frame(f) for f in frames)


def test_problems_with_the_whole_file_still_stop_the_run(inputs, capsys):
    nrrd_path, csv_path, mp4_path = inputs
    csv_path.write_text("0,0,0\n1,1,1\n2,2,2\n3,3,3\n4,4,4\n")
    assert main([str(nrrd_path), str(csv_path), str(mp4_path)]) == EXIT_INVALID
    assert "got 5 rows" in capsys.readouterr().err
    assert not mp4_path.exists()


def test_world_coordinates_without_metadata_stop_the_run(inputs, capsys):
    nrrd_path, csv_path, mp4_path = inputs
    assert main([str(nrrd_path), str(csv_path), str(mp4_path), "--world"]) == EXIT_INVALID
    assert "--world needs 'space origin'" in capsys.readouterr().err
    assert not mp4_path.exists()


def test_a_failing_projection_gives_an_error_frame_and_exit_code_1(inputs, capsys, monkeypatch):
    import projection.cli as cli

    class Failing(NumpyProjector):
        calls = 0

        def project(self, volume, plane, out=None):
            Failing.calls += 1
            if Failing.calls == 3:          # call 1 is the warm-up, call 3 is the second plane
                raise RuntimeError("boom")
            return super().project(volume, plane, out)

    monkeypatch.setattr(cli, "make_projector", lambda backend, interpolation: Failing(interpolation))
    nrrd_path, csv_path, mp4_path = inputs
    assert main([str(nrrd_path), str(csv_path), str(mp4_path)]) == EXIT_INTERNAL
    out, err = capsys.readouterr()
    assert "internal error while projecting (RuntimeError: boom)" in err
    assert "1 of 4 frames are error frames" in out
    frames = read_frames(mp4_path)
    assert frames.shape[0] == 4 and is_error_frame(frames[1])


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


def test_non_isosceles_plane_is_projected_with_a_warning(inputs, capsys):
    nrrd_path, csv_path, mp4_path = inputs
    csv_path.write_text(f"{CENTER_PLANE}\n16,6,16, 6,26,16, 26,20,16\n")
    assert main([str(nrrd_path), str(csv_path), str(mp4_path), "--backend", "numpy"]) == EXIT_OK
    err = capsys.readouterr().err
    assert "warning: " in err and "planes.csv:2: plane is not isosceles" in err
    assert read_frames(mp4_path).shape[0] == 2
    assert main([str(nrrd_path), str(csv_path), str(mp4_path), "--force", "--backend", "numpy"]) == EXIT_OK
    assert "warning" not in capsys.readouterr().err


def test_lecturer_csv_format(tmp_path, capsys):
    """The lecturer's example sheet: a point count column, a header, empty cells for a triangle's fourth point,
    triangles and trapezoids mixed, and one triangle that is not isosceles (row 4)."""
    nrrd_path = tmp_path / "volume.nrrd"
    synthetic.save_nrrd(nrrd_path, synthetic.gradient((500, 8, 8), np.uint16))
    csv_path = tmp_path / "lecturer.csv"
    csv_path.write_text(
        "Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4\n"
        "3,200,200,0,0,200,400,400,200,400,,,\n"
        "4,200,0,0,200,0,500,200,400,0,200,400,500\n"
        "3,200,200,-30,-50,200,400,400,250,400,,,\n"
        "4,200,50,0,200,0,500,200,350,0,200,400,500\n"
    )
    mp4_path = tmp_path / "out.mp4"
    assert main([str(nrrd_path), str(csv_path), str(mp4_path), "--backend", "numpy"]) == EXIT_OK
    out, err = capsys.readouterr()
    assert "wrote 4 frames" in out
    assert err.count("warning: ") == 1 and "lecturer.csv:4: plane is not isosceles" in err
    assert read_frames(mp4_path).shape[0] == 4


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
