import pytest

from projection.errors import InvalidInputError
from projection.geometry import Trapezoid, Triangle
from projection.plane_csv import parse_planes

TRI = "5,0,0,0,10,0,10,10,0"
TRAP = "2,0,0,0,10,0,8,0,0,10,10,0"


def write(tmp_path, text):
    path = tmp_path / "planes.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_triangles(tmp_path):
    planes = parse_planes(write(tmp_path, f"{TRI}\n{TRI}\n"))
    assert len(planes) == 2
    assert all(isinstance(p, Triangle) for p in planes)


def test_trapezoids(tmp_path):
    planes = parse_planes(write(tmp_path, f"{TRAP}\n"))
    assert isinstance(planes[0], Trapezoid)


def test_header_is_skipped(tmp_path):
    planes = parse_planes(write(tmp_path, f"ax,ay,az,bx,by,bz,cx,cy,cz\n{TRI}\n"))
    assert len(planes) == 1


def test_blank_lines_and_spaces_are_ignored(tmp_path):
    planes = parse_planes(write(tmp_path, f"\n 5, 0, 0, 0, 10, 0, 10, 10, 0 \n\n{TRI}\n"))
    assert len(planes) == 2


def test_utf8_bom(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbf" + TRI.encode())
    assert len(parse_planes(path)) == 1


def test_bad_value_reports_line_number(tmp_path):
    with pytest.raises(InvalidInputError, match=r"planes\.csv:3: non-numeric"):
        parse_planes(write(tmp_path, f"{TRI}\n{TRI}\n5,0,x,0,10,0,10,10,0\n"))


def test_wrong_column_count_reports_line_number(tmp_path):
    with pytest.raises(InvalidInputError, match=r":2: expected 9 or 12"):
        parse_planes(write(tmp_path, f"{TRI}\n1,2,3,4\n"))


def test_mixed_point_counts_rejected(tmp_path):
    with pytest.raises(InvalidInputError, match=r":2: expected 9 values"):
        parse_planes(write(tmp_path, f"{TRI}\n{TRAP}\n"))


def test_invalid_plane_reports_line_number(tmp_path):
    with pytest.raises(InvalidInputError, match=r":2: .*isosceles"):
        parse_planes(write(tmp_path, f"{TRI}\n0,0,0,10,0,0,0,6,0\n"))


def test_force_skips_isosceles_check(tmp_path):
    assert len(parse_planes(write(tmp_path, "0,0,0,10,0,0,0,6,0\n"), force=True)) == 1


def test_header_only_file_rejected(tmp_path):
    with pytest.raises(InvalidInputError, match="no planes"):
        parse_planes(write(tmp_path, "a,b,c\n"))


def test_empty_file_rejected(tmp_path):
    with pytest.raises(InvalidInputError, match="no planes"):
        parse_planes(write(tmp_path, ""))


def test_missing_file(tmp_path):
    with pytest.raises(InvalidInputError, match="not found"):
        parse_planes(tmp_path / "nope.csv")


def test_to_voxel_is_applied(tmp_path):
    shift = lambda pts: pts + 100.0
    plane = parse_planes(write(tmp_path, f"{TRI}\n"), to_voxel=shift)[0]
    assert plane.rails().A1.tolist() == [105.0, 100.0, 100.0]
