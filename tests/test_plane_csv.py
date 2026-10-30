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
    def shift(pts):
        return pts + 100.0

    plane = parse_planes(write(tmp_path, f"{TRI}\n"), to_voxel=shift)[0]
    assert plane.rails().A1.tolist() == [105.0, 100.0, 100.0]


def test_point_per_row_triangle(tmp_path):
    planes = parse_planes(write(tmp_path, "5,0,0\n0,10,0\n10,10,0\n"))
    assert len(planes) == 1 and isinstance(planes[0], Triangle)
    assert planes[0].rails().A1.tolist() == [5.0, 0.0, 0.0]


def test_point_per_row_trapezoid_with_header_and_blank_lines(tmp_path):
    planes = parse_planes(write(tmp_path, "x,y,z\n\n2,0,0\n0,10,0\n8,0,0\n10,10,0\n\n"))
    assert len(planes) == 1 and isinstance(planes[0], Trapezoid)


def test_point_per_row_matches_plane_per_row(tmp_path):
    by_point = parse_planes(write(tmp_path, "2,0,0\n0,10,0\n8,0,0\n10,10,0\n"))[0].rails()
    by_row = parse_planes(write(tmp_path, f"{TRAP}\n"))[0].rails()
    assert (by_point.H, by_point.W) == (by_row.H, by_row.W)
    assert by_point.A1.tolist() == by_row.A1.tolist() and by_point.v2.tolist() == by_row.v2.tolist()


def test_point_per_row_wrong_row_count(tmp_path):
    with pytest.raises(InvalidInputError, match=":1: .*got 5 rows"):
        parse_planes(write(tmp_path, "0,0,0\n1,1,1\n2,2,2\n3,3,3\n4,4,4\n"))
    with pytest.raises(InvalidInputError, match=":1: .*got 2 rows"):
        parse_planes(write(tmp_path, "0,0,0\n1,1,1\n"))


TRI_POINTS = "5,0,0\n0,10,0\n10,10,0\n"
TRAP_POINTS = "2,0,0\n0,10,0\n8,0,0\n10,10,0\n"


def test_point_per_row_contiguous_triangles(tmp_path):
    planes = parse_planes(write(tmp_path, TRI_POINTS * 3))
    assert len(planes) == 3 and all(isinstance(p, Triangle) for p in planes)


def test_point_per_row_contiguous_trapezoids(tmp_path):
    planes = parse_planes(write(tmp_path, TRAP_POINTS * 2))
    assert len(planes) == 2 and all(isinstance(p, Trapezoid) for p in planes)


def test_point_per_row_blank_line_blocks_may_mix_shapes(tmp_path):
    planes = parse_planes(write(tmp_path, f"x,y,z\n{TRI_POINTS}\n{TRAP_POINTS}\n\n{TRI_POINTS}"))
    assert [type(p) for p in planes] == [Triangle, Trapezoid, Triangle]


def test_point_per_row_blank_line_blocks_resolve_multiples_of_12(tmp_path):
    planes = parse_planes(write(tmp_path, "\n".join([TRAP_POINTS] * 3)))
    assert len(planes) == 3 and all(isinstance(p, Trapezoid) for p in planes)


def test_point_per_row_multiple_of_12_without_blank_lines_is_ambiguous(tmp_path):
    with pytest.raises(InvalidInputError, match="12 point rows .*blank lines"):
        parse_planes(write(tmp_path, TRAP_POINTS * 3))


def test_point_per_row_bad_block_reports_its_line(tmp_path):
    with pytest.raises(InvalidInputError, match=r":5: .*this block has 2"):
        parse_planes(write(tmp_path, f"{TRI_POINTS}\n0,0,0\n1,1,1\n"))


def test_point_per_row_invalid_later_plane_reports_its_first_line(tmp_path):
    with pytest.raises(InvalidInputError, match=r":4: .*isosceles"):
        parse_planes(write(tmp_path, f"{TRI_POINTS}0,0,0\n10,0,0\n0,6,0\n"))


def test_trailing_commas_are_ignored(tmp_path):
    assert len(parse_planes(write(tmp_path, f"{TRI},\n{TRI}, ,\n"))) == 2
    assert len(parse_planes(write(tmp_path, "5,0,0,\n0,10,0,\n10,10,0,\n"))) == 1


def test_empty_cell_inside_a_row_is_rejected(tmp_path):
    with pytest.raises(InvalidInputError, match=r":1: non-numeric"):
        parse_planes(write(tmp_path, "5,,0,0,10,0,10,10,0\n"))


def test_point_per_row_bad_row_reports_line(tmp_path):
    with pytest.raises(InvalidInputError, match=r":2: expected 3 values"):
        parse_planes(write(tmp_path, "5,0,0\n0,10\n10,10,0\n"))
    with pytest.raises(InvalidInputError, match=r":3: non-numeric"):
        parse_planes(write(tmp_path, "5,0,0\n0,10,0\n10,q,0\n"))


def test_point_per_row_invalid_plane_reports_first_line(tmp_path):
    with pytest.raises(InvalidInputError, match=r":1: .*isosceles"):
        parse_planes(write(tmp_path, "0,0,0\n10,0,0\n0,6,0\n"))


def test_point_per_row_applies_to_voxel(tmp_path):
    plane = parse_planes(write(tmp_path, "5,0,0\n0,10,0\n10,10,0\n"), to_voxel=lambda p: p + 100.0)[0]
    assert plane.rails().A1.tolist() == [105.0, 100.0, 100.0]
