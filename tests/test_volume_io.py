from pathlib import Path

import nrrd
import numpy as np
import pytest

from projection import synthetic
from projection.errors import InvalidInputError
from projection.io_nrrd import load_nrrd
from projection.volume import Volume

MRHEAD = Path(__file__).resolve().parent.parent / "data" / "MRHead.nrrd"


def save(tmp_path, data, header=None):
    path = tmp_path / "v.nrrd"
    nrrd.write(str(path), data, header or {}, index_order="F")
    return path


@pytest.mark.parametrize("dtype", synthetic.DTYPES)
def test_roundtrip_keeps_axis_order_and_dtype(tmp_path, dtype):
    data = synthetic.gradient((6, 7, 8), dtype)
    vol = load_nrrd(save(tmp_path, data))
    assert vol.shape == (6, 7, 8)
    assert vol.dtype == np.dtype(dtype)
    assert vol.data[3, 0, 0] == 3 and vol.data[3, 6, 7] == 3
    assert vol.value_offset == 0


def test_axis_order_distinguishes_axes(tmp_path):
    data = np.zeros((4, 5, 6), dtype=np.uint8)
    data[3, 1, 2] = 200
    vol = load_nrrd(save(tmp_path, data))
    assert vol.data[3, 1, 2] == 200


def test_volume_is_contiguous_and_read_only(tmp_path):
    vol = load_nrrd(save(tmp_path, synthetic.random_volume((5, 6, 7))))
    assert vol.data.flags.c_contiguous
    assert not vol.data.flags.writeable
    assert vol.flat.shape == (5 * 6 * 7,)
    assert vol.flat[(2 * 6 + 3) * 7 + 4] == vol.data[2, 3, 4]


def test_signed_non_negative_is_cast_without_offset(tmp_path):
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    vol = load_nrrd(save(tmp_path, data))
    assert vol.dtype == np.dtype(np.uint16) and vol.value_offset == 0
    assert np.array_equal(vol.data, data)


def test_signed_negative_is_shifted(tmp_path):
    data = np.array([-1000, 0, 500, 3000], dtype=np.int16).reshape(1, 2, 2)
    vol = load_nrrd(save(tmp_path, data))
    assert vol.dtype == np.dtype(np.uint16)
    assert vol.value_offset == -1000
    assert vol.data.min() == 0 and vol.data.max() == 4000


def test_rejects_4d(tmp_path):
    with pytest.raises(InvalidInputError, match="3D"):
        load_nrrd(save(tmp_path, np.zeros((2, 3, 4, 3), dtype=np.uint8)))


def test_rejects_float(tmp_path):
    with pytest.raises(InvalidInputError, match="unsupported dtype"):
        load_nrrd(save(tmp_path, np.zeros((2, 3, 4), dtype=np.float32)))


def test_rejects_missing_file(tmp_path):
    with pytest.raises(InvalidInputError, match="not found"):
        load_nrrd(tmp_path / "nope.nrrd")


def test_rejects_garbage_file(tmp_path):
    path = tmp_path / "bad.nrrd"
    path.write_bytes(b"this is not an nrrd file")
    with pytest.raises(InvalidInputError, match="cannot read"):
        load_nrrd(path)


def test_rejects_oversize_dimension():
    with pytest.raises(InvalidInputError, match="dimension"):
        Volume(np.zeros((1024, 2, 2), dtype=np.uint8))


def test_volume_rejects_bad_dtype_and_rank():
    with pytest.raises(InvalidInputError, match="dtype"):
        Volume(np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(InvalidInputError, match="3D"):
        Volume(np.zeros((2, 2), dtype=np.uint8))


def test_world_to_voxel_roundtrip(tmp_path):
    header = {
        "space": "left-posterior-superior",
        "space origin": [10.0, 20.0, 30.0],
        "space directions": [[2.0, 0, 0], [0, 3.0, 0], [0, 0, 4.0]],
    }
    vol = load_nrrd(save(tmp_path, np.zeros((4, 4, 4), dtype=np.uint8), header))
    idx = vol.world_to_voxel(np.array([[12.0, 26.0, 38.0]]))
    np.testing.assert_allclose(idx, [[1.0, 2.0, 2.0]])


def test_world_to_voxel_requires_metadata():
    with pytest.raises(InvalidInputError, match="spatial metadata"):
        Volume(np.zeros((2, 2, 2), dtype=np.uint8)).world_to_voxel(np.zeros((1, 3)))


@pytest.mark.skipif(not MRHEAD.is_file(), reason="data/MRHead.nrrd not downloaded")
def test_loads_mrhead():
    vol = load_nrrd(MRHEAD)
    assert vol.shape == (256, 256, 130)
    assert vol.dtype == np.dtype(np.uint16)
    assert vol.value_offset == 0
    assert vol.origin is not None and vol.directions.shape == (3, 3)
    idx = vol.world_to_voxel(vol.origin[None, :])
    np.testing.assert_allclose(idx, [[0, 0, 0]], atol=1e-9)
