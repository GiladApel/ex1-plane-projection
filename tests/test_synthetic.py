import numpy as np
import pytest

from projection import synthetic


@pytest.mark.parametrize("dtype", synthetic.DTYPES)
def test_gradient_value_is_x(dtype):
    g = synthetic.gradient((10, 4, 5), dtype)
    assert g.dtype == np.dtype(dtype)
    assert all(np.all(g[x] == x) for x in range(10))


def test_gradient_rejects_values_that_do_not_fit():
    with pytest.raises(ValueError):
        synthetic.gradient((300, 2, 2), np.uint8)


@pytest.mark.parametrize("dtype", synthetic.DTYPES)
def test_sphere_center_filled_corner_empty(dtype):
    s = synthetic.sphere((21, 21, 21), dtype)
    assert s[10, 10, 10] == np.iinfo(dtype).max
    assert s[0, 0, 0] == 0


def test_checkerboard_alternates():
    c = synthetic.checkerboard((16, 16, 16), np.uint8, cell=4)
    assert c[0, 0, 0] == 0 and c[4, 0, 0] == 255 and c[4, 4, 0] == 0


def test_random_is_deterministic_per_seed():
    a = synthetic.random_volume((8, 8, 8), seed=3)
    b = synthetic.random_volume((8, 8, 8), seed=3)
    c = synthetic.random_volume((8, 8, 8), seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
