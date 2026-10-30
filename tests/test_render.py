import numpy as np
import pytest

from projection import synthetic
from projection.geometry import Trapezoid, Triangle
from projection.render import FrameComposer, IntensityWindow
from projection.video import Mp4Writer
from projection.volume import Volume


def test_window_percentiles_of_a_gradient():
    window = IntensityWindow.from_volume(Volume(synthetic.gradient((101, 4, 4), np.uint8)))
    assert window.lo == pytest.approx(1.0) and window.hi == pytest.approx(99.0)


def test_window_maps_lo_hi_and_clips():
    window = IntensityWindow(100.0, 200.0)
    image = np.array([[0, 100, 150, 200, 60000]], dtype=np.uint16)
    assert window.apply(image).tolist() == [[0, 0, 128, 255, 255]]


def test_window_on_constant_volume_does_not_divide_by_zero():
    window = IntensityWindow.from_volume(Volume(np.full((4, 4, 4), 7, dtype=np.uint8)))
    assert window.hi > window.lo
    assert window.apply(np.full((2, 2), 7, dtype=np.uint8)).tolist() == [[0, 0], [0, 0]]


def test_window_ignores_large_background_at_the_minimum():
    data = np.zeros((10, 10, 10), dtype=np.uint16)
    data[:, :, 5:] = synthetic.gradient((10, 10, 5), np.uint16) + 2000
    window = IntensityWindow.from_volume(Volume(data))
    assert window.lo >= 2000


def test_window_keeps_a_small_minimum_share():
    data = synthetic.gradient((100, 10, 10), np.uint8)
    data[:3] = 0
    window = IntensityWindow.from_volume(Volume(data))
    assert window.lo == 0.0


def test_window_on_a_binary_mask_maps_the_object_to_white():
    window = IntensityWindow.from_volume(Volume(synthetic.sphere((20, 20, 20), np.uint8)))
    assert (window.lo, window.hi) == (0.0, 255.0)


def test_window_subsamples_large_volumes():
    volume = Volume(synthetic.gradient((255, 100, 100), np.uint8))
    window = IntensityWindow.from_volume(volume)
    assert 0 <= window.lo < window.hi <= 254


def test_canvas_is_largest_plane_rounded_up_to_even():
    planes = [Triangle((5, 0, 0), (0, 10, 0), (10, 10, 0)), Trapezoid((2, 0, 0), (0, 14, 0), (9, 0, 0), (11, 14, 0))]
    composer = FrameComposer.for_planes(planes)
    assert max(p.shape[0] for p in planes) <= composer.height and composer.height % 2 == 0
    assert max(p.shape[1] for p in planes) <= composer.width and composer.width % 2 == 0


def test_compose_centers_and_clears_between_calls():
    composer = FrameComposer(8, 10)
    out = composer.compose(np.full((4, 6), 9, dtype=np.uint8))
    assert out.shape == (8, 10) and out.sum() == 9 * 24
    assert np.all(out[2:6, 2:8] == 9) and out[:2].sum() == 0 and out[:, :2].sum() == 0
    out = composer.compose(np.full((2, 2), 5, dtype=np.uint8))
    assert out.sum() == 5 * 4


def test_compose_rejects_oversize_image():
    with pytest.raises(ValueError, match="canvas"):
        FrameComposer(4, 4).compose(np.zeros((5, 4), dtype=np.uint8))


def test_writer_rejects_odd_size_and_bad_frames(tmp_path):
    with pytest.raises(ValueError, match="even"):
        Mp4Writer(tmp_path / "a.mp4", 10, 5, 6)
    with Mp4Writer(tmp_path / "a.mp4", 10, 4, 6) as writer:
        with pytest.raises(ValueError, match="expected uint8"):
            writer.append(np.zeros((4, 4), dtype=np.uint8))
        writer.append(np.zeros((4, 6), dtype=np.uint8))
    assert writer.frames == 1
