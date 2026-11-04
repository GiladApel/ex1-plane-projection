"""The Streamlit demo (demo/app.py): its step-by-step explanation must give the program's own values, and both tabs
must render. Needs the demo's packages (pip install -r demo/requirements.txt); CI runs it in its own job."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

from projection import make_projector, synthetic  # noqa: E402
from projection.sampler import Interpolation, sample_nearest, sample_trilinear  # noqa: E402
from projection.volume import Volume  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "demo" / "app.py"
spec = importlib.util.spec_from_file_location("demo_app", APP)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

SAMPLERS = {Interpolation.NEAREST: sample_nearest, Interpolation.TRILINEAR: sample_trilinear}


def explained_image(volume, plane, mode):
    """The image rebuilt only from the demo's walk(): what the explanation says each pixel is."""
    rails = plane.rails()
    image = np.zeros(plane.shape, dtype=volume.dtype)
    for y, row in enumerate(app.walk(rails)):
        for _, x, p in row["samples"]:
            if 0 <= x < rails.W:
                image[y, x] = SAMPLERS[mode](volume.data, *p)
    return image


@pytest.mark.parametrize("mode", list(Interpolation), ids=lambda m: m.value)
def test_the_explanation_gives_the_programs_pixels(mode):
    volume = Volume(synthetic.random_volume((16, 16, 16), seed=3))
    rng = np.random.default_rng(5)
    projector = make_projector("reference", mode)
    for _ in range(40):
        shape = rng.choice(["Triangle", "Trapezoid"])
        angles = rng.uniform(-180, 180, 3)
        size = rng.uniform(2, 14, 3)
        plane = app.plane_from_points(app.plane_points(shape, [7.5] * 3, app.rotation(*angles), *size), force=True)
        np.testing.assert_array_equal(explained_image(volume, plane, mode), projector.project(volume, plane))


def test_trilinear_corner_weights_add_up_to_one():
    corners = app.trilinear_corners(app.teaching_volume().data, np.array([3.25, 7.5, 0.875]))
    assert len(corners) == 8
    assert sum(w for _, w, _ in corners) == pytest.approx(1.0)


def run_app(**state):
    test = AppTest.from_file(str(APP), default_timeout=120)
    test.session_state["teach_pixel"] = state.pop("pixel", (0, 0))
    test.run()
    assert not test.exception, test.exception
    return test


def test_both_tabs_render_and_the_check_passes():
    test = run_app(pixel=(4, 5))
    assert [t.label for t in test.tabs] == ["Explore", "How it works"]
    assert any("Pixel y = 4, x = 5" in m.value for m in test.markdown)
    assert test.success and "reference backend" in test.success[0].value
    assert not test.error


@pytest.mark.parametrize("shape", ["Triangle", "Trapezoid"])
@pytest.mark.parametrize("mode", ["nearest", "trilinear"])
def test_every_pixel_of_the_teaching_image_explains_itself(shape, mode):
    test = run_app()
    test.radio(key="teach_shape").set_value(shape).run()
    test.radio(key="teach_mode").set_value(mode).run()
    height = int(test.markdown[[i for i, m in enumerate(test.markdown) if "The image E" in m.value][0]]
                 .value.split("H = ")[1].split()[0])
    for y in (0, height // 2, height - 1):
        test.session_state["teach_pixel"] = (y, 0)
        test.run()
        assert not test.exception and not test.error and test.success, (shape, mode, y)


def test_a_pixel_beyond_a_smaller_image_is_clamped():
    test = run_app(pixel=(99, 99))
    assert any("Pixel y = " in m.value for m in test.markdown)
    assert test.success
