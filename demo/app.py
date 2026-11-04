"""Interactive demo: place a triangle or trapezoid in a volume and see its projection.

streamlit run demo/app.py            # pip install -r demo/requirements.txt

The sliders set the plane's center, orientation and size; the page shows where the plane lies in the volume,
the projected image with its size and timing, and the plane as a row of the lecturer's CSV format, ready to paste
into a file for Ex0.py. MRHead is downloaded on first use (as scripts/download_data.py does).
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import download_data  # noqa: E402

from projection import InvalidInputError, Volume, load_nrrd, make_projector, plane_from_points, synthetic  # noqa: E402
from projection.render import IntensityWindow  # noqa: E402
from projection.sampler import Interpolation  # noqa: E402

VOLUMES = ["MRHead (MRI of a head, 256 x 256 x 130)", "Synthetic sphere and cube (no download)"]


@st.cache_resource(show_spinner="Loading the volume...")
def load_volume(name: str) -> tuple[Volume, IntensityWindow]:
    if name == VOLUMES[0]:
        download_data.DATA_DIR.mkdir(exist_ok=True)
        if not download_data.fetch("MRHead"):
            raise RuntimeError("MRHead could not be downloaded")
        volume = load_nrrd(download_data.DATA_DIR / "MRHead.nrrd")
    else:
        data = synthetic.sphere((160, 160, 160)) // 2
        data[100:140, 20:60, 30:130] = 255
        volume = Volume(data)
    return volume, IntensityWindow.from_volume(volume)


@st.cache_resource(show_spinner="Preparing the projection engine (Numba compiles on first use)...")
def engine(backend: str, interpolation: str):
    return make_projector(backend, Interpolation(interpolation))


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Rotation matrix for angles in degrees: roll about x, then pitch about y, then yaw about z."""
    a, b, c = np.radians([yaw, pitch, roll])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    return rz @ ry @ rx


def plane_points(shape: str, center, frame: np.ndarray, height: float, base: float, top: float) -> np.ndarray:
    """Points of an isosceles triangle (a, b, c) or trapezoid (a', b, a'', c) centered on center.
    Column 0 of frame is the base direction, column 1 points from the top to the base."""
    center = np.asarray(center, dtype=np.float64)
    across, down = frame[:, 0], frame[:, 1]
    top_mid, base_mid = center - height / 2 * down, center + height / 2 * down
    b, c = base_mid - base / 2 * across, base_mid + base / 2 * across
    if shape == "Triangle":
        return np.array([top_mid, b, c])
    return np.array([top_mid - top / 2 * across, b, top_mid + top / 2 * across, c])


def sketch(volume_shape, points: np.ndarray):
    """The volume's box and the plane, in voxel coordinates."""
    figure = plt.figure(figsize=(4.2, 4.2))
    axis = figure.add_subplot(projection="3d")
    sx, sy, sz = (n - 0.5 for n in volume_shape)
    low = -0.5
    corners = np.array([[x, y, z] for x in (low, sx) for y in (low, sy) for z in (low, sz)])
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(corners[i] != corners[j]) == 1:
                axis.plot(*zip(corners[i], corners[j], strict=True), color="#8a8f98", linewidth=0.8)
    outline = points if len(points) == 3 else points[[0, 2, 3, 1]]
    closed = np.vstack([outline, outline[:1]])
    axis.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="#D55E00", linewidth=2)
    axis.add_collection3d(Poly3DCollection([outline], facecolor="#D55E00", alpha=0.25))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    extent = np.vstack([corners, points])
    low_corner, high_corner = extent.min(axis=0), extent.max(axis=0)
    axis.set_xlim(low_corner[0], high_corner[0])
    axis.set_ylim(low_corner[1], high_corner[1])
    axis.set_zlim(low_corner[2], high_corner[2])
    axis.set_box_aspect(tuple(high_corner - low_corner))   # true proportions, voxels are cubes
    figure.tight_layout()
    return figure


def csv_row(points: np.ndarray) -> str:
    """The plane as a row of the lecturer's format: number of points, then x, y, z of each point."""
    numbers = ",".join(f"{v:.3f}".rstrip("0").rstrip(".") for v in points.reshape(-1))
    return f"{len(points)},{numbers}" + (",,," if len(points) == 3 else "")


def main() -> None:
    st.set_page_config(page_title="Plane projection demo", layout="wide")
    st.title("Plane projection of a 3D volume")
    st.caption("Ex1, Introduction to Software Engineering. Move the plane; the image is its projection.")

    with st.sidebar:
        name = st.selectbox("Volume", VOLUMES)
        volume, window = load_volume(name)
        sx, sy, sz = volume.shape
        shape = st.radio("Plane", ["Triangle", "Trapezoid"], horizontal=True)
        st.subheader("Position")
        cx = st.slider("center x", 0.0, float(sx - 1), (sx - 1) / 2)
        cy = st.slider("center y", 0.0, float(sy - 1), (sy - 1) / 2)
        cz = st.slider("center z", 0.0, float(sz - 1), (sz - 1) / 2)
        st.subheader("Orientation (degrees)")
        yaw = st.slider("turn about z", -180, 180, 0)
        pitch = st.slider("tilt about y", -90, 90, 0)
        roll = st.slider("tilt about x", -180, 180, 90)
        st.subheader("Size (voxels)")
        largest = float(max(volume.shape))
        height = st.slider("height", 2.0, 1.5 * largest, 0.8 * largest)
        base = st.slider("base", 2.0, 1.5 * largest, 0.8 * largest)
        top = st.slider("top (trapezoid)", 0.0, 1.5 * largest, 0.4 * largest) if shape == "Trapezoid" else 0.0
        st.subheader("Engine")
        interpolation = st.radio("Sampling", ["nearest", "trilinear"], horizontal=True)
        backend = st.selectbox("Backend", ["auto", "numba", "numpy", "reference"])

    points = plane_points(shape, (cx, cy, cz), rotation(yaw, pitch, roll), height, base, top)
    left, right = st.columns([1, 1.4])
    with left:
        st.pyplot(sketch(volume.shape, points), clear_figure=True)
        st.markdown("**The plane as a CSV row** (the lecturer's format):")
        st.code(csv_row(points), language="text")
    with right:
        try:
            plane = plane_from_points(points, force=True)
            projector = engine(backend, interpolation)
            projector.project(volume, plane)
            start = time.perf_counter()
            image = projector.project(volume, plane)
            elapsed = time.perf_counter() - start
        except InvalidInputError as exc:
            st.error(f"This plane cannot be projected: {exc}")
            return
        gray = window.apply(image)
        zoom = max(1, 480 // max(gray.shape))
        st.image(np.kron(gray, np.ones((zoom, zoom), dtype=np.uint8)), clamp=True)
        h, w = image.shape
        st.markdown(
            f"**{h} x {w} pixels** · {type(projector).__name__}, {interpolation} · "
            f"{elapsed * 1000:.2f} ms ({1 / elapsed:.0f} planes per second)"
        )


main()
