"""Interactive demo: place a triangle or trapezoid in a volume and see its projection.

streamlit run demo/app.py            # pip install -r demo/requirements.txt

Two tabs:
- Explore: the sliders set the plane's center, orientation and size; the page shows where the plane lies in the
  volume, the projected image with its size and timing, and the plane as a row of the lecturer's CSV format, ready
  to paste into a file for Ex0.py. MRHead is downloaded on first use (as scripts/download_data.py does).
- How it works: a small plane in a 16^3 volume, with the rails, rows and sample points drawn in 3D. Clicking a
  pixel of the image shows every step of the algorithm for that pixel with its numbers, and checks the result
  against the program's reference backend.
"""
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import download_data  # noqa: E402

from projection import InvalidInputError, Volume, load_nrrd, make_projector, plane_from_points, synthetic  # noqa: E402
from projection.render import IntensityWindow  # noqa: E402
from projection.sampler import HALF, SNAP, Interpolation, sample_nearest, sample_trilinear  # noqa: E402

VOLUMES = ["MRHead, MRI, 256x256x130", "Sphere and cube, 160^3"]
TEACH_SIDE = 16
RAIL_1, RAIL_2, CHOSEN = "#0072B2", "#E69F00", "#D55E00"


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


@st.cache_resource
def teaching_volume() -> Volume:
    """Brightest at the center and 22 gray levels darker per voxel of distance, so neighboring voxels differ."""
    grids = np.ogrid[tuple(slice(0, TEACH_SIDE) for _ in range(3))]
    center = (TEACH_SIDE - 1) / 2
    distance = np.sqrt(sum((g - center) ** 2 for g in grids))
    return Volume(np.clip(np.round(250 - 22 * distance), 0, 255).astype(np.uint8))


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


def walk(rails) -> list[dict]:
    """Every row of the walk with its samples, computed with the same floating-point steps as ReferenceProjector,
    so each pixel's value can be shown together with the numbers that produced it.

    Pseudo-code (one dict per row y):
        a1 = A1 + y*v1;  a2 = A2 + y*v2;  d12 = |a2 - a1|;  dp = (a2 - a1)/d12;  off = (W-1)/2 - d12/2
        samples: for x1 in 0 .. floor(d12 + SNAP): x = floor(off + x1 + HALF), p = a1 + x1*dp
    """
    rows = []
    for y in range(rails.H):
        a1 = rails.A1 + y * rails.v1
        a2 = rails.A2 + y * rails.v2
        d = a2 - a1
        d12 = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        dp = d / d12 if d12 > 0 else d
        off = (rails.W - 1) / 2.0 - d12 / 2.0
        samples = [(x1, math.floor(off + x1 + HALF), a1 + x1 * dp) for x1 in range(math.floor(d12 + SNAP) + 1)]
        rows.append({"a1": a1, "a2": a2, "d12": d12, "dp": dp, "off": off, "samples": samples})
    return rows


def trilinear_corners(data: np.ndarray, p: np.ndarray) -> list[tuple[tuple[int, int, int], float, int]]:
    """The 8 voxels around p with their weights and values (0 outside the volume), in the sampler's order."""
    base = np.floor(p).astype(int)
    f = p - base
    corners = []
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                index = (int(base[0]) + di, int(base[1]) + dj, int(base[2]) + dk)
                weight = ((f[0] if di else 1 - f[0]) * (f[1] if dj else 1 - f[1]) * (f[2] if dk else 1 - f[2]))
                inside = all(0 <= i < n for i, n in zip(index, data.shape, strict=True))
                corners.append((index, float(weight), int(data[index]) if inside else 0))
    return corners


def vec(v) -> str:
    return "(" + ", ".join(f"{c:.3f}" for c in v) + ")"


def box_edges(low, high) -> np.ndarray:
    """The 12 edges of an axis-aligned box as one polyline with None breaks (for one plotly trace)."""
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    points: list = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(corners[i] != corners[j]) == 1:
                points += [corners[i], corners[j], [None] * 3]
    return np.array(points, dtype=object)


def line_trace(points, color: str, width: float, name: str, show: bool = True) -> go.Scatter3d:
    points = np.asarray(points, dtype=object)
    return go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="lines", name=name, showlegend=show,
                        line={"color": color, "width": width}, hoverinfo="skip")


def scene_3d(volume: Volume, plane, rows: list[dict], image: np.ndarray, pixel: tuple[int, int], mode: str):
    """The volume box, the plane, both rails, every row and every sample point; the chosen pixel's row, sample
    point and voxel(s) are highlighted."""
    a1_0, b1, a2_0, b2 = (np.asarray(c, dtype=np.float64) for c in plane.corners)
    shape = np.array(volume.shape)
    traces = [line_trace(box_edges(-0.5 * np.ones(3), shape - 0.5), "#8a8f98", 2, "volume box")]
    outline = [a1_0, b1, b2, a2_0]
    corners = np.array(outline)
    traces.append(go.Mesh3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2], i=[0, 0], j=[1, 2], k=[2, 3],
                            color=CHOSEN, opacity=0.12, name="plane", hoverinfo="skip"))
    traces.append(line_trace([a1_0, b1], RAIL_1, 7, "rail 1: a1 = A1 + y·v1"))
    traces.append(line_trace([a2_0, b2], RAIL_2, 7, "rail 2: a2 = A2 + y·v2"))
    segments: list = []
    for row in rows:
        segments += [row["a1"], row["a2"], [None] * 3]
    traces.append(line_trace(segments, "#8a8f98", 1.5, "rows a1 → a2"))

    samples = [(y, x, p) for y, row in enumerate(rows) for _, x, p in row["samples"] if 0 <= x < image.shape[1]]
    points = np.array([p for *_, p in samples])
    values = [int(image[y, x]) for y, x, _ in samples]
    traces.append(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", name="sample points p",
        marker={"size": 3.5, "color": values, "colorscale": "gray", "cmin": 0, "cmax": 255,
                "line": {"color": "#8a8f98", "width": 0.5}},
        text=[f"pixel (y={y}, x={x}), value {v}" for (y, x, _), v in zip(samples, values, strict=True)],
        hoverinfo="text"))

    y, x = pixel
    row = rows[y]
    traces.append(line_trace([row["a1"], row["a2"]], CHOSEN, 5, f"row y = {y}"))
    hit = next((p for _, xx, p in row["samples"] if xx == x), None)
    if hit is not None:
        traces.append(go.Scatter3d(x=[hit[0]], y=[hit[1]], z=[hit[2]], mode="markers", name="chosen p",
                                   marker={"size": 7, "color": CHOSEN}, hoverinfo="skip"))
        if mode == Interpolation.NEAREST.value:
            voxel = np.floor(hit + HALF)
            traces.append(line_trace(box_edges(voxel - 0.5, voxel + 0.5), CHOSEN, 4, "nearest voxel"))
        else:
            used = np.array([c for c, w, _ in trilinear_corners(volume.data, hit)], dtype=np.float64)
            traces.append(go.Scatter3d(x=used[:, 0], y=used[:, 1], z=used[:, 2], mode="markers",
                                       name="8 corner voxels", marker={"size": 4, "color": CHOSEN, "symbol": "diamond"},
                                       hoverinfo="skip"))
    normal = np.cross(b1 - a1_0, b2 - a1_0 if np.allclose(a1_0, a2_0) else a2_0 - a1_0)
    normal /= np.linalg.norm(normal)
    side = np.cross(normal, [0, 0, 1]) if abs(normal[2]) < 0.9 else np.cross(normal, [1, 0, 0])
    eye = 1.6 * normal + 0.6 * side / np.linalg.norm(side)   # face the plane, turned a little to show depth
    figure = go.Figure(traces)
    figure.update_layout(scene={"aspectmode": "data", "xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z",
                                "camera": {"eye": dict(zip("xyz", eye.tolist(), strict=True))}},
                         margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=520,
                         legend={"yanchor": "top", "y": 0.98, "xanchor": "left", "x": 0.0})
    return figure


def image_figure(image: np.ndarray, pixel: tuple[int, int]) -> go.Figure:
    """The projected image as clickable squares, one per pixel, with row 0 at the top."""
    h, w = image.shape
    ys, xs = np.mgrid[0:h, 0:w]
    size = max(6, min(30, 330 // max(h, w)))
    figure = go.Figure(go.Scatter(
        x=xs.ravel(), y=ys.ravel(), mode="markers",
        marker={"symbol": "square", "size": size, "color": image.ravel(), "colorscale": "gray", "cmin": 0, "cmax": 255,
                "line": {"color": "#8a8f98", "width": 0.5}},
        text=[f"pixel (y={y}, x={x}), value {v}"
              for y, x, v in zip(ys.ravel(), xs.ravel(), image.ravel(), strict=True)],
        hoverinfo="text"))
    figure.add_trace(go.Scatter(x=[pixel[1]], y=[pixel[0]], mode="markers", hoverinfo="skip",
                                marker={"symbol": "square-open", "size": size + 4, "color": CHOSEN,
                                        "line": {"width": 3}}))
    figure.update_layout(showlegend=False, height=size * h + 80, width=size * w + 110, dragmode=False,
                         margin={"l": 50, "r": 10, "t": 10, "b": 40}, clickmode="event+select",
                         xaxis={"title": "x (column)", "range": [-0.6, w - 0.4], "dtick": 1 if w <= 16 else 5},
                         yaxis={"title": "y (row)", "range": [h - 0.4, -0.6], "dtick": 1 if h <= 16 else 5,
                                "scaleanchor": "x"})
    return figure


def pick_pixel() -> None:
    points = st.session_state["teach_image"]["selection"]["points"]
    if points:
        st.session_state["teach_pixel"] = (int(points[0]["y"]), int(points[0]["x"]))


def explain_pixel(volume: Volume, rails, rows: list[dict], image: np.ndarray, pixel: tuple[int, int],
                  mode: str) -> None:
    """The algorithm's steps for one pixel, with the numbers of this plane."""
    y, x = pixel
    row = rows[y]
    st.markdown(f"#### Pixel y = {y}, x = {x}")
    st.markdown(
        f"1. **Row y = {y} walks between the rails.**  \n"
        f"a1 = A1 + y·v1 = {vec(rails.A1)} + {y}·{vec(rails.v1)} = **{vec(row['a1'])}**  \n"
        f"a2 = A2 + y·v2 = {vec(rails.A2)} + {y}·{vec(rails.v2)} = **{vec(row['a2'])}**\n"
        f"2. **Row length and direction.** d12 = |a2 − a1| = **{row['d12']:.3f}**, so the row takes "
        f"floor(d12) + 1 = **{len(row['samples'])}** samples, one voxel apart: dp = (a2 − a1)/d12 = {vec(row['dp'])}.\n"
        f"3. **Centering.** The row is centered in the image width W = {rails.W}: "
        f"off = (W − 1)/2 − d12/2 = **{row['off']:.3f}**, and sample x1 goes to column x = floor(off + x1 + ½)."
    )
    hit = next(((x1, p) for x1, xx, p in row["samples"] if xx == x), None)
    if hit is None:
        used = [xx for _, xx, _ in row["samples"]]
        st.markdown(f"4. **No sample of this row lands on column {x}** (row {y} covers columns {min(used)} to "
                    f"{max(used)}), so the pixel keeps its initial value **0**: it lies outside the plane's shape.")
        value = 0
    else:
        x1, p = hit
        st.markdown(
            f"4. **Sample x1 = {x1}** lands on column floor({row['off']:.3f} + {x1} + ½) = **{x}**.  \n"
            f"p = a1 + x1·dp = {vec(row['a1'])} + {x1}·{vec(row['dp'])} = **{vec(p)}**"
        )
        if mode == Interpolation.NEAREST.value:
            voxel = tuple(int(math.floor(c + HALF)) for c in p)
            value = int(sample_nearest(volume.data, *p))
            inside = all(0 <= i < n for i, n in zip(voxel, volume.shape, strict=True))
            st.markdown(
                f"5. **Nearest voxel**: floor(p + ½) = **{voxel}**, "
                + (f"whose value is M{list(voxel)} = **{value}**." if inside else "outside the volume, so **0**.")
            )
        else:
            corners = trilinear_corners(volume.data, p)
            value = sample_trilinear(volume.data, *p)
            st.markdown("5. **Trilinear**: the 8 voxels around p, each weighted by its closeness to p on every axis:")
            st.dataframe({"voxel": [str(c) for c, _, _ in corners], "weight": [round(w, 4) for _, w, _ in corners],
                          "value": [v for _, _, v in corners],
                          "weight × value": [round(w * v, 3) for _, w, v in corners]}, hide_index=True)
            total = sum(w * v for _, w, v in corners)
            st.markdown(f"Sum = {total:.3f}, rounded: floor(sum + ½) = **{value}**.")
    program = int(image[y, x])
    if program == value:
        st.success(f"E[{y}][{x}] = {value}, the same value the program's reference backend wrote for this pixel.")
    else:
        st.error(f"The steps give {value} but the reference backend wrote {program}.")


def teaching_tab() -> None:
    st.markdown(
        "The plane is covered by **two rails** that start at the top and run along the plane's two legs, "
        "one voxel per step. Row y of the image is the segment from the point a1 on rail 1 to the point a2 on "
        "rail 2; the row is sampled once per voxel of its length and centered in the image. For a triangle both "
        "rails start at the apex a, so the first row is a single point. **Click a pixel** to see how it was computed."
    )
    controls, picture = st.columns([1, 1.3])
    with controls:
        shape = st.radio("Plane", ["Triangle", "Trapezoid"], horizontal=True, key="teach_shape")
        mode = st.radio("Sampling", ["nearest", "trilinear"], horizontal=True, key="teach_mode")
        yaw = st.slider("turn about z", -180, 180, 30, key="teach_yaw")
        pitch = st.slider("tilt about y", -90, 90, 20, key="teach_pitch")
        roll = st.slider("tilt about x", -180, 180, 70, key="teach_roll")
        height = st.slider("height (voxels)", 2.0, 14.0, 9.0, 0.5, key="teach_height")
        base = st.slider("base (voxels)", 2.0, 14.0, 10.0, 0.5, key="teach_base")
        top = st.slider("top (voxels)", 0.0, 14.0, 4.0, 0.5, key="teach_top") if shape == "Trapezoid" else 0.0

    volume = teaching_volume()
    center = [(TEACH_SIDE - 1) / 2] * 3
    try:
        plane = plane_from_points(plane_points(shape, center, rotation(yaw, pitch, roll), height, base, top),
                                  force=True)
    except InvalidInputError as exc:
        st.error(f"This plane cannot be projected: {exc}")
        return
    rails = plane.rails()
    image = make_projector("reference", Interpolation(mode)).project(volume, plane)
    rows = walk(rails)
    y, x = st.session_state.get("teach_pixel", (2 * rails.H // 3, rails.W // 2))
    pixel = (min(y, rails.H - 1), min(x, rails.W - 1))

    with picture:
        st.markdown(f"**The image E: H = {rails.H} rows × W = {rails.W} columns**")
        st.plotly_chart(image_figure(image, pixel), key="teach_image", on_select=pick_pixel,
                        selection_mode="points", width="content", config={"displayModeBar": False})
        st.caption(f"A1 = {vec(rails.A1)}, v1 = {vec(rails.v1)}; A2 = {vec(rails.A2)}, v2 = {vec(rails.v2)}. "
                   f"|v1| = |v2| = {np.linalg.norm(rails.v1):.3f} voxel per row.")
    left, right = st.columns([1.3, 1])
    with left:
        st.plotly_chart(scene_3d(volume, plane, rows, image, pixel, mode), key="teach_scene",
                        config={"displaylogo": False})
        st.caption(f"A {TEACH_SIDE}³ volume, brightest at its center. Drag to turn the view; hover a point to "
                   "see its pixel.")
    with right:
        explain_pixel(volume, rails, rows, image, pixel, mode)


def explore_tab(volume: Volume, window: IntensityWindow, controls: dict) -> None:
    points = plane_points(controls["shape"], controls["center"], rotation(*controls["angles"]), *controls["size"])
    left, right = st.columns([1, 1.4])
    with left:
        st.pyplot(sketch(volume.shape, points), clear_figure=True)
        st.markdown("**The plane as a CSV row** (the lecturer's format):")
        st.code(csv_row(points), language="text", wrap_lines=True)
    with right:
        interpolation, backend = controls["interpolation"], controls["backend"]
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


def main() -> None:
    st.set_page_config(page_title="Plane projection demo", layout="wide")
    st.title("Plane projection of a 3D volume")
    st.caption("Ex1, Introduction to Software Engineering. Move the plane; the image is its projection.")

    with st.sidebar:
        st.header("Explore tab")
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

    explore, teach = st.tabs(["Explore", "How it works"])
    with explore:
        explore_tab(volume, window, {"shape": shape, "center": (cx, cy, cz), "angles": (yaw, pitch, roll),
                                     "size": (height, base, top), "interpolation": interpolation,
                                     "backend": backend})
    with teach:
        teaching_tab()


if __name__ == "__main__":
    main()
