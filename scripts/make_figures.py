"""Draw the figures of docs/ALGORITHM.md from the real implementation.

python scripts/make_figures.py        # needs matplotlib: pip install -e ".[figures]"

Writes docs/algorithm_overview.png and docs/algorithm_changes.png. Every number and image in the
figures is computed with the project's own classes, and the script stops with an assertion error if
the drawing and the implementation disagree.
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from projection.geometry import Trapezoid, Triangle  # noqa: E402
from projection.projector import ReferenceProjector  # noqa: E402
from projection.volume import Volume  # noqa: E402

BLUE, ORANGE, GREEN, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#8a8f98", "#1f2328"
PALE, MIDBLUE = "#dbe6f3", "#9ec5e8"
HALO = [pe.withStroke(linewidth=3.2, foreground="white")]
SIDE = 12
HIGHLIGHT_ROW = 10
TRIANGLE = ((6, -3, 5), (1, 11, 8), (11, 11, 8))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "mathtext.fontset": "dejavusans"})


def note(ax, x, y, text, **kw):
    kw.setdefault("color", INK)
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    return ax.text(x, y, text, path_effects=HALO, **kw)


# ---------------------------------------------------------------- example data

def example():
    """The plane, volume and projection used by the figures."""
    values = (30 + 20 * np.arange(SIDE, dtype=np.int64)).astype(np.uint8)
    volume = Volume(np.broadcast_to(values[:, None, None], (SIDE,) * 3).copy())
    plane = Triangle(*TRIANGLE)
    image = ReferenceProjector().project(volume, plane)
    return volume, plane, image


def row_geometry(rails, y):
    """a1, a2, d12, dp, offset and pixel columns of row y, with the formulas of docs/ALGORITHM.md."""
    a1 = rails.A1 + y * rails.v1
    a2 = rails.A2 + y * rails.v2
    d12 = float(np.linalg.norm(a2 - a1))
    dp = (a2 - a1) / d12 if d12 > 0 else np.zeros(3)
    off = (rails.W - 1) / 2.0 - d12 / 2.0
    columns = [math.floor(off + x1 + 0.5) for x1 in range(math.floor(d12) + 1)]
    return a1, a2, d12, dp, off, columns


# ---------------------------------------------------------------- 3D drawing helpers

class View:
    """Orthographic view of voxel space; screen up is -y so that y runs downwards like image rows."""

    def __init__(self, azimuth_deg, elevation_deg, center):
        a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
        self.ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        self.rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
        self.center = np.asarray(center, dtype=float)

    def project(self, pts):
        pts = np.atleast_2d(np.asarray(pts, dtype=float)) - self.center
        rotated = (pts * np.array([1.0, -1.0, 1.0])) @ self.ry.T @ self.rx.T
        return rotated[:, :2]

    def facing(self, normal):
        rotated = (np.asarray(normal, dtype=float) * np.array([1.0, -1.0, 1.0])) @ self.ry.T @ self.rx.T
        return rotated[2] > 0


def clip_polygon(poly, axis, bound, keep_above):
    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        p_in = (p[axis] >= bound) if keep_above else (p[axis] <= bound)
        q_in = (q[axis] >= bound) if keep_above else (q[axis] <= bound)
        if p_in:
            out.append(p)
        if p_in != q_in:
            t = (bound - p[axis]) / (q[axis] - p[axis])
            out.append(p + t * (q - p))
    return out


def inside_part(p0, p1, lo, hi):
    """Parameter interval of segment p0-p1 inside the box [lo, hi]^3, or None."""
    t0, t1 = 0.0, 1.0
    d = p1 - p0
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if not lo <= p0[k] <= hi:
                return None
            continue
        ta, tb = sorted(((lo - p0[k]) / d[k], (hi - p0[k]) / d[k]))
        t0, t1 = max(t0, ta), min(t1, tb)
    return (t0, t1) if t0 < t1 else None


def draw_segment(ax, view, p0, p1, color, width, z=3, lo=-0.5, hi=SIDE - 0.5):
    """Solid inside the volume box, dashed grey outside it."""
    span = inside_part(p0, p1, lo, hi)
    pieces = [(0.0, 1.0, False)] if span is None else [
        (0.0, span[0], False), (span[0], span[1], True), (span[1], 1.0, False)]
    for a, b, inside in pieces:
        if b - a < 1e-9:
            continue
        xy = view.project(np.array([p0 + a * (p1 - p0), p0 + b * (p1 - p0)]))
        if inside:
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=width, solid_capstyle="round", zorder=z)
        else:
            ax.plot(xy[:, 0], xy[:, 1], color=GREY, lw=max(width * 0.8, 0.8), ls=(0, (3, 2)), zorder=z)


def draw_box(ax, view, lo=-0.5, hi=SIDE - 0.5):
    corners = np.array([[x, y, z] for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)], dtype=float)
    faces = []
    for axis in range(3):
        for sign in (-1, 1):
            normal = np.zeros(3)
            normal[axis] = sign
            faces.append((axis, sign, view.facing(normal)))
    for i, a in enumerate(corners):
        for j in range(i + 1, 8):
            b = corners[j]
            diff = np.flatnonzero(a != b)
            if len(diff) != 1:
                continue
            shared = [k for k in range(3) if k != diff[0]]
            visible = any(vis for axis, sign, vis in faces
                          if axis in shared and np.isclose(a[axis], hi if sign == 1 else lo))
            xy = view.project(np.array([a, b]))
            ax.plot(xy[:, 0], xy[:, 1], color=INK if visible else GREY, lw=1.3 if visible else 0.9,
                    ls="-" if visible else (0, (3, 3)), zorder=2 if visible else 1, alpha=0.9 if visible else 0.7)


# ---------------------------------------------------------------- figure 1, panel A

def panel_scene(ax, plane):
    rails = plane.rails()
    view = View(azimuth_deg=-32, elevation_deg=24, center=[(SIDE - 1) / 2] * 3)
    A, B, C = (np.array(p, dtype=float) for p in TRIANGLE)
    assert np.allclose(rails.A1, A)

    draw_box(ax, view)

    full = view.project(np.array([A, B, C]))
    ax.add_patch(Polygon(full, closed=True, facecolor=PALE, edgecolor="none", alpha=0.7, zorder=2))
    clipped = [A, B, C]
    for axis in range(3):
        clipped = clip_polygon(clipped, axis, -0.5, True)
        clipped = clip_polygon(clipped, axis, SIDE - 0.5, False)
    ax.add_patch(Polygon(view.project(np.array(clipped)), closed=True, facecolor=MIDBLUE, edgecolor="none",
                         alpha=0.6, zorder=2))

    for y in range(rails.H):
        if y != HIGHLIGHT_ROW:
            a1, a2, *_ = row_geometry(rails, y)
            draw_segment(ax, view, a1, a2, color="#6f8fb0", width=0.7, z=3)

    a1, a2, d12, dp, off, cols = row_geometry(rails, HIGHLIGHT_ROW)
    draw_segment(ax, view, a1, a2, color=INK, width=2.0, z=5)
    samples = np.array([a1 + k * dp for k in range(len(cols))])
    dots = view.project(samples)
    ax.scatter(dots[:, 0], dots[:, 1], s=36, color=GREEN, edgecolor="white", linewidth=0.8, zorder=6)

    draw_segment(ax, view, A, B, color=BLUE, width=2.6, z=4)
    draw_segment(ax, view, A, C, color=ORANGE, width=2.6, z=4)
    draw_segment(ax, view, B, C, color=INK, width=1.6, z=4)

    pa, pb, pc = view.project(np.array([A, B, C]))
    note(ax, pa[0], pa[1] + 0.9, "A", fontsize=15, fontweight="bold")
    note(ax, pb[0] - 0.75, pb[1] - 0.15, "B", fontsize=15, fontweight="bold", color=BLUE)
    note(ax, pc[0] + 0.75, pc[1] - 0.15, "C", fontsize=15, fontweight="bold", color=ORANGE)

    dab = float(np.linalg.norm(B - A))
    dbc = float(np.linalg.norm(C - B))
    mid_ab, mid_ac, mid_bc = (view.project(np.array([m]))[0] for m in ((A + B) / 2, (A + C) / 2, (B + C) / 2))
    note(ax, mid_ab[0] - 1.95, mid_ab[1] + 0.35, f"$D_{{ab}}={dab:.1f}$", fontsize=12, color=BLUE)
    note(ax, mid_ac[0] + 1.95, mid_ac[1] + 0.35, f"$D_{{ac}}={dab:.1f}$", fontsize=12, color=ORANGE)
    note(ax, mid_bc[0], mid_bc[1] - 0.95, f"$D_{{bc}}={dbc:.1f}$", fontsize=12)

    pa1, pa2 = view.project(np.array([a1, a2]))
    note(ax, pa1[0] - 1.05, pa1[1] + 0.2, "$a_1 = p_0$", fontsize=13, color=BLUE)
    note(ax, pa2[0] + 0.55, pa2[1] + 0.7, "$a_2$", fontsize=13, color=ORANGE)
    middle = (pa1 + pa2) / 2
    note(ax, middle[0] - 0.2, middle[1] - 1.75, f"row $y={HIGHLIGHT_ROW}$:  $d_{{12}}={d12:.1f}$", fontsize=11.5)
    note(ax, middle[0] - 0.2, middle[1] - 2.45, "$p$ steps from $a_1$ to $a_2$ by $dp$", fontsize=11.5,
         color=GREEN)

    inside_point = view.project(np.array([A + 0.1 * (B - A) + 0.1 * (C - A)]))[0]
    note(ax, inside_point[0] - 4.6, inside_point[1] + 1.2, "outside $M$: value 0", fontsize=11.5, color="#555b64")
    ax.annotate("", xy=inside_point + np.array([-0.15, 0.0]), xytext=(inside_point[0] - 3.4, inside_point[1] + 1.1),
                arrowprops={"arrowstyle": "->", "color": "#555b64", "lw": 1.1}, zorder=7)

    corner = np.array([-0.5, SIDE - 0.5, SIDE - 0.5])
    base = view.project(np.array([corner]))[0]
    origin = base + np.array([-2.6, -0.4])
    tips = []
    for name, direction in (("x", [3, 0, 0]), ("y", [0, 3, 0]), ("z", [0, 0, 3])):
        shift = view.project(np.array([corner + np.array(direction)]))[0] - base
        tip = origin + shift
        tips.append(tip)
        ax.annotate("", xy=tip, xytext=origin, arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.4}, zorder=7)
        label = tip + 0.7 * shift / np.linalg.norm(shift)
        note(ax, label[0], label[1], name, fontsize=12.5)
        tips.append(label)

    box_pts = view.project(np.array([[x, y, z] for x in (-0.5, SIDE - 0.5) for y in (-0.5, SIDE - 0.5)
                                      for z in (-0.5, SIDE - 0.5)]))
    note(ax, box_pts[:, 0].max() - 1.2, box_pts[:, 1].max() + 0.2, "volume $M$", fontsize=14, fontweight="bold")

    ax.set_aspect("equal")
    ax.axis("off")
    pts = np.vstack([box_pts, full, origin[None, :], np.array(tips)])
    ax.set_xlim(pts[:, 0].min() - 1.0, pts[:, 0].max() + 2.0)
    ax.set_ylim(pts[:, 1].min() - 0.8, pts[:, 1].max() + 1.6)


# ---------------------------------------------------------------- figure 1, panel B

def panel_image(ax, plane, image):
    rails = plane.rails()
    height, width = rails.H, rails.W
    visited = np.zeros((height, width), dtype=bool)
    for y in range(height):
        *_, columns = row_geometry(rails, y)
        visited[y, [c for c in columns if 0 <= c < width]] = True
    assert np.all(visited[image > 0]), "image has values outside the visited region"

    cmap = plt.get_cmap("viridis")
    peak = float(image.max())
    rgb = np.full((height, width, 3), 0.97)
    for y in range(height):
        for x in range(width):
            if image[y, x] > 0:
                rgb[y, x] = cmap(0.15 + 0.85 * image[y, x] / peak)[:3]
    ax.imshow(rgb, extent=(-0.5, width - 0.5, height - 0.5, -0.5), interpolation="nearest", zorder=1)
    for y, x in zip(*np.nonzero(visited & (image == 0)), strict=True):
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#e6ebf2", edgecolor="#8b97a8", hatch="////",
                               lw=0, zorder=2))
    for k in range(height + 1):
        ax.plot([-0.5, width - 0.5], [k - 0.5, k - 0.5], color="#9aa3b0", lw=0.4, alpha=0.5, zorder=3)
    for k in range(width + 1):
        ax.plot([k - 0.5, k - 0.5], [-0.5, height - 0.5], color="#9aa3b0", lw=0.4, alpha=0.5, zorder=3)
    ax.add_patch(Rectangle((-0.5, -0.5), width, height, fill=False, edgecolor=INK, lw=1.4, zorder=4))

    a1, a2, d12, dp, off, columns = row_geometry(rails, HIGHLIGHT_ROW)
    y = HIGHLIGHT_ROW
    ax.add_patch(Rectangle((-0.5, y - 0.5), width, 1, fill=False, edgecolor=ORANGE, lw=2.4, zorder=6))
    for c in columns:
        ax.plot(c, y, "o", color=GREEN, ms=7.5, markeredgecolor="white", markeredgewidth=0.9, zorder=7)
    centre = (width - 1) / 2.0
    ax.plot([centre, centre], [-0.5, height - 0.5], color=ORANGE, lw=1.1, ls=(0, (4, 3)), zorder=5)
    note(ax, centre, -1.55, r"center $\frac{W-1}{2}$", fontsize=11.5, color=ORANGE)

    callout = (f"row $y={y}$\n$d_{{12}}={d12:.1f}$  →  {len(columns)} samples\n"
               f"$x=\\frac{{W-1}}{{2}}-\\frac{{d_{{12}}}}{{2}}+x_1$\n"
               f"$={centre:.0f}-{d12 / 2:.2f}+x_1$\n→ columns {columns[0]} … {columns[-1]}")
    ax.annotate(callout, xy=(width - 0.4, y), xytext=(width + 1.0, y - 0.3), fontsize=10.5, color=INK,
                va="center", ha="left", annotation_clip=False,
                arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.4}, zorder=8,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": ORANGE, "lw": 1.0})

    never_rows = [r for r in range(height) if visited[r].any() and not (image[r] > 0).any()]
    top, bottom = min(never_rows), max(never_rows)
    ax.annotate("", xy=(width + 0.15, top - 0.45), xytext=(width + 0.15, bottom + 0.45),
                arrowprops={"arrowstyle": "-", "color": GREY, "lw": 2.2}, zorder=7, annotation_clip=False)
    note(ax, width + 0.65, (top + bottom) / 2, f"rows {top}–{bottom}: $p$ is\noutside $M$, value 0", fontsize=10.5,
         color="#555b64", ha="left")

    ax.annotate("", xy=(-2.0, height - 0.5), xytext=(-2.0, -0.5),
                arrowprops={"arrowstyle": "<->", "color": INK, "lw": 1.3}, annotation_clip=False)
    note(ax, -3.1, (height - 1) / 2, f"$H=\\lfloor D_{{ab}}\\rfloor+1={height}$", fontsize=12, rotation=90)
    ax.annotate("", xy=(-0.5, height + 0.7), xytext=(width - 0.5, height + 0.7),
                arrowprops={"arrowstyle": "<->", "color": INK, "lw": 1.3}, annotation_clip=False)
    note(ax, centre, height + 1.5, f"$W=\\lfloor D_{{bc}}\\rfloor+1={width}$", fontsize=12)

    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10, 15])
    ax.xaxis.tick_top()
    ax.tick_params(length=3, labelsize=10, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    note(ax, width + 0.35, -1.55, "$x\\;\\rightarrow$", fontsize=12.5, ha="left")
    note(ax, -1.2, -1.55, "$y\\downarrow$", fontsize=12.5)

    legend_y = height + 3.0
    swatches = (("#3b8f8a", None, "value $M[p]$"), ("#e6ebf2", "////", "$p$ outside $M$: 0"),
                ("#f7f7f7", None, "never visited"))
    for i, (color, hatch, text) in enumerate(swatches):
        x0 = -3.0 + i * 5.2
        ax.add_patch(Rectangle((x0, legend_y - 0.35), 0.8, 0.8, facecolor=color, edgecolor="#8b97a8", hatch=hatch,
                               lw=0.8, clip_on=False))
        ax.text(x0 + 1.1, legend_y + 0.05, text, fontsize=10, va="center", ha="left", color=INK)

    ax.text(-3.0, legend_y + 1.6, "example volume: $M[x][y][z]=30+20\\,x$, so brighter means larger $x$",
            fontsize=10, va="center", ha="left", color="#555b64")
    ax.set_xlim(-4.2, width + 9.6)
    ax.set_ylim(legend_y + 2.3, -2.5)
    ax.set_aspect("equal")


def figure_overview(out_path):
    _, plane, image = example()
    rails = plane.rails()
    A, B, C = (np.array(p, dtype=float) for p in TRIANGLE)
    assert plane.shape == (rails.H, rails.W)
    assert rails.H == math.floor(np.linalg.norm(B - A)) + 1
    assert rails.W == math.floor(np.linalg.norm(C - B)) + 1

    fig, (left, right) = plt.subplots(1, 2, figsize=(18, 8.4), gridspec_kw={"width_ratios": [1.0, 1.05]})
    fig.patch.set_facecolor("white")
    panel_scene(left, plane)
    panel_image(right, plane, image)
    fig.text(0.015, 0.975, "A   The plane inside the volume", fontsize=15, fontweight="bold", ha="left")
    fig.text(0.51, 0.975, "B   The output image E", fontsize=15, fontweight="bold", ha="left")
    fig.subplots_adjust(left=0.01, right=0.995, top=0.915, bottom=0.02, wspace=0.03)
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2

def triangle_outline(ax, apex, length, theta, color="#333", lw=1.4):
    """Isosceles triangle drawn upwards-down: apex at `apex`, edges of `length` at half-angle theta (y points down)."""
    ax_, ay = apex
    corner = (length * math.sin(theta), length * math.cos(theta))
    pts = [(ax_, ay), (ax_ - corner[0], ay + corner[1]), (ax_ + corner[0], ay + corner[1])]
    ax.add_patch(Polygon(pts, closed=True, facecolor=PALE, edgecolor=color, lw=lw, alpha=0.8, zorder=1))
    return pts


def row_segment(apex, steps, theta):
    half, depth = steps * math.sin(theta), steps * math.cos(theta)
    return (apex[0] - half, apex[1] + depth), (apex[0] + half, apex[1] + depth)


def panel_rows(ax):
    """D3: the assignment steps a1 before use; we sample row 0 at the apex."""
    theta = math.radians(20)
    length = 6.3
    plane = Triangle((0, 0, 0), (-length * math.sin(theta), length * math.cos(theta), 0),
                     (length * math.sin(theta), length * math.cos(theta), 0))
    ours = list(range(plane.shape[0]))
    assignment = list(range(math.ceil(length)))
    assert len(ours) == len(assignment) == 7, "D3 example: both loops should run 7 rows"

    for centre, title, steps in ((-3.6, "assignment: $a_1 = a_1 + v$ before use", [y + 1 for y in assignment]),
                                 (3.6, "ours: $a_1 = A + y\\cdot v$", ours)):
        apex = (centre, 0.0)
        triangle_outline(ax, apex, length, theta)
        ax.plot([apex[0]], [apex[1]], "o", color=INK, ms=5, zorder=6)
        note(ax, apex[0], apex[1] - 0.55, "A", fontsize=12.5, fontweight="bold")
        for y, s in enumerate(steps):
            (x0, y0), (x1, y1) = row_segment(apex, s, theta)
            beyond = s > length + 1e-9
            colour = "#c0392b" if beyond else (GREEN if s == 0 else BLUE)
            ax.plot([x0, x1], [y0, y1], color=colour, lw=2.0 if not beyond else 1.8, zorder=4,
                    ls=(0, (3, 2)) if beyond else "-")
            ax.plot([x0, x1], [y0, y1], "o", color=colour, ms=3.8, zorder=5)
            note(ax, x1 + 0.45, y1, str(y), fontsize=10, color=colour, ha="left")
        note(ax, centre, -3.5, title, fontsize=11.5, fontweight="bold")

    ax.plot([-3.6], [0], marker="x", color="#c0392b", ms=11, mew=2.4, zorder=7)
    note(ax, -3.6, -1.6, "apex row is skipped", fontsize=10.5, color="#c0392b")
    (x0, y0), _ = row_segment((-3.6, 0.0), 7, theta)
    note(ax, -3.6, y0 + 0.7, "row 6 lies beyond the base", fontsize=10.5, color="#c0392b")
    note(ax, 3.6, -1.6, "row 0 is the apex", fontsize=10.5, color=GREEN)
    base_y = length * math.cos(theta)
    for centre in (-3.6, 3.6):
        ax.plot([centre - length * math.sin(theta) - 0.3, centre + length * math.sin(theta) + 0.3], [base_y, base_y],
                color=GREY, lw=0.8, ls=(0, (1, 2)), zorder=1)
    note(ax, 0.0, base_y + 0.3, "base", fontsize=9.5, color="#555b64")
    ax.set_xlim(-7.4, 7.4)
    ax.set_ylim(base_y + 1.9, -4.2)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_width(ax):
    """D4/D5: the widest row has Dbc + 1 samples, so E needs floor(Dbc) + 1 columns."""
    dbc = 5.0
    samples = list(range(int(dbc) + 1))
    plane = Triangle((0, 0, 0), (-dbc / 2, 6.0, 0), (dbc / 2, 6.0, 0), force=True)
    assert plane.shape[1] == len(samples) == 6

    def strip(y0, columns, label, subtitle, bad):
        for c in range(columns):
            ax.add_patch(Rectangle((c - 0.5, y0 - 0.5), 1, 1, facecolor="#e8eef6", edgecolor=INK, lw=1.2, zorder=2))
            note(ax, c, y0 + 0.9, str(c), fontsize=10, color="#555b64")
        for x in samples:
            colour = "#c0392b" if x in bad else GREEN
            ax.plot(x, y0, "o", color=colour, ms=10, markeredgecolor="white", markeredgewidth=1.0, zorder=5)
        centre = (columns - 1) / 2.0 if "ours" in label else dbc / 2.0
        ax.plot([centre, centre], [y0 - 0.95, y0 + 0.65], color=ORANGE, lw=1.3, ls=(0, (4, 3)), zorder=4)
        note(ax, centre, y0 - 1.25, label, fontsize=11, color=ORANGE)
        note(ax, -0.5, y0 - 2.35, subtitle, fontsize=11.5, fontweight="bold", ha="left")

    strip(2.0, 5, "center $D_{bc}/2$ = 2.5", "assignment: E has $D_{bc}$ = 5 columns", bad=[5])
    ax.annotate("sample $x_1 = d_{12} = 5$\nhas no column", xy=(5.15, 2.0), xytext=(7.7, 2.0),
                fontsize=10.5, color="#c0392b", ha="left", va="center", annotation_clip=False,
                arrowprops={"arrowstyle": "->", "color": "#c0392b", "lw": 1.3}, zorder=8)
    note(ax, 2.5, 4.2, "widest row: $x_1 = 0 \\ldots d_{12}$, that is $D_{bc}+1$ = 6 samples", fontsize=10.5, color=INK)
    strip(8.6, 6, "center $(W-1)/2$ = 2.5", "ours: $W=\\lfloor D_{bc}\\rfloor+1$ = 6 columns", bad=[])
    note(ax, 2.5, 10.6, "same center when $D_{bc}$ is an integer;\nless than one pixel apart otherwise",
         fontsize=10.5, color="#555b64")
    ax.set_xlim(-1.2, 11.2)
    ax.set_ylim(11.6, -3.3)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_trapezoid(ax):
    """D10: a trapezoid uses the same procedure with rails a'->b and a''->c."""
    a1_, b_, a2_, c_ = (-1.3, 0.0), (-3.2, 5.0), (1.3, 0.0), (3.2, 5.0)
    plane = Trapezoid((a1_[0], a1_[1], 0), (b_[0], b_[1], 0), (a2_[0], a2_[1], 0), (c_[0], c_[1], 0))
    height, width = plane.shape
    ax.add_patch(Polygon([a1_, a2_, c_, b_], closed=True, facecolor=PALE, edgecolor="none", alpha=0.85, zorder=1))
    rails = plane.rails()
    for y in range(height):
        _, _, d12, _, _, _ = row_geometry(rails, y)
        t = y / max(rails.H - 1, 1)
        left = (a1_[0] + t * (b_[0] - a1_[0]), a1_[1] + t * (b_[1] - a1_[1]))
        right = (a2_[0] + t * (c_[0] - a2_[0]), a2_[1] + t * (c_[1] - a2_[1]))
        hot = y == 3
        ax.plot([left[0], right[0]], [left[1], right[1]], color=INK if hot else "#6f8fb0", lw=2.2 if hot else 0.8,
                zorder=3)
        if hot:
            ax.plot([left[0], right[0]], [left[1], right[1]], "o", color=GREEN, ms=5, zorder=5)
            note(ax, left[0] - 0.3, left[1] + 0.05, "$a_1$", fontsize=12.5, color=BLUE, ha="right")
            note(ax, right[0] + 0.3, right[1] + 0.05, "$a_2$", fontsize=12.5, color=ORANGE, ha="left")
    ax.plot([a1_[0], b_[0]], [a1_[1], b_[1]], color=BLUE, lw=2.6, zorder=4)
    ax.plot([a2_[0], c_[0]], [a2_[1], c_[1]], color=ORANGE, lw=2.6, zorder=4)
    ax.plot([a1_[0], a2_[0]], [a1_[1], a2_[1]], color=INK, lw=1.4, zorder=4)
    ax.plot([b_[0], c_[0]], [b_[1], c_[1]], color=INK, lw=1.4, zorder=4)
    note(ax, a1_[0] - 0.4, a1_[1] - 0.45, "$a'$", fontsize=13, color=BLUE)
    note(ax, a2_[0] + 0.4, a2_[1] - 0.45, "$a''$", fontsize=13, color=ORANGE)
    note(ax, b_[0] - 0.35, b_[1] + 0.45, "b", fontsize=13, color=BLUE, fontweight="bold")
    note(ax, c_[0] + 0.35, c_[1] + 0.45, "c", fontsize=13, color=ORANGE, fontweight="bold")
    note(ax, 0, -0.75, f"$|a'a''|$ = {abs(a2_[0] - a1_[0]):.1f}", fontsize=11)
    note(ax, 0, 5.75, f"$|bc|$ = {abs(c_[0] - b_[0]):.1f}", fontsize=11)
    note(ax, -3.0, 2.0, "rail 1", fontsize=11, color=BLUE, rotation=75)
    note(ax, 3.0, 2.0, "rail 2", fontsize=11, color=ORANGE, rotation=-75)
    note(ax, 0, 7.05, f"$H=\\lfloor |a'b|\\rfloor+1$ = {height},   $W=\\lfloor\\max(|a'a''|,|bc|)\\rfloor+1$ = {width}",
         fontsize=11, fontweight="bold")
    note(ax, 0, 7.95, "triangle = the case $a' = a''$ (top edge shrinks to the apex)", fontsize=10.5, color="#555b64")
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(8.6, -1.5)
    ax.set_aspect("equal")
    ax.axis("off")


def figure_changes(out_path):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), gridspec_kw={"width_ratios": [1.15, 0.95, 0.9]})
    fig.patch.set_facecolor("white")
    panel_rows(axes[0])
    panel_width(axes[1])
    panel_trapezoid(axes[2])
    fig.subplots_adjust(left=0.01, right=0.995, top=0.9, bottom=0.03, wspace=0.05)
    titles = ("D3   Row indexing: loop $y < D_{ab}$, $D_{ab}$ = 6.3", "D4, D5   Output width and centering",
              "D10   Trapezoid: same procedure, two rails")
    for ax, title in zip(axes, titles, strict=True):
        fig.text(ax.get_position().x0 + 0.01, 0.955, title, fontsize=13.5, fontweight="bold", ha="left")
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)


def worked_example_table(rows=(0, 1, 3, 5, 10, 15)):
    """Markdown table of the numbers behind Figure 1 (the same triangle), for docs/ALGORITHM.md."""
    _, plane, image = example()
    rails = plane.rails()
    fmt = lambda p: "(" + ", ".join(f"{v:.1f}" for v in p) + ")"  # noqa: E731
    lines = ["| y | a1 | a2 | d12 | off | columns | non-zero pixels |", "|---|---|---|---|---|---|---|"]
    for y in rows:
        a1, a2, d12, _, off, columns = row_geometry(rails, y)
        lines.append(f"| {y} | {fmt(a1)} | {fmt(a2)} | {d12:.2f} | {off:.2f} | {columns[0]}..{columns[-1]} | "
                     f"{int(np.count_nonzero(image[y]))} of {len(columns)} |")
    return "\n".join(lines)


def main():
    figure_overview(ROOT / "docs" / "algorithm_overview.png")
    figure_changes(ROOT / "docs" / "algorithm_changes.png")
    print("wrote docs/algorithm_overview.png and docs/algorithm_changes.png\n")
    print(worked_example_table())


if __name__ == "__main__":
    main()
