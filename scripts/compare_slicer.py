"""Compare our projection with vtkImageReslice, the filter 3D Slicer draws its slice views with.

python scripts/compare_slicer.py VOLUME.nrrd [PLANES.csv ...] [--random N] [--interp nearest|trilinear]

For every plane, every pixel our program fills is sampled again by vtkImageReslice at the same point of
the volume, and the two values are compared. The sample points follow ALGORITHM.md section 2; VTK is
given only the points and does the reading of the volume itself: its own axis order, rounding to the
nearest voxel or linear interpolation, and bounds. Needs the VTK package (pip install vtk).

--random N adds N random oblique planes through the volume (half triangles, half trapezoids, seed 0).
With trilinear sampling only pixels at least one voxel inside the volume are compared, because VTK and
this program treat the half-voxel border differently there.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import vtk  # noqa: E402
from vtk.util import numpy_support  # noqa: E402

from projection import Interpolation, Plane, Triangle, Volume, load_nrrd, make_projector, parse_planes  # noqa: E402
from projection.synthetic import random_plane  # noqa: E402

HALF_TOLERANCE = 1e-5


def vtk_image(volume: Volume) -> vtk.vtkImageData:
    image = vtk.vtkImageData()
    image.SetDimensions(*volume.shape)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.SetOrigin(0.0, 0.0, 0.0)
    flat = np.ascontiguousarray(volume.data.ravel(order="F"))
    image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(flat, deep=True))
    return image


def row_samples(plane: Plane, y: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Start point, unit step and first and last column of row y (ALGORITHM.md section 2): pixel x of the
    row samples start + (x - first) * step."""
    rails = plane.rails()
    a1 = rails.A1 + y * rails.v1
    diff = (rails.A2 + y * rails.v2) - a1
    d12 = float(np.sqrt(diff @ diff))
    step = diff / d12 if d12 > 0 else np.zeros(3)
    first = math.floor((rails.W - 1) / 2.0 - d12 / 2.0 + 0.5)
    return a1, step, first, first + math.floor(d12)


def resliced_row(source: vtk.vtkImageData, start, step, count: int, linear: bool) -> np.ndarray:
    """count samples of the volume at start, start + step, ... read by vtkImageReslice."""
    direction = step if step @ step > 0 else np.array([1.0, 0.0, 0.0])
    other = np.cross(direction, [0.0, 0.0, 1.0] if abs(direction[2]) < 0.9 else [1.0, 0.0, 0.0])
    other /= np.linalg.norm(other)
    normal = np.cross(direction, other)
    matrix = vtk.vtkMatrix4x4()
    for row, column in enumerate((direction, other, normal)):
        for k in range(3):
            matrix.SetElement(k, row, column[k])
    for k in range(3):
        matrix.SetElement(k, 3, start[k])
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(source)
    reslice.SetResliceAxes(matrix)
    reslice.SetOutputDimensionality(2)
    reslice.SetOutputSpacing(1.0, 1.0, 1.0)
    reslice.SetOutputOrigin(0.0, 0.0, 0.0)
    reslice.SetOutputExtent(0, count - 1, 0, 0, 0, 0)
    reslice.SetBackgroundLevel(0)
    if linear:
        reslice.SetInterpolationModeToLinear()
    else:
        reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    return numpy_support.vtk_to_numpy(reslice.GetOutput().GetPointData().GetScalars()).reshape(-1)


def compare(volume: Volume, source: vtk.vtkImageData, planes: list[Plane], linear: bool) -> dict:
    """Pseudo-code:
        for each plane:
            ours = our projection of the plane
            for each row y: (start, step, first, last) = row_samples(plane, y)
                theirs = vtkImageReslice samples at start + i * step, i = 0 .. last - first
                compare ours[y][first + i] with theirs[i] for the columns inside the image whose sample
                point lies inside the volume (outside, both are 0 by definition); trilinear: only samples
                at least one voxel inside the volume
                a nearest-voxel difference is "at a half" when the sample point lies within 1e-5 of
                the boundary between two voxels on some axis
    """
    projector = make_projector("auto", Interpolation.TRILINEAR if linear else Interpolation.NEAREST)
    shape = np.array(volume.shape)
    stats = {"planes": len(planes), "pixels": 0, "identical": 0, "at_half": 0, "max_diff": 0}
    for plane in planes:
        ours = projector.project(volume, plane)
        for y in range(plane.shape[0]):
            start, step, first, last = row_samples(plane, y)
            theirs = resliced_row(source, start, step, last - first + 1, linear)
            columns = np.arange(first, last + 1)
            points = start + (columns - first)[:, None] * step
            keep = (columns >= 0) & (columns < plane.shape[1])
            keep &= np.all((points >= -0.5) & (points < shape - 0.5), axis=1)
            if linear:
                keep &= np.all((points >= 1) & (points <= shape - 2), axis=1)
            mine = ours[y, columns[keep]].astype(np.int64)
            other = theirs[keep].astype(np.int64)
            differ = mine != other
            stats["pixels"] += int(keep.sum())
            stats["identical"] += int((~differ).sum())
            if differ.any():
                stats["max_diff"] = max(stats["max_diff"], int(np.abs(mine - other).max()))
                fractions = points[keep][differ] % 1.0
                stats["at_half"] += int(np.any(np.abs(fractions - 0.5) < HALF_TOLERANCE, axis=1).sum())
    return stats


def random_planes(volume: Volume, count: int) -> list[Plane]:
    rng = np.random.default_rng(0)
    center = (np.array(volume.shape) - 1) / 2
    size = float(min(volume.shape))
    planes: list[Plane] = []
    while len(planes) < count:
        offset = rng.uniform(-0.25, 0.25, size=3) * size
        length, bottom = rng.uniform(0.3, 0.9) * size, rng.uniform(0.3, 0.9) * size
        top = 0.0 if len(planes) % 2 == 0 else rng.uniform(0.1, 0.8) * bottom
        trapezoid = random_plane(rng, center + offset, length, top, bottom)
        a, b, _, c = trapezoid.corners
        planes.append(Triangle(a, b, c) if top == 0.0 else trapezoid)
    return planes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nrrd", type=Path)
    parser.add_argument("csv", type=Path, nargs="*")
    parser.add_argument("--random", type=int, default=0)
    parser.add_argument("--interp", choices=[m.value for m in Interpolation], default="nearest")
    args = parser.parse_args()

    volume = load_nrrd(args.nrrd)
    source = vtk_image(volume)
    linear = args.interp == Interpolation.TRILINEAR.value
    sets = [(path.name, parse_planes(path, force=True)) for path in args.csv]
    if args.random:
        sets.append((f"{args.random} random oblique planes", random_planes(volume, args.random)))
    print(f"{args.nrrd.name} {volume.shape} {volume.dtype}, {args.interp}, VTK {vtk.vtkVersion.GetVTKVersion()}")
    for name, planes in sets:
        s = compare(volume, source, planes, linear)
        different = s["pixels"] - s["identical"]
        detail = f", {s['at_half']} of them at a half" if not linear else f", largest difference {s['max_diff']}"
        print(f"  {name}: {s['planes']} planes, {s['pixels']} pixels, identical {s['identical']} "
              f"({s['identical'] / s['pixels']:.5%}), different {different}{detail if different else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
