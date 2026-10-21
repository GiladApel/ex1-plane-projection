# Experiment

All commands are run from the repository root after `python scripts/download_data.py`. The CSV plane lists and output videos are in [experiment/](experiment/).

![Frames from the three runs: MRHead axial, sagittal, coronal; CTChest](experiment/preview.png)

## Runs

| # | Volume | Planes | Command | Output |
|---|---|---|---|---|
| 1 | MRHead | 256 rectangles, one per slice along the head's superior-inferior axis (voxel axis y) | `python scripts/make_planes.py sweep data/MRHead.nrrd docs/experiment/mrhead_sweep.csv --axis y`<br>`python Ex0.py data/MRHead.nrrd docs/experiment/mrhead_sweep.csv docs/experiment/mrhead_sweep.mp4 --fps 25` | [mrhead_sweep.mp4](experiment/mrhead_sweep.mp4) |
| 2 | MRHead | 180 planes through the center, rotating 360 degrees around axis y in 2 degree steps | `python scripts/make_planes.py rotate data/MRHead.nrrd docs/experiment/mrhead_rotate.csv --axis y --frames 180`<br>`python Ex0.py data/MRHead.nrrd docs/experiment/mrhead_rotate.csv docs/experiment/mrhead_rotate.mp4 --fps 30` | [mrhead_rotate.mp4](experiment/mrhead_rotate.mp4) |
| 3 | MRHead | same planes as run 2, trilinear sampling | add `--interp trilinear` | [mrhead_rotate_trilinear.mp4](experiment/mrhead_rotate_trilinear.mp4) |
| 4 | CTChest | 180 planes rotating around the long axis z | same as run 2 with `data/CTChest.nrrd`, `--axis z` | [ctchest_rotate.mp4](experiment/ctchest_rotate.mp4) |

## Results

**Correct geometry.** Run 1 reproduces the 256 axial slices of the head in order (frame 0 is the top of the head and frame 255 the neck, following the file's axis direction). In run 2 the plane starts as a sagittal cut (the head seen from the side) and passes through the coronal cut (seen from the front) after a quarter turn, as expected for a plane that contains the rotation axis. The sweep frames are exact slices: the test suite verifies for every axis that a sweep plane returns the same values as the NumPy slice.

**No jitter in the rotation.** The rotating plane has a half-integer side length. With an integer side, floating-point noise made the image size flip by one pixel between frames; a test caught this during development.

**Speed.** Projection-only and end-to-end rates reported by `Ex0.py` on the test machine (i7-13620H, 16 threads, Numba backend, nearest sampling):

| Run | Frame size | Projection fps | End-to-end fps (with mp4 encoding) |
|---|---|---|---|
| 1. MRHead sweep | 256 x 130 | 830 | 311 |
| 2. MRHead rotation | 386 x 386 | 1774 | 108 |
| 3. MRHead rotation, trilinear | 386 x 386 | 821 | 113 |
| 4. CTChest rotation | 740 x 740 | 716 | 33 |

The projection is far above the 100 fps target in every run. The end-to-end rate is limited by H.264 encoding, which grows with frame area; it falls to 33 fps only for the 740 x 740 CT frames. Detailed projection benchmarks on a 500^3 volume are in [experiment/benchmark.md](experiment/benchmark.md): Numba reaches about 1000 fps with nearest sampling and about 390 fps with trilinear, and still exceeds 100 fps on a single thread (170 fps). The NumPy fallback reaches 53 to 96 fps with nearest sampling.

**Findings from the benchmark.**
- Axis-aligned planes are faster than oblique ones (Numba uint8: 1463 vs 1009 fps; NumPy: 110 vs 53). The likely cause (not measured directly) is memory locality: axis-aligned samples walk the volume along regularly strided lines, while oblique samples jump across the 125 MB volume and miss the processor cache.
- Larger voxel types are slower only for oblique planes (Numba nearest: 1009 fps for uint8, 808 for uint16, 581 for uint32), probably the same memory effect, since the volume is 125, 250 and 500 MB.
- Trilinear costs 2.5 to 3 times more than nearest on Numba, because it reads 8 voxels per sample.
- Threads: 170, 332, 456, 997, 1052 fps for 1, 2, 4, 8, 16 threads. Scaling is close to linear up to 8 threads and flat beyond. This CPU has 6 performance and 4 efficiency cores (16 threads); the flat part is probably memory bandwidth and the shared cores, but this was not investigated.

**Nearest versus trilinear.** Comparing frame 20 of runs 2 and 3 (an oblique plane), nearest sampling shows blocky, stair-stepped tissue edges and trilinear is visibly smoother; 31% of the pixels differ, by 2 gray levels on average. The price is the 2.5x lower projection speed, which is still more than 800 fps.

## Limitations observed

- **CT contrast.** The CTChest video is pale. The brightness window uses the 1st and 99th percentile of the whole volume (SRS D12); this volume has a large block of padding at its minimum value and the loader shifted the Hounsfield values up by 3024, so air, soft tissue and bone all land in the upper part of the range. A tissue-specific window (for example lung or soft tissue) would give better contrast but needs user-supplied bounds, which are out of scope.
- **Anisotropic voxels.** The CTChest frames look flattened vertically because the slice spacing is 2.5 mm against 0.76 mm in plane. Projection works in voxel space by design (SRS D15).
- **NumPy fallback speed.** Without Numba, nearest sampling runs at 53 to 96 fps (above the 30 fps minimum, below the 100 fps target) and trilinear at about 6 fps.
- **Single machine.** All timings come from one laptop; results on other hardware will differ, especially the Numba thread scaling.
