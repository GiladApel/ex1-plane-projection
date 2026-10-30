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
| 1. MRHead sweep | 256 x 130 | 988 | 393 |
| 2. MRHead rotation | 386 x 386 | 1540 | 137 |
| 3. MRHead rotation, trilinear | 386 x 386 | 890 | 132 |
| 4. CTChest rotation | 740 x 740 | 777 | 46 |

The projection is far above the 100 fps target in every run. The end-to-end rate is limited by H.264 encoding, which grows with frame area; it falls to 46 fps only for the 740 x 740 CT frames. Detailed projection benchmarks on a 500^3 volume are in [experiment/benchmark.md](experiment/benchmark.md): Numba reaches about 840 fps with nearest sampling on oblique planes and about 410 fps with trilinear, and still exceeds 100 fps on a single thread (169 fps). The NumPy fallback reaches about 150 fps with nearest sampling and 16 to 35 fps with trilinear.

**Findings from the benchmark.**
- Axis-aligned planes are faster than oblique ones (Numba uint8 nearest: 1946 vs 837 fps; NumPy: 230 vs 153). The likely cause (not measured directly) is memory locality: axis-aligned samples walk the volume along regularly strided lines, while oblique samples jump across the 125 MB volume and miss the processor cache.
- Larger voxel types are slower only for oblique planes (Numba nearest: 837 fps for uint8, 708 for uint16, 519 for uint32), probably the same memory effect, since the volume is 125, 250 and 500 MB.
- Trilinear costs 2 to 3.4 times more than nearest on Numba, because it reads 8 voxels per sample.
- Numba threads: 169, 366, 648, 1000, 500 fps for 1, 2, 4, 8, 16 threads. Scaling is close to linear up to 8 threads. In an earlier run 16 threads gave 1052 fps, so the drop at 16 is partly run-to-run noise from oversubscribing this CPU (6 performance and 4 efficiency cores, 16 threads); this was not investigated further.

**Nearest versus trilinear.** Comparing frame 20 of runs 2 and 3 (an oblique plane), nearest sampling shows blocky, stair-stepped tissue edges and trilinear is visibly smoother; 31% of the pixels differ, by 2 gray levels on average. The price is the 2.5x lower projection speed, which is still more than 800 fps.

## Limitations observed

- **CT contrast.** In the first version the CTChest video was pale: 21% of the volume is padding at the minimum value (-3024 HU, shifted to 0 by the loader), which pulled the 1st percentile of the brightness window down to the padding, so air, soft tissue and bone all landed in the upper part of the gray range. The window now ignores the minimum value when it covers at least 5% of the volume (SRS D12): for CTChest the window moved from 0..3327 to 2014..3372, so air is black and the lungs are visible. MRHead hardly changes (0..134 to 1..136). Soft tissue still appears light gray, since one global window covers air to bone; a tissue-specific window (lung or soft tissue) would need user-supplied bounds, which are out of scope.
- **Anisotropic voxels.** The CTChest frames look flattened vertically because the slice spacing is 2.5 mm against 0.76 mm in plane. Projection works in voxel space by design (SRS D15).
- **NumPy fallback speed.** Without Numba, nearest sampling runs at about 150 fps on this machine because the frame is split across 4 threads; with fewer cores it will be slower. Trilinear runs at 16 to 35 fps, below the 30 fps minimum in the worst case, which is why trilinear is only required on Numba.
- **Single machine.** All timings come from one laptop; results on other hardware will differ, especially the Numba thread scaling.
