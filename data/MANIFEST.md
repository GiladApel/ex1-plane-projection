# Data set

The NRRD files are not committed. `python scripts/download_data.py` downloads and checksum-verifies the real volumes below into this folder. Synthetic volumes are generated on demand.

## Real volumes (3D Slicer sample data)

Source: `https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/<sha256>`

| Name | File | Shape (x, y, z) | Stored type | Loaded as | Voxel spacing (mm) | Purpose |
|---|---|---|---|---|---|---|
| MRHead | `MRHead.nrrd` | 256 x 256 x 130 | int16 | uint16, offset 0 | 1.0 x 1.0 x 1.3 | Main demo and experiments; the file the assignment links to. |
| MRBrainTumor1 | `MRBrainTumor1.nrrd` | 256 x 256 x 112 | int16 | uint16, offset 0 | 0.94 x 0.94 x 1.4 | Second MRI, different intensity range. |
| CTChest | `CTChest.nrrd` | 512 x 512 x 139 | int32 | uint32, offset -3024 | 0.76 x 0.76 x 2.5 | Negative values: exercises the signed-to-unsigned rule (SRS D17) and the 32-bit path. |
| CTACardio | `CTACardio.nrrd` | 512 x 512 x 321 | int16 | uint16, offset -1024 | 0.93 x 0.93 x 1.25 | Largest real volume; negative values; 16-bit path. |

SHA-256 of each file is in `scripts/download_data.py`.

Voxel spacing is anisotropic in all four. The projection works in voxel space (SRS D15), so frames show the data with square voxels, not square millimeters.

## Synthetic volumes

`python scripts/make_synthetic.py OUT_DIR --size N` writes, for uint8, uint16 and uint32, a gradient (`M[x][y][z] = x`), a centered sphere, a checkerboard and seeded random noise. They have known answers, so the tests use them as oracles; random 500^3 volumes are also what the benchmark runs on, because projection speed does not depend on the content.

## Plane lists

CSV files for the experiments are in `docs/experiment/`; they are produced by `scripts/make_planes.py`.
