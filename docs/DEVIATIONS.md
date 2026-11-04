# Deviations from the assignment

The assignment describes the projection as pseudo-code and leaves some points open. This page lists every place where this project does something other than what the text says, and why. Gaps the text leaves open (with no text to deviate from) are resolved in the [SRS](SRS.md#5-design-decisions); the decision numbers (D1 to D18) are the same in both documents.

The assignment's pseudo-code, as this project reads it (triangle `a, b, c`):

```
Dab = |b - a|, Dbc = |c - b|, vab = (b - a) / Dab, vac = (c - a) / |c - a|
E = matrix of Dab x Dbc;  a1 = a2 = a
for y < Dab:
    a1 = a1 + vab;  a2 = a2 + vac
    p = a1;  d12 = |a2 - a1|;  dp = (a2 - a1) / d12
    for x1 <= d12:
        v = M[p];  x = Dbc/2 - d12/2 + x1;  E[x][y] = v
        p = p + dp
```

## Deviations

| # | Topic | Assignment | This project | Reason | Effect on the image |
|---|---|---|---|---|---|
| D3 | First row | `a1` and `a2` move before the first row is read, so the apex row is never sampled; with `y < Dab` the last row can land beyond the base | Row `y` reads `a1 = a + y * vab`, for `y = 0 .. floor(Dab)` | The apex is part of the triangle, and no row should fall outside it | Our image has the apex as row 0; the assignment's row `y` is our row `y + 1` |
| D4 | Image size | `Dab x Dbc`, not integers in general; the widest row has `d12 + 1` samples (`x1 = 0 .. d12`), one more than `Dbc` columns | `H = floor(Dab) + 1` rows, `W = floor(Dbc) + 1` columns | Every sample needs a pixel, including both ends of each row | One more row and column than the text, so nothing is clipped |
| D5 | Column of a sample | `x = Dbc/2 - d12/2 + x1`, a float | `x = floor((W - 1)/2 - d12/2 + x1 + 0.5)` | An index must be an integer; centering on `(W - 1)/2` fits the widest row exactly | Same center as the text when `Dbc` is an integer, less than one pixel away otherwise |
| D6 | Indexing | `E[x][y]` | `E[y][x]` | Row-first is the layout of NumPy images and of video frames | None: the same picture, stored the usual way |
| D7 | Position along a row | `p = p + dp`, accumulated | `p = a1 + x1 * dp`, computed directly | No accumulated rounding error, and the three backends give bit-identical results | Mathematically equal; differences only in the last floating-point bits |
| D9 | Shape of the plane | Isosceles triangle or trapezoid | Any triangle or trapezoid with the stated point order; a plane that is not isosceles gets a warning and is projected as given | The lecturer's own example sheet has a triangle with legs 497.4 and 476.9 | None for isosceles planes; others are drawn instead of rejected |
| D14 | Planes file | "A list of 3 or 4 points" | The lecturer's sheet format (a point count, then the coordinates) and two simpler layouts | The text leaves the layout open; the sheet fixes it | None |
| D17 | Value types | Unsigned 8, 16 and 32 bits | Signed 8, 16 and 32 bits too, shifted to unsigned when negative | Real scans are often signed: the lecturer's `MRHead_1.nrrd` is `int16`, CT data is negative in air | None for unsigned data |
| D2 | Sampling | `v = M[p]` for a point `p` with fractions | Nearest voxel, `floor(p + 0.5)` on each axis (D8); trilinear interpolation as an option | The text does not say how a fractional point picks a voxel | See the measurement below |

Additions that the text does not ask for, and that do not change the default output: trilinear sampling (`--interp trilinear`), world coordinates (`--world`), the choice of backend (`--backend`).

## Where this project follows the text on purpose

**Row spacing along the legs.** Each row moves one unit along the legs (`a1 += vab`), as in the text, so a triangle's image has `floor(Dab) + 1` rows. Another reading spaces rows one unit along the triangle's height instead, which keeps the true proportions (as 3D Slicer shows a cut) but has `floor(height) + 1` rows: for a triangle whose base equals its height, `Dab = 1.118 * height`, so that image is about 11 percent shorter. The lecturer said outputs should match his "up to pixels", so this project keeps his spacing. Within each row, the values read are the same as 3D Slicer's for the same points ([EXPERIMENT.md](EXPERIMENT.md#comparison-with-3d-slicer)).

## How far the deviations move the output

A literal transcription of the pseudo-code above was compared once with `ReferenceProjector` on 200 random isosceles triangles in a 40^3 volume whose every voxel holds a unique value, and on 40 random triangles through MRHead. The script was a one-off check and is not part of the repository.

- With `M[p]` read as the nearest voxel, every pixel of the literal version appears in our image within one pixel; 76 to 80 percent are in exactly the same pixel once our image is shifted by the one row of D3. On MRHead the mean difference was about 2.9 percent of the mean value.
- With `M[p]` read by truncation (`int(p)`, as a C or Java cast would), only about 48 percent match, and the mean difference on MRHead was about 10 percent. The rounding of `M[p]` is therefore the one open point that changes the output noticeably (assumption A3 in the [SRS](SRS.md#6-assumptions-to-confirm-with-the-lecturer)).
