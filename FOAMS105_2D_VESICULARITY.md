# FOAMS 1.0.5 two-dimensional vesicularity

This stage reproduces the bounded two-dimensional vesicularity arithmetic in
the pinned FOAMS 1.0.5 `analysis.m` and its per-image area inputs from
`project.m`.

## Public API

`calculate_foams105_2d_vesicularity(raw_selection, groups)` accepts a canonical
`Foams105RawSelectionResult` and one `Foams105VesicularityGroupInput` per
source slot. Each group supplies accepted image-area results and all measured
pore objects before diameter filtering and raw cutoffs.

The result retains:

- the complete canonical raw-selection replay;
- source-selected and unfiltered object identities and area sums;
- accepted per-image and aggregate area provenance;
- independent source area assignments, recurrence multipliers, removed areas,
  and residuals;
- integrated, per-group, and arithmetic-mean percentages;
- contextual diagnostics for reassignment and signed nonphysical results.

## Source arithmetic

Both integrated and per-group calculations use original `area2`, represented by
`border_corrected_area_mm2`. Original `area1`, represented by
`phase_corrected_area_mm2`, and its exclusion diagnostics remain provenance and
do not become vesicularity denominators.

For integrated vesicularity, image scales are sorted ascending while individual
image area2 values are independently sorted descending, with equal areas
retaining flattened input order. Area values are paired positionally with the
sorted scales and summed by scale slot. This source behavior can reassign area
values between actual image groups and is preserved explicitly.

With source denominators $A_s$, selected raw-object areas $S_s$, and residuals
$R_s$, the recurrence is:

$$
R_1 = A_1 - S_1
$$

$$
R_s = R_{s-1} - \left(\frac{R_{s-1}}{A_s}\right) S_s
$$

The integrated percentage is:

$$
100\left(1 - \frac{R_G}{A_1}\right)
$$

Per-group image vesicularity instead uses actual image membership. For
unfiltered measured pore area $U_s$ and actual group area2 sum $D_s$:

$$
V_s = 100\frac{U_s}{D_s}
$$

The reported mean is the arithmetic mean of all declared group percentages,
including disabled trailing raw-selection groups.

## Validation boundary

The public function replays raw selection once and requires complete result
equality. It also requires stable global image/component identities, exact
object geometry in the unfiltered inventory, compatible accepted area records,
positive per-image area2, strictly increasing source-slot scales, and no more
than 20 images. Nonfinite arithmetic and nonzero underflow raise typed errors.
Signed finite percentages and residuals are preserved and diagnosed rather
than clipped.

The bundled 13-case fixture contains independent hand-derived numerical
expectations. Public integration tests separately exercise coherent masks,
measurements, areas, raw cutoffs, object ownership, disabled groups, pooled
images, tie order, and source reassignment. The fixture is not MATLAB output,
workbook evidence, or original acquisition data. This stage does not establish
volume normalization, density, shape, GUI, export, original-image, or complete
workflow parity.

Run the synthetic public-API report with:

```bash
python -m main.examples.foams105_2d_vesicularity
```
