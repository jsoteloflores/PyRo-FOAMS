# FOAMS 1.0.5 Shape Measures and Statistics

This module implements the bounded Brief 17 shape calculations from
`jsoteloflores/FOAMS-1.0.5` at commit
`179663203f2d0f86b2863d5ea7f8f70dadca02f8`. It consumes accepted raw
selection, moment-geometry, and MATLAB R2025b perimeter-profile results. It
does not segment images, add a second diameter filter, create shape histograms,
write Shape/Stats exports, or orchestrate the complete pipeline.

## API

`derive_foams105_shapes(raw_selection, image_geometry_results,
image_perimeter_results)` joins by `(image_id, component_id)` in raw output
order. Each supplied image is canonically replayed once by the moment adapter
and once by the perimeter adapter. The two geometry families must have the
same per-image phase measurement, including masks, labels, counts, calibration,
and provenance. Result rows are immutable snapshots with their raw source
slot/object indices, original component records, moment orientation fields,
perimeter profile, and derived measures.

`summarize_foams105_shapes(shape_result, *, configured_scales_px_per_mm,
minimum_diameter_px, mode)` changes only the cached row view and statistics;
it does not replay moment or perimeter calculations. The scale vector is
retained in the result, including configured scales not represented by a
surviving object. Threshold millimeters are
`minimum_diameter_px / max(configured_scales_px_per_mm)`.

The explicit `initial_analysis` mode selects plot membership with strict `>`
but summarizes every post-nesting row. `updated_minimum` uses inclusive `>=`
and summarizes only those selected rows. The result reports threshold and
statistics identities/counts separately; neither operation removes or
renumbers derived rows.

## Measures and Values

Rows contain calibrated area, major/minor axes, measured perimeter, ellipse
area, regularity, form factor, elongation, aspect ratio, fitted-ellipse
perimeter, eccentricity, and the already project-shifted orientation. The
expressions use area `A`, major axis `L`, minor axis `S`, and perimeter `P`:

- Ellipse area: `((L / 2) * (S / 2)) * pi`
- Regularity: `A / ellipse_area`
- Form factor: `(4 * pi * A) / (P ** 2)`
- Elongation: `(L - S) / (L + S)`
- Aspect ratio: `S / L`
- Fitted-ellipse perimeter: `2 * L * (2 + (pi - 2) * (S / L) ** 1.456)`

The fitted perimeter preserves the original factor and exponent. Aspect ratio
is `S / L`; ratios are not clamped. Eccentricity and the shifted orientation
are carried from the validated moment result without further transformation.

`Foams105ShapeValue` carries ordinary finite results with status `defined`.
For zero perimeter, form factor is represented as `value=None`, status
`undefined`, reason `zero_perimeter`, and legacy class `positive_infinity`.
The object and its other measures remain available. Arithmetic overflow and
nonzero underflow raise contextual numerical errors instead of being hidden.

The four statistics rows and seven columns follow the source order:
`aspect ratio`, `elongation`, `regularity`, `eccentricity`; `N`, max, min, mean,
sample standard deviation, median, mode. Aspect ratio, elongation, and
regularity are finite-filtered independently with included/excluded identities
retained. Eccentricity comes from validated finite geometry and is not
independently filtered. Form factor is not a statistics row; area and diameter
statistics are not presented as original outputs.

The rows provide exact integer `N`, unrounded six numeric statistics, and
four-decimal display values. Presentation rounding follows the observed
R2025b `roundn(x,-4)` operation order: multiply the binary64 value by
`10000.0`, round the scaled absolute value to nearest integer with ties away
from zero, restore the sign, then divide by `10000.0`. The implementation uses
`floor` plus the fractional remainder, not Python's ties-to-even `round` and
not Decimal quantization of the original binary value. Literal MATLAB checks
give `0.00035 -> 0.0004`, `-0.00035 -> -0.0004`, and `0.03125 -> 0.0313`; the
adjacent binary64 values around `0.00035` produce `0.0003` below and `0.0004`
above. The independent MATLAB commands executed `roundn.m` from the installed
R2025b environment and both exited successfully. This narrowly validates the
rounding primitive only; no MATLAB `basicstats.m` execution, full historical
statistics parity, or original workflow parity is claimed.

Reduction is sequential float64 in raw-selection row order; bitwise MATLAB
summation parity is not claimed. Modes are counted on full-precision values,
with the smallest value selected for frequency ties. Empty populations return
`N=0`, unavailable status, reason `empty_population`, and `None` for the other
six values. This is an explicit API behavior, not a claim about `basicstats.m`
empty-input output.

## Verification Boundary

Moment geometry retains its analytical verification level and unchanged
`pending_historical_profile` perimeter marker. The combined result records
moment and perimeter provenance separately; it does not upgrade moment or
orientation verification. MATLAB R2025b perimeter-profile reference evidence
does not establish the historical runtime used for original workbooks or
full-workflow parity. Cached shape results require `mm`, `mm^2`, and `degree`
units; summary validation rejects metadata changes without rescanning image
pixels or replaying geometry. Each unit mismatch raises a contextual
`incompatible_shape_result_unit` validation error with field, expected, and
actual values. Floating-point reductions are ordinary source-order
calculations, not bitwise MATLAB comparisons.

Run the executable sample with:

```powershell
python -m main.examples.foams105_shape_statistics
```

It demonstrates the singleton's zero perimeter, the `initial_analysis`
population discrepancy, and the recalculated `updated_minimum` population.
Shape histogram binning and actual Shape/Stats writers remain explicit Brief
18 work; original-run reconciliation is Brief 19.