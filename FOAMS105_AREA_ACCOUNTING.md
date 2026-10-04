# FOAMS 1.0.5 area accounting

This stage reproduces the bounded per-image area arithmetic used by the pinned
FOAMS 1.0.5 source and makes magnification-group aggregation explicit. It
consumes `Foams105ImageMeasurementResult`; it does not rerun mask topology.

## Public API

- `estimate_foams105_binary_area(mask)` applies the documented padded 2x2
  weighted-neighborhood estimator and returns pixel-area units.
- `calculate_foams105_image_areas(image_id, measurement_result,
  excluded_min_area_mm2)` computes full, border, and four positional excluded
  areas. A component is selected only when its measured area is strictly
  greater than its slot threshold.
- `aggregate_foams105_group_areas(group_id, image_area_results)` sums compatible
  image terms in caller-supplied order and retains the contributing image IDs.

The original variable mapping is explicit: `area2` is
`border_corrected_area_mm2`, while `area1` is
`phase_corrected_area_mm2`. Denominators are signed results. Zero or negative
values are preserved and diagnosed rather than clipped or silently repaired.
Duplicate phase values remain independent positional slots, so overlapping or
identical exclusions are subtracted once per enabled slot and reported through
collision diagnostics.

## Validation boundary

The bundled fixture contains six weighted-mask expectations and two integrated
analytical image cases. It is hand-authored from documented weights and pinned
source formulas; MATLAB and original images were not run. This stage does not
implement vesicularity, automatic magnification nesting, shape statistics,
GUI behavior, or full original-workflow equivalence.

Run the report with:

```bash
python -m main.examples.foams105_area_accounting
```