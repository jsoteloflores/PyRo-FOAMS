# FOAMS 1.0.5 mask measurements

`main.core.measure_foams105_phase` provides a separate source-style path for
one exact image phase. It performs the sequence from `data_meas.m` at pinned
FOAMS commit `179663203f2d0f86b2863d5ea7f8f70dadca02f8`:

1. Select pixels by exact equality without dtype conversion.
2. Remove complete border-touching components with 8-connectivity.
3. Fill holes using 4-connected background reachability.
4. Label filled foreground with 4-connectivity in column-major discovery order.
5. Count integer pixels and calculate calibrated area and equivalent diameter.

`main.core.measure_foams105_image` measures the pore phase and exactly four
stable optional exclusion slots. `None` replaces the original infinity-based
disabled convention. Enabled phases are processed independently; duplicate
values and pore/exclusion collisions are preserved and reported rather than
merged.

Result records are frozen. Mask and label arrays are independently owned,
immutable-buffer-backed NumPy arrays, so ordinary writes fail and later caller
mutation cannot alter results. Component area is a pixel count, not contour
area or `bwarea`.

## Verification boundary

The bundled fixture contains seven hand-authored analytical masks covering
border connectivity, 4-connected object separation and hole filling,
column-major IDs, and empty output. It is not MATLAB-generated output or
original acquisition data. The report compares all complete masks, label
images, component areas, calibrated values, and conservation identities with
explicit runtime failures:

```bash
python -m main.examples.foams105_mask_measurements
```

This stage does not implement `regionprops` axes, orientation, eccentricity,
perimeter, `bwarea`, excluded-component thresholds, corrected denominators,
magnification aggregation, nesting, or GUI integration. Existing modern
contour and ellipse measurements remain separate and are not relabeled as
original equivalents.