# FOAMS 1.0.5 size-class preparation

The focused APIs in `main.core` replay four operations from `project.m` in
`jsoteloflores/FOAMS-1.0.5` commit
`179663203f2d0f86b2863d5ea7f8f70dadca02f8`:

- `build_foams105_bin_labels` creates 45 unrounded recurrent values and rounds
  them to five decimal places using the documented exact-scaled-half-ties-up
  profile. This profile has not been benchmarked against the historical MATLAB
  runtime.
- `filter_foams105_diameters` applies the original unrounded equivalent-diameter
  threshold for one caller-identified magnification group.
- `count_foams105_histogram` counts already-filtered diameters with retained
  `histc` ownership and reports the discarded upper tail.
- `normalize_foams105_counts` divides counts by a finite positive corrected area
  supplied by the caller.

All results are immutable and carry the pinned source repository, commit, and
file alongside their distinct method and unit metadata. Normalization source
provenance identifies the arithmetic; its area remains explicitly
`caller_supplied_not_computed_here`. Rounded zero or duplicate labels are
preserved when positive raw values round to zero. A raw first label that
underflows to zero during division is rejected. The accepted post-nesting
converter remains stricter and continues to require positive selected labels.

```python
from main.core import (
    build_foams105_bin_labels,
    count_foams105_histogram,
    filter_foams105_diameters,
    normalize_foams105_counts,
)

labels = build_foams105_bin_labels(10, (100,))
filtered = filter_foams105_diameters((0.05, 0.1, 1.0), 10, 100)
histogram = count_foams105_histogram(
    (1.0, 2.0, 4.0),
    (0.5, 1.0, 1.5, 2.0, 3.9, 4.0, 5.0),
)
normalized = normalize_foams105_counts(histogram, 2.0)
```

## Verification boundary

The bundled JSON fixture is an independent extraction from `Raw_data.xls` and
`NA_mag.xls`. Tests compare all 180 count cells across four groups containing
777, 135, 2,076, and zero observations. All 2,988 supplied observations are
conserved, and none occupy the discarded upper tail.

The workbook diameters were already thresholded by the original pipeline. Its
minimum-pixel setting, scales, corrected areas, and image-processing settings
are not established, so the fixture does not verify threshold recovery, area
calculation, image measurements, manual range selection, AutoSmart, or full
image-to-output compatibility. Source `N_A` values are retained only for later
area reconciliation and are not used to derive an area here.

Run the headless report with:

```bash
python -m main.examples.foams105_size_classes
```