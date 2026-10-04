# FOAMS 1.0.5 post-nesting conversion

`main.core.convert_foams105_nv(bin_labels_mm, na_per_mm2)` replays the
post-nesting `N_A`-to-`N_V` conversion in `analysis.m` from
`jsoteloflores/FOAMS-1.0.5` commit
`179663203f2d0f86b2863d5ea7f8f70dadca02f8`.

The function accepts one to 45 already nested original-style labels in mm and
bin-integrated `N_A` in mm^-2. Labels must be finite, positive, and
nondecreasing. The pinned workbook contains a repeated label, and the original
nesting code permits repeated labels through concatenation and inclusive
selection. The workbook's acquisition settings have not been reconstructed,
so the cause of its repeated pair is not established. Descending labels are
rejected. Optional `length_unit` and `input_density_unit` arguments
must remain `mm` and `mm^-2`; the converter never guesses units.

The immutable result includes the source probability sequence, alpha
coefficients, cube-volume midpoint heights, larger-class contributions, signed
`N_V` in mm^-3, negative indices, adjacent duplicate-label index pairs, and
pinned source provenance. Its smallest
selected class deliberately uses
`legacy_no_larger_class_subtraction`, matching this source version rather than
repairing that behavior.

```python
from main.core import convert_foams105_nv

result = convert_foams105_nv(
    (1.0, 2.0, 4.0),
    (8.0, 3.0, 1.0),
)
print(result.signed_nv_per_mm3)
```

## Verification boundary

The bundled `Calc_out.xls` JSON extraction contains 29 independently supplied
workbook rows. Tests compare every candidate value with relative tolerance
`1e-12`. The fixture was extracted from the workbook; ordinary tests do not
run Excel or MATLAB.

This verifies only the FOAMS 1.0.5 **post-nesting conversion stage**. It does
not establish complete image-to-output compatibility. Original histogram edge
ownership, rounded-label preparation, area corrections, measurements,
automatic transitions, vesicularity, volume normalization, shape outputs, UI,
and exports remain separate milestones. The modern
`spherical_upper_edge_triangular_v1` solver remains an independent API.

Run the comparison report with:

```bash
python -m main.examples.foams105_compatibility
```
