# FOAMS 1.0.5 raw-object cutoff selection

`select_foams105_raw_objects()` applies original source-style cutoffs to already-thresholded, calibrated component records while retaining image and component identity.

## Contract

- Labels are a finite, positive, strictly increasing grid of 1 to 45 unrounded values.
- One to four group records retain contiguous original slots `1..G` and immutable ordered object tuples.
- Every object has a unique `(image_id, component_id)` identity across the call and finite positive diameter and area values.
- Ranges reuse `Foams105BinnedRange`, are positional by source slot and must enable a prefix `1..K`. Slot 1 is required; disabled middle slots raise `noncontiguous_active_slots`.
- Lower-cutoff lookup rounds labels and required lower bounds with the accepted Brief 07 five-decimal half-up binary-float profile. Lookup requires exactly one match.
- For each slot below the last active slot, the effective lower bound is the unrounded global label preceding its own matched lower label. The last active slot uses its own matched unrounded label unless it is slot 4.
- Slot 4 always uses zero as its effective lower bound. Its supplied positive lower cutoff is retained only as provenance and is not looked up.
- Literal upper bounds and object diameters are not rounded. Selection is inclusive at both bounds.
- Selected rows concatenate active slots in ascending `1, 2, 3, 4` order and preserve object order within each slot. This intentionally differs from Brief 11 binned row concatenation.
- Empty groups and empty selections are valid. The selector does not measure masks, repeat threshold filtering, infer geometry, run binned selection or calculate vesicularity.

The result contains a trace for every group, complete selected/rejected index partitions, stable object identities and aligned diameter/area projections.

## Fixture integrity

The eleven hand-derived analytical cases are pinned after normalizing only CRLF pairs to LF. Their normalized SHA-256 is `f1f778b01fa96346f18ada4134a125fe48c1d776bffbbc789c13b3793e6ea407`. They are not MATLAB output or an original-workbook raw-selection oracle.

## Scope

This is a bounded replay of raw-object cutoff behavior from pinned `analysis.m`. Shape measurements, 2D vesicularity, volume normalization, density summaries, interface integration, export parity and complete original-image reconciliation remain outside this stage.

Run the report with:

```bash
python -m main.examples.foams105_raw_selection
```
