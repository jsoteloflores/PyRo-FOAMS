# FOAMS 1.0.5 Pipeline and Exports

The headless workflow accepts already segmented, label-valued 2D images. It
composes the accepted phase measurement, area accounting, size-label/filter,
histogram/normalization, binned selection, raw selection, N_A-to-N_V
conversion, vesicularity, volume, summary, moment, perimeter, and shape APIs.
It does not replace segmentation or the modern GUI workflow.

## Public Workflow

`prepare_foams105_analysis(groups, preparation_settings, *, source_metadata)`
owns image snapshots and caches phase measurements, area1/area2 records,
moment/perimeter geometry, minimum-filter partitions, complete histograms,
and source-normalized N_A where area1 is positive. Group order and source slots
must already be contiguous and scale-ascending. It never sorts groups to make
them acceptable. If area1 is nonpositive, the prepared result retains the
histogram and marks normalization unavailable; analysis then fails at the
required binned-selection stage with the typed cause.

`analyze_foams105_prepared(prepared, analysis_settings)` requires an explicit
`manual` or `autosmart` selection mode and one explicit volume normalization
mode. Manual and AutoSmart never fall back to one another. Raw-object
selection is independently replayed from the same requested ranges and the
diameter-filtered object identities; it is not derived from binned rows.
Prepared phase arrays are owned and read-only, so caller mutation cannot alter
the cached preparation. Reanalysis reuses measurements, areas, filtered
objects, and complete histograms. Accepted result adapters may perform their
required canonical replays.

`run_foams105_analysis(...)` is a convenience wrapper around the two explicit
stages. Table projection and writing are separate:

- `build_foams105_export_tables(analysis_result)` projects stored results only.
- `write_foams105_export_bundle(analysis_result, output_directory)` writes a
  UTF-8 TSV/JSON bundle atomically into a new directory and refuses collisions.

Run the synthetic example in PowerShell:

```powershell
python -m main.examples.foams105_pipeline_export C:\temp\foams105-export-demo
```

The example uses active checks (including under `python -O`) and does not claim
original-image parity.

## Source-to-Field Mapping

| File | Source population and field mapping |
|---|---|
| `NA_mag.txt` | All 45 prepared global labels and each group's full pre-nesting histogram counts/N_A; slots absent from configuration receive source-style zero columns, while configured disabled groups retain their actual complete vectors. |
| `Raw_data.txt` | Every diameter-filtered pre-nesting component, ordered by source slot then original per-group object order; `EqDi`, area, perimeter, major/minor axes, project-shifted orientation, and integer slot as `Mag type`. |
| `Rawcutoffs_data.txt` | Post-nesting raw selection in the accepted raw-selection order, with the same geometry fields and identities retained in a sidecar. |
| `Calc_out.txt` | Selected binned/converter order: labels, N_A, signed N_V, adjusted volume fraction, cumulative adjusted fraction, log cumulative N_V, log label, and log number density per width. Undefined real values serialize as `NA` with a diagnostic. |
| `Area_out.txt` | Per-image phase-corrected area1 in the pinned source-key order: `mag1`, `first`, `second`, `third`, `fifteenth`, `fourth`, `fifth`, `sixth`, `seventh`, `eigth`, `ninth`, `tenth`, `eleventh`, `sixteenth`, `seventeenth`, `twelth`, `thirteenth`, `fourteenth`, `eighteenth`, `nineteenth`. Selected image rows use source-style one-based image numbers; the identity sidecar retains the source key. |
| `Ves_out.txt` | Mean group percentage; per-slot percentages; N_V-derived percentage; volume normalization factor; caller-supplied density-derived vesicularity; and integrated 2D percentage. Absent slots are zero. Missing density-derived metadata is an explicit `NA`/partial table, never a guessed zero. |
| `Misc_out.txt` | Post-nesting object count, included image count, total N_V, corrected N_V, supplied category cutoffs, then upper/middle/lower category percentages. |
| `Shape_out.txt` | All post-nesting objects, summarized as captured MATLAB `hist`/`histc` counts for eccentricity, elongation, aspect ratio, and regularity. The additional shape-threshold population is not used here. |
| `Stats_out.txt` | Four source-order metric rows (`asp_rat`, `elong`, `regularity`, `eccent`) with exact integer N and the accepted four-decimal display values. The top-left header is empty. |

The source slot-4 bulk-N_A quirk is preserved: its density indices select slot
3 counts and use slot 3 area1 as denominator, while its x coordinate remains
slot 4 area1. `source_slot4_uses_slot3_counts_and_area` is explicit metadata.
Literal x values are filtered by the source's nonzero-area operation, while y
retains one status-bearing entry per configured slot. A zero area can therefore
produce mismatched vector lengths; the projection is marked unplottable rather
than compacted, zipped, or truncated.

For `Area_out`, exact-zero area values are removed only for the source table
projection. Because source image numbers are constructed before that removal,
any length mismatch makes only `Area_out.txt` unavailable; other tables remain
eligible. A separate identity sidecar retains every input image and its area.
Negative nonzero area values remain signed and ordered.

## Integrity and Limits

The export bundle includes a deterministic manifest, SHA-256 checksums,
diagnostics, raw/shape identity sidecars, and selected density source-index
sidecars. Input hashes include dtype and shape plus contiguous C-order bytes.
Scientific floats use 17 significant digits; only the `Stats_out.txt` display
statistics use four decimals. The manifest contains no timestamp or random ID.

The MATLAB shape-histogram fixture validates only the captured R2025b `hist`
and `histc` arrays. Moment orientation remains analytical, perimeter profiles
remain R2025b-specific, and this handoff is not complete original-workflow or
original-image parity. Brief 19 remains responsible for original-run
reconciliation and performance; GUI integration and improvements remain
Briefs 20 and 21.

The unchanged `main/tests/fixtures/18_foams105_histogram_reference.json` has
SHA-256 `6ee281b1a7be253e04b07d5c1d461b18fa9b2324e88af37127b49522c14c674f`.
Its R2025b capture log is `18_matlab_histogram_review.log` with SHA-256
`35c4a92bb07917db5cc436ebd64ee0bcc3336b21644070a359cabd6ddb0cb18f`. The
preceding Brief 17a runtime evidence is retained as
`17a_matlab_rounding_review.log` (SHA-256
`a6808737ee413bbc42f3b842d7b8d40a9aca296cc6e276a14e6e6e978400323a`) and
`17a_matlab_rounding_boundaries.log` (SHA-256
`00e66ac3529207989686ae27b6c3be5aec37cb6f76bee271d7a4a427f4b63937`).