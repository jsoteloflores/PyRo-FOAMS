# FOAMS 1.0.5 perimeter reference tooling (Brief 16a)

This handoff adds capture and comparison tooling for perimeter reference
artifacts. It does not implement a production perimeter estimator and does not
claim historical MATLAB parity.

## Scope and boundary

- Source pin: `jsoteloflores/FOAMS-1.0.5@179663203f2d0f86b2863d5ea7f8f70dadca02f8`.
- Reference workflow is tooling-only: case preparation, optional MATLAB capture,
  strict import validation, and deterministic numerical comparison reports.
- Perimeter values in this workflow are evidence inputs, not production outputs.
- `historical_compatible` is always reported separately from schema/topology/
  numerical validation states.

## Delivered artifacts

- Case catalog:
  - `main/tests/fixtures/16a_foams105_perimeter_cases.json`
- Python reference module:
  - `main/reference_tools/perimeter_reference.py`
- Python CLI wrappers:
  - `reference-tools/perimeter/prepare_perimeter_reference.py`
  - `reference-tools/perimeter/import_perimeter_reference.py`
  - `reference-tools/perimeter/compare_perimeter_reference.py`
- MATLAB capture runner (legacy-friendly TSV output):
  - `reference-tools/matlab/foams105_perimeter_capture_tsv.m`
- Existing MATLAB moment harness ordering fix:
  - `reference-tools/matlab/foams105_moment_regionprops_reference.m`

## Preparation step (required)

Generate deterministic MATLAB constructor + hash manifest:

```bash
python reference-tools/perimeter/prepare_perimeter_reference.py
```

Default outputs:

- Constructor: `reference-tools/matlab/foams105_perimeter_cases_generated.m`
- Manifest: `reference-tools/perimeter/foams105_perimeter_manifest.json`

The manifest records:

- `case_catalog_hash`
- `preparation_hash`
- `runner_hash`
- `constructor_hash`
- case count and source pin

## MATLAB capture step (optional external runtime)

Run in MATLAB (when available):

```matlab
foams105_perimeter_capture_tsv('foams105_perimeter_cases_generated', 'reference-tools/matlab/perimeter-run', 'manual capture command string')
```

Expected output directory files:

- `metadata.tsv`
- `cases.tsv`
- `labels.tsv`
- `components.tsv`
- `completed.marker` (must contain `COMPLETE`)

Notes:

- Raw-phase cases are replayed in MATLAB via the pinned sequence:
  phase selection -> `imclearborder(...,8)` -> `imfill(...,'holes')` ->
  `bwlabel(...,4)`.
- Final-label cases are used directly as provided in the prepared constructor.
- Runner preserves native labels and exports both native and canonical IDs.
- Canonical IDs are defined by ascending first pixel in MATLAB column-major
  order.
- `Perimeter` and `PerimeterOld` statuses are independent and explicit.
- `PerimeterOld` unsupported detection is probed independently from per-case
  runtime failures.
- No synthetic replacement values are emitted for unsupported or error states.

## Import validation

Validate a capture directory against the catalog and expected hashes:

```bash
python reference-tools/perimeter/import_perimeter_reference.py reference-tools/matlab/perimeter-run
```

Importer checks include:

- schema/source/hash compatibility
- complete case coverage, stage/shape/scale consistency, and unique identities
- full label matrix reconstruction and exact topology match
- finite nonnegative perimeter values for successful rows
- explicit property statuses and completion marker

## Comparison reports

Perimeter vs PerimeterOld within one run:

```bash
python reference-tools/perimeter/compare_perimeter_reference.py reference-tools/matlab/perimeter-run --mode within-run --output-json reference-tools/perimeter/report-within-run.json
```

Run-to-run perimeter comparison:

```bash
python reference-tools/perimeter/compare_perimeter_reference.py reference-tools/matlab/perimeter-run --mode run-to-run --other-run-dir path/to/other/run --output-json reference-tools/perimeter/report-run-to-run.json
```

Console summary includes mode, schema/topology/numerical/historical states,
expected/comparable/skipped/mismatch counts, and explicit skipped/error reasons.

JSON output includes compared rows with identities, pixel and calibrated values,
absolute and relative differences, tolerance verdicts, and skipped/error reasons.

CLI exit status:

- `0`: complete numerical agreement for selected mode
- `1`: numerical disagreement
- `2`: invalid input/evidence or incomplete/no-comparable comparison

Comparison tolerances:

- absolute: `1e-12` pixels
- relative: `1e-12`

Relative difference zero-reference rule:

- both zero -> relative difference `0`
- zero reference, nonzero candidate -> relative difference undefined (reported as
  absent), absolute difference still reported

Run comparison coverage contract:

- requires identical case coverage, stage, dimensions, and scale
- requires full per-case label matrix equality before numerical comparison
- reports skipped/error row reasons explicitly when statuses are not comparable

## Reviewer handoff checklist

Provide these files from a runtime capture:

1. `metadata.tsv`
2. `cases.tsv`
3. `labels.tsv`
4. `components.tsv`
5. `completed.marker`
6. command string used to invoke MATLAB capture
7. constructor and manifest hashes used for the run
8. comparison command(s) executed
9. generated JSON report file(s)

## Current evidence status

- Tooling and CI-safe synthetic transport tests are implemented.
- No MATLAB runtime evidence is bundled by default in this repository.
- MATLAB availability is an external dependency for runtime evidence capture.
- Historical perimeter equivalence remains pending Brief 16b review and method
  acceptance.
