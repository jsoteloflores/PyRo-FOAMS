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

Run all documented commands from the checked repository root. The three direct
Python wrappers locate the repository package relative to their own script
paths and do not require `PYTHONPATH`; their default input/output paths are
still relative to the repository-root working directory.

Generate deterministic MATLAB constructor + hash manifest:

```bash
python reference-tools/perimeter/prepare_perimeter_reference.py --constructor reference-tools/matlab/foams105_perimeter_cases_generated.m --manifest reference-tools/perimeter/foams105_perimeter_manifest.json
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
addpath('reference-tools/matlab');
foams105_perimeter_capture_tsv('foams105_perimeter_cases_generated', 'reference-tools/matlab/perimeter-run-YYYYMMDD-HHMMSS', 'MATLAB <release> perimeter reference capture <unique-id>')
```

Use a new, unique output directory for every attempt. The runner replaces an
existing output directory and its `.tmp` sibling. Retain the full MATLAB
console log and process exit status separately, including failed attempts.

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
python reference-tools/perimeter/import_perimeter_reference.py reference-tools/matlab/perimeter-run-YYYYMMDD-HHMMSS --catalog main/tests/fixtures/16a_foams105_perimeter_cases.json --manifest reference-tools/perimeter/foams105_perimeter_manifest.json
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
python reference-tools/perimeter/compare_perimeter_reference.py reference-tools/matlab/perimeter-run-YYYYMMDD-HHMMSS --mode within-run --catalog main/tests/fixtures/16a_foams105_perimeter_cases.json --manifest reference-tools/perimeter/foams105_perimeter_manifest.json --output-json reference-tools/perimeter/report-within-run.json
```

Run-to-run perimeter comparison:

```bash
python reference-tools/perimeter/compare_perimeter_reference.py reference-tools/matlab/perimeter-run-YYYYMMDD-HHMMSS --mode run-to-run --other-run-dir path/to/other/run --catalog main/tests/fixtures/16a_foams105_perimeter_cases.json --manifest reference-tools/perimeter/foams105_perimeter_manifest.json --output-json reference-tools/perimeter/report-run-to-run.json
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
10. full MATLAB log, runtime/toolbox information, and MATLAB process exit status
11. SHA-256 hashes for the retained catalog, generated constructor, manifest,
    runner, capture TSV files, and report

## Current evidence status

- Tooling and CI-safe synthetic transport tests remain available for preparing,
  importing, and comparing new runtime captures.
- Reviewer-supplied MATLAB R2025b evidence is bundled under
  `main/tests/fixtures/16b_perimeter_evidence/`. It contains the 29-case,
  31-object primary capture and 64 supplementary masks with 526 components.
  The primary `narrow_neck_final` component is retained as diagnostic input; it
  is not a valid four-connected production measurement.
- `main.core` exposes `measure_foams105_perimeter` and
  `select_foams105_perimeter` with explicit profiles
  `matlab_perimeter_old_r2025b_verified_v1` and
  `matlab_perimeter_r2025b_verified_v1`. The example
  `python -m main.examples.foams105_perimeter_profiles` replays the 526
  supplementary objects and makes intentional profile differences visible.
- Both profiles match the valid primary reference components and all 526
  supplementary MATLAB values within relative/absolute `1e-12` pixel
  tolerance. This verifies these R2025b properties only; it does not identify
  the historical MATLAB release used for original FOAMS workbooks or establish
  full original-workflow parity.
- A separate local R2025b capture attempt is retained in
  `reference-tools/matlab/perimeter-evidence-r2025b-20261016-004729-c559cf4f`.
  That attempt is blocked because local Image Processing Toolbox is
  unavailable and is not the source of the bundled evidence above.
