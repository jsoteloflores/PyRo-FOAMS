# Original FOAMS completion tracker

Last updated: 2026-10-16.

**Original FOAMS implementation: NOT COMPLETE.**

Target reference: `jsoteloflores/FOAMS-1.0.5` at commit
`179663203f2d0f86b2863d5ea7f8f70dadca02f8`.

## Milestones

| Stage | Status | Evidence required or retained |
|---|---|---|
| 1. Post-nesting N_A-to-N_V conversion | Brief 06 accepted against bundled Calc_out fixture | All 29 rows, source alpha order, heights, duplicate metadata, validation, and smallest-class exception tested |
| 2. Measurement, area, and histogram conventions | Briefs 07, 08, and 09 accepted; shape measurements pending | 45-label preparation, all 180 histogram cells, exact mask topology, calibrated component geometry, six weighted-area cases, strict four-slot exclusion thresholds, signed per-image denominators, and explicit compatible-group sums tested |
| 3. Automatic magnification cutoffs | Briefs 10-12 accepted in configured CI with Brief 13 commit `ffc59f5` | Signed log-density transitions, exact binned source slices, LF/CRLF fixture pins, raw predecessor bounds, stable component identities, inclusive unrounded filtering, and ascending raw-slot provenance tested |
| 4. 2D vesicularity and phase accounting | Brief 13 accepted in configured CI run `38417127328` | Independent scale/area sorting, source recurrence, actual-membership summaries, area2 denominators, area1 provenance, signed diagnostics, diameter filtering, and cutoff-separated inventories tested |
| 5. Volume distributions and normalization | Brief 14 accepted in configured CI run `38842930126` | Equivalent-sphere row volumes, all three normalization modes, signed and duplicate rows, ordered cumulative fractions, and contextual numerical failures tested |
| 6. Density summaries and plotting quantities | Brief 15 accepted after Brief 15a correction commit `82033e0`; configured CI run `38846116435` | NV/NVcorr, exact-grid widths, ordered forward and reverse sums, logarithms, paired volume bins, strict size-category percentages, and Python 3.11/3.12 portability tested |
| 7. Shape outputs and statistics | Briefs 16-18a accepted within bounded analytical/R2025b scopes | Moment orientation remains analytical; R2025b perimeter and roundn checks are narrow reference tests; original-run reconciliation exposed strict-threshold identity and eccentricity-statistics convention mismatches |
| 8. Analysis interface | Headless Brief 18 workflow accepted; GUI integration remains Brief 20 | Preparation/analysis APIs compose accepted stages and preserve source slots; GUI remains pending |
| 9. Original-equivalent exports | Nine deterministic TSV projections accepted in Brief 18a | All nine table families and sidecars are produced; comparison found no differences outside Calc_out and Stats_out, subject to documented Area_out/Ves_out availability rules |
| 10. Full workflow reconciliation | Brief 19 first MATLAB replay and benchmark complete; parity blocked on recorded deviations | Pinned source/runtime capture, 7-image/10-case catalog, 7,786 comparisons, mismatch report, and synthetic benchmarks are retained; hosted CI and original-image/GUI parity remain pending |

## Current verified boundary

The FOAMS 1.0.5 post-nesting `N_A`-to-`N_V` converter is verified against the
bundled 29-row `Calc_out.xls` extraction. Size-class preparation is separately
verified for source-derived analytical boundaries and all 180 saved-workbook
count cells from 2,988 already-thresholded observations. The mask-measurement
path is checked against seven hand-authored analytical topology cases; these
are not original images or MATLAB-generated outputs. The artifacts record
workbook hashes and the pinned source version. MATLAB was not executed as part
of these tests. The current boundary does not establish original-image parity,
shape-property equivalence, end-to-end nesting selection, or full workflow
parity. Corrected-area calculations are implemented from the measured masks
and components, with signed nonpositive denominators retained as diagnostics.
AutoSmart default cutoff suggestions are accepted. Explicit and suggested
binned ranges now replay checked source-style selection and concatenation,
including duplicate boundary rows and source-slot provenance. Raw-object
selection retains stable component identities and checked source-slot cutoffs.
The 2D vesicularity stage now preserves the original independent scale/area
sorting recurrence while reporting actual-membership per-group summaries.

Brief 11a changes only the test-local integrity calculation: CRLF checkout
pairs are normalized to LF before hashing the fixture. The existing LF digest
is retained, and regression checks prove both LF/CRLF equivalence and
sensitivity to a changed numerical value. No fixture values or scientific
selection code changed. Acceptance remains pending until the correction has a
commit and all configured CI jobs pass.

Brief 12 adds a bounded raw-object cutoff replay with active-prefix validation,
the original predecessor-label rules, slot 4's zero lower bound, stable image
and component identities, and aligned diameter/area projections. Its eleven
cases are hand-derived analytical expectations, not a MATLAB run or workbook
raw-selection oracle. The reconstructed 29 binned workbook rows do not verify
raw selection. Shape calculations remain outside this stage.

Brief 13 adds integrated 2D vesicularity from canonical raw selection and
per-group image vesicularity from all measured pore objects before filtering.
Both paths use border-corrected `area2`; excluded-phase `area1` is retained as
provenance. Thirteen independent numerical cases and coherent public pipeline
tests cover source area reassignment, disabled groups, pooled images, signed
nonphysical results, identity ownership, overflow, and underflow. These tests
are not MATLAB output, workbook vesicularity parity, or original-image parity.

Brief 14 adds equivalent-sphere volume fractions from canonical converter
heights and signed N_V rows. Unnormalized, integrated-2D, and known-percentage
modes preserve converter order and values while adjusting only volume
fractions. Ten calculated analytical cases cover signed rows, duplicate labels,
zero targets, zero and cancellation denominators, and signed integrated
targets. The expectations are not original workbook references. Plotting
reductions, cumulative counts, NV/NVcorr summaries, shape, exports, GUI, and
complete original-image parity remain pending.

Brief 15 adds immutable distribution summaries from canonical volume results
and size-category percentages from complete histogram vectors. Exact global
grid membership preserves duplicate selected rows and explicit zero widths;
reverse cumulative N_V and forward total N_V remain independent signed sums.
Expected zero, negative-log, and zero-denominator domains are represented as
typed undefined real values instead of NaN, infinity, or complex payloads.
Five distribution and four category cases are calculated analytical evidence,
not MATLAB output or workbook references. The unchanged 29-row workbook
selection demonstrates only converter parity and duplicate-width handling.
Shape, GUI plots, exports, and full-workflow reconciliation remain pending.

Brief 15a replaces one Python-version-dependent built-in `sum` test oracle
with an explicit source-order loop and adds two literal cancellation cases that
distinguish forward order, reverse order, and compensated summation. Production
scientific arithmetic and all fixtures are unchanged. Commit `82033e0` passes
485 tests with zero skips on Python 3.11 and 3.12 on both Linux and Windows in
configured CI run `38846116435`; Ruff also passes. All 13 repository examples
pass locally under `python -O`. Original shape geometry remains pending for
Brief 16.

Brief 16 adds component moment geometry on accepted filled four-connected
labels: central second moments with the unit-square `1/12` contribution,
ordered major/minor eigen-axes, eccentricity, source orientation, and the
project's `+90` representation. Shape identity is preserved through a
canonical raw-selection replay and explicit `(image_id, component_id)` joins.
Perimeter is intentionally omitted from the verified API with
`pending_historical_profile`; a guarded MATLAB reference-export harness is
added for independent later comparison. This stage is analytically verified,
not MATLAB parity, and derived shape statistics remain pending for Brief 17.

Brief 16 review correction hardens the public selection boundary: each supplied
image geometry record is now validated once against a canonical moment replay
from its retained phase measurement and declared image ID before selection
joins. Parent/component metadata, strict scalar types, finite physical bounds,
and ordered component geometry are enforced with typed validation failures and
context. Returned selection rows now reference canonical owned records so
caller-owned writable measurement buffers cannot mutate post-selection results.
Historical MATLAB moment/perimeter equivalence is still not established.

Brief 16 final validation fix extends strict scalar contracts to retained
phase-measurement components before canonical replay uses nested IDs,
coordinates, counts, and calibrated fields. Fractional/Boolean/nonfinite and
nonscalar payloads now fail with typed contextual validation errors instead of
incidental conversion/arithmetic exceptions. Valid NumPy integer scalar inputs
remain supported. This remains a boundary correction only: perimeter,
historical MATLAB equivalence, and derived shape statistics are still pending.

friendly MATLAB TSV capture runner, Python preparation/import/comparison
Brief 16a adds perimeter reference capture/comparison tooling rather than a
production perimeter estimator. A deterministic input case catalog, legacy-
friendly MATLAB TSV capture runner, Python preparation/import/comparison
utilities, strict schema/topology/hash validation, and CI-safe transport
fixtures are in place. Direct-script wrappers now bootstrap the repository root
without `PYTHONPATH`, and subprocess tests exercise all three wrappers plus a
synthetic prepare/import/compare sequence. Local tests and lint are recorded in
the implementation handoff; configured hosted Linux/Windows Python 3.11/3.12
CI is still required for acceptance.

The real capture attempt is retained under
`reference-tools/matlab/perimeter-evidence-r2025b-20261016-004729-c559cf4f`.
MATLAB 25.2.0.2998904 (R2025b) starts successfully, but Image Processing
Toolbox is unavailable (`ver('images')` returns no installation), and capture
fails at `regionprops`. The partial `.tmp` TSVs and complete logs are preserved;
they are not importable reference evidence. No component perimeter values,
within-run report, or historical compatibility result is claimed. Install or
license Image Processing Toolbox and repeat capture into a new unique directory
before this implementation work began. Separately supplied reviewer evidence
was later bundled under `main/tests/fixtures/16b_perimeter_evidence/`; it comes
from MATLAB 25.2.0.2998904 (R2025b) with Image Processing Toolbox 25.2 and is
not produced by the blocked local capture attempt.

Brief 16b adds two explicit perimeter profiles, Moore pixel-center boundary
tracing, linear grouped label/topology validation, calibrated perimeter
records, and canonical raw-selection identity joins. The profile
`matlab_perimeter_old_r2025b_verified_v1` matches MATLAB `PerimeterOld`; the
profile `matlab_perimeter_r2025b_verified_v1` matches MATLAB `Perimeter`.
Regression checks match the 30 valid primary-capture objects and all 526
supplementary objects within relative/absolute `1e-12` pixel tolerance. The
captured `narrow_neck_final` values remain diagnostic because its same-ID
region is not four-connected. Brief 16b is accepted within this evidence
boundary: configured CI run `38908191207` passes lint and all four
Linux/Windows Python 3.11/3.12 jobs, including 539 tests on Linux 3.12. This
establishes R2025b property equivalence on the supplied evidence, not the
historical runtime for original workbooks or full original-workflow parity.
Moment-only verification metadata and its pending-perimeter marker remain
unchanged. The implementation commit is `47c075d`; its tracker follow-up is
`62147c1`.

Brief 17 adds immutable combined shape results from canonical raw-selection,
moment, and perimeter inputs. It verifies that both geometry families retain
the same complete phase measurement, including label/mask arrays, and joins
selected rows by stable image/component identity. Derived measures preserve
the original regularity, form-factor, elongation, aspect-ratio, and fitted-
ellipse-perimeter expressions. Initial analysis keeps all post-nesting rows
for statistics while using a strict size threshold for plot membership;
updated-minimum mode uses inclusive membership for both. Statistics retain
unrounded values and four-decimal display values, sample
standard deviation, smallest-value mode ties, and explicit empty-population
statuses. The example exercises a real singleton and threshold exclusion.
Brief 17a replaces Decimal quantization with the observed MATLAB `roundn(x,-4)`
binary64 sequence: scale by `10000.0`, round the absolute scaled value using
floor/fraction ties-away-from-zero, restore sign, and divide by `10000.0`.
Literal MATLAB R2025b execution of the installed `roundn.m` produced
`0.00035 -> 0.0004`, `-0.00035 -> -0.0004`, `0.03125 -> 0.0313`, and the
specified adjacent-float boundary results; the two supplied review commands
exited 0. This validates that primitive only, not original `basicstats.m` or
historical workflow parity. Cached summary inputs also require `mm`, `mm^2`,
and `degree`; mismatches fail with contextual `incompatible_shape_result_unit`
errors before image or row replay.

Local verification after 17a: 561 tests pass (the 539-test baseline plus 22),
Ruff passes, and all 16 scientific examples pass under `python -O`. Hosted run
`38913323389` passes Ruff and all four Linux/Windows Python 3.11/3.12 test
jobs. Earlier run `38911650997` was for pre-17a code and is not the acceptance
run for these corrections. Independent review remains pending.
Moment/orientation parity and full original-run reconciliation remain
unresolved. Brief 17a implementation commit: `f54657f`.

Brief 18 adds the headless segmented-image workflow, complete pre-nesting
histograms, explicit manual/AutoSmart binned and independent raw selections,
accepted area/vesicularity/conversion/volume/statistics stages, bulk N_A and
MATLAB R2025b shape histogram arrays, and nine deterministic UTF-8 TSV table
projections with identity sidecars, diagnostics, checksums, and atomic
fresh-directory bundle writing. `Shape_out.txt` contains one row per
post-nesting selected raw object; shape histograms remain plot/sidecar data.
The unchanged histogram fixture and three runtime logs are tracked with
byte-preserving Git attributes and raw SHA-256 pins. Local verification before
Brief 18a corrections was 586 tests. With the corrections, 594 tests pass with
no local skips, Ruff passes, and all 17 examples pass under `python -O`. A
clean checkout with `core.autocrlf=true` passes all 594 tests and reproduces
the four pinned raw SHA-256 values. Hosted CI run `38926177590` passes Ruff and
the Linux/Windows Python 3.11/3.12 matrix on acceptance commit `015b1c9`.
Prepared/final results carry builder-issued private cache-validation records,
cached reanalysis avoids image-array scans, and raw table rows use checked
indexed lookup. The only repository-defined conditional skip remains the
existing Tk integration test when a display is unavailable; Brief 18a adds no
skip behavior.
`Area_out.txt` is
explicitly unavailable when zero-area removal misaligns source image indices;
`Ves_out.txt` is partial if caller-supplied density-derived vesicularity is
missing. These cases are not filled with invented zeros. Binary `.xls`
writing, rendered plots, original-image parity, and full original-workflow
parity are not claimed by this handoff.

Brief 19 remains original-run reconciliation/core performance; Brief 20 is GUI
integration; Brief 21 is GUI improvements/release and the final same-real-image
comparison after GUI tests. The original FOAMS implementation must remain NOT
COMPLETE until those later gates and intentional deviations are resolved.

The original FOAMS functionality may be called implemented and verified only
after all milestones pass and intentional deviations are explicitly resolved.

## Brief 19 First Replay And Benchmark

The pinned MATLAB workflow was executed locally against the external checkout
of `jsoteloflores/FOAMS-1.0.5` at
`179663203f2d0f86b2863d5ea7f8f70dadca02f8`, using MATLAB
`25.2.0.2998904 (R2025b)` and Image Processing Toolbox `25.2`. The source
checkout remains outside this repository. This is an exact-fragment numerical
replay, not an original GUI session or a real-image comparison.

The complete external capture contains 10 catalog cases, 9 successful source
runs, 1 explicitly expected MATLAB `MATLAB:badsubscript` source-domain failure,
and 7 deduplicated image-evidence records. Capture SHA-256:
`e3c03cf8794da48d58e256b8efe6f2403791c92e585cd8cb28245412383d5c54`.
The extraction-map, generated constructor, runner, and catalog hashes are
stored in the bundle manifest. The bundle is kept under the local temporary
directory, not committed. Rerun `reference-tools/workflow/run_workflow_reference.py`
against the pinned external source checkout to create a fresh bundle, then
run `reference-tools/workflow/compare_workflow_reference.py <capture-dir>`.

The strict comparison emitted 7,786 rows: 113 mismatches and 1 unavailable
expected-failure stage. Captured K0/K1/border-removal/hole-fill/label arrays,
component memberships, direct measurements, Raw_data and Rawcutoffs_data
identities, initial per-object shape vectors, histogram/area tables, and the
other compared table families match. The first divergence is
`shape_and_initial_threshold`: in the strict-boundary case, source identities
contain 8 threshold objects while Python contains 5. The updated inclusive
threshold case matches. `Stats_out` has one additional source convention
difference: source `basicstats` includes an eccentricity value of zero
(N=8, mean 0.3540, standard deviation 0.4899); Python classifies that value
as undefined and excludes it (N=7, mean 0.4045, standard deviation 0.5061).
These are reported as mismatches, not normalized away.

The remaining 109 mismatches are in `Calc_out`: 46 source `Ln(n)` cells are
negative infinity, 23 are complex logarithms, and 40 `log(NV>L)` cells are
negative infinity; Python returns typed `log_of_zero`/`log_of_negative`
undefined values instead. Source signed/complex values and Python undefined
reasons are preserved in JSON/Markdown comparison rows. No science formula
was changed to force parity. These threshold, statistics, and logarithm
conventions block full original-workflow equivalence. The empty catalog case
also shows MATLAB failing in the project image-area/histogram stage rather
than at Python's AutoSmart cutoff stage.

The comparator uses exact array and identity checks, source-order/schema
checks, unit-specific floating tolerances, exact Stats_out display rounding,
and separate undefined/NaN/infinity/complex classifications. The comparison
report is generated beside the external capture and is not treated as a
permanent original-image fixture.

The Brief 19 benchmark measures fresh preparation, complete
prepare-plus-analysis, cached analysis re-run, and export projection/write on
three deterministic synthetic masks (512^2/100, 1024^2/500, and
2048^2/1000 pixels/components), using one warm-up and five measured runs.
At 2048^2/1000, final medians were 1.242 s fresh preparation, 1.746 s full
prepare-plus-analysis, 0.520 s cached re-analysis, and 0.119 s export/write.
The synthetic workload is not the MATLAB catalog or original images. Windows
working-set memory is sampled every 5 ms; short peaks can be missed and the
observed deltas varied between runs.

Profiling located the preparation bottleneck in per-component
connectivity/hole validation. Replacing Python point-set traversal and Euler
bookkeeping with OpenCV 4-connected foreground labeling plus a 4-connected
exterior flood fill reduced profiled validation from 2.57 s to 0.38 s on the
2048^2/1000 workload. The new check matched the previous validator on all
3x3 masks and 1,000 seeded 8x8 masks; all 15 perimeter tests and the captured
source comparison still pass. In five-run benchmarks, fresh preparation
improved from 1.926 s to 1.242 s (about 36%), and full prepare-plus-analysis
from 2.443 s to 1.746 s (about 29%). No memory reduction is claimed.

Local validation after this work: all 608 repository tests and all 17 examples
under `python -O` pass. Configured Ruff scope (`main/`, the Brief 19 benchmark,
and workflow wrappers) passes; unrestricted `ruff check .` still reports
import-order/E402 issues in three older perimeter wrappers. Hosted
Linux/Windows Python 3.11/3.12 CI is not yet recorded. Brief 19 remains
incomplete until strict-threshold, eccentricity-statistics, and logarithm
domain differences are dispositioned and the final example/CI gates pass.
GUI integration and original-image comparison remain later Brief 20/21 gates.
