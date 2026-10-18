The durable, MATLAB-free five-case projection is
`main/tests/fixtures/19a_foams105_reference.json.gz`, with sibling manifest
`19a_foams105_reference.json.gz.manifest.json`. The manifest pins compressed
and decompressed SHA-256, parent capture hash, selected catalog hash and exact
case IDs. Final compressed SHA-256:
`32379cd9313e24f12ea4ea22468b0eb649b52f72dc1375911ebf7ca33fac7aa6`; payload
SHA-256: `60ed38a96f05fd3dae301e788eda1f2f6fdab76ea411e5a34f5a2183c39b09b7`.
# Original FOAMS completion tracker

Last updated: 2026-10-18.

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
| 7. Shape outputs and statistics | Briefs 16-18a accepted within bounded analytical/R2025b scopes | Moment orientation remains analytical; R2025b perimeter and roundn checks are narrow reference tests; corrected MATLAB replay now confirms threshold identities and updated eccentricity statistics |
| 8. Analysis interface | Headless Brief 18 workflow accepted; GUI integration remains Brief 20 | Preparation/analysis APIs compose accepted stages and preserve source slots; GUI remains pending |
| 9. Original-equivalent exports | Nine deterministic TSV projections accepted in Brief 18a | All nine table families and sidecars are produced; corrected comparison has no unexpected finite/table mismatch outside the documented Calc_out logarithm representation extension, subject to Area_out/Ves_out availability rules |
| 10. Full workflow reconciliation | Brief 19 corrected replay and durable reference complete; literal raw parity intentionally differs at log serialization; hosted CI pending | Exact pinned source/runtime capture and compressed five-case real-MATLAB reference; 7,719 full-capture rows with 117 operand-verified log representation differences and one unavailable source failure; redesigned benchmarks include bounded topology tradeoffs |

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

## Brief 19a Corrected Replay And Evidence

The pinned MATLAB workflow was replayed from the external checkout of
`jsoteloflores/FOAMS-1.0.5` at
`179663203f2d0f86b2863d5ea7f8f70dadca02f8`, using MATLAB
`25.2.0.2998904 (R2025b)` and Image Processing Toolbox `25.2`. The original
source remains external. This is a numerical source-fragment replay, not an
original GUI session or real-image comparison.

The `updated_shape_minimum_statistics` extraction now spans `newstats.m`
lines 8-23, LF SHA-256
`37e8c28f537a8c0fbd08f13486da466627ce3fff704303bc522c8853b556075e`. Its
declared `stats_asp_rat`, `stats_elong`, `stats_reg`, and `stats_ecc` outputs
are assignment-checked; MATLAB clears the prior values before evaluation and
requires all four outputs afterward. A regression proves an updated subset
counts a genuine zero-eccentricity object. The previous “eccentricity
convention mismatch” diagnosis was a stale MATLAB `stats_ecc` caused by the
truncated fragment, not a Python exclusion rule.

Every original `initial_analysis` catalog run now uses the same project
`minpix` and statistics minimum. The independent Python summary API remains
available, and `updated_minimum` still uses the distinct `min_diameter` from
`newstats.m`. Capture/report metadata records project and statistics minima,
maximum scale, both thresholds in mm, and strict `>` versus inclusive `>=`.
Exact boundary and neighboring-double behavior remains covered separately by
shape-statistics API tests; a full replay is not used to imply an equality
object survives the earlier project diameter filter.

The fresh complete capture contains 10 cases, 9 successful source runs, one
expected MATLAB `MATLAB:badsubscript` failure, and 7 deduplicated source-image
records. Capture SHA-256:
`ad5fce5ea56d406877b5948821ececa8bc0ca298dd9f650bfce0b90e5bde6d4c`.
The previously reviewed, unmodified capture remains external at
`%TEMP%/foams105-workflow-r2025b-5729f079fbc54a6eaa74df3e713e3aae`; it is
superseded and is not the parent of the committed package.
The durable, MATLAB-free five-case projection is
`main/tests/fixtures/19a_foams105_reference.json.gz`, with sibling manifest
`19a_foams105_reference.json.gz.manifest.json`. The manifest pins compressed
and decompressed SHA-256, parent capture hash, selected catalog hash and exact
case IDs. Final compressed SHA-256:
`7b2acdbd5e564a0f5774e82a39f7aa2751fe79ab061a7ad88ba808b462728b93`; payload
SHA-256: `afecf0ab4c187e480051268f4734e24e703a4e787eb83443ccf8a4f4b831f0b5`.
Decompression is deterministic gzip (`mtime=0`) over canonical UTF-8 JSON.
The projection retains every selected case field, table cell,
identity, direct image evidence, expected failure, and strict/policy report;
it does not filter mismatches. No private original source files are included.

On the full corrected capture, strict raw comparison produced 7,719 rows:
7,601 exact/toleranced matches, 117 raw differences, and one unavailable
expected-failure stage. All 117 raw differences are confined to `Calc_out`
logarithm representation: 52 `Ln(n)` negative-infinity values, 18 `Ln(n)`
complex values, and 47 `log(NV>L)` negative-infinity values. Source operands
and exact operations are captured (`ln(n_numb)` and
`log10(NV_cumsum)`). The named policy
`foams105-real-log-classification-v1` verifies source/Python operand equality,
operation/base, Python reason/classification, and the source result before
qualifying the existing real-valued API extension. Policy mode reports 117
qualified representation differences and zero unexpected mismatches; strict
mode reports 117 raw differences and fails. No finite scientific table or
image-stage mismatch remains in the supported cases, but literal
`Calc_out` export parity is not established. The expected source failure remains unavailable:
MATLAB fails at project image-area/histogram processing, Python at
`autosmart_cutoffs`; both stages and reasons are retained.

Candidate code identity is the actual Git commit `d71c17931dbb3625ca68dcefb304d275d7dfa7fe`; `brief19-catalog-v1` is recorded separately as a catalog label. Reports include dirty source-file hashes while excluding the self-referential reference package and generated benchmark result files, which have their own checksums.

The topology optimization now bounds the OpenCV dense path by padded
bounding-box area `<= 8 * component_pixel_count`, computed with Python integer
arithmetic before allocating the dense mask. Longer/sparser components use
the previous sparse four-connected traversal plus Euler hole check. Tests
forbid dense allocation/OpenCV calls on long L, C and staircase shapes, retain
disconnected/hole rejection and Euler failure context, and compare every
nonempty 3x3 mask to the prior validator. The 1024^2 L workload has 1,791
foreground pixels inside an 802,816-pixel padded bounding box (448.25x); its
fresh preparation is about 0.057 s without allocating that box.

The benchmark now has a fixed-image-size component-count series, an
approximately fixed-density image-size series, sparse-L input, actual cached
cutoff and normalization edits, a shape-minimum-only call to
`summarize_foams105_shapes`, and separate pure projection versus disk writing
using a prebuilt projection. The original Brief 19 results remain preserved
as `benchmarks/results_foams105_brief19_pre19a.{md,json}`. New results and raw
samples are in `benchmarks/results_foams105_brief19.{md,json}`. Each operation
uses one warm-up and five measured iterations; RSS is a 5 ms process-working-
set sample and can miss short peaks. Workloads are synthetic, not MATLAB
catalog or original images. In the final 2048^2/1000 run, fresh preparation
is 1.271 s and full prepare-plus-analysis is 1.671 s, versus pre-19a medians
1.242 s and 1.746 s (+2.3% and -4.3%). Repeated runs varied appreciably; no
general compact-mask speedup or memory reduction is claimed. The value of the
bound is the explicit sparse-shape work/memory cap.

Final local validation: 624 tests passed with no skips; all 17 examples passed
under `python -O`; configured Ruff passes. Unrestricted `ruff check .` still
reports import-order/E402 issues in three older perimeter wrappers outside
this brief. Hosted Linux/Windows Python 3.11/3.12 CI remains pending; this
dirty worktree was not pushed or committed, so the four hosted jobs could not
be run on this exact candidate. Brief 19 remains pending independent review,
hosted CI, and explicit acceptance of the documented raw logarithm
serialization extension. GUI integration and final real-image comparison
remain Briefs 20/21.
