# Original FOAMS completion tracker

Last updated: 2026-10-15.

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
| 7. Shape outputs and statistics | Pending | Original conventions and ellipse-algorithm differences resolved |
| 8. Analysis interface | Pending | Original analyses available through the application |
| 9. Original-equivalent exports | Pending | Calc/Area/Ves/Misc/Shape-equivalent outputs verified |
| 10. Full workflow reconciliation | Pending | Pinned stage-by-stage and final output benchmarks |

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

The original FOAMS functionality may be called implemented and verified only
after all milestones pass and intentional deviations are explicitly resolved.
