# Original FOAMS completion tracker

Last updated: 2026-10-05.

**Original FOAMS implementation: NOT COMPLETE.**

Target reference: `jsoteloflores/FOAMS-1.0.5` at commit
`179663203f2d0f86b2863d5ea7f8f70dadca02f8`.

## Milestones

| Stage | Status | Evidence required or retained |
|---|---|---|
| 1. Post-nesting N_A-to-N_V conversion | Brief 06 accepted against bundled Calc_out fixture | All 29 rows, source alpha order, heights, duplicate metadata, validation, and smallest-class exception tested |
| 2. Measurement, area, and histogram conventions | Briefs 07, 08, and 09 accepted; shape measurements pending | 45-label preparation, all 180 histogram cells, exact mask topology, calibrated component geometry, six weighted-area cases, strict four-slot exclusion thresholds, signed per-image denominators, and explicit compatible-group sums tested |
| 3. Automatic magnification cutoffs | Brief 10 accepted; Brief 11 scientific scope accepted; Brief 11a CI queued at `732bbd0920eb1b758cb11b499dcaf2bf12eba762`; Brief 12 raw-object selection implemented pending review | Signed log-density transitions, exact binned source slices, LF/CRLF fixture pins, raw predecessor bounds, stable component identities, inclusive unrounded filtering, and ascending raw-slot provenance tested |
| 4. 2D vesicularity and phase accounting | Pending | Multiscale area integration and excluded-phase denominators |
| 5. Volume distributions and normalization | Pending | Original normalization modes and cumulative volume outputs |
| 6. Density summaries and plotting quantities | Pending | NV/NVcorr, per-width density, cumulative counts, and logarithms |
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
shape-property equivalence, vesicularity, nesting selection, or full workflow
parity. Corrected-area calculations are implemented from the measured masks
and components, with signed nonpositive denominators retained as diagnostics.
AutoSmart default cutoff suggestions are accepted. Explicit and suggested
binned ranges now replay checked source-style selection and concatenation,
including duplicate boundary rows and source-slot provenance. Raw-object
selection remains outside this boundary.

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
raw selection. Vesicularity and shape calculations remain outside this stage.

The original FOAMS functionality may be called implemented and verified only
after all milestones pass and intentional deviations are explicitly resolved.
