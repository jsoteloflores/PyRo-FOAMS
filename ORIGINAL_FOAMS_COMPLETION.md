# Original FOAMS completion tracker

Last updated: 2026-10-04.

**Original FOAMS implementation: NOT COMPLETE.**

Target reference: `jsoteloflores/FOAMS-1.0.5` at commit
`179663203f2d0f86b2863d5ea7f8f70dadca02f8`.

## Milestones

| Stage | Status | Evidence required or retained |
|---|---|---|
| 1. Post-nesting N_A-to-N_V conversion | Brief 06 accepted against bundled Calc_out fixture | All 29 rows, source alpha order, heights, duplicate metadata, validation, and smallest-class exception tested |
| 2. Measurement, area, and histogram conventions | Pending | Connectivity, border/phase area, rounded labels, histc ownership, and trimming comparisons |
| 3. Automatic magnification cutoffs | Pending | Source algorithm, ties, zeros, no-overlap behavior, and selected-bin comparisons |
| 4. 2D vesicularity and phase accounting | Pending | Multiscale area integration and excluded-phase denominators |
| 5. Volume distributions and normalization | Pending | Original normalization modes and cumulative volume outputs |
| 6. Density summaries and plotting quantities | Pending | NV/NVcorr, per-width density, cumulative counts, and logarithms |
| 7. Shape outputs and statistics | Pending | Original conventions and ellipse-algorithm differences resolved |
| 8. Analysis interface | Pending | Original analyses available through the application |
| 9. Original-equivalent exports | Pending | Calc/Area/Ves/Misc/Shape-equivalent outputs verified |
| 10. Full workflow reconciliation | Pending | Pinned stage-by-stage and final output benchmarks |

## Current verified boundary

The FOAMS 1.0.5 post-nesting `N_A`-to-`N_V` converter is verified against the
bundled 29-row `Calc_out.xls` extraction. The artifact records the workbook
hash and pinned source version. MATLAB was not executed as part of this test.
The converter does not establish source-compatible image measurements,
histogram preparation, area denominators, nesting selection, or full workflow
parity.

The original FOAMS functionality may be called implemented and verified only
after all milestones pass and intentional deviations are explicitly resolved.
