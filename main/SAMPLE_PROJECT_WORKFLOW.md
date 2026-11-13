# Sample Project Workspace

## Create and save a sample

Launch the application with `python -m main`. The top bar names one physical sample. Use **Load Images** to add images to that sample; duplicate filenames are allowed because each image receives a stable project ID. Use **Rename Image** to change the display name without changing identity.

Use **Save Project** to choose a project folder. Schema 2 stores exact uint8/uint16 grayscale, BGR, or BGRA source pixels in non-executable `.npy` assets; working images, masks, and labels use separate lossless assets. Schema-1 projects remain readable as `legacy_derived_unknown`; their stored pixels are retained as the only available source representation, without claiming lost precision was recovered. Migration occurs only on explicit save. Reopen the folder or copy it elsewhere; load validates checksums, dimensions, schema, and portable asset paths. Unsaved changes prompt for Save, Discard, or Cancel when opening a project or closing the app.

## Prepare and accept segmentations

The active image is outlined in blue. Batch targets are outlined in green. Click a thumbnail to change the active image; Ctrl-click or right-click toggles its batch-target membership. **Process…** opens the existing threshold/separation controls. **Apply to selected** runs in a background coordinator, leaves the UI responsive, and creates drafts only for green-selected images; it does not accept or include them. Cancel or close invalidates pending results. A result commits only if the project session, stable image ID, source/draft revisions, and processing settings still match.

**Edit Masks…** edits the draft. **Import Mask…** accepts dimension-matched binary grayscale raster masks (0/1/255) or numeric integer `.npy` label maps; label maps are explicitly converted with `labels > 0`. Ambiguous color/phase rasters and non-binary grayscale rasters are rejected. Imported masks remain drafts and are recorded as imports.

In a standalone processing window, **Save Draft Masks…** saves every available mask, including all-zero masks; **Save Draft Labels…** writes both the raw `.npy` label map and color preview. Filenames use stable image IDs, not display names. Failed/partial writes are reported per output. Canceling the folder picker writes nothing.

**Accept Draft** creates/replaces the accepted full-resolution snapshot. Draft edits do not change the accepted snapshot. **Include** and **Exclude** are independent and inclusion requires acceptance. An included image can have a newer draft; analysis and measurement export continue to use the accepted snapshot until it is accepted again.

Cropping preserves the owned original source pixels. The cropped working image and its source-coordinate ROI are saved separately. Changing the working ROI clears acceptance and inclusion because the prior segmentation no longer matches the processing image.

## Calibration and analysis

Use **Set Calibration…** for the active image or selected batch images. The value is spatial units per pixel; supported units are m, cm, mm, um, and nm. The image grid shows valid resolutions normalized to mm/px. Processing and acceptance do not require calibration.

**Analyze Included** now opens an integrated sample-analysis workspace that uses a validated, owned snapshot of included accepted labels, masks, working images, calibrations, stable IDs, accepted snapshot IDs, and revisions. The workspace supports three run presets: **Original FOAMS 1.0.5**, **Modern analysis**, and **Compare methods**. The processing window's **Export accepted measurements (N)…** uses the same snapshot adapter; it never falls back to transient drafts. Physical analysis and export are blocked with image-specific errors until every included input has a valid calibration. Calibration units are normalized to mm/px before measurements, preserving the existing CSV columns. Optional accepted-label previews use stable image IDs in filenames. Standalone processing windows expose a separately labeled draft export.

## Current limitations

- The workspace retains the existing thumbnail grid rather than a new dedicated image-list/editor layout.
- Preparation actions wrap by measured group width; horizontal/vertical scrollbars appear only when content exceeds the viewport. Tab enters the preparation controls and cycles through actions/visible overflow controls; Shift+Tab reverses the order.
- Layout was exercised at 1280×800 and 1400×900 using ordinary fonts and doubled Tk named-font sizes. This is not OS-DPI certification.
- Batch threshold/separation runs use the active processing controls for every selected target. Background workers operate on one owned dispatch snapshot; only Tk's UI thread drains events and commits results.
- No immutable analysis-run history or automatic result-cache persistence is implemented.
- Brief 21 remains pending independent acceptance. Original FOAMS remains NOT COMPLETE; accepted multiphase project inputs and same-real-image MATLAB comparison remain outstanding. No material phases are inferred from object labels.

## Analysis corrections (Brief 21a, pending)

The shared analysis window now offers **Prepare / inspect**, **Suggest cutoffs**, **Review selection**, **Run analysis**, **Save setup**, and **Update shape minimum**. Preparation/suggestion/calculation run on owned workers. Conflicting edits are disabled during jobs; Cancel/close invalidates completion rather than interrupting scientific kernels. The Measurements action retains access to the existing modern inspection tool.

The Distributions tab contains native prepared and selected values with counts, corrected/eligible areas, N_A, and available signed N_V/volume values. Select contiguous prepared rows and choose a group to apply exact legacy labels or explicit modern half-open bin ranges. Raw legacy memberships and method-specific diagnostics are in Details. Log plots omit nonpositive/unavailable values only visually; tables retain them.

Modern multi-group analysis requires explicit `group:start:stop` segments; transition numbers alone are rejected. Group IDs retain coarse-to-fine calibration ordering, but ownership of small/large bins is chosen explicitly, not inferred from that order. Minimum detection and maximum reliability remain `None` when unspecified. Measurements can be inspected even when missing detection limits block N_A or when a composite plan is not ready. Reconstruction requires separate selection of **No 3D size classes above the selected upper edge**.

Setup schema 2 saves independent method settings, editable drafts, stable image/source-key assignments, and referenced image IDs. Schema-1 null inactive-method fields no longer become the text `None`. Future schemas and malformed mappings are reported, not silently treated as completed runs. Save Setup stores configuration without running science; Save Project persists it using existing atomic project I/O. Reopening restores configuration, not a completed calculation.

Captured jobs retain settings, project/session and accepted input identities. Inclusion, calibration and reacceptance changes mark results stale and reject late completions; Run captures fresh inputs. Draft-only edits and display renaming do not change scientific identity. Legacy preparation keys omit display/list order and unrelated project revisions. Selection, normalization, categories and shape summaries reuse canonical prepared results. Calculation/render failures retain earlier results and identify the failed method/stage.

Regression coverage is in `tests/test_sample_analysis.py` and the live `test_analysis_settings_workers_and_native_rendering` / `test_analysis_explicit_modern_plan_and_cancellation` tests. It includes both perimeter renders, direct API result checks, manual raw membership, normalization/shape preparation-call counts, settings disk round trips, asymmetric group ownership, explicit reconstruction consent, both Compare partial-success directions, cancellation, reacceptance and stale completion. Local populated screenshots include `analysis-distributions.png`, `analysis-manual-cutoffs.png`, `analysis-comparison.png`, `analysis-validation-error.png`, and plots at both window/font settings.

These corrections are uncommitted work on base `4a5fcb4c2affde05ff912440ff48a370b899eca8`; no corrected hosted run is claimed. Reviewed foundation commit `025ed6dd64db63477789188ea463b6f471d9d22b` had [run 43363606911](https://github.com/jsoteloflores/PyRo-FOAMS/actions/runs/43363606911): lint and all four unit jobs passed, but the Windows live-Tk job failed after its 10-minute workflow-test timeout. Brief 21 requires corrected-commit hosted evidence and independent acceptance before Brief 22.

Local validation of this worktree: `python -m ruff check main/` passed; `python -m unittest discover main/tests` passed (676 discovered, 670 executed, 6 live tests skipped). A separate `PYRO_FOAMS_LIVE_TK=1 python -m unittest -v main.tests.test_sample_project_gui` run passed all 6 live tests with no skips. Populated plot captures were visually checked after fixing owned-window foreground capture; these are local evidence, not a hosted-CI result or complete OS-DPI certification.

## Live GUI test

On a display-capable Windows host, run the committed widget-level smoke and capture empty and populated workspace screenshots at both sizes and font settings:

```powershell
$env:PYRO_FOAMS_LIVE_TK = "1"
$env:PYRO_FOAMS_GUI_ARTIFACTS = "artifacts/gui-smoke"
python -m unittest -v main.tests.test_sample_project_gui
```

The test drives real workspace and processing-window buttons, delays batch computation to verify Tk timer responsiveness, checks accepted-input export routing, draft file round-trips and partial failures, cancellation-safe cleanup, keyboard traversal, and layout bounds. Screenshots are `workspace_1280x800_font1x.png`, `workspace_1280x800_font2x.png`, `workspace_1400x900_font1x.png`, `workspace_1400x900_font2x.png`, and `workspace_populated_1280x800_font1x.png`, plus the analysis captures above. GitHub Actions runs it in a dedicated Windows Python 3.12 job and uploads the screenshots. Run without `PYRO_FOAMS_LIVE_TK=1` only for discovery; live tests will be skipped, which is not GUI acceptance.

Commit `4928e03f3844919ded6716d0f93075174e530ec9` passed hosted run [42262702398](https://github.com/jsoteloflores/PyRo-FOAMS/actions/runs/42262702398): lint, all four Linux/Windows Python 3.11/3.12 unit jobs, and the Windows Python 3.12 live-Tk job succeeded. The GUI job uploaded artifact `windows-live-tk-gui-smoke` (artifact ID `13526245699`), containing the screenshots listed above.

Earlier run `42244733714` stalled during dependency installation for more than 54 minutes. Run `42259395489` did not complete its GUI step; local reproduction identified an unmocked standalone-export success dialog. The test now mocks that dialog and captures Tk stderr as a failure; the passing run above verifies the correction. Same-real-image comparison against MATLAB FOAMS remains future work; no scientific formulas changed.
