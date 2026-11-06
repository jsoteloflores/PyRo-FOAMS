# Sample Project Workspace

## Create and save a sample

Launch the application with `python -m main`. The top bar names one physical sample. Use **Load Images** to add images to that sample; duplicate filenames are allowed because each image receives a stable project ID. Use **Rename Image** to change the display name without changing identity.

Use **Save Project** to choose a project folder. Schema 2 stores exact uint8/uint16 grayscale, BGR, or BGRA source pixels in non-executable `.npy` assets; working images, masks, and labels use separate lossless assets. Schema-1 projects remain readable as `legacy_derived_unknown`; their stored pixels are retained as the only available source representation, without claiming lost precision was recovered. Migration occurs only on explicit save. Reopen the folder or copy it elsewhere; load validates checksums, dimensions, schema, and portable asset paths. Unsaved changes prompt for Save, Discard, or Cancel when opening a project or closing the app.

## Prepare and accept segmentations

The active image is outlined in blue. Batch targets are outlined in green. Click a thumbnail to change the active image; Ctrl-click or right-click toggles its batch-target membership. **Process…** opens the existing threshold/separation controls. **Apply to selected** runs in a background coordinator, leaves the UI responsive, and creates drafts only for green-selected images; it does not accept or include them. Cancel or close invalidates pending results. A result commits only if the project session, stable image ID, source/draft revisions, and processing settings still match.

**Edit Masks…** edits the draft. **Import Mask…** accepts dimension-matched binary grayscale raster masks (0/1/255) or numeric integer `.npy` label maps; label maps are explicitly converted with `labels > 0`. Ambiguous color/phase rasters and non-binary grayscale rasters are rejected. Imported masks remain drafts and are recorded as imports.

**Accept Draft** creates/replaces the accepted full-resolution snapshot. Draft edits do not change the accepted snapshot. **Include** and **Exclude** are independent and inclusion requires acceptance. An included image can have a newer draft; analysis and measurement export continue to use the accepted snapshot until it is accepted again.

Cropping preserves the owned original source pixels. The cropped working image and its source-coordinate ROI are saved separately. Changing the working ROI clears acceptance and inclusion because the prior segmentation no longer matches the processing image.

## Calibration and analysis

Use **Set Calibration…** for the active image or selected batch images. The value is spatial units per pixel; supported units are m, cm, mm, um, and nm. The image grid shows valid resolutions normalized to mm/px. Processing and acceptance do not require calibration.

**Analyze Included** uses a validated, owned snapshot of included accepted labels, masks, working images, calibrations, stable IDs, accepted snapshot IDs, and revisions. The processing window's **Export accepted measurements (N)…** uses the same adapter; it never falls back to transient drafts. Physical analysis and export are blocked with image-specific errors until every included input has a valid calibration. Calibration units are normalized to mm/px before measurements, preserving the existing CSV columns. Optional accepted-label previews use stable image IDs in filenames. Standalone processing windows expose a separately labeled draft export.

## Current limitations

- The workspace retains the existing thumbnail grid rather than a new dedicated image-list/editor layout.
- Batch threshold/separation runs use the active processing controls for every selected target. Background workers operate on one owned dispatch snapshot; only Tk's UI thread drains events and commits results.
- No immutable analysis-run history or automatic result-cache persistence is implemented.
- Original FOAMS setup, immutable analysis history, the future joinable export bundle, and same-real-image MATLAB comparison remain future work.

## Live GUI test

On a display-capable Windows host, run the committed widget-level smoke and capture the 1280×800 and 1400×900 screenshots:

```powershell
$env:PYRO_FOAMS_LIVE_TK = "1"
$env:PYRO_FOAMS_GUI_ARTIFACTS = "artifacts/gui-smoke"
python -m unittest -v main.tests.test_sample_project_gui
```

The test drives real workspace and processing-window buttons, delays batch computation to verify Tk timer responsiveness, checks accepted-input export routing and cancellation-safe cleanup, and fails on Tk callback exceptions. GitHub Actions runs it in a dedicated Windows Python 3.12 job and uploads the screenshots. Without `PYRO_FOAMS_LIVE_TK=1`, ordinary headless suite discovery reports this integration test as skipped; that skip is not GUI acceptance.
- Same-real-image comparison against MATLAB FOAMS remains future work; no scientific formulas were changed in this foundation.
