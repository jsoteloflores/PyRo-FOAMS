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
- Original FOAMS setup, immutable analysis history, the future joinable export bundle, and same-real-image MATLAB comparison remain future work.

## Live GUI test

On a display-capable Windows host, run the committed widget-level smoke and capture empty and populated workspace screenshots at both sizes and font settings:

```powershell
$env:PYRO_FOAMS_LIVE_TK = "1"
$env:PYRO_FOAMS_GUI_ARTIFACTS = "artifacts/gui-smoke"
python -m unittest -v main.tests.test_sample_project_gui
```

The test drives real workspace and processing-window buttons, delays batch computation to verify Tk timer responsiveness, checks accepted-input export routing, draft file round-trips and partial failures, cancellation-safe cleanup, keyboard traversal, and layout bounds. Screenshots are `workspace_1280x800_font1x.png`, `workspace_1280x800_font2x.png`, `workspace_1400x900_font1x.png`, `workspace_1400x900_font2x.png`, and `workspace_populated_1280x800_font1x.png`. GitHub Actions runs it in a dedicated Windows Python 3.12 job and uploads the screenshots. Run without `PYRO_FOAMS_LIVE_TK=1` only for discovery; the four live tests will be skipped, which is not GUI acceptance.

Commit `4928e03f3844919ded6716d0f93075174e530ec9` passed hosted run [42262702398](https://github.com/jsoteloflores/PyRo-FOAMS/actions/runs/42262702398): lint, all four Linux/Windows Python 3.11/3.12 unit jobs, and the Windows Python 3.12 live-Tk job succeeded. The GUI job uploaded artifact `windows-live-tk-gui-smoke` (artifact ID `13526245699`), containing the screenshots listed above.

Earlier run `42244733714` stalled during dependency installation for more than 54 minutes. Run `42259395489` did not complete its GUI step; local reproduction identified an unmocked standalone-export success dialog. The test now mocks that dialog and captures Tk stderr as a failure; the passing run above verifies the correction. Same-real-image comparison against MATLAB FOAMS remains future work; no scientific formulas changed.
