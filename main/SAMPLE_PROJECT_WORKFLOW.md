# Sample Project Workspace

## Create and save a sample

Launch the application with `python -m main`. The top bar names one physical sample. Use **Load Images** to add images to that sample; duplicate filenames are allowed because each image receives a stable project ID. Use **Rename Image** to change the display name without changing identity.

Use **Save Project** to choose a project folder. The folder owns a JSON manifest and lossless source, working-image, mask, and label assets. Reopen it with **Open Project** or copy the folder before reopening it elsewhere. Opening rejects unsupported schemas, missing/corrupt assets, invalid dimensions, and paths that escape the project folder. Unsaved changes prompt for Save, Discard, or Cancel when opening a project or closing the app.

## Prepare and accept segmentations

The active image is outlined in blue. Batch targets are outlined in green. Click a thumbnail to change the active image; Ctrl-click or right-click toggles its batch-target membership. **Proceed with Processing** opens the existing threshold/separation controls. **Apply to selected** creates drafts only for green-selected images; it does not accept or include them.

**Edit Masks…** edits the draft. **Import Mask…** accepts dimension-matched binary grayscale raster masks (0/1/255) or numeric integer `.npy` label maps; label maps are explicitly converted with `labels > 0`. Ambiguous color/phase rasters and non-binary grayscale rasters are rejected. Imported masks remain drafts and are recorded as imports.

**Accept Draft** creates/replaces the accepted full-resolution snapshot. Draft edits do not change the accepted snapshot. **Toggle Include** is independent and requires an accepted snapshot. An included image can have a newer draft; analysis continues to use the accepted snapshot until it is accepted again.

Cropping preserves the owned original source pixels. The cropped working image and its source-coordinate ROI are saved separately. Changing the working ROI clears acceptance and inclusion because the prior segmentation no longer matches the processing image.

## Calibration and analysis

Use **Set Calibration…** for the active image or selected batch images. The value is spatial units per pixel; supported units are m, cm, mm, um, and nm. The image grid shows valid resolutions normalized to mm/px. Processing and acceptance do not require calibration.

**Analyze Included** and the processing window's stereology action use accepted labels for included images. Physical analysis is blocked with an image-specific list until every included input has a valid calibration. Current modern stereology and existing export tools remain available; this foundation does not add the Original FOAMS analysis setup, results history, or the future joinable export bundle.

## Current limitations

- The workspace retains the existing thumbnail grid rather than a new dedicated image-list/editor layout.
- Batch threshold/separation runs use the active processing controls for every selected target.
- No immutable analysis-run history or automatic result-cache persistence is implemented.
- An automated live-Tk Windows smoke passed the create/import/process/batch/edit/accept/include/calibrate/analyze/save/reopen sequence; a non-fatal Tk callback warning appeared during teardown. Manual visual/user-driven acceptance remains open.
- Same-real-image comparison against MATLAB FOAMS remains future work; no scientific formulas were changed in this foundation.
