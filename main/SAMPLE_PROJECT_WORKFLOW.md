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

The Brief 21a corrections were pushed as `5afb74e76130bf7bb532c67e09af32efcd0a0d66` ("fixins"). Hosted [run 43367128530](https://github.com/jsoteloflores/PyRo-FOAMS/actions/runs/43367128530) passed all six jobs: lint, Linux/Windows Python 3.11/3.12 unit tests, and Windows Python 3.12 live Tk. This baseline still contained the saved-setup and comparison-provenance defects corrected in Brief 21b below. Earlier foundation commit `025ed6dd64db63477789188ea463b6f471d9d22b` had [run 43363606911](https://github.com/jsoteloflores/PyRo-FOAMS/actions/runs/43363606911): lint and all four unit jobs passed, but the Windows live-Tk job failed after its 10-minute workflow-test timeout.

Recorded Brief 21a local validation: lint passed; unit discovery passed (676 discovered, 670 executed, 6 live tests skipped). A separate live-Tk run passed all 6 live tests with no skips. Populated plot captures were visually checked after fixing owned-window foreground capture; these are not complete OS-DPI certification.

## Brief 21b corrections (pending acceptance)

- Saved setup: `core/sample_analysis.py::normalize_analysis_setup` copies historical records, normalizes only schema-1 missing/null inactive method payloads, preserves active settings and meaningful unspecified limits, and validates canonical schema 2. Restore, Save Setup, and Run use this common baseline. Unknown versions, active null payloads, and malformed method payloads remain explicit errors. Tests cover both historical method directions, Save Setup before Run, successful dispatch, switching defaults, and disk save/reopen without restoring calculations.
- Comparison provenance: frozen `CapturedAnalysisRequest` and `MethodResultEnvelope` records retain settings, project/session identity, accepted inputs and a unique request ID. Current comparisons render only their own successful sides. A visible status on every result tab identifies failed sides. The result selector exposes earlier successes separately as **Previous successful result**, with their captured metadata and view-specific stale status; no old rows/curves are borrowed by a failing current side. These envelopes are session-only, not durable history.
- Rollback/lifecycle: failed rendering restores the selected data, provenance and selection together. Stale completion cannot replace a selected earlier result. Operation diagnostics are tied to the input identity at which they occurred, so a subsequent reacceptance can still mark the viewed result stale. The broad workflow smoke explicitly asserts its expected `binned_selection` area1 diagnostic instead of accepting an arbitrary success/failure.
- Regression evidence: `test_schema1_null_method_setup_restores_runs_and_saves` and `test_comparison_current_and_previous_provenance_are_separate` exercise current and prior views, both partial-failure directions, both failures/successes, table values, plotted legends/data, metadata, renderer rollback and stale completion. Screenshots `analysis-21b-current-partial-failure.png` and `analysis-21b-previous-success.png` were captured and visually inspected.

Brief 21b was pushed as `0cd4ac85b155cbfbef01220cfb6047ff947e4b12` ("b side"). Recorded local gates passed: Ruff, focused analysis/project tests (36), full unit discovery (680 discovered, 672 executed, 8 live tests skipped), and all 8 separately executed live tests. Hosted [run 43368913949](https://github.com/jsoteloflores/PyRo-FOAMS/actions/runs/43368913949) passed lint and all four unit jobs, but [the Windows/Python 3.12 live job](https://github.com/jsoteloflores/PyRo-FOAMS/actions/runs/43368913949/job/131699882497) failed. Brief 21c addresses its presentation failure below; no complete hosted acceptance is claimed for Brief 21b.

## Brief 21c layout correction (pending hosted acceptance)

The result notebook now occupies the expanding weighted grid area. First entry to Plot collapses setup and its secondary action toolbar; **Show setup / Hide setup** reopens or hides them without changing drafts, requests or science. Run, Cancel, setup toggle and Back to preparation stay in the primary row, with the result selector below. Setup scrolls and its height is constrained to preserve useful result space when reopened. The minimum window size and chosen fonts are unchanged.

The sample heading contains only a width-fitted name and included-image count/binary scope. Full names, images, calibrations, request IDs and settings remain in Details. Current/previous identity, stale state and per-method outcomes remain visible above every result tab. Labels wrap to available width through coalesced, value-guarded resize callbacks; long operational status text is limited to two visible lines with scrolling rather than consuming plot space.

`test_analysis_results_focused_layout_preserves_state` covers populated Original, Modern and Compare success, partial failure, both failures, and a separately selected prior success with changed inputs. Each is checked at 1280x800 and 1400x900 with normal/doubled named fonts, both collapsed and reopened setup (48 measured layouts). Checks preserve edits, envelope identity and provenance, verify keyboard access and unchanged preparation-call counts, and require at least 240 canvas pixels and 120 axes pixels. The original greater-than-100 assertion remains in `test_analysis_settings_workers_and_native_rendering`.

Every layout assertion logs `ANALYSIS_LAYOUT` JSON with state, requested/actual size, font scale, Tk scaling, plot width/height and setup visibility. An owned-window screenshot is captured before assertions and cleanup, including failing states. Screenshots use `analysis-21c-{state}-{width}x{height}-font{scale}x.png`; doubled-font current Compare and previous-inputs-changed captures were visually inspected for readable axes/legends and clear identity/stale distinctions.

Local measurements at Tk scaling 1.3335775335775335: results-focused canvas heights were 485-719 pixels across the six states; reopened setup retained at least 245 pixels. At 1280x800/doubled fonts, long-status current success/failure plots measured 485 pixels, and the separately selected stale prior result measured 517 pixels. These are measured local results, not predictions of hosted font/DPI behavior.

The reviewed hosted failure reported a 70-pixel plot in `test_analysis_settings_workers_and_native_rendering`. Its exact size/font iteration was not identified in the provided log evidence and is not inferred from local screenshots. New per-iteration diagnostics are intended to resolve that ambiguity on the corrected hosted run. Public job pages confirm failure but require sign-in for detailed logs.

Brief 21c was pushed as `db70a769864b5bb8c1ddaf59cf6057494318d73a` ("more fixin"). Recorded local checks passed: Ruff, unit discovery (681 discovered, 672 executed, 9 live tests skipped), and all 9 separately executed live tests. Hosted [run 43370370979](https://github.com/jsoteloflores/PyRo-FOAMS/actions/runs/43370370979) passed lint and all four unit jobs, but [the Windows GUI job](https://github.com/jsoteloflores/PyRo-FOAMS/actions/runs/43370370979/job/131704518570) failed actual-size equality in two tests: requested 1280x800, actual 1028x749, with plots of 536/568 pixels. This is not the earlier 70-pixel regression and does not establish completion of the full hosted matrix. The exact desktop/WM constraint needs the geometry diagnostics below.

## Brief 21d explicit display coverage (pending hosted gates)

CI now declares two distinct tracks without removing Windows GUI tests:

- **Windows available display**, `PYRO_FOAMS_GUI_DISPLAY_MODE=available`: derive client-size targets from measured work area, native decorations and WM maximum, with the supported 980x640 minimum. Preserve normal/doubled fonts, all outcomes, open/closed setup and the nine scientific/migration/provenance/lifecycle tests. Actual-size equality is still required against these declared targets. Duplicate size targets on a small desktop are reported once, not mislabeled as the two full review sizes.
- **Linux exact-size display**, `PYRO_FOAMS_GUI_DISPLAY_MODE=exact`: run the same tests at both original 1280x800/1400x900 sizes on a configured 1920x1200, 96-DPI Xvfb display with Openbox, Tk, screenshot and X11 dependencies. A bounded `_NET_SUPPORTING_WM_CHECK` readiness probe requires a window manager before testing focus. Exact mode fails if target client geometry is unavailable; it never silently clamps targets.

The harness settles mapped client geometry through a bounded Tk event loop, not arbitrary sleeps. Requests are positioned at the measured work-area origin. JSON artifacts record platform/Python/Tk versions, screen and virtual-root dimensions, Tk scaling, WM min/max, requested/actual client sizes, client bounds, Windows decorated bounds/work area, focus, screenshot dimensions/scope and matrix state/font scale. Mismatches record whether the target exceeds reported WM/work-area limits. These distinguish a measured constraint from an unexplained settled mismatch; no root cause is inferred solely from the old 1028x749 result.

Owned-window PNGs are captured before assertions and cleanup. Windows captures include decorated bounds; other platforms explicitly record client capture scope when decorated bounds are unavailable. Screenshot dimensions and on-screen bounds are checked; cropped diagnostics are marked incomplete and cannot be accepted as visual coverage. Available-mode filenames are prefixed `available-` and use actual tested dimensions. Both CI tracks upload `display-*.json` and PNGs, including failures. Existing greater-than-100, 240-pixel canvas and 120-pixel axes requirements remain unchanged.

Local Windows/Python 3.11 evidence: Ruff passed; unit discovery passed (683 discovered, 674 executed, 9 live tests skipped). All nine live tests passed in exact mode on the measured 3440x1440 primary screen (work area 3440x1392; Tk scaling 1.3335775335775335), exercising both original sizes and font scales. All nine also passed with an explicitly simulated native WM limit of 1008x689 on that otherwise large desktop. This simulation is not hosted Windows desktop coverage. Its analysis plots retained at least 374 pixels collapsed and 245 pixels with setup open; complete 1024x728 decorated screenshots were checked. Product layout did not require changes.

Brief 21d was pushed as `c006a073b03139036298a0edb801b325c5cb8d2b` ("fix"). Hosted [run 43373538516](https://github.com/jsoteloflores/PyRo-FOAMS/actions/runs/43373538516) passed lint, all four unit jobs, and Windows available-display GUI coverage. Review evidence records actual 1008x681 Windows clients and successful Linux analysis-layout matrices at both original sizes/font scales. The Linux exact-display job still failed the broad workflow's reverse-Tab assertion; it did not complete the full GUI gate. Both display tracks remain required.

## Brief 21e reverse navigation (pending hosted verification)

Preparation controls, the image canvas and participating scrollbars retain their widget-local Tab/Shift-Tab handlers. On X11 only, `ISO_Left_Tab` is now bound to the existing reverse handler, which returns `break`; no global key interception or new focus order is introduced. `_preparationFocusOrder` uses ttk's state API for ttk widgets, including scrollbars that have no `state` configuration option, while preserving Canvas state handling.

Observed local Windows event: generated Shift-Tab delivered `keysym=Tab`, `keycode=9`, `state=1`. Binding tags were widget, `TButton`/`Canvas`/`TScrollbar`, toplevel, and `all`; the widget binding called `_onPreparationShiftTab` and consumed the event. Add Images and the image canvas each reached Save Project with one actual focus transition. Visible scrollbar traversal also produces a `NotifyNonlinearVirtual` notification for the ancestor canvas followed by `NotifyNonlinear` for the focused child; these are logged separately rather than miscounted as two transitions.

The regression dispatches real Tk key events, records the delivered keysym/modifiers, applicable binding scripts/tags, raw focus details and final target in the existing JSON artifacts, and asserts one actual transition. X11 runs both the existing Shift-Tab simulation and native `ISO_Left_Tab`; Windows retains its existing simulation. Ordinary predecessors, disabled-control skipping, focused-control visibility and visible vertical/horizontal scrollbars are checked at normal/doubled fonts. A reversible one-group-per-row overflow fixture exercises actual scrollbar bindings without retaining layout edits or changing the user's font. The broad test still continues to batch, analysis, accepted export and project reopen.

The prior hosted Linux failure did not record its delivered keysym, so the ISO_Left_Tab diagnosis is not claimed as an observed X11 fact yet. The missing native binding is corrected and the next hosted JSON will establish delivery/binding order. Local Linux execution remains unavailable on this Windows host.

Brief 21e was pushed as `dc61136341757b7853c65386edd70fbba7bb833a`. Recorded local validation passed Ruff, unit discovery (683 discovered, 674 executed, 9 live tests skipped), and the GUI module's 11 tests (9 live plus 2 policy). Hosted [run 43376334915](https://github.com/jsoteloflores/PyRo-FOAMS/actions/runs/43376334915) passed lint and all four unit jobs. Both Python 3.12 GUI jobs failed during root destruction and then reported cascading font errors, despite passing keyboard assertions; their complete acceptance remains open. Review identified the probe callback ownership mismatch addressed below.

## Brief 21f probe ownership (pending Python 3.12/hosted verification)

The keyboard diagnostic probe is now explicitly registered through `self.root.bind_class` and removed through `self.root.deletecommand`. [CPython 3.12.10's tkinter source](https://github.com/python/cpython/blob/v3.12.10/Lib/tkinter/__init__.py) routes class binding registration through the root; deleting through a child can leave a root-tracked name that root destruction later tries to delete again. Registration and deletion now share one owner on both supported versions.

Both callback handles start as `None` before `try`. Cleanup restores source bindtags, removes the temporary probe script before deleting its successfully registered callback, restores original FocusIn scripts, and removes the separately root-owned focus callback. Partial registration failures preserve their original exception. No private callback lists or tkinter default-root state are cleared or reset.

`test_keyboard_probe_cleanup_allows_fresh_root_fonts` exercises an injected registration failure, repeated real probe creation/cleanup on child controls, exact tag/script restoration, normal `root.destroy`, then creation of a fresh root with working implicit/explicit TkDefaultFont lookups. Normal teardown propagates Tcl errors when destroying live roots. A Tcl window-command existence check avoids operating on the root after the broad workflow has already closed it intentionally; it does not catch or suppress a live-root destruction failure.

Local Windows/Python 3.11 validation: Ruff passed; unit discovery passed (684 discovered, 674 executed, 10 live tests skipped); all 12 GUI-module tests passed separately (10 live plus 2 policy). The new ownership regression, broad close/reopen workflow and setup-migration test also passed in that explicit order in one process. Existing navigation, single-transition, provenance, layout and both CI display modes are unchanged.

Only Python 3.11 is listed by the installed launcher on this host, so the required Python 3.12 sequence has not been executed locally. Both existing GUI CI tracks use Python 3.12 and will run the new regression with the broad workflow and migration in the same module/process. Brief 21f changes are uncommitted work on `dc61136341757b7853c65386edd70fbba7bb833a`; no corrected pushed SHA or final hosted outcome is claimed. Successful corrected-revision lint, four unit jobs and both GUI jobs remain required before independent review and Brief 22. Scope is the diagnostic harness only; full original-FOAMS readiness remains unclaimed.

## Live GUI test

On a display-capable Windows host, run the committed widget-level smoke and capture empty and populated workspace screenshots at both sizes and font settings:

```powershell
$env:PYRO_FOAMS_LIVE_TK = "1"
$env:PYRO_FOAMS_GUI_DISPLAY_MODE = "exact"
$env:PYRO_FOAMS_GUI_ARTIFACTS = "artifacts/gui-smoke"
python -m unittest -v main.tests.test_sample_project_gui
```

The test drives real workspace and processing-window buttons, delays batch computation to verify Tk timer responsiveness, checks accepted-input export routing, draft file round-trips and partial failures, cancellation-safe cleanup, keyboard traversal, and layout bounds. Screenshots include `workspace_1280x800_font1x.png`, `workspace_1280x800_font2x.png`, `workspace_1400x900_font1x.png`, `workspace_1400x900_font2x.png`, populated workspace captures, and the analysis captures above. Use explicit `available` mode on a constrained Windows desktop; its measured sizes must not be reported as exact-size coverage. GitHub Actions now has separate Windows-available and Linux-exact Python 3.12 jobs, uploading JSON and PNG evidence. Run without `PYRO_FOAMS_LIVE_TK=1` only for discovery; live tests will be skipped, which is not GUI acceptance.

Commit `4928e03f3844919ded6716d0f93075174e530ec9` passed hosted run [42262702398](https://github.com/jsoteloflores/PyRo-FOAMS/actions/runs/42262702398): lint, all four Linux/Windows Python 3.11/3.12 unit jobs, and the Windows Python 3.12 live-Tk job succeeded. The GUI job uploaded artifact `windows-live-tk-gui-smoke` (artifact ID `13526245699`), containing the screenshots listed above.

Earlier run `42244733714` stalled during dependency installation for more than 54 minutes. Run `42259395489` did not complete its GUI step; local reproduction identified an unmocked standalone-export success dialog. The test now mocks that dialog and captures Tk stderr as a failure; the passing run above verifies the correction. Same-real-image comparison against MATLAB FOAMS remains future work; no scientific formulas changed.
