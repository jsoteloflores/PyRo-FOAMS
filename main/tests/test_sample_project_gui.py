import contextlib
import io
import os
import sys
import tempfile
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import unittest
from pathlib import Path
from tkinter import ttk
from unittest.mock import patch

import cv2
import numpy as np
from PIL import ImageGrab

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main.core.sample_project import (
    accept_draft,
    add_image,
    create_analysis_snapshot,
    create_project,
    set_analysis_included,
    set_calibration,
    update_draft,
)
from main.core.stereology import colorize_labels as real_colorize_labels
from main.gui import batch_controller, postprocessing, preprocessing, processing, sample_analysis

LIVE_TK = os.environ.get("PYRO_FOAMS_LIVE_TK") == "1"


class TestLiveSampleProjectGui(unittest.TestCase):
    def setUp(self):
        if not LIVE_TK:
            self.skipTest("Set PYRO_FOAMS_LIVE_TK=1 on a display-capable host to run live Tk integration.")
        self.callback_errors = []
        self.tk_stderr = io.StringIO()
        self.stderr_redirect = contextlib.redirect_stderr(self.tk_stderr)
        self.stderr_redirect.__enter__()
        self.root = tk.Tk()
        self.root.report_callback_exception = self._report_callback_exception
        self.root.geometry("1280x800+0+0")
        self.root.deiconify()
        self.app = preprocessing.PreprocessApp(self.root)
        self.root.update()

    def tearDown(self):
        if hasattr(self, "root"):
            try:
                for child in tuple(self.root.winfo_children()):
                    if isinstance(child, tk.Toplevel):
                        child.destroy()
                preprocessing.cancel_debounced_callbacks(self.root)
                self.root.destroy()
            except tk.TclError:
                pass
        self.stderr_redirect.__exit__(None, None, None)
        self.assertNotRegex(
            self.tk_stderr.getvalue(),
            r"invalid command name|Exception in Tkinter callback|Traceback \(most recent call last\)",
        )

    def _report_callback_exception(self, exception_type, exception, traceback):
        self.callback_errors.append((exception_type, exception, traceback))

    def _checkpoint(self, name):
        print(f"LIVE_TK_GATE: {name}", flush=True)

    def _top_levels(self, root):
        return [child for child in root.winfo_children() if isinstance(child, tk.Toplevel)]

    def _walk(self, widget):
        yield widget
        for child in widget.winfo_children():
            yield from self._walk(child)

    def _wait_until(self, predicate, root, timeout=8.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            root.update()
            self.assertFalse(self.callback_errors, repr(self.callback_errors))
            threading.Event().wait(0.005)
        self.assertTrue(predicate(), "timed out waiting for live Tk workflow state")
        self.assertFalse(self.callback_errors, repr(self.callback_errors))

    def test_analysis_settings_workers_and_native_rendering(self):
        from main.core.sample_analysis import prepare_legacy_analysis, run_legacy_analysis, run_modern_analysis
        from main.core.sample_project_io import open_project, save_project
        from main.tests.test_sample_analysis import TestSampleAnalysisAdapters

        project, _ = TestSampleAnalysisAdapters()._snapshot_with_three_images()
        set_analysis_included(project.images[2], False)
        project.analysis_setup = {"schema": 1, "mode": "Modern analysis", "legacy": {
            "minimum_diameter_px": None, "size_lower_cutoff_mm": None, "size_upper_cutoff_mm": None,
            "selection_mode": None, "perimeter_profile": None, "shape_statistics_mode": None}}
        window = sample_analysis.SampleAnalysisWorkspace(self.root, project, create_analysis_snapshot(project))
        window.modeVar.set("Original FOAMS 1.0.5")
        artifacts = Path(os.environ.get("PYRO_FOAMS_GUI_ARTIFACTS", "main/tests/gui_smoke_artifacts"))

        def finish(allow_render_error=False):
            self._wait_until(lambda: not window._busy, self.root)
            if not allow_render_error:
                self.assertNotIn("Rendering failed", window.statusVar.get())

        def capture(name):
            artifacts.mkdir(parents=True, exist_ok=True)
            window.deiconify()
            window.lift()
            window.attributes("-topmost", True)
            self.root.update()
            left, top = window.winfo_rootx(), window.winfo_rooty()
            ImageGrab.grab(bbox=(left, top, left + window.winfo_width(), top + window.winfo_height())).save(artifacts / name)
            window.attributes("-topmost", False)

        with patch("tkinter.messagebox.showerror", side_effect=AssertionError("unexpected modal error")):
            with patch("main.gui.sample_analysis.prepare_legacy_analysis", wraps=prepare_legacy_analysis) as preparation:
                for profile in ("R2025b", "PerimeterOld"):
                    window.perimeterProfileVar.set(profile)
                    window.runButton.invoke()
                    finish()
                    self.assertEqual(window.statusVar.get(), "Original run complete", window.resultsText.get("1.0", "end"))
                    direct = run_legacy_analysis(window.snapshot, window._parse_legacy_config())
                    actual = window._latest_legacy.analysis
                    self.assertEqual(actual.binned_selection_result.rows, direct.analysis.binned_selection_result.rows)
                    self.assertEqual(actual.raw_selection_result.selected_identities, direct.analysis.raw_selection_result.selected_identities)
                    self.assertEqual(actual.volume_result.adjusted_volume_fractions, direct.analysis.volume_result.adjusted_volume_fractions)
                    self.assertGreater(len(window.distributionTable.get_children()), 0)
                capture("analysis-distributions.png")
                calls = preparation.call_count
                window.volumeModeVar.set("integrated_2d")
                window.runButton.invoke()
                finish()
                self.assertEqual(preparation.call_count, calls)
                rows = [row_id for row_id, row in window._table_rows.items() if row[0] == "legacy"]
                window.distributionTable.selection_set(rows)
                window.binGroupVar.set("group-1")
                window.applyBinsButton.invoke()
                self.assertEqual(window.selectionModeVar.get(), "manual")
                window.runButton.invoke()
                finish()
                self.assertEqual(window.statusVar.get(), "Original run complete")
                self.assertEqual(preparation.call_count, calls)
                capture("analysis-manual-cutoffs.png")
                direct = run_legacy_analysis(window.snapshot, window._parse_legacy_config())
                self.assertEqual(window._latest_legacy.analysis.raw_selection_result.selected_identities,
                                 direct.analysis.raw_selection_result.selected_identities)
                self.assertGreater(window._latest_legacy.analysis.raw_selection_result.selected_count, 0)
                window.volumeModeVar.set("known_vesicularity")
                window.knownVesicularityVar.set("25")
                window.runButton.invoke()
                finish()
                self.assertEqual(window.statusVar.get(), "Original run complete")
                self.assertEqual(preparation.call_count, calls)
                window.shapeMinimumVar.set("2.5")
                window.shapeUpdateButton.invoke()
                finish()
                self.assertEqual(window._latest_legacy.analysis.analysis_settings.shape_statistics_mode, "updated_minimum")
                self.assertEqual(preparation.call_count, calls)
                window.suggestButton.invoke()
                finish()
                self.assertIn("Suggested ranges", window.statusVar.get())
                suggestion = window.manualRangesVar.get()
                window.selectionModeVar.set("autosmart")
                window.manualRangesVar.set(suggestion)
                self.assertEqual(window.selectionModeVar.get(), "manual")
            window.modeVar.set("Modern analysis")
            window.modernMinDetectableVar.set("0.001")
            window.modernEdgesVar.set("0.001,0.01,0.1")
            window.minimumDiameterVar.set("unfinished")
            window.runButton.invoke()
            finish()
            self.assertEqual(window.statusVar.get(), "Modern run complete")
            window.saveSetupButton.invoke()
            setup = dict(project.analysis_setup)
            window._on_close()
            with tempfile.TemporaryDirectory() as folder:
                save_project(project, Path(folder))
                project = open_project(Path(folder))
            window = sample_analysis.SampleAnalysisWorkspace(self.root, project, create_analysis_snapshot(project))
            self.assertEqual(window.modernEdgesVar.get(), "0.001,0.01,0.1")
            self.assertEqual(window.minimumDiameterVar.get(), "unfinished")
            self.assertEqual(window.manualRangesVar.get(), setup["drafts"]["manualRangesVar"])
            self.assertEqual(window.volumeModeVar.get(), "known_vesicularity")
            self.assertEqual(window.shapeModeVar.get(), "updated_minimum")
            window.minimumDiameterVar.set("2")
            window.modeVar.set("Original FOAMS 1.0.5")
            window.runButton.invoke()
            finish()
            self.assertEqual(window.statusVar.get(), "Original run complete")
            self.assertEqual(tuple(tuple(item) for item in project.analysis_setup["legacy"]["manual_ranges"]),
                             window._parse_manual_ranges())
            window.modeVar.set("Compare methods")
            window.runButton.invoke()
            finish()
            self.assertIsNotNone(window._latest_compare.legacy_result)
            self.assertIsNotNone(window._latest_compare.modern_result)
            direct = run_modern_analysis(window.snapshot, window._parse_modern_config())
            self.assertEqual(window._latest_compare.modern_result.nesting_result.counts, direct.nesting_result.counts)
            capture("analysis-comparison.png")
            window.modernMinDetectableVar.set("")
            window.runButton.invoke()
            finish()
            self.assertIsNotNone(window._latest_compare.legacy_result)
            self.assertIsNone(window._latest_compare.modern_result)
            window.modernMinDetectableVar.set("0.001")
            window.runButton.invoke()
            finish()
            fonts = {name: tkfont.nametofont(name).cget("size") for name in tkfont.names()}
            try:
                for scale in (1, 2):
                    for name, size in fonts.items():
                        tkfont.nametofont(name).configure(size=size * scale)
                    for width, height in ((1280, 800), (1400, 900)):
                        window.geometry(f"{width}x{height}+0+0")
                        self.root.update()
                        window.plotCanvas.get_tk_widget().master.master.select(1)
                        self.root.update()
                        capture(f"analysis-plot-{width}x{height}-font{scale}x.png")
                        self.assertGreater(window.plotCanvas.get_tk_widget().winfo_height(), 100)
                        window.plotCanvas.draw()
                        pixels = np.asarray(window.plotCanvas.buffer_rgba())
                        self.assertGreater(np.count_nonzero(np.any(pixels[:, :, :3] < 200, axis=2)), 100)
                        self.assertTrue(window.runButton.winfo_ismapped())
                        self.assertLessEqual(window.statusVar.get().count("\n"), 1)
            finally:
                for name, size in fonts.items():
                    tkfont.nametofont(name).configure(size=size)
                window.plotCanvas.get_tk_widget().master.master.select(0)
            previous_text = window.resultsText.get("1.0", "end")
            previous_values = [window.distributionTable.item(row_id, "values") for row_id in window.distributionTable.get_children()]
            with patch.object(window, "_render_compare", side_effect=RuntimeError("injected renderer failure")):
                window.runButton.invoke()
                finish(allow_render_error=True)
            self.assertIn(previous_text.strip(), window.resultsText.get("1.0", "end"))
            self.assertEqual(previous_values, [window.distributionTable.item(row_id, "values") for row_id in window.distributionTable.get_children()])
            self.assertIn("Rendering failed", window.statusVar.get())
            window.manualRangesVar.set("group-1:123:456")
            window.runButton.invoke()
            finish()
            self.assertIsNone(window._latest_compare.legacy_result)
            self.assertIsNotNone(window._latest_compare.modern_result)
            self.assertIn("failed", window.statusVar.get())
            capture("analysis-validation-error.png")

            entered, release = threading.Event(), threading.Event()
            thread_ids = []
            real_prepare = prepare_legacy_analysis

            def slow_prepare(*args, **kwargs):
                thread_ids.append(threading.get_ident())
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test worker timeout")
                return real_prepare(*args, **kwargs)

            window._legacy_prepare_cache.clear()
            with patch("main.gui.sample_analysis.prepare_legacy_analysis", side_effect=slow_prepare):
                window.suggestButton.invoke()
                self._wait_until(entered.is_set, self.root)
                self.assertNotEqual(thread_ids[0], threading.get_ident())
                self.assertEqual(str(window.setupControls.children['!combobox'].cget("state")), "disabled")
                set_calibration(project.images[0], 0.004, "mm")
                release.set()
                finish()
                self.assertIn("Discarded stale", window.statusVar.get())
            window._on_close()

    def _capture_workspace(self, artifact_dir, name):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.update_idletasks()
        self.root.update()
        left = self.root.winfo_rootx()
        top = self.root.winfo_rooty()
        right = left + self.root.winfo_width()
        bottom = top + self.root.winfo_height()
        ImageGrab.grab(bbox=(left, top, right, bottom)).save(artifact_dir / name)
        self.root.attributes("-topmost", False)

    def test_analysis_explicit_modern_plan_and_cancellation(self):
        from main.core.sample_analysis import prepare_modern_analysis
        from main.tests.test_sample_analysis import TestSampleAnalysisAdapters

        project, snapshot = TestSampleAnalysisAdapters()._snapshot_with_three_images()
        window = sample_analysis.SampleAnalysisWorkspace(self.root, project, snapshot)
        window.modeVar.set("Modern analysis")
        window.modernMinDetectableVar.set("0.001")
        window.prepareButton.invoke()
        self._wait_until(lambda: not window._busy, self.root)
        self.assertIsNotNone(window._prepared_modern.distribution_result)
        self.assertIsNone(window._latest_modern)
        for group, start, stop in (("group-2", 0, 4), ("group-1", 4, 8)):
            rows = [row_id for row_id, row in window._table_rows.items()
                    if row[0] == "modern" and row[1] == group and start <= row[2] < stop]
            self.assertEqual(len(rows), stop - start)
            window.distributionTable.selection_set(rows)
            window.binGroupVar.set(group)
            window.applyBinsButton.invoke()
        self.assertEqual(window.modernSegmentsVar.get(), "group-2:0:4;group-1:4:8")
        window.runButton.invoke()
        self._wait_until(lambda: not window._busy, self.root)
        nested = window._latest_modern.nesting_result
        self.assertEqual(nested.source_group_ids, ("group-2",) * 4 + ("group-1",) * 4)
        for index, group in enumerate(nested.source_group_ids):
            native = window._latest_modern.distribution_result.groups[(project.project_id, group)]
            self.assertEqual(nested.eligible_areas_mm2[index], native.eligible_areas_mm2[index])
            self.assertEqual(nested.counts[index], native.counts[index])

        completed = window._latest_modern
        entered, release = threading.Event(), threading.Event()

        def delayed(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test cancellation timeout")
            return prepare_modern_analysis(*args, **kwargs)

        try:
            with patch("main.gui.sample_analysis.prepare_modern_analysis", side_effect=delayed):
                window.runButton.invoke()
                self._wait_until(entered.is_set, self.root)
                window.cancelButton.invoke()
                self.assertFalse(window._busy)
                release.set()
                self._wait_until(lambda: not window._worker.is_alive(), self.root)
                self.assertIs(window._latest_modern, completed)
                self.assertIn("Cancelled", window.statusVar.get())
        finally:
            release.set()
        image = project.images[0]
        update_draft(image, mask=np.zeros_like(image.accepted_mask), labels=None, settings={})
        self.assertEqual(window._completed_inputs, window._input_signature())
        accept_draft(image)
        self._wait_until(lambda: "Inputs changed" in window.statusVar.get(), self.root)
        window.runButton.invoke()
        self._wait_until(lambda: not window._busy, self.root)
        self.assertEqual(window._last_completed_snapshot.images[0].snapshot_id, image.accepted_snapshot.snapshot_id)
        self.assertIsNot(window._latest_modern, completed)
        window._on_close()

    def _assert_keyboard_traversal_reaches_primary_controls(self):
        expected = set(self.app.workspaceFocusControls)
        expected.discard(self.app.includeButton)
        visited = set()
        for scrollbar in (self.app.imagesVerticalScrollbar, self.app.imagesHorizontalScrollbar):
            if scrollbar.winfo_ismapped():
                expected.add(scrollbar)
        self.app.sampleEntry.focus_set()
        self.root.update()
        for _ in range(len(expected) + 3):
            current = self.root.focus_get()
            self.assertIsNotNone(current)
            if current in expected:
                visited.add(current)
            current.event_generate("<Tab>")
            self.root.update()
        self.assertTrue(expected.issubset(visited), "Tab traversal omitted a primary workspace control")
        self.app.addImagesButton.focus_set()
        self.root.update()
        self.app.addImagesButton.event_generate("<Shift-Tab>")
        self.root.update()
        self.assertIs(self.root.focus_get(), self.app.saveProjectButton)

    def _assert_workspace_controls_fit(self, width, height, artifact_dir, font_scale):
        self.root.geometry(f"{width}x{height}+0+0")
        self.root.deiconify()
        self.root.lift()
        self.root.update_idletasks()
        self.app._reflowPreparationToolbar()
        self.root.update()
        left = self.root.winfo_rootx()
        top = self.root.winfo_rooty()
        right = left + self.root.winfo_width()
        bottom = top + self.root.winfo_height()
        controls = (
            self.app.newProjectButton,
            self.app.openProjectButton,
            self.app.saveProjectButton,
            self.app.addImagesButton,
            self.app.renameImageButton,
            self.app.setCalibrationButton,
            self.app.cropButton,
            self.app.processButton,
            self.app.editMasksButton,
            self.app.importMaskButton,
            self.app.acceptDraftButton,
            self.app.includeButton,
            self.app.analyzeIncludedButton,
        )
        preparation_controls = {
            self.app.addImagesButton,
            self.app.renameImageButton,
            self.app.setCalibrationButton,
            self.app.cropButton,
            self.app.cropMarginsCheckbutton,
            self.app.processButton,
            self.app.editMasksButton,
            self.app.importMaskButton,
        }
        for control in controls:
            self.assertTrue(control.winfo_ismapped(), control.cget("text"))
            self.assertGreater(control.winfo_width(), 1, control.cget("text"))
            if control in preparation_controls:
                group = control.master
                canvas = self.app.imagesCanvas
                bounds = canvas.bbox("all") or (0, 0, canvas.winfo_width(), canvas.winfo_height())
                canvas.xview_moveto(min(1.0, group.winfo_x() / max(1, bounds[2])))
                canvas.yview_moveto(min(1.0, group.winfo_y() / max(1, bounds[3])))
                self.root.update_idletasks()
                self.root.update()
                canvas_left = canvas.winfo_rootx()
                canvas_top = canvas.winfo_rooty()
                canvas_right = canvas_left + canvas.winfo_width()
                canvas_bottom = canvas_top + canvas.winfo_height()
                self.assertGreaterEqual(control.winfo_rootx(), canvas_left, control.cget("text"))
                self.assertLessEqual(control.winfo_rootx() + control.winfo_width(), canvas_right, control.cget("text"))
                self.assertGreaterEqual(control.winfo_rooty(), canvas_top, control.cget("text"))
                self.assertLessEqual(control.winfo_rooty() + control.winfo_height(), canvas_bottom, control.cget("text"))
            else:
                self.assertGreaterEqual(control.winfo_rootx(), left, control.cget("text"))
                self.assertLessEqual(control.winfo_rootx() + control.winfo_width(), right, control.cget("text"))
                self.assertGreaterEqual(control.winfo_rooty(), top, control.cget("text"))
                self.assertLessEqual(control.winfo_rooty() + control.winfo_height(), bottom, control.cget("text"))
            if str(control.cget("state")) != "disabled":
                control.focus_set()
                self.root.update()
                self.assertIs(self.root.focus_get(), control, f"{control.cget('text')} must be keyboard focusable")

        self.app.imagesCanvas.xview_moveto(0)
        self.app.imagesCanvas.yview_moveto(0)
        self._capture_workspace(
            artifact_dir,
            f"workspace_{width}x{height}_font{font_scale}x.png",
        )

    def _run_batch(self, window, root, target_indices, delayed=False):
        window.targetIndices = list(target_indices)
        started = threading.Event()
        release = threading.Event()
        timer_fired = threading.Event()
        real_process = batch_controller.process_batch_parallel

        def process(images, **kwargs):
            if delayed:
                started.set()
                release.wait(5.0)
            return real_process(images, **kwargs)

        messagebox_patch = patch("main.gui.processing.messagebox.showinfo")
        messagebox_patch.start()
        try:
            if delayed:
                with patch("main.gui.batch_controller.process_batch_parallel", side_effect=process):
                    root.after(20, timer_fired.set)
                    window.applySelectedButton.invoke()
                    self.assertTrue(started.wait(2.0))
                    self._wait_until(timer_fired.is_set, root)
                    self.assertTrue(timer_fired.is_set(), "Tk timer must fire while batch work is blocked")
                    release.set()
                    try:
                        self._wait_until(lambda: window._batchJobId is None, root)
                    except AssertionError as exc:
                        self.fail(
                            f"batch stalled: status={window.batchStatusVar.get()!r}, "
                            f"queue={window._batchCoordinator.events.qsize()}, "
                            f"workers={tuple(window._batchCoordinator._threads)}, "
                            f"Tk errors={self.callback_errors!r}; {exc}"
                        )
            else:
                with patch("main.gui.batch_controller.process_batch_parallel", side_effect=process):
                    window.applySelectedButton.invoke()
                    self._wait_until(lambda: window._batchJobId is None, root)
        finally:
            release.set()
            messagebox_patch.stop()
        self.assertEqual(window._batchSuccessCount, len(target_indices))
        self.assertEqual(window._batchFailureCount, 0)

    def _edit_active_draft(self, index):
        self.app._onThumbClick(index)
        accepted_snapshot = self.app.project.images[index].accepted_snapshot
        accepted_mask = (
            self.app.project.images[index].accepted_mask.copy()
            if self.app.project.images[index].accepted_mask is not None
            else None
        )
        self.app.editMasksButton.invoke()
        self.root.update()
        editor = next(window for window in self._top_levels(self.root) if isinstance(window, postprocessing.PostprocessWindow))
        before = editor.masks[index].copy()
        background = np.argwhere(~before)
        if background.size:
            y, x = (int(value) for value in background[0])
            paint = True
        else:
            foreground = np.argwhere(before)
            self.assertGreater(foreground.size, 0)
            y, x = (int(value) for value in foreground[0])
            paint = False
        editor.radiusVar.set(1)
        editor._stamp((x, y), paint)
        self.assertFalse(np.array_equal(before, editor.masks[index]))
        editor._notify_parent()
        self.assertIs(self.app.project.images[index].accepted_snapshot, accepted_snapshot)
        if accepted_mask is not None:
            np.testing.assert_array_equal(self.app.project.images[index].accepted_mask, accepted_mask)
        editor._on_close()
        self.root.update()

    def _set_active_calibration(self, index, value, unit):
        self.app._onThumbClick(index)
        self.app.setCalibrationButton.invoke()
        self.root.update()
        dialog = next(window for window in self._top_levels(self.root) if window.title() == "Set Calibration")
        entries = [widget for widget in self._walk(dialog) if isinstance(widget, ttk.Entry)]
        combos = [widget for widget in self._walk(dialog) if isinstance(widget, ttk.Combobox)]
        apply_button = next(
            widget for widget in self._walk(dialog)
            if isinstance(widget, ttk.Button) and widget.cget("text") == "Apply"
        )
        entries[0].delete(0, "end")
        entries[0].insert(0, str(value))
        combos[0].set(unit)
        apply_button.invoke()
        self.root.update()

    def test_project_workflow_export_responsive_batch_layout_and_round_trip(self):
        artifact_dir = Path(os.environ.get("PYRO_FOAMS_GUI_ARTIFACTS", tempfile.gettempdir()))
        with patch("main.gui.processing.runSeparationPipeline") as pipeline, patch(
            "main.core.stereology.measure_dataset"
        ) as measure_dataset:
            self._assert_workspace_controls_fit(1280, 800, artifact_dir, 1)
            self._assert_workspace_controls_fit(1400, 900, artifact_dir, 1)
            self._assert_keyboard_traversal_reaches_primary_controls()
            fonts = []
            for font_name in tkfont.names(self.root):
                font = tkfont.nametofont(font_name, root=self.root)
                original_size = int(font.cget("size"))
                if original_size != 0:
                    fonts.append((font, original_size))
                    font.configure(size=original_size * 2)
                    self.assertEqual(int(font.cget("size")), original_size * 2)
            try:
                self.app._reflowPreparationToolbar()
                self._assert_workspace_controls_fit(1280, 800, artifact_dir, 2)
                self._assert_workspace_controls_fit(1400, 900, artifact_dir, 2)
                self._assert_keyboard_traversal_reaches_primary_controls()
            finally:
                for font, original_size in fonts:
                    font.configure(size=original_size)
                    self.assertEqual(int(font.cget("size")), original_size)
                self.app._reflowPreparationToolbar()
                self.root.update()
            pipeline.assert_not_called()
            measure_dataset.assert_not_called()
        self._checkpoint("layout-and-screenshots")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_paths = []
            for index, name in enumerate(("same.tif", "same.tif", "excluded.png")):
                folder = root / str(index)
                folder.mkdir()
                image = np.zeros((72, 72), dtype=np.uint8)
                cv2.circle(image, (22 + index * 10, 36), 13, 210, -1)
                if index == 0:
                    source_image = image.astype(np.uint16) * 257
                    source_image[0, 0] = 10
                    source_image[0, 1] = 11
                    image = source_image
                path = folder / name
                self.assertTrue(cv2.imwrite(str(path), image))
                source_paths.append(str(path))

            project_path = root / "saved-project"
            csv_path = root / "accepted.csv"
            preview_path = root / "accepted-previews"
            preview_path.mkdir()
            with patch.object(preprocessing.filedialog, "askopenfilenames", return_value=tuple(source_paths)):
                self.app.addImagesButton.invoke()
            self.root.update()
            self._capture_workspace(artifact_dir, "workspace_populated_1280x800_font1x.png")
            self.assertEqual(len(self.app.project.images), 3)
            ids = [image.image_id for image in self.app.project.images]
            self.assertEqual(len(set(ids)), 3)
            self.assertEqual(self.app.project.images[0].display_name, self.app.project.images[1].display_name)
            self.assertEqual(self.app.project.images[0].source_image.dtype, np.uint16)
            self.assertEqual(self.app.project.images[0].source_image[0, 0], 10)
            self.assertEqual(self.app.project.images[0].source_image[0, 1], 11)
            self.assertEqual(self.app.project.images[0].working_image[0, 0], 0)
            self.assertEqual(self.app.project.images[0].working_image[0, 1], 0)
            self._checkpoint("import-duplicate-basenames")

            with patch("main.gui.preprocessing.messagebox.showwarning") as warning, patch.object(
                preprocessing.filedialog, "asksaveasfilename"
            ) as save_dialog:
                self.app.onExportAcceptedMeasurements()
                warning.assert_called_once()
                save_dialog.assert_not_called()

            self.app.batchSelection = {0, 1}
            self.app.processButton.invoke()
            self.root.update()
            processing_window = next(
                window for window in self._top_levels(self.root)
                if isinstance(window, processing.ProcessingWindow)
            )
            self._run_batch(processing_window, self.root, {0, 1}, delayed=True)
            self.assertIsNotNone(self.app.project.images[0].draft_mask)
            self.assertIsNotNone(self.app.project.images[1].draft_mask)
            self.assertIsNone(self.app.project.images[2].draft_mask)
            self.assertFalse(self.app.project.images[0].analysis_included)
            self._checkpoint("selected-responsive-batch")
            processing_window._onClose()
            self.root.update()

            self._edit_active_draft(0)
            self.app.acceptDraftButton.invoke()
            accepted_a = self.app.project.images[0].accepted_mask.copy()
            self.app.includeButton.invoke()
            self.assertEqual(self.app.includeButton.cget("text"), "Exclude")
            self.app.includeButton.focus_set()
            self.root.update()
            self.assertIs(self.root.focus_get(), self.app.includeButton)
            self._checkpoint("manual-edit-accept-include-first")

            self.app._onThumbClick(2)
            self.app.batchSelection = {2}
            self.app.processButton.invoke()
            self.root.update()
            processing_window = next(
                window for window in self._top_levels(self.root)
                if isinstance(window, processing.ProcessingWindow)
            )
            self._run_batch(processing_window, self.root, {2})
            self.assertIsNotNone(self.app.project.images[2].draft_mask)
            self.assertIsNone(self.app.project.images[2].accepted_snapshot)
            self.app._onThumbClick(2)
            self.app.acceptDraftButton.invoke()
            self.assertFalse(self.app.project.images[2].analysis_included)
            self.assertEqual(self.app.includeButton.cget("text"), "Include")
            processing_window._onClose()
            self.root.update()
            self._checkpoint("accept-excluded-third")

            self.app._onThumbClick(1)
            self.app.acceptDraftButton.invoke()
            self.app.includeButton.invoke()
            self._set_active_calibration(0, 2.0, "um")
            self._set_active_calibration(1, 0.003, "mm")
            self._set_active_calibration(2, 0.01, "cm")
            self._edit_active_draft(0)
            self._checkpoint("calibration-and-new-draft")
            self.assertFalse(np.array_equal(self.app.project.images[0].draft_mask, accepted_a))
            self.assertTrue(np.array_equal(self.app.project.images[0].accepted_mask, accepted_a))

            self.app.analyzeIncludedButton.invoke()
            self.root.update()
            analysis_window = next(
                window for window in self._top_levels(self.root)
                if isinstance(window, sample_analysis.SampleAnalysisWorkspace)
            )
            self.assertEqual(len(analysis_window.snapshot.images), 2)
            np.testing.assert_array_equal(
                analysis_window.snapshot.images[0].accepted_labels,
                self.app.project.images[0].accepted_labels,
            )
            with patch("tkinter.messagebox.showerror") as analysis_error:
                analysis_window.runButton.invoke()
                self._wait_until(lambda: str(analysis_window.runButton.cget("state")) == "normal", self.root)
                analysis_error.assert_not_called()
            original_text = analysis_window.resultsText.get("1.0", "end")
            self.assertTrue(
                ("Original FOAMS" in original_text) or ("Analysis run failed" in original_text),
                original_text,
            )

            analysis_window.modeVar.set("Compare methods")
            analysis_window.modernMinDetectableVar.set("0.001")
            analysis_window.modernSegmentsVar.set("group-2:0:4;group-1:4:8")
            with patch("tkinter.messagebox.showerror") as compare_error:
                analysis_window.runButton.invoke()
                self._wait_until(lambda: str(analysis_window.runButton.cget("state")) == "normal", self.root)
                compare_error.assert_not_called()
            compare_text = analysis_window.resultsText.get("1.0", "end")
            self.assertIn("Compare methods", compare_text)
            self.assertIn("Legacy path:", compare_text)
            self.assertIn("Modern path:", compare_text)
            self._checkpoint("analyze-included")

            with patch.object(preprocessing.filedialog, "askdirectory", return_value=str(project_path)):
                self.app.saveProjectButton.invoke()
            self.assertTrue((project_path / "manifest.json").is_file())
            self._checkpoint("save-project")

            processing_window = None
            self.app._onThumbClick(0)
            self._checkpoint("before-export-processing-window")
            self.app.processButton.invoke()
            self._checkpoint("process-button-invoked-for-export")
            self.root.update()
            self._checkpoint("root-updated-for-export")
            processing_window = next(
                window for window in self._top_levels(self.root)
                if isinstance(window, processing.ProcessingWindow)
            )
            self._checkpoint("found-export-processing-window")
            self.assertIn("2", processing_window.exportMeasurementsButton.cget("text"))
            project_export_callback = processing_window.projectMeasurementExportCallback
            processing_window.projectMeasurementExportCallback = None
            with patch.object(processing.messagebox, "showerror") as export_error, patch.object(
                processing.filedialog, "asksaveasfilename"
            ) as save_dialog:
                processing_window.exportMeasurementsButton.invoke()
                export_error.assert_called_once()
                save_dialog.assert_not_called()
            processing_window.projectMeasurementExportCallback = project_export_callback

            def failing_project_export():
                raise RuntimeError("synthetic project export failure")

            processing_window.projectMeasurementExportCallback = failing_project_export
            with patch.object(processing.messagebox, "showerror") as export_error, patch.object(
                processing.filedialog, "asksaveasfilename"
            ) as save_dialog:
                processing_window.exportMeasurementsButton.invoke()
                export_error.assert_called_once()
                save_dialog.assert_not_called()
            processing_window.projectMeasurementExportCallback = project_export_callback
            captured = {}
            real_measure_dataset = __import__("main.core.stereology", fromlist=["measure_dataset"]).measure_dataset

            def capture_measurement_inputs(labels, scales):
                captured["labels"] = [label.copy() for label in labels]
                captured["scales"] = [dict(scale) for scale in scales]
                return real_measure_dataset(labels, scales)

            with patch("main.core.stereology.measure_dataset", side_effect=capture_measurement_inputs), patch.object(
                processing.filedialog, "asksaveasfilename", return_value=str(csv_path)
            ), patch.object(processing.messagebox, "askyesno", return_value=True), patch.object(
                processing.filedialog, "askdirectory", return_value=str(preview_path)
            ), patch.object(processing.messagebox, "showinfo"), patch(
                "main.core.stereology.colorize_labels", wraps=real_colorize_labels
            ) as colorize:
                self._checkpoint("before-real-export-invoke")
                processing_window.exportMeasurementsButton.invoke()
                self._checkpoint("after-real-export-invoke")
            self.assertTrue(csv_path.is_file())
            self.assertEqual(len(captured["labels"]), 2)
            np.testing.assert_array_equal(captured["labels"][0], accepted_a.astype(np.int32) // 255)
            self.assertAlmostEqual(captured["scales"][0]["unitsPerPx"], 0.002)
            self.assertEqual({scale["unitName"] for scale in captured["scales"]}, {"mm"})
            self.assertEqual(set(self.app.lastMeasurementExportImageIds), {ids[0], ids[1]})
            self.assertTrue((preview_path / f"{ids[0]}_accepted_labels.png").is_file())
            self.assertTrue((preview_path / f"{ids[1]}_accepted_labels.png").is_file())
            self.assertEqual(colorize.call_count, 2)
            np.testing.assert_array_equal(
                colorize.call_args_list[0].args[0],
                self.app.project.images[0].accepted_labels,
            )
            np.testing.assert_array_equal(
                colorize.call_args_list[1].args[0],
                self.app.project.images[1].accepted_labels,
            )
            self._checkpoint("accepted-measurement-export")

            with patch("main.gui.preprocessing.messagebox.showwarning") as warning, patch.object(
                preprocessing.filedialog, "asksaveasfilename"
            ) as save_dialog:
                set_calibration(self.app.project.images[0], None, None)
                processing_window.exportMeasurementsButton.invoke()
                warning.assert_called_once()
                self.assertIn("same.tif", warning.call_args.args[1])
                save_dialog.assert_not_called()
                set_calibration(self.app.project.images[0], 2.0, "um")
            self._checkpoint("calibration-blocked-export")

            processing_window._onClose()
            analysis_window._on_close()
            self.root.update()
            ids_before = [image.image_id for image in self.app.project.images]
            source_before = [image.source_image.copy() for image in self.app.project.images]
            accepted_before = [image.accepted_mask.copy() for image in self.app.project.images]
            with patch.object(preprocessing.messagebox, "askyesnocancel", return_value=False):
                self.app._onCloseRequested()

            reopened_root = tk.Tk()
            reopened_root.report_callback_exception = self._report_callback_exception
            reopened_root.geometry("1280x800+0+0")
            reopened_app = preprocessing.PreprocessApp(reopened_root)
            reopened_root.update()
            with patch.object(preprocessing.filedialog, "askdirectory", return_value=str(project_path)):
                reopened_app.openProjectButton.invoke()
            reopened_root.update()
            self._checkpoint("reopen-project")
            self.assertEqual([image.image_id for image in reopened_app.project.images], ids_before)
            self.assertEqual(reopened_app.project.images[0].source_fidelity, "preserved")
            self.assertTrue(reopened_app.project.images[0].analysis_included)
            self.assertTrue(reopened_app.project.images[1].analysis_included)
            self.assertFalse(reopened_app.project.images[2].analysis_included)
            self.assertAlmostEqual(preprocessing.calibration_mm_per_px(reopened_app.project.images[0]), 0.002)
            self.assertAlmostEqual(preprocessing.calibration_mm_per_px(reopened_app.project.images[1]), 0.003)
            for index, image in enumerate(reopened_app.project.images):
                np.testing.assert_array_equal(image.source_image, source_before[index])
                if accepted_before[index] is not None:
                    np.testing.assert_array_equal(image.accepted_mask, accepted_before[index])
            reopened_app._onCloseRequested()

        self.assertEqual(self.callback_errors, [])

    def test_close_cancels_delayed_batch_without_late_project_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "delayed.png"
            image = np.zeros((48, 48), dtype=np.uint8)
            cv2.circle(image, (24, 24), 10, 200, -1)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            with patch.object(preprocessing.filedialog, "askopenfilenames", return_value=(str(image_path),)):
                self.app.addImagesButton.invoke()
            self.root.update()

            callback_calls = []
            self.app._acceptBatchDraftResult = lambda result: callback_calls.append(result) or True
            self.app.batchSelection = {0}
            self.app.processButton.invoke()
            self.root.update()
            window = next(
                item for item in self._top_levels(self.root)
                if isinstance(item, processing.ProcessingWindow)
            )
            started = threading.Event()
            release = threading.Event()
            real_process = batch_controller.process_batch_parallel

            def delayed(images, **kwargs):
                started.set()
                release.wait(5.0)
                return real_process(images, **kwargs)

            try:
                with patch("main.gui.batch_controller.process_batch_parallel", side_effect=delayed):
                    window.applySelectedButton.invoke()
                    self.assertTrue(started.wait(2.0))
                    window._onClose()
                    release.set()
                    deadline = time.monotonic() + 5.0
                    while window._batchCoordinator._threads and time.monotonic() < deadline:
                        self.root.update()
                        threading.Event().wait(0.005)
                    self.assertFalse(window._batchCoordinator._threads)
            finally:
                release.set()

            self.assertIsNone(self.app.project.images[0].draft_mask)
            self.assertEqual(callback_calls, [])
            self.root.update()
            self.assertEqual(self.callback_errors, [])
            with patch.object(preprocessing.messagebox, "askyesnocancel", return_value=False):
                self.app._onCloseRequested()

    def test_standalone_processing_export_is_explicitly_labeled_draft(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = np.zeros((16, 16), dtype=np.uint8)
            labels = np.zeros((16, 16), dtype=np.int32)
            labels[4:12, 4:12] = 1
            window = processing.ProcessingWindow(
                parent=self.root,
                images=[image],
                paths=["standalone.png"],
                labels=[labels],
                binaries=[(labels > 0).astype(np.uint8) * 255],
            )
            self.root.update()
            self.assertIn("draft", window.exportMeasurementsButton.cget("text").lower())
            output = Path(temporary) / "draft.csv"
            with patch.object(processing.filedialog, "asksaveasfilename", return_value=str(output)), patch.object(
                processing.messagebox, "askyesno", return_value=False
            ), patch.object(processing.messagebox, "showinfo"):
                window.exportMeasurementsButton.invoke()
            self.assertTrue(output.is_file())
            window._onClose()

    def test_draft_save_buttons_round_trip_and_report_failures(self):
        project = create_project("Draft Save")
        source = np.zeros((12, 12), dtype=np.uint8)
        records = [
            add_image(project, source, f"source-{index}.png", "duplicate.png")
            for index in range(3)
        ]
        accepted = np.zeros((12, 12), dtype=np.uint8)
        accepted[2:7, 2:7] = 255
        for record in records[:2]:
            update_draft(record, mask=accepted, labels=None, settings={})
            accept_draft(record)
            set_calibration(record, 0.01, "mm")
            set_analysis_included(record, True)
        project_snapshot = create_analysis_snapshot(project)
        accepted_before = [record.accepted_mask.copy() if record.accepted_mask is not None else None for record in records]
        inclusion_before = [record.analysis_included for record in records]
        analysis_revision_before = project.analysis_revision

        draft_masks = [np.zeros((12, 12), np.uint8), accepted.copy(), None]
        draft_labels = [np.zeros((12, 12), np.int32), np.where(accepted > 0, 7, 0).astype(np.int32), None]
        window = processing.ProcessingWindow(
            parent=self.root,
            images=[source.copy() for _ in records],
            paths=[record.display_name for record in records],
            binaries=draft_masks,
            labels=draft_labels,
            imageIds=[record.image_id for record in records],
        )
        self.root.update()

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            with patch.object(processing.filedialog, "askdirectory", return_value=str(output_dir)), patch.object(
                processing.messagebox, "showinfo"
            ) as show_info, patch.object(processing.messagebox, "showwarning") as show_warning:
                window.saveDraftMasksButton.invoke()
                show_info.assert_called_once()
                show_warning.assert_not_called()
            mask_paths = [output_dir / f"{records[index].image_id}_draft_mask.png" for index in (0, 1)]
            self.assertTrue(all(path.is_file() for path in mask_paths))
            for index, path in enumerate(mask_paths):
                saved_mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                np.testing.assert_array_equal(saved_mask, draft_masks[index])
            self.assertFalse((output_dir / f"{records[2].image_id}_draft_mask.png").exists())

            with patch.object(processing.filedialog, "askdirectory", return_value=str(output_dir)), patch.object(
                processing.messagebox, "showinfo"
            ) as show_info, patch.object(processing.messagebox, "showwarning") as show_warning:
                window.saveDraftLabelsButton.invoke()
                show_info.assert_called_once()
                show_warning.assert_not_called()
            for index in (0, 1):
                raw_path = output_dir / f"{records[index].image_id}_draft_labels.npy"
                preview_path = output_dir / f"{records[index].image_id}_draft_labels_color.png"
                self.assertTrue(raw_path.is_file())
                self.assertTrue(preview_path.is_file())
                np.testing.assert_array_equal(np.load(raw_path, allow_pickle=False), draft_labels[index])
            self.assertEqual(len(list(output_dir.glob("*_draft_mask.png"))), 2)

            contents_before_cancel = set(output_dir.iterdir())
            with patch.object(processing.filedialog, "askdirectory", return_value="") as picker:
                window.saveDraftMasksButton.invoke()
                picker.assert_called_once()
            self.assertEqual(set(output_dir.iterdir()), contents_before_cancel)

            window.binaries = [None, None, None]
            with patch.object(processing.filedialog, "askdirectory") as picker, patch.object(
                processing.messagebox, "showwarning"
            ) as warning:
                window.saveDraftMasksButton.invoke()
                warning.assert_called_once()
                picker.assert_not_called()
            window.binaries = [draft_masks[0], None, None]

            window.labels = [None, None, None]
            with patch.object(processing.filedialog, "askdirectory") as picker, patch.object(
                processing.messagebox, "showwarning"
            ) as warning:
                window.saveDraftLabelsButton.invoke()
                warning.assert_called_once()
                picker.assert_not_called()
            window.labels = [draft_labels[0], None, None]
            contents_before_label_cancel = set(output_dir.iterdir())
            with patch.object(processing.filedialog, "askdirectory", return_value="") as picker:
                window.saveDraftLabelsButton.invoke()
                picker.assert_called_once()
            self.assertEqual(set(output_dir.iterdir()), contents_before_label_cancel)

            with patch.object(processing.filedialog, "askdirectory", return_value=str(output_dir)), patch.object(
                processing.cv2, "imwrite", return_value=False
            ), patch.object(processing.messagebox, "showwarning") as warning:
                window.saveDraftMasksButton.invoke()
                warning.assert_called_once()
                self.assertIn("Saved 0 of 1", warning.call_args.args[1])

            partial_dir = output_dir / "partial"
            partial_dir.mkdir()
            with patch.object(processing.filedialog, "askdirectory", return_value=str(partial_dir)), patch.object(
                processing.np, "save", side_effect=OSError("synthetic disk failure")
            ), patch.object(processing.messagebox, "showwarning") as warning:
                window.saveDraftLabelsButton.invoke()
                warning.assert_called_once()
                self.assertIn("Partial pairs: 1", warning.call_args.args[1])
                self.assertIn("raw label map failed", warning.call_args.args[1])
            self.assertTrue((partial_dir / f"{records[0].image_id}_draft_labels_color.png").is_file())
            self.assertFalse((partial_dir / f"{records[0].image_id}_draft_labels.npy").exists())

        self.assertEqual([record.analysis_included for record in records], inclusion_before)
        self.assertEqual(project.analysis_revision, analysis_revision_before)
        for record, prior in zip(records, accepted_before):
            if prior is not None:
                np.testing.assert_array_equal(record.accepted_mask, prior)
            else:
                self.assertIsNone(record.accepted_mask)
        after = create_analysis_snapshot(project)
        self.assertEqual(
            [image.accepted_checksum_sha256 for image in after.images],
            [image.accepted_checksum_sha256 for image in project_snapshot.images],
        )
        window._onClose()


if __name__ == "__main__":
    unittest.main()
