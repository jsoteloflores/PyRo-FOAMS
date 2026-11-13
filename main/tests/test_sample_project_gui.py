import contextlib
import ctypes
import io
import json
import os
import platform
import subprocess
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


def review_client_sizes(info, mode):
    targets = ((1280, 800), (1400, 900))
    if mode == "exact":
        return targets
    if mode != "available":
        raise ValueError("GUI display mode must be exact or available")
    left, top, right, bottom = info["work_area"]
    outer = info["decorated_outer_bounds"]
    client_width, client_height = info["client_size"]
    border_width = outer[2] - outer[0] - client_width if outer is not None else 0
    border_height = outer[3] - outer[1] - client_height if outer is not None else 0
    available_width = min(right - left - border_width, info["wm_max_size"][0])
    available_height = min(bottom - top - border_height, info["wm_max_size"][1])
    minimum_width = max(980, info["wm_min_size"][0])
    minimum_height = max(640, info["wm_min_size"][1])
    if available_width < minimum_width or available_height < minimum_height:
        raise ValueError(f"Available client area {available_width}x{available_height} is below the supported minimum {minimum_width}x{minimum_height}")
    return tuple(dict.fromkeys((min(width, available_width), min(height, available_height)) for width, height in targets))


class TestGuiDisplayPolicy(unittest.TestCase):
    def test_available_mode_accounts_for_decorations_and_wm_limits(self):
        info = {"work_area": [0, 0, 1024, 728], "decorated_outer_bounds": [0, 0, 1016, 739],
                "client_size": [1000, 700], "wm_min_size": [980, 640], "wm_max_size": [1028, 749]}
        self.assertEqual(review_client_sizes(info, "available"), ((1008, 689),))
        self.assertEqual(review_client_sizes(info, "exact"), ((1280, 800), (1400, 900)))

    def test_available_mode_rejects_unsupported_display(self):
        info = {"work_area": [0, 0, 800, 600], "decorated_outer_bounds": None,
                "client_size": [800, 600], "wm_min_size": [980, 640], "wm_max_size": [800, 600]}
        with self.assertRaisesRegex(ValueError, "below the supported minimum"):
            review_client_sizes(info, "available")


class TestLiveSampleProjectGui(unittest.TestCase):
    def setUp(self):
        if not LIVE_TK:
            self.skipTest("Set PYRO_FOAMS_LIVE_TK=1 on a display-capable host to run live Tk integration.")
        self.callback_errors = []
        self._setup_ready = False
        self.tk_stderr = io.StringIO()
        self.stderr_redirect = contextlib.redirect_stderr(self.tk_stderr)
        self.stderr_redirect.__enter__()
        self.addCleanup(self._cleanup_setup_failure)
        self.root = tk.Tk()
        self.root.report_callback_exception = self._report_callback_exception
        self.display_mode = os.environ.get("PYRO_FOAMS_GUI_DISPLAY_MODE", "exact")
        self.assertIn(self.display_mode, ("exact", "available"))
        self.geometry_records = []
        self.root.geometry("1280x800+0+0")
        self.root.deiconify()
        self.app = preprocessing.PreprocessApp(self.root)
        self.root.update()
        self.window_manager_probe = None
        self._settle_geometry(self.root)
        self._record_geometry(self.root, "environment", requested_size=[1280, 800], settled=True)
        if os.environ.get("PYRO_FOAMS_GUI_REQUIRE_WM") == "1":
            def manager_ready():
                probe = subprocess.run(["xprop", "-root", "_NET_SUPPORTING_WM_CHECK"],
                                       capture_output=True, text=True, timeout=1)
                self.window_manager_probe = probe.stdout.strip()
                return probe.returncode == 0 and "window id # 0x" in probe.stdout
            try:
                self._wait_until(manager_ready, self.root)
            finally:
                self._record_geometry(self.root, "window-manager-readiness")
        width, height = self._review_sizes(self.root)[0]
        self._request_geometry(self.root, width, height)
        self._setup_ready = True

    def _cleanup_setup_failure(self):
        if self._setup_ready:
            return
        if hasattr(self, "root"):
            try:
                preprocessing.cancel_debounced_callbacks(self.root)
                self.root.destroy()
            except tk.TclError:
                pass
        self.stderr_redirect.__exit__(None, None, None)

    def tearDown(self):
        try:
            if hasattr(self, "root") and self.root.tk.call("info", "commands", str(self.root)):
                for child in tuple(self.root.winfo_children()):
                    if isinstance(child, tk.Toplevel):
                        close = getattr(child, "_on_close", None)
                        if callable(close):
                            close()
                        else:
                            child.destroy()
                preprocessing.cancel_debounced_callbacks(self.root)
                self.root.destroy()
        finally:
            self.stderr_redirect.__exit__(None, None, None)
        self.assertNotRegex(
            self.tk_stderr.getvalue(),
            r"invalid command name|Exception in Tkinter callback|Traceback \(most recent call last\)",
        )

    def _report_callback_exception(self, exception_type, exception, traceback):
        self.callback_errors.append((exception_type, exception, traceback))

    def _geometry_info(self, window):
        work_area = [0, 0, window.winfo_screenwidth(), window.winfo_screenheight()]
        outer = None
        if sys.platform == "win32":
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            rect = wintypes.RECT()
            if user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0):
                work_area = [rect.left, rect.top, rect.right, rect.bottom]
            handle = int(window.wm_frame(), 16)
            user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
            if user32.GetWindowRect(handle, ctypes.byref(rect)):
                outer = [rect.left, rect.top, rect.right, rect.bottom]
        left, top = window.winfo_rootx(), window.winfo_rooty()
        return {"platform": platform.platform(), "python": sys.version, "mode": self.display_mode,
            "windowing_system": str(window.tk.call("tk", "windowingsystem")),
            "display": os.environ.get("DISPLAY"), "window_manager_probe": self.window_manager_probe,
            "focused_widget": str(window.focus_get()) if window.focus_get() is not None else None,
                "screen": [window.winfo_screenwidth(), window.winfo_screenheight()],
                "virtual_root": [window.winfo_vrootx(), window.winfo_vrooty(), window.winfo_vrootwidth(), window.winfo_vrootheight()],
                "tk_scaling": float(window.tk.call("tk", "scaling")),
                "tk_patchlevel": str(window.tk.call("info", "patchlevel")),
                "wm_min_size": list(window.wm_minsize()), "wm_max_size": list(window.wm_maxsize()),
                "client_size": [window.winfo_width(), window.winfo_height()],
                "client_bounds": [left, top, left + window.winfo_width(), top + window.winfo_height()],
                "decorated_outer_bounds": outer, "work_area": work_area}

    def _record_geometry(self, window, state, **fields):
        record = self._geometry_info(window)
        record.update(state=state, **fields)
        self.geometry_records.append(record)
        artifacts = Path(os.environ.get("PYRO_FOAMS_GUI_ARTIFACTS", "main/tests/gui_smoke_artifacts"))
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / f"display-{self._testMethodName}-{self.display_mode}.json").write_text(
            json.dumps(self.geometry_records, indent=2), encoding="utf-8")
        print("GUI_DISPLAY: " + json.dumps(record), flush=True)
        return record

    def _review_sizes(self, window):
        self._settle_geometry(window)
        info = self._geometry_info(window)
        try:
            sizes = review_client_sizes(info, self.display_mode)
        except ValueError:
            self._record_geometry(window, "unsupported-available-display")
            raise
        self._record_geometry(window, "declared-matrix", tested_sizes=sizes)
        return sizes

    def _request_geometry(self, window, width, height):
        work = self._geometry_info(window)["work_area"]
        window.geometry(f"{width}x{height}+{work[0]}+{work[1]}")
        window.deiconify()
        self._settle_geometry(window)
        info = self._record_geometry(window, "geometry-request", requested_size=[width, height])
        if info["client_size"] != [width, height]:
            outer = info["decorated_outer_bounds"]
            border_width = outer[2] - outer[0] - info["client_size"][0] if outer else 0
            border_height = outer[3] - outer[1] - info["client_size"][1] if outer else 0
            work = info["work_area"]
            self._record_geometry(window, "geometry-mismatch", requested_size=[width, height],
                exceeds_wm_max=width > info["wm_max_size"][0] or height > info["wm_max_size"][1],
                exceeds_work_area=width + border_width > work[2] - work[0] or height + border_height > work[3] - work[1])
            self._capture_owned_window(window, f"geometry-mismatch-{width}x{height}.png", check_bounds=False)
        self.assertEqual(info["client_size"], [width, height], json.dumps(info))

    def _capture_owned_window(self, window, name, directory=None, check_bounds=True):
        window.deiconify()
        window.lift()
        window.attributes("-topmost", True)
        self._settle_geometry(window)
        info = self._geometry_info(window)
        if self.display_mode == "available" and check_bounds:
            width, height = review_client_sizes(info, self.display_mode)[-1]
            if window.winfo_width() > width or window.winfo_height() > height:
                self._request_geometry(window, width, height)
                info = self._geometry_info(window)
        bounds = info["decorated_outer_bounds"] or info["client_bounds"]
        directory = directory or Path(os.environ.get("PYRO_FOAMS_GUI_ARTIFACTS", "main/tests/gui_smoke_artifacts"))
        directory.mkdir(parents=True, exist_ok=True)
        if self.display_mode == "available":
            actual = f"{window.winfo_width()}x{window.winfo_height()}"
            name = "available-" + name.replace("1280x800", actual).replace("1400x900", actual)
        try:
            image = ImageGrab.grab(bbox=tuple(bounds))
            image.save(directory / name)
            work = info["work_area"]
            complete = bounds[0] >= work[0] and bounds[1] >= work[1] and bounds[2] <= work[2] and bounds[3] <= work[3]
            self._record_geometry(window, "screenshot", screenshot=name, screenshot_bounds=bounds,
                                  screenshot_size=list(image.size), complete_owned_window=complete,
                                  screenshot_scope="decorated" if info["decorated_outer_bounds"] else "client")
            self.assertEqual(image.size, (bounds[2] - bounds[0], bounds[3] - bounds[1]))
            if check_bounds:
                self.assertGreaterEqual(bounds[0], work[0], json.dumps(info))
                self.assertGreaterEqual(bounds[1], work[1], json.dumps(info))
                self.assertLessEqual(bounds[2], work[2], json.dumps(info))
                self.assertLessEqual(bounds[3], work[3], json.dumps(info))
        finally:
            window.attributes("-topmost", False)

    def _settle_geometry(self, window):
        deadline = time.monotonic() + 3.0
        last = None
        stable_since = None
        while time.monotonic() < deadline:
            window.update()
            self.assertFalse(self.callback_errors, repr(self.callback_errors))
            geometry = (window.winfo_width(), window.winfo_height(), window.winfo_rootx(), window.winfo_rooty())
            now = time.monotonic()
            if geometry != last:
                last, stable_since = geometry, now
            elif now - stable_since >= 0.1 and window.winfo_ismapped():
                return
            tick = []
            timer = window.after(10, lambda: tick.append(True))
            try:
                while not tick and time.monotonic() < deadline:
                    window.update()
            finally:
                if not tick:
                    window.after_cancel(timer)
        self._record_geometry(window, "settling-timeout")
        self.fail("Window geometry did not settle within the bounded event-loop deadline")

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
            self._capture_owned_window(window, name, artifacts)

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
                    for width, height in self._review_sizes(window):
                        self._request_geometry(window, width, height)
                        self.root.update()
                        window.resultsNotebook.select(window.plotCanvas.get_tk_widget().master)
                        self.root.update()
                        capture(f"analysis-plot-{width}x{height}-font{scale}x.png")
                        self._assert_analysis_results_layout(window, width, height, scale, "existing-compare-success")
                        self.assertGreater(window.plotCanvas.get_tk_widget().winfo_height(), 100)
                        window.plotCanvas.draw()
                        pixels = np.asarray(window.plotCanvas.buffer_rgba())
                        self.assertGreater(np.count_nonzero(np.any(pixels[:, :, :3] < 200, axis=2)), 100)
                        self.assertTrue(window.runButton.winfo_ismapped())
                        self.assertLessEqual(window.statusVar.get().count("\n"), 1)
            finally:
                for name, size in fonts.items():
                    tkfont.nametofont(name).configure(size=size)
                window.resultsNotebook.select(window.distributionTable.master)
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
        self._capture_owned_window(self.root, name, artifact_dir)

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
                window.cancelButton.focus_force()
                self.root.update()
                self.assertIs(self.root.focus_get(), window.cancelButton)
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

    def test_schema1_null_method_setup_restores_runs_and_saves(self):
        from main.core.legacy_perimeter import FOAMS105_PERIMETER_PROFILE_CURRENT
        from main.core.sample_analysis import LegacyRunConfig, prepare_legacy_analysis
        from main.core.sample_project_io import open_project, save_project
        from main.tests.test_sample_analysis import TestSampleAnalysisAdapters

        for mode in ("Original FOAMS 1.0.5", "Modern analysis"):
            with self.subTest(mode=mode):
                project, _ = TestSampleAnalysisAdapters()._snapshot_with_three_images()
                set_analysis_included(project.images[2], False)
                snapshot = create_analysis_snapshot(project)
                labels = prepare_legacy_analysis(snapshot, LegacyRunConfig(minimum_diameter_px=3.0)).prepared.label_preparation_result.bin_labels_mm
                legacy = {"minimum_diameter_px": 3.0, "perimeter_profile": FOAMS105_PERIMETER_PROFILE_CURRENT,
                    "selection_mode": "manual", "manual_ranges": [["group-1", labels[0], labels[-1]]],
                    "size_lower_cutoff_mm": 0.3, "size_upper_cutoff_mm": 0.8}
                modern = {"min_detectable_mm": 0.001, "max_reliable_mm": None,
                          "bin_edges_mm": [0.001, 0.01, 0.1]}
                project.analysis_setup = {"schema": 1, "mode": mode,
                    "legacy": legacy if mode.startswith("Original") else None,
                    "modern": modern if mode == "Modern analysis" else None}
                window = sample_analysis.SampleAnalysisWorkspace(self.root, project, snapshot)
                self.assertEqual(window.modeVar.get(), mode)
                self.assertIsNone(window._displayed_request)
                self.assertFalse(window._successful_results)
                self.assertIsNone(project.analysis_setup["modern" if mode.startswith("Original") else "legacy"])
                window.saveSetupButton.invoke()
                self.assertEqual(window.statusVar.get(), "Setup saved; no calculation performed")
                self.assertEqual(project.analysis_setup["schema"], 2)
                self.assertIsInstance(project.analysis_setup["legacy"], dict)
                self.assertIsInstance(project.analysis_setup["modern"], dict)
                if mode.startswith("Original"):
                    self.assertEqual(window.minimumDiameterVar.get(), "3.0")
                    self.assertEqual(window.sizeLowerCutoffVar.get(), "0.3")
                    self.assertEqual(window._parse_modern_config().min_detectable_mm, None)
                else:
                    self.assertEqual(window.modernMinDetectableVar.get(), "0.001")
                    self.assertEqual(window.modernMaxReliableVar.get(), "")
                    self.assertEqual(window.minimumDiameterVar.get(), "2.0")
                window.runButton.invoke()
                self._wait_until(lambda: not window._busy, self.root)
                self.assertEqual(window.statusVar.get(), "Original run complete" if mode.startswith("Original") else "Modern run complete",
                                 window.resultsText.get("1.0", "end"))
                window.modeVar.set("Modern analysis" if mode.startswith("Original") else "Original FOAMS 1.0.5")
                for variable in (window.minimumDiameterVar, window.sizeLowerCutoffVar, window.sizeUpperCutoffVar,
                                 window.modernMinDetectableVar, window.modernMaxReliableVar):
                    self.assertNotEqual(variable.get(), "None")
                window.modeVar.set(mode)
                window.saveSetupButton.invoke()
                window._on_close()
                with tempfile.TemporaryDirectory() as folder:
                    save_project(project, Path(folder))
                    restored = open_project(Path(folder))
                self.assertEqual(restored.analysis_setup["schema"], 2)
                if mode == "Modern analysis":
                    self.assertIsNone(restored.analysis_setup["modern"]["max_reliable_mm"])
                reopened = sample_analysis.SampleAnalysisWorkspace(self.root, restored, create_analysis_snapshot(restored))
                self.assertEqual(reopened.modeVar.get(), mode)
                self.assertIsNone(reopened._displayed_request)
                self.assertFalse(reopened._successful_results)
                reopened._on_close()

    def test_comparison_current_and_previous_provenance_are_separate(self):
        from main.core.sample_analysis import prepare_legacy_analysis, prepare_modern_analysis
        from main.tests.test_sample_analysis import TestSampleAnalysisAdapters

        project, snapshot = TestSampleAnalysisAdapters()._snapshot_with_three_images()
        window = sample_analysis.SampleAnalysisWorkspace(self.root, project, snapshot)
        labels = prepare_legacy_analysis(snapshot).prepared.label_preparation_result.bin_labels_mm
        window.selectionModeVar.set("manual")
        window.manualRangesVar.set(f"group-1:{labels[0]!r}:{labels[-1]!r}")

        def run():
            window.runButton.invoke()
            self._wait_until(lambda: not window._busy, self.root)
            self.assertNotIn("Rendering failed", window.statusVar.get())

        def table():
            return [window.distributionTable.item(row_id, "values") for row_id in window.distributionTable.get_children()]

        def capture(name):
            self._capture_owned_window(window, name)

        run()
        self.assertEqual(window.statusVar.get(), "Original run complete", window.resultsText.get("1.0", "end"))
        original_a = window._displayed_envelopes[0]
        window.modeVar.set("Modern analysis")
        window.modernMinDetectableVar.set("0.001")
        window.modernSegmentsVar.set("group-2:0:4;group-1:4:8")
        run()
        self.assertEqual(window.statusVar.get(), "Modern run complete")
        modern_a = window._displayed_envelopes[0]
        table_a = table()
        curves_a = [(line.get_label(), line.get_xdata().copy(), line.get_ydata().copy()) for line in window.plotAxes.lines]
        label_a = next(label for label, envelope in window._successful_results.items() if envelope is modern_a)
        self.assertNotEqual(original_a.request.request_id, modern_a.request.request_id)
        set_calibration(project.images[0], 0.004, "mm")
        set_calibration(project.images[1], 0.004, "mm")
        window.modeVar.set("Compare methods")
        for legacy_fails, modern_fails in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(legacy_fails=legacy_fails, modern_fails=modern_fails):
                window.minimumDiameterVar.set("-1" if legacy_fails else "2")
                window.modernMinDetectableVar.set("" if modern_fails else "0.001")
                if not legacy_fails:
                    labels = prepare_legacy_analysis(create_analysis_snapshot(project)).prepared.label_preparation_result.bin_labels_mm
                    window.manualRangesVar.set(f"group-1:{labels[0]!r}:{labels[-1]!r}")
                run()
                request_b = window._displayed_request
                self.assertNotEqual(request_b.request_id, modern_a.request.request_id)
                self.assertEqual(window.resultViewVar.get(), "Current request")
                self.assertEqual(len(window._displayed_envelopes), int(not legacy_fails) + int(not modern_fails))
                self.assertTrue(all(envelope.request is request_b for envelope in window._displayed_envelopes))
                self.assertEqual(request_b.snapshot.images[0].mm_per_px, 0.004)
                for fails, method, label in ((legacy_fails, "legacy", "Original"), (modern_fails, "modern", "Modern")):
                    self.assertIn(f"{method}: {'failed' if fails else 'success'}", window.resultProvenanceVar.get())
                    method_rows = [row for row in table() if row[0].startswith(label)]
                    legends = [line.get_label() for line in window.plotAxes.lines if line.get_label().startswith(label)]
                    if fails:
                        self.assertFalse(method_rows)
                        self.assertFalse(legends)
                    else:
                        self.assertTrue(method_rows)
                        self.assertTrue(legends)
                if modern_fails and not legacy_fails:
                    capture("analysis-21b-current-partial-failure.png")
                window.resultViewVar.set(label_a)
                window.resultViewSelector.event_generate("<<ComboboxSelected>>")
                self.root.update()
                self.assertIs(window._displayed_request, modern_a.request)
                self.assertEqual(window._displayed_envelopes, (modern_a,))
                self.assertEqual(table(), table_a)
                self.assertEqual(len(window.plotAxes.lines), len(curves_a))
                for line, (label, diameters, values) in zip(window.plotAxes.lines, curves_a):
                    self.assertEqual(line.get_label(), label)
                    np.testing.assert_array_equal(line.get_xdata(), diameters)
                    np.testing.assert_array_equal(line.get_ydata(), values)
                self.assertIn("Previous successful result", window.resultProvenanceVar.get())
                self.assertIn("Inputs changed", window.resultProvenanceVar.get())
                self.assertIn(modern_a.request.request_id, window.resultsText.get("1.0", "end"))
                self.assertEqual(modern_a.request.modern_config.min_detectable_mm, 0.001)
                self.assertEqual(modern_a.request.snapshot.images[0].mm_per_px, 0.003)
                if modern_fails and not legacy_fails:
                    capture("analysis-21b-previous-success.png")
                window.resultViewVar.set("Current request")
                window.resultViewSelector.event_generate("<<ComboboxSelected>>")
                self.root.update()
                self.assertIs(window._displayed_request, request_b)
        window.resultViewVar.set(label_a)
        window.resultViewSelector.event_generate("<<ComboboxSelected>>")
        self.root.update()
        provenance_a = window.resultProvenanceVar.get()

        def failed_render(_result):
            window._clear_distribution()
            raise RuntimeError("injected partial render failure")

        with patch.object(window, "_render_compare", side_effect=failed_render):
            window.runButton.invoke()
            self._wait_until(lambda: not window._busy, self.root)
        self.assertIs(window._displayed_request, modern_a.request)
        self.assertEqual(window.resultViewVar.get(), label_a)
        self.assertEqual(window.resultProvenanceVar.get(), provenance_a)
        self.assertEqual(table(), table_a)
        entered, release = threading.Event(), threading.Event()

        def delayed(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test worker timeout")
            return prepare_modern_analysis(*args, **kwargs)

        try:
            with patch("main.gui.sample_analysis.prepare_modern_analysis", side_effect=delayed):
                window.runButton.invoke()
                self._wait_until(entered.is_set, self.root)
                set_calibration(project.images[0], 0.006, "mm")
                release.set()
                self._wait_until(lambda: not window._busy, self.root)
                self.assertIn("Discarded stale", window.statusVar.get())
                self.assertIs(window._displayed_request, modern_a.request)
                self.assertEqual(table(), table_a)
                self.assertEqual(window.resultViewVar.get(), label_a)
        finally:
            release.set()
            window._on_close()

    def _assert_analysis_results_layout(self, window, width, height, font_scale, state, focused=True):
        window.deiconify()
        window.lift()
        window.attributes("-topmost", True)
        self._settle_geometry(window)
        window.plotCanvas.draw()
        plot = window.plotCanvas.get_tk_widget()
        metrics = {"state": state, "requested_size": [width, height],
                   "actual_size": [window.winfo_width(), window.winfo_height()],
                   "font_scale": font_scale, "tk_scaling": float(window.tk.call("tk", "scaling")),
                   "plot_height": plot.winfo_height(), "plot_width": plot.winfo_width(),
                   "setup_visible": window._setup_visible}
        print("ANALYSIS_LAYOUT: " + json.dumps(metrics), flush=True)
        self._record_geometry(window, state, **{key: value for key, value in metrics.items() if key != "state"})
        self._capture_owned_window(window, f"analysis-21c-{state}-{width}x{height}-font{font_scale}x.png")
        diagnostic = json.dumps(metrics)
        self.assertEqual(metrics["actual_size"], metrics["requested_size"], diagnostic)
        self.assertEqual(window._setup_visible, not focused, diagnostic)
        self.assertGreaterEqual(plot.winfo_height(), 240, diagnostic)
        self.assertGreaterEqual(window.plotAxes.get_window_extent(window.plotCanvas.get_renderer()).height, 120, diagnostic)
        for control in (window.runButton, window.cancelButton, window.setupToggleButton,
                        window.resultViewSelector, window.backButton):
            self.assertTrue(control.winfo_ismapped(), diagnostic)
            self.assertGreaterEqual(control.winfo_rooty(), window.winfo_rooty(), diagnostic)
            self.assertLessEqual(control.winfo_rooty() + control.winfo_height(), window.winfo_rooty() + window.winfo_height(), diagnostic)
            self.assertLessEqual(control.winfo_rootx() + control.winfo_width(), window.winfo_rootx() + window.winfo_width(), diagnostic)
        self.assertLessEqual(window.statusLabel.winfo_height() if not window._status_scroll.winfo_ismapped()
                             else window._status_canvas.winfo_height(), tkfont.nametofont("TkDefaultFont").metrics("linespace") * 2 + 4)

    def test_analysis_results_focused_layout_preserves_state(self):
        from main.core.sample_analysis import prepare_legacy_analysis, prepare_modern_analysis
        from main.tests.test_sample_analysis import TestSampleAnalysisAdapters

        project, _ = TestSampleAnalysisAdapters()._snapshot_with_three_images()
        set_analysis_included(project.images[2], False)
        project.sample_name = "Long physical sample name with several acquisition details " * 8
        snapshot = create_analysis_snapshot(project)
        labels = prepare_legacy_analysis(snapshot).prepared.label_preparation_result.bin_labels_mm
        window = sample_analysis.SampleAnalysisWorkspace(self.root, project, snapshot)
        window.manualRangesVar.set(f"group-1:{labels[0]!r}:{labels[-1]!r}")
        window.modernMinDetectableVar.set("0.001")
        window.modernEdgesVar.set("0.001,0.01,0.1")
        font_sizes = {name: tkfont.nametofont(name).cget("size") for name in tkfont.names()}
        previous_label = None
        try:
            with patch("main.gui.sample_analysis.prepare_legacy_analysis", wraps=prepare_legacy_analysis) as legacy_calls, \
                 patch("main.gui.sample_analysis.prepare_modern_analysis", wraps=prepare_modern_analysis) as modern_calls:
                for state in ("original-success", "modern-success", "compare-success", "partial-failure", "both-failed", "previous-inputs-changed"):
                    if state == "previous-inputs-changed":
                        set_calibration(project.images[0], 0.004, "mm")
                        window.resultViewVar.set(previous_label)
                        window.resultViewSelector.event_generate("<<ComboboxSelected>>")
                        self.root.update()
                        self.assertIn("Previous successful result", window.resultProvenanceVar.get())
                        self.assertIn("Inputs changed", window.resultProvenanceVar.get())
                    else:
                        window.modeVar.set("Original FOAMS 1.0.5" if state == "original-success" else
                                           "Modern analysis" if state == "modern-success" else "Compare methods")
                        window.minimumDiameterVar.set("-1" if state in ("partial-failure", "both-failed") else "2")
                        window.modernMinDetectableVar.set("" if state == "both-failed" else "0.001")
                        window.runButton.invoke()
                        self._wait_until(lambda: not window._busy, self.root)
                        if state == "original-success":
                            self.assertEqual(window.statusVar.get(), "Original run complete")
                        elif state == "modern-success":
                            self.assertEqual(window.statusVar.get(), "Modern run complete")
                            envelope = window._displayed_envelopes[0]
                            previous_label = next(label for label, saved in window._successful_results.items() if saved is envelope)
                        else:
                            self.assertEqual(window._latest_compare.legacy_result is None, state in ("partial-failure", "both-failed"))
                            self.assertEqual(window._latest_compare.modern_result is None, state == "both-failed")
                    window.resultsNotebook.select(window.plotCanvas.get_tk_widget().master)
                    self.root.update()
                    self.assertFalse(window._setup_visible)
                    for font_scale in (1, 2):
                        for name, size in font_sizes.items():
                            tkfont.nametofont(name).configure(size=size * font_scale)
                        for width, height in self._review_sizes(window):
                            self._request_geometry(window, width, height)
                            window.statusVar.set("A long operational diagnostic, retained in the scrollable status area. " * 12)
                            self.root.update()
                            self._assert_analysis_results_layout(window, width, height, font_scale, state)
                            request = window._displayed_request
                            envelopes = window._displayed_envelopes
                            provenance = window.resultProvenanceVar.get()
                            signature = window._settings_signature()
                            counts = (legacy_calls.call_count, modern_calls.call_count)
                            window.setupToggleButton.invoke()
                            self.root.update()
                            self.assertTrue(window.setupBody.winfo_ismapped())
                            self._assert_analysis_results_layout(window, width, height, font_scale,
                                                                 state + "-setup-open", focused=False)
                            window.setupToggleButton.invoke()
                            self.root.update()
                            self.assertFalse(window.setupBody.winfo_ismapped())
                            self.assertIs(window._displayed_request, request)
                            self.assertEqual(window._displayed_envelopes, envelopes)
                            self.assertEqual(window.resultProvenanceVar.get(), provenance)
                            self.assertEqual(window._settings_signature(), signature)
                            self.assertEqual((legacy_calls.call_count, modern_calls.call_count), counts)
                            window.runButton.focus_force()
                            self.root.update()
                            reached = set()
                            for _ in range(12):
                                focused = self.root.focus_get()
                                reached.add(focused)
                                focused.event_generate("<Tab>")
                                self.root.update()
                            self.assertTrue({window.runButton, window.setupToggleButton, window.backButton,
                                             window.resultViewSelector}.issubset(reached))
                    for name, size in font_sizes.items():
                        tkfont.nametofont(name).configure(size=size)
        finally:
            for name, size in font_sizes.items():
                tkfont.nametofont(name).configure(size=size)
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
        sequences = ["<Shift-Tab>"]
        if self.root.tk.call("tk", "windowingsystem") == "x11":
            sequences.append("<ISO_Left_Tab>")
        for sequence in sequences:
            for source in (self.app.addImagesButton, self.app.imagesCanvas):
                self._assert_dispatched_preparation_tab(source, self.app.saveProjectButton, sequence)
            self._assert_dispatched_preparation_tab(self.app.setCalibrationButton, self.app.renameImageButton, sequence)
            previous = self.app.importMaskButton
            for scrollbar in (self.app.imagesVerticalScrollbar, self.app.imagesHorizontalScrollbar):
                if scrollbar.winfo_ismapped():
                    self._assert_dispatched_preparation_tab(scrollbar, previous, sequence)
                    previous = scrollbar
            original_state = self.app.renameImageButton.cget("state")
            try:
                self.app.renameImageButton.configure(state="disabled")
                self._assert_dispatched_preparation_tab(self.app.setCalibrationButton, self.app.addImagesButton, sequence)
                self._assert_dispatched_preparation_tab(self.app.addImagesButton, self.app.setCalibrationButton, "<Tab>")
            finally:
                self.app.renameImageButton.configure(state=original_state)
        original_groups = [(group, group.cget("width"), group.cget("height"), group.pack_propagate())
                           for group in self.app.imagesToolbarGroups]
        try:
            width = self.app.imagesCanvas.winfo_width() + 100
            for index, (group, _width, _height, _propagation) in enumerate(original_groups):
                height = 200 if index == 0 else group.winfo_reqheight()
                group.pack_propagate(False)
                group.configure(width=width, height=height)
            self.app._reflowPreparationToolbar()
            self._settle_geometry(self.root)
            self.assertTrue(self.app.imagesVerticalScrollbar.winfo_ismapped())
            self.assertTrue(self.app.imagesHorizontalScrollbar.winfo_ismapped())
            for sequence in sequences:
                self._assert_dispatched_preparation_tab(self.app.imagesVerticalScrollbar, self.app.importMaskButton, sequence)
                self._assert_dispatched_preparation_tab(self.app.imagesHorizontalScrollbar, self.app.imagesVerticalScrollbar, sequence)
        finally:
            for group, width, height, propagation in original_groups:
                group.configure(width=width, height=height)
                group.pack_propagate(propagation)
            self.app._reflowPreparationToolbar()
            self._settle_geometry(self.root)

    def _assert_dispatched_preparation_tab(self, source, expected, sequence="<Shift-Tab>"):
        source.focus_set()
        self.root.update()
        original_tags = source.bindtags()
        probe_tag = f"ReverseTabProbe{id(source)}"
        events = []
        transitions = []
        focus_notifications = []

        def record_key(event):
            delivered_sequence = "<ISO_Left_Tab>" if event.keysym == "ISO_Left_Tab" else sequence
            events.append({"keysym": event.keysym, "keycode": event.keycode, "state": event.state,
                           "widget": str(event.widget), "focus_before": str(self.root.focus_get()),
                           "bindtags": list(original_tags), "delivered_sequence": delivered_sequence,
                           "widget_binding": source.bind(delivered_sequence),
                           "shift_tab_binding": source.bind("<Shift-Tab>"),
                           "native_reverse_binding": source.bind("<ISO_Left_Tab>") if self.root.tk.call("tk", "windowingsystem") == "x11" else None,
                           "class_binding": source.bind_class(source.winfo_class(), delivered_sequence),
                           "all_binding": source.bind_class("all", delivered_sequence)})

        def record_focus(widget, detail):
            focus_notifications.append({"widget": widget, "detail": detail})
            if detail not in ("NotifyVirtual", "NotifyNonlinearVirtual"):
                transitions.append(widget)

        bindings = []
        focus_command = None
        probe_binding = None
        try:
            focus_command = self.root.register(record_focus)
            source.bindtags((probe_tag, *original_tags))
            probe_binding = self.root.bind_class(probe_tag, "<KeyPress>", record_key)
            for control in dict.fromkeys((*self.app.workspaceFocusControls,
                                         self.app.imagesVerticalScrollbar, self.app.imagesHorizontalScrollbar)):
                original_binding = control.bind("<FocusIn>")
                bindings.append((control, original_binding))
                self.root.tk.call("bind", str(control), "<FocusIn>", f"+{focus_command} %W %d")
            source.event_generate(sequence, when="tail")
            self.root.update()
            self._record_geometry(self.root, "forward-tab" if sequence == "<Tab>" else "reverse-tab", generated_sequence=sequence,
                                  source=str(source), expected=str(expected), delivered_events=events,
                                  focus_transitions=transitions, final_focus=str(self.root.focus_get()),
                                  focus_notifications=focus_notifications,
                                  font_size=tkfont.nametofont("TkDefaultFont").cget("size"))
            if self.root.focus_get() is not expected or transitions != [str(expected)]:
                artifacts = Path(os.environ.get("PYRO_FOAMS_GUI_ARTIFACTS", "main/tests/gui_smoke_artifacts"))
                self._capture_workspace(artifacts, "keyboard-navigation-failure.png")
            self.assertEqual(len(events), 1, repr(events))
            self.assertEqual(transitions, [str(expected)], repr(events))
            self.assertIs(self.root.focus_get(), expected, repr(events))
            if expected in self.app._preparationControls:
                canvas = self.app.imagesCanvas
                self.assertGreaterEqual(expected.winfo_rootx(), canvas.winfo_rootx(), repr(events))
                self.assertGreaterEqual(expected.winfo_rooty(), canvas.winfo_rooty(), repr(events))
                self.assertLessEqual(expected.winfo_rootx() + expected.winfo_width(), canvas.winfo_rootx() + canvas.winfo_width(), repr(events))
                self.assertLessEqual(expected.winfo_rooty() + expected.winfo_height(), canvas.winfo_rooty() + canvas.winfo_height(), repr(events))
        finally:
            source.bindtags(original_tags)
            if probe_binding is not None:
                self.root.unbind_class(probe_tag, "<KeyPress>")
                self.root.deletecommand(probe_binding)
            for control, binding in bindings:
                self.root.tk.call("bind", str(control), "<FocusIn>", binding)
            if focus_command is not None:
                self.root.deletecommand(focus_command)

    def test_keyboard_probe_cleanup_allows_fresh_root_fonts(self):
        source = self.app.addImagesButton
        original_tags = source.bindtags()
        original_focus = source.bind("<FocusIn>")
        with patch.object(self.root, "bind_class", side_effect=RuntimeError("injected probe registration failure")):
            with self.assertRaisesRegex(RuntimeError, "injected probe registration failure"):
                self._assert_dispatched_preparation_tab(source, self.app.saveProjectButton)
        self.assertEqual(source.bindtags(), original_tags)
        self.assertEqual(source.bind("<FocusIn>"), original_focus)
        for _ in range(3):
            for source, expected in ((self.app.addImagesButton, self.app.saveProjectButton),
                                     (self.app.imagesCanvas, self.app.saveProjectButton),
                                     (self.app.setCalibrationButton, self.app.renameImageButton)):
                original_tags = source.bindtags()
                original_focus = source.bind("<FocusIn>")
                self._assert_dispatched_preparation_tab(source, expected)
                self.assertEqual(source.bindtags(), original_tags)
                self.assertEqual(source.bind("<FocusIn>"), original_focus)
                self.assertEqual(self.root.bind_class(f"ReverseTabProbe{id(source)}", "<KeyPress>"), "")
        preprocessing.cancel_debounced_callbacks(self.root)
        self.root.destroy()
        self.root = tk.Tk()
        self.root.report_callback_exception = self._report_callback_exception
        self.assertGreater(tkfont.nametofont("TkDefaultFont").metrics("linespace"), 0)
        self.assertGreater(tkfont.nametofont("TkDefaultFont", root=self.root).metrics("linespace"), 0)
        self.root.update()

    def test_saved_runs_reopen_inspect_and_export_without_science(self):
        from main.core.sample_analysis import prepare_legacy_analysis
        from main.core.sample_project_io import open_project, save_project
        from main.tests.test_sample_analysis import TestSampleAnalysisAdapters

        project, _ = TestSampleAnalysisAdapters()._snapshot_with_three_images()
        set_analysis_included(project.images[2], False)
        project.sample_name = 'Sample, "pore"\n\u00e9vidence'
        snapshot = create_analysis_snapshot(project)
        labels = prepare_legacy_analysis(snapshot).prepared.label_preparation_result.bin_labels_mm
        window = sample_analysis.SampleAnalysisWorkspace(self.root, project, snapshot)
        window.manualRangesVar.set(f"group-1:{labels[0]!r}:{labels[-1]!r}")
        window.modernMinDetectableVar.set("0.001")
        window.modernEdgesVar.set("0.001,0.01,0.1")
        for mode in ("Original FOAMS 1.0.5", "Modern analysis", "Compare methods"):
            window.modeVar.set(mode)
            window.runButton.invoke()
            self._wait_until(lambda: not window._busy, self.root)
            self.assertIsNotNone(window._displayed_request)
            self.assertEqual(project.analysis_runs[-1].run_id, window._displayed_request.request_id)
        record_a = project.analysis_runs[0]
        before = tuple(run.payload for run in project.analysis_runs)
        set_calibration(project.images[0], 0.004, "mm")
        set_analysis_included(project.images[1], False)
        project.sample_name = "Changed sample display name"
        window.modeVar.set("Modern analysis")
        window.runButton.invoke()
        self._wait_until(lambda: not window._busy, self.root)
        self.assertEqual(tuple(run.payload for run in project.analysis_runs[:3]), before)
        self.assertEqual(len(project.analysis_runs), 4)
        window._on_close()
        with tempfile.TemporaryDirectory() as folder:
            project_folder = Path(folder) / "project"
            save_project(project, project_folder)
            with contextlib.ExitStack() as mocks:
                for name in ("prepare_legacy_analysis", "prepare_modern_analysis", "analyze_legacy_prepared", "analyze_modern_prepared"):
                    mocks.enter_context(patch("main.gui.sample_analysis." + name, side_effect=AssertionError("history reran science")))
                restored = open_project(project_folder)
                self.assertEqual(tuple(run.payload for run in restored.analysis_runs[:3]), before)
                window = sample_analysis.SampleAnalysisWorkspace(self.root, restored, create_analysis_snapshot(restored))
                self.root.update()
                self.assertIsNone(window._displayed_request)
                label = next(label for label, record in window._saved_labels.items() if record.run_id == record_a.run_id)
                window.resultViewVar.set(label)
                window.resultViewSelector.event_generate("<<ComboboxSelected>>")
                self._wait_until(lambda: window._selected_saved_run is not None, self.root)
                self.assertEqual(window._selected_saved_run.payload, record_a.payload)
                self.assertEqual(window._distribution_rows, record_a.data()["tables"]["distributions"])
                self.assertEqual(window._object_rows, record_a.data()["tables"]["objects"])
                self.assertIn("Inputs changed", window.resultProvenanceVar.get())
                self.assertIn("Previous saved run", window.resultProvenanceVar.get())
                inputs = tuple(image.accepted_snapshot.snapshot_id for image in restored.images if image.analysis_included)
                window.useRunSettingsButton.invoke()
                self.assertEqual(tuple(image.accepted_snapshot.snapshot_id for image in restored.images if image.analysis_included), inputs)
                self.assertEqual(len(restored.analysis_runs), 4)
                for tab in (window._overview_frame, window.objectsTable.master, window.distributionTable.master, window._plot_frame):
                    window.resultsNotebook.select(tab)
                    self.root.update()
                artifacts = Path(os.environ.get("PYRO_FOAMS_GUI_ARTIFACTS", "main/tests/gui_smoke_artifacts"))
                self._capture_owned_window(window, "analysis-22-reopened-stale-run.png", artifacts)
                export_parent = Path(folder) / "exports"
                export_parent.mkdir()
                with patch("main.gui.sample_analysis.filedialog.askdirectory", return_value=str(export_parent)):
                    window.exportRunButton.invoke()
                self._wait_until(lambda: not window._exporting, self.root)
                self.assertIn("Export complete", window.statusVar.get())
                destination = export_parent / ("run-" + record_a.run_id)
                manifest = json.loads((destination / "manifest.json").read_text())
                self.assertEqual(manifest["run_id"], record_a.run_id)
                self.assertEqual((destination / "original_compatible" / "manifest.json").read_bytes(),
                    restored.analysis_assets[next(item["sha256"] for item in record_a.data()["canonical_original"] if item["filename"] == "manifest.json")])
                self._capture_owned_window(window, "analysis-22-export-complete.png", artifacts)
                window._on_close()

    def test_large_saved_object_pages_and_history_cancel_do_not_rerun(self):
        project = create_project("Large native object table")
        image = add_image(project, np.zeros((128, 128), dtype=np.uint8), "large.png")
        labels = np.zeros((128, 128), dtype=np.int32)
        for index in range(900):
            row, column = 2 + (index // 30) * 4, 2 + (index % 30) * 4
            labels[row:row + 2, column:column + 2] = index + 1
        update_draft(image, mask=(labels > 0).astype(np.uint8) * 255, labels=labels, settings={})
        accept_draft(image)
        set_calibration(image, 0.001, "mm")
        set_analysis_included(image, True)
        window = sample_analysis.SampleAnalysisWorkspace(self.root, project, create_analysis_snapshot(project))
        window.modeVar.set("Modern analysis")
        window.modernMinDetectableVar.set("0.0001")
        window.modernEdgesVar.set("0.0001,0.001,0.01")
        responsive = []
        self.root.after(10, lambda: responsive.append(True))
        window.runButton.invoke()
        self._wait_until(lambda: not window._busy, self.root)
        self.assertTrue(responsive)
        self.assertEqual(len(window._object_rows), 900)
        self.assertEqual(len(window.objectsTable.get_children()), 250)
        record = project.analysis_runs[-1]
        label = next(label for label, saved in window._saved_labels.items() if saved is record)
        original_request = window._displayed_request
        with patch("main.gui.sample_analysis.prepare_modern_analysis", side_effect=AssertionError("paging reran science")):
            started = time.monotonic()
            window._page_objects(1)
            self.root.update()
            self.assertLess(time.monotonic() - started, 2)
            self.assertEqual(len(window.objectsTable.get_children()), 250)
            window.resultViewVar.set(label)
            window.resultViewSelector.event_generate("<<ComboboxSelected>>")
            window.cancelButton.invoke()
            self.root.update()
            self.assertIs(window._displayed_request, original_request)
            self.assertIsNone(window._selected_saved_run)
            self.assertEqual(len(project.analysis_runs), 1)
            window.resultViewVar.set(label)
            window.resultViewSelector.event_generate("<<ComboboxSelected>>")
            self._wait_until(lambda: window._selected_saved_run is not None, self.root)
            self.assertEqual(window._selected_saved_run.run_id, record.run_id)
            self.assertEqual(len(window.objectsTable.get_children()), 250)
        window._on_close()

    def test_history_route_without_valid_current_inputs(self):
        from main.core.sample_project_io import open_project, save_project
        from main.tests.test_analysis_runs import TestSavedAnalysisRuns

        for case in ("excluded", "uncalibrated", "unavailable"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                project, projection, _ = TestSavedAnalysisRuns().completed("Modern analysis")
                if case == "uncalibrated":
                    set_calibration(project.images[0], None, None)
                else:
                    for image in project.images:
                        set_analysis_included(image, False)
                root = Path(folder) / "project"
                save_project(project, root)
                if case == "unavailable":
                    (root / projection.record.index_entry()["path"]).write_bytes(b"corrupt history")
                restored = open_project(root)
                self.app.project = restored
                self.app._refreshProjectBackedLists()
                with contextlib.ExitStack() as mocks:
                    mocks.enter_context(patch("main.gui.preprocessing.create_analysis_snapshot", side_effect=AssertionError("history required current inputs")))
                    for function in ("prepare_legacy_analysis", "prepare_modern_analysis", "analyze_legacy_prepared", "analyze_modern_prepared"):
                        mocks.enter_context(patch("main.gui.sample_analysis." + function, side_effect=AssertionError("history ran science")))
                    previous_windows = set(self._top_levels(self.root))
                    self.app.resultsHistoryButton.invoke()
                    self.root.update()
                    window = next(child for child in self._top_levels(self.root)
                                  if child not in previous_windows and isinstance(child, sample_analysis.SampleAnalysisWorkspace))
                    self.assertIsNone(window.snapshot)
                    if case == "unavailable":
                        self.assertIn("unavailable", window.statusVar.get())
                        self.assertIn("Unavailable historical runs", window.resultsText.get("1.0", "end"))
                    else:
                        self._wait_until(lambda: window._selected_saved_run is not None, self.root)
                        self.assertEqual(window._selected_saved_run.payload, projection.record.payload)
                        self.assertIn("Current inputs unavailable", window.resultProvenanceVar.get())
                        window.saveSetupButton.invoke()
                        self.assertIn("Setup saved", window.statusVar.get())
                        window.useRunSettingsButton.invoke()
                        self.assertIsNone(window.snapshot)
                        output = Path(folder) / "exports"
                        output.mkdir()
                        with patch("main.gui.sample_analysis.filedialog.askdirectory", return_value=str(output)):
                            window.exportRunButton.invoke()
                        self._wait_until(lambda: not window._exporting, self.root)
                        manifest = json.loads((output / ("run-" + projection.record.run_id) / "manifest.json").read_text())
                        self.assertEqual(manifest["run_id"], projection.record.run_id)
                        window.runButton.invoke()
                        self.assertIn("Validation failed", window.statusVar.get())
                        self.assertEqual(window._selected_saved_run.run_id, projection.record.run_id)
                        self._capture_owned_window(window, "analysis-22a-history-only-" + case + ".png")
                    window._on_close()

    def test_delayed_history_selection_is_owned_by_view_intent(self):
        from main.core.run_exports import run_tables as real_tables
        from main.tests.test_sample_analysis import TestSampleAnalysisAdapters

        project, _ = TestSampleAnalysisAdapters()._snapshot_with_three_images()
        set_analysis_included(project.images[2], False)
        window = sample_analysis.SampleAnalysisWorkspace(self.root, project, create_analysis_snapshot(project))
        window.modeVar.set("Modern analysis")
        window.modernMinDetectableVar.set("0.001")
        window.modernEdgesVar.set("0.001,0.01,0.1")
        for _ in range(2):
            window.runButton.invoke()
            self._wait_until(lambda: not window._busy, self.root)
        run_a, run_b = project.analysis_runs
        label_a = next(label for label, run in window._saved_labels.items() if run.run_id == run_a.run_id)
        label_b = next(label for label, run in window._saved_labels.items() if run.run_id == run_b.run_id)
        previous_live = next(label for label, envelope in window._successful_results.items() if envelope.request.request_id == run_a.run_id)

        def select(label):
            window.resultViewVar.set(label)
            window._select_result_view()

        def signature():
            return (window.resultViewVar.get(), window._selected_saved_run, window._displayed_request,
                    window.resultProvenanceVar.get(), window.overviewText.get("1.0", "end"),
                    tuple(window.distributionTable.item(row, "values") for row in window.distributionTable.get_children()),
                    tuple(window.objectsTable.item(row, "values") for row in window.objectsTable.get_children()),
                    tuple((line.get_label(), tuple(line.get_xdata()), tuple(line.get_ydata())) for line in window.plotAxes.lines),
                    window._table_run_id)

        expected = signature()
        with patch("main.gui.sample_analysis.run_tables", side_effect=ValueError("injected history load failure")):
            select(label_a)
            self._wait_until(lambda: not window._history_loading, self.root)
        self.assertEqual(signature(), expected)
        self.assertIn("History load failed", window.statusVar.get())
        count = len(project.analysis_runs)
        window.prepareButton.invoke()
        self._wait_until(lambda: not window._busy, self.root)
        self.assertFalse(window._object_rows)
        self.assertFalse(window._distribution_rows)
        self.assertFalse(window.overviewText.get("1.0", "end").strip())
        self.assertIsNone(window._table_run_id)
        self.assertEqual(len(project.analysis_runs), count)
        window.runButton.invoke()
        self._wait_until(lambda: not window._busy, self.root)

        for destination in ("Current request", previous_live, label_b, "calculate", "cancel", "session", "close"):
            with self.subTest(destination=destination):
                select("Current request")
                expected_request = window._displayed_request
                entered, release, completed = threading.Event(), threading.Event(), threading.Event()
                queued = threading.Event()
                put = window._history_queue.put
                def record_put(item):
                    put(item)
                    if item[1].run_id == run_a.run_id:
                        queued.set()
                def delayed(record):
                    if record.run_id == run_a.run_id:
                        entered.set()
                        if not release.wait(5):
                            raise RuntimeError("history barrier timeout")
                    try:
                        return real_tables(record)
                    finally:
                        completed.set()
                try:
                    with patch("main.gui.sample_analysis.run_tables", side_effect=delayed), patch.object(window._history_queue, "put", side_effect=record_put):
                        select(label_a)
                        self._wait_until(entered.is_set, self.root)
                        if destination == "calculate":
                            window.runButton.invoke()
                            self.assertEqual(str(window.cancelButton.cget("state")), "normal")
                        elif destination == "cancel":
                            window.cancelButton.invoke()
                        elif destination == "close":
                            window._on_close()
                        elif destination == "session":
                            original_project_id = project.project_id
                            project.project_id = "replaced-project-session"
                        else:
                            select(destination)
                            if destination == label_b:
                                self._wait_until(lambda: window._selected_saved_run is not None, self.root)
                        if destination not in ("close", "session"):
                            expected = signature()
                        release.set()
                        self._wait_until(queued.is_set, self.root)
                        if destination == "close":
                            self.assertFalse(window.winfo_exists())
                            continue
                        if destination == "session":
                            self._wait_until(lambda: not window._history_loading, self.root)
                            self.assertIs(window._displayed_request, expected_request)
                            project.project_id = original_project_id
                            continue
                        if destination == "calculate":
                            self._wait_until(lambda: not window._busy, self.root)
                            self.assertNotEqual(window._displayed_request.request_id, run_a.run_id)
                            self.assertIsNone(window._selected_saved_run)
                            self.assertEqual(str(window.cancelButton.cget("state")), "disabled")
                        else:
                            self.root.update()
                            window._poll_history()
                            self.assertEqual(signature(), expected)
                finally:
                    release.set()
        self.assertEqual(project.analysis_runs[0].run_id, run_a.run_id)

    def _assert_workspace_controls_fit(self, width, height, artifact_dir, font_scale):
        self._request_geometry(self.root, width, height)
        self._record_geometry(self.root, "preparation-layout", requested_size=[width, height], font_scale=font_scale)
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
            for width, height in self._review_sizes(self.root):
                self._assert_workspace_controls_fit(width, height, artifact_dir, 1)
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
                for width, height in self._review_sizes(self.root):
                    self._assert_workspace_controls_fit(width, height, artifact_dir, 2)
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
            self.assertIsNone(analysis_window._latest_legacy)
            self.assertIn("cannot analyze a group without positive phase-corrected area1 normalization", original_text)
            self.assertIn("stage=binned_selection", original_text)

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
            width, height = self._review_sizes(reopened_root)[0]
            self._request_geometry(reopened_root, width, height)
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
