import os
import sys
import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main.gui.preview_controller import (
    CommittedProcessingConfig,
    PreviewCompletion,
    PreviewOwnership,
    make_processing_key,
)
from main.gui.processing import ProcessingWindow


class FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeStatus(FakeVar):
    pass


class FakeCanvas:
    def __init__(self):
        self.delete_count = 0

    def delete(self, *_args):
        self.delete_count += 1


class FakeWidget:
    def __init__(self):
        self.states = []

    def state(self, states):
        self.states.append(tuple(states))


class FakeWorker:
    def __init__(self):
        self.discard_count = 0

    def discard_pending(self):
        self.discard_count += 1


class TestProcessingEventContract(unittest.TestCase):
    def _commit_window(self, *, auto=True):
        window = object.__new__(ProcessingWindow)
        old_threshold = {"method": "otsu", "medianK": 3}
        window._committedConfig = CommittedProcessingConfig.create(
            old_threshold, {"method": "none"}
        )
        window._ownership = PreviewOwnership()
        old_key = make_processing_key("image", 0, old_threshold, {"method": "none"})
        window._ownership.set_desired(old_key)
        window._currentThreshParams = lambda: {"method": "otsu", "medianK": 5}
        window._currentSepParams = lambda: {"method": "none"}
        window._imageKey = lambda: ("image", 0)
        window.autoPreviewVar = FakeVar(auto)
        window.statusVar = FakeStatus("")
        window.rightCanvas = FakeCanvas()
        window._cachedRightNp = object()
        window._draftDirty = True
        window._invalidEntry = None
        window._updateApplyState = lambda: None
        window.schedule_count = 0

        def schedule():
            window.schedule_count += 1

        window._schedulePreview = schedule
        return window

    def test_slider_writes_only_mark_draft_and_release_commits_once(self):
        window = object.__new__(ProcessingWindow)
        window._updatingVars = False
        window._draftDirty = False
        window._updateApplyState = lambda: None
        window.useDefaultsVar = FakeVar(False)
        commits = []
        window._commitProcessingParams = lambda **_kwargs: commits.append(True)

        for _ in range(100):
            window._onDraftEdited()
        self.assertTrue(window._draftDirty)
        self.assertEqual(commits, [])

        event = type("Event", (), {"keysym": "Right", "widget": FakeWidget()})()
        window._onScaleKeyRelease(event)
        self.assertEqual(len(commits), 1)

    def test_enter_then_focus_loss_schedules_one_effective_change(self):
        window = self._commit_window(auto=True)
        self.assertTrue(window._commitProcessingParams())
        self.assertTrue(window._commitProcessingParams())
        self.assertEqual(window.schedule_count, 1)

    def test_manual_commit_waits_for_explicit_recompute(self):
        window = self._commit_window(auto=False)
        self.assertTrue(window._commitProcessingParams())
        self.assertEqual(window.schedule_count, 0)
        window.recomputeNow()
        self.assertEqual(window.schedule_count, 1)

    def test_invalid_entry_starts_no_work_and_marks_field(self):
        window = self._commit_window(auto=True)
        window._currentThreshParams = lambda: (_ for _ in ()).throw(ValueError("bad"))
        field = FakeWidget()
        self.assertFalse(window._commitProcessingParams(error_widget=field))
        self.assertEqual(window.schedule_count, 0)
        self.assertEqual(field.states, [("invalid",)])
        self.assertTrue(window._draftDirty)

    def test_auto_off_cancels_dispatch_and_invalidates_running_request(self):
        window = object.__new__(ProcessingWindow)
        window.autoPreviewVar = FakeVar(False)
        window._dispatchAfterId = "dispatch"
        window._ownership = PreviewOwnership()
        key = make_processing_key("image", 0, {"method": "otsu"}, {"method": "none"})
        window._ownership.set_desired(key)
        generation = window._ownership.request_started()
        window._worker = FakeWorker()
        window.statusVar = FakeStatus("")
        window.after_cancel_calls = []
        window.after_cancel = window.after_cancel_calls.append
        window._updateApplyState = lambda: None

        window._onAutoPreviewChanged()
        completion = PreviewCompletion(key, generation, binary=object())
        self.assertEqual(window.after_cancel_calls, ["dispatch"])
        self.assertEqual(window._worker.discard_count, 1)
        self.assertFalse(window._ownership.accept(completion))
        self.assertFalse(window._ownership.can_apply)

    def test_auto_off_preserves_valid_idle_result(self):
        window = object.__new__(ProcessingWindow)
        window.autoPreviewVar = FakeVar(False)
        window._dispatchAfterId = None
        window._ownership = PreviewOwnership()
        key = make_processing_key("image", 0, {"method": "otsu"}, {"method": "none"})
        window._ownership.set_desired(key)
        generation = window._ownership.request_started()
        window._ownership.accept(PreviewCompletion(key, generation, binary=object()))
        window._worker = FakeWorker()
        window.statusVar = FakeStatus("")
        window._updateApplyState = lambda: None

        window._onAutoPreviewChanged()
        self.assertTrue(window._ownership.can_apply)

    def test_display_changes_only_recompose_cached_result(self):
        window = object.__new__(ProcessingWindow)
        renders = []
        window._renderAcceptedPreview = lambda: renders.append(True)
        for _ in range(20):
            window._onViewChanged()
        self.assertEqual(len(renders), 20)

    def test_defaults_leave_separation_controls_unchanged(self):
        window = object.__new__(ProcessingWindow)
        window.methodVar = FakeVar("adaptive")
        for name, value in {
            "polarityVar": "custom", "useCLAHEVar": True, "claheClipVar": 9.0,
            "claheTileVar": 99, "medianKVar": 99, "gaussianKVar": 99,
            "applyOpenCloseVar": True, "morphKVar": 99,
            "adaptiveBlockVar": 99, "adaptiveCVar": 99,
            "percentileVar": 99.0, "pickTolVar": 99,
        }.items():
            setattr(window, name, FakeVar(value))
        separation = {
            "sepMethodVar": "none", "fillHolesVar": False, "minAreaVar": 777,
            "distanceBlurVar": 777, "peakMinDistVar": 777,
            "peakRelThrVar": 0.77, "connectivityVar": 4,
            "clearBorderVar": True, "overlayAlphaVar": 0.77,
        }
        for name, value in separation.items():
            setattr(window, name, FakeVar(value))

        window._restoreDefaultVars()

        for name, value in separation.items():
            self.assertEqual(getattr(window, name).get(), value)
        self.assertNotEqual(window.adaptiveBlockVar.get(), 99)


class TestProcessingTkIntegration(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
            self.root.withdraw()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")

    def tearDown(self):
        if hasattr(self, "root"):
            try:
                self.root.destroy()
            except tk.TclError:
                pass

    def test_idle_and_display_changes_do_not_segment(self):
        image = np.zeros((24, 24), dtype=np.uint8)
        with patch("main.gui.processing.runSeparationPipeline") as pipeline:
            window = ProcessingWindow(self.root, [image], paths=["image.png"])
            self.root.update_idletasks()
            self.root.update()
            for index in range(20):
                window.overlayAlphaVar.set(index / 20.0)
                window.viewModeVar.set("binary" if index % 2 else "labels")
                self.root.update_idletasks()
            self.root.update()
            pipeline.assert_not_called()
            window._onClose()


if __name__ == "__main__":
    unittest.main()
