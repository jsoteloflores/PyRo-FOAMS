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


class FakeButton:
    def __init__(self):
        self.states = []

    def configure(self, *, state):
        self.states.append(state)


class TestProcessingEventContract(unittest.TestCase):
    def _commit_window(self, *, auto=True, accepted=False):
        window = object.__new__(ProcessingWindow)
        old_threshold = {"method": "otsu", "medianK": 3}
        window._committedConfig = CommittedProcessingConfig.create(
            old_threshold, {"method": "none"}
        )
        window._ownership = PreviewOwnership()
        old_key = make_processing_key("image", 0, old_threshold, {"method": "none"})
        window._ownership.set_desired(old_key)
        if accepted:
            generation = window._ownership.request_started()
            window._ownership.accept(
                PreviewCompletion(old_key, generation, binary=np.ones((1, 1), np.uint8))
            )
        window._currentThreshParams = lambda: {"method": "otsu", "medianK": 5}
        window._currentSepParams = lambda: {"method": "none"}
        window._imageKey = lambda: ("image", 0)
        window.autoPreviewVar = FakeVar(auto)
        window.useDefaultsVar = FakeVar(False)
        window.statusVar = FakeStatus("")
        window.rightCanvas = FakeCanvas()
        window._cachedRightNp = object()
        window._updatingVars = False
        window._draftDirty = True
        window._invalidEntry = None
        window.applyCurrentButton = FakeButton()
        window._dispatchAfterId = None
        window._worker = FakeWorker()
        window.after_cancel = lambda _callback: None
        window.schedule_count = 0

        def schedule():
            if window._dispatchAfterId is None:
                window.schedule_count += 1
                window._dispatchAfterId = "scheduled"

        window._schedulePreview = schedule
        return window

    @staticmethod
    def _install_threshold_controls(window):
        values = {
            "methodVar": "otsu", "polarityVar": "auto",
            "useCLAHEVar": False, "claheClipVar": 2.0, "claheTileVar": 8,
            "medianKVar": 3, "gaussianKVar": 0,
            "applyOpenCloseVar": False, "morphKVar": 3,
            "adaptiveBlockVar": 31, "adaptiveCVar": 2,
            "percentileVar": 50.0, "pickTolVar": 10, "pickValueVar": 128,
            "useDefaultsVar": False,
        }
        for name, value in values.items():
            setattr(window, name, FakeVar(value))

    def test_unchanged_valid_commit_restores_apply_without_work(self):
        window = self._commit_window(auto=True, accepted=True)
        original_result = window._ownership.result
        original_key = window._ownership.desired_key
        original_generation = window._ownership.generation
        window._currentThreshParams = lambda: {"method": "otsu", "medianK": 3}

        window._onDraftEdited()
        self.assertEqual(window.applyCurrentButton.states[-1], "disabled")
        self.assertTrue(window._commitProcessingParams())

        self.assertEqual(window.applyCurrentButton.states[-1], "normal")
        self.assertEqual(window.schedule_count, 0)
        self.assertIs(window._ownership.result, original_result)
        self.assertEqual(window._ownership.desired_key, original_key)
        self.assertEqual(window._ownership.generation, original_generation)

    def test_inactive_parameter_change_restores_apply_without_work(self):
        window = self._commit_window(auto=True, accepted=True)
        window._currentThreshParams = lambda: {
            "method": "otsu", "medianK": 3,
            "useCLAHE": False, "claheClip": 999.0,
        }
        window._onDraftEdited()
        self.assertTrue(window._commitProcessingParams())
        self.assertEqual(window.applyCurrentButton.states[-1], "normal")
        self.assertEqual(window.schedule_count, 0)

    def test_enter_then_focus_loss_unchanged_starts_no_work(self):
        window = self._commit_window(auto=True, accepted=True)
        window._currentThreshParams = lambda: {"method": "otsu", "medianK": 3}

        window._onComputationChanged(type("Event", (), {"widget": FakeWidget()})())
        window._onComputationChanged(type("Event", (), {"widget": FakeWidget()})())

        self.assertEqual(window.schedule_count, 0)
        self.assertEqual(window.applyCurrentButton.states[-1], "normal")

    def test_effective_change_disables_apply_until_matching_completion(self):
        window = self._commit_window(auto=True, accepted=True)
        self.assertTrue(window._commitProcessingParams())
        self.assertEqual(window.schedule_count, 1)
        self.assertEqual(window.applyCurrentButton.states[-1], "disabled")

        generation = window._ownership.request_started()
        key = window._ownership.desired_key
        window._ownership.accept(PreviewCompletion(key, generation, binary=object()))
        window._updateApplyState()
        self.assertEqual(window.applyCurrentButton.states[-1], "normal")

    def test_correcting_invalid_draft_to_original_reuses_result(self):
        window = self._commit_window(auto=True, accepted=True)
        original = window._committedConfig
        window._currentThreshParams = lambda: {
            "method": "percentile", "percentile": float("nan")
        }
        self.assertFalse(window._commitProcessingParams())

        window._currentThreshParams = lambda: {"method": "otsu", "medianK": 3}
        self.assertTrue(window._commitProcessingParams())
        self.assertEqual(window.applyCurrentButton.states[-1], "normal")
        self.assertEqual(window.schedule_count, 0)
        self.assertEqual(window._committedConfig, original)

    def test_unchanged_commit_does_not_enable_pending_or_failed_result(self):
        pending = self._commit_window(auto=True)
        pending._currentThreshParams = lambda: {"method": "otsu", "medianK": 3}
        pending._ownership.request_started()
        self.assertTrue(pending._commitProcessingParams())
        self.assertEqual(pending.applyCurrentButton.states[-1], "disabled")
        self.assertEqual(pending.schedule_count, 0)

        failed = self._commit_window(auto=True)
        failed._currentThreshParams = lambda: {"method": "otsu", "medianK": 3}
        generation = failed._ownership.request_started()
        failed._ownership.accept(
            PreviewCompletion(
                failed._ownership.desired_key, generation, error=RuntimeError("failed")
            )
        )
        self.assertTrue(failed._commitProcessingParams())
        self.assertEqual(failed.applyCurrentButton.states[-1], "disabled")
        self.assertEqual(failed.schedule_count, 1)

    def test_invalid_nonfinite_commit_preserves_config_and_starts_no_work(self):
        window = self._commit_window(auto=True, accepted=True)
        original = window._committedConfig
        window._currentThreshParams = lambda: {
            "method": "percentile", "percentile": float("nan")
        }
        field = FakeWidget()

        self.assertFalse(window._commitProcessingParams(error_widget=field))
        self.assertIs(window._committedConfig, original)
        self.assertEqual(window.schedule_count, 0)
        self.assertEqual(window.applyCurrentButton.states[-1], "disabled")
        self.assertIn("percentile", window.statusVar.get())

    def test_invalid_commit_cancels_queued_automatic_dispatch(self):
        window = self._commit_window(auto=True, accepted=True)
        window._dispatchAfterId = "queued"
        cancelled = []
        window.after_cancel = cancelled.append
        window._currentThreshParams = lambda: {
            "method": "percentile", "percentile": float("inf")
        }

        self.assertFalse(window._commitProcessingParams())
        self.assertEqual(cancelled, ["queued"])
        self.assertIsNone(window._dispatchAfterId)
        self.assertEqual(window._worker.discard_count, 1)

    def test_running_completion_cannot_enable_apply_while_draft_invalid(self):
        window = self._commit_window(auto=True)
        generation = window._ownership.request_started()
        key = window._ownership.desired_key
        window._currentThreshParams = lambda: {
            "method": "percentile", "percentile": float("-inf")
        }
        self.assertFalse(window._commitProcessingParams())

        window._ownership.accept(PreviewCompletion(key, generation, binary=object()))
        window._updateApplyState()
        self.assertEqual(window.applyCurrentButton.states[-1], "disabled")

    def test_invalid_event_marks_its_source_widget(self):
        window = self._commit_window(auto=True)
        window._currentThreshParams = lambda: {
            "method": "otsu", "medianK": 3.5
        }
        source = FakeWidget()
        window.useDefaultsVar = FakeVar(False)

        window._onComputationChanged(type("Event", (), {"widget": source})())

        self.assertEqual(source.states, [("invalid",)])
        self.assertIn("medianK", window.statusVar.get())

    def test_advanced_invalid_candidate_is_atomic(self):
        window = self._commit_window(auto=True, accepted=True)
        self._install_threshold_controls(window)
        original = window._committedConfig
        original_controls = (window.useCLAHEVar.get(), window.claheTileVar.get())

        with self.assertRaisesRegex(ValueError, "claheTile"):
            window._commitAdvancedValues({"useCLAHE": True, "claheTile": 0})

        self.assertIs(window._committedConfig, original)
        self.assertEqual(
            (window.useCLAHEVar.get(), window.claheTileVar.get()), original_controls
        )
        self.assertEqual(window.schedule_count, 0)

    def test_advanced_valid_candidate_installs_once(self):
        window = self._commit_window(auto=True, accepted=True)
        self._install_threshold_controls(window)

        self.assertTrue(window._commitAdvancedValues({"medianK": 4}))

        self.assertEqual(window._committedConfig.threshold_dict()["medianK"], 5)
        self.assertEqual(window.medianKVar.get(), 5)
        self.assertFalse(window.useDefaultsVar.get())
        self.assertEqual(window.schedule_count, 1)

    def test_advanced_cancel_without_commit_changes_nothing(self):
        window = self._commit_window(auto=True, accepted=True)
        self._install_threshold_controls(window)
        original = window._committedConfig
        original_controls = window.medianKVar.get()

        # Closing the dialog does not call its OK-only commit boundary.
        self.assertIs(window._committedConfig, original)
        self.assertEqual(window.medianKVar.get(), original_controls)
        self.assertEqual(window.schedule_count, 0)

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
