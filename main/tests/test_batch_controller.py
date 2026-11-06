import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main.core.sample_project import add_image, create_project, update_draft
from main.gui.batch_controller import BatchJobCoordinator, BatchTargetResult, BatchTargetSnapshot
from main.gui.preprocessing import PreprocessApp
from main.gui.processing import ProcessingWindow


class FakeStatus:
    value = ""

    def set(self, _value):
        self.value = _value


class FakeProgress:
    def __getitem__(self, _name):
        return 0


class TestBatchJobCoordinator(unittest.TestCase):
    def test_failed_target_retains_previous_draft(self):
        window = object.__new__(ProcessingWindow)
        previous_binary = np.full((2, 2), 255, np.uint8)
        previous_labels = np.ones((2, 2), np.int32)
        window.imageIds = ["image-a"]
        window.projectId = "sample-1"
        window.projectSessionId = "session-1"
        window.imageRevisions = [2]
        window.draftRevisions = [3]
        window._batchJobId = "job-1"
        window._committedConfig = SimpleNamespace(
            threshold=(("method", "otsu"),),
            separation=(("method", "none"),),
        )
        window._batchFailureCount = 0
        window._batchStaleCount = 0
        window._batchSuccessCount = 0
        window._batchTotal = 1
        window.batchProgress = FakeProgress()
        window.batchStatusVar = FakeStatus()
        window.binaries = [previous_binary.copy()]
        window.labels = [previous_labels.copy()]
        window.labelRevisions = [4]
        window.batchTargetCallback = lambda _result: self.fail("failed results must not reach project commit")
        result = BatchTargetResult(
            job_id="job-1",
            project_id="sample-1",
            session_id="session-1",
            index=0,
            image_id="image-a",
            source_revision=2,
            draft_revision=3,
            threshold=(("method", "otsu"),),
            separation=(("method", "none"),),
            error="ValueError: synthetic target failure",
        )

        window._commitBatchTarget(result)

        self.assertEqual(window._batchFailureCount, 1)
        np.testing.assert_array_equal(window.binaries[0], previous_binary)
        np.testing.assert_array_equal(window.labels[0], previous_labels)

    def test_project_controller_rejects_stale_session_and_draft_revisions(self):
        project = create_project("Batch revisions")
        image = add_image(project, np.zeros((4, 4), dtype=np.uint8), "image.png")
        previous = np.zeros((4, 4), dtype=np.uint8)
        previous[1:3, 1:3] = 255
        update_draft(image, mask=previous, labels=None, settings={})

        app = object.__new__(PreprocessApp)
        app.project = project
        app.projectSessionId = "session-current"
        app.masks = [image.draft_mask.copy()]
        app.labels = [image.draft_labels.copy()]
        app.maskRevisions = [image.draft_revision]
        app.labelRevisions = [image.draft_revision]
        app.draftRevisions = [image.draft_revision]
        app._markUnsaved = lambda: None
        app._redrawThumb = lambda _index: None
        app.statusVar = FakeStatus()

        replacement = np.zeros((4, 4), dtype=np.uint8)
        replacement[0:2, 0:2] = 255
        base = dict(
            job_id="job-1",
            project_id=project.project_id,
            index=0,
            image_id=image.image_id,
            source_revision=image.source_revision,
            draft_revision=image.draft_revision,
            threshold=(("method", "otsu"),),
            separation=(("method", "none"),),
            binary=replacement,
        )
        stale_revision = BatchTargetResult(session_id="session-current", **{**base, "draft_revision": 0})
        stale_session = BatchTargetResult(session_id="old-session", **base)
        self.assertFalse(app._acceptBatchDraftResult(stale_revision))
        self.assertFalse(app._acceptBatchDraftResult(stale_session))
        np.testing.assert_array_equal(image.draft_mask, previous)

        current = BatchTargetResult(session_id="session-current", **base)
        self.assertTrue(app._acceptBatchDraftResult(current))
        np.testing.assert_array_equal(image.draft_mask, replacement)

    def test_selected_targets_return_stable_ids_and_independent_failures(self):
        def process(images, **_kwargs):
            value = int(images[0][0, 0])
            if value == 2:
                raise ValueError("synthetic failure")
            return [np.full((2, 2), 255, np.uint8)], [np.ones((2, 2), np.int32)], []

        targets = (
            BatchTargetSnapshot(3, "image-a", 4, 8, np.full((2, 2), 1, np.uint8)),
            BatchTargetSnapshot(7, "image-b", 6, 9, np.full((2, 2), 2, np.uint8)),
        )
        coordinator = BatchJobCoordinator()
        with patch("main.gui.batch_controller.process_batch_parallel", side_effect=process):
            job_id = coordinator.start(
                project_id="sample-1",
                targets=targets,
                threshold={"method": "otsu"},
                separation={"method": "none"},
                max_workers=2,
            )
            events = []
            while not any(event.kind == "finished" for event in events):
                events.append(coordinator.events.get(timeout=3.0))

        results = [event.result for event in events if event.kind == "target"]
        self.assertEqual({result.image_id for result in results}, {"image-a", "image-b"})
        self.assertTrue(all(result.job_id == job_id for result in results))
        self.assertTrue(all(result.project_id == "sample-1" for result in results))
        failed = next(result for result in results if result.image_id == "image-b")
        self.assertEqual(failed.error, "ValueError: synthetic failure")
        succeeded = next(result for result in results if result.image_id == "image-a")
        self.assertEqual(succeeded.source_revision, 4)
        self.assertEqual(succeeded.draft_revision, 8)
        self.assertTrue(np.all(succeeded.binary == 255))

    def test_cancelled_running_target_emits_no_late_target_result(self):
        started = threading.Event()
        release = threading.Event()

        def process(images, **_kwargs):
            started.set()
            release.wait(3.0)
            return [np.full((2, 2), 255, np.uint8)], [np.ones((2, 2), np.int32)], []

        coordinator = BatchJobCoordinator()
        with patch("main.gui.batch_controller.process_batch_parallel", side_effect=process):
            job_id = coordinator.start(
                project_id="sample-1",
                targets=(BatchTargetSnapshot(0, "image-a", 1, 2, np.zeros((2, 2), np.uint8)),),
                threshold={"method": "otsu"},
                separation={"method": "none"},
                max_workers=1,
            )
            self.assertTrue(started.wait(1.0))
            self.assertTrue(coordinator.cancel(job_id))
            release.set()
            events = []
            while not any(event.kind == "finished" for event in events):
                events.append(coordinator.events.get(timeout=3.0))

        self.assertFalse(any(event.kind == "target" for event in events))
        self.assertTrue(events[-1].cancelled)


if __name__ == "__main__":
    unittest.main()
