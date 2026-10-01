import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gui.preview_controller import (
    CommittedProcessingConfig,
    LatestPreviewWorker,
    PreviewCompletion,
    PreviewOwnership,
    PreviewRequest,
    make_processing_key,
)


def key(image="a", revision=0, **threshold):
    values = {
        "method": "otsu",
        "useCLAHE": False,
        "claheClip": 2.0,
        "applyOpenClose": False,
        "morphK": 3,
        **threshold,
    }
    return make_processing_key(image, revision, values, {"method": "none"})


class TestPreviewOwnership(unittest.TestCase):
    def test_disabled_parameters_do_not_change_effective_key(self):
        self.assertEqual(key(claheClip=2.0, morphK=3), key(claheClip=99.0, morphK=101))
        self.assertNotEqual(key(useCLAHE=True, claheClip=2.0), key(useCLAHE=True, claheClip=3.0))

    def test_core_equivalent_values_have_same_key(self):
        self.assertEqual(key(medianK=4), key(medianK=5))
        self.assertEqual(
            key(method="adaptive", adaptiveBlock=2),
            key(method="adaptive", adaptiveBlock=3),
        )
        self.assertEqual(
            make_processing_key(
                "a", 0, {"method": "otsu"},
                {"method": "watershed", "distanceBlurK": 4},
            ),
            make_processing_key(
                "a", 0, {"method": "otsu"},
                {"method": "watershed", "distanceBlurK": 5},
            ),
        )

    def test_committed_configuration_is_detached_from_mutable_inputs(self):
        threshold = {"method": "otsu", "medianK": 3}
        config = CommittedProcessingConfig.create(threshold, {"method": "none"})
        threshold["medianK"] = 99
        self.assertEqual(config.threshold_dict()["medianK"], 3)

    def test_navigation_and_newer_generation_reject_old_results(self):
        ownership = PreviewOwnership()
        key_a = key("a")
        ownership.set_desired(key_a)
        generation_a = ownership.request_started()
        key_b = key("b")
        ownership.set_desired(key_b)
        generation_b = ownership.request_started()

        stale = PreviewCompletion(key_a, generation_a, np.ones((1, 1), np.uint8))
        self.assertFalse(ownership.accept(stale))
        self.assertFalse(ownership.can_apply)

        current = PreviewCompletion(key_b, generation_b, np.ones((1, 1), np.uint8))
        self.assertTrue(ownership.accept(current))
        self.assertTrue(ownership.can_apply)

    def test_failure_for_new_state_disables_apply(self):
        ownership = PreviewOwnership()
        current_key = key()
        ownership.set_desired(current_key)
        generation = ownership.request_started()
        ownership.accept(PreviewCompletion(current_key, generation, error=ValueError("bad")))
        self.assertFalse(ownership.can_apply)


class TestLatestPreviewWorker(unittest.TestCase):
    def test_busy_worker_runs_only_latest_pending_request(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def compute(request):
            calls.append(request.key.image_id)
            if request.key.image_id == "a":
                started.set()
                release.wait(2.0)
            return np.full((1, 1), len(calls), np.uint8), None

        worker = LatestPreviewWorker(compute)
        image = np.zeros((1, 1), np.uint8)
        try:
            worker.submit(PreviewRequest.create(key("a"), 1, image, {}, {}))
            self.assertTrue(started.wait(1.0))
            worker.submit(PreviewRequest.create(key("b"), 2, image, {}, {}))
            worker.submit(PreviewRequest.create(key("c"), 3, image, {}, {}))
            release.set()
            deadline = time.monotonic() + 2.0
            while worker.has_work and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(calls, ["a", "c"])
            self.assertEqual(worker.results.qsize(), 2)
        finally:
            worker.close()

    def test_close_discards_pending_request(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def compute(request):
            calls.append(request.key.image_id)
            started.set()
            release.wait(2.0)
            return np.zeros((1, 1), np.uint8), None

        worker = LatestPreviewWorker(compute)
        image = np.zeros((1, 1), np.uint8)
        worker.submit(PreviewRequest.create(key("a"), 1, image, {}, {}))
        self.assertTrue(started.wait(1.0))
        worker.submit(PreviewRequest.create(key("b"), 2, image, {}, {}))
        worker.close()
        release.set()
        time.sleep(0.02)
        self.assertEqual(calls, ["a"])

    def test_discard_pending_does_not_interrupt_running_request(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def compute(request):
            calls.append(request.key.image_id)
            started.set()
            release.wait(2.0)
            return np.zeros((1, 1), np.uint8), None

        worker = LatestPreviewWorker(compute)
        image = np.zeros((1, 1), np.uint8)
        try:
            worker.submit(PreviewRequest.create(key("a"), 1, image, {}, {}))
            self.assertTrue(started.wait(1.0))
            worker.submit(PreviewRequest.create(key("b"), 2, image, {}, {}))
            worker.discard_pending()
            release.set()
            deadline = time.monotonic() + 1.0
            while worker.has_work and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(calls, ["a"])
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
