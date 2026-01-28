# main/tests/test_batch.py
# Unit tests for main/core/batch.py - parallel batch processing

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.batch import (
    _process_single_image,
    measure_batch,
    process_batch_parallel,
    process_batch_sequential,
    threshold_batch,
)
from core.processing import DEFAULTS
from core.stereology import PoreProps


def _make_circle_image(h=100, w=100, cx=50, cy=50, r=20) -> np.ndarray:
    """Create a grayscale image with a white circle on black background."""
    img = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
    img[mask] = 255
    return img


def _make_multi_circle_image(h=200, w=200) -> np.ndarray:
    """Create image with multiple circles."""
    img = np.zeros((h, w), dtype=np.uint8)
    for cx, cy, r in [(50, 50, 20), (150, 50, 25), (100, 150, 30)]:
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
        img[mask] = 255
    return img


class TestProcessSingleImage(unittest.TestCase):
    """Tests for the worker function _process_single_image."""

    def test_returns_tuple_of_five(self):
        img = _make_circle_image()
        thresh_params = {"method": "otsu", "polarity": "auto"}
        sep_params = dict(DEFAULTS["separation"])
        result = _process_single_image(img, thresh_params, sep_params, image_index=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)

    def test_index_preserved(self):
        img = _make_circle_image()
        thresh_params = {"method": "otsu", "polarity": "auto"}
        sep_params = dict(DEFAULTS["separation"])
        result = _process_single_image(img, thresh_params, sep_params, image_index=42)
        self.assertEqual(result[0], 42)

    def test_binary_is_uint8(self):
        img = _make_circle_image()
        thresh_params = {"method": "otsu", "polarity": "auto"}
        sep_params = dict(DEFAULTS["separation"])
        _, binary, _, _, _ = _process_single_image(img, thresh_params, sep_params)
        self.assertEqual(binary.dtype, np.uint8)
        self.assertTrue(set(np.unique(binary)).issubset({0, 255}))

    def test_labels_is_int32_or_none(self):
        img = _make_circle_image()
        thresh_params = {"method": "otsu", "polarity": "auto"}
        sep_params = dict(DEFAULTS["separation"])
        sep_params["method"] = "watershed"
        _, _, labels, _, _ = _process_single_image(img, thresh_params, sep_params)
        if labels is not None:
            self.assertEqual(labels.dtype, np.int32)

    def test_props_returned_when_measure_true(self):
        img = _make_circle_image()
        thresh_params = {"method": "otsu", "polarity": "auto"}
        sep_params = dict(DEFAULTS["separation"])
        sep_params["method"] = "watershed"
        _, _, _, _, props = _process_single_image(img, thresh_params, sep_params, measure=True)
        self.assertIsNotNone(props)
        self.assertIsInstance(props, list)

    def test_props_none_when_measure_false(self):
        img = _make_circle_image()
        thresh_params = {"method": "otsu", "polarity": "auto"}
        sep_params = dict(DEFAULTS["separation"])
        _, _, _, _, props = _process_single_image(img, thresh_params, sep_params, measure=False)
        self.assertIsNone(props)

    def test_scale_applied_to_props(self):
        img = _make_circle_image()
        thresh_params = {"method": "otsu", "polarity": "auto"}
        sep_params = dict(DEFAULTS["separation"])
        sep_params["method"] = "watershed"
        scale = {"unitsPerPx": 0.1, "unitName": "mm"}
        _, _, _, _, props = _process_single_image(img, thresh_params, sep_params, scale=scale)
        if props:
            self.assertIsNotNone(props[0].area_units2)


class TestProcessBatchSequential(unittest.TestCase):
    """Tests for process_batch_sequential."""

    def test_empty_list_returns_empty(self):
        binaries, labels, props = process_batch_sequential([])
        self.assertEqual(len(binaries), 0)
        self.assertEqual(len(labels), 0)
        self.assertEqual(len(props), 0)

    def test_single_image(self):
        img = _make_circle_image()
        binaries, labels, props = process_batch_sequential([img])
        self.assertEqual(len(binaries), 1)
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(props), 1)

    def test_multiple_images(self):
        imgs = [_make_circle_image(cx=30+i*10) for i in range(5)]
        binaries, labels, props = process_batch_sequential(imgs)
        self.assertEqual(len(binaries), 5)
        self.assertEqual(len(labels), 5)

    def test_all_binaries_valid(self):
        imgs = [_make_circle_image(), _make_multi_circle_image()]
        binaries, _, _ = process_batch_sequential(imgs)
        for b in binaries:
            self.assertEqual(b.dtype, np.uint8)
            self.assertTrue(b.shape[0] > 0 and b.shape[1] > 0)

    def test_progress_callback_called(self):
        imgs = [_make_circle_image() for _ in range(3)]
        progress_calls = []
        def cb(completed, total):
            progress_calls.append((completed, total))
        process_batch_sequential(imgs, progress_callback=cb)
        self.assertEqual(len(progress_calls), 3)
        self.assertEqual(progress_calls[-1], (3, 3))


class TestProcessBatchParallel(unittest.TestCase):
    """Tests for process_batch_parallel (uses ProcessPoolExecutor)."""

    def test_empty_list_returns_empty(self):
        binaries, labels, props = process_batch_parallel([])
        self.assertEqual(len(binaries), 0)

    def test_single_image_works(self):
        img = _make_circle_image()
        binaries, labels, props = process_batch_parallel([img], max_workers=1)
        self.assertEqual(len(binaries), 1)
        self.assertEqual(binaries[0].dtype, np.uint8)

    def test_multiple_images(self):
        imgs = [_make_circle_image(cx=30+i*10) for i in range(4)]
        binaries, labels, props = process_batch_parallel(imgs, max_workers=2)
        self.assertEqual(len(binaries), 4)

    def test_results_match_sequential(self):
        """Parallel and sequential should produce identical results."""
        np.random.seed(42)
        imgs = [_make_circle_image(cx=40+i*15, r=15+i*2) for i in range(3)]
        thresh_params = {"method": "otsu", "polarity": "auto"}
        sep_params = dict(DEFAULTS["separation"])
        sep_params["method"] = "none"  # deterministic

        bin_seq, lab_seq, _ = process_batch_sequential(imgs, thresh_params, sep_params, measure=False)
        bin_par, lab_par, _ = process_batch_parallel(imgs, thresh_params, sep_params, measure=False, max_workers=2)

        for i in range(3):
            np.testing.assert_array_equal(bin_seq[i], bin_par[i])

    def test_max_workers_respected(self):
        imgs = [_make_circle_image() for _ in range(10)]
        # Just ensure it doesn't crash with various worker counts
        binaries, _, _ = process_batch_parallel(imgs, max_workers=1)
        self.assertEqual(len(binaries), 10)
        binaries2, _, _ = process_batch_parallel(imgs, max_workers=4)
        self.assertEqual(len(binaries2), 10)


class TestThresholdBatch(unittest.TestCase):
    """Tests for threshold_batch convenience function."""

    def test_returns_binaries_only(self):
        imgs = [_make_circle_image(), _make_multi_circle_image()]
        binaries = threshold_batch(imgs)
        self.assertEqual(len(binaries), 2)
        for b in binaries:
            self.assertEqual(b.dtype, np.uint8)

    def test_custom_params(self):
        imgs = [_make_circle_image()]
        # threshold_batch takes method/polarity as kwargs, not thresh_params dict
        binaries = threshold_batch(imgs, method="percentile", polarity="auto", percentile=50.0)
        self.assertEqual(len(binaries), 1)


class TestMeasureBatch(unittest.TestCase):
    """Tests for measure_batch convenience function.

    Note: measure_batch returns a FLAT list of all PoreProps across all images,
    not a list-of-lists. Use measure_dataset directly if you need per-image grouping.
    """

    def test_returns_flat_list(self):
        # Create simple label arrays
        lab1 = np.zeros((50, 50), dtype=np.int32)
        lab1[10:20, 10:20] = 1
        lab2 = np.zeros((50, 50), dtype=np.int32)
        lab2[5:15, 5:15] = 1
        lab2[25:40, 25:40] = 2

        props = measure_batch([lab1, lab2])
        # Flat list: 1 from lab1 + 2 from lab2 = 3 total
        self.assertEqual(len(props), 3)
        self.assertTrue(all(isinstance(p, PoreProps) for p in props))

    def test_handles_none_labels(self):
        lab1 = np.zeros((50, 50), dtype=np.int32)
        lab1[10:20, 10:20] = 1
        props = measure_batch([lab1, None, lab1])
        # Should have 1 + 0 + 1 = 2 props total (None is skipped)
        self.assertEqual(len(props), 2)

    def test_scales_applied(self):
        lab = np.zeros((100, 100), dtype=np.int32)
        lab[20:40, 20:40] = 1
        scales = [{"unitsPerPx": 0.5, "unitName": "um"}]
        props = measure_batch([lab], scales=scales)
        self.assertEqual(len(props), 1)
        self.assertIsNotNone(props[0].area_units2)


if __name__ == "__main__":
    unittest.main()
