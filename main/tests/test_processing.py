# main/tests/test_processing.py
# Unit tests for core/processing.py: thresholding, cleanup, separation

import unittest
import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.processing import (
    thresholdImageAdvanced,
    removeSmallAreas,
    clearBorderTouching,
    fillHoles,
    watershedSeparate,
    postSeparateCleanup,
    runSeparationPipeline,
    labelsToColor,
    DEFAULTS,
)


class TestThresholdImageAdvanced(unittest.TestCase):
    """Tests for thresholdImageAdvanced() with different methods."""

    def setUp(self):
        # Create a simple test image: dark background with bright circle
        self.img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(self.img, (50, 50), 20, 200, -1)

    def test_otsu_returns_binary(self):
        binary, meta = thresholdImageAdvanced(self.img, method="otsu")
        self.assertEqual(binary.dtype, np.uint8)
        self.assertTrue(set(np.unique(binary)).issubset({0, 255}))
        self.assertEqual(meta["method"], "otsu")

    def test_adaptive_returns_binary(self):
        binary, meta = thresholdImageAdvanced(
            self.img, method="adaptive", adaptiveBlock=11, adaptiveC=2
        )
        self.assertEqual(binary.dtype, np.uint8)
        self.assertTrue(set(np.unique(binary)).issubset({0, 255}))
        self.assertEqual(meta["method"], "adaptive")

    def test_percentile_returns_binary(self):
        binary, meta = thresholdImageAdvanced(
            self.img, method="percentile", percentile=50.0
        )
        self.assertEqual(binary.dtype, np.uint8)
        self.assertTrue(set(np.unique(binary)).issubset({0, 255}))
        self.assertIn("thresh", meta)

    def test_pick_method_requires_value(self):
        with self.assertRaises(ValueError):
            thresholdImageAdvanced(self.img, method="pick", pickValue=None)

    def test_pick_method_with_value(self):
        binary, meta = thresholdImageAdvanced(
            self.img, method="pick", pickValue=200, pickTolerance=10
        )
        self.assertEqual(binary.dtype, np.uint8)
        self.assertEqual(meta["pickValue"], 200)
        self.assertEqual(meta["pickTolerance"], 10)
        # Should select the circle area (gray=200)
        self.assertGreater(binary.sum(), 0)

    def test_polarity_poresDarker(self):
        binary, _ = thresholdImageAdvanced(self.img, method="otsu", polarity="poresDarker")
        # Background is darker (0), so with poresDarker, background should be foreground
        self.assertEqual(binary.dtype, np.uint8)

    def test_polarity_poresBrighter(self):
        binary, _ = thresholdImageAdvanced(self.img, method="otsu", polarity="poresBrighter")
        self.assertEqual(binary.dtype, np.uint8)

    def test_clahe_preprocessing(self):
        binary, _ = thresholdImageAdvanced(
            self.img, method="otsu", useCLAHE=True, claheClip=2.0, claheTile=8
        )
        self.assertEqual(binary.dtype, np.uint8)

    def test_median_blur_preprocessing(self):
        binary, _ = thresholdImageAdvanced(self.img, method="otsu", medianK=5)
        self.assertEqual(binary.dtype, np.uint8)

    def test_morph_cleanup(self):
        binary, _ = thresholdImageAdvanced(
            self.img, method="otsu", applyOpenClose=True, morphK=3
        )
        self.assertEqual(binary.dtype, np.uint8)

    def test_bgr_input_converted(self):
        bgr = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        binary, _ = thresholdImageAdvanced(bgr, method="otsu")
        self.assertEqual(binary.dtype, np.uint8)
        self.assertEqual(binary.shape, (100, 100))


class TestRemoveSmallAreas(unittest.TestCase):
    """Tests for vectorized removeSmallAreas()."""

    def test_removes_small_components(self):
        # Create binary with one large and one small component
        binary = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(binary, (30, 30), 20, 255, -1)  # large
        cv2.circle(binary, (80, 80), 3, 255, -1)   # small (~28 px)

        result = removeSmallAreas(binary, minArea=50, connectivity=8)

        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(set(np.unique(result)).issubset({0, 255}))
        # Large circle should remain
        self.assertGreater(result[30, 30], 0)
        # Small circle should be removed
        self.assertEqual(result[80, 80], 0)

    def test_keeps_all_when_minArea_1(self):
        binary = np.zeros((50, 50), dtype=np.uint8)
        binary[10, 10] = 255  # single pixel
        result = removeSmallAreas(binary, minArea=1)
        self.assertEqual(result[10, 10], 255)

    def test_empty_image(self):
        binary = np.zeros((50, 50), dtype=np.uint8)
        result = removeSmallAreas(binary, minArea=10)
        self.assertEqual(result.sum(), 0)

    def test_connectivity_4_vs_8(self):
        # Diagonal neighbors: connected with 8, not with 4
        binary = np.zeros((10, 10), dtype=np.uint8)
        binary[3, 3] = 255
        binary[4, 4] = 255
        binary[5, 5] = 255

        result4 = removeSmallAreas(binary, minArea=2, connectivity=4)
        result8 = removeSmallAreas(binary, minArea=2, connectivity=8)

        # With 4-connectivity, each pixel is separate (area=1), so all removed
        self.assertEqual(result4.sum(), 0)
        # With 8-connectivity, they form one component (area=3), so kept
        self.assertGreater(result8.sum(), 0)


class TestClearBorderTouching(unittest.TestCase):
    """Tests for vectorized clearBorderTouching()."""

    def test_removes_border_components(self):
        binary = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(binary, (50, 50), 15, 255, -1)  # center, not touching
        cv2.circle(binary, (5, 50), 10, 255, -1)   # left edge, touching

        result = clearBorderTouching(binary, connectivity=8)

        # Center circle should remain
        self.assertGreater(result[50, 50], 0)
        # Left-edge circle should be removed
        self.assertEqual(result[5, 50], 0)

    def test_keeps_interior_components(self):
        binary = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(binary, (20, 20), (80, 80), 255, -1)

        result = clearBorderTouching(binary, connectivity=8)

        # Interior rectangle should remain
        self.assertGreater(result[50, 50], 0)

    def test_empty_image(self):
        binary = np.zeros((50, 50), dtype=np.uint8)
        result = clearBorderTouching(binary)
        self.assertEqual(result.sum(), 0)


class TestFillHoles(unittest.TestCase):
    """Tests for fillHoles()."""

    def test_fills_internal_holes(self):
        # Ring shape (outer circle minus inner circle)
        binary = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(binary, (50, 50), 30, 255, -1)  # outer
        cv2.circle(binary, (50, 50), 10, 0, -1)    # hole

        result = fillHoles(binary)

        # Center (the hole) should now be filled
        self.assertEqual(result[50, 50], 255)

    def test_preserves_solid_regions(self):
        binary = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(binary, (50, 50), 20, 255, -1)

        result = fillHoles(binary)

        # Should be identical (no holes to fill)
        np.testing.assert_array_equal(result, binary)


class TestWatershedSeparate(unittest.TestCase):
    """Tests for watershedSeparate()."""

    def test_separates_touching_circles(self):
        # Two overlapping circles
        binary = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(binary, (35, 50), 20, 255, -1)
        cv2.circle(binary, (65, 50), 20, 255, -1)

        labels, dist = watershedSeparate(
            binary, distanceBlurK=3, peakMinDistance=10, peakRelThreshold=0.2
        )

        self.assertEqual(labels.dtype, np.int32)
        # Should have at least 2 distinct labels (plus background 0)
        unique = np.unique(labels)
        self.assertGreaterEqual(len(unique[unique > 0]), 2)

    def test_single_circle_stays_one_label(self):
        binary = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(binary, (50, 50), 20, 255, -1)

        labels, _ = watershedSeparate(binary, peakMinDistance=50)

        unique = np.unique(labels)
        # Should have exactly 1 object label
        self.assertEqual(len(unique[unique > 0]), 1)


class TestPostSeparateCleanup(unittest.TestCase):
    """Tests for vectorized postSeparateCleanup()."""

    def test_removes_small_labels(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:30, 10:30] = 1  # 400 px
        labels[40:42, 40:42] = 2  # 4 px

        result = postSeparateCleanup(labels, minAreaPx=50)

        self.assertIn(1, result)  # large label kept (renumbered)
        self.assertNotIn(2, result)  # small label removed

    def test_removes_border_labels(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:20, 10:20] = 1  # interior
        labels[0:5, 20:30] = 2    # touches top border

        result = postSeparateCleanup(labels, clearBorder=True)

        # Interior label should remain
        self.assertGreater(result[15, 15], 0)
        # Border label should be removed
        self.assertEqual(result[2, 25], 0)

    def test_relabels_sequentially(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[5:15, 5:15] = 5    # non-sequential label (100 px)
        labels[30:40, 30:40] = 10 # non-sequential label (100 px)

        # minAreaPx > 1 triggers the relabeling path
        result = postSeparateCleanup(labels, minAreaPx=2)

        unique = np.unique(result)
        unique = unique[unique > 0]
        # Should have exactly 2 labels, relabeled sequentially starting at 1
        self.assertEqual(len(unique), 2)
        self.assertEqual(unique.min(), 1)
        self.assertEqual(unique.max(), 2)


class TestLabelsToColor(unittest.TestCase):
    """Tests for labelsToColor()."""

    def test_returns_bgr_uint8(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:20, 10:20] = 1

        color = labelsToColor(labels)

        self.assertEqual(color.dtype, np.uint8)
        self.assertEqual(color.shape, (50, 50, 3))

    def test_background_is_black(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:20, 10:20] = 1

        color = labelsToColor(labels)

        # Background pixel should be black
        np.testing.assert_array_equal(color[0, 0], [0, 0, 0])

    def test_overlay_on_gray(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:20, 10:20] = 1
        gray = np.full((50, 50), 128, dtype=np.uint8)

        color = labelsToColor(labels, bgGray=gray, alpha=0.5)

        self.assertEqual(color.dtype, np.uint8)
        self.assertEqual(color.shape, (50, 50, 3))


class TestRunSeparationPipeline(unittest.TestCase):
    """End-to-end tests for runSeparationPipeline()."""

    def test_basic_pipeline(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 20, 200, -1)

        tparams = {"method": "otsu", "polarity": "auto"}
        sparams = {"method": "none", "fillHoles": False, "minAreaPx": 0}

        binary, labels, meta = runSeparationPipeline(img, tparams, sparams)

        self.assertEqual(binary.dtype, np.uint8)
        self.assertIsNone(labels)  # method="none"
        self.assertEqual(meta["method"], "otsu")

    def test_pipeline_with_watershed(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (35, 50), 18, 200, -1)
        cv2.circle(img, (65, 50), 18, 200, -1)

        tparams = {"method": "otsu", "polarity": "auto"}
        sparams = {
            "method": "watershed",
            "fillHoles": True,
            "minAreaPx": 10,
            "distanceBlurK": 3,
            "peakMinDistance": 10,
            "peakRelThreshold": 0.2,
            "connectivity": 8,
            "clearBorder": False,
        }

        binary, labels, meta = runSeparationPipeline(img, tparams, sparams)

        self.assertEqual(binary.dtype, np.uint8)
        self.assertIsNotNone(labels)
        self.assertEqual(labels.dtype, np.int32)


if __name__ == "__main__":
    unittest.main()
