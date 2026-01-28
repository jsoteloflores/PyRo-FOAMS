# main/tests/test_stereology.py
# Unit tests for core/stereology.py: measurements, colorization, CSV export

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.stereology import (
    PoreProps,
    colorize_labels,
    mask_from_labels,
    measure_dataset,
    measure_labels,
    save_props_csv,
)


class TestMeasureLabels(unittest.TestCase):
    """Tests for measure_labels() per-pore measurements."""

    def setUp(self):
        # Create a simple label map with one circular pore
        self.labels = np.zeros((100, 100), dtype=np.int32)
        cv2.circle(self.labels, (50, 50), 20, 1, -1)

    def test_returns_list_of_poreprops(self):
        props = measure_labels(self.labels)

        self.assertIsInstance(props, list)
        self.assertEqual(len(props), 1)
        self.assertIsInstance(props[0], PoreProps)

    def test_area_calculation(self):
        props = measure_labels(self.labels)
        p = props[0]

        # Circle area ~ π * r² ≈ 1256 px
        self.assertGreater(p.area_px, 1200)
        self.assertLess(p.area_px, 1300)

    def test_centroid_calculation(self):
        props = measure_labels(self.labels)
        p = props[0]

        # Centroid should be near (50, 50)
        self.assertAlmostEqual(p.centroid_x, 50, delta=1)
        self.assertAlmostEqual(p.centroid_y, 50, delta=1)

    def test_circularity_near_one(self):
        props = measure_labels(self.labels)
        p = props[0]

        # Circle should have circularity close to 1.0
        self.assertGreater(p.circularity, 0.9)
        self.assertLessEqual(p.circularity, 1.0)

    def test_equivalent_diameter(self):
        props = measure_labels(self.labels)
        p = props[0]

        # eq_diam = sqrt(4*A/π) ≈ 2*r ≈ 40
        self.assertGreater(p.eq_diam_px, 38)
        self.assertLess(p.eq_diam_px, 42)

    def test_border_touching_detection(self):
        # Pore touching left edge
        labels = np.zeros((100, 100), dtype=np.int32)
        cv2.circle(labels, (5, 50), 10, 1, -1)  # touches left

        props = measure_labels(labels)
        self.assertTrue(props[0].touches_border)

        # Pore in center
        labels2 = np.zeros((100, 100), dtype=np.int32)
        cv2.circle(labels2, (50, 50), 10, 1, -1)

        props2 = measure_labels(labels2)
        self.assertFalse(props2[0].touches_border)

    def test_scale_applied(self):
        scale = {"unitsPerPx": 0.01, "unitName": "mm"}
        props = measure_labels(self.labels, scale=scale)
        p = props[0]

        self.assertIsNotNone(p.area_units2)
        self.assertIsNotNone(p.eq_diam_units)
        self.assertEqual(p.unit_name, "mm")
        # area_units2 = area_px * (0.01)^2
        self.assertAlmostEqual(p.area_units2, p.area_px * 0.0001, places=6)

    def test_multiple_labels(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        cv2.circle(labels, (25, 25), 10, 1, -1)
        cv2.circle(labels, (75, 75), 15, 2, -1)

        props = measure_labels(labels)

        self.assertEqual(len(props), 2)
        labels_found = {p.label for p in props}
        self.assertEqual(labels_found, {1, 2})

    def test_empty_labels(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        props = measure_labels(labels)

        self.assertEqual(props, [])


class TestMeasureDataset(unittest.TestCase):
    """Tests for measure_dataset() across multiple images."""

    def test_aggregates_all_images(self):
        labels1 = np.zeros((50, 50), dtype=np.int32)
        cv2.circle(labels1, (25, 25), 10, 1, -1)

        labels2 = np.zeros((50, 50), dtype=np.int32)
        cv2.circle(labels2, (25, 25), 8, 1, -1)
        cv2.circle(labels2, (40, 40), 5, 2, -1)

        props = measure_dataset([labels1, labels2])

        self.assertEqual(len(props), 3)  # 1 + 2 pores

    def test_handles_none_labels(self):
        labels1 = np.zeros((50, 50), dtype=np.int32)
        cv2.circle(labels1, (25, 25), 10, 1, -1)

        props = measure_dataset([labels1, None, None])

        self.assertEqual(len(props), 1)

    def test_image_index_tracked(self):
        labels1 = np.zeros((50, 50), dtype=np.int32)
        cv2.circle(labels1, (25, 25), 10, 1, -1)

        labels2 = np.zeros((50, 50), dtype=np.int32)
        cv2.circle(labels2, (25, 25), 10, 1, -1)

        props = measure_dataset([labels1, labels2])

        indices = {p.image_index for p in props}
        self.assertEqual(indices, {0, 1})


class TestColorizeLabels(unittest.TestCase):
    """Tests for colorize_labels()."""

    def test_returns_bgr_uint8(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        cv2.circle(labels, (25, 25), 10, 1, -1)

        color = colorize_labels(labels)

        self.assertEqual(color.dtype, np.uint8)
        self.assertEqual(color.shape, (50, 50, 3))

    def test_background_preserved(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        cv2.circle(labels, (25, 25), 10, 1, -1)

        color = colorize_labels(labels)

        # Background should be dark (not exactly black due to blending)
        self.assertLess(color[0, 0].sum(), 50)

    def test_overlay_on_gray(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        cv2.circle(labels, (25, 25), 10, 1, -1)
        gray = np.full((50, 50), 128, dtype=np.uint8)

        color = colorize_labels(labels, bg_gray=gray, alpha=0.5)

        self.assertEqual(color.dtype, np.uint8)
        self.assertEqual(color.shape, (50, 50, 3))

    def test_different_labels_different_colors(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[10:30, 10:30] = 1
        labels[60:80, 60:80] = 2

        color = colorize_labels(labels, seed=123)

        color1 = tuple(color[20, 20])
        color2 = tuple(color[70, 70])
        self.assertNotEqual(color1, color2)


class TestSavePropsCSV(unittest.TestCase):
    """Tests for save_props_csv()."""

    def test_writes_csv_file(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        cv2.circle(labels, (25, 25), 10, 1, -1)
        props = measure_labels(labels)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name

        try:
            save_props_csv(path, props)
            self.assertTrue(os.path.exists(path))

            with open(path, 'r') as f:
                content = f.read()
                self.assertIn("area_px", content)
                self.assertIn("eq_diam_px", content)
        finally:
            os.unlink(path)

    def test_empty_props_writes_header(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name

        try:
            save_props_csv(path, [])
            self.assertTrue(os.path.exists(path))

            with open(path, 'r') as f:
                content = f.read()
                self.assertIn("area_px", content)
        finally:
            os.unlink(path)


class TestMaskFromLabels(unittest.TestCase):
    """Tests for mask_from_labels()."""

    def test_returns_uint8_binary(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        cv2.circle(labels, (25, 25), 10, 1, -1)

        mask = mask_from_labels(labels)

        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(set(np.unique(mask)).issubset({0, 255}))

    def test_foreground_is_255(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:20, 10:20] = 5

        mask = mask_from_labels(labels)

        self.assertEqual(mask[15, 15], 255)
        self.assertEqual(mask[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
