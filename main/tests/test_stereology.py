# main/tests/test_stereology.py
# Unit tests for core/stereology.py: measurements, colorization, CSV export

import math
import os
import sys
import tempfile
import unittest
from dataclasses import fields

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


def measure_labels_full_frame_reference(labels, image_index=0, scale=None):
    """Reviewed-commit full-frame implementation retained for equivalence tests."""
    height, width = labels.shape
    units_per_px = None
    unit_name = None
    if isinstance(scale, dict) and "unitsPerPx" in scale:
        try:
            units_per_px = float(scale["unitsPerPx"])
        except Exception:
            units_per_px = None
        unit_name = str(scale.get("unitName", "") or "")
    result = []
    scratch = np.zeros_like(labels, dtype=np.uint8)
    for label in np.unique(labels):
        if label <= 0:
            continue
        np.equal(labels, label, out=scratch)
        area = int(np.count_nonzero(scratch))
        contours, _ = cv2.findContours(
            scratch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        moments = cv2.moments(scratch, binaryImage=True)
        centroid_x = float(moments["m10"] / moments["m00"])
        centroid_y = float(moments["m01"] / moments["m00"])
        ys, xs = np.nonzero(scratch)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        touches_border = x0 == 0 or y0 == 0 or x1 >= width or y1 >= height
        equivalent_diameter = math.sqrt(4.0 * area / math.pi)
        circularity = (
            4.0 * math.pi * area / (perimeter * perimeter)
            if perimeter > 0 else float("nan")
        )

        major = minor = orientation = None
        flat_points = None
        if contours:
            flat_points = np.vstack(contours).squeeze(1)
            if flat_points.shape[0] >= 5:
                _, (ellipse_width, ellipse_height), angle = cv2.fitEllipse(flat_points)
                if ellipse_width >= ellipse_height:
                    major, minor = float(ellipse_width), float(ellipse_height)
                    orientation = float(angle)
                else:
                    major, minor = float(ellipse_height), float(ellipse_width)
                    orientation = float((angle + 90.0) % 180.0)
            else:
                points = flat_points.astype(np.float32)
                _, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=None)
                try:
                    diameter1 = 4.0 * float(np.sqrt(max(eigenvalues[0, 0], 0.0)))
                    diameter2 = 4.0 * float(np.sqrt(max(eigenvalues[1, 0], 0.0)))
                    major, minor = max(diameter1, diameter2), min(diameter1, diameter2)
                    vector_x, vector_y = eigenvectors[0]
                    orientation = float(
                        (math.degrees(math.atan2(vector_y, vector_x)) + 360.0) % 180.0
                    )
                except Exception:
                    major = minor = orientation = None

        aspect = float(major / minor) if major and minor and minor > 0 else None
        feret_max = feret_min = None
        if flat_points is not None and flat_points.shape[0] >= 2:
            hull_points = cv2.convexHull(flat_points).reshape(-1, 2)
            rectangle = cv2.minAreaRect(hull_points)
            rectangle_width, rectangle_height = rectangle[1]
            feret_min = float(min(rectangle_width, rectangle_height))
            sampled_hull = (
                hull_points[np.linspace(0, len(hull_points) - 1, 600, dtype=int)]
                if len(hull_points) > 1200 else hull_points
            )
            differences = sampled_hull[None, :, :] - sampled_hull[:, None, :]
            squared = differences[:, :, 0] ** 2 + differences[:, :, 1] ** 2
            feret_max = float(np.sqrt(squared.max()))

        prop = PoreProps(
            image_index, int(label), area, perimeter, centroid_x, centroid_y,
            x0, y0, x1, y1, bool(touches_border), equivalent_diameter,
            circularity, major, minor, aspect, orientation, feret_max, feret_min,
            units_per_px=units_per_px, unit_name=unit_name,
        )
        if units_per_px and units_per_px > 0:
            prop.area_units2 = area * units_per_px**2
            prop.eq_diam_units = prop.eq_diam_px * units_per_px
            if prop.major_axis_px:
                prop.major_axis_units = prop.major_axis_px * units_per_px
            if prop.minor_axis_px:
                prop.minor_axis_units = prop.minor_axis_px * units_per_px
            if prop.feret_max_px:
                prop.feret_max_units = prop.feret_max_px * units_per_px
            if prop.feret_min_px:
                prop.feret_min_units = prop.feret_min_px * units_per_px
        result.append(prop)
    return result


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

    def test_sparse_ids_match_full_frame_geometry_reference(self):
        labels = np.zeros((80, 120), dtype=np.int32)
        cv2.circle(labels, (25, 30), 9, 7, -1)
        cv2.rectangle(labels, (70, 10), (90, 25), 1_000_003, -1)
        cv2.rectangle(labels, (95, 50), (105, 60), 1_000_003, -1)

        measured = {prop.label: prop for prop in measure_labels(labels)}
        self.assertEqual(set(measured), {7, 1_000_003})
        for label, prop in measured.items():
            mask = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            moments = cv2.moments(mask, binaryImage=True)
            ys, xs = np.nonzero(mask)
            self.assertEqual(prop.area_px, int(np.count_nonzero(mask)))
            self.assertAlmostEqual(
                prop.perimeter_px,
                sum(cv2.arcLength(contour, True) for contour in contours),
            )
            self.assertAlmostEqual(prop.centroid_x, moments["m10"] / moments["m00"])
            self.assertAlmostEqual(prop.centroid_y, moments["m01"] / moments["m00"])
            self.assertEqual(
                (prop.bbox_x0, prop.bbox_y0, prop.bbox_x1, prop.bbox_y1),
                (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            )

    def test_all_fields_match_reviewed_full_frame_implementation(self):
        labels = np.zeros((180, 220), dtype=np.int32)
        cv2.ellipse(labels, (60, 65), (28, 13), 27, 0, 360, 7, -1)
        cv2.circle(labels, (60, 65), 5, 0, -1)
        cv2.rectangle(labels, (130, 25), (150, 45), 1_000_003, -1)
        cv2.rectangle(labels, (175, 80), (205, 95), 1_000_003, -1)
        cv2.rectangle(labels, (0, 130), (18, 160), 42, -1)
        scale = {"unitsPerPx": 0.025, "unitName": "mm"}

        expected = measure_labels_full_frame_reference(labels, image_index=4, scale=scale)
        actual = measure_labels(labels, image_index=4, scale=scale)

        self.assertEqual([prop.label for prop in actual], [prop.label for prop in expected])
        for actual_prop, expected_prop in zip(actual, expected):
            for field in fields(PoreProps):
                with self.subTest(label=actual_prop.label, field=field.name):
                    actual_value = getattr(actual_prop, field.name)
                    expected_value = getattr(expected_prop, field.name)
                    if isinstance(expected_value, float):
                        self.assertAlmostEqual(actual_value, expected_value, places=5)
                    else:
                        self.assertEqual(actual_value, expected_value)


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

    def test_sparse_ids_preserve_legacy_palette(self):
        labels = np.array([[0, 7, 7], [1_000_003, 0, 7]], dtype=np.int32)
        unique = np.unique(labels)
        unique = unique[unique > 0]
        rng = np.random.default_rng(123)
        hues = np.linspace(0, 179, num=len(unique), endpoint=False).astype(np.uint8)
        rng.shuffle(hues)
        hsv = np.stack(
            [hues, np.full_like(hues, 200), np.full_like(hues, 255)], axis=1
        ).reshape(-1, 1, 3)
        colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
        expected = np.zeros((2, 3, 3), dtype=np.uint8)
        for index, label in enumerate(unique):
            expected[labels == label] = colors[index]

        np.testing.assert_array_equal(colorize_labels(labels, seed=123), expected)


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
