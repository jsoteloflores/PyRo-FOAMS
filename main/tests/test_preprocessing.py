# main/tests/test_preprocessing.py
# Unit tests for main/core/preprocessing.py - image I/O and crop utilities

import unittest
import numpy as np
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.preprocessing import (
    loadImage,
    clampRectToImage,
    rectToMargins,
    marginsToRect,
    cropWithRect,
    cropWithMargins,
    applyCropBatch,
)


class TestClampRectToImage(unittest.TestCase):
    """Tests for clampRectToImage function."""

    def test_valid_rect_unchanged(self):
        rect = (10, 10, 50, 50)
        result = clampRectToImage(rect, (100, 100))
        self.assertEqual(result, (10, 10, 50, 50))

    def test_clamps_negative_coords(self):
        rect = (-10, -5, 50, 50)
        result = clampRectToImage(rect, (100, 100))
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)

    def test_clamps_exceeding_coords(self):
        rect = (10, 10, 150, 120)
        result = clampRectToImage(rect, (100, 100))
        self.assertEqual(result[2], 100)
        self.assertEqual(result[3], 100)

    def test_returns_none_for_degenerate_rect(self):
        # x0 >= x1
        rect = (50, 10, 30, 50)
        result = clampRectToImage(rect, (100, 100))
        self.assertIsNone(result)

    def test_returns_none_for_zero_width(self):
        rect = (50, 10, 50, 50)
        result = clampRectToImage(rect, (100, 100))
        self.assertIsNone(result)

    def test_small_image_shape(self):
        rect = (0, 0, 10, 10)
        result = clampRectToImage(rect, (5, 5))
        self.assertEqual(result, (0, 0, 5, 5))


class TestRectToMargins(unittest.TestCase):
    """Tests for rectToMargins function."""

    def test_full_image_zero_margins(self):
        rect = (0, 0, 100, 100)
        margins = rectToMargins(rect, (100, 100))
        self.assertEqual(margins, (0, 0, 0, 0))

    def test_centered_crop(self):
        rect = (10, 20, 90, 80)
        margins = rectToMargins(rect, (100, 100))
        # left=10, top=20, right=100-90=10, bottom=100-80=20
        self.assertEqual(margins, (10, 20, 10, 20))

    def test_top_left_corner(self):
        rect = (0, 0, 50, 50)
        margins = rectToMargins(rect, (100, 100))
        self.assertEqual(margins, (0, 0, 50, 50))

    def test_bottom_right_corner(self):
        rect = (50, 50, 100, 100)
        margins = rectToMargins(rect, (100, 100))
        self.assertEqual(margins, (50, 50, 0, 0))


class TestMarginsToRect(unittest.TestCase):
    """Tests for marginsToRect function."""

    def test_zero_margins_full_image(self):
        margins = (0, 0, 0, 0)
        rect = marginsToRect(margins, (100, 100))
        self.assertEqual(rect, (0, 0, 100, 100))

    def test_symmetric_margins(self):
        margins = (10, 10, 10, 10)
        rect = marginsToRect(margins, (100, 100))
        self.assertEqual(rect, (10, 10, 90, 90))

    def test_asymmetric_margins(self):
        margins = (5, 10, 15, 20)
        rect = marginsToRect(margins, (100, 100))
        # x0=5, y0=10, x1=100-15=85, y1=100-20=80
        self.assertEqual(rect, (5, 10, 85, 80))

    def test_returns_none_for_excessive_margins(self):
        margins = (50, 50, 60, 60)  # overlap
        rect = marginsToRect(margins, (100, 100))
        self.assertIsNone(rect)

    def test_roundtrip_rect_margins_rect(self):
        original_rect = (15, 25, 85, 75)
        shape = (100, 100)
        margins = rectToMargins(original_rect, shape)
        recovered_rect = marginsToRect(margins, shape)
        self.assertEqual(recovered_rect, original_rect)


class TestCropWithRect(unittest.TestCase):
    """Tests for cropWithRect function."""

    def test_basic_crop(self):
        img = np.arange(100).reshape(10, 10).astype(np.uint8)
        cropped = cropWithRect(img, (2, 3, 7, 8))
        self.assertEqual(cropped.shape, (5, 5))

    def test_crop_returns_copy(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        cropped = cropWithRect(img, (10, 10, 50, 50))
        cropped[0, 0] = 255
        self.assertEqual(img[10, 10], 0)  # original unchanged

    def test_color_image_crop(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = cropWithRect(img, (0, 0, 50, 50))
        self.assertEqual(cropped.shape, (50, 50, 3))

    def test_full_image_crop(self):
        img = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
        cropped = cropWithRect(img, (0, 0, 60, 50))
        np.testing.assert_array_equal(cropped, img)


class TestCropWithMargins(unittest.TestCase):
    """Tests for cropWithMargins function."""

    def test_zero_margins_returns_full_image(self):
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        cropped = cropWithMargins(img, (0, 0, 0, 0))
        np.testing.assert_array_equal(cropped, img)

    def test_uniform_margins(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        cropped = cropWithMargins(img, (10, 10, 10, 10))
        self.assertEqual(cropped.shape, (80, 80))

    def test_returns_none_for_invalid_margins(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        cropped = cropWithMargins(img, (30, 30, 30, 30))  # overlapping
        self.assertIsNone(cropped)


class TestApplyCropBatch(unittest.TestCase):
    """Tests for applyCropBatch function."""

    def test_empty_list_returns_empty(self):
        result = applyCropBatch([], rect=(0, 0, 10, 10))
        self.assertEqual(result, [])

    def test_single_image_with_rect(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        result = applyCropBatch([img], rect=(10, 10, 50, 50), useMargins=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (40, 40))

    def test_multiple_images_same_size(self):
        imgs = [np.zeros((100, 100), dtype=np.uint8) for _ in range(3)]
        result = applyCropBatch(imgs, rect=(0, 0, 50, 50), useMargins=True)
        self.assertEqual(len(result), 3)
        for cropped in result:
            self.assertEqual(cropped.shape, (50, 50))

    def test_use_margins_true_adapts_to_size(self):
        # Different sized images with useMargins=True should adapt
        img1 = np.zeros((100, 100), dtype=np.uint8)
        img2 = np.zeros((200, 200), dtype=np.uint8)
        # Rect (10,10,90,90) on first image -> margins (10,10,10,10)
        # Applied to second image -> crop (10,10,190,190) -> 180x180
        result = applyCropBatch([img1, img2], rect=(10, 10, 90, 90), useMargins=True)
        self.assertEqual(result[0].shape, (80, 80))
        self.assertEqual(result[1].shape, (180, 180))

    def test_use_margins_false_clamps_per_image(self):
        img1 = np.zeros((100, 100), dtype=np.uint8)
        img2 = np.zeros((50, 50), dtype=np.uint8)
        # Rect (10,10,90,90) clamped on second image -> (10,10,50,50)
        result = applyCropBatch([img1, img2], rect=(10, 10, 90, 90), useMargins=False)
        self.assertEqual(result[0].shape, (80, 80))
        self.assertEqual(result[1].shape, (40, 40))

    def test_rect_clamped_to_valid_region(self):
        # clampRectToImage clamps coords to image bounds, doesn't return None
        # for out-of-bounds rects unless they become degenerate
        img = np.zeros((100, 100), dtype=np.uint8)
        # This rect (200,200,300,300) gets clamped to (99,99,100,100) = 1x1 pixel
        result = applyCropBatch([img], rect=(200, 200, 300, 300), useMargins=True)
        # Should return a small cropped image, not None
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0])

    def test_raises_without_rect_or_margins(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaises(ValueError):
            applyCropBatch([img])


class TestLoadImage(unittest.TestCase):
    """Tests for loadImage function (requires file I/O)."""

    def setUp(self):
        # Create a temp image file for testing
        import cv2
        self.temp_dir = tempfile.mkdtemp()
        self.test_gray_path = os.path.join(self.temp_dir, "test_gray.png")
        self.test_color_path = os.path.join(self.temp_dir, "test_color.png")
        
        gray_img = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
        color_img = np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8)
        
        cv2.imwrite(self.test_gray_path, gray_img)
        cv2.imwrite(self.test_color_path, color_img)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_grayscale(self):
        img = loadImage(self.test_gray_path, asGray=True)
        self.assertEqual(img.ndim, 2)
        self.assertEqual(img.dtype, np.uint8)

    def test_load_color(self):
        img = loadImage(self.test_color_path, asGray=False)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)
        self.assertEqual(img.dtype, np.uint8)

    def test_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            loadImage("/nonexistent/path/image.png")


if __name__ == "__main__":
    unittest.main()
