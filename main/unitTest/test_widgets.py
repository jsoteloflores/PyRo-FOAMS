# main/unitTest/test_widgets.py
# Unit tests for widgets.py: dtype helpers, debounce

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from widgets import ensure_mask_uint8, ensure_labels_int32


class TestEnsureMaskUint8(unittest.TestCase):
    """Tests for ensure_mask_uint8() dtype normalization."""

    def test_bool_to_uint8(self):
        mask = np.array([[True, False], [False, True]], dtype=bool)
        result = ensure_mask_uint8(mask, (2, 2))

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result[0, 0], 255)
        self.assertEqual(result[0, 1], 0)

    def test_uint8_passthrough(self):
        mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)
        result = ensure_mask_uint8(mask, (2, 2))

        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result, mask)

    def test_uint8_binary_scaled(self):
        # uint8 with values 0/1 should be scaled to 0/255
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        result = ensure_mask_uint8(mask, (2, 2))

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result[0, 0], 255)
        self.assertEqual(result[0, 1], 0)

    def test_none_returns_zeros(self):
        result = ensure_mask_uint8(None, (5, 5))

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.sum(), 0)

    def test_resizes_if_shape_mismatch(self):
        mask = np.ones((10, 10), dtype=np.uint8) * 255
        result = ensure_mask_uint8(mask, (20, 20))

        self.assertEqual(result.shape, (20, 20))
        self.assertEqual(result.dtype, np.uint8)

    def test_int32_nonzero_to_255(self):
        mask = np.array([[0, 5], [10, 0]], dtype=np.int32)
        result = ensure_mask_uint8(mask, (2, 2))

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[0, 1], 255)
        self.assertEqual(result[1, 0], 255)

    def test_float_nonzero_to_255(self):
        mask = np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32)
        result = ensure_mask_uint8(mask, (2, 2))

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[0, 1], 255)
        self.assertEqual(result[1, 0], 255)


class TestEnsureLabelsInt32(unittest.TestCase):
    """Tests for ensure_labels_int32() dtype normalization."""

    def test_uint8_to_int32(self):
        labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        result = ensure_labels_int32(labels, (2, 2))

        self.assertEqual(result.dtype, np.int32)
        np.testing.assert_array_equal(result, labels.astype(np.int32))

    def test_int32_passthrough(self):
        labels = np.array([[0, 1], [2, 3]], dtype=np.int32)
        result = ensure_labels_int32(labels, (2, 2))

        self.assertEqual(result.dtype, np.int32)
        np.testing.assert_array_equal(result, labels)

    def test_none_returns_zeros(self):
        result = ensure_labels_int32(None, (5, 5))

        self.assertEqual(result.dtype, np.int32)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.sum(), 0)

    def test_resizes_if_shape_mismatch(self):
        labels = np.arange(100, dtype=np.int32).reshape(10, 10)
        result = ensure_labels_int32(labels, (20, 20))

        self.assertEqual(result.shape, (20, 20))
        self.assertEqual(result.dtype, np.int32)

    def test_float_to_int32(self):
        labels = np.array([[0.0, 1.5], [2.9, 3.0]], dtype=np.float64)
        result = ensure_labels_int32(labels, (2, 2))

        self.assertEqual(result.dtype, np.int32)
        # Floats truncated to int
        self.assertEqual(result[0, 1], 1)
        self.assertEqual(result[1, 0], 2)

    def test_int32_direct_passthrough(self):
        # int32 arrays should pass through directly
        labels = np.array([[0, 5], [10, 1]], dtype=np.int32)
        result = ensure_labels_int32(labels, (2, 2))

        self.assertEqual(result.dtype, np.int32)
        np.testing.assert_array_equal(result, labels)


class TestDebounce(unittest.TestCase):
    """Tests for debounce() decorator (basic behavior without Tk)."""

    def test_debounce_import(self):
        # Just verify we can import the debounce function
        from widgets import debounce
        self.assertTrue(callable(debounce))


if __name__ == "__main__":
    unittest.main()
