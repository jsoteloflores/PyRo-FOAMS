import json
import math
import os
import sys
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.legacy_measurements import (
    FOAMS105_BORDER_CONNECTIVITY,
    FOAMS105_COMPONENT_CONNECTIVITY,
    FOAMS105_COMPONENT_ORDER_POLICY,
    FOAMS105_HOLE_BACKGROUND_CONNECTIVITY,
    FOAMS105_MEASUREMENT_METHOD,
    FOAMS105_MEASUREMENT_SCOPE,
    FOAMS105_MEASUREMENT_SOURCE_COMMIT,
    FOAMS105_MEASUREMENT_SOURCE_FILE,
    FOAMS105_MEASUREMENT_SOURCE_REPOSITORY,
    Foams105MeasurementNumericalError,
    Foams105MeasurementValidationError,
    measure_foams105_image,
    measure_foams105_phase,
)

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "08_foams105_measurement_cases.json"
)


class TestFoams105MeasurementFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_PATH, encoding="utf-8") as stream:
            cls.fixture = json.load(stream)

    def test_all_seven_complete_topology_cases_match(self):
        self.assertEqual(len(self.fixture["cases"]), 7)
        for case in self.fixture["cases"]:
            with self.subTest(case=case["id"]):
                result = measure_foams105_phase(
                    np.asarray(case["image"]),
                    case["foreground_value"],
                    case["scale_px_per_mm"],
                )
                np.testing.assert_array_equal(
                    result.removed_border_mask,
                    np.asarray(case["expected_removed_border_mask"], dtype=bool),
                )
                np.testing.assert_array_equal(
                    result.filled_mask,
                    np.asarray(case["expected_filled_mask"], dtype=bool),
                )
                np.testing.assert_array_equal(
                    result.label_image,
                    np.asarray(case["expected_label_image"], dtype=np.int32),
                )
                self.assertEqual(
                    tuple(component.area_px for component in result.components),
                    tuple(case["expected_area_px"]),
                )
                for component, area_px in zip(
                    result.components, case["expected_area_px"]
                ):
                    expected_area = area_px / 4.0
                    expected_diameter = 2.0 * math.sqrt(expected_area / math.pi)
                    self.assertTrue(
                        math.isclose(
                            component.area_mm2,
                            expected_area,
                            rel_tol=1e-12,
                            abs_tol=0.0,
                        )
                    )
                    self.assertTrue(
                        math.isclose(
                            component.equivalent_diameter_mm,
                            expected_diameter,
                            rel_tol=1e-12,
                            abs_tol=0.0,
                        )
                    )

    def test_fixture_origin_and_source_are_explicit(self):
        self.assertEqual(
            self.fixture["origin"],
            "Hand-authored analytical masks; not MATLAB-generated or original-acquisition exports",
        )
        self.assertEqual(
            self.fixture["source_commit"], FOAMS105_MEASUREMENT_SOURCE_COMMIT
        )
        self.assertEqual(self.fixture["source_file"], "data_meas.m")


class TestFoams105Topology(unittest.TestCase):
    def test_all_foreground_and_one_dimensional_images_are_removed(self):
        for image in (
            np.ones((5, 5), dtype=np.uint8),
            np.ones((1, 7), dtype=np.uint8),
            np.ones((7, 1), dtype=np.uint8),
        ):
            with self.subTest(shape=image.shape):
                result = measure_foams105_phase(image, 1, 2)
                self.assertEqual(result.selected_pixel_count, image.size)
                self.assertEqual(result.removed_border_pixel_count, image.size)
                self.assertEqual(result.final_foreground_pixel_count, 0)
                self.assertEqual(result.components, ())

    def test_frame_connected_background_cavity_is_not_filled(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[1:6, 1] = 1
        image[1:6, 5] = 1
        image[5, 1:6] = 1
        image[1, 1:3] = 1
        image[1, 4:6] = 1
        result = measure_foams105_phase(image, 1, 1)
        self.assertFalse(result.filled_mask[3, 3])
        self.assertEqual(result.filled_added_pixel_count, 0)

    def test_noncontiguous_input_and_unselected_phase(self):
        source = np.zeros((12, 12), dtype=np.float64)
        source[4, 4] = 2.5
        image = source[::2, ::2]
        self.assertFalse(image.flags.c_contiguous)
        selected = measure_foams105_phase(image, 2.5, 2)
        empty = measure_foams105_phase(image, 9.0, 2)
        self.assertEqual(tuple(component.area_px for component in selected.components), (1,))
        self.assertEqual(empty.components, ())

    def test_zero_and_255_are_exact_values_without_polarity_inference(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[3, 3] = 255
        zero = measure_foams105_phase(image, 0, 1)
        high = measure_foams105_phase(image, 255, 1)
        self.assertEqual(zero.components, ())
        self.assertEqual(zero.removed_border_pixel_count, 48)
        self.assertEqual(tuple(component.area_px for component in high.components), (1,))

    def test_column_major_order_and_first_pixel_coordinates(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[1, 4] = 1
        image[4, 1] = 1
        result = measure_foams105_phase(image, 1, 1)
        self.assertEqual(
            tuple(component.first_pixel_rc for component in result.components),
            ((4, 1), (1, 4)),
        )
        self.assertEqual(result.label_image[4, 1], 1)
        self.assertEqual(result.label_image[1, 4], 2)

    def test_pixel_conservation_for_border_and_filled_cases(self):
        cases = []
        border = np.zeros((7, 7), dtype=np.uint8)
        border[0, 0] = border[1, 1] = border[2, 2] = 1
        cases.append(border)
        ring = np.zeros((7, 7), dtype=np.uint8)
        ring[1:6, 1] = ring[1:6, 5] = 1
        ring[1, 1:6] = ring[5, 1:6] = 1
        cases.append(ring)
        for image in cases:
            result = measure_foams105_phase(image, 1, 2)
            self.assertEqual(
                result.selected_pixel_count,
                result.removed_border_pixel_count
                + result.retained_before_fill_pixel_count,
            )
            self.assertEqual(
                result.final_foreground_pixel_count,
                result.retained_before_fill_pixel_count
                + result.filled_added_pixel_count,
            )
            self.assertEqual(
                result.final_foreground_pixel_count,
                sum(component.area_px for component in result.components),
            )


class TestFoams105CalibrationAndOwnership(unittest.TestCase):
    def test_scale_covariance_keeps_topology_and_pixel_counts(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[2:5, 2:5] = 1
        first = measure_foams105_phase(image, 1, 2)
        doubled = measure_foams105_phase(image, 1, 4)
        np.testing.assert_array_equal(first.filled_mask, doubled.filled_mask)
        np.testing.assert_array_equal(first.label_image, doubled.label_image)
        self.assertEqual(first.components[0].area_px, doubled.components[0].area_px)
        self.assertTrue(
            math.isclose(
                doubled.components[0].area_mm2,
                first.components[0].area_mm2 / 4.0,
                rel_tol=1e-15,
            )
        )
        self.assertTrue(
            math.isclose(
                doubled.components[0].equivalent_diameter_mm,
                first.components[0].equivalent_diameter_mm / 2.0,
                rel_tol=1e-15,
            )
        )

    def test_input_mutation_does_not_change_results_and_arrays_reject_writes(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        image[2, 2] = 1
        result = measure_foams105_phase(image, 1, 2)
        image[:, :] = 1
        self.assertEqual(result.final_foreground_pixel_count, 1)
        self.assertEqual(result.filled_mask[2, 2], 1)
        for field in ("removed_border_mask", "filled_mask", "label_image"):
            array = getattr(result, field)
            self.assertFalse(array.flags.writeable)
            with self.subTest(field=field), self.assertRaises(ValueError):
                array.flat[0] = array.flat[0]

    def test_results_records_and_metadata_are_frozen_and_stable(self):
        image = np.pad(np.ones((1, 1), dtype=np.uint8), 2)
        result = measure_foams105_phase(image, 1, 2)
        self.assertEqual(result.method, FOAMS105_MEASUREMENT_METHOD)
        self.assertEqual(result.scope, FOAMS105_MEASUREMENT_SCOPE)
        self.assertEqual(result.source_repository, FOAMS105_MEASUREMENT_SOURCE_REPOSITORY)
        self.assertEqual(result.source_commit, FOAMS105_MEASUREMENT_SOURCE_COMMIT)
        self.assertEqual(result.source_file, FOAMS105_MEASUREMENT_SOURCE_FILE)
        self.assertEqual(result.border_connectivity, FOAMS105_BORDER_CONNECTIVITY)
        self.assertEqual(result.hole_background_connectivity, FOAMS105_HOLE_BACKGROUND_CONNECTIVITY)
        self.assertEqual(result.component_connectivity, FOAMS105_COMPONENT_CONNECTIVITY)
        self.assertEqual(result.component_order_policy, FOAMS105_COMPONENT_ORDER_POLICY)
        with self.assertRaises(FrozenInstanceError):
            result.method = "changed"
        with self.assertRaises(FrozenInstanceError):
            result.components[0].area_px = 2

    def test_calibration_underflow_and_overflow_fail_contextually(self):
        image = np.pad(np.ones((1, 1), dtype=np.uint8), 1)
        for scale in (1e308, 5e-324):
            with self.subTest(scale=scale), self.assertRaisesRegex(
                Foams105MeasurementNumericalError, "component 1"
            ):
                measure_foams105_phase(image, 1, scale)


class TestFoams105ImageWrapper(unittest.TestCase):
    def test_disabled_enabled_empty_and_each_stable_exclusion_slot(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[3, 3] = 1
        image[2, 2] = 2
        result = measure_foams105_image(image, 1, 2, (2, 3, 4, None))
        self.assertEqual(len(result.excluded_phase_results), 4)
        self.assertEqual(result.excluded_phase_results[0].components[0].area_px, 1)
        self.assertEqual(result.excluded_phase_results[1].components, ())
        self.assertEqual(result.excluded_phase_results[2].components, ())
        self.assertIsNone(result.excluded_phase_results[3])
        self.assertEqual(result.excluded_phase_values, (2, 3, 4, None))

    def test_duplicate_and_pore_collisions_are_independent_and_reported(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[3, 3] = 1
        result = measure_foams105_image(image, 1, 2, (1, 2, 1, 2))
        self.assertEqual(
            result.phase_value_collision_slot_pairs,
            (
                ("pore", "excluded_phase_1"),
                ("pore", "excluded_phase_3"),
                ("excluded_phase_1", "excluded_phase_3"),
                ("excluded_phase_2", "excluded_phase_4"),
            ),
        )
        self.assertEqual(result.pore_result.final_foreground_pixel_count, 1)
        self.assertEqual(result.excluded_phase_results[0].final_foreground_pixel_count, 1)
        self.assertEqual(result.excluded_phase_results[2].final_foreground_pixel_count, 1)

    def test_excluded_ring_and_interior_fill_independently(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[1:6, 1] = image[1:6, 5] = 2
        image[1, 1:6] = image[5, 1:6] = 2
        image[3, 3] = 3
        result = measure_foams105_image(image, 1, 2, (2, 3, None, None))
        ring = result.excluded_phase_results[0]
        interior = result.excluded_phase_results[1]
        self.assertTrue(ring.filled_mask[3, 3])
        self.assertTrue(interior.filled_mask[3, 3])
        self.assertEqual(ring.components[0].area_px, 25)
        self.assertEqual(interior.components[0].area_px, 1)

    def test_wrapper_requires_exactly_four_slots(self):
        image = np.zeros((3, 3), dtype=np.uint8)
        for exclusions in ((), (None,), (None,) * 5, "none", None):
            with self.subTest(exclusions=exclusions), self.assertRaisesRegex(
                Foams105MeasurementValidationError, "exactly four"
            ):
                measure_foams105_image(image, 1, 2, exclusions)


class TestFoams105MeasurementValidation(unittest.TestCase):
    def test_invalid_images_fail_contextually(self):
        invalid = (
            None,
            [],
            np.array([], dtype=np.uint8),
            np.zeros((0, 2), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=object),
            np.zeros((2, 2), dtype="U1"),
            np.zeros((2, 2), dtype=np.complex128),
            np.array([[math.nan]]),
            np.array([[math.inf]]),
        )
        for image in invalid:
            with self.subTest(type=type(image).__name__), self.assertRaisesRegex(
                Foams105MeasurementValidationError, "image"
            ):
                measure_foams105_phase(image, 1, 2)

    def test_invalid_phase_values_and_scale_fail_contextually(self):
        image = np.zeros((3, 3), dtype=np.uint8)
        for phase in (True, "1", 1 + 0j, np.array(1), math.nan, math.inf, 10**400):
            with self.subTest(phase=phase), self.assertRaisesRegex(
                Foams105MeasurementValidationError, "phase_value"
            ):
                measure_foams105_phase(image, phase, 2)
        for scale in (True, "2", 1 + 0j, np.array(2), 0, -1, math.nan, math.inf, 10**400):
            with self.subTest(scale=scale), self.assertRaisesRegex(
                Foams105MeasurementValidationError, "scale_px_per_mm"
            ):
                measure_foams105_phase(image, 1, scale)

    def test_boolean_phase_values_are_supported_for_boolean_images(self):
        image = np.zeros((5, 5), dtype=bool)
        image[2, 2] = True
        result = measure_foams105_phase(image, True, 2)
        self.assertIs(result.phase_value, True)
        self.assertEqual(result.components[0].area_px, 1)


if __name__ == "__main__":
    unittest.main()
