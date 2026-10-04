from __future__ import annotations

import json
import math
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np

from main.core.legacy_areas import (
    DIAGNOSTIC_BORDER_AREA_NONPOSITIVE,
    DIAGNOSTIC_CONSTITUENT_PHASE_AREA_NONPOSITIVE,
    DIAGNOSTIC_CONSTITUENT_PHASE_VALUE_COLLISION,
    DIAGNOSTIC_PHASE_AREA_NONPOSITIVE,
    DIAGNOSTIC_PHASE_VALUE_COLLISION,
    FOAMS105_AREA1_MAPPING,
    FOAMS105_AREA2_MAPPING,
    FOAMS105_AREA_SCOPE,
    FOAMS105_AREA_SOURCE_COMMIT,
    FOAMS105_BINARY_AREA_METHOD,
    FOAMS105_GROUP_AREA_METHOD,
    FOAMS105_IMAGE_AREA_METHOD,
    Foams105AreaNumericalError,
    Foams105AreaValidationError,
    aggregate_foams105_group_areas,
    calculate_foams105_image_areas,
    estimate_foams105_binary_area,
)
from main.core.legacy_measurements import measure_foams105_image

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "09_foams105_area_cases.json"


def _measure_case(case):
    image = np.asarray(case["image"], dtype=np.uint8)
    return measure_foams105_image(
        image,
        case["pore_value"],
        case["scale_px_per_mm"],
        tuple(case["excluded_phase_values"]),
    )


def _blank_measurement(shape=(5, 5), scale=1.0, exclusions=(None,) * 4):
    return measure_foams105_image(
        np.zeros(shape, dtype=np.uint8), 1, scale, exclusions
    )


class Foams105BinaryAreaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_fixture_weighted_areas(self):
        for case in self.fixture["weighted_area_cases"]:
            with self.subTest(case=case["id"]):
                result = estimate_foams105_binary_area(
                    np.asarray(case["mask"], dtype=np.uint8)
                )
                self.assertEqual(
                    result.weighted_area_px,
                    case["expected_weighted_area_px"],
                )

    def test_rotation_reflection_and_translation_invariance(self):
        mask = np.zeros((8, 9), dtype=np.uint8)
        mask[2:4, 3:6] = ((1, 1, 0), (1, 0, 1))
        expected = estimate_foams105_binary_area(mask).weighted_area_px
        transforms = (
            np.rot90(mask),
            np.rot90(mask, 2),
            np.rot90(mask, 3),
            np.fliplr(mask),
            np.flipud(mask),
        )
        for transformed in transforms:
            self.assertEqual(
                estimate_foams105_binary_area(transformed).weighted_area_px,
                expected,
            )
        translated = np.zeros_like(mask)
        translated[4:6, 1:4] = ((1, 1, 0), (1, 0, 1))
        self.assertEqual(
            estimate_foams105_binary_area(translated).weighted_area_px,
            expected,
        )

    def test_filled_rectangles_include_one_dimensional_shapes(self):
        for shape in ((1, 7), (6, 1), (3, 8)):
            with self.subTest(shape=shape):
                result = estimate_foams105_binary_area(np.ones(shape, dtype=bool))
                self.assertEqual(result.weighted_area_px, math.prod(shape))

    def test_binary_area_metadata_and_immutability(self):
        result = estimate_foams105_binary_area(np.ones((1, 1), dtype=np.uint8))
        self.assertEqual(result.method, FOAMS105_BINARY_AREA_METHOD)
        self.assertEqual(result.scope, FOAMS105_AREA_SCOPE)
        self.assertEqual(result.foreground_pixel_count, 1)
        with self.assertRaises(FrozenInstanceError):
            result.weighted_area_px = 2.0

    def test_binary_area_rejects_malformed_masks(self):
        malformed = (
            [[1]],
            np.asarray([], dtype=np.uint8),
            np.zeros((2, 0), dtype=np.uint8),
            np.zeros((1, 1, 1), dtype=np.uint8),
            np.asarray([[255]], dtype=np.uint8),
            np.asarray([[2]], dtype=np.int64),
            np.asarray([[np.nan]]),
            np.asarray([[np.inf]]),
            np.asarray([["1"]]),
        )
        for mask in malformed:
            with self.subTest(mask=repr(mask)):
                with self.assertRaises(Foams105AreaValidationError):
                    estimate_foams105_binary_area(mask)


class Foams105ImageAreaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_fixture_area_cases(self):
        for case in self.fixture["area_cases"]:
            with self.subTest(case=case["id"]):
                result = calculate_foams105_image_areas(
                    case["id"],
                    _measure_case(case),
                    tuple(case["excluded_min_area_mm2"]),
                )
                self.assertEqual(result.full_area_mm2, case["expected_full_area_mm2"])
                self.assertEqual(
                    result.border_weighted_area_px,
                    case["expected_border_weighted_area_px"],
                )
                self.assertEqual(result.border_area_mm2, case["expected_border_area_mm2"])
                self.assertEqual(
                    result.excluded_area_mm2_by_slot,
                    tuple(case["expected_excluded_area_mm2_by_slot"]),
                )
                self.assertEqual(
                    result.border_corrected_area_mm2, case["expected_area2_mm2"]
                )
                self.assertEqual(
                    result.phase_corrected_area_mm2, case["expected_area1_mm2"]
                )

    def test_threshold_is_strict_at_adjacent_float_values(self):
        case = self.fixture["area_cases"][0]
        measurement = _measure_case(case)
        area = 0.25
        below = math.nextafter(area, -math.inf)
        above = math.nextafter(area, math.inf)
        selected = calculate_foams105_image_areas(
            "below", measurement, (below, 0, 0, 0)
        )
        equal = calculate_foams105_image_areas(
            "equal", measurement, (area, 0, 0, 0)
        )
        rejected = calculate_foams105_image_areas(
            "above", measurement, (above, 0, 0, 0)
        )
        self.assertEqual(selected.excluded_slots[0].selected_component_ids, (1,))
        self.assertEqual(equal.excluded_slots[0].rejected_component_ids, (1,))
        self.assertEqual(rejected.excluded_slots[0].rejected_component_ids, (1,))

    def test_disabled_and_enabled_empty_slots_are_distinct(self):
        measurement = _blank_measurement(exclusions=(None, 2, None, None))
        result = calculate_foams105_image_areas("slots", measurement)
        self.assertFalse(result.excluded_slots[0].enabled)
        self.assertTrue(result.excluded_slots[1].enabled)
        self.assertEqual(result.excluded_slots[1].selected_component_ids, ())
        self.assertEqual(result.excluded_area_mm2_by_slot, (0.0, 0.0, 0.0, 0.0))

    def test_colliding_exclusions_are_subtracted_per_slot(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[1:6, 1:6] = 2
        measurement = measure_foams105_image(image, 1, 1.0, (2, 2, None, None))
        result = calculate_foams105_image_areas("collision", measurement)
        self.assertEqual(result.excluded_area_mm2_by_slot, (25.0, 25.0, 0.0, 0.0))
        self.assertEqual(result.phase_corrected_area_mm2, -1.0)
        self.assertFalse(result.phase_corrected_area_is_positive)
        self.assertIn(DIAGNOSTIC_PHASE_AREA_NONPOSITIVE, result.diagnostic_codes)
        self.assertIn(DIAGNOSTIC_PHASE_VALUE_COLLISION, result.diagnostic_codes)

    def test_all_pore_image_preserves_zero_border_denominator(self):
        image = np.ones((5, 5), dtype=np.uint8)
        measurement = measure_foams105_image(image, 1, 1.0)
        result = calculate_foams105_image_areas("all-pore", measurement)
        self.assertEqual(result.full_area_mm2, 25.0)
        self.assertEqual(result.border_area_mm2, 25.0)
        self.assertEqual(result.border_corrected_area_mm2, 0.0)
        self.assertFalse(result.border_corrected_area_is_positive)
        self.assertIn(DIAGNOSTIC_BORDER_AREA_NONPOSITIVE, result.diagnostic_codes)

    def test_scale_covariance(self):
        case = self.fixture["area_cases"][0]
        image = np.asarray(case["image"], dtype=np.uint8)
        first = calculate_foams105_image_areas(
            "scale-2",
            measure_foams105_image(image, 1, 2.0, (2, None, None, None)),
        )
        second = calculate_foams105_image_areas(
            "scale-4",
            measure_foams105_image(image, 1, 4.0, (2, None, None, None)),
        )
        self.assertEqual(second.border_weighted_area_px, first.border_weighted_area_px)
        for first_value, second_value in (
            (first.full_area_mm2, second.full_area_mm2),
            (first.border_area_mm2, second.border_area_mm2),
            (first.total_excluded_area_mm2, second.total_excluded_area_mm2),
            (first.border_corrected_area_mm2, second.border_corrected_area_mm2),
            (first.phase_corrected_area_mm2, second.phase_corrected_area_mm2),
        ):
            self.assertEqual(second_value, first_value / 4.0)

    def test_image_area_metadata(self):
        result = calculate_foams105_image_areas("metadata", _blank_measurement())
        self.assertEqual(result.method, FOAMS105_IMAGE_AREA_METHOD)
        self.assertEqual(result.source_commit, FOAMS105_AREA_SOURCE_COMMIT)
        self.assertEqual(result.original_area1_mapping, FOAMS105_AREA1_MAPPING)
        self.assertEqual(result.original_area2_mapping, FOAMS105_AREA2_MAPPING)
        with self.assertRaises(FrozenInstanceError):
            result.image_id = "changed"

    def test_image_area_rejects_bad_ids_thresholds_and_wrappers(self):
        measurement = _blank_measurement()
        bad_thresholds = (
            (0, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, -1),
            (0, 0, 0, math.inf),
            (0, 0, 0, math.nan),
            (0, 0, 0, True),
            "0000",
        )
        for thresholds in bad_thresholds:
            with self.subTest(thresholds=thresholds):
                with self.assertRaises(Foams105AreaValidationError):
                    calculate_foams105_image_areas("bad", measurement, thresholds)
        for image_id in ("", "   ", None, 3):
            with self.assertRaises(Foams105AreaValidationError):
                calculate_foams105_image_areas(image_id, measurement)
        with self.assertRaises(Foams105AreaValidationError):
            calculate_foams105_image_areas("bad", object())
        inconsistent = replace(
            measurement,
            pore_result=replace(measurement.pore_result, image_shape=(4, 4)),
        )
        with self.assertRaises(Foams105AreaValidationError):
            calculate_foams105_image_areas("bad", inconsistent)

    def test_calibration_rejects_underflow_and_overflow(self):
        for scale in (np.finfo(float).max, np.nextafter(0.0, 1.0)):
            measurement = _blank_measurement(shape=(2, 2), scale=float(scale))
            with self.subTest(scale=scale):
                with self.assertRaises(Foams105AreaNumericalError):
                    calculate_foams105_image_areas("numerical", measurement)


class Foams105GroupAreaTests(unittest.TestCase):
    def test_group_sums_terms_in_supplied_order(self):
        first = calculate_foams105_image_areas("first", _blank_measurement((2, 3)))
        second = calculate_foams105_image_areas("second", _blank_measurement((4, 5)))
        result = aggregate_foams105_group_areas("group", (first, second))
        self.assertEqual(result.image_ids, ("first", "second"))
        self.assertEqual(result.full_area_mm2, 26.0)
        self.assertEqual(result.border_corrected_area_mm2, 26.0)
        self.assertEqual(result.phase_corrected_area_mm2, 26.0)
        self.assertEqual(result.method, FOAMS105_GROUP_AREA_METHOD)

    def test_group_preserves_constituent_diagnostics(self):
        good = calculate_foams105_image_areas(
            "good", _blank_measurement(exclusions=(2, 2, None, None))
        )
        image = np.zeros((7, 7), dtype=np.uint8)
        image[1:6, 1:6] = 2
        bad = calculate_foams105_image_areas(
            "bad", measure_foams105_image(image, 1, 1.0, (2, 2, None, None))
        )
        result = aggregate_foams105_group_areas("group", (good, bad))
        self.assertEqual(result.phase_corrected_area_mm2, 24.0)
        self.assertEqual(result.nonpositive_phase_corrected_image_ids, ("bad",))
        self.assertEqual(result.phase_collision_image_ids, ("good", "bad"))
        self.assertIn(
            DIAGNOSTIC_CONSTITUENT_PHASE_AREA_NONPOSITIVE,
            result.diagnostic_codes,
        )
        self.assertIn(
            DIAGNOSTIC_CONSTITUENT_PHASE_VALUE_COLLISION,
            result.diagnostic_codes,
        )

    def test_group_rejects_empty_duplicate_and_incompatible_inputs(self):
        first = calculate_foams105_image_areas("first", _blank_measurement())
        different_scale = calculate_foams105_image_areas(
            "scale", _blank_measurement(scale=2.0)
        )
        different_phases = calculate_foams105_image_areas(
            "phases", _blank_measurement(exclusions=(2, None, None, None))
        )
        different_thresholds = calculate_foams105_image_areas(
            "thresholds", _blank_measurement(), (1, 0, 0, 0)
        )
        invalid_groups = (
            (),
            (first, first),
            (first, different_scale),
            (first, different_phases),
            (first, different_thresholds),
            (first, object()),
            "bad",
        )
        for images in invalid_groups:
            with self.subTest(images=images):
                with self.assertRaises(Foams105AreaValidationError):
                    aggregate_foams105_group_areas("group", images)
        for group_id in ("", "  ", None, 1):
            with self.assertRaises(Foams105AreaValidationError):
                aggregate_foams105_group_areas(group_id, (first,))


if __name__ == "__main__":
    unittest.main()
