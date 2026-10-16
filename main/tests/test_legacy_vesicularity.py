from __future__ import annotations

import json
import math
import unittest
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from main.core.legacy_areas import calculate_foams105_image_areas
from main.core.legacy_measurements import measure_foams105_image
from main.core.legacy_raw_selection import (
    Foams105RawGroupInput,
    Foams105RawObject,
    select_foams105_raw_objects,
)
from main.core.legacy_selection import Foams105BinnedRange
from main.core.legacy_vesicularity import (
    DIAGNOSTIC_NEGATIVE_RESIDUAL,
    DIAGNOSTIC_PERCENTAGE_OUT_OF_RANGE,
    DIAGNOSTIC_SELECTED_AREA_EXCEEDS_SOURCE_DENOMINATOR,
    DIAGNOSTIC_SOURCE_AREA_REASSIGNED,
    FOAMS105_GROUP_SUMMARY_DENOMINATOR_POLICY,
    FOAMS105_INTEGRATED_DENOMINATOR_POLICY,
    FOAMS105_UNFILTERED_POPULATION_POLICY,
    FOAMS105_VESICULARITY_METHOD,
    FOAMS105_VESICULARITY_SCOPE,
    Foams105VesicularityDomainError,
    Foams105VesicularityGroupInput,
    Foams105VesicularityNumericalError,
    Foams105VesicularityValidationError,
    _calculate_foams105_vesicularity_numerical,
    _checked_divide,
    calculate_foams105_2d_vesicularity,
)

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "13_foams105_vesicularity_cases.json"
)
FIXTURE_LF_SHA256 = "2911877ac5758bc76b98ba0a6df60af052cc4d953d16a9604d86de9f4b64420a"


def _assert_close_sequence(test_case, actual, expected):
    test_case.assertEqual(len(actual), len(expected))
    for candidate, reference in zip(actual, expected):
        test_case.assertTrue(
            math.isclose(candidate, reference, rel_tol=1e-12, abs_tol=1e-12),
            f"{candidate!r} != {reference!r}",
        )


def _context_from_images(group_id, source_slot, scale, images, exclusions=None):
    if exclusions is None:
        exclusions = (None, None, None, None)
    area_results = []
    pore_objects = []
    for image_index, image in enumerate(images, start=1):
        image_id = f"{group_id}-image-{image_index}"
        measurement = measure_foams105_image(image, 1, scale, exclusions)
        area_results.append(
            calculate_foams105_image_areas(image_id, measurement)
        )
        pore_objects.extend(
            Foams105RawObject(
                image_id,
                component.component_id,
                component.equivalent_diameter_mm,
                component.area_mm2,
            )
            for component in measurement.pore_result.components
        )
    return Foams105VesicularityGroupInput(
        group_id, source_slot, tuple(area_results), tuple(pore_objects)
    )


def _raw_selection(contexts, raw_objects_by_group=None, active_count=None):
    if raw_objects_by_group is None:
        raw_objects_by_group = tuple(
            context.unfiltered_pore_objects for context in contexts
        )
    if active_count is None:
        active_count = len(contexts)
    groups = tuple(
        Foams105RawGroupInput(
            context.group_id, context.source_slot, tuple(raw_objects)
        )
        for context, raw_objects in zip(contexts, raw_objects_by_group)
    )
    ranges = tuple(
        Foams105BinnedRange(context.group_id, 1.0 if index == 0 else 0.1, 10.0)
        if index < active_count
        else None
        for index, context in enumerate(contexts)
    )
    return select_foams105_raw_objects((0.1, 1.0, 10.0), groups, ranges)


class TestFoams105VesicularityNumericalFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_all_thirteen_independent_numerical_cases(self):
        self.assertEqual(len(self.fixture["cases"]), 13)
        successful = 0
        expected_rejections = 0
        for case in self.fixture["cases"]:
            with self.subTest(case=case["id"]):
                try:
                    result = _calculate_foams105_vesicularity_numerical(
                        case["scales"],
                        case["image_area2_by_group"],
                        case["selected_area_sums"],
                        case["unfiltered_area_sums"],
                    )
                except (
                    Foams105VesicularityDomainError,
                    Foams105VesicularityNumericalError,
                ) as exc:
                    self.assertEqual(exc.code, case.get("expected_error"))
                    expected_rejections += 1
                    continue

                self.assertNotIn("expected_error", case)
                _assert_close_sequence(
                    self,
                    result.source_sorted_area2_by_slot_mm2,
                    case["expected_source_denominators"],
                )
                _assert_close_sequence(
                    self,
                    result.residuals_reference_area_mm2,
                    case["expected_residuals"],
                )
                self.assertTrue(
                    math.isclose(
                        result.integrated_vesicularity_percent,
                        case["expected_integrated_percent"],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
                _assert_close_sequence(
                    self,
                    result.group_vesicularity_percent,
                    case["expected_group_percent"],
                )
                self.assertTrue(
                    math.isclose(
                        result.mean_group_vesicularity_percent,
                        case["expected_mean_percent"],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
                self.assertEqual(
                    result.reassigned_slots,
                    tuple(case.get("expected_reassignment_slots", ())),
                )
                if case.get("expected_nonphysical"):
                    codes = tuple(diagnostic.code for diagnostic in result.diagnostics)
                    self.assertIn(DIAGNOSTIC_NEGATIVE_RESIDUAL, codes)
                    self.assertIn(
                        DIAGNOSTIC_SELECTED_AREA_EXCEEDS_SOURCE_DENOMINATOR, codes
                    )
                    self.assertIn(DIAGNOSTIC_PERCENTAGE_OUT_OF_RANGE, codes)
                successful += 1
        self.assertEqual((successful, expected_rejections), (10, 3))

    def test_fixture_normalized_text_hash_is_pinned(self):
        normalized = FIXTURE_PATH.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(sha256(normalized).hexdigest(), FIXTURE_LF_SHA256)

    def test_nonzero_division_underflow_is_contextual(self):
        with self.assertRaises(Foams105VesicularityNumericalError) as caught:
            _checked_divide(
                5e-324,
                1e308,
                stage="test_division",
                source_slot=2,
                code="nonfinite_recurrence",
            )
        self.assertEqual(caught.exception.code, "nonzero_underflow")
        self.assertEqual(caught.exception.context["source_slot"], 2)

    def test_nonzero_recurrence_multiplication_underflow_is_rejected(self):
        with self.assertRaises(Foams105VesicularityNumericalError) as caught:
            _calculate_foams105_vesicularity_numerical(
                (1.0, 2.0),
                ((1e-308,), (1e-308,)),
                (5e-309, 5e-324),
                (0.0, 0.0),
            )
        self.assertEqual(caught.exception.code, "nonzero_underflow")
        self.assertEqual(caught.exception.context["stage"], "recurrence_removed_area")


class TestFoams105VesicularityPublicIntegration(unittest.TestCase):
    def test_area2_selected_population_and_unfiltered_population_stay_distinct(self):
        image = np.zeros((12, 12), dtype=np.uint8)
        image[2, 2] = 1
        image[5:7, 5:7] = 1
        image[2:5, 8:11] = 2
        context = _context_from_images("only", 1, 1.0, (image,), (2, None, None, None))
        raw = _raw_selection((context,), ((context.unfiltered_pore_objects[1],),))

        result = calculate_foams105_2d_vesicularity(raw, (context,))

        trace = result.group_traces[0]
        self.assertEqual(trace.source_sorted_area2_mm2, 144.0)
        self.assertEqual(trace.actual_group_area2_mm2, 144.0)
        self.assertEqual(trace.phase_corrected_area1_mm2, 135.0)
        self.assertEqual(trace.selected_area_sum_mm2, 4.0)
        self.assertEqual(trace.unfiltered_area_sum_mm2, 5.0)
        self.assertTrue(
            math.isclose(
                result.integrated_vesicularity_percent,
                100.0 * 4.0 / 144.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        _assert_close_sequence(
            self, result.group_vesicularity_percent, (100.0 * 5.0 / 144.0,)
        )
        self.assertEqual(trace.selected_count, 1)
        self.assertEqual(trace.unfiltered_count, 2)

    def test_source_sort_reassigns_areas_but_group_summary_uses_membership(self):
        coarse = np.zeros((4, 4), dtype=np.uint8)
        coarse[1, 1] = 1
        fine = np.zeros((20, 20), dtype=np.uint8)
        fine[5, 5] = 1
        contexts = (
            _context_from_images("coarse", 1, 1.0, (coarse,)),
            _context_from_images("fine", 2, 2.0, (fine,)),
        )

        result = calculate_foams105_2d_vesicularity(
            _raw_selection(contexts), contexts
        )

        self.assertEqual(result.source_sorted_area2_by_slot_mm2, (100.0, 16.0))
        self.assertEqual(result.actual_group_area2_by_slot_mm2, (16.0, 100.0))
        self.assertEqual(result.group_vesicularity_percent, (6.25, 0.25))
        self.assertEqual(
            tuple(
                assignment.original_image_id
                for assignment in result.source_area_assignments
            ),
            ("fine-image-1", "coarse-image-1"),
        )
        reassignment = tuple(
            diagnostic
            for diagnostic in result.diagnostics
            if diagnostic.code == DIAGNOSTIC_SOURCE_AREA_REASSIGNED
        )
        self.assertEqual(tuple(item.source_slot for item in reassignment), (1, 2))
        self.assertEqual(reassignment[0].image_ids, ("coarse-image-1",))

    def test_multiple_images_pool_area_before_arithmetic_group_mean(self):
        first = np.zeros((6, 10), dtype=np.uint8)
        first[2, 2] = 1
        second = np.zeros((4, 10), dtype=np.uint8)
        second[2, 2] = 1
        fine = np.zeros((10, 10), dtype=np.uint8)
        fine[3, 3] = 1
        contexts = (
            _context_from_images("pooled", 1, 1.0, (first, second)),
            _context_from_images("fine", 2, 2.0, (fine,)),
        )
        result = calculate_foams105_2d_vesicularity(
            _raw_selection(contexts), contexts
        )
        self.assertEqual(result.actual_group_area2_by_slot_mm2, (100.0, 25.0))
        self.assertEqual(result.group_vesicularity_percent, (2.0, 1.0))
        self.assertEqual(result.mean_group_vesicularity_percent, 1.5)

    def test_disabled_trailing_group_still_has_image_summary(self):
        images = []
        for size in (10, 10):
            image = np.zeros((size, size), dtype=np.uint8)
            image[2, 2] = 1
            images.append(image)
        contexts = (
            _context_from_images("active", 1, 1.0, (images[0],)),
            _context_from_images("disabled", 2, 2.0, (images[1],)),
        )
        raw = _raw_selection(contexts, active_count=1)
        result = calculate_foams105_2d_vesicularity(raw, contexts)
        self.assertEqual(result.selected_area_by_slot_mm2, (1.0, 0.0))
        self.assertEqual(result.unfiltered_area_by_slot_mm2, (1.0, 0.25))
        self.assertEqual(result.group_traces[1].selected_count, 0)
        self.assertEqual(result.group_traces[1].unfiltered_count, 1)

    def test_equal_area_sort_ties_retain_flattened_image_order(self):
        first = np.zeros((5, 5), dtype=np.uint8)
        first[2, 2] = 1
        second = np.zeros((10, 10), dtype=np.uint8)
        second[3, 3] = 1
        contexts = (
            _context_from_images("first", 1, 1.0, (first,)),
            _context_from_images("second", 2, 2.0, (second,)),
        )
        result = calculate_foams105_2d_vesicularity(
            _raw_selection(contexts), contexts
        )
        self.assertEqual(result.sorted_area2_mm2, (25.0, 25.0))
        self.assertEqual(
            tuple(item.original_image_id for item in result.source_area_assignments),
            ("first-image-1", "second-image-1"),
        )

    def test_result_is_immutable_and_records_source_policies(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        image[2, 2] = 1
        context = _context_from_images("only", 1, 1.0, (image,))
        result = calculate_foams105_2d_vesicularity(
            _raw_selection((context,)), (context,)
        )
        self.assertEqual(result.method, FOAMS105_VESICULARITY_METHOD)
        self.assertEqual(result.scope, FOAMS105_VESICULARITY_SCOPE)
        self.assertEqual(
            result.integrated_denominator_policy,
            FOAMS105_INTEGRATED_DENOMINATOR_POLICY,
        )
        self.assertEqual(
            result.group_summary_denominator_policy,
            FOAMS105_GROUP_SUMMARY_DENOMINATOR_POLICY,
        )
        self.assertEqual(
            result.unfiltered_population_policy,
            FOAMS105_UNFILTERED_POPULATION_POLICY,
        )
        self.assertEqual(result.raw_selection, _raw_selection((context,)))
        with self.assertRaises(FrozenInstanceError):
            result.integrated_vesicularity_percent = 0.0


class TestFoams105VesicularityValidation(unittest.TestCase):
    def setUp(self):
        image = np.zeros((8, 8), dtype=np.uint8)
        image[2, 2] = 1
        self.context = _context_from_images("only", 1, 1.0, (image,))
        self.raw = _raw_selection((self.context,))

    def test_altered_raw_selection_result_is_rejected(self):
        tampered = replace(self.raw, selected_count=self.raw.selected_count + 1)
        with self.assertRaises(Foams105VesicularityValidationError) as caught:
            calculate_foams105_2d_vesicularity(tampered, (self.context,))
        self.assertEqual(caught.exception.code, "inconsistent_raw_selection")

    def test_altered_area_provenance_is_rejected(self):
        image = replace(self.context.image_area_results[0], method="other")
        context = replace(self.context, image_area_results=(image,))
        with self.assertRaises(Foams105VesicularityValidationError) as caught:
            calculate_foams105_2d_vesicularity(self.raw, (context,))
        self.assertEqual(caught.exception.code, "inconsistent_area_provenance")

    def test_nonpositive_area2_is_rejected_with_image_context(self):
        image = replace(
            self.context.image_area_results[0],
            border_corrected_area_mm2=0.0,
            border_corrected_area_is_positive=False,
        )
        context = replace(self.context, image_area_results=(image,))
        with self.assertRaises(Foams105VesicularityDomainError) as caught:
            calculate_foams105_2d_vesicularity(self.raw, (context,))
        self.assertEqual(caught.exception.code, "nonpositive_border_corrected_area")
        self.assertEqual(caught.exception.context["image_id"], "only-image-1")

    def test_foreign_object_image_is_rejected(self):
        source = self.context.unfiltered_pore_objects[0]
        context = replace(
            self.context,
            unfiltered_pore_objects=(replace(source, image_id="foreign"),),
        )
        with self.assertRaises(Foams105VesicularityValidationError) as caught:
            calculate_foams105_2d_vesicularity(self.raw, (context,))
        self.assertEqual(caught.exception.code, "foreign_image_membership")

    def test_changed_object_geometry_is_rejected(self):
        source = self.context.unfiltered_pore_objects[0]
        context = replace(
            self.context,
            unfiltered_pore_objects=(replace(source, area_mm2=source.area_mm2 * 2.0),),
        )
        with self.assertRaises(Foams105VesicularityValidationError) as caught:
            calculate_foams105_2d_vesicularity(self.raw, (context,))
        self.assertEqual(caught.exception.code, "changed_object_geometry")

    def test_group_identity_and_owned_tuple_contracts_are_rejected(self):
        malformed = (
            replace(self.context, group_id="other"),
            replace(self.context, source_slot=2),
            replace(self.context, image_area_results=list(self.context.image_area_results)),
            replace(
                self.context,
                unfiltered_pore_objects=list(self.context.unfiltered_pore_objects),
            ),
        )
        for context in malformed:
            with self.subTest(context=context):
                with self.assertRaises(Foams105VesicularityValidationError):
                    calculate_foams105_2d_vesicularity(self.raw, (context,))

    def test_source_slot_rejects_booleans_and_nonintegers(self):
        for source_slot in (True, np.bool_(True), 1.0):
            context = replace(self.context, source_slot=source_slot)
            with self.subTest(source_slot=source_slot), self.assertRaises(
                Foams105VesicularityValidationError
            ) as caught:
                calculate_foams105_2d_vesicularity(self.raw, (context,))
            self.assertIn("source_slot must be an integer", str(caught.exception))

    def test_source_slot_accepts_and_normalizes_numpy_integer(self):
        context = replace(self.context, source_slot=np.int64(1))
        result = calculate_foams105_2d_vesicularity(self.raw, (context,))
        self.assertEqual(result.groups[0].source_slot, 1)
        self.assertIsInstance(result.groups[0].source_slot, int)

    def test_malformed_scalar_is_typed(self):
        image = replace(self.context.image_area_results[0], scale_px_per_mm=True)
        context = replace(self.context, image_area_results=(image,))
        with self.assertRaises(Foams105VesicularityValidationError):
            calculate_foams105_2d_vesicularity(self.raw, (context,))


if __name__ == "__main__":
    unittest.main()
