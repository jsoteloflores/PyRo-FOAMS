from __future__ import annotations

import json
import math
import unittest
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path

import numpy as np

from main.core.legacy_raw_selection import (
    FOAMS105_RAW_SELECTION_BOUNDARY_POLICY,
    FOAMS105_RAW_SELECTION_CONCATENATION_POLICY,
    FOAMS105_RAW_SELECTION_METHOD,
    FOAMS105_RAW_SELECTION_PREFIX_POLICY,
    FOAMS105_RAW_SELECTION_ROUNDING_PROFILE,
    FOAMS105_RAW_SELECTION_SCOPE,
    FOAMS105_RAW_SELECTION_SOURCE_COMMIT,
    Foams105RawGroupInput,
    Foams105RawObject,
    Foams105RawSelectionDomainError,
    Foams105RawSelectionNumericalError,
    Foams105RawSelectionValidationError,
    select_foams105_raw_objects,
)
from main.core.legacy_selection import Foams105BinnedRange
from main.core.legacy_size_classes import FOAMS105_ROUNDING_POLICY

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "12_foams105_raw_selection_cases.json"
FIXTURE_LF_SHA256 = "f1f778b01fa96346f18ada4134a125fe48c1d776bffbbc789c13b3793e6ea407"


def _object(image_id, component_id, diameter, area=None):
    if area is None:
        area = math.pi * float(diameter) ** 2 / 4.0
    return Foams105RawObject(image_id, component_id, diameter, area)


def _group(group_id, slot, diameters):
    return Foams105RawGroupInput(
        group_id,
        slot,
        tuple(
            _object(f"image-{slot}", index + 1, diameter)
            for index, diameter in enumerate(diameters)
        ),
    )


class TestFoams105RawSelectionFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_all_eleven_complete_source_rule_cases(self):
        self.assertEqual(len(self.fixture["cases"]), 11)
        successful = 0
        expected_rejections = 0
        for case in self.fixture["cases"]:
            groups = tuple(
                _group(f"slot-{slot}", slot, diameters)
                for slot, diameters in enumerate(case["diameters"], start=1)
            )
            ranges = tuple(
                None
                if bounds is None
                else Foams105BinnedRange(groups[index].group_id, *bounds)
                for index, bounds in enumerate(case["ranges"])
            )
            with self.subTest(case=case["id"]):
                if "expected_error" in case:
                    with self.assertRaises(Foams105RawSelectionDomainError) as caught:
                        select_foams105_raw_objects(case["labels"], groups, ranges)
                    self.assertEqual(caught.exception.code, case["expected_error"])
                    expected_rejections += 1
                    continue

                result = select_foams105_raw_objects(case["labels"], groups, ranges)
                expected_indices = tuple(
                    tuple(indices) for indices in case["expected_indices"]
                )
                self.assertEqual(
                    tuple(trace.selected_indices for trace in result.group_traces),
                    expected_indices,
                )
                self.assertEqual(
                    tuple(trace.effective_lower_mm for trace in result.group_traces),
                    tuple(case["effective_lowers"]),
                )
                for trace, group in zip(result.group_traces, result.groups):
                    self.assertEqual(
                        set(trace.selected_indices) | set(trace.rejected_indices),
                        set(range(len(group.objects))),
                    )
                    self.assertFalse(
                        set(trace.selected_indices) & set(trace.rejected_indices)
                    )
                    self.assertEqual(
                        trace.selected_objects,
                        tuple(group.objects[index] for index in trace.selected_indices),
                    )
                expected_rows = tuple(
                    (slot, source_index)
                    for slot, indices in enumerate(expected_indices, start=1)
                    for source_index in indices
                )
                self.assertEqual(
                    tuple(
                        (row.source_slot, row.source_object_index)
                        for row in result.rows
                    ),
                    expected_rows,
                )
                self.assertEqual(
                    result.equivalent_diameters_mm,
                    tuple(row.equivalent_diameter_mm for row in result.rows),
                )
                self.assertEqual(
                    result.areas_mm2, tuple(row.area_mm2 for row in result.rows)
                )
                self.assertEqual(
                    result.selected_identities,
                    tuple((row.image_id, row.component_id) for row in result.rows),
                )
                successful += 1
        self.assertEqual((successful, expected_rejections), (7, 4))

    def test_fixture_normalized_text_hash_is_pinned(self):
        normalized = FIXTURE_PATH.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(sha256(normalized).hexdigest(), FIXTURE_LF_SHA256)


class TestFoams105RawSelectionBoundaries(unittest.TestCase):
    def test_exact_and_adjacent_endpoints_are_inclusive_without_tolerance(self):
        lower = 1.0
        upper = 2.0
        diameters = (
            math.nextafter(lower, 0.0),
            lower,
            math.nextafter(lower, math.inf),
            math.nextafter(upper, 0.0),
            upper,
            math.nextafter(upper, math.inf),
        )
        result = select_foams105_raw_objects(
            (lower, upper),
            (_group("only", 1, diameters),),
            (Foams105BinnedRange("only", lower, upper),),
        )
        self.assertEqual(result.group_traces[0].selected_indices, (1, 2, 3, 4))
        self.assertEqual(result.group_traces[0].rejected_indices, (0, 5))

    def test_unrelated_rounded_ambiguity_does_not_block_unique_requested_key(self):
        result = select_foams105_raw_objects(
            (1.000001, 1.000002, 2.0),
            (_group("only", 1, (1.5, 2.0)),),
            (Foams105BinnedRange("only", 2.0, 2.0),),
        )
        self.assertEqual(result.group_traces[0].matched_global_index, 2)
        self.assertEqual(result.group_traces[0].selected_indices, (1,))

    def test_slot_four_lower_is_provenance_only_and_never_looked_up(self):
        groups = tuple(_group(f"g{slot}", slot, (0.5, 1.0, 2.0, 4.0)) for slot in range(1, 5))
        ranges = (
            Foams105BinnedRange("g1", 4.0, 4.0),
            Foams105BinnedRange("g2", 2.0, 2.0),
            Foams105BinnedRange("g3", 2.0, 2.0),
            Foams105BinnedRange("g4", 999.0, 1000.0),
        )
        result = select_foams105_raw_objects((1.0, 2.0, 4.0), groups, ranges)
        trace = result.group_traces[3]
        self.assertEqual(trace.lower_policy, "zero_for_slot4")
        self.assertIsNone(trace.rounded_lower_key)
        self.assertIsNone(trace.matched_global_index)
        self.assertEqual(trace.effective_lower_mm, 0.0)
        self.assertEqual(trace.supplied_range.lower_label_mm, 999.0)


class TestFoams105RawSelectionIdentity(unittest.TestCase):
    def test_component_numbers_can_repeat_across_images_and_fields_stay_aligned(self):
        objects = (
            _object("image-a", 1, 1.0, 11.0),
            _object("image-b", 1, 2.0, 22.0),
        )
        result = select_foams105_raw_objects(
            (1.0, 2.0),
            (Foams105RawGroupInput("only", 1, objects),),
            (Foams105BinnedRange("only", 1.0, 2.0),),
        )
        self.assertEqual(result.selected_identities, (("image-a", 1), ("image-b", 1)))
        self.assertEqual(result.equivalent_diameters_mm, (1.0, 2.0))
        self.assertEqual(result.areas_mm2, (11.0, 22.0))
        self.assertEqual(
            tuple(
                (
                    row.image_id,
                    row.component_id,
                    row.equivalent_diameter_mm,
                    row.area_mm2,
                )
                for row in result.rows
            ),
            (("image-a", 1, 1.0, 11.0), ("image-b", 1, 2.0, 22.0)),
        )

    def test_duplicate_image_component_identity_is_rejected_across_groups(self):
        groups = (
            Foams105RawGroupInput("g1", 1, (_object("same", 1, 2.0),)),
            Foams105RawGroupInput("g2", 2, (_object("same", 1, 1.0),)),
        )
        ranges = (
            Foams105BinnedRange("g1", 2.0, 2.0),
            Foams105BinnedRange("g2", 1.0, 1.0),
        )
        with self.assertRaises(Foams105RawSelectionValidationError) as caught:
            select_foams105_raw_objects((1.0, 2.0), groups, ranges)
        self.assertEqual(caught.exception.code, "duplicate_object_identity")
        self.assertEqual(caught.exception.context["identity"], ("same", 1))


class TestFoams105RawSelectionEmptyAndImmutable(unittest.TestCase):
    def test_empty_active_groups_and_all_empty_result_are_valid(self):
        groups = (
            Foams105RawGroupInput("g1", 1, ()),
            Foams105RawGroupInput("g2", 2, ()),
        )
        result = select_foams105_raw_objects(
            (1.0, 2.0),
            groups,
            (
                Foams105BinnedRange("g1", 2.0, 2.0),
                Foams105BinnedRange("g2", 1.0, 1.0),
            ),
        )
        self.assertEqual(result.rows, ())
        self.assertEqual(result.selected_objects, ())
        self.assertEqual(result.selected_count, 0)
        self.assertEqual(
            tuple(trace.selected_indices for trace in result.group_traces), ((), ())
        )

    def test_result_owns_snapshots_and_survives_mutable_outer_input_changes(self):
        labels = [1.0, 2.0]
        groups = [_group("only", 1, (1.0, 2.0))]
        ranges = [Foams105BinnedRange("only", 1.0, 2.0)]
        result = select_foams105_raw_objects(labels, groups, ranges)
        labels[0] = 99.0
        groups.clear()
        ranges.clear()
        self.assertEqual(result.bin_labels_mm, (1.0, 2.0))
        self.assertEqual(result.selected_count, 2)
        with self.assertRaises(FrozenInstanceError):
            result.selected_count = 0
        with self.assertRaises(FrozenInstanceError):
            result.rows[0].component_id = 99


class TestFoams105RawSelectionValidation(unittest.TestCase):
    def test_malformed_inputs_fail_with_selection_errors(self):
        valid_group = _group("only", 1, (1.0,))
        valid_range = Foams105BinnedRange("only", 1.0, 1.0)
        cases = (
            ((True,), (valid_group,), (valid_range,)),
            ((1.0,), (), ()),
            (
                (1.0,),
                (Foams105RawGroupInput("only", 2, ()),),
                (valid_range,),
            ),
            (
                (1.0,),
                (Foams105RawGroupInput("only", 1, []),),
                (valid_range,),
            ),
            ((1.0,), (valid_group,), (None,)),
            (
                (1.0,),
                (valid_group,),
                (Foams105BinnedRange("wrong", 1.0, 1.0),),
            ),
            (
                (1.0,),
                (valid_group,),
                (Foams105BinnedRange("only", 2.0, 1.0),),
            ),
        )
        for labels, groups, ranges in cases:
            with self.subTest(labels=labels, groups=groups, ranges=ranges):
                with self.assertRaises(Foams105RawSelectionValidationError):
                    select_foams105_raw_objects(labels, groups, ranges)

    def test_invalid_objects_including_disabled_groups_are_rejected(self):
        invalid_objects = (
            Foams105RawObject("", 1, 1.0, 1.0),
            Foams105RawObject("image", True, 1.0, 1.0),
            Foams105RawObject("image", 1, 0.0, 1.0),
            Foams105RawObject("image", 1, math.nan, 1.0),
            Foams105RawObject("image", 1, 1.0, math.inf),
            Foams105RawObject("image", 1, 1 + 0j, 1.0),
        )
        for object_ in invalid_objects:
            groups = (
                Foams105RawGroupInput("g1", 1, ()),
                Foams105RawGroupInput("g2", 2, (object_,)),
            )
            with self.subTest(object_=object_), self.assertRaises(
                Foams105RawSelectionValidationError
            ):
                select_foams105_raw_objects(
                    (1.0,),
                    groups,
                    (Foams105BinnedRange("g1", 1.0, 1.0), None),
                )

    def test_disabled_middle_slot_is_typed_domain_error(self):
        groups = tuple(_group(f"g{slot}", slot, ()) for slot in range(1, 4))
        ranges = (
            Foams105BinnedRange("g1", 2.0, 2.0),
            None,
            Foams105BinnedRange("g3", 1.0, 1.0),
        )
        with self.assertRaises(Foams105RawSelectionDomainError) as caught:
            select_foams105_raw_objects((1.0, 2.0), groups, ranges)
        self.assertEqual(caught.exception.code, "noncontiguous_active_slots")
        self.assertEqual(caught.exception.context["later_active_slots"], (3,))

    def test_rounding_scale_overflow_is_typed_numerical_error(self):
        with self.assertRaises(Foams105RawSelectionNumericalError) as caught:
            select_foams105_raw_objects(
                (1.0,),
                (_group("only", 1, (1.0,)),),
                (Foams105BinnedRange("only", 1e304, 1e304),),
            )
        self.assertEqual(caught.exception.code, "rounding_overflow")
        self.assertEqual(caught.exception.context["field"], "ranges[0].lower_label_mm")

    def test_numpy_scalars_are_normalized_but_booleans_are_rejected(self):
        result = select_foams105_raw_objects(
            (np.float64(1.0),),
            (
                Foams105RawGroupInput(
                    "only",
                    np.int64(1),
                    (Foams105RawObject("image", np.int64(1), np.float64(1), np.float64(2)),),
                ),
            ),
            (Foams105BinnedRange("only", np.float64(1), np.float64(1)),),
        )
        self.assertEqual(result.areas_mm2, (2.0,))
        self.assertIsInstance(result.rows[0].component_id, int)


class TestFoams105RawSelectionMetadata(unittest.TestCase):
    def test_metadata_and_policies_are_explicit(self):
        result = select_foams105_raw_objects(
            (1.0,),
            (_group("only", 1, (1.0,)),),
            (Foams105BinnedRange("only", 1.0, 1.0),),
        )
        self.assertEqual(result.method, FOAMS105_RAW_SELECTION_METHOD)
        self.assertEqual(result.source_commit, FOAMS105_RAW_SELECTION_SOURCE_COMMIT)
        self.assertEqual(result.scope, FOAMS105_RAW_SELECTION_SCOPE)
        self.assertEqual(result.supported_prefix_policy, FOAMS105_RAW_SELECTION_PREFIX_POLICY)
        self.assertEqual(
            result.concatenation_policy, FOAMS105_RAW_SELECTION_CONCATENATION_POLICY
        )
        self.assertEqual(result.boundary_policy, FOAMS105_RAW_SELECTION_BOUNDARY_POLICY)
        self.assertEqual(result.rounding_profile, FOAMS105_RAW_SELECTION_ROUNDING_PROFILE)
        self.assertEqual(result.rounding_profile, FOAMS105_ROUNDING_POLICY)


if __name__ == "__main__":
    unittest.main()
