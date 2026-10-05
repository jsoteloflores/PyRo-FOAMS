import json
import os
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.legacy_cutoffs import (
    FOAMS105_CUTOFF_METHOD,
    Foams105CutoffGroupInput,
    suggest_foams105_cutoffs,
)
from core.legacy_foams import convert_foams105_nv
from core.legacy_selection import (
    FOAMS105_SELECTION_AUTOSMART_ORIGIN,
    FOAMS105_SELECTION_MANUAL_ORIGIN,
    FOAMS105_SELECTION_METHOD,
    FOAMS105_SELECTION_MISALIGNMENT_POLICY,
    FOAMS105_SELECTION_SCOPE,
    FOAMS105_SELECTION_SOURCE_COMMIT,
    FOAMS105_SELECTION_SOURCE_FILE,
    FOAMS105_SELECTION_SOURCE_REPOSITORY,
    Foams105BinnedGroupInput,
    Foams105BinnedRange,
    Foams105SelectionDomainError,
    Foams105SelectionValidationError,
    select_foams105_binned_ranges,
    select_foams105_suggested_ranges,
)

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "11_foams105_selection_cases.json"
)
HISTOGRAM_FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "07_foams105_histogram_reference.json"
)
CONVERSION_FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "06_foams105_nv_reference.json"
)
SELECTION_FIXTURE_LF_SHA256 = (
    "acaa9f08ac3aaa3326f6d661b6df02c724d09eedb3b41020e1985da1a06f1a4b"
)


def _normalized_text_sha256(raw_bytes):
    return sha256(raw_bytes.replace(b"\r\n", b"\n")).hexdigest()


class TestFoams105SelectionFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_PATH, encoding="utf-8") as stream:
            cls.fixture = json.load(stream)

    def test_all_seven_checked_source_slice_cases(self):
        self.assertEqual(len(self.fixture["cases"]), 7)
        for case in self.fixture["cases"]:
            groups = tuple(
                Foams105BinnedGroupInput(
                    group_id=f"slot-{slot}",
                    source_slot=slot,
                    na_per_mm2=tuple(densities),
                )
                for slot, densities in enumerate(case["densities"], start=1)
            )
            ranges = tuple(
                None
                if bounds is None
                else Foams105BinnedRange(
                    group_id=groups[index].group_id,
                    lower_label_mm=bounds[0],
                    upper_label_mm=bounds[1],
                )
                for index, bounds in enumerate(case["ranges"])
            )
            with self.subTest(case=case["id"]):
                if "expected_error" in case:
                    with self.assertRaises(Foams105SelectionDomainError) as caught:
                        select_foams105_binned_ranges(
                            case["labels"], groups, ranges
                        )
                    self.assertEqual(caught.exception.code, case["expected_error"])
                    continue

                result = select_foams105_binned_ranges(
                    case["labels"], groups, ranges
                )
                self.assertEqual(result.bin_labels_mm, tuple(case["expected_labels"]))
                self.assertEqual(result.na_per_mm2, tuple(case["expected_na"]))
                self.assertEqual(
                    tuple(row.source_slot for row in result.rows),
                    tuple(case["expected_source_slots"]),
                )
                self.assertEqual(
                    tuple(row.global_bin_index for row in result.rows),
                    tuple(case["expected_global_indices"]),
                )

    def test_vendored_fixture_hash_is_pinned(self):
        raw_bytes = Path(FIXTURE_PATH).read_bytes()
        lf_bytes = raw_bytes.replace(b"\r\n", b"\n")
        self.assertEqual(_normalized_text_sha256(lf_bytes), SELECTION_FIXTURE_LF_SHA256)

    def test_vendored_fixture_hash_accepts_crlf_checkout(self):
        raw_bytes = Path(FIXTURE_PATH).read_bytes()
        lf_bytes = raw_bytes.replace(b"\r\n", b"\n")
        crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
        self.assertEqual(
            _normalized_text_sha256(crlf_bytes), SELECTION_FIXTURE_LF_SHA256
        )

    def test_vendored_fixture_hash_rejects_changed_numerical_value(self):
        raw_bytes = Path(FIXTURE_PATH).read_bytes()
        lf_bytes = raw_bytes.replace(b"\r\n", b"\n")
        original = b'"expected_na": [\n        1,\n        2,\n        10,'
        replacement = b'"expected_na": [\n        1,\n        2,\n        10.5,'
        self.assertEqual(lf_bytes.count(original), 1)
        changed_bytes = lf_bytes.replace(original, replacement, 1)
        self.assertNotEqual(changed_bytes, lf_bytes)
        self.assertEqual(changed_bytes.count(replacement), 1)
        self.assertNotEqual(
            _normalized_text_sha256(changed_bytes), SELECTION_FIXTURE_LF_SHA256
        )


class TestFoams105WorkbookSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(HISTOGRAM_FIXTURE_PATH, encoding="utf-8") as stream:
            cls.histogram_fixture = json.load(stream)
        with open(CONVERSION_FIXTURE_PATH, encoding="utf-8") as stream:
            cls.conversion_fixture = json.load(stream)
        cls.labels = tuple(cls.histogram_fixture["bin_labels_mm"])
        cls.groups = tuple(
            Foams105BinnedGroupInput(
                group_id=f"group-{group['group_id']}",
                source_slot=slot,
                na_per_mm2=tuple(group["source_na_per_mm2"]),
            )
            for slot, group in enumerate(cls.histogram_fixture["groups"], start=1)
        )
        cls.ranges = (
            Foams105BinnedRange("group-1", cls.labels[14], cls.labels[27]),
            None,
            Foams105BinnedRange("group-3", cls.labels[0], cls.labels[14]),
            None,
        )
        cls.result = select_foams105_binned_ranges(
            cls.labels, cls.groups, cls.ranges
        )

    def test_exact_29_selected_workbook_rows_and_source_identity(self):
        expected = self.conversion_fixture
        self.assertEqual(self.result.bin_labels_mm, tuple(expected["bin_labels_mm"]))
        self.assertEqual(self.result.na_per_mm2, tuple(expected["na_per_mm2"]))
        self.assertEqual(self.result.output_row_count, 29)
        self.assertEqual(
            tuple(row.source_slot for row in self.result.rows),
            (3,) * 15 + (1,) * 14,
        )
        self.assertEqual(
            tuple(row.global_bin_index for row in self.result.rows),
            tuple(range(15)) + tuple(range(14, 28)),
        )
        self.assertEqual(self.result.disabled_group_ids, ("group-2", "group-4"))
        self.assertEqual(
            self.result.adjacent_duplicate_output_index_pairs, ((14, 15),)
        )
        self.assertEqual(self.result.overlapping_global_indices, (14,))
        self.assertEqual(self.result.uncovered_global_indices, ())
        self.assertTrue(self.result.converter_length_supported)

    def test_explicit_converter_call_matches_all_workbook_rows(self):
        converted = convert_foams105_nv(
            self.result.bin_labels_mm, self.result.na_per_mm2
        )
        expected = self.conversion_fixture["expected_nv_per_mm3"]
        self.assertEqual(len(converted.signed_nv_per_mm3), 29)
        for candidate, reference in zip(converted.signed_nv_per_mm3, expected):
            self.assertLessEqual(abs(candidate - reference), 1e-12 * abs(reference))


class TestFoams105SelectionDiagnostics(unittest.TestCase):
    def test_gap_overlap_and_shared_boundary_are_reported_without_repair(self):
        labels = (1.0, 2.0, 3.0, 4.0, 5.0)
        groups = (
            Foams105BinnedGroupInput("coarse", 1, (1.0,) * 5),
            Foams105BinnedGroupInput("fine", 2, (1.0,) * 5),
        )
        gap = select_foams105_binned_ranges(
            labels,
            groups,
            (
                Foams105BinnedRange("coarse", 4.0, 5.0),
                Foams105BinnedRange("fine", 1.0, 2.0),
            ),
        )
        self.assertEqual(gap.uncovered_global_indices, (2,))
        self.assertEqual(gap.overlapping_global_indices, ())

        overlap = select_foams105_binned_ranges(
            labels,
            groups,
            (
                Foams105BinnedRange("coarse", 3.0, 5.0),
                Foams105BinnedRange("fine", 1.0, 3.0),
            ),
        )
        self.assertEqual(overlap.overlapping_global_indices, (2,))
        self.assertEqual(overlap.adjacent_duplicate_output_index_pairs, ((2, 3),))

    def test_more_than_45_rows_are_retained_but_converter_is_unsupported(self):
        labels = tuple(float(index + 1) for index in range(45))
        groups = tuple(
            Foams105BinnedGroupInput(f"g{slot}", slot, (1.0,) * 45)
            for slot in range(1, 5)
        )
        ranges = (
            Foams105BinnedRange("g1", labels[30], labels[44]),
            Foams105BinnedRange("g2", labels[20], labels[30]),
            Foams105BinnedRange("g3", labels[10], labels[20]),
            Foams105BinnedRange("g4", labels[0], labels[10]),
        )
        result = select_foams105_binned_ranges(labels, groups, ranges)
        self.assertEqual(result.output_row_count, 48)
        self.assertFalse(result.converter_length_supported)
        self.assertEqual(result.overlapping_global_indices, (10, 20, 30))
        self.assertEqual(
            result.adjacent_duplicate_output_index_pairs,
            ((10, 11), (21, 22), (32, 33)),
        )

    def test_literal_bounds_are_retained_and_middle_slots_stay_disabled(self):
        groups = (
            Foams105BinnedGroupInput("g1", 1, (1.0, 1.0, 1.0)),
            Foams105BinnedGroupInput("g2", 2, (1.0, 1.0, 1.0)),
            Foams105BinnedGroupInput("g3", 3, (1.0, 1.0, 1.0)),
        )
        result = select_foams105_binned_ranges(
            (1.0, 2.0, 3.0),
            groups,
            (
                Foams105BinnedRange("g1", 2.5, 3.5),
                None,
                Foams105BinnedRange("g3", 0.5, 1.5),
            ),
        )
        self.assertEqual(result.disabled_group_ids, ("g2",))
        self.assertEqual(tuple(row.source_slot for row in result.rows), (3, 1))
        self.assertEqual(result.requested_ranges[0].lower_label_mm, 2.5)
        self.assertEqual(result.requested_ranges[2].upper_label_mm, 1.5)


class TestFoams105SuggestedRangeAdapter(unittest.TestCase):
    def test_valid_suggestion_delegates_with_provenance(self):
        suggestion = suggest_foams105_cutoffs(
            (1.0, 2.0, 3.0, 4.0),
            (Foams105CutoffGroupInput("only", 10.0, (0.0, 2.0, 0.0, 3.0)),),
        )
        result = select_foams105_suggested_ranges(suggestion)
        self.assertEqual(result.bin_labels_mm, (2.0, 3.0, 4.0))
        self.assertEqual(result.na_per_mm2, (2.0, 0.0, 3.0))
        self.assertEqual(result.range_origin, FOAMS105_SELECTION_AUTOSMART_ORIGIN)
        self.assertEqual(result.suggestion_method, FOAMS105_CUTOFF_METHOD)
        self.assertEqual(result.suggestion_source_commit, suggestion.source_commit)

    def test_inconsistent_manual_suggestion_is_rejected_before_selection(self):
        suggestion = suggest_foams105_cutoffs(
            (1.0, 2.0, 3.0),
            (Foams105CutoffGroupInput("only", 10.0, (1.0, 2.0, 3.0)),),
        )
        malformed_range = replace(
            suggestion.suggested_ranges[0], lower_label_mm=2.0
        )
        malformed = replace(suggestion, suggested_ranges=(malformed_range,))
        with self.assertRaises(Foams105SelectionValidationError) as caught:
            select_foams105_suggested_ranges(malformed)
        self.assertEqual(caught.exception.code, "inconsistent_suggestion")

    def test_valid_multi_group_suggestion_matches_explicit_manual_delegation(self):
        suggestion = suggest_foams105_cutoffs(
            (1.0, 2.0, 3.0, 4.0),
            (
                Foams105CutoffGroupInput("coarse", 1.0, (1.0, 1.0, 1.0, 1.0)),
                Foams105CutoffGroupInput("fine", 2.0, (0.0, 2.0, 4.0, 8.0)),
            ),
        )
        adapted = select_foams105_suggested_ranges(suggestion)
        manual = select_foams105_binned_ranges(
            suggestion.bin_labels_mm,
            tuple(
                Foams105BinnedGroupInput(
                    group.group_id, index + 1, group.na_per_mm2
                )
                for index, group in enumerate(suggestion.groups)
            ),
            tuple(
                Foams105BinnedRange(
                    range_.group_id,
                    range_.lower_label_mm,
                    range_.upper_label_mm,
                )
                for range_ in suggestion.suggested_ranges
            ),
        )
        self.assertEqual(adapted.rows, manual.rows)
        self.assertEqual(adapted.group_traces, manual.group_traces)
        self.assertEqual(adapted.bin_labels_mm, manual.bin_labels_mm)
        self.assertEqual(adapted.na_per_mm2, manual.na_per_mm2)


class TestFoams105SelectionValidation(unittest.TestCase):
    def test_invalid_records_and_grids_fail_contextually(self):
        group = Foams105BinnedGroupInput("g1", 1, (1.0, 1.0))
        valid_range = Foams105BinnedRange("g1", 1.0, 2.0)
        cases = (
            ((1.0, 1.0), (group,), (valid_range,), "unsupported_grid"),
            ((1.0, 2.0), (), (), "invalid_input"),
            (
                (1.0, 2.0),
                (Foams105BinnedGroupInput("g1", 2, (1.0, 1.0)),),
                (valid_range,),
                "invalid_input",
            ),
            ((1.0, 2.0), (group,), (None,), "invalid_input"),
            (
                (1.0, 2.0),
                (group,),
                (Foams105BinnedRange("other", 1.0, 2.0),),
                "invalid_input",
            ),
            (
                (1.0, 2.0),
                (group,),
                (Foams105BinnedRange("g1", 2.0, 1.0),),
                "invalid_input",
            ),
        )
        for labels, groups, ranges, code in cases:
            with self.subTest(labels=labels, groups=groups, ranges=ranges):
                with self.assertRaises(Foams105SelectionValidationError) as caught:
                    select_foams105_binned_ranges(labels, groups, ranges)
                self.assertEqual(caught.exception.code, code)

    def test_empty_numeric_range_is_a_typed_domain_error(self):
        with self.assertRaises(Foams105SelectionDomainError) as caught:
            select_foams105_binned_ranges(
                (1.0, 2.0),
                (Foams105BinnedGroupInput("g1", 1, (1.0, 1.0)),),
                (Foams105BinnedRange("g1", 3.0, 4.0),),
            )
        self.assertEqual(caught.exception.code, "empty_selected_range")

    def test_inputs_outputs_metadata_and_error_context_are_immutable(self):
        group = Foams105BinnedGroupInput("g1", 1, (1.0, 2.0))
        range_ = Foams105BinnedRange("g1", 1.0, 2.0)
        result = select_foams105_binned_ranges((1.0, 2.0), (group,), (range_,))
        self.assertEqual(result.range_origin, FOAMS105_SELECTION_MANUAL_ORIGIN)
        self.assertEqual(result.method, FOAMS105_SELECTION_METHOD)
        self.assertEqual(result.source_repository, FOAMS105_SELECTION_SOURCE_REPOSITORY)
        self.assertEqual(result.source_commit, FOAMS105_SELECTION_SOURCE_COMMIT)
        self.assertEqual(result.source_file, FOAMS105_SELECTION_SOURCE_FILE)
        self.assertEqual(result.scope, FOAMS105_SELECTION_SCOPE)
        self.assertEqual(
            result.misalignment_policy, FOAMS105_SELECTION_MISALIGNMENT_POLICY
        )
        self.assertIsInstance(result.rows, tuple)
        self.assertIsInstance(result.group_traces, tuple)
        with self.assertRaises(FrozenInstanceError):
            result.output_row_count = 0
        with self.assertRaises(FrozenInstanceError):
            result.rows[0].source_slot = 2
        with self.assertRaises(Foams105SelectionDomainError) as caught:
            select_foams105_binned_ranges(
                (1.0, 2.0),
                (group,),
                (Foams105BinnedRange("g1", 3.0, 4.0),),
            )
        with self.assertRaises(TypeError):
            caught.exception.context["new"] = "value"


if __name__ == "__main__":
    unittest.main()
