from __future__ import annotations

import json
import math
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from main.core.legacy_cutoffs import (
    FOAMS105_CUTOFF_METHOD,
    FOAMS105_CUTOFF_OCCURRENCE_TIE_POLICY,
    FOAMS105_CUTOFF_SCOPE,
    FOAMS105_CUTOFF_SIGN_TIE_POLICY,
    FOAMS105_CUTOFF_SOURCE_COMMIT,
    FOAMS105_CUTOFF_STATUS,
    FOAMS105_CUTOFF_ZERO_DIFFERENCE_POLICY,
    Foams105CutoffDomainError,
    Foams105CutoffGroupInput,
    Foams105CutoffValidationError,
    suggest_foams105_cutoffs,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "10_foams105_autosmart_cases.json"


def _groups(case):
    return tuple(
        Foams105CutoffGroupInput(
            f"group-{index + 1}",
            scale,
            tuple(densities),
        )
        for index, (scale, densities) in enumerate(
            zip(case["scales_px_per_mm"], case["na_per_mm2"])
        )
    )


def _group(group_id, scale, densities):
    return Foams105CutoffGroupInput(group_id, scale, tuple(densities))


class Foams105CutoffFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_all_fixture_cases(self):
        self.assertEqual(len(self.fixture["cases"]), 7)
        for case in self.fixture["cases"]:
            with self.subTest(case=case["id"]):
                if "expected_error" in case:
                    with self.assertRaises(Foams105CutoffDomainError) as raised:
                        suggest_foams105_cutoffs(case["labels_mm"], _groups(case))
                    self.assertEqual(raised.exception.code, case["expected_error"])
                    continue
                result = suggest_foams105_cutoffs(
                    case["labels_mm"], _groups(case)
                )
                self.assertEqual(
                    tuple(item.selected_index for item in result.transitions),
                    tuple(case["expected_transition_indices"]),
                )
                self.assertEqual(
                    tuple(
                        (item.lower_index, item.upper_index)
                        for item in result.suggested_ranges
                    ),
                    tuple(tuple(value) for value in case["expected_ranges"]),
                )
                for suggested_range in result.suggested_ranges:
                    self.assertEqual(
                        suggested_range.lower_label_mm,
                        result.bin_labels_mm[suggested_range.lower_index],
                    )
                    self.assertEqual(
                        suggested_range.upper_label_mm,
                        result.bin_labels_mm[suggested_range.upper_index],
                    )


class Foams105CutoffTransitionTests(unittest.TestCase):
    def test_positive_only_and_negative_only_candidates(self):
        labels = (1, 2, 3, 4)
        coarse = _group("coarse", 1, (1, 1, 1, 1))
        cases = (
            (_group("fine-positive", 2, (2, 4, 8, 16)), 2.0),
            (_group("fine-negative", 2, (0.5, 0.25, 0.125, 0.0625)), 0.5),
        )
        for fine, expected_ratio in cases:
            with self.subTest(group=fine.group_id):
                transition = suggest_foams105_cutoffs(
                    labels, (coarse, fine)
                ).transitions[0]
                self.assertEqual(transition.selected_index, 0)
                self.assertTrue(
                    math.isclose(
                        transition.selected_signed_difference,
                        math.log(expected_ratio),
                        rel_tol=1e-15,
                    )
                )
                if expected_ratio > 1:
                    self.assertIsNone(transition.largest_negative_difference)
                else:
                    self.assertIsNone(transition.smallest_positive_difference)

    def test_equal_sign_magnitude_selects_negative_last_occurrence(self):
        result = suggest_foams105_cutoffs(
            (1, 2, 3, 4, 5),
            (
                _group("coarse", 1, (1, 1, 1, 1, 1)),
                _group("fine", 2, (math.e, 1 / math.e, 1 / math.e, math.e, 0)),
            ),
        )
        transition = result.transitions[0]
        self.assertEqual(transition.selected_signed_difference, -1.0)
        self.assertEqual(transition.tied_indices, (1, 2))
        self.assertEqual(transition.selected_index, 2)

    def test_exact_zero_differences_are_excluded(self):
        transition = suggest_foams105_cutoffs(
            (1, 2, 3),
            (
                _group("coarse", 1, (1, 1, 1)),
                _group("fine", 2, (1, 2, 4)),
            ),
        ).transitions[0]
        self.assertEqual(transition.overlap_indices, (0, 1, 2))
        self.assertEqual(transition.signed_log_differences[0], 0.0)
        self.assertEqual(transition.selected_index, 1)

    def test_pair_errors_include_pair_context(self):
        cases = (
            (
                "no_positive_overlap",
                _group("coarse", 1, (1, 0)),
                _group("fine", 2, (0, 1)),
            ),
            (
                "no_nonzero_log_difference",
                _group("coarse", 1, (1, 2)),
                _group("fine", 2, (1, 2)),
            ),
        )
        for code, coarse, fine in cases:
            with self.subTest(code=code):
                with self.assertRaises(Foams105CutoffDomainError) as raised:
                    suggest_foams105_cutoffs((1, 2), (coarse, fine))
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(
                    raised.exception.context["pair_group_ids"],
                    ("coarse", "fine"),
                )

    def test_extreme_finite_densities_use_finite_separate_logs(self):
        transition = suggest_foams105_cutoffs(
            (1, 2),
            (
                _group("coarse", 1, (1e-300, 1)),
                _group("fine", 2, (1e300, 1)),
            ),
        ).transitions[0]
        self.assertEqual(transition.selected_index, 0)
        self.assertTrue(math.isfinite(transition.selected_signed_difference))
        self.assertGreater(transition.selected_signed_difference, 1300)


class Foams105CutoffRangeTests(unittest.TestCase):
    def test_one_group_internal_zeros_remain_inside_support_bounds(self):
        result = suggest_foams105_cutoffs(
            (1, 2, 3, 4, 5),
            (_group("only", 1, (0, 2, 0, 3, 0)),),
        )
        suggested_range = result.suggested_ranges[0]
        self.assertEqual(
            (suggested_range.lower_index, suggested_range.upper_index), (1, 3)
        )

    def test_one_group_all_zero_errors(self):
        with self.assertRaises(Foams105CutoffDomainError) as raised:
            suggest_foams105_cutoffs(
                (1, 2, 3), (_group("only", 1, (0, 0, 0)),)
            )
        self.assertEqual(raised.exception.code, "empty_group_support")
        self.assertEqual(raised.exception.context["group_id"], "only")

    def test_each_three_group_pair_is_independent(self):
        result = suggest_foams105_cutoffs(
            (1, 2, 3, 4, 5),
            (
                _group("coarse", 1, (1, 1, 1, 1, 1)),
                _group("middle", 2, (16, 8, 4, 2, 4)),
                _group("fine", 3, (0, 0, 8, 8, 32)),
            ),
        )
        self.assertEqual(
            tuple(item.selected_index for item in result.transitions), (3, 2)
        )
        self.assertEqual(
            tuple(
                (item.lower_index, item.upper_index)
                for item in result.suggested_ranges
            ),
            ((4, 4), (3, 3), (2, 2)),
        )

    def test_crossing_transitions_error_without_repair(self):
        with self.assertRaises(Foams105CutoffDomainError) as raised:
            suggest_foams105_cutoffs(
                (1, 2, 3, 4),
                (
                    _group("coarse", 1, (1, 1, 1, 1)),
                    _group("middle", 2, (4, 2, 8, 16)),
                    _group("fine", 3, (64, 16, 16, 64)),
                ),
            )
        self.assertEqual(raised.exception.code, "invalid_suggested_range")
        self.assertEqual(raised.exception.context["group_id"], "middle")
        self.assertEqual(raised.exception.context["proposed_lower_index"], 3)
        self.assertEqual(raised.exception.context["proposed_upper_index"], 1)

    def test_finest_lower_bound_differs_between_two_and_four_groups(self):
        two_group = suggest_foams105_cutoffs(
            (1, 2, 3, 4),
            (
                _group("coarse", 1, (1, 1, 1, 1)),
                _group("fine", 2, (0, 0, 2, 4)),
            ),
        )
        self.assertEqual(two_group.suggested_ranges[-1].lower_index, 2)

        four_group = suggest_foams105_cutoffs(
            (1, 2, 3, 4, 5, 6),
            (
                _group("g1", 1, (1, 1, 1, 1, 1, 1)),
                _group("g2", 2, (32, 16, 8, 4, 2, 4)),
                _group("g3", 3, (1024, 256, 64, 8, 8, 32)),
                _group("g4", 4, (0, 4096, 128, 32, 64, 256)),
            ),
        )
        self.assertEqual(
            tuple(item.selected_index for item in four_group.transitions),
            (4, 3, 2),
        )
        self.assertEqual(four_group.groups[-1].na_per_mm2[0], 0.0)
        self.assertEqual(four_group.suggested_ranges[-1].lower_index, 0)

    def test_transition_at_final_label_has_no_successor(self):
        with self.assertRaises(Foams105CutoffDomainError) as raised:
            suggest_foams105_cutoffs(
                (1, 2, 3),
                (
                    _group("coarse", 1, (1, 1, 1)),
                    _group("fine", 2, (1, 1, 2)),
                ),
            )
        self.assertEqual(raised.exception.code, "transition_has_no_successor")
        self.assertEqual(raised.exception.context["transition_index"], 2)


class Foams105CutoffContractTests(unittest.TestCase):
    def test_malformed_labels_fail_contextually(self):
        bad_labels = (
            (),
            tuple(range(1, 47)),
            (0, 1),
            (1, 1),
            (2, 1),
            (1, math.nan),
            (1, math.inf),
            (1, True),
            (1, "2"),
            (1, 2 + 0j),
        )
        group = _group("group", 1, (1, 1))
        for labels in bad_labels:
            density_count = len(labels) if 1 <= len(labels) <= 45 else 2
            candidate = _group("group", 1, (1,) * density_count)
            with self.subTest(labels=labels):
                with self.assertRaises(Foams105CutoffValidationError):
                    suggest_foams105_cutoffs(labels, (candidate,))
        self.assertEqual(group.na_per_mm2, (1, 1))

    def test_malformed_groups_fail_contextually(self):
        labels = (1, 2)
        bad_groups = (
            (),
            tuple(_group(f"g{i}", i + 1, (1, 1)) for i in range(5)),
            (object(),),
            (_group("", 1, (1, 1)),),
            (_group("g", 0, (1, 1)),),
            (_group("g", math.inf, (1, 1)),),
            (_group("g", True, (1, 1)),),
            (Foams105CutoffGroupInput("g", 1, [1, 1]),),
            (_group("g", 1, (1,)),),
            (_group("g", 1, (1, -1)),),
            (_group("g", 1, (1, math.nan)),),
            (_group("g", 1, (1, True)),),
            (_group("g", 1, (1, "1")),),
        )
        for groups in bad_groups:
            with self.subTest(groups=groups):
                with self.assertRaises(Foams105CutoffValidationError):
                    suggest_foams105_cutoffs(labels, groups)

    def test_duplicate_ids_scales_and_unsorted_scales_fail(self):
        cases = (
            (_group("same", 1, (1, 1)), _group("same", 2, (1, 1))),
            (_group("a", 1, (1, 1)), _group("b", 1, (1, 1))),
            (_group("a", 2, (1, 1)), _group("b", 1, (1, 1))),
        )
        for groups in cases:
            with self.subTest(groups=groups):
                with self.assertRaises(Foams105CutoffValidationError):
                    suggest_foams105_cutoffs((1, 2), groups)

    def test_inputs_unchanged_results_deterministic_frozen_and_complete(self):
        labels = [1, 2, 3, 4]
        coarse_values = (1, 1, 1, 1)
        fine_values = (2, 1, 4, 8)
        groups = (
            _group("coarse", 1, coarse_values),
            _group("fine", 2, fine_values),
        )
        first = suggest_foams105_cutoffs(labels, groups)
        second = suggest_foams105_cutoffs(labels, groups)
        self.assertEqual(first, second)
        self.assertEqual(labels, [1, 2, 3, 4])
        self.assertEqual(groups[0].na_per_mm2, coarse_values)
        self.assertEqual(groups[1].na_per_mm2, fine_values)
        self.assertIsInstance(first.bin_labels_mm, tuple)
        self.assertIsInstance(first.groups, tuple)
        self.assertIsInstance(first.transitions, tuple)
        self.assertIsInstance(first.suggested_ranges, tuple)
        with self.assertRaises(FrozenInstanceError):
            first.bin_labels_mm = ()
        with self.assertRaises(FrozenInstanceError):
            first.transitions[0].selected_index = 2
        self.assertEqual(first.method, FOAMS105_CUTOFF_METHOD)
        self.assertEqual(first.source_commit, FOAMS105_CUTOFF_SOURCE_COMMIT)
        self.assertEqual(first.scope, FOAMS105_CUTOFF_SCOPE)
        self.assertEqual(
            first.zero_difference_policy,
            FOAMS105_CUTOFF_ZERO_DIFFERENCE_POLICY,
        )
        self.assertEqual(first.sign_tie_policy, FOAMS105_CUTOFF_SIGN_TIE_POLICY)
        self.assertEqual(
            first.occurrence_tie_policy,
            FOAMS105_CUTOFF_OCCURRENCE_TIE_POLICY,
        )
        self.assertEqual(first.suggestion_status, FOAMS105_CUTOFF_STATUS)
        self.assertEqual(first.length_unit, "mm")
        self.assertEqual(first.scale_unit, "pixels/mm")
        self.assertEqual(first.density_unit, "mm^-2")


if __name__ == "__main__":
    unittest.main()
