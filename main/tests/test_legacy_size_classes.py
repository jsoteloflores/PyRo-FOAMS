import json
import math
import os
import sys
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.legacy_size_classes import (
    FOAMS105_AREA_PROVENANCE,
    FOAMS105_EDGE_POLICY,
    FOAMS105_HISTOGRAM_METHOD,
    FOAMS105_LABEL_METHOD,
    FOAMS105_ROUNDING_VERIFICATION,
    Foams105SizeClassNumericalError,
    Foams105SizeClassValidationError,
    _round_nonnegative_5_decimals,
    build_foams105_bin_labels,
    count_foams105_histogram,
    filter_foams105_diameters,
    normalize_foams105_counts,
)
from examples.foams105_size_classes import _count_mismatches

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "07_foams105_histogram_reference.json"
)


class TestFoams105HistogramFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_PATH, encoding="utf-8") as stream:
            cls.fixture = json.load(stream)

    def test_all_four_groups_and_180_count_cells_match_exactly(self):
        labels = self.fixture["bin_labels_mm"]
        groups = self.fixture["groups"]
        self.assertEqual(len(labels), 45)
        self.assertEqual(len(groups), 4)
        compared_rows = 0
        for group, expected_input_count in zip(groups, (777, 135, 2076, 0)):
            diameters = group["already_thresholded_equivalent_diameters_mm"]
            expected = group["expected_counts"]
            result = count_foams105_histogram(labels, diameters)
            self.assertEqual(len(expected), 45)
            self.assertEqual(len(result.counts), 45)
            self.assertEqual(len(diameters), expected_input_count)
            self.assertEqual(result.counts, tuple(expected))
            self.assertEqual(result.input_count, expected_input_count)
            self.assertEqual(result.retained_count, expected_input_count)
            self.assertEqual(result.discarded_upper_count, 0)
            compared_rows += len(result.counts)
        self.assertEqual(compared_rows, 180)

    def test_fixture_provenance_and_total_observations(self):
        self.assertEqual(
            self.fixture["source_commit"],
            "179663203f2d0f86b2863d5ea7f8f70dadca02f8",
        )
        self.assertEqual(
            sum(
                len(group["already_thresholded_equivalent_diameters_mm"])
                for group in self.fixture["groups"]
            ),
            2988,
        )
        self.assertEqual(
            tuple(self.fixture["workbook_sha256"]),
            ("NA_mag.xls", "Raw_data.xls"),
        )


class TestFoams105LabelPreparation(unittest.TestCase):
    def test_literal_anchors_and_separate_scalar_recurrence(self):
        result = build_foams105_bin_labels(10, (100,))
        self.assertEqual(
            result.bin_labels_mm[:5],
            (0.1, 0.12589, 0.15849, 0.19953, 0.25119),
        )
        self.assertEqual(result.bin_labels_mm[-1], 2511.88643)
        expected_raw = [0.1]
        for _ in range(44):
            expected_raw.append(expected_raw[-1] * (10.0 ** 0.1))
        self.assertEqual(result.raw_bin_labels_mm, tuple(expected_raw))
        self.assertEqual(result.method, FOAMS105_LABEL_METHOD)
        self.assertEqual(
            result.rounding_verification, FOAMS105_ROUNDING_VERIFICATION
        )

    def test_rounding_below_at_and_above_representable_scaled_tie(self):
        scaled_values = (
            math.nextafter(1.5, 0.0),
            1.5,
            math.nextafter(1.5, math.inf),
        )
        rounded = tuple(
            _round_nonnegative_5_decimals(value / 100000.0, index)
            for index, value in enumerate(scaled_values)
        )
        self.assertEqual(rounded, (0.00001, 0.00002, 0.00002))

    def test_uses_largest_of_one_to_four_scales_without_mutation(self):
        scales = [20.0, 100.0, 50.0]
        result = build_foams105_bin_labels(10, scales)
        self.assertEqual(result.raw_bin_labels_mm[0], 0.1)
        self.assertEqual(result.scales_px_per_mm, (20.0, 100.0, 50.0))
        self.assertEqual(scales, [20.0, 100.0, 50.0])

    def test_rounding_preserves_and_reports_zero_and_duplicate_labels(self):
        result = build_foams105_bin_labels(1e-320, (1.0,))
        self.assertEqual(len(result.bin_labels_mm), 45)
        self.assertEqual(result.zero_label_indices, tuple(range(45)))
        self.assertEqual(
            result.adjacent_duplicate_label_index_pairs,
            tuple((index - 1, index) for index in range(1, 45)),
        )

    def test_label_input_and_numerical_failures_are_contextual(self):
        invalid = (
            (True, (1.0,), "minimum_diameter_px"),
            ("1", (1.0,), "minimum_diameter_px"),
            (1.0, (), "scales_px_per_mm"),
            (1.0, (1, 2, 3, 4, 5), "scales_px_per_mm"),
            (1.0, (np.array(1.0),), "scales_px_per_mm\\[0\\]"),
            (1.0, (10**400,), "scales_px_per_mm\\[0\\]"),
        )
        for minimum, scales, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(
                Foams105SizeClassValidationError, message
            ):
                build_foams105_bin_labels(minimum, scales)
        with self.assertRaisesRegex(Foams105SizeClassNumericalError, "recurrence"):
            build_foams105_bin_labels(1e308, (1.0,))
        with self.assertRaisesRegex(Foams105SizeClassNumericalError, "rounding scale"):
            build_foams105_bin_labels(2e303, (1.0,))


class TestFoams105DiameterFiltering(unittest.TestCase):
    def test_threshold_neighbors_equality_and_indices(self):
        threshold = (10.0 / 100.0) * (10.0 ** -0.1)
        diameters = [
            math.nextafter(threshold, 0.0),
            threshold,
            math.nextafter(threshold, math.inf),
        ]
        result = filter_foams105_diameters(diameters, 10, 100)
        self.assertEqual(result.threshold_mm, threshold)
        self.assertEqual(result.retained_indices, (1, 2))
        self.assertEqual(result.rejected_below_threshold_indices, (0,))
        self.assertEqual(result.retained_diameters_mm, tuple(diameters[1:]))
        self.assertEqual(
            len(result.retained_indices)
            + len(result.rejected_below_threshold_indices),
            result.input_count,
        )
        self.assertEqual(diameters[0], math.nextafter(threshold, 0.0))

    def test_two_scales_and_empty_observations(self):
        first = filter_foams105_diameters((), 10, 100)
        second = filter_foams105_diameters((), 10, 200)
        self.assertEqual(first.retained_diameters_mm, ())
        self.assertEqual(first.retained_indices, ())
        self.assertEqual(first.rejected_below_threshold_indices, ())
        self.assertEqual(second.threshold_mm, first.threshold_mm / 2.0)

    def test_invalid_diameters_fail_instead_of_becoming_rejections(self):
        for value in (0.0, -1.0, math.nan, math.inf, True, "1", 1 + 0j):
            with self.subTest(value=value), self.assertRaisesRegex(
                Foams105SizeClassValidationError, "diameters_mm\\[0\\]"
            ):
                filter_foams105_diameters((value,), 10, 100)


class TestFoams105HistogramOwnership(unittest.TestCase):
    def test_literal_analytical_fixture_and_upper_tail(self):
        result = count_foams105_histogram(
            (1, 2, 4), (0.5, 1, 1.5, 2, 3.9, 4, 5)
        )
        self.assertEqual(result.counts, (1, 2, 2))
        self.assertEqual(result.retained_count, 5)
        self.assertEqual(result.discarded_upper_count, 2)
        self.assertEqual(result.input_count, 7)
        self.assertEqual(result.method, FOAMS105_HISTOGRAM_METHOD)
        self.assertEqual(result.edge_policy, FOAMS105_EDGE_POLICY)

    def test_exact_and_nextafter_boundary_ownership(self):
        below_one = math.nextafter(1.0, 0.0)
        above_one = math.nextafter(1.0, math.inf)
        below_two = math.nextafter(2.0, 0.0)
        above_two = math.nextafter(2.0, math.inf)
        below_four = math.nextafter(4.0, 0.0)
        result = count_foams105_histogram(
            (1, 2, 4),
            (below_one, 1.0, above_one, below_two, 2.0, above_two, below_four, 4.0),
        )
        self.assertEqual(result.counts, (1, 3, 3))
        self.assertEqual(result.discarded_upper_count, 1)

    def test_duplicate_run_zero_singleton_and_empty_cases(self):
        duplicate = count_foams105_histogram((1, 2, 2, 2, 4), (2,))
        self.assertEqual(duplicate.counts, (0, 0, 0, 0, 1))
        self.assertEqual(
            duplicate.adjacent_duplicate_label_index_pairs,
            ((1, 2), (2, 3)),
        )
        zero = count_foams105_histogram((0, 1, 2), (0.5, 1.0))
        self.assertEqual(zero.counts, (0, 1, 1))
        singleton = count_foams105_histogram((1,), (0.5, 1.0))
        self.assertEqual(singleton.counts, (1,))
        self.assertEqual(singleton.discarded_upper_count, 1)
        empty = count_foams105_histogram((1, 2), ())
        self.assertEqual(empty.counts, (0, 0))
        self.assertEqual(empty.input_count, 0)

    def test_histogram_validation_is_contextual(self):
        invalid_labels = ((), (2, 1), (-1,), tuple(range(46)), (True,), (10**400,))
        for labels in invalid_labels:
            with self.subTest(labels=labels), self.assertRaises(
                Foams105SizeClassValidationError
            ):
                count_foams105_histogram(labels, ())
        for diameter in (0, -1, math.nan, math.inf, True, "1", np.array(1.0), 10**400):
            with self.subTest(diameter=diameter), self.assertRaisesRegex(
                Foams105SizeClassValidationError, "diameters_mm\\[0\\]"
            ):
                count_foams105_histogram((1,), (diameter,))


class TestFoams105Normalization(unittest.TestCase):
    def test_analytical_normalization_area_scaling_and_zero_counts(self):
        histogram = count_foams105_histogram(
            (1, 2, 4), (0.5, 1, 1.5, 2, 3.9, 4, 5)
        )
        first = normalize_foams105_counts(histogram, 2)
        doubled = normalize_foams105_counts(histogram, 4)
        self.assertEqual(first.na_per_mm2, (0.5, 1.0, 1.0))
        self.assertEqual(
            doubled.na_per_mm2,
            tuple(value / 2.0 for value in first.na_per_mm2),
        )
        empty = normalize_foams105_counts(
            count_foams105_histogram((1, 2), ()), 2
        )
        self.assertEqual(empty.na_per_mm2, (0.0, 0.0))
        self.assertEqual(first.area_provenance, FOAMS105_AREA_PROVENANCE)

    def test_invalid_area_and_histogram_type_fail(self):
        histogram = count_foams105_histogram((1,), ())
        for area in (0, -1, math.nan, math.inf, True, "1", 10**400):
            with self.subTest(area=area), self.assertRaises(
                Foams105SizeClassValidationError
            ):
                normalize_foams105_counts(histogram, area)
        with self.assertRaisesRegex(
            Foams105SizeClassValidationError, "histogram_result"
        ):
            normalize_foams105_counts(object(), 1)
        with self.assertRaisesRegex(
            Foams105SizeClassNumericalError, "index 0"
        ):
            normalize_foams105_counts(
                count_foams105_histogram((1,), (0.5,)), 5e-324
            )

    def test_example_comparison_rejects_truncation_and_mismatch(self):
        complete = tuple(0 for _ in range(45))
        self.assertEqual(_count_mismatches(complete, complete, 1), ())
        with self.assertRaisesRegex(ValueError, "candidate length"):
            _count_mismatches(complete[:-1], complete, 1)
        mismatched = list(complete)
        mismatched[7] = 1
        self.assertEqual(_count_mismatches(mismatched, complete, 1), (7,))

    def test_results_are_immutable_and_deterministic(self):
        built = build_foams105_bin_labels(10, (100,))
        filtered = filter_foams105_diameters((0.1, 0.2), 10, 100)
        first = count_foams105_histogram((1, 2, 2, 4), (1, 2, 4))
        second = count_foams105_histogram((1, 2, 2, 4), (1, 2, 4))
        normalized = normalize_foams105_counts(first, 2)
        self.assertEqual(first, second)
        for result in (built, filtered, first, normalized):
            with self.subTest(result=type(result).__name__), self.assertRaises(
                FrozenInstanceError
            ):
                result.method = "changed"


if __name__ == "__main__":
    unittest.main()
