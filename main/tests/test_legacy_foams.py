import json
import math
import os
import sys
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.legacy_foams import (
    FOAMS105_HEIGHT_POLICY,
    FOAMS105_METHOD,
    FOAMS105_SCOPE,
    FOAMS105_SMALLEST_CLASS_POLICY,
    Foams105NumericalError,
    Foams105ValidationError,
    convert_foams105_nv,
)
from core.reconstruction import RECONSTRUCTION_METHOD

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "06_foams105_nv_reference.json"
)


class TestFoams105ReferenceFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_PATH, encoding="utf-8") as stream:
            cls.fixture = json.load(stream)
        cls.result = convert_foams105_nv(
            cls.fixture["bin_labels_mm"], cls.fixture["na_per_mm2"]
        )

    def test_all_29_original_workbook_rows_match_relative_tolerance(self):
        expected = self.fixture["expected_nv_per_mm3"]
        self.assertEqual(len(expected), 29)
        errors = []
        relative_errors = []
        for index, (candidate, reference) in enumerate(
            zip(self.result.signed_nv_per_mm3, expected)
        ):
            error = abs(candidate - reference)
            errors.append(error)
            relative_errors.append(error / abs(reference))
            self.assertLessEqual(
                error,
                1e-12 * abs(reference),
                f"Workbook row {index} exceeds relative tolerance",
            )
        self.assertEqual(errors.index(max(errors)), 0)
        self.assertLessEqual(max(relative_errors), 1e-12)

    def test_reference_provenance_and_duplicate_transition_label_are_retained(self):
        self.assertEqual(
            self.result.bin_labels_mm, tuple(self.fixture["bin_labels_mm"])
        )
        self.assertEqual(self.result.bin_labels_mm[14:16], (0.15132, 0.15132))
        self.assertEqual(self.result.na_per_mm2, tuple(self.fixture["na_per_mm2"]))
        self.assertEqual(self.result.method, FOAMS105_METHOD)
        self.assertEqual(self.result.scope, FOAMS105_SCOPE)
        self.assertEqual(
            self.result.source_repository, self.fixture["source_repo"]
        )
        self.assertEqual(self.result.source_commit, self.fixture["source_commit"])

    def test_alpha_prefix_matches_source_replay_anchors_in_lag_order(self):
        expected = (
            1.6461208533433853,
            0.45612264530845004,
            0.11619047775207883,
            0.04149451147417596,
            0.0172711049259736,
        )
        for candidate, reference in zip(self.result.alpha_coefficients, expected):
            self.assertTrue(
                math.isclose(candidate, reference, rel_tol=1e-14, abs_tol=0.0)
            )
        self.assertGreater(self.result.probabilities[0], self.result.probabilities[1])


class TestFoams105ConversionArithmetic(unittest.TestCase):
    def test_first_height_uses_zero_lower_term_for_cropped_labels(self):
        result = convert_foams105_nv((10.0, 20.0), (1.0, 1.0))
        expected = 10.0 * (0.5 ** (1.0 / 3.0))
        self.assertTrue(
            math.isclose(
                result.mean_projected_heights_mm[0],
                expected,
                rel_tol=1e-15,
                abs_tol=0.0,
            )
        )
        self.assertEqual(result.height_policy, FOAMS105_HEIGHT_POLICY)

    def test_one_two_and_three_classes_preserve_source_subtraction_rules(self):
        one = convert_foams105_nv((1.0,), (2.0,))
        self.assertEqual(one.larger_contributions_per_mm2, (0.0,))
        self.assertEqual(
            one.signed_nv_per_mm3[0],
            one.alpha_coefficients[0] * 2.0 / one.mean_projected_heights_mm[0],
        )

        two = convert_foams105_nv((1.0, 2.0), (2.0, 3.0))
        self.assertEqual(two.larger_contributions_per_mm2, (0.0, 0.0))

        three = convert_foams105_nv((1.0, 2.0, 4.0), (2.0, 3.0, 5.0))
        self.assertEqual(three.larger_contributions_per_mm2[0], 0.0)
        self.assertEqual(
            three.larger_contributions_per_mm2[1],
            three.alpha_coefficients[1] * 5.0,
        )
        self.assertEqual(three.larger_contributions_per_mm2[2], 0.0)
        self.assertEqual(
            three.smallest_class_policy, FOAMS105_SMALLEST_CLASS_POLICY
        )

    def test_larger_class_changes_interior_but_not_smallest_class(self):
        baseline = convert_foams105_nv((1.0, 2.0, 4.0), (2.0, 3.0, 1.0))
        changed = convert_foams105_nv((1.0, 2.0, 4.0), (2.0, 3.0, 10.0))
        self.assertEqual(
            baseline.signed_nv_per_mm3[0], changed.signed_nv_per_mm3[0]
        )
        self.assertNotEqual(
            baseline.signed_nv_per_mm3[1], changed.signed_nv_per_mm3[1]
        )

    def test_negative_interior_is_retained_without_clipping(self):
        result = convert_foams105_nv((1.0, 2.0, 4.0), (1.0, 0.0, 1.0))
        self.assertLess(result.signed_nv_per_mm3[1], 0.0)
        self.assertEqual(result.negative_indices, (1,))

    def test_all_zero_input_is_exact_and_rows_remain_in_place(self):
        result = convert_foams105_nv((1.0, 2.0, 3.0, 4.0), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(result.na_per_mm2, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(result.signed_nv_per_mm3, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(len(result.alpha_coefficients), 4)

    def test_density_linearity(self):
        base = convert_foams105_nv((1.0, 2.0, 4.0), (2.0, 3.0, 5.0))
        scaled = convert_foams105_nv((1.0, 2.0, 4.0), (14.0, 21.0, 35.0))
        for candidate, reference in zip(
            scaled.signed_nv_per_mm3, base.signed_nv_per_mm3
        ):
            self.assertTrue(
                math.isclose(candidate, 7.0 * reference, rel_tol=1e-14, abs_tol=0.0)
            )

    def test_physical_scale_covariance_keeps_fixed_coefficients(self):
        labels = (0.5, 1.0, 2.0, 4.0)
        densities = (8.0, 3.0, 1.0, 0.25)
        reference = convert_foams105_nv(labels, densities)
        for scale in (1e-3, 1e3):
            scaled = convert_foams105_nv(
                tuple(label * scale for label in labels),
                tuple(density / scale**2 for density in densities),
            )
            self.assertEqual(scaled.probabilities, reference.probabilities)
            self.assertEqual(scaled.alpha_coefficients, reference.alpha_coefficients)
            for candidate, expected in zip(
                scaled.signed_nv_per_mm3, reference.signed_nv_per_mm3
            ):
                self.assertTrue(
                    math.isclose(
                        candidate,
                        expected / scale**3,
                        rel_tol=1e-12,
                        abs_tol=0.0,
                    )
                )

    def test_inputs_are_unchanged_results_immutable_and_deterministic(self):
        labels = [1.0, 2.0, 2.0, 4.0]
        densities = [1.0, 0.0, 2.0, 3.0]
        original_labels = labels.copy()
        original_densities = densities.copy()
        first = convert_foams105_nv(labels, densities)
        second = convert_foams105_nv(labels, densities)
        self.assertEqual(first, second)
        self.assertEqual(labels, original_labels)
        self.assertEqual(densities, original_densities)
        self.assertIsInstance(first.bin_labels_mm, tuple)
        with self.assertRaises(FrozenInstanceError):
            first.method = "changed"

    def test_modern_method_identifier_is_unchanged(self):
        self.assertEqual(RECONSTRUCTION_METHOD, "spherical_upper_edge_triangular_v1")
        self.assertNotEqual(RECONSTRUCTION_METHOD, FOAMS105_METHOD)


class TestFoams105Validation(unittest.TestCase):
    def test_invalid_lengths_fail(self):
        invalid = (
            ((), ()),
            ((1.0,), ()),
            (tuple(range(1, 47)), tuple(0.0 for _ in range(46))),
        )
        for labels, densities in invalid:
            with self.subTest(length=len(labels)), self.assertRaises(
                Foams105ValidationError
            ):
                convert_foams105_nv(labels, densities)

    def test_invalid_scalar_types_and_values_fail(self):
        invalid = (
            ((True,), (1.0,)),
            ((np.bool_(True),), (1.0,)),
            (("1.0",), (1.0,)),
            ((np.array(1.0),), (1.0,)),
            ((math.nan,), (1.0,)),
            ((math.inf,), (1.0,)),
            ((0.0,), (1.0,)),
            ((2.0, 1.0), (1.0, 1.0)),
            ((1.0,), (True,)),
            ((1.0,), ("1.0",)),
            ((1.0,), (math.nan,)),
            ((1.0,), (math.inf,)),
            ((1.0,), (-1.0,)),
        )
        for labels, densities in invalid:
            with self.subTest(labels=labels, densities=densities), self.assertRaises(
                Foams105ValidationError
            ):
                convert_foams105_nv(labels, densities)

    def test_invalid_units_fail_without_guessing(self):
        with self.assertRaisesRegex(Foams105ValidationError, "length_unit"):
            convert_foams105_nv((1.0,), (1.0,), length_unit="um")
        with self.assertRaisesRegex(Foams105ValidationError, "input_density_unit"):
            convert_foams105_nv(
                (1.0,), (1.0,), input_density_unit="um^-2"
            )

    def test_numerical_overflow_fails_contextually(self):
        with self.assertRaisesRegex(Foams105NumericalError, "index 0"):
            convert_foams105_nv((1.0,), (1e308,))


if __name__ == "__main__":
    unittest.main()
