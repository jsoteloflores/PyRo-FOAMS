import math
import os
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.distributions import create_diameter_bin_spec, geometric_diameter_bin_spec
from core.nesting import NestingPlan, NestingSegment
from core.reconstruction import (
    COMPATIBILITY_STATUS,
    RECONSTRUCTION_METHOD,
    UPPER_TAIL_ASSUMPTION,
    ReconstructionNumericalError,
    ReconstructionValidationError,
    build_spherical_section_operator,
    project_spherical_number_densities,
    reconstruct_spherical_3d,
    solve_spherical_number_densities,
)
from tests.test_nesting import make_group, make_result, make_source_image


class TestSphericalSectionOperator(unittest.TestCase):
    def test_independent_two_class_radicals(self):
        operator = build_spherical_section_operator(
            create_diameter_bin_spec((1.0, 2.0, 4.0))
        )
        expected = (
            (math.sqrt(3.0), math.sqrt(15.0) - 2.0 * math.sqrt(3.0)),
            (0.0, 2.0 * math.sqrt(3.0)),
        )

        for actual_row, expected_row in zip(operator.coefficients_mm, expected):
            for actual, target in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, target, places=15)
        self.assertEqual(operator.representative_diameters_mm, (2.0, 4.0))
        self.assertEqual(operator.method, RECONSTRUCTION_METHOD)
        self.assertEqual(operator.compatibility_status, COMPATIBILITY_STATUS)

    def test_one_bin_diagonal_and_basis_projection(self):
        operator = build_spherical_section_operator(
            create_diameter_bin_spec((3.0, 5.0))
        )
        self.assertAlmostEqual(operator.coefficients_mm[0][0], 4.0)
        self.assertEqual(project_spherical_number_densities(operator, (2.5,)), (10.0,))

    def test_geometric_and_irregular_grids_are_strictly_upper_triangular(self):
        specifications = (
            geometric_diameter_bin_spec(0.1, 6, 2.0),
            create_diameter_bin_spec((0.1, 0.13, 0.7, 0.72, 3.0)),
        )
        for specification in specifications:
            with self.subTest(edges=specification.edges_mm):
                operator = build_spherical_section_operator(specification)
                for row_index, row in enumerate(operator.coefficients_mm):
                    self.assertGreater(row[row_index], 0.0)
                    self.assertTrue(all(value == 0.0 for value in row[:row_index]))
                    for column_index in range(len(row)):
                        basis = tuple(
                            1.0 if index == column_index else 0.0
                            for index in range(len(row))
                        )
                        projected = project_spherical_number_densities(operator, basis)
                        self.assertEqual(projected[row_index], row[column_index])

    def test_each_column_sum_matches_sectionable_center_interval(self):
        edges = (0.25, 0.4, 1.1, 2.0, 7.0)
        operator = build_spherical_section_operator(create_diameter_bin_spec(edges))
        for column_index, diameter in enumerate(edges[1:]):
            column_sum = math.fsum(
                operator.coefficients_mm[row][column_index]
                for row in range(column_index + 1)
            )
            expected = math.sqrt(
                (diameter - edges[0]) * (diameter + edges[0])
            )
            self.assertAlmostEqual(column_sum, expected, places=14)

    def test_scale_covariance_of_operator_and_inverse(self):
        edges = (0.2, 0.5, 1.7, 4.0)
        na = (8.0, 3.0, 1.0)
        reference_operator = build_spherical_section_operator(
            create_diameter_bin_spec(edges)
        )
        reference = solve_spherical_number_densities(
            reference_operator,
            na,
            upper_tail_assumption=UPPER_TAIL_ASSUMPTION,
        )
        for scale in (1e-3, 1e3):
            scaled_operator = build_spherical_section_operator(
                create_diameter_bin_spec(tuple(edge * scale for edge in edges))
            )
            scaled = solve_spherical_number_densities(
                scaled_operator,
                na,
                upper_tail_assumption=UPPER_TAIL_ASSUMPTION,
            )
            for reference_row, scaled_row in zip(
                reference_operator.coefficients_mm, scaled_operator.coefficients_mm
            ):
                for reference_value, scaled_value in zip(reference_row, scaled_row):
                    self.assertAlmostEqual(
                        scaled_value / scale, reference_value, places=13
                    )
            for reference_value, scaled_value in zip(
                reference.signed_nv_per_mm3, scaled.signed_nv_per_mm3
            ):
                self.assertAlmostEqual(
                    scaled_value * scale, reference_value, places=11
                )

    def test_narrow_and_wide_finite_grids_build_deterministically(self):
        narrow = (1.0, np.nextafter(1.0, math.inf), np.nextafter(1.0, math.inf, dtype=np.float64))
        narrow = (narrow[0], narrow[1], np.nextafter(narrow[1], math.inf))
        grids = (narrow, (1e-200, 1.0, 1e200))
        for edges in grids:
            with self.subTest(edges=edges):
                first = build_spherical_section_operator(create_diameter_bin_spec(edges))
                second = build_spherical_section_operator(create_diameter_bin_spec(edges))
                self.assertEqual(first, second)
                self.assertTrue(
                    all(math.isfinite(value) for row in first.coefficients_mm for value in row)
                )

    def test_operator_and_result_records_are_immutable(self):
        operator = build_spherical_section_operator(
            create_diameter_bin_spec((1.0, 2.0))
        )
        result = solve_spherical_number_densities(
            operator, (1.0,), upper_tail_assumption=UPPER_TAIL_ASSUMPTION
        )
        with self.assertRaises(FrozenInstanceError):
            operator.edges_mm = (2.0, 3.0)
        with self.assertRaises(FrozenInstanceError):
            result.status = "changed"


class TestSignedTriangularInverse(unittest.TestCase):
    def setUp(self):
        self.operator = build_spherical_section_operator(
            create_diameter_bin_spec((1.0, 2.0, 4.0))
        )

    def solve(self, values):
        return solve_spherical_number_densities(
            self.operator,
            values,
            upper_tail_assumption=UPPER_TAIL_ASSUMPTION,
        )

    def test_round_trip_and_linear_density_scaling(self):
        original = (2.0, 3.0)
        projected = project_spherical_number_densities(self.operator, original)
        result = self.solve(projected)
        scaled = self.solve(tuple(value * 7.5 for value in projected))
        for actual, expected in zip(result.signed_nv_per_mm3, original):
            self.assertAlmostEqual(actual, expected, places=14)
        for actual, expected in zip(scaled.signed_nv_per_mm3, original):
            self.assertAlmostEqual(actual, expected * 7.5, places=13)
        self.assertTrue(result.forward_consistency_passed)
        self.assertEqual(result.status, "nonnegative")

    def test_zero_input_is_exact_and_has_zero_tolerances(self):
        result = self.solve((0.0, 0.0))
        self.assertEqual(result.signed_nv_per_mm3, (0.0, 0.0))
        self.assertEqual(result.fitted_na_per_mm2, (0.0, 0.0))
        self.assertEqual(result.residuals_per_mm2, (0.0, 0.0))
        self.assertEqual(result.forward_consistency_tolerance_per_mm2, 0.0)
        self.assertEqual(result.negative_classification_tolerance_per_mm3, 0.0)
        self.assertTrue(result.forward_consistency_passed)

    def test_material_negative_solution_is_preserved(self):
        result = self.solve((0.0, 1.0))
        self.assertLess(result.signed_nv_per_mm3[0], 0.0)
        self.assertGreater(result.signed_nv_per_mm3[1], 0.0)
        self.assertEqual(result.status, "negative_solution")
        self.assertEqual(result.negative_local_bin_indices, (0,))
        self.assertEqual(result.materially_negative_local_bin_indices, (0,))
        self.assertEqual(result.roundoff_negative_local_bin_indices, ())

    def test_roundoff_negative_classification_does_not_modify_value(self):
        tolerance_scale = 1.0
        tiny_negative = -32.0 * np.finfo(np.float64).eps * tolerance_scale
        result = self.solve((
            self.operator.coefficients_mm[0][1] * tolerance_scale
            + self.operator.coefficients_mm[0][0] * tiny_negative,
            self.operator.coefficients_mm[1][1] * tolerance_scale,
        ))
        self.assertLess(result.signed_nv_per_mm3[0], 0.0)
        self.assertEqual(result.status, "roundoff_negative")
        self.assertEqual(result.roundoff_negative_local_bin_indices, (0,))

    def test_tail_assumption_is_required_and_exact(self):
        with self.assertRaisesRegex(ReconstructionValidationError, "explicitly"):
            solve_spherical_number_densities(self.operator, (1.0, 1.0))
        with self.assertRaises(ReconstructionValidationError):
            solve_spherical_number_densities(
                self.operator, (1.0, 1.0), upper_tail_assumption="implicit"
            )

    def test_nonrepresentable_solution_raises_numerical_error(self):
        tiny_operator = build_spherical_section_operator(
            create_diameter_bin_spec((1e-300, 2e-300))
        )
        with self.assertRaisesRegex(ReconstructionNumericalError, "nonfinite"):
            solve_spherical_number_densities(
                tiny_operator,
                (1e308,),
                upper_tail_assumption=UPPER_TAIL_ASSUMPTION,
            )

    def test_public_vector_validation(self):
        invalid = ((1.0,), (1.0, -1.0), (1.0, math.nan), (1.0, math.inf), (1.0, True))
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ReconstructionValidationError):
                project_spherical_number_densities(self.operator, values)


class TestReconstructionValidation(unittest.TestCase):
    def setUp(self):
        self.operator = build_spherical_section_operator(
            create_diameter_bin_spec((1.0, 2.0, 4.0))
        )

    def test_rejects_malformed_direct_operators(self):
        matrix = self.operator.coefficients_mm
        cases = (
            replace(self.operator, edges_mm=[1.0, 2.0, 4.0]),
            replace(self.operator, representative_diameters_mm=(2.0, 5.0)),
            replace(self.operator, coefficients_mm=(matrix[0],)),
            replace(self.operator, coefficients_mm=(matrix[0], (1.0, matrix[1][1]))),
            replace(
                self.operator,
                coefficients_mm=(
                    (np.nextafter(matrix[0][0], math.inf), matrix[0][1]),
                    matrix[1],
                ),
            ),
            replace(self.operator, method="other"),
            replace(self.operator, input_density_unit="um^-3"),
        )
        for malformed in cases:
            with self.subTest(malformed=malformed), self.assertRaises(
                ReconstructionValidationError
            ):
                project_spherical_number_densities(malformed, (1.0, 1.0))

    def test_operator_requires_valid_immutable_bin_spec(self):
        valid = create_diameter_bin_spec((1.0, 2.0))
        with self.assertRaises(ReconstructionValidationError):
            build_spherical_section_operator(replace(valid, edges_mm=[1.0, 2.0]))


class TestNestingReconstructionAdapter(unittest.TestCase):
    def setUp(self):
        bins = create_diameter_bin_spec((0.1, 0.2, 0.4, 0.8, 1.6))
        fine_image = make_source_image(0, "fine-image", "fine", 1.0)
        coarse_image = make_source_image(1, "coarse-image", "coarse", 10.0)
        fine = make_group(
            "sample", "fine", bins, (fine_image,), (20, 10, 5, 2)
        )
        coarse = make_group(
            "sample", "coarse", bins, (coarse_image,), (180, 120, 40, 10)
        )
        self.source = make_result(
            bins, (fine, coarse), (fine_image, coarse_image)
        )

    def test_restricted_nesting_retains_exact_provenance_and_tail_advisory(self):
        plan = NestingPlan("sample", 1, 3, (NestingSegment("fine", 1, 3),))
        result = reconstruct_spherical_3d(
            self.source, plan, upper_tail_assumption=UPPER_TAIL_ASSUMPTION
        )
        nested = result.nested_distribution
        self.assertIs(nested.bin_spec, self.source.bin_spec)
        self.assertIs(nested.source_images, self.source.source_images)
        self.assertIs(nested.source_diagnostics, self.source.diagnostics)
        self.assertEqual(nested.selected_bin_indices, (1, 2))
        self.assertEqual(result.operator.edges_mm, (0.2, 0.4, 0.8))
        self.assertEqual(result.input_na_per_mm2, nested.number_densities_per_mm2)
        self.assertTrue(result.upper_range_truncated)
        self.assertIsNotNone(result.upper_range_advisory)
        self.assertEqual(result.upper_tail_assumption, UPPER_TAIL_ASSUMPTION)

    def test_full_range_has_no_truncation_advisory_and_retains_nesting_warnings(self):
        plan = NestingPlan(
            "sample",
            0,
            4,
            (NestingSegment("fine", 0, 2), NestingSegment("coarse", 2, 4)),
        )
        result = reconstruct_spherical_3d(
            self.source, plan, upper_tail_assumption=UPPER_TAIL_ASSUMPTION
        )
        self.assertFalse(result.upper_range_truncated)
        self.assertIsNone(result.upper_range_advisory)
        self.assertEqual(len(result.nested_distribution.overlap_diagnostics), 1)
        self.assertFalse(result.nested_distribution.exclude_border)

    def test_negative_diagnostics_use_absolute_source_indices(self):
        plan = NestingPlan("sample", 1, 3, (NestingSegment("fine", 1, 3),))
        sparse_group = replace(
            self.source.groups[("sample", "fine")],
            counts=(20, 0, 1, 2),
            number_densities_per_mm2=(20.0, 0.0, 1.0, 2.0),
        )
        groups = dict(self.source.groups)
        groups[("sample", "fine")] = sparse_group
        source = replace(self.source, groups=groups)
        result = reconstruct_spherical_3d(
            source, plan, upper_tail_assumption=UPPER_TAIL_ASSUMPTION
        )
        self.assertEqual(result.materially_negative_absolute_bin_indices, (1,))
        self.assertEqual(result.status, "negative_solution")

    def test_adapter_calls_nesting_once(self):
        plan = NestingPlan("sample", 0, 4, (NestingSegment("fine", 0, 4),))
        from core import reconstruction

        original = reconstruction.nest_2d_distribution
        with patch.object(
            reconstruction, "nest_2d_distribution", wraps=original
        ) as nested_call:
            reconstruct_spherical_3d(
                self.source, plan, upper_tail_assumption=UPPER_TAIL_ASSUMPTION
            )
        nested_call.assert_called_once_with(self.source, plan)

    def test_adapter_requires_explicit_tail_assumption_before_nesting(self):
        plan = NestingPlan("sample", 0, 1, (NestingSegment("fine", 0, 1),))
        with self.assertRaisesRegex(ReconstructionValidationError, "explicitly"):
            reconstruct_spherical_3d(self.source, plan)


if __name__ == "__main__":
    unittest.main()
