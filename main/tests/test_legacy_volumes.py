from __future__ import annotations

import json
import math
import unittest
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from main.core.legacy_areas import calculate_foams105_image_areas
from main.core.legacy_foams import convert_foams105_nv
from main.core.legacy_measurements import measure_foams105_image
from main.core.legacy_raw_selection import (
    Foams105RawGroupInput,
    Foams105RawObject,
    select_foams105_raw_objects,
)
from main.core.legacy_selection import Foams105BinnedRange
from main.core.legacy_vesicularity import (
    Foams105VesicularityGroupInput,
    calculate_foams105_2d_vesicularity,
)
from main.core.legacy_volumes import (
    DIAGNOSTIC_CUMULATIVE_DECREASE,
    DIAGNOSTIC_NEGATIVE_ADJUSTED_FRACTION_ROW,
    DIAGNOSTIC_NEGATIVE_NORMALIZATION_FACTOR,
    DIAGNOSTIC_NEGATIVE_NV_ROW,
    DIAGNOSTIC_NEGATIVE_RAW_FRACTION_ROW,
    DIAGNOSTIC_NEGATIVE_RAW_FRACTION_SUM,
    DIAGNOSTIC_RAW_FRACTION_SUM_ABOVE_ONE,
    DIAGNOSTIC_TARGET_PERCENT_OUT_OF_RANGE,
    FOAMS105_VOLUME_ADJUSTMENT_POLICY,
    FOAMS105_VOLUME_CUMULATIVE_POLICY,
    FOAMS105_VOLUME_METHOD,
    FOAMS105_VOLUME_RELATIONSHIP_POLICY,
    FOAMS105_VOLUME_SCOPE,
    Foams105VolumeDomainError,
    Foams105VolumeNumericalError,
    Foams105VolumeValidationError,
    _calculate_foams105_volume_numerical,
    _checked_divide,
    _checked_sum,
    calculate_foams105_volume_distribution,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "14_foams105_volume_cases.json"
FIXTURE_LF_SHA256 = "a4e2752da6c71a1aea59c2f15f18100bf939751b8da1c974dfca0aaa7e08d4c5"


def _assert_close_sequence(test_case, actual, expected):
    test_case.assertEqual(len(actual), len(expected))
    for candidate, reference in zip(actual, expected):
        test_case.assertTrue(
            math.isclose(candidate, reference, rel_tol=1e-12, abs_tol=1e-12),
            f"{candidate!r} != {reference!r}",
        )


def _vesicularity_group(group_id, source_slot, scale, image):
    image_id = f"{group_id}-image"
    measurement = measure_foams105_image(
        image, 1, scale, (None, None, None, None)
    )
    area_result = calculate_foams105_image_areas(image_id, measurement)
    objects = tuple(
        Foams105RawObject(
            image_id,
            component.component_id,
            component.equivalent_diameter_mm,
            component.area_mm2,
        )
        for component in measurement.pore_result.components
    )
    return (
        Foams105RawGroupInput(group_id, source_slot, objects),
        Foams105VesicularityGroupInput(
            group_id, source_slot, (area_result,), objects
        ),
    )


def _canonical_vesicularity():
    coarse_image = np.zeros((10, 10), dtype=np.uint8)
    coarse_image[2, 2] = 1
    fine_image = np.zeros((10, 10), dtype=np.uint8)
    fine_image[3, 3] = 1
    coarse_raw, coarse_context = _vesicularity_group(
        "coarse", 1, 1.0, coarse_image
    )
    fine_raw, fine_context = _vesicularity_group("fine", 2, 2.0, fine_image)
    raw = select_foams105_raw_objects(
        (0.1, 1.0, 10.0),
        (coarse_raw, fine_raw),
        (
            Foams105BinnedRange("coarse", 1.0, 10.0),
            Foams105BinnedRange("fine", 0.1, 10.0),
        ),
    )
    return calculate_foams105_2d_vesicularity(
        raw, (coarse_context, fine_context)
    )


class TestFoams105VolumeNumericalFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_all_ten_independent_numerical_cases(self):
        self.assertEqual(len(self.fixture["cases"]), 10)
        successes = 0
        expected_rejections = 0
        for case in self.fixture["cases"]:
            with self.subTest(case=case["id"]):
                try:
                    result = _calculate_foams105_volume_numerical(
                        case["heights"],
                        case["nv"],
                        case["mode"],
                        target_vesicularity_percent=case.get("target_percent"),
                    )
                except (Foams105VolumeDomainError, Foams105VolumeNumericalError) as exc:
                    self.assertEqual(exc.code, case.get("expected_error"))
                    expected_rejections += 1
                    continue

                self.assertNotIn("expected_error", case)
                _assert_close_sequence(
                    self, result.raw_volume_fractions, case["raw"]
                )
                self.assertTrue(
                    math.isclose(
                        result.raw_fraction_sum,
                        case["sum"],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        result.normalization_factor,
                        case["factor"],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
                _assert_close_sequence(
                    self, result.adjusted_volume_fractions, case["adjusted"]
                )
                _assert_close_sequence(
                    self,
                    result.cumulative_adjusted_volume_fractions,
                    case["cumulative"],
                )
                successes += 1
        self.assertEqual((successes, expected_rejections), (8, 2))

    def test_fixture_normalized_text_hash_is_pinned(self):
        normalized = FIXTURE_PATH.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(sha256(normalized).hexdigest(), FIXTURE_LF_SHA256)

    def test_signed_diagnostics_are_retained_without_clipping(self):
        result = _calculate_foams105_volume_numerical(
            (1.0, 1.0, 1.0),
            (-1.909859317102744, 5.729577951308232, 0.0),
            "known_vesicularity",
            target_vesicularity_percent=50.0,
        )
        codes = tuple(diagnostic.code for diagnostic in result.diagnostics)
        self.assertIn(DIAGNOSTIC_NEGATIVE_NV_ROW, codes)
        self.assertIn(DIAGNOSTIC_NEGATIVE_RAW_FRACTION_ROW, codes)
        self.assertIn(DIAGNOSTIC_NEGATIVE_ADJUSTED_FRACTION_ROW, codes)
        self.assertIn(DIAGNOSTIC_CUMULATIVE_DECREASE, codes)

    def test_negative_total_and_out_of_range_integrated_target_are_diagnosed(self):
        negative = _calculate_foams105_volume_numerical(
            (1.0,),
            (-1.909859317102744,),
            "known_vesicularity",
            target_vesicularity_percent=50.0,
        )
        self.assertTrue(
            math.isclose(
                negative.normalization_factor,
                -0.5,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        negative_codes = tuple(item.code for item in negative.diagnostics)
        self.assertIn(DIAGNOSTIC_NEGATIVE_RAW_FRACTION_SUM, negative_codes)
        self.assertIn(DIAGNOSTIC_NEGATIVE_NORMALIZATION_FACTOR, negative_codes)

        integrated = _calculate_foams105_volume_numerical(
            (1.0,),
            (1.909859317102744,),
            "integrated_2d",
            target_vesicularity_percent=120.0,
        )
        self.assertIn(
            DIAGNOSTIC_TARGET_PERCENT_OUT_OF_RANGE,
            tuple(item.code for item in integrated.diagnostics),
        )

    def test_zero_and_cancellation_are_valid_only_without_normalization(self):
        zero = _calculate_foams105_volume_numerical((1.0,), (0.0,))
        self.assertEqual(zero.raw_fraction_sum, 0.0)
        self.assertEqual(zero.adjusted_fraction_sum, 0.0)
        cancelled = _calculate_foams105_volume_numerical(
            (1.0, 1.0),
            (1.909859317102744, -1.909859317102744),
        )
        self.assertEqual(cancelled.raw_fraction_sum, 0.0)
        with self.assertRaises(Foams105VolumeDomainError) as caught:
            _calculate_foams105_volume_numerical(
                (1.0,),
                (0.0,),
                "known_vesicularity",
                target_vesicularity_percent=0.0,
            )
        self.assertEqual(caught.exception.code, "zero_raw_volume_sum")


class TestFoams105VolumeNumericalFailures(unittest.TestCase):
    def test_height_cube_overflow_and_positive_underflow_are_contextual(self):
        cases = (
            (1e200, "nonfinite_volume_arithmetic"),
            (1e-200, "positive_sphere_volume_underflow"),
        )
        for height, code in cases:
            with self.subTest(height=height), self.assertRaises(
                Foams105VolumeNumericalError
            ) as caught:
                _calculate_foams105_volume_numerical((height,), (0.0,))
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(caught.exception.context["stage"], "height_cube")
            self.assertEqual(caught.exception.context["row_index"], 0)

    def test_pi_multiplication_overflow_is_contextual(self):
        height = 4e102
        with self.assertRaises(Foams105VolumeNumericalError) as caught:
            _calculate_foams105_volume_numerical((height,), (1.0,))
        self.assertEqual(caught.exception.code, "nonfinite_volume_arithmetic")
        self.assertEqual(caught.exception.context["stage"], "sphere_volume_pi")

    def test_division_and_volume_nv_underflow_are_contextual(self):
        with self.assertRaises(Foams105VolumeNumericalError) as divided:
            _checked_divide(5e-324, 6.0, stage="sphere_volume_divide_six", row_index=0)
        self.assertEqual(divided.exception.code, "nonzero_underflow")
        with self.assertRaises(Foams105VolumeNumericalError) as multiplied:
            _calculate_foams105_volume_numerical((0.9,), (5e-324,))
        self.assertEqual(multiplied.exception.code, "nonzero_underflow")
        self.assertEqual(
            multiplied.exception.context["stage"], "sphere_volume_times_nv"
        )

    def test_target_scaling_factor_and_adjustment_failures_are_contextual(self):
        with self.assertRaises(Foams105VolumeNumericalError) as target:
            _calculate_foams105_volume_numerical(
                (1.0,),
                (1.909859317102744,),
                "integrated_2d",
                target_vesicularity_percent=5e-324,
            )
        self.assertEqual(target.exception.context["stage"], "target_percent_divide_100")

        with self.assertRaises(Foams105VolumeNumericalError) as factor:
            _calculate_foams105_volume_numerical(
                (1.0,),
                (1e-323,),
                "integrated_2d",
                target_vesicularity_percent=50.0,
            )
        self.assertEqual(factor.exception.context["stage"], "normalization_factor")

        with self.assertRaises(Foams105VolumeNumericalError) as adjusted:
            _calculate_foams105_volume_numerical(
                (1.0, 1.0),
                (1e-323, 1.909859317102744),
                "integrated_2d",
                target_vesicularity_percent=49.0,
            )
        self.assertEqual(adjusted.exception.context["stage"], "adjusted_volume_fraction")

    def test_prefix_sum_overflow_has_stage_and_row(self):
        with self.assertRaises(Foams105VolumeNumericalError) as caught:
            _checked_sum(
                1e308,
                1e308,
                stage="cumulative_adjusted_volume_fraction",
                row_index=3,
            )
        self.assertEqual(caught.exception.context["stage"], "cumulative_adjusted_volume_fraction")
        self.assertEqual(caught.exception.context["row_index"], 3)


class TestFoams105VolumePublicAPI(unittest.TestCase):
    def setUp(self):
        self.conversion = convert_foams105_nv((1.0, 2.0, 4.0), (8.0, 3.0, 1.0))

    def test_unnormalized_preserves_converter_rows_and_nv(self):
        result = calculate_foams105_volume_distribution(self.conversion)
        self.assertEqual(result.mode, "unnormalized")
        self.assertEqual(result.normalization_factor, 1.0)
        self.assertIsNone(result.target_vesicularity_percent)
        self.assertIsNone(result.normalization_residual)
        self.assertEqual(
            tuple(row.signed_nv_per_mm3 for row in result.rows),
            self.conversion.signed_nv_per_mm3,
        )
        self.assertEqual(result.adjusted_volume_fractions, result.raw_volume_fractions)
        self.assertEqual(result.adjusted_fraction_sum, result.raw_fraction_sum)

    def test_known_mode_treats_input_as_percent_and_accepts_zero_and_100(self):
        forty = calculate_foams105_volume_distribution(
            self.conversion,
            "known_vesicularity",
            known_vesicularity_percent=40.0,
        )
        fraction_like = calculate_foams105_volume_distribution(
            self.conversion,
            "known_vesicularity",
            known_vesicularity_percent=0.4,
        )
        zero = calculate_foams105_volume_distribution(
            self.conversion,
            "known_vesicularity",
            known_vesicularity_percent=0.0,
        )
        full = calculate_foams105_volume_distribution(
            self.conversion,
            "known_vesicularity",
            known_vesicularity_percent=100.0,
        )
        self.assertTrue(math.isclose(forty.adjusted_fraction_sum, 0.4, abs_tol=1e-12))
        self.assertTrue(
            math.isclose(fraction_like.adjusted_fraction_sum, 0.004, abs_tol=1e-12)
        )
        self.assertEqual(zero.normalization_factor, 0.0)
        self.assertEqual(zero.adjusted_volume_fractions, (0.0, 0.0, 0.0))
        self.assertTrue(math.isclose(full.adjusted_fraction_sum, 1.0, abs_tol=1e-12))

    def test_integrated_mode_uses_integrated_not_group_mean_percentage(self):
        vesicularity = _canonical_vesicularity()
        self.assertTrue(
            math.isclose(
                vesicularity.integrated_vesicularity_percent,
                1.99,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        self.assertEqual(vesicularity.mean_group_vesicularity_percent, 1.0)
        result = calculate_foams105_volume_distribution(
            self.conversion,
            "integrated_2d",
            vesicularity_result=vesicularity,
        )
        self.assertEqual(result.vesicularity_result, vesicularity)
        self.assertTrue(
            math.isclose(
                result.target_vesicularity_percent,
                1.99,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(result.adjusted_fraction_sum, 0.0199, abs_tol=1e-12)
        )
        self.assertEqual(
            result.upstream_vesicularity_diagnostics, vesicularity.diagnostics
        )

    def test_duplicate_rows_survive_and_signed_nv_is_never_changed(self):
        conversion = convert_foams105_nv(
            (1.0, 2.0, 2.0, 4.0), (1.0, 0.0, 2.0, 3.0)
        )
        result = calculate_foams105_volume_distribution(conversion)
        self.assertEqual(result.adjacent_duplicate_label_index_pairs, ((1, 2),))
        self.assertEqual(tuple(row.bin_label_mm for row in result.rows), conversion.bin_labels_mm)
        self.assertEqual(
            tuple(row.signed_nv_per_mm3 for row in result.rows),
            conversion.signed_nv_per_mm3,
        )
        self.assertEqual(len(result.rows), 4)

    def test_metadata_immutability_and_owned_outputs(self):
        labels = [1.0, 2.0]
        densities = [2.0, 1.0]
        conversion = convert_foams105_nv(labels, densities)
        result = calculate_foams105_volume_distribution(conversion)
        labels[0] = 99.0
        densities[0] = 99.0
        self.assertEqual(result.conversion_result.bin_labels_mm, (1.0, 2.0))
        self.assertIsInstance(result.rows, tuple)
        self.assertEqual(result.method, FOAMS105_VOLUME_METHOD)
        self.assertEqual(result.scope, FOAMS105_VOLUME_SCOPE)
        self.assertEqual(result.adjustment_policy, FOAMS105_VOLUME_ADJUSTMENT_POLICY)
        self.assertEqual(result.cumulative_policy, FOAMS105_VOLUME_CUMULATIVE_POLICY)
        self.assertEqual(
            result.analysis_relationship_policy, FOAMS105_VOLUME_RELATIONSHIP_POLICY
        )
        with self.assertRaises(FrozenInstanceError):
            result.normalization_factor = 2.0

    def test_raw_sum_above_one_is_diagnosed(self):
        result = calculate_foams105_volume_distribution(self.conversion)
        if result.raw_fraction_sum > 1.0:
            self.assertIn(
                DIAGNOSTIC_RAW_FRACTION_SUM_ABOVE_ONE,
                tuple(item.code for item in result.diagnostics),
            )


class TestFoams105VolumeValidation(unittest.TestCase):
    def setUp(self):
        self.conversion = convert_foams105_nv((1.0, 2.0), (2.0, 1.0))

    def test_canonical_conversion_rejects_altered_fields_and_provenance(self):
        mutations = (
            {"mean_projected_heights_mm": (1.0, 2.0)},
            {"signed_nv_per_mm3": (1.0, 2.0)},
            {"alpha_coefficients": (1.0, 2.0)},
            {"bin_labels_mm": (1.0, 3.0)},
            {"source_commit": "other"},
            {"output_density_unit": "cm^-3"},
        )
        for mutation in mutations:
            forged = replace(self.conversion, **mutation)
            with self.subTest(mutation=mutation), self.assertRaises(
                Foams105VolumeValidationError
            ) as caught:
                calculate_foams105_volume_distribution(forged)
            self.assertEqual(caught.exception.code, "inconsistent_conversion_result")

    def test_canonical_vesicularity_rejects_altered_result(self):
        vesicularity = _canonical_vesicularity()
        forged = replace(
            vesicularity,
            integrated_vesicularity_percent=(
                vesicularity.integrated_vesicularity_percent + 1.0
            ),
        )
        with self.assertRaises(Foams105VolumeValidationError) as caught:
            calculate_foams105_volume_distribution(
                self.conversion, "integrated_2d", vesicularity_result=forged
            )
        self.assertEqual(caught.exception.code, "inconsistent_vesicularity_result")

    def test_modes_reject_missing_contradictory_and_unused_targets(self):
        vesicularity = _canonical_vesicularity()
        calls = (
            ("other", {}, "unsupported_mode"),
            ("unnormalized", {"known_vesicularity_percent": 20.0}, "unused_target"),
            ("unnormalized", {"vesicularity_result": vesicularity}, "unused_target"),
            ("integrated_2d", {}, "missing_target"),
            (
                "integrated_2d",
                {
                    "vesicularity_result": vesicularity,
                    "known_vesicularity_percent": 20.0,
                },
                "unused_target",
            ),
            ("known_vesicularity", {}, "missing_target"),
            (
                "known_vesicularity",
                {
                    "vesicularity_result": vesicularity,
                    "known_vesicularity_percent": 20.0,
                },
                "unused_target",
            ),
        )
        for mode, kwargs, code in calls:
            with self.subTest(mode=mode, kwargs=kwargs), self.assertRaises(
                Foams105VolumeValidationError
            ) as caught:
                calculate_foams105_volume_distribution(
                    self.conversion, mode, **kwargs
                )
            self.assertEqual(caught.exception.code, code)

    def test_known_target_rejects_invalid_types_and_limits(self):
        invalid = (True, np.bool_(True), "40", math.nan, math.inf, -0.1, 100.1)
        for target in invalid:
            with self.subTest(target=target), self.assertRaises(
                Foams105VolumeValidationError
            ):
                calculate_foams105_volume_distribution(
                    self.conversion,
                    "known_vesicularity",
                    known_vesicularity_percent=target,
                )

    def test_public_api_requires_exact_result_types(self):
        with self.assertRaises(Foams105VolumeValidationError):
            calculate_foams105_volume_distribution(object())
        with self.assertRaises(Foams105VolumeValidationError):
            calculate_foams105_volume_distribution(
                self.conversion,
                "integrated_2d",
                vesicularity_result=object(),
            )


if __name__ == "__main__":
    unittest.main()
