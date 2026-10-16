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
from main.core.legacy_summaries import (
    DIAGNOSTIC_NEGATIVE_NV_ROW,
    DIAGNOSTIC_NEGATIVE_PER_WIDTH_DENSITY,
    DIAGNOSTIC_NEGATIVE_SOLID_FRACTION,
    FOAMS105_CATEGORY_BOUNDARY_POLICY,
    FOAMS105_CATEGORY_METHOD,
    FOAMS105_CUMULATIVE_POLICY,
    FOAMS105_MAX_EXACT_COUNT,
    FOAMS105_PAIRING_POLICY,
    FOAMS105_SUMMARY_METHOD,
    LEGACY_COMPLEX,
    LEGACY_INDETERMINATE,
    LEGACY_NEGATIVE_INFINITY,
    LEGACY_POSITIVE_INFINITY,
    REASON_LOG_OF_NEGATIVE,
    REASON_LOG_OF_ZERO,
    REASON_ZERO_BIN_WIDTH,
    REASON_ZERO_SOLID_FRACTION,
    REASON_ZERO_TOTAL_COUNT,
    Foams105CountGroupInput,
    Foams105SummaryNumericalError,
    Foams105SummaryValidationError,
    _calculate_foams105_distribution_numerical,
    summarize_foams105_distribution,
    summarize_foams105_size_categories,
)
from main.core.legacy_vesicularity import (
    Foams105VesicularityGroupInput,
    calculate_foams105_2d_vesicularity,
)
from main.core.legacy_volumes import calculate_foams105_volume_distribution

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "15_foams105_summary_cases.json"
FIXTURE_LF_SHA256 = "e2c3be0ce28f31072168d5aa6258ea9ba09d96a401bba7f516be87b1462f88c5"


def _assert_close(test_case, actual, expected):
    test_case.assertTrue(
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"{actual!r} != {expected!r}",
    )


def _assert_close_sequence(test_case, actual, expected):
    test_case.assertEqual(len(actual), len(expected))
    for candidate, reference in zip(actual, expected):
        _assert_close(test_case, candidate, reference)


def _values(records):
    return tuple(record.value for record in records)


def _reasons(records):
    return tuple(record.reason for record in records)


def _canonical_vesicularity():
    image = np.zeros((10, 10), dtype=np.uint8)
    image[3:5, 3:5] = 1
    measurement = measure_foams105_image(
        image, 1, 1.0, (None, None, None, None)
    )
    area = calculate_foams105_image_areas("summary-image", measurement)
    objects = tuple(
        Foams105RawObject(
            "summary-image",
            component.component_id,
            component.equivalent_diameter_mm,
            component.area_mm2,
        )
        for component in measurement.pore_result.components
    )
    raw = select_foams105_raw_objects(
        (0.1, 1.0, 10.0),
        (Foams105RawGroupInput("summary", 1, objects),),
        (Foams105BinnedRange("summary", 0.1, 10.0),),
    )
    return calculate_foams105_2d_vesicularity(
        raw,
        (Foams105VesicularityGroupInput("summary", 1, (area,), objects),),
    )


class TestFoams105SummaryFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_all_five_distribution_cases(self):
        self.assertEqual(len(self.fixture["distribution_cases"]), 5)
        for case in self.fixture["distribution_cases"]:
            with self.subTest(case=case["id"]):
                result = _calculate_foams105_distribution_numerical(
                    case["global_labels"],
                    case["labels"],
                    case["nv"],
                    case["adjusted"],
                    case["correction_percent"],
                )
                _assert_close_sequence(self, result.lower_edges_mm, case["lower_edges"])
                _assert_close_sequence(self, result.widths_mm, case["widths"])
                _assert_close_sequence(self, result.midsizes_mm, case["midsizes"])
                _assert_close_sequence(
                    self, result.cumulative_nv_per_mm3, case["cumulative_nv"]
                )
                if "per_width" in case:
                    self.assertEqual(len(result.number_density_per_width_mm4), len(case["per_width"]))
                    for actual, expected in zip(
                        result.number_density_per_width_mm4, case["per_width"]
                    ):
                        if expected is None:
                            self.assertIsNone(actual.value)
                        else:
                            _assert_close(self, actual.value, expected)
                if "ln_per_width" in case:
                    for actual, expected in zip(
                        result.ln_number_density_per_width, case["ln_per_width"]
                    ):
                        if expected is None:
                            self.assertIsNone(actual.value)
                        else:
                            _assert_close(self, actual.value, expected)
                if "ln_reasons" in case:
                    self.assertEqual(
                        _reasons(result.ln_number_density_per_width),
                        tuple(case["ln_reasons"]),
                    )
                if "log10_cumulative" in case:
                    for actual, expected in zip(
                        result.log10_cumulative_nv, case["log10_cumulative"]
                    ):
                        if expected is None:
                            self.assertIsNone(actual.value)
                        else:
                            _assert_close(self, actual.value, expected)
                if "log10_cumulative_reasons" in case:
                    self.assertEqual(
                        _reasons(result.log10_cumulative_nv),
                        tuple(case["log10_cumulative_reasons"]),
                    )
                _assert_close(self, result.total_nv_per_mm3, case["total_nv"])
                if case["corrected_nv"] is None:
                    self.assertIsNone(result.corrected_nv_per_mm3.value)
                    self.assertEqual(
                        result.corrected_nv_per_mm3.reason,
                        case["corrected_reason"],
                    )
                else:
                    _assert_close(
                        self, result.corrected_nv_per_mm3.value, case["corrected_nv"]
                    )
                self.assertEqual(
                    tuple(item.bin_label_mm for item in result.paired_volume_bins),
                    tuple(case["paired_labels"]),
                )
                _assert_close_sequence(
                    self,
                    tuple(
                        item.adjusted_volume_fraction
                        for item in result.paired_volume_bins
                    ),
                    case["paired_fractions"],
                )

    def test_all_four_category_cases(self):
        self.assertEqual(len(self.fixture["category_cases"]), 4)
        for case in self.fixture["category_cases"]:
            groups = tuple(
                Foams105CountGroupInput(f"group-{index}", index, tuple(counts))
                for index, counts in enumerate(case["counts"], 1)
            )
            with self.subTest(case=case["id"]):
                result = summarize_foams105_size_categories(
                    case["global_labels"], groups, case["lower"], case["upper"]
                )
                self.assertEqual(result.total_count, case["total"])
                self.assertEqual(
                    (result.lower_count, result.middle_count, result.upper_count),
                    tuple(case["category_counts"]),
                )
                self.assertEqual(result.boundary_count, case["boundary_count"])
                percentages = (
                    result.lower_percentage,
                    result.middle_percentage,
                    result.upper_percentage,
                )
                for actual, expected in zip(percentages, case["percentages"]):
                    if expected is None:
                        self.assertIsNone(actual.value)
                        self.assertEqual(actual.reason, case["reason"])
                    else:
                        _assert_close(self, actual.value, expected)

    def test_fixture_lf_hash_is_pinned(self):
        normalized = FIXTURE_PATH.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(sha256(normalized).hexdigest(), FIXTURE_LF_SHA256)

    def test_undefined_records_retain_legacy_classifications(self):
        zero_width = _calculate_foams105_distribution_numerical(
            (1.0, 2.0),
            (1.0, 1.0, 1.0),
            (2.0, -2.0, 0.0),
            (0.0, 0.0, 0.0),
            100.0,
        )
        self.assertEqual(
            tuple(item.legacy_classification for item in zero_width.number_density_per_width_mm4[1:]),
            (LEGACY_NEGATIVE_INFINITY, LEGACY_INDETERMINATE),
        )
        self.assertEqual(
            zero_width.ln_number_density_per_width[1].upstream_reason,
            REASON_ZERO_BIN_WIDTH,
        )
        self.assertEqual(
            zero_width.corrected_nv_per_mm3.legacy_classification,
            LEGACY_INDETERMINATE,
        )
        signed = _calculate_foams105_distribution_numerical(
            (1.0, 2.0), (2.0,), (1.0,), (0.0,), 100.0
        )
        self.assertEqual(
            signed.corrected_nv_per_mm3.legacy_classification,
            LEGACY_POSITIVE_INFINITY,
        )
        negative_solid = _calculate_foams105_distribution_numerical(
            (1.0, 2.0), (2.0,), (2.0,), (0.4,), 120.0
        )
        self.assertIn(
            DIAGNOSTIC_NEGATIVE_SOLID_FRACTION,
            tuple(item.code for item in negative_solid.diagnostics),
        )

    def test_log_domain_status_is_explicit_and_finite_values_remain_visible(self):
        result = _calculate_foams105_distribution_numerical(
            (1.0, 2.0, 4.0),
            (2.0, 4.0),
            (1.0, -1.0),
            (0.1, -0.1),
            0.0,
        )
        self.assertEqual(result.cumulative_nv_per_mm3, (0.0, -1.0))
        self.assertEqual(
            tuple(item.reason for item in result.log10_cumulative_nv),
            (REASON_LOG_OF_ZERO, REASON_LOG_OF_NEGATIVE),
        )
        self.assertEqual(
            tuple(item.legacy_classification for item in result.log10_cumulative_nv),
            (LEGACY_NEGATIVE_INFINITY, LEGACY_COMPLEX),
        )
        self.assertEqual(_values(result.number_density_per_width_mm4), (1.0, -0.5))


class TestFoams105SummaryNumericalFailures(unittest.TestCase):
    def test_forward_total_does_not_use_compensated_sum(self):
        result = _calculate_foams105_distribution_numerical(
            (1.0, 2.0, 4.0),
            (1.0, 2.0, 4.0),
            (1e16, 1.0, -1e16),
            (0.0, 0.0, 0.0),
            0.0,
        )
        self.assertEqual(result.total_nv_per_mm3, 0.0)

    def test_forward_total_and_reverse_cumulative_use_distinct_orders(self):
        result = _calculate_foams105_distribution_numerical(
            (1.0, 2.0, 4.0),
            (1.0, 2.0, 4.0),
            (1e16, -1e16, 1.0),
            (0.0, 0.0, 0.0),
            0.0,
        )
        self.assertEqual(result.total_nv_per_mm3, 1.0)
        self.assertEqual(
            result.cumulative_nv_per_mm3,
            (0.0, -1e16, 1.0),
        )

    def test_reverse_and_forward_sum_overflow_have_distinct_stages(self):
        with self.assertRaises(Foams105SummaryNumericalError) as reverse:
            _calculate_foams105_distribution_numerical(
                (1.0, 2.0), (1.0, 2.0), (1e308, 1e308), (0.0, 0.0), 0.0
            )
        self.assertEqual(reverse.exception.context["stage"], "reverse_cumulative_nv")
        with self.assertRaises(Foams105SummaryNumericalError) as forward:
            _calculate_foams105_distribution_numerical(
                (1e307, 1.1e307, 1.2e307),
                (1e307, 1.1e307, 1.2e307),
                (1e308, 1e308, -1e308),
                (0.0, 0.0, 0.0),
                0.0,
            )
        self.assertEqual(forward.exception.context["stage"], "forward_total_nv")

    def test_per_width_overflow_and_underflow_are_contextual(self):
        cases = (
            ((0.1, 0.2), (0.2,), (1e308,), "nonfinite_summary_arithmetic"),
            ((1.0, 3.0), (3.0,), (5e-324,), "nonzero_underflow"),
        )
        for grid, labels, nv, code in cases:
            with self.subTest(code=code), self.assertRaises(
                Foams105SummaryNumericalError
            ) as caught:
                _calculate_foams105_distribution_numerical(
                    grid, labels, nv, (0.0,), 0.0
                )
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(
                caught.exception.context["stage"], "number_density_per_width"
            )

    def test_correction_and_pair_arithmetic_failures_are_contextual(self):
        cases = (
            ((1.0,), (1.0,), (0.0,), (0.0,), 5e-324, "correction_percent_times_0_01"),
            ((1.0, 2.0), (1.0, 2.0), (0.0, 0.0), (1e308, 1e308), 0.0, "paired_adjusted_volume_fraction"),
        )
        for grid, labels, nv, adjusted, correction, stage in cases:
            with self.subTest(stage=stage), self.assertRaises(
                Foams105SummaryNumericalError
            ) as caught:
                _calculate_foams105_distribution_numerical(
                    grid, labels, nv, adjusted, correction
                )
            self.assertEqual(caught.exception.context["stage"], stage)

    def test_corrected_nv_overflow_and_underflow_are_contextual(self):
        cases = (
            ((1e307,), (1e307,), (1e308,), 99.9, "nonfinite_summary_arithmetic"),
            ((1.0, 1.5), (1.5,), (5e-324,), -100.0, "nonzero_underflow"),
        )
        for grid, labels, nv, correction, code in cases:
            with self.subTest(code=code), self.assertRaises(
                Foams105SummaryNumericalError
            ) as caught:
                _calculate_foams105_distribution_numerical(
                    grid, labels, nv, (0.0,), correction
                )
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(caught.exception.context["stage"], "corrected_nv")


class TestFoams105DistributionPublicAPI(unittest.TestCase):
    def setUp(self):
        self.vesicularity = _canonical_vesicularity()
        self.conversion = convert_foams105_nv((1.0, 2.0, 4.0), (8.0, 3.0, 1.0))

    def test_all_modes_select_the_required_correction_source(self):
        unnormalized_volume = calculate_foams105_volume_distribution(self.conversion)
        unnormalized = summarize_foams105_distribution(
            unnormalized_volume,
            self.conversion.bin_labels_mm,
            vesicularity_result=self.vesicularity,
        )
        integrated_volume = calculate_foams105_volume_distribution(
            self.conversion,
            "integrated_2d",
            vesicularity_result=self.vesicularity,
        )
        integrated = summarize_foams105_distribution(
            integrated_volume, self.conversion.bin_labels_mm
        )
        known_volume = calculate_foams105_volume_distribution(
            self.conversion,
            "known_vesicularity",
            known_vesicularity_percent=40.0,
        )
        known = summarize_foams105_distribution(
            known_volume, self.conversion.bin_labels_mm
        )
        _assert_close(self, unnormalized.correction_percent, 4.0)
        _assert_close(self, integrated.correction_percent, 4.0)
        self.assertEqual(known.correction_percent, 40.0)
        expected_forward_total = 0.0
        for value in self.conversion.signed_nv_per_mm3:
            expected_forward_total += value
        self.assertEqual(
            unnormalized.total_nv_per_mm3,
            expected_forward_total,
        )
        self.assertEqual(
            tuple(row.signed_nv_per_mm3 for row in known.rows),
            self.conversion.signed_nv_per_mm3,
        )

    def test_known_100_percent_reports_undefined_corrected_nv(self):
        volume = calculate_foams105_volume_distribution(
            self.conversion,
            "known_vesicularity",
            known_vesicularity_percent=100.0,
        )
        result = summarize_foams105_distribution(
            volume, self.conversion.bin_labels_mm
        )
        self.assertEqual(result.solid_fraction, 0.0)
        self.assertIsNone(result.corrected_nv_per_mm3.value)
        self.assertEqual(
            result.corrected_nv_per_mm3.reason, REASON_ZERO_SOLID_FRACTION
        )

    def test_predecessor_can_come_from_outside_selected_subset(self):
        conversion = convert_foams105_nv((2.0, 4.0), (2.0, 1.0))
        volume = calculate_foams105_volume_distribution(conversion)
        result = summarize_foams105_distribution(
            volume, (1.0, 2.0, 4.0), vesicularity_result=self.vesicularity
        )
        self.assertEqual(result.lower_edges_mm, (1.0, 2.0))

    def test_first_grid_label_fallback_duplicate_width_and_pair_provenance(self):
        conversion = convert_foams105_nv(
            (1.0, 2.0, 2.0, 4.0), (1.0, 0.0, 2.0, 3.0)
        )
        volume = calculate_foams105_volume_distribution(conversion)
        result = summarize_foams105_distribution(
            volume, (1.0, 2.0, 4.0), vesicularity_result=self.vesicularity
        )
        _assert_close(self, result.lower_edges_mm[0], 1.0 / (10.0**0.1))
        self.assertEqual(result.widths_mm[2], 0.0)
        self.assertEqual(
            result.number_density_per_width_mm4[2].reason, REASON_ZERO_BIN_WIDTH
        )
        self.assertEqual(
            tuple(item.source_row_indices for item in result.paired_volume_bins),
            ((0, 1), (2, 3)),
        )
        self.assertEqual(
            tuple(item.bin_label_mm for item in result.paired_volume_bins),
            (2.0, 4.0),
        )

    def test_signed_values_have_row_eligibility_without_dropping_rows(self):
        conversion = convert_foams105_nv(
            (1.0, 2.0, 2.0, 4.0), (1.0, 0.0, 2.0, 3.0)
        )
        volume = calculate_foams105_volume_distribution(conversion)
        result = summarize_foams105_distribution(
            volume, (1.0, 2.0, 4.0), vesicularity_result=self.vesicularity
        )
        codes = tuple(item.code for item in result.diagnostics)
        self.assertIn(DIAGNOSTIC_NEGATIVE_NV_ROW, codes)
        self.assertIn(DIAGNOSTIC_NEGATIVE_PER_WIDTH_DENSITY, codes)
        self.assertEqual(len(result.rows), 4)
        self.assertFalse(result.rows[1].ln_density_eligible)
        self.assertEqual(result.rows[1].ln_density_reason, REASON_LOG_OF_NEGATIVE)

    def test_singleton_and_odd_pairing(self):
        for labels, densities, expected_indices in (
            ((1.0,), (1.0,), ((0,),)),
            ((1.0, 2.0, 4.0), (3.0, 2.0, 1.0), ((0, 1), (2,))),
        ):
            conversion = convert_foams105_nv(labels, densities)
            volume = calculate_foams105_volume_distribution(conversion)
            result = summarize_foams105_distribution(
                volume, labels, vesicularity_result=self.vesicularity
            )
            self.assertEqual(
                tuple(item.source_row_indices for item in result.paired_volume_bins),
                expected_indices,
            )

    def test_metadata_and_results_are_immutable_owned_records(self):
        grid = [1.0, 2.0, 4.0]
        volume = calculate_foams105_volume_distribution(self.conversion)
        result = summarize_foams105_distribution(
            volume, grid, vesicularity_result=self.vesicularity
        )
        grid[0] = 99.0
        self.assertEqual(result.global_bin_labels_mm, (1.0, 2.0, 4.0))
        self.assertEqual(result.method, FOAMS105_SUMMARY_METHOD)
        self.assertEqual(result.cumulative_policy, FOAMS105_CUMULATIVE_POLICY)
        self.assertEqual(result.pairing_policy, FOAMS105_PAIRING_POLICY)
        with self.assertRaises(FrozenInstanceError):
            result.total_nv_per_mm3 = 0.0


class TestFoams105DistributionValidation(unittest.TestCase):
    def setUp(self):
        self.vesicularity = _canonical_vesicularity()
        self.conversion = convert_foams105_nv((1.0, 2.0), (2.0, 1.0))
        self.volume = calculate_foams105_volume_distribution(self.conversion)

    def test_requires_canonical_volume_and_vesicularity_records(self):
        with self.assertRaises(Foams105SummaryValidationError):
            summarize_foams105_distribution(object(), (1.0, 2.0))
        forged_volume = replace(self.volume, total_nv_per_mm3=1.0) if hasattr(self.volume, "total_nv_per_mm3") else replace(self.volume, raw_fraction_sum=self.volume.raw_fraction_sum + 1.0)
        with self.assertRaises(Foams105SummaryValidationError) as caught:
            summarize_foams105_distribution(
                forged_volume, (1.0, 2.0), vesicularity_result=self.vesicularity
            )
        self.assertEqual(caught.exception.code, "inconsistent_volume_result")
        forged_vesicularity = replace(
            self.vesicularity,
            integrated_vesicularity_percent=(
                self.vesicularity.integrated_vesicularity_percent + 1.0
            ),
        )
        with self.assertRaises(Foams105SummaryValidationError) as caught:
            summarize_foams105_distribution(
                self.volume, (1.0, 2.0), vesicularity_result=forged_vesicularity
            )
        self.assertEqual(caught.exception.code, "inconsistent_vesicularity_result")

    def test_mode_specific_vesicularity_parameters_are_strict(self):
        with self.assertRaises(Foams105SummaryValidationError) as missing:
            summarize_foams105_distribution(self.volume, (1.0, 2.0))
        self.assertEqual(missing.exception.code, "missing_vesicularity_result")
        integrated = calculate_foams105_volume_distribution(
            self.conversion,
            "integrated_2d",
            vesicularity_result=self.vesicularity,
        )
        known = calculate_foams105_volume_distribution(
            self.conversion,
            "known_vesicularity",
            known_vesicularity_percent=40.0,
        )
        for volume in (integrated, known):
            with self.subTest(mode=volume.mode), self.assertRaises(
                Foams105SummaryValidationError
            ) as caught:
                summarize_foams105_distribution(
                    volume, (1.0, 2.0), vesicularity_result=self.vesicularity
                )
            self.assertEqual(caught.exception.code, "unused_vesicularity_result")

    def test_grid_rejects_absent_ambiguous_descending_and_invalid_values(self):
        invalid = (
            ((1.0, 3.0), "selected_label_absent_from_global_grid"),
            ((1.0, 2.0, 2.0), "ambiguous_global_grid"),
            ((2.0, 1.0), "ambiguous_global_grid"),
            ((1.0, math.nan), "invalid_input"),
            ((1.0, True), "invalid_input"),
        )
        for grid, code in invalid:
            with self.subTest(grid=grid), self.assertRaises(
                Foams105SummaryValidationError
            ) as caught:
                summarize_foams105_distribution(
                    self.volume, grid, vesicularity_result=self.vesicularity
                )
            self.assertEqual(caught.exception.code, code)


class TestFoams105SizeCategories(unittest.TestCase):
    def test_four_groups_are_accepted_and_five_are_rejected(self):
        four_groups = tuple(
            Foams105CountGroupInput(f"group-{slot}", slot, (slot,))
            for slot in range(1, 5)
        )
        result = summarize_foams105_size_categories(
            (1.0,), four_groups, 1.0, 1.0
        )
        self.assertEqual(result.total_count, 10)
        five_groups = four_groups + (Foams105CountGroupInput("group-5", 5, (5,)),)
        with self.assertRaises(Foams105SummaryValidationError):
            summarize_foams105_size_categories(
                (1.0,), five_groups, 1.0, 1.0
            )

    def test_empty_groups_contribute_owned_zero_vectors(self):
        counts = [0, 0, 0]
        result = summarize_foams105_size_categories(
            (1.0, 2.0, 3.0),
            (
                Foams105CountGroupInput("empty", 1, counts),
                Foams105CountGroupInput("active", 2, (1, 2, 3)),
            ),
            1.0,
            3.0,
        )
        counts[0] = 99
        self.assertEqual(result.count_groups[0].counts, (0, 0, 0))
        self.assertEqual(result.pooled_counts, (1, 2, 3))
        self.assertEqual(result.boundary_indices, (0, 2))
        self.assertEqual(result.method, FOAMS105_CATEGORY_METHOD)
        self.assertEqual(result.boundary_policy, FOAMS105_CATEGORY_BOUNDARY_POLICY)

    def test_numpy_integers_are_normalized_but_booleans_and_floats_rejected(self):
        accepted = summarize_foams105_size_categories(
            (1.0,),
            (Foams105CountGroupInput("numpy", np.int64(1), (np.int64(2),)),),
            1.0,
            1.0,
        )
        self.assertEqual(accepted.count_groups[0], Foams105CountGroupInput("numpy", 1, (2,)))
        for value in (True, np.bool_(True), 1.0, -1):
            with self.subTest(value=value), self.assertRaises(
                Foams105SummaryValidationError
            ):
                summarize_foams105_size_categories(
                    (1.0,),
                    (Foams105CountGroupInput("bad", 1, (value,)),),
                    1.0,
                    1.0,
                )

    def test_group_identity_slot_length_and_count_limits_are_validated(self):
        cases = (
            ((), "invalid_input"),
            ((Foams105CountGroupInput("a", 2, (1,)),), "invalid_input"),
            (
                (
                    Foams105CountGroupInput("a", 1, (1,)),
                    Foams105CountGroupInput("a", 2, (1,)),
                ),
                "duplicate_group_id",
            ),
            ((Foams105CountGroupInput("a", 1, (1, 2)),), "invalid_input"),
            (
                (Foams105CountGroupInput("a", 1, (FOAMS105_MAX_EXACT_COUNT + 1,)),),
                "count_total_exceeds_exact_limit",
            ),
        )
        for groups, code in cases:
            with self.subTest(code=code), self.assertRaises(
                Foams105SummaryValidationError
            ) as caught:
                summarize_foams105_size_categories((1.0,), groups, 1.0, 1.0)
            self.assertEqual(caught.exception.code, code)

    def test_cutoffs_and_global_grid_are_validated(self):
        group = (Foams105CountGroupInput("a", 1, (1, 2)),)
        for lower, upper in ((0.0, 1.0), (2.0, 1.0), (True, 2.0), (1.0, math.inf)):
            with self.subTest(lower=lower, upper=upper), self.assertRaises(
                Foams105SummaryValidationError
            ):
                summarize_foams105_size_categories(
                    (1.0, 2.0), group, lower, upper
                )

    def test_zero_total_has_explicit_undefined_percentages(self):
        result = summarize_foams105_size_categories(
            (1.0, 2.0),
            (Foams105CountGroupInput("empty", 1, (0, 0)),),
            1.0,
            2.0,
        )
        for percentage in (
            result.lower_percentage,
            result.middle_percentage,
            result.upper_percentage,
        ):
            self.assertIsNone(percentage.value)
            self.assertEqual(percentage.reason, REASON_ZERO_TOTAL_COUNT)
            self.assertEqual(percentage.legacy_classification, LEGACY_INDETERMINATE)


if __name__ == "__main__":
    unittest.main()
