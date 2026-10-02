import math
import os
import sys
import unittest
from dataclasses import replace
from types import MappingProxyType

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.distributions import (
    DETECTION_ELIGIBILITY_POLICY,
    DiameterBin,
    DistributionDiagnostics,
    DistributionResult,
    Group2DDistribution,
    create_diameter_bin_spec,
)
from core.nesting import (
    NESTING_METHOD,
    NestingPlan,
    NestingSegment,
    NestingValidationError,
    nest_2d_distribution,
)
from core.sampling import create_image_sampling_record
from examples.manual_nesting import _format_interval


def make_source_image(
    image_index,
    image_id,
    group_id,
    area_mm2,
    *,
    sample_id="sample",
    minimum=0.1,
    maximum=1.6,
    parent_image_id=None,
):
    width = int(round(area_mm2 * 100))
    labels = np.zeros((100, width), dtype=np.int32)
    return create_image_sampling_record(
        sample_id=sample_id,
        image_id=image_id,
        image_index=image_index,
        magnification_group_id=group_id,
        label_map=labels,
        calibration=0.01,
        calibration_unit="mm",
        min_detectable_diameter=minimum,
        max_reliable_diameter=maximum,
        parent_image_id=parent_image_id,
    )


def make_group(sample_id, group_id, bin_spec, images, counts):
    areas = []
    contributors = []
    for bin_ in bin_spec.bins:
        eligible = tuple(
            image
            for image in images
            if image.sample_id == sample_id
            and image.magnification_group_id == group_id
            and image.min_detectable_diameter_mm <= bin_.lower_mm
            and (
                image.max_reliable_diameter_mm is None
                or image.max_reliable_diameter_mm >= bin_.upper_mm
            )
        )
        areas.append(math.fsum(image.analyzed_area_mm2 for image in eligible))
        contributors.append(tuple((image.sample_id, image.image_id) for image in eligible))
    supported = tuple(area > 0 for area in areas)
    densities = tuple(
        count / area if flag else math.nan
        for count, area, flag in zip(counts, areas, supported)
    )
    return Group2DDistribution(
        sample_id=sample_id,
        magnification_group_id=group_id,
        counts=tuple(counts),
        eligible_image_counts=tuple(len(value) for value in contributors),
        eligible_areas_mm2=tuple(areas),
        number_densities_per_mm2=densities,
        supported=supported,
        contributing_images=tuple(contributors),
    )


def make_result(bin_spec, groups, images, *, diagnostics=None):
    return DistributionResult(
        bin_spec=bin_spec,
        groups=MappingProxyType(
            {(group.sample_id, group.magnification_group_id): group for group in groups}
        ),
        diagnostics=diagnostics or DistributionDiagnostics(),
        exclude_border=False,
        source_images=tuple(images),
    )


class TestManualMagnificationNesting(unittest.TestCase):
    def setUp(self):
        self.bins = create_diameter_bin_spec([0.1, 0.2, 0.4, 0.8, 1.6])

    def worked_result(self):
        fine_image = make_source_image(0, "fine-image", "fine", 1.0)
        coarse_image = make_source_image(1, "coarse-image", "coarse", 10.0)
        fine = make_group("sample", "fine", self.bins, (fine_image,), (20, 10, 5, 2))
        coarse = make_group(
            "sample", "coarse", self.bins, (coarse_image,), (180, 120, 40, 10)
        )
        return make_result(self.bins, (fine, coarse), (fine_image, coarse_image))

    def worked_plan(self):
        return NestingPlan(
            "sample",
            0,
            4,
            (NestingSegment("fine", 0, 2), NestingSegment("coarse", 2, 4)),
        )

    def test_worked_fixture_preserves_source_rows_and_transition_ownership(self):
        source = self.worked_result()
        nested = nest_2d_distribution(source, self.worked_plan())

        self.assertEqual(nested.counts, (20, 10, 40, 10))
        self.assertEqual(nested.eligible_areas_mm2, (1.0, 1.0, 10.0, 10.0))
        self.assertEqual(nested.number_densities_per_mm2, (20.0, 10.0, 4.0, 1.0))
        self.assertEqual(nested.source_group_ids, ("fine", "fine", "coarse", "coarse"))
        self.assertEqual(nested.selected_bin_indices, (0, 1, 2, 3))
        self.assertEqual(nested.transitions[0].edge_index, 2)
        self.assertEqual(nested.transitions[0].diameter_mm, 0.4)
        self.assertEqual(nested.method, NESTING_METHOD)
        self.assertEqual(nested.density_unit, "mm^-2")
        self.assertEqual(
            nested.detection_eligibility_policy, DETECTION_ELIGIBILITY_POLICY
        )
        self.assertEqual(source.density_unit, "mm^-2")

        comparisons = nested.overlap_diagnostics[0].comparisons
        self.assertEqual(nested.overlap_diagnostics[0].common_supported_bin_count, 4)
        self.assertEqual(nested.overlap_diagnostics[0].informative_bin_count, 4)
        expected = (-2 / 19, 2 / 11, -2 / 9, -2 / 3)
        self.assertEqual(tuple(item.bin_index for item in comparisons), (0, 1, 2, 3))
        for comparison, value in zip(comparisons, expected):
            self.assertAlmostEqual(comparison.symmetric_relative_difference, value)

    def test_three_groups_two_transitions_and_varying_areas(self):
        fine_image = make_source_image(0, "fine", "fine", 1.0, maximum=0.4)
        medium_image = make_source_image(
            1, "medium", "medium", 2.0, minimum=0.2, maximum=0.8
        )
        coarse_image = make_source_image(
            2, "coarse", "coarse", 3.0, minimum=0.4, maximum=1.6
        )
        groups = (
            make_group("sample", "fine", self.bins, (fine_image,), (2, 1, 0, 0)),
            make_group("sample", "medium", self.bins, (medium_image,), (0, 6, 4, 0)),
            make_group("sample", "coarse", self.bins, (coarse_image,), (0, 0, 9, 12)),
        )
        result = make_result(
            self.bins, groups, (fine_image, medium_image, coarse_image)
        )
        plan = NestingPlan(
            "sample",
            0,
            4,
            (
                NestingSegment("fine", 0, 1),
                NestingSegment("medium", 1, 3),
                NestingSegment("coarse", 3, 4),
            ),
        )

        nested = nest_2d_distribution(result, plan)

        self.assertEqual(nested.counts, (2, 6, 4, 12))
        self.assertEqual(nested.eligible_areas_mm2, (1.0, 2.0, 2.0, 3.0))
        self.assertEqual(nested.number_densities_per_mm2, (2.0, 3.0, 2.0, 4.0))
        self.assertEqual(tuple(item.edge_index for item in nested.transitions), (1, 3))

    def test_single_group_identity_slice_and_restricted_range(self):
        source = self.worked_result()
        plan = NestingPlan("sample", 1, 3, (NestingSegment("fine", 1, 3),))

        nested = nest_2d_distribution(source, plan)

        fine = source.groups[("sample", "fine")]
        self.assertEqual(nested.selected_bin_indices, (1, 2))
        self.assertEqual(nested.counts, fine.counts[1:3])
        self.assertEqual(nested.eligible_areas_mm2, fine.eligible_areas_mm2[1:3])
        self.assertEqual(nested.transitions, ())
        self.assertEqual(nested.overlap_diagnostics, ())
        self.assertIs(nested.bin_spec, source.bin_spec)

    def test_invalid_plan_shapes_and_indices_fail_clearly(self):
        source = self.worked_result()
        invalid_plans = (
            NestingPlan("sample", 0, 4, (NestingSegment("fine", 0, 1),)),
            NestingPlan(
                "sample", 0, 4,
                (NestingSegment("fine", 0, 1), NestingSegment("coarse", 2, 4)),
            ),
            NestingPlan(
                "sample", 0, 4,
                (NestingSegment("fine", 0, 3), NestingSegment("coarse", 2, 4)),
            ),
            NestingPlan("sample", 0, 4, (NestingSegment("fine", 2, 1),)),
            NestingPlan("sample", 0, 4, (NestingSegment("fine", 0, 0),)),
            NestingPlan(
                "sample", 0, 4,
                (NestingSegment("fine", 0, 2), NestingSegment("fine", 2, 4)),
            ),
            NestingPlan("sample", True, 4, (NestingSegment("fine", 0, 4),)),
            NestingPlan("sample", 0, 4, (NestingSegment("fine", False, 4),)),
        )
        for plan in invalid_plans:
            with self.subTest(plan=plan), self.assertRaises(NestingValidationError):
                nest_2d_distribution(source, plan)

    def test_missing_and_cross_sample_groups_fail(self):
        source = self.worked_result()
        with self.assertRaisesRegex(NestingValidationError, "missing"):
            nest_2d_distribution(
                source,
                NestingPlan("sample", 0, 1, (NestingSegment("unknown", 0, 1),)),
            )

        other_image = make_source_image(
            2, "other", "shared", 1.0, sample_id="other"
        )
        other_group = make_group(
            "other", "shared", self.bins, (other_image,), (0, 0, 0, 0)
        )
        other_result = make_result(self.bins, (other_group,), (other_image,))
        with self.assertRaisesRegex(NestingValidationError, "another sample"):
            nest_2d_distribution(
                other_result,
                NestingPlan("sample", 0, 1, (NestingSegment("shared", 0, 1),)),
            )

    def test_selected_unsupported_bin_fails_without_fallback(self):
        fine_image = make_source_image(0, "fine", "fine", 1.0, maximum=0.4)
        coarse_image = make_source_image(1, "coarse", "coarse", 1.0)
        fine = make_group("sample", "fine", self.bins, (fine_image,), (0, 0, 0, 0))
        coarse = make_group("sample", "coarse", self.bins, (coarse_image,), (0, 0, 0, 0))
        source = make_result(self.bins, (fine, coarse), (fine_image, coarse_image))

        with self.assertRaisesRegex(NestingValidationError, "unsupported bin 2"):
            nest_2d_distribution(
                source,
                NestingPlan("sample", 2, 3, (NestingSegment("fine", 2, 3),)),
            )

    def test_unsupported_bins_outside_plan_range_are_allowed(self):
        image = make_source_image(
            0, "middle", "middle", 2.0, minimum=0.2, maximum=0.8
        )
        group = make_group("sample", "middle", self.bins, (image,), (0, 3, 4, 0))
        nested = nest_2d_distribution(
            make_result(self.bins, (group,), (image,)),
            NestingPlan("sample", 1, 3, (NestingSegment("middle", 1, 3),)),
        )
        self.assertEqual(nested.counts, (3, 4))

    def test_zero_density_overlap_cases(self):
        left_image = make_source_image(0, "left", "left", 1.0)
        right_image = make_source_image(1, "right", "right", 1.0)
        left = make_group("sample", "left", self.bins, (left_image,), (0, 1, 1, 1))
        right = make_group("sample", "right", self.bins, (right_image,), (0, 0, 1, 1))
        source = make_result(self.bins, (left, right), (left_image, right_image))
        plan = NestingPlan(
            "sample", 0, 4,
            (NestingSegment("left", 0, 2), NestingSegment("right", 2, 4)),
        )

        nested = nest_2d_distribution(source, plan)
        comparisons = nested.overlap_diagnostics[0].comparisons

        self.assertEqual(nested.number_densities_per_mm2[0], 0.0)
        self.assertTrue(comparisons[0].both_zero)
        self.assertEqual(comparisons[0].delta_per_mm2, 0.0)
        self.assertIsNone(comparisons[0].symmetric_relative_difference)
        self.assertFalse(comparisons[1].both_zero)
        self.assertEqual(comparisons[1].symmetric_relative_difference, -2.0)

    def test_no_shared_support_is_advisory_and_unbracketed(self):
        fine_image = make_source_image(0, "fine", "fine", 1.0, maximum=0.4)
        coarse_image = make_source_image(
            1, "coarse", "coarse", 2.0, minimum=0.4, maximum=1.6
        )
        fine = make_group("sample", "fine", self.bins, (fine_image,), (1, 1, 0, 0))
        coarse = make_group("sample", "coarse", self.bins, (coarse_image,), (0, 0, 2, 2))
        nested = nest_2d_distribution(
            make_result(self.bins, (fine, coarse), (fine_image, coarse_image)),
            NestingPlan(
                "sample", 0, 4,
                (NestingSegment("fine", 0, 2), NestingSegment("coarse", 2, 4)),
            ),
        )
        diagnostic = nested.overlap_diagnostics[0]
        self.assertTrue(diagnostic.no_shared_supported_bins)
        self.assertTrue(diagnostic.no_informative_overlap)
        self.assertTrue(diagnostic.transition_not_bracketed_by_shared_support)
        self.assertFalse(diagnostic.below_transition_common_supported)
        self.assertFalse(diagnostic.above_transition_common_supported)

    def test_no_informative_overlap_is_separate_from_shared_support(self):
        left_image = make_source_image(0, "left", "left", 1.0)
        right_image = make_source_image(1, "right", "right", 1.0)
        left = make_group("sample", "left", self.bins, (left_image,), (0, 0, 1, 1))
        right = make_group("sample", "right", self.bins, (right_image,), (0, 0, 2, 2))
        nested = nest_2d_distribution(
            make_result(self.bins, (left, right), (left_image, right_image)),
            NestingPlan(
                "sample", 0, 2,
                (NestingSegment("left", 0, 1), NestingSegment("right", 1, 2)),
            ),
        )
        diagnostic = nested.overlap_diagnostics[0]
        self.assertFalse(diagnostic.no_shared_supported_bins)
        self.assertTrue(diagnostic.no_informative_overlap)
        self.assertFalse(diagnostic.transition_not_bracketed_by_shared_support)

    def test_one_sided_shared_support_reports_unbracketed_transition(self):
        left_image = make_source_image(0, "left", "left", 1.0, maximum=0.4)
        right_image = make_source_image(
            1, "right", "right", 1.0, minimum=0.2, maximum=1.6
        )
        left = make_group("sample", "left", self.bins, (left_image,), (1, 1, 0, 0))
        right = make_group("sample", "right", self.bins, (right_image,), (0, 1, 1, 1))
        nested = nest_2d_distribution(
            make_result(self.bins, (left, right), (left_image, right_image)),
            NestingPlan(
                "sample", 0, 4,
                (NestingSegment("left", 0, 2), NestingSegment("right", 2, 4)),
            ),
        )
        diagnostic = nested.overlap_diagnostics[0]
        self.assertEqual(diagnostic.common_supported_bin_indices, (1,))
        self.assertTrue(diagnostic.below_transition_common_supported)
        self.assertFalse(diagnostic.above_transition_common_supported)
        self.assertTrue(diagnostic.transition_not_bracketed_by_shared_support)

    def test_counts_areas_and_densities_never_pool(self):
        nested = nest_2d_distribution(self.worked_result(), self.worked_plan())
        self.assertEqual(nested.counts[2], 40)
        self.assertEqual(nested.eligible_areas_mm2[2], 10.0)
        self.assertEqual(nested.number_densities_per_mm2[2], 4.0)
        self.assertNotEqual(nested.counts[2], 45)
        self.assertNotEqual(nested.eligible_areas_mm2[2], 11.0)

    def test_provenance_diagnostics_and_inputs_remain_immutable(self):
        diagnostic = DistributionDiagnostics(unsupported_by_image=((0, 7),))
        source = self.worked_result()
        source = replace(source, diagnostics=diagnostic)
        before_images = source.source_images
        before_groups = tuple(source.groups.items())

        nested = nest_2d_distribution(source, self.worked_plan())

        self.assertEqual(nested.source_images, source.source_images)
        self.assertIs(nested.source_diagnostics, diagnostic)
        self.assertEqual(source.source_images, before_images)
        self.assertEqual(tuple(source.groups.items()), before_groups)
        self.assertEqual(nested.contributing_images[0], (("sample", "fine-image"),))
        self.assertIsInstance(nested.bin_spec.edges_mm, tuple)
        self.assertIsInstance(nested.bin_spec.bins, tuple)
        self.assertIsInstance(nested.source_images, tuple)
        self.assertIsInstance(nested.source_diagnostics.unsupported_by_image, tuple)
        self.assertTrue(
            all(isinstance(image.included_labels, tuple) for image in nested.source_images)
        )

    def test_example_interval_notation_uses_absolute_source_bin(self):
        self.assertEqual(_format_interval(0, 0.1, 0.2, 4), "[0.1, 0.2)")
        self.assertEqual(_format_interval(2, 0.4, 0.8, 4), "[0.4, 0.8)")
        self.assertEqual(_format_interval(3, 0.8, 1.6, 4), "[0.8, 1.6]")

    def test_inconsistent_direct_source_data_fails_validation(self):
        source = self.worked_result()
        fine = source.groups[("sample", "fine")]
        cases = (
            replace(
                fine,
                number_densities_per_mm2=(999.0,) + fine.number_densities_per_mm2[1:],
            ),
            replace(
                fine,
                contributing_images=((('sample', 'missing'),),) + fine.contributing_images[1:],
            ),
            replace(
                fine,
                eligible_image_counts=(2,) + fine.eligible_image_counts[1:],
            ),
            replace(fine, counts=fine.counts[:-1]),
            replace(
                fine,
                eligible_areas_mm2=("invalid",) + fine.eligible_areas_mm2[1:],
            ),
        )
        for invalid in cases:
            groups = dict(source.groups)
            groups[("sample", "fine")] = invalid
            malformed = replace(source, groups=MappingProxyType(groups))
            with self.subTest(invalid=invalid), self.assertRaises(NestingValidationError):
                nest_2d_distribution(malformed, self.worked_plan())

    def test_ambiguous_source_image_provenance_fails_validation(self):
        source = self.worked_result()
        duplicate_index = replace(source.source_images[1], image_index=0)
        malformed = replace(
            source, source_images=(source.source_images[0], duplicate_index)
        )
        with self.assertRaisesRegex(NestingValidationError, "image_index"):
            nest_2d_distribution(malformed, self.worked_plan())

        with self.assertRaisesRegex(NestingValidationError, "mapping"):
            nest_2d_distribution(
                replace(source, groups=()), self.worked_plan()
            )

    def test_unsupported_scientific_metadata_fails_without_mutation(self):
        source = self.worked_result()
        cases = (
            ("density_unit", ""),
            ("density_unit", None),
            ("density_unit", "unknown"),
            ("density_unit", "um^-2"),
            ("detection_eligibility_policy", ""),
            ("detection_eligibility_policy", None),
            ("detection_eligibility_policy", "unknown"),
            ("detection_eligibility_policy", "center-bin coverage"),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                malformed = replace(source, **{field: value})
                with self.assertRaisesRegex(NestingValidationError, field):
                    nest_2d_distribution(malformed, self.worked_plan())
                self.assertEqual(getattr(malformed, field), value)

    def test_mutable_or_malformed_grid_collections_fail(self):
        source = self.worked_result()
        malformed_specs = (
            replace(source.bin_spec, edges_mm=list(source.bin_spec.edges_mm)),
            replace(source.bin_spec, bins=list(source.bin_spec.bins)),
            replace(source.bin_spec, bins=(None,) + source.bin_spec.bins[1:]),
            replace(
                source.bin_spec,
                edges_mm=("0.1",) + source.bin_spec.edges_mm[1:],
            ),
            replace(
                source.bin_spec,
                bins=(
                    DiameterBin("0.1", 0.2, 0.1, math.sqrt(0.02)),
                ) + source.bin_spec.bins[1:],
            ),
        )
        for bin_spec in malformed_specs:
            with self.subTest(bin_spec=bin_spec), self.assertRaises(
                NestingValidationError
            ):
                nest_2d_distribution(
                    replace(source, bin_spec=bin_spec), self.worked_plan()
                )

    def test_mutable_diagnostics_and_image_label_collections_fail(self):
        source = self.worked_result()
        mutable_diagnostics = replace(
            source.diagnostics, unsupported_by_image=[(0, 7)]
        )
        with self.assertRaisesRegex(NestingValidationError, "unsupported_by_image"):
            nest_2d_distribution(
                replace(source, diagnostics=mutable_diagnostics), self.worked_plan()
            )

        malformed_diagnostics = replace(
            source.diagnostics, unsupported_by_image=((0, True),)
        )
        with self.assertRaisesRegex(NestingValidationError, "unsupported_by_image"):
            nest_2d_distribution(
                replace(source, diagnostics=malformed_diagnostics), self.worked_plan()
            )

        for field in ("included_labels", "omitted_labels"):
            image = replace(source.source_images[0], **{field: [1]})
            malformed = replace(
                source, source_images=(image,) + source.source_images[1:]
            )
            with self.subTest(field=field), self.assertRaisesRegex(
                NestingValidationError, field
            ):
                nest_2d_distribution(malformed, self.worked_plan())

        duplicate_labels = replace(
            source.source_images[0], included_labels=(1, 1)
        )
        with self.assertRaisesRegex(NestingValidationError, "duplicate labels"):
            nest_2d_distribution(
                replace(
                    source,
                    source_images=(duplicate_labels,) + source.source_images[1:],
                ),
                self.worked_plan(),
            )

    def test_supported_bin_requires_real_positive_source_support(self):
        source = self.worked_result()
        fine = source.groups[("sample", "fine")]
        fabricated = replace(
            fine,
            eligible_image_counts=(0,) * 4,
            contributing_images=((),) * 4,
            eligible_areas_mm2=(1e-16,) * 4,
            number_densities_per_mm2=tuple(count / 1e-16 for count in fine.counts),
        )
        groups = dict(source.groups)
        groups[("sample", "fine")] = fabricated
        malformed = replace(
            source,
            groups=MappingProxyType(groups),
            source_images=(source.source_images[1],),
        )
        with self.assertRaisesRegex(NestingValidationError, "positive source support"):
            nest_2d_distribution(malformed, self.worked_plan())

    def test_zero_area_source_fails_even_when_group_total_is_positive(self):
        labels = np.zeros((1, 1), dtype=np.int32)
        tiny = create_image_sampling_record(
            sample_id="sample",
            image_id="tiny",
            image_index=10,
            magnification_group_id="fine",
            label_map=labels,
            calibration=1e-9,
            calibration_unit="mm",
            min_detectable_diameter=0.1,
            max_reliable_diameter=1.6,
        )
        positive = make_source_image(11, "positive", "fine", 1.0)
        hidden_zero = replace(tiny, analyzed_area_mm2=0.0)
        group = make_group(
            "sample", "fine", self.bins, (hidden_zero, positive), (0, 0, 0, 0)
        )
        source = make_result(self.bins, (group,), (hidden_zero, positive))
        with self.assertRaisesRegex(NestingValidationError, "analyzed_area_mm2"):
            nest_2d_distribution(
                source,
                NestingPlan("sample", 0, 1, (NestingSegment("fine", 0, 1),)),
            )

    def test_tiny_positive_area_is_valid_but_relative_mismatch_fails(self):
        labels = np.zeros((1, 1), dtype=np.int32)
        image = create_image_sampling_record(
            sample_id="tiny",
            image_id="tiny-image",
            image_index=10,
            magnification_group_id="tiny-group",
            label_map=labels,
            calibration=1e-9,
            calibration_unit="mm",
            min_detectable_diameter=0.1,
            max_reliable_diameter=1.6,
        )
        group = make_group("tiny", "tiny-group", self.bins, (image,), (0, 0, 0, 0))
        source = make_result(self.bins, (group,), (image,))
        plan = NestingPlan("tiny", 0, 1, (NestingSegment("tiny-group", 0, 1),))

        nested = nest_2d_distribution(source, plan)
        self.assertEqual(nested.eligible_areas_mm2, (1e-18,))
        self.assertEqual(nested.number_densities_per_mm2, (0.0,))

        mismatched = replace(
            group,
            eligible_areas_mm2=(1.01e-18,) + group.eligible_areas_mm2[1:],
        )
        with self.assertRaisesRegex(NestingValidationError, "eligible area"):
            nest_2d_distribution(
                make_result(self.bins, (mismatched,), (image,)), plan
            )

    def test_source_area_accumulation_overflow_is_contextual(self):
        labels = np.zeros((1, 1), dtype=np.int32)
        images = tuple(
            create_image_sampling_record(
                sample_id="huge",
                image_id=f"huge-{index}",
                image_index=20 + index,
                magnification_group_id="huge-group",
                label_map=labels,
                calibration=1e154,
                calibration_unit="mm",
                min_detectable_diameter=0.1,
                max_reliable_diameter=1.6,
            )
            for index in range(2)
        )
        contributors = tuple((image.sample_id, image.image_id) for image in images)
        group = Group2DDistribution(
            sample_id="huge",
            magnification_group_id="huge-group",
            counts=(0, 0, 0, 0),
            eligible_image_counts=(2, 2, 2, 2),
            eligible_areas_mm2=(1e308, 1e308, 1e308, 1e308),
            number_densities_per_mm2=(0.0, 0.0, 0.0, 0.0),
            supported=(True, True, True, True),
            contributing_images=(contributors,) * 4,
        )
        source = make_result(self.bins, (group,), images)
        plan = NestingPlan(
            "huge", 0, 1, (NestingSegment("huge-group", 0, 1),)
        )
        with self.assertRaisesRegex(
            NestingValidationError, "bin 0.*accumulation overflowed"
        ):
            nest_2d_distribution(source, plan)

    def test_malformed_group_mapping_keys_fail_with_domain_error(self):
        source = self.worked_result()
        malformed_keys = (None, ("sample",), "sample:fine", ("sample", "fine", "extra"))
        for key in malformed_keys:
            groups = dict(source.groups)
            groups[key] = groups.pop(("sample", "fine"))
            with self.subTest(key=key), self.assertRaisesRegex(
                NestingValidationError, "group key"
            ):
                nest_2d_distribution(
                    replace(source, groups=MappingProxyType(groups)),
                    self.worked_plan(),
                )

    def test_repeated_calls_and_reordered_group_mapping_are_deterministic(self):
        source = self.worked_result()
        first = nest_2d_distribution(source, self.worked_plan())
        second = nest_2d_distribution(source, self.worked_plan())
        reversed_groups = MappingProxyType(dict(reversed(tuple(source.groups.items()))))
        third = nest_2d_distribution(
            replace(source, groups=reversed_groups), self.worked_plan()
        )
        self.assertEqual(first, second)
        self.assertEqual(first, third)


if __name__ == "__main__":
    unittest.main()
