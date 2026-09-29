import math
import os
import sys
import unittest
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.distributions import (
    DistributionValidationError,
    calculate_2d_number_densities,
    create_diameter_bin_spec,
    geometric_diameter_bin_spec,
)
from core.sampling import (
    SamplingDataset,
    SamplingValidationError,
    build_sampling_dataset,
    create_image_sampling_record,
)
from core.stereology import measure_labels


def make_image(
    image_index,
    image_id,
    area_mm2,
    diameters,
    *,
    sample_id="sample",
    group_id="group",
    minimum=0.1,
    maximum=None,
    border_labels=(),
):
    width = int(round(area_mm2 * 100))
    labels = np.zeros((100, width), dtype=np.int32)
    for label, _ in enumerate(diameters, start=1):
        labels[label, label] = label
    image = create_image_sampling_record(
        sample_id=sample_id,
        image_id=image_id,
        image_index=image_index,
        magnification_group_id=group_id,
        label_map=labels,
        calibration=0.01,
        calibration_unit="mm",
        min_detectable_diameter=minimum,
        max_reliable_diameter=maximum,
    )
    pores = measure_labels(labels, image_index=image_index)
    measurements = [
        replace(
            pore,
            eq_diam_px=diameters[pore.label - 1] / 0.01,
            touches_border=pore.label in border_labels,
        )
        for pore in pores
    ]
    return image, measurements


def make_dataset(*fixtures):
    images = [fixture[0] for fixture in fixtures]
    measurements = [pore for fixture in fixtures for pore in fixture[1]]
    return build_sampling_dataset(images, measurements)


class TestDiameterBinSpec(unittest.TestCase):
    def test_geometric_grid(self):
        spec = geometric_diameter_bin_spec(0.01, 3)
        self.assertEqual(len(spec.edges_mm), 4)
        for lower, upper in zip(spec.edges_mm, spec.edges_mm[1:]):
            self.assertAlmostEqual(upper / lower, 10**0.1)
        self.assertAlmostEqual(
            spec.bins[0].geometric_midpoint_mm,
            math.sqrt(spec.edges_mm[0] * spec.edges_mm[1]),
        )

    def test_explicit_edges_are_copied_and_describe_bins(self):
        edges = [0.1, 0.2, 0.4]
        spec = create_diameter_bin_spec(edges)
        edges[1] = 99.0
        self.assertEqual(spec.edges_mm, (0.1, 0.2, 0.4))
        self.assertEqual(spec.bins[1].width_mm, 0.2)

    def test_invalid_explicit_and_geometric_grids(self):
        for edges in ([0.1], [0, 0.1], [0.2, 0.1], [0.1, math.nan], [0.1, math.inf], [True, 2.0]):
            with self.subTest(edges=edges), self.assertRaises(DistributionValidationError):
                create_diameter_bin_spec(edges)
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(DistributionValidationError):
                geometric_diameter_bin_spec(0.1, count)
        for start, step in ((0, 0.1), (math.nan, 0.1), (0.1, 0), (0.1, math.inf)):
            with self.subTest(start=start, step=step), self.assertRaises(DistributionValidationError):
                geometric_diameter_bin_spec(start, 2, step)
        with self.assertRaises(DistributionValidationError):
            geometric_diameter_bin_spec(True, 2)
        with self.assertRaises(DistributionValidationError):
            geometric_diameter_bin_spec(1e308, 2, 1.0)


class TestNumberDensities(unittest.TestCase):
    def setUp(self):
        self.bins = create_diameter_bin_spec([0.1, 0.2, 0.4])

    def test_edge_ownership_and_representable_neighbors(self):
        below_edge = np.nextafter(0.2, 0.0)
        above_edge = np.nextafter(0.2, math.inf)
        fixture = make_image(
            0,
            "edges",
            1.0,
            [0.1, below_edge, 0.2, above_edge, 0.4, 0.09, 0.41],
            maximum=0.4,
        )
        result = calculate_2d_number_densities(make_dataset(fixture), self.bins)
        group = result.groups[("sample", "group")]
        self.assertEqual(group.counts, (2, 3))
        self.assertEqual(len(result.diagnostics.below_grid), 1)
        self.assertEqual(len(result.diagnostics.above_grid), 1)

    def test_area_weighting_and_empty_image_denominator(self):
        one = make_image(0, "one", 1.0, [0.15, 0.15])
        three = make_image(1, "three", 3.0, [0.15, 0.15, 0.15])
        result = calculate_2d_number_densities(make_dataset(one, three), self.bins)
        group = result.groups[("sample", "group")]
        self.assertEqual(group.counts[0], 5)
        self.assertAlmostEqual(group.eligible_areas_mm2[0], 4.0)
        self.assertAlmostEqual(group.number_densities_per_mm2[0], 1.25)

        empty = make_image(2, "empty", 2.0, [])
        result = calculate_2d_number_densities(make_dataset(one, three, empty), self.bins)
        group = result.groups[("sample", "group")]
        self.assertAlmostEqual(group.eligible_areas_mm2[0], 6.0)
        self.assertAlmostEqual(group.number_densities_per_mm2[0], 5 / 6)

    def test_heterogeneous_detection_windows(self):
        first = make_image(0, "a", 1.0, [0.15, 0.15, 0.3], maximum=0.4)
        second = make_image(1, "b", 3.0, [0.3, 0.3, 0.3], minimum=0.2, maximum=0.4)
        group = calculate_2d_number_densities(make_dataset(first, second), self.bins).groups[
            ("sample", "group")
        ]
        self.assertEqual(group.counts, (2, 4))
        self.assertEqual(group.eligible_areas_mm2, (1.0, 4.0))
        self.assertEqual(group.number_densities_per_mm2, (2.0, 1.0))
        self.assertEqual(group.eligible_image_counts, (1, 2))

    def test_partial_window_is_unsupported_for_whole_bin(self):
        lower_cut = make_image(0, "lower", 1.0, [0.18], minimum=0.15, maximum=0.3)
        upper_cut = make_image(1, "upper", 1.0, [0.3], minimum=0.15, maximum=0.3)
        result = calculate_2d_number_densities(make_dataset(lower_cut, upper_cut), self.bins)
        group = result.groups[("sample", "group")]
        self.assertEqual(group.counts, (0, 0))
        self.assertEqual(group.eligible_areas_mm2, (0.0, 0.0))
        self.assertEqual(len(result.diagnostics.unsupported_by_image), 2)

    def test_supported_empty_is_distinct_from_unsupported(self):
        image = make_image(0, "limited", 1.0, [], minimum=0.2, maximum=0.4)
        group = calculate_2d_number_densities(make_dataset(image), self.bins).groups[
            ("sample", "group")
        ]
        self.assertEqual(group.counts, (0, 0))
        self.assertEqual(group.supported, (False, True))
        self.assertTrue(math.isnan(group.number_densities_per_mm2[0]))
        self.assertEqual(group.number_densities_per_mm2[1], 0.0)

    def test_window_boundaries_and_missing_limit(self):
        exact = make_image(0, "exact", 1.0, [], minimum=0.1, maximum=0.2)
        open_upper = make_image(1, "open", 1.0, [], minimum=0.1)
        group = calculate_2d_number_densities(make_dataset(exact, open_upper), self.bins).groups[
            ("sample", "group")
        ]
        self.assertEqual(group.eligible_image_counts, (2, 1))

        image, _ = make_image(2, "missing", 1.0, [])
        missing = replace(
            image,
            min_detectable_diameter=None,
            min_detectable_diameter_mm=None,
        )
        with self.assertRaisesRegex(SamplingValidationError, "missing"):
            calculate_2d_number_densities(SamplingDataset((missing,), ()), self.bins)

    def test_group_isolation_and_empty_group_retention(self):
        first = make_image(0, "a", 1.0, [0.15], sample_id="one", group_id="same")
        second = make_image(1, "b", 1.0, [], sample_id="two", group_id="same")
        result = calculate_2d_number_densities(make_dataset(first, second), self.bins)
        self.assertEqual(set(result.groups), {("one", "same"), ("two", "same")})
        self.assertEqual(result.groups[("two", "same")].counts, (0, 0))

    def test_equivalent_mm_and_um_inputs_match(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[1, 1] = 1
        mm_image = create_image_sampling_record(
            sample_id="sample", image_id="mm", image_index=0,
            magnification_group_id="mm", label_map=labels,
            calibration=0.01, calibration_unit="mm", min_detectable_diameter=0.1,
        )
        um_image = create_image_sampling_record(
            sample_id="sample", image_id="um", image_index=1,
            magnification_group_id="um", label_map=labels,
            calibration=10.0, calibration_unit="um", min_detectable_diameter=100.0,
        )
        mm_pore = replace(measure_labels(labels, image_index=0)[0], eq_diam_px=15.0)
        um_pore = replace(measure_labels(labels, image_index=1)[0], eq_diam_px=15.0)
        result = calculate_2d_number_densities(
            build_sampling_dataset([mm_image, um_image], [mm_pore, um_pore]), self.bins
        )
        self.assertEqual(result.groups[("sample", "mm")].counts, result.groups[("sample", "um")].counts)
        self.assertEqual(
            result.groups[("sample", "mm")].number_densities_per_mm2,
            result.groups[("sample", "um")].number_densities_per_mm2,
        )

    def test_border_policy_and_diagnostic_precedence_reconcile(self):
        fixture = make_image(
            0, "diagnostics", 1.0, [0.15, 0.05, 0.5, 0.18],
            minimum=0.2, border_labels=(1,)
        )
        result = calculate_2d_number_densities(
            make_dataset(fixture), self.bins, exclude_border=True
        )
        self.assertEqual(result.groups[("sample", "group")].counts, (0, 0))
        self.assertEqual(len(result.diagnostics.excluded_border), 1)
        self.assertEqual(len(result.diagnostics.below_grid), 1)
        self.assertEqual(len(result.diagnostics.above_grid), 1)
        self.assertEqual(len(result.diagnostics.unsupported_by_image), 1)
        self.assertEqual(result.diagnostics.uncounted_input_count, 4)
        self.assertEqual(result.groups[("sample", "group")].eligible_areas_mm2, (0.0, 1.0))

    def test_empty_dataset_and_input_integrity(self):
        dataset = SamplingDataset((), ())
        result = calculate_2d_number_densities(dataset, self.bins)
        self.assertEqual(dict(result.groups), {})
        self.assertEqual(result.bin_spec, self.bins)
        self.assertEqual(dataset, SamplingDataset((), ()))

        fixture = make_image(0, "bad", 1.0, [0.15])
        valid = make_dataset(fixture)
        bad_pore = replace(valid.pores[0], equivalent_diameter_mm=math.nan)
        with self.assertRaisesRegex(DistributionValidationError, "diameter"):
            calculate_2d_number_densities(SamplingDataset(valid.images, (bad_pore,)), self.bins)
        string_pore = replace(valid.pores[0], equivalent_diameter_mm="0.15")
        with self.assertRaisesRegex(DistributionValidationError, "diameter"):
            calculate_2d_number_densities(SamplingDataset(valid.images, (string_pore,)), self.bins)

        bad_image = object.__new__(type(valid.images[0]))
        for name, value in valid.images[0].__dict__.items():
            object.__setattr__(bad_image, name, value)
        object.__setattr__(bad_image, "analyzed_area_mm2", math.inf)
        with self.assertRaisesRegex(DistributionValidationError, "area"):
            calculate_2d_number_densities(SamplingDataset((bad_image,), ()), self.bins)


if __name__ == "__main__":
    unittest.main()
