import math
import os
import sys
import unittest
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.sampling import (
    AnalysisDomainError,
    MeasurementMappingError,
    SamplingValidationError,
    UnsupportedUnitError,
    build_sampling_dataset,
    create_image_sampling_record,
)
from core.stereology import measure_labels


def make_record(label_map, **overrides):
    values = {
        "sample_id": "sample-a",
        "image_id": "image-a",
        "image_index": 0,
        "magnification_group_id": "high",
        "label_map": label_map,
        "calibration": 0.01,
        "calibration_unit": "mm",
    }
    values.update(overrides)
    return create_image_sampling_record(**values)


class TestImageSamplingRecord(unittest.TestCase):
    def test_full_and_masked_physical_area(self):
        labels = np.zeros((100, 200), dtype=np.int32)
        full = make_record(labels)
        self.assertEqual(full.analyzed_pixel_count, 20_000)
        self.assertAlmostEqual(full.analyzed_area_mm2, 2.0)

        domain = np.zeros_like(labels, dtype=bool)
        domain[:50, :] = True
        masked = make_record(labels, analysis_domain_mask=domain)
        self.assertEqual(masked.analyzed_pixel_count, 10_000)
        self.assertAlmostEqual(masked.analyzed_area_mm2, 1.0)

    def test_micrometers_and_millimeters_are_equivalent(self):
        labels = np.zeros((10, 20), dtype=np.int32)
        mm = make_record(labels)
        um = make_record(
            labels,
            image_id="image-um",
            image_index=1,
            calibration=10.0,
            calibration_unit="µm",
        )
        self.assertEqual(mm.calibration_mm_per_px, um.calibration_mm_per_px)
        self.assertEqual(mm.analyzed_area_mm2, um.analyzed_area_mm2)

    def test_detection_limits_are_optional_and_validated(self):
        labels = np.zeros((2, 2), dtype=np.int32)
        record = make_record(labels)
        self.assertFalse(record.detection_window_is_set)
        self.assertIsNone(record.min_detectable_diameter_mm)
        with self.assertRaises(SamplingValidationError):
            make_record(labels, min_detectable_diameter=1.0, max_reliable_diameter=1.0)
        with self.assertRaises(SamplingValidationError):
            make_record(labels, max_reliable_diameter=1.0)
        with self.assertRaises(SamplingValidationError):
            make_record(labels, min_detectable_diameter=math.inf)
        with self.assertRaises(SamplingValidationError):
            make_record(labels, min_detectable_diameter=-1.0)

    def test_invalid_calibration_unit_and_mask_are_rejected(self):
        labels = np.zeros((2, 2), dtype=np.int32)
        with self.assertRaises(SamplingValidationError):
            make_record(labels, calibration=0)
        with self.assertRaises(UnsupportedUnitError):
            make_record(labels, calibration_unit="px")
        with self.assertRaises(AnalysisDomainError):
            make_record(labels, analysis_domain_mask=np.ones((3, 2), dtype=bool))
        with self.assertRaises(AnalysisDomainError):
            make_record(labels, analysis_domain_mask=np.ones((2, 2), dtype=np.uint8))

    def test_domain_omits_external_label_and_rejects_crossing_label(self):
        labels = np.zeros((4, 6), dtype=np.int32)
        labels[1:3, 1:3] = 1
        labels[1:3, 4:6] = 2
        domain = np.zeros_like(labels, dtype=bool)
        domain[:, :4] = True
        record = make_record(labels, analysis_domain_mask=domain)
        self.assertEqual(record.included_labels, (1,))
        self.assertEqual(record.omitted_labels, (2,))

        crossing_domain = domain.copy()
        crossing_domain[:, 1] = False
        with self.assertRaisesRegex(AnalysisDomainError, "cuts through label 1"):
            make_record(labels, analysis_domain_mask=crossing_domain)

    def test_sparse_label_ids_are_preserved(self):
        labels = np.zeros((6, 6), dtype=np.int32)
        labels[1, 1] = 7
        labels[4, 4] = 1_000_003
        record = make_record(labels)
        self.assertEqual(record.included_labels, (7, 1_000_003))


class TestSamplingDataset(unittest.TestCase):
    def test_converts_diameter_and_retains_source_measurement(self):
        labels = np.zeros((30, 30), dtype=np.int32)
        labels[5:15, 5:15] = 1
        pore = measure_labels(labels, image_index=0)[0]
        pore = replace(pore, eq_diam_px=20.0)
        image = make_record(labels)
        dataset = build_sampling_dataset([image], [pore])
        self.assertAlmostEqual(dataset.pores[0].equivalent_diameter_mm, 0.2)
        self.assertIs(dataset.pores[0].measurement, pore)
        self.assertFalse(dataset.pores[0].measurement.touches_border)

    def test_group_summary_retains_unequal_and_empty_images(self):
        labels_1 = np.zeros((100, 100), dtype=np.int32)
        labels_1[10:20, 10:20] = 1
        labels_3 = np.zeros((100, 300), dtype=np.int32)
        image_1 = make_record(labels_1, image_id="one", image_index=0)
        image_3 = make_record(labels_3, image_id="three", image_index=1)
        pore = measure_labels(labels_1, image_index=0)[0]

        dataset = build_sampling_dataset([image_1, image_3], [pore])
        summary = dataset.summarize_groups()[("sample-a", "high")]
        self.assertEqual([image.analyzed_area_mm2 for image in dataset.images], [1.0, 3.0])
        self.assertEqual(summary.image_count, 2)
        self.assertEqual(summary.observed_pore_count, 1)
        self.assertAlmostEqual(summary.total_analyzed_area_mm2, 4.0)

    def test_same_group_name_in_different_samples_is_isolated(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        first = make_record(labels, sample_id="a", image_id="a", image_index=0)
        second = make_record(labels, sample_id="b", image_id="b", image_index=1)
        summaries = build_sampling_dataset([first, second], []).summarize_groups()
        self.assertEqual(set(summaries), {("a", "high"), ("b", "high")})

    def test_external_label_measurement_is_recorded_as_omitted(self):
        labels = np.zeros((4, 6), dtype=np.int32)
        labels[1:3, 1:3] = 1
        labels[1:3, 4:6] = 2
        domain = np.zeros_like(labels, dtype=bool)
        domain[:, :4] = True
        image = make_record(labels, analysis_domain_mask=domain)
        pores = measure_labels(labels, image_index=0)
        dataset = build_sampling_dataset([image], pores)
        self.assertEqual([p.measurement.label for p in dataset.pores], [1])
        self.assertEqual(dataset.omitted_pore_identities, ((0, 2),))

    def test_invalid_measurement_mapping_is_rejected(self):
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[1:3, 1:3] = 1
        image = make_record(labels)
        pore = measure_labels(labels, image_index=0)[0]
        with self.assertRaisesRegex(MeasurementMappingError, "unknown image_index"):
            build_sampling_dataset([image], [replace(pore, image_index=8)])
        with self.assertRaisesRegex(MeasurementMappingError, "Duplicate pore"):
            build_sampling_dataset([image], [pore, pore])
        with self.assertRaisesRegex(MeasurementMappingError, "Ambiguous image_index"):
            build_sampling_dataset(
                [image, make_record(labels, image_id="other", image_index=0)], []
            )
        with self.assertRaisesRegex(MeasurementMappingError, "Duplicate image identity"):
            build_sampling_dataset(
                [image, make_record(labels, image_index=1)], []
            )
        with self.assertRaisesRegex(MeasurementMappingError, "positive and finite"):
            build_sampling_dataset([image], [replace(pore, eq_diam_px=math.nan)])

    def test_calibrated_measurement_must_match_image(self):
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[1:3, 1:3] = 1
        image = make_record(labels)
        pore = measure_labels(
            labels,
            image_index=0,
            scale={"unitsPerPx": 10.0, "unitName": "um"},
        )[0]
        dataset = build_sampling_dataset([image], [pore])
        self.assertAlmostEqual(
            dataset.pores[0].equivalent_diameter_mm,
            pore.eq_diam_units * 1e-3,
        )
        with self.assertRaisesRegex(MeasurementMappingError, "calibrated diameter"):
            build_sampling_dataset([image], [replace(pore, eq_diam_units=99.0)])
        with self.assertRaisesRegex(MeasurementMappingError, "image calibration"):
            build_sampling_dataset([image], [replace(pore, units_per_px=11.0)])

    def test_detection_readiness_is_explicit(self):
        labels = np.zeros((2, 2), dtype=np.int32)
        pending = make_record(labels)
        dataset = build_sampling_dataset([pending], [])
        self.assertFalse(dataset.detection_limits_are_set)
        with self.assertRaisesRegex(SamplingValidationError, "image-a"):
            dataset.require_detection_limits()

        ready = make_record(labels, min_detectable_diameter=0.01)
        ready_dataset = build_sampling_dataset([ready], [])
        self.assertTrue(ready_dataset.detection_limits_are_set)
        ready_dataset.require_detection_limits()


if __name__ == "__main__":
    unittest.main()
