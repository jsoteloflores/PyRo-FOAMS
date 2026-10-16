from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from main.core.legacy_perimeter import FOAMS105_PERIMETER_PROFILE_CURRENT
from main.core.legacy_pipeline import (
    Foams105AnalysisSettings,
    Foams105PipelineGroupInput,
    Foams105PipelineStageError,
    Foams105PreparationSettings,
    Foams105SegmentedImageInput,
    Foams105SourceMetadata,
    analyze_foams105_prepared,
    prepare_foams105_analysis,
)
from main.core.legacy_selection import Foams105BinnedRange
from main.core.legacy_shape_statistics import Foams105ShapeValue

VOLUME_UNNORMALIZED = "unnormalized"
VOLUME_KNOWN_VESICULARITY = "known_vesicularity"


def _inputs():
    first = np.zeros((18, 18), dtype=np.uint8)
    first[3, 16] = 1
    first[7:9, 7:9] = 1
    first[2, 12] = 2
    first[0, 0] = 1
    first[10, 10:13] = 1
    first[12, 10:13] = 1
    first[10:13, 10] = 1
    first[10:13, 12] = 1
    second = np.zeros((18, 18), dtype=np.uint8)
    second[5:8, 5:8] = 1
    return first, second


def _settings(mode=VOLUME_UNNORMALIZED):
    return Foams105AnalysisSettings(
        selection_mode="manual",
        manual_ranges=(Foams105BinnedRange("coarse", 1.5, 2.6),),
        volume_normalization_mode=mode,
        known_vesicularity_percent=25.0 if mode == VOLUME_KNOWN_VESICULARITY else None,
        size_lower_cutoff_mm=0.5,
        size_upper_cutoff_mm=1.0,
        shape_minimum_diameter_px=2.0,
        shape_statistics_mode="updated_minimum",
    )


class TestFoams105Pipeline(unittest.TestCase):
    def setUp(self):
        self.first, self.second = _inputs()
        self.groups = (
            Foams105PipelineGroupInput(
                "coarse",
                1,
                (
                    Foams105SegmentedImageInput(
                        "image-a",
                        "first",
                        self.first,
                        1.0,
                        1,
                        (2, None, None, None),
                        (0.0, 0.0, 0.0, 0.0),
                    ),
                    Foams105SegmentedImageInput(
                        "image-b",
                        "second",
                        self.second,
                        1.0,
                        1,
                        (2, None, None, None),
                        (0.0, 0.0, 0.0, 0.0),
                    ),
                ),
            ),
        )
        self.prepared = prepare_foams105_analysis(
            self.groups,
            Foams105PreparationSettings(1.5, FOAMS105_PERIMETER_PROFILE_CURRENT),
            source_metadata=Foams105SourceMetadata(25.0, "test-revision"),
        )

    def test_preparation_snapshots_images_and_preserves_stage_populations(self):
        first_prepared = self.prepared.groups[0].images[0]
        original_hash = first_prepared.input_sha256
        self.first[:, :] = 9
        self.assertEqual(first_prepared.input_sha256, original_hash)
        self.assertFalse(first_prepared.image_input.segmented_image.flags.writeable)
        with self.assertRaises(ValueError):
            first_prepared.image_input.segmented_image.setflags(write=True)
        self.assertEqual(first_prepared.image_input.segmented_image[3, 16], 1)
        self.assertEqual(len(self.prepared.groups[0].images), 2)
        self.assertGreater(self.prepared.groups[0].histogram_result.input_count, 0)
        self.assertEqual(self.prepared.source_image_mapping[0].source_image_key, "first")
        self.assertEqual(self.prepared.source_metadata.density_derived_vesicularity_percent, 25.0)
        first_measurement = first_prepared.measurement_result.pore_result
        self.assertEqual(first_measurement.removed_border_pixel_count, 1)
        self.assertEqual(first_measurement.filled_added_pixel_count, 1)
        self.assertTrue(first_measurement.filled_mask[11, 11])

    def test_manual_analysis_keeps_distinct_population_routes_and_geometry(self):
        result = analyze_foams105_prepared(self.prepared, _settings())
        self.assertEqual(result.cutoff_suggestion_result, None)
        self.assertEqual(result.binned_selection_result.range_origin, "explicit_manual")
        self.assertEqual(result.raw_selection_result.selected_count, len(result.rawcutoffs_data_rows))
        self.assertEqual(len(result.raw_data_rows), len(self.prepared.groups[0].diameter_filtered_objects))
        self.assertGreaterEqual(len(result.raw_data_rows), result.raw_selection_result.selected_count)
        self.assertGreater(len(result.raw_data_rows), len(result.rawcutoffs_data_rows))
        self.assertEqual(
            sum(len(group.unfiltered_pore_objects) - len(group.diameter_filtered_objects) for group in self.prepared.groups),
            1,
        )
        selected = set(result.raw_selection_result.selected_identities)
        self.assertTrue(any(row.component_id == 1 and row.image_id == "image-a" for row in result.raw_data_rows))
        self.assertTrue(any(row.component_id == 1 and row.image_id == "image-b" for row in result.raw_data_rows))
        self.assertGreater(len(result.raw_data_rows), len(selected))
        self.assertEqual(result.shape_statistics_result.shape_result.perimeter_profile, FOAMS105_PERIMETER_PROFILE_CURRENT)
        self.assertEqual(result.shape_histogram_result.population_identities, result.shape_statistics_result.threshold_identities)
        self.assertEqual(result.vesicularity_result.group_vesicularity_percent[0], result.vesicularity_result.mean_group_vesicularity_percent)
        self.assertEqual(len(result.distribution_summary_result.rows), len(result.volume_result.rows))

    def test_autosmart_is_explicit_and_no_fallback_occurs(self):
        autosmart = replace(_settings(), selection_mode="autosmart", manual_ranges=())
        result = analyze_foams105_prepared(self.prepared, autosmart)
        self.assertIsNotNone(result.cutoff_suggestion_result)
        self.assertEqual(result.binned_selection_result.range_origin, "autosmart_suggestion")

        no_objects = np.zeros((18, 18), dtype=np.uint8)
        empty_group = replace(
            self.groups[0],
            images=(replace(self.groups[0].images[0], segmented_image=no_objects),),
        )
        empty_prepared = prepare_foams105_analysis(
            (empty_group,),
            Foams105PreparationSettings(0.5, FOAMS105_PERIMETER_PROFILE_CURRENT),
        )
        with self.assertRaises(Foams105PipelineStageError) as caught:
            analyze_foams105_prepared(empty_prepared, autosmart)
        self.assertEqual(caught.exception.stage, "autosmart_cutoffs")
        self.assertEqual(caught.exception.cause_code, "empty_group_support")

    def test_four_source_slots_preserve_input_order_and_slot_four_quirk(self):
        groups = []
        for slot, scale, source_key, pixel in (
            (1, 1.0, "mag1", (3, 3)),
            (2, 2.0, "first", (3, 3)),
            (3, 3.0, "second", (3, 3)),
            (4, 4.0, "third", (3, 3)),
        ):
            image = np.zeros((18, 18), dtype=np.uint8)
            image[pixel] = 1
            groups.append(
                Foams105PipelineGroupInput(
                    f"group-{slot}",
                    slot,
                    (
                        Foams105SegmentedImageInput(
                            f"image-{slot}", source_key, image, scale, 1,
                            (None, None, None, None), (0.0, 0.0, 0.0, 0.0),
                        ),
                    ),
                )
            )
        prepared = prepare_foams105_analysis(
            tuple(groups),
            Foams105PreparationSettings(0.5, FOAMS105_PERIMETER_PROFILE_CURRENT),
            source_metadata=Foams105SourceMetadata(30.0, "four-slot-fixture"),
        )
        labels = prepared.label_preparation_result.bin_labels_mm
        ranges = []
        for group in prepared.groups:
            positive = tuple(index for index, count in enumerate(group.histogram_result.counts) if count > 0)
            if not positive:
                raise RuntimeError(f"four-slot fixture has no histogram support for {group.group_id}")
            lower_index = 0 if group.source_slot == 4 else positive[0]
            ranges.append(Foams105BinnedRange(group.group_id, labels[lower_index], labels[positive[-1]]))
        settings = replace(
            _settings(),
            manual_ranges=tuple(ranges),
        )
        result = analyze_foams105_prepared(prepared, settings)
        self.assertEqual(tuple(group.source_slot for group in result.prepared_result.groups), (1, 2, 3, 4))
        self.assertEqual(tuple(mapping.source_slot for mapping in result.prepared_result.source_image_mapping), (1, 2, 3, 4))
        self.assertEqual(tuple(slot.source_slot for slot in result.bulk_na_plot_data_result.slots_descending), (4, 3, 2, 1))
        slot4 = result.bulk_na_plot_data_result.slots_descending[0]
        self.assertEqual(slot4.count_source_slot, 3)
        self.assertEqual(slot4.denominator_source_slot, 3)
        self.assertIn("source_slot4_uses_slot3_counts_and_area", slot4.diagnostics)

    def test_integrated_2d_normalization_path(self):
        integrated_settings = replace(
            _settings(),
            volume_normalization_mode="integrated_2d",
            known_vesicularity_percent=None,
        )
        result = analyze_foams105_prepared(self.prepared, integrated_settings)
        self.assertEqual(result.volume_result.mode, "integrated_2d")
        self.assertEqual(result.volume_result.vesicularity_result, result.vesicularity_result)

    def test_normalization_reanalysis_reuses_prepared_measurements_and_histograms(self):
        with patch("main.core.legacy_pipeline.measure_foams105_image") as measure, patch(
            "main.core.legacy_pipeline.count_foams105_histogram"
        ) as histogram, patch(
            "main.core.legacy_pipeline.measure_foams105_moment_geometry"
        ) as moment_geometry, patch(
            "main.core.legacy_pipeline.measure_foams105_perimeter"
        ) as perimeter, patch(
            "main.core.legacy_pipeline._phase_measurements_match",
            side_effect=AssertionError("cached reanalysis must not scan phase arrays"),
        ) as phase_measurement_match:
            result = analyze_foams105_prepared(self.prepared, _settings(VOLUME_KNOWN_VESICULARITY))
        measure.assert_not_called()
        histogram.assert_not_called()
        moment_geometry.assert_not_called()
        perimeter.assert_not_called()
        phase_measurement_match.assert_not_called()
        self.assertEqual(result.volume_result.mode, VOLUME_KNOWN_VESICULARITY)
        self.assertEqual(result.volume_result.target_vesicularity_percent, 25.0)

    def test_shape_minimum_reanalysis_uses_cached_summary_without_image_work(self):
        changed = replace(_settings(), shape_minimum_diameter_px=0.0)
        from main.core.legacy_shape_statistics import summarize_foams105_shapes

        with patch("main.core.legacy_pipeline.measure_foams105_image") as measure, patch(
            "main.core.legacy_pipeline.count_foams105_histogram"
        ) as histogram, patch(
            "main.core.legacy_pipeline.summarize_foams105_shapes",
            wraps=summarize_foams105_shapes,
        ) as shape_summary, patch(
            "main.core.legacy_pipeline.measure_foams105_moment_geometry"
        ) as moment_geometry, patch(
            "main.core.legacy_pipeline.measure_foams105_perimeter"
        ) as perimeter:
            result = analyze_foams105_prepared(self.prepared, changed)
        measure.assert_not_called()
        histogram.assert_not_called()
        moment_geometry.assert_not_called()
        perimeter.assert_not_called()
        self.assertEqual(shape_summary.call_count, 1)
        self.assertEqual(result.shape_statistics_result.threshold_count, len(result.shape_result.rows))

    def test_raw_table_lookup_is_indexed_and_preserves_original_indices(self):
        lower_label = self.prepared.label_preparation_result.bin_labels_mm[2]
        settings = replace(
            _settings(),
            manual_ranges=(Foams105BinnedRange("coarse", lower_label, 4.0),),
        )
        from main.core.legacy_pipeline import _raw_selected_object

        with patch(
            "main.core.legacy_pipeline._raw_selected_object",
            wraps=_raw_selected_object,
        ) as indexed_lookup:
            result = analyze_foams105_prepared(self.prepared, settings)
        self.assertEqual(indexed_lookup.call_count, result.raw_selection_result.selected_count)
        self.assertEqual(
            tuple((row.image_id, row.component_id) for row in result.rawcutoffs_data_rows),
            result.raw_selection_result.selected_identities,
        )
        self.assertEqual(
            tuple(row.image_id for row in result.raw_selection_result.rows),
            ("image-a", "image-b"),
        )
        self.assertEqual(
            tuple(row.source_object_index for row in result.raw_selection_result.rows),
            (1, 2),
        )
        self.assertEqual(
            tuple(row.source_object_index for row in result.rawcutoffs_data_rows),
            (1, 3),
        )

    def test_changed_area_count_normalization_and_settings_snapshots_are_rejected(self):
        group = self.prepared.groups[0]
        changed_group_area = replace(
            group,
            group_area_result=replace(
                group.group_area_result,
                phase_corrected_area_mm2=group.group_area_result.phase_corrected_area_mm2 + 1.0,
            ),
        )
        altered_counts = (group.histogram_result.counts[0] + 1,) + group.histogram_result.counts[1:]
        changed_histogram = replace(group, histogram_result=replace(group.histogram_result, counts=altered_counts))
        changed_normalization = replace(
            group,
            normalization_result=replace(
                group.normalization_result,
                na_per_mm2=(group.normalization_result.na_per_mm2[0] + 123.0,)
                + group.normalization_result.na_per_mm2[1:],
            ),
        )
        for name, altered_group in (
            ("area", changed_group_area),
            ("counts", changed_histogram),
            ("normalization", changed_normalization),
        ):
            with self.subTest(snapshot=name), self.assertRaises(Foams105PipelineStageError):
                analyze_foams105_prepared(
                    replace(self.prepared, groups=(altered_group,)),
                    _settings(),
                )
        stale_settings = replace(
            self.prepared,
            preparation_settings=replace(
                self.prepared.preparation_settings,
                minimum_diameter_px=self.prepared.preparation_settings.minimum_diameter_px + 0.5,
            ),
        )
        with self.assertRaises(Foams105PipelineStageError):
            analyze_foams105_prepared(stale_settings, _settings())

    def test_changed_cached_shape_value_is_rejected(self):
        group = self.prepared.groups[0]
        image = group.images[0]
        cached = image.component_shape_measures[0]
        altered_cached = replace(cached, regularity=Foams105ShapeValue(999.0, "defined"))
        altered_image = replace(
            image,
            component_shape_measures=(altered_cached,) + image.component_shape_measures[1:],
        )
        altered_group = replace(group, images=(altered_image,) + group.images[1:])
        altered_prepared = replace(self.prepared, groups=(altered_group,))
        with self.assertRaises(Foams105PipelineStageError) as caught:
            analyze_foams105_prepared(altered_prepared, _settings())
        self.assertEqual(caught.exception.stage, "prepared_validation")

    def test_prepared_result_without_builder_record_is_rejected(self):
        unvalidated = replace(self.prepared, _validation_record=None)
        with self.assertRaises(Foams105PipelineStageError) as caught:
            analyze_foams105_prepared(unvalidated, _settings())
        self.assertEqual(caught.exception.cause_code, "missing_prepared_validation_record")

    def test_changed_cached_identity_is_rejected(self):
        altered_group = replace(self.prepared.groups[0], group_id="renamed")
        altered_prepared = replace(self.prepared, groups=(altered_group,))
        with self.assertRaises(Foams105PipelineStageError) as caught:
            analyze_foams105_prepared(altered_prepared, _settings())
        self.assertEqual(caught.exception.stage, "prepared_validation")
        self.assertEqual(caught.exception.cause_code, "stale_group_area")

        altered_mapping = replace(self.prepared, source_image_mapping=())
        with self.assertRaises(Foams105PipelineStageError) as caught:
            analyze_foams105_prepared(altered_mapping, _settings())
        self.assertEqual(caught.exception.stage, "prepared_validation")
        self.assertEqual(caught.exception.cause_code, "stale_source_mapping")


if __name__ == "__main__":
    unittest.main()
