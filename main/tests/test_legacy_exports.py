from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from main.core.legacy_exports import (
    AREA_HEADERS,
    CALC_HEADERS,
    MISC_HEADERS,
    NA_MAG_HEADERS,
    RAW_HEADERS,
    SHAPE_HEADERS,
    STATS_HEADERS,
    VES_HEADERS,
    Foams105ExportError,
    _area_out_projection,
    _format_cell,
    _project_foams105_export_tables,
    build_foams105_export_tables,
    write_foams105_export_bundle,
)
from main.core.legacy_perimeter import FOAMS105_PERIMETER_PROFILE_CURRENT
from main.core.legacy_pipeline import (
    Foams105AnalysisSettings,
    Foams105PipelineGroupInput,
    Foams105PreparationSettings,
    Foams105SegmentedImageInput,
    Foams105SourceMetadata,
    analyze_foams105_prepared,
    prepare_foams105_analysis,
)
from main.core.legacy_selection import Foams105BinnedRange
from main.core.legacy_summaries import Foams105RealValue


def _build_result(density_setting=True):
    first = np.zeros((18, 18), dtype=np.uint8)
    first[3, 3] = 1
    first[7:9, 7:9] = 1
    first[2, 12] = 2
    second = np.zeros((18, 18), dtype=np.uint8)
    second[5:7, 5:7] = 1
    groups = (
        Foams105PipelineGroupInput(
            "coarse",
            1,
            (
                Foams105SegmentedImageInput("image-b", "second", second, 1.0, 1, (2, None, None, None), (0.0, 0.0, 0.0, 0.0)),
                Foams105SegmentedImageInput("image-a", "first", first, 1.0, 1, (2, None, None, None), (0.0, 0.0, 0.0, 0.0)),
            ),
        ),
    )
    metadata = Foams105SourceMetadata(25.0, "test-revision") if density_setting else Foams105SourceMetadata()
    prepared = prepare_foams105_analysis(
        groups,
        Foams105PreparationSettings(0.5, FOAMS105_PERIMETER_PROFILE_CURRENT),
        source_metadata=metadata,
    )
    settings = Foams105AnalysisSettings(
        selection_mode="manual",
        manual_ranges=(Foams105BinnedRange("coarse", 0.5, 100.0),),
        volume_normalization_mode="unnormalized",
        known_vesicularity_percent=None,
        size_lower_cutoff_mm=0.5,
        size_upper_cutoff_mm=1.0,
        shape_minimum_diameter_px=1.5,
        shape_statistics_mode="updated_minimum",
    )
    return analyze_foams105_prepared(prepared, settings)


def _tables_by_name(result):
    return {table.filename: table for table in build_foams105_export_tables(result).tables}


class TestFoams105ExportProjections(unittest.TestCase):
    def test_all_nine_schemas_and_source_population_routes(self):
        result = _build_result()
        projections = build_foams105_export_tables(result)
        tables = {table.filename: table for table in projections.tables}
        self.assertEqual(
            tuple(tables),
            (
                "NA_mag.txt",
                "Raw_data.txt",
                "Rawcutoffs_data.txt",
                "Calc_out.txt",
                "Area_out.txt",
                "Ves_out.txt",
                "Misc_out.txt",
                "Shape_out.txt",
                "Stats_out.txt",
            ),
        )
        self.assertTrue(projections.complete)
        self.assertEqual(tables["NA_mag.txt"].headers, NA_MAG_HEADERS)
        self.assertEqual(tables["Raw_data.txt"].headers, RAW_HEADERS)
        self.assertEqual(tables["Rawcutoffs_data.txt"].headers, RAW_HEADERS)
        self.assertEqual(tables["Calc_out.txt"].headers, CALC_HEADERS)
        self.assertEqual(tables["Area_out.txt"].headers, AREA_HEADERS)
        self.assertEqual(tables["Ves_out.txt"].headers, VES_HEADERS)
        self.assertEqual(tables["Misc_out.txt"].headers, MISC_HEADERS)
        self.assertEqual(tables["Shape_out.txt"].headers, SHAPE_HEADERS)
        self.assertEqual(tables["Stats_out.txt"].headers, STATS_HEADERS)
        self.assertEqual(len(tables["NA_mag.txt"].rows), 45)
        self.assertTrue(all(row[3:] == (0, 0.0, 0, 0.0, 0, 0.0) for row in tables["NA_mag.txt"].rows))
        self.assertEqual(len(tables["Raw_data.txt"].rows), len(result.raw_data_rows))
        self.assertEqual(len(tables["Rawcutoffs_data.txt"].rows), result.raw_selection_result.selected_count)
        self.assertEqual(len(tables["Calc_out.txt"].rows), len(result.conversion_result.bin_labels_mm))
        self.assertEqual(len(tables["Area_out.txt"].rows), 2)
        shape_rows = tables["Shape_out.txt"].rows
        self.assertEqual(len(shape_rows), result.raw_selection_result.selected_count)
        self.assertEqual(
            tuple((row.image_id, row.component_id) for row in result.shape_result.rows),
            tuple((row.image_id, row.component_id) for row in result.rawcutoffs_data_rows),
        )
        self.assertEqual(
            shape_rows,
            (
                (0.0, 0.0, 1.0, 0.9549296585513721),
                (0.0, 0.0, 1.0, 0.9549296585513721),
                (0.0, 0.0, 1.0, 0.9549296585513721),
            ),
        )
        self.assertEqual(tuple(row[0] for row in tables["Stats_out.txt"].rows), ("asp_rat", "elong", "regularity", "eccent"))
        self.assertEqual(result.shape_statistics_result.statistic_rows[2].display_mean, 0.9549)
        self.assertEqual(tables["Area_out.txt"].rows[0][0], 1)
        self.assertEqual(
            tuple(item.source_image_key for item in result.prepared_result.source_image_mapping),
            ("second", "first"),
        )
        image_a_area = next(
            image.image_area_result.phase_corrected_area_mm2
            for image in result.prepared_result.groups[0].images
            if image.image_input.source_image_key == "first"
        )
        self.assertEqual(tables["Area_out.txt"].rows[0][1], image_a_area)
        self.assertEqual(tables["Raw_data.txt"].rows[0][-1], 1)

    def test_shape_minimum_changes_plot_population_not_shape_table(self):
        result = _build_result()
        lower_minimum = analyze_foams105_prepared(
            result.prepared_result,
            replace(result.analysis_settings, shape_minimum_diameter_px=0.0),
        )
        self.assertNotEqual(
            result.shape_histogram_result.population_identities,
            lower_minimum.shape_histogram_result.population_identities,
        )
        self.assertEqual(result.shape_statistics_result.statistic_rows[0].n, 2)
        self.assertEqual(lower_minimum.shape_statistics_result.statistic_rows[0].n, 3)
        self.assertEqual(
            _tables_by_name(result)["Shape_out.txt"].rows,
            _tables_by_name(lower_minimum)["Shape_out.txt"].rows,
        )

    def test_missing_density_vesicularity_is_explicit_partial_cell(self):
        result = _build_result(density_setting=False)
        projections = build_foams105_export_tables(result)
        projection = {table.filename: table for table in projections.tables}["Ves_out.txt"]
        self.assertEqual(projection.status, "partial")
        self.assertIsNone(projection.rows[0][7])
        self.assertFalse(projections.complete)
        with tempfile.TemporaryDirectory() as root:
            bundle = write_foams105_export_bundle(result, Path(root) / "partial")
            self.assertFalse(bundle.complete)

    def test_area_out_unavailable_on_zero_area_index_mismatch(self):
        result = _build_result()
        group = result.prepared_result.groups[0]
        first_image = group.images[0]
        zero_area = replace(first_image.image_area_result, phase_corrected_area_mm2=0.0)
        modified_image = replace(first_image, image_area_result=zero_area)
        modified_group = replace(group, images=(modified_image,) + group.images[1:])
        altered_prepared = replace(result.prepared_result, groups=(modified_group,))
        altered_result = replace(result, prepared_result=altered_prepared)
        area_table = _area_out_projection(altered_result, [])
        self.assertEqual(area_table.status, "unavailable")
        self.assertEqual(area_table.unavailable_reason, "area_count_does_not_match_source_image_indices")

        negative_area = replace(first_image.image_area_result, phase_corrected_area_mm2=-2.0)
        negative_image = replace(first_image, image_area_result=negative_area)
        negative_group = replace(group, images=(negative_image,) + group.images[1:])
        negative_result = replace(result, prepared_result=replace(result.prepared_result, groups=(negative_group,)))
        negative_table = _area_out_projection(negative_result, [])
        self.assertEqual(negative_table.status, "available")
        self.assertEqual(negative_table.rows[1][1], -2.0)

    def test_export_projection_does_not_recompute_science(self):
        result = _build_result()
        with patch("main.core.legacy_pipeline.measure_foams105_image") as measure, patch(
            "main.core.legacy_pipeline.count_foams105_histogram"
        ) as histogram:
            build_foams105_export_tables(result)
        measure.assert_not_called()
        histogram.assert_not_called()

    def test_mixed_binned_density_is_rejected_before_bundle_directory_creation(self):
        result = _build_result()
        original = result.binned_selection_result.na_per_mm2
        mixed_binned = replace(
            result.binned_selection_result,
            na_per_mm2=(123456.0,) + original[1:],
        )
        altered_result = replace(result, binned_selection_result=mixed_binned)
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "must-not-exist"
            with self.assertRaises(Foams105ExportError) as caught:
                write_foams105_export_bundle(altered_result, destination)
            self.assertEqual(caught.exception.code, "invalid_analysis_result_cache")
            self.assertFalse(destination.exists())

    def test_replaced_final_settings_are_rejected_by_public_export(self):
        result = _build_result()
        altered_result = replace(
            result,
            analysis_settings=replace(result.analysis_settings, shape_minimum_diameter_px=0.0),
        )
        with self.assertRaises(Foams105ExportError) as caught:
            build_foams105_export_tables(altered_result)
        self.assertEqual(caught.exception.code, "invalid_analysis_result_cache")

    def test_final_result_without_builder_record_is_rejected(self):
        result = replace(_build_result(), _validation_record=None)
        with self.assertRaises(Foams105ExportError) as caught:
            build_foams105_export_tables(result)
        self.assertEqual(caught.exception.code, "invalid_analysis_result_cache")


class TestFoams105ExportBundle(unittest.TestCase):
    def test_deterministic_tsv_bundle_checksums_and_collision_refusal(self):
        result = _build_result()
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            projection = build_foams105_export_tables(result)
            first = write_foams105_export_bundle(result, root_path / "one")
            second = write_foams105_export_bundle(result, root_path / "two", projection_result=projection)
            self.assertTrue(first.complete)
            self.assertEqual(first.table_filenames, tuple(
                ("NA_mag.txt", "Raw_data.txt", "Rawcutoffs_data.txt", "Calc_out.txt", "Area_out.txt", "Ves_out.txt", "Misc_out.txt", "Shape_out.txt", "Stats_out.txt")
            ))
            self.assertEqual(first.checksums_sha256, second.checksums_sha256)
            self.assertIn("manifest.json", dict(first.checksums_sha256))
            for filename in first.table_filenames:
                data = (root_path / "one" / filename).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), dict(first.checksums_sha256)[filename])
                self.assertTrue(data.endswith(b"\n"))
            with (root_path / "one" / "Stats_out.txt").open("r", encoding="utf-8", newline="") as handle:
                parsed = tuple(csv.reader(handle, delimiter="\t"))
            self.assertEqual(tuple(parsed[0]), STATS_HEADERS)
            self.assertEqual(parsed[3][4], "0.9549")
            with (root_path / "one" / "Raw_data.txt").open("r", encoding="utf-8", newline="") as handle:
                raw_rows = tuple(csv.reader(handle, delimiter="\t"))
            self.assertEqual(float(raw_rows[1][0]), result.raw_data_rows[0].equivalent_diameter_mm)
            manifest = json.loads((root_path / "one" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source"]["actual_pyro_revision"], "test-revision")
            self.assertIn("manifest.json", manifest["output_filenames"])
            identities = json.loads((root_path / "one" / "input_identity.json").read_text(encoding="utf-8"))
            self.assertEqual(tuple(item["source_image_key"] for item in identities["images"]), ("second", "first"))
            self.assertTrue(all("area1_phase_corrected_mm2" in item and "area2_border_corrected_mm2" in item for item in identities["images"]))
            shape_identity = json.loads((root_path / "one" / "Shape_out.identity.json").read_text(encoding="utf-8"))
            self.assertEqual(
                tuple((item["image_id"], item["component_id"]) for item in shape_identity["rows"]),
                tuple((row.image_id, row.component_id) for row in result.rawcutoffs_data_rows),
            )
            self.assertEqual(
                tuple(tuple(item) for item in shape_identity["plot_population_identities"]),
                result.shape_histogram_result.population_identities,
            )
            with self.assertRaises(Foams105ExportError) as caught:
                write_foams105_export_bundle(result, root_path / "one")
            self.assertEqual(caught.exception.code, "output_directory_exists")

    def test_prebuilt_projection_must_belong_to_exact_analysis_result(self):
        result = _build_result()
        projection = build_foams105_export_tables(result)
        other_result = _build_result()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(Foams105ExportError) as caught:
                write_foams105_export_bundle(
                    other_result,
                    Path(directory) / "bundle",
                    projection_result=projection,
                )
        self.assertEqual(caught.exception.code, "projection_analysis_mismatch")

    def test_write_failure_cleans_only_temporary_sibling(self):
        result = _build_result()
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "bundle"
            with patch("main.core.legacy_exports._write_bytes", side_effect=OSError("simulated disk error")):
                with self.assertRaises(Foams105ExportError) as caught:
                    write_foams105_export_bundle(result, destination)
            self.assertEqual(caught.exception.code, "bundle_write_failed")
            self.assertFalse(destination.exists())
            self.assertEqual(tuple(Path(root).iterdir()), ())

    def test_unavailable_area_projection_does_not_block_other_tables(self):
        result = _build_result()
        group = result.prepared_result.groups[0]
        image = group.images[0]
        zero_area = replace(image.image_area_result, phase_corrected_area_mm2=0.0)
        altered_image = replace(image, image_area_result=zero_area)
        altered_group = replace(group, images=(altered_image,) + group.images[1:])
        altered_result = replace(
            result,
            prepared_result=replace(result.prepared_result, groups=(altered_group,)),
        )
        projections = _project_foams105_export_tables(altered_result)
        tables = {table.filename: table for table in projections.tables}
        self.assertFalse(projections.complete)
        self.assertEqual(tables["Area_out.txt"].status, "unavailable")
        self.assertEqual(tables["Calc_out.txt"].status, "available")

    def test_undefined_numeric_cells_serialize_as_na_with_classification(self):
        diagnostics = []
        formatted = _format_cell(
            Foams105RealValue(
                value=None,
                reason="zero_solid_fraction",
                legacy_classification="positive_infinity",
            ),
            table="Misc_out.txt",
            row_index=0,
            column="NVcorr",
            diagnostics=diagnostics,
        )
        self.assertEqual(formatted, "NA")
        self.assertEqual(diagnostics[0].reason, "zero_solid_fraction")
        self.assertEqual(diagnostics[0].legacy_classification, "positive_infinity")


if __name__ == "__main__":
    unittest.main()
