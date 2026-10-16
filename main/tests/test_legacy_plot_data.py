from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from main.core.legacy_areas import (
    aggregate_foams105_group_areas,
    calculate_foams105_image_areas,
)
from main.core.legacy_measurements import measure_foams105_image, measure_foams105_phase
from main.core.legacy_perimeter import FOAMS105_PERIMETER_PROFILE_CURRENT, measure_foams105_perimeter
from main.core.legacy_plot_data import (
    FOAMS105_HIST_CENTERS,
    FOAMS105_HIST_EDGES,
    FOAMS105_ORIENTATION_HIST_CENTERS,
    Foams105PlotDataValidationError,
    _center_histogram,
    _edge_histogram,
    build_foams105_bulk_na_plot_data,
    build_foams105_shape_histograms,
)
from main.core.legacy_raw_selection import Foams105RawGroupInput, Foams105RawObject, select_foams105_raw_objects
from main.core.legacy_selection import (
    Foams105BinnedGroupInput,
    Foams105BinnedRange,
    select_foams105_binned_ranges,
)
from main.core.legacy_shape_statistics import derive_foams105_shapes, summarize_foams105_shapes
from main.core.legacy_shapes import measure_foams105_moment_geometry
from main.core.legacy_summaries import Foams105CountGroupInput

REFERENCE_PATH = Path(__file__).parent / "fixtures" / "18_foams105_histogram_reference.json"
REFERENCE_SHA256 = "6ee281b1a7be253e04b07d5c1d461b18fa9b2324e88af37127b49522c14c674f"
REFERENCE_LOG_HASHES = {
    "18_matlab_histogram_review.log": "35c4a92bb07917db5cc436ebd64ee0bcc3336b21644070a359cabd6ddb0cb18f",
    "17a_matlab_rounding_review.log": "a6808737ee413bbc42f3b842d7b8d40a9aca296cc6e276a14e6e6e978400323a",
    "17a_matlab_rounding_boundaries.log": "00e66ac3529207989686ae27b6c3be5aec37cb6f76bee271d7a4a427f4b63937",
}


def _fixture():
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


class TestFoams105HistogramReference(unittest.TestCase):
    def test_fixture_sha256_is_pinned(self):
        self.assertEqual(sha256(REFERENCE_PATH.read_bytes()).hexdigest(), REFERENCE_SHA256)
        for filename, expected in REFERENCE_LOG_HASHES.items():
            path = REFERENCE_PATH.with_name(filename)
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), expected, filename)

    def test_exact_reference_centers_and_captured_hist_counts(self):
        reference = _fixture()
        self.assertEqual(FOAMS105_HIST_CENTERS, tuple(reference["shape_centers"]))
        self.assertEqual(FOAMS105_HIST_EDGES, tuple(reference["shape_centers"]))
        samples = tuple(
            (("shape-fixture", index + 1), value)
            for index, value in enumerate(reference["shape_inputs"])
        )
        center = _center_histogram("fixture_hist", samples, FOAMS105_HIST_CENTERS)
        edge = _edge_histogram("fixture_histc", samples, FOAMS105_HIST_EDGES)
        self.assertEqual(center.counts, tuple(reference["hist_counts"]))
        self.assertEqual(edge.counts, tuple(reference["histc_counts"]))
        self.assertEqual(center.under_range_count, 1)
        self.assertEqual(center.over_range_count, 1)
        self.assertEqual(center.included_count, 7)
        self.assertEqual(edge.under_range_count, 1)
        self.assertEqual(edge.over_range_count, 1)
        self.assertEqual(edge.included_count, 5)

    def test_orientation_midpoint_and_centers_match_matlab_capture(self):
        reference = _fixture()
        self.assertEqual(FOAMS105_ORIENTATION_HIST_CENTERS, tuple(reference["orientation_centers"]))
        samples = tuple(
            (("orientation-fixture", index + 1), value)
            for index, value in enumerate(reference["orientation_inputs"])
        )
        result = _center_histogram(
            "orientation_project_deg",
            samples,
            FOAMS105_ORIENTATION_HIST_CENTERS,
            orientation_flags={identity: True for identity, _ in samples},
        )
        self.assertEqual(result.counts, tuple(reference["orientation_counts"]))
        self.assertEqual(result.orientation_defined_identities, tuple(identity for identity, _ in samples))

    def test_nextafter_neighbors_and_exact_final_histc_edge(self):
        center_boundary = 0.025
        samples = tuple(
            (("center", index), value)
            for index, value in enumerate(
                (
                    math.nextafter(center_boundary, -math.inf),
                    center_boundary,
                    math.nextafter(center_boundary, math.inf),
                )
            )
        )
        center = _center_histogram("boundary", samples, FOAMS105_HIST_CENTERS)
        self.assertEqual(center.counts[0], 2)
        self.assertEqual(center.counts[1], 1)

        final_edge = FOAMS105_HIST_EDGES[-1]
        edge_values = tuple(
            (("edge", index), value)
            for index, value in enumerate(
                (
                    0.0,
                    math.nextafter(final_edge, -math.inf),
                    final_edge,
                    math.nextafter(final_edge, math.inf),
                    math.nextafter(0.0, -math.inf),
                )
            )
        )
        edge = _edge_histogram("final-edge", edge_values, FOAMS105_HIST_EDGES)
        self.assertEqual(edge.counts[0], 1)
        self.assertEqual(edge.counts[-2], 1)
        self.assertEqual(edge.counts[-1], 1)
        self.assertEqual(edge.under_range_count, 1)
        self.assertEqual(edge.over_range_count, 1)
        self.assertEqual(edge.omitted_count, 2)

        orientation_boundary = 5.0
        orientation_samples = tuple(
            (("orientation", index), value)
            for index, value in enumerate(
                (
                    math.nextafter(orientation_boundary, -math.inf),
                    orientation_boundary,
                    math.nextafter(orientation_boundary, math.inf),
                )
            )
        )
        orientation = _center_histogram(
            "orientation_project_deg",
            orientation_samples,
            FOAMS105_ORIENTATION_HIST_CENTERS,
        )
        self.assertEqual(orientation.counts[0], 2)
        self.assertEqual(orientation.counts[1], 1)


class TestFoams105ShapeHistogramIntegration(unittest.TestCase):
    def test_histograms_use_threshold_identities_not_initial_summary_population(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        image[2, 2] = 1
        image[5:7, 5:7] = 1
        measurement = measure_foams105_phase(image, 1, 1.0)
        moments = measure_foams105_moment_geometry("hist-image", measurement)
        perimeter = measure_foams105_perimeter(
            "hist-image", measurement, profile=FOAMS105_PERIMETER_PROFILE_CURRENT
        )
        objects = tuple(
            Foams105RawObject("hist-image", c.component_id, c.equivalent_diameter_mm, c.area_mm2)
            for c in measurement.components
        )
        raw = select_foams105_raw_objects(
            (0.01, 0.1, 3.0),
            (Foams105RawGroupInput("g", 1, objects),),
            (Foams105BinnedRange("g", 0.1, 3.0),),
        )
        shape = derive_foams105_shapes(raw, (moments,), (perimeter,))
        statistics = summarize_foams105_shapes(
            shape,
            configured_scales_px_per_mm=(1.0,),
            minimum_diameter_px=1.5,
            mode="initial_analysis",
        )
        self.assertEqual(statistics.summary_population_count, 2)
        self.assertEqual(statistics.threshold_count, 1)
        result = build_foams105_shape_histograms(statistics)
        self.assertEqual(result.population_identities, statistics.threshold_identities)
        self.assertEqual(result.eccentricity.population_count, 1)
        self.assertEqual(result.regularity.population_count, 1)
        self.assertEqual(len(result.eccentricity.counts), 21)
        self.assertEqual(len(result.orientation_project_deg.counts), 19)

    def test_empty_histograms_have_exact_zero_counts(self):
        empty_raw = select_foams105_raw_objects(
            (0.1, 1.0, 2.0),
            (Foams105RawGroupInput("empty", 1, ()),),
            (Foams105BinnedRange("empty", 1.0, 2.0),),
        )
        empty_shapes = derive_foams105_shapes(empty_raw, (), ())
        stats = summarize_foams105_shapes(
            empty_shapes,
            configured_scales_px_per_mm=(1.0,),
            minimum_diameter_px=0.0,
            mode="updated_minimum",
        )
        result = build_foams105_shape_histograms(stats)
        self.assertEqual(result.eccentricity.counts, (0,) * 21)
        self.assertEqual(result.orientation_project_deg.counts, (0,) * 19)
        self.assertEqual(result.regularity.counts, (0,) * 21)

    def test_invalid_statistics_result_is_rejected(self):
        with self.assertRaises(Foams105PlotDataValidationError):
            build_foams105_shape_histograms(None)


class TestFoams105BulkNaPlotData(unittest.TestCase):
    @staticmethod
    def _area(group_id, phase_area):
        measurement = measure_foams105_image(np.zeros((2, 2), dtype=np.uint8), 1, 1.0)
        image_area = calculate_foams105_image_areas(f"image-{group_id}", measurement)
        grouped = aggregate_foams105_group_areas(group_id, (image_area,))
        return replace(grouped, phase_corrected_area_mm2=phase_area)

    def _selection(self, fourth_enabled=True):
        groups = tuple(
            Foams105BinnedGroupInput(
                f"group-{slot}",
                slot,
                (1.0, 1.0, 1.0),
            )
            for slot in range(1, 5)
        )
        ranges = tuple(
            Foams105BinnedRange(
                group.group_id,
                1.0 if group.source_slot == 4 else 3.0,
                2.0 if group.source_slot == 4 else 3.0,
            )
            if group.source_slot != 4 or fourth_enabled
            else None
            for group in groups
        )
        return select_foams105_binned_ranges((1.0, 2.0, 3.0), groups, ranges)

    def test_slot_four_uses_slot_three_counts_and_area_but_fourth_x_area(self):
        selection = self._selection()
        counts = tuple(
            Foams105CountGroupInput(
                f"group-{slot}",
                slot,
                (2, 3, 5) if slot == 3 else (100, 200, 300) if slot == 4 else (1, 1, 1),
            )
            for slot in range(1, 5)
        )
        areas = tuple(
            self._area(f"group-{slot}", {1: 4.0, 2: 6.0, 3: 10.0, 4: 20.0}[slot])
            for slot in range(1, 5)
        )
        result = build_foams105_bulk_na_plot_data(selection, counts, areas)
        slot_four = next(item for item in result.slots_descending if item.source_slot == 4)
        self.assertEqual(slot_four.density_source_indices, (0, 1))
        self.assertEqual(slot_four.numerator, 5)
        self.assertEqual(slot_four.count_source_slot, 3)
        self.assertEqual(slot_four.denominator_source_slot, 3)
        self.assertEqual(slot_four.denominator_area1_mm2, 10.0)
        self.assertEqual(slot_four.plot_x_area1_mm2, 20.0)
        self.assertEqual(slot_four.value.value, 0.5)
        self.assertIn("source_slot4_uses_slot3_counts_and_area", slot_four.diagnostics)
        self.assertNotEqual(slot_four.value.value, 15.0)
        self.assertEqual(result.source_x_areas_nonzero_mm2, (20.0, 10.0, 6.0, 4.0))
        self.assertTrue(result.plot_ready)

    def test_disabled_slice_and_zero_area_keep_literal_mismatched_vectors(self):
        selection = self._selection(fourth_enabled=False)
        counts = tuple(
            Foams105CountGroupInput(f"group-{slot}", slot, (1, 1, 1))
            for slot in range(1, 5)
        )
        areas = tuple(
            self._area(f"group-{slot}", 0.0 if slot == 1 else float(slot * 5))
            for slot in range(1, 5)
        )
        result = build_foams105_bulk_na_plot_data(selection, counts, areas)
        slot_four = next(item for item in result.slots_descending if item.source_slot == 4)
        self.assertIsNone(slot_four.value.value)
        self.assertEqual(slot_four.value.reason, "no_source_slice")
        self.assertEqual(result.source_x_areas_nonzero_mm2, (20.0, 15.0, 10.0))
        self.assertEqual(len(result.source_y_values), 4)
        self.assertNotEqual(len(result.source_x_areas_nonzero_mm2), len(result.source_y_values))
        self.assertFalse(result.plot_ready)
        self.assertIn("bulk_na_source_slice_unavailable", result.diagnostics)
        self.assertIn("bulk_na_source_x_y_length_mismatch", result.diagnostics)


if __name__ == "__main__":
    unittest.main()
