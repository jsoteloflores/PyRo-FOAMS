from __future__ import annotations

import unittest

from main.core.analysis_plots import reconstruction_series, visible_points
from main.core.run_exports import run_tables
from main.core.sample_analysis import ModernRunConfig, run_modern_analysis


class TestReconstructionPlots(unittest.TestCase):
    def test_original_points_are_native_source_order_and_joinable(self):
        from main.tests.test_analysis_runs import TestSavedAnalysisRuns
        _, projection, results = TestSavedAnalysisRuns().completed()
        series, messages = reconstruction_series(projection.record.data(), run_tables(projection.record))
        original = next(item for item in series if item["method_id"] == "legacy")
        self.assertEqual([point["diameter_mm"] for point in original["points"]],
                         [row.label_mm for row in results["legacy"].analysis.binned_selection_result.rows])
        self.assertEqual([point["signed_nv_per_mm3"] for point in original["points"]],
                         list(results["legacy"].analysis.conversion_result.signed_nv_per_mm3))
        self.assertIn("3D reconstruction was not calculated", messages[0])
        selected = [row for row in run_tables(projection.record)["distributions"] if row["method_id"] == "legacy" and row["stage"] == "selected"]
        self.assertEqual([point["row_ordinal"] for point in original["points"]], [row["row_ordinal"] for row in selected])

        from main.core.analysis_runs import SavedAnalysisRun, json_bytes
        data = projection.record.data()
        data["schema"] = 1
        data.pop("table_projection_schema")
        old = SavedAnalysisRun(projection.record.run_id, projection.record.project_id, json_bytes(data))
        before = old.payload
        historical, _ = reconstruction_series(old.data(), run_tables(old))
        self.assertEqual(historical, series)
        self.assertEqual(old.payload, before)

    def test_signed_duplicate_zero_and_single_point_visibility(self):
        points = [{"diameter_mm": 0.1, "signed_nv_per_mm3": value, "row_ordinal": index}
                  for index, value in enumerate((-2.0, 0.0, 3.0, -4.0))]
        visible, counts = visible_points(points)
        self.assertEqual(visible, points)
        self.assertFalse(any(counts.values()))
        visible, counts = visible_points(points, log_y=True)
        self.assertEqual(visible, [points[2]])
        self.assertEqual(counts["negative"], 2)
        self.assertEqual(counts["zero"], 1)
        self.assertEqual(visible_points(points[:1])[0], points[:1])

    def test_modern_coordinate_contract_is_upper_class_edge(self):
        from main.core.sample_project import create_analysis_snapshot, set_analysis_included
        from main.tests.test_sample_analysis import TestSampleAnalysisAdapters
        project, _ = TestSampleAnalysisAdapters()._snapshot_with_three_images()
        set_analysis_included(project.images[2], False)
        result = run_modern_analysis(create_analysis_snapshot(project), ModernRunConfig(min_detectable_mm=0.001,
            bin_edges_mm=(0.001, 0.01, 0.1), enable_reconstruction=True,
            upper_tail_assumption="zero_beyond_selected_upper_edge"))
        from main.core.analysis_runs import structured
        native = {"reconstruction": structured(result.reconstruction_result)}
        rows = [{"method_id": "modern", "stage": "selected", "row_ordinal": index,
                 "group_id": "group-1", "bin_index": index} for index in range(2)]
        series, _ = reconstruction_series({"project_id": "project", "run_id": "run", "methods": [
            {"method_id": "modern", "outcome": "success", "native": native}]}, {"distributions": rows})
        self.assertEqual([point["diameter_mm"] for point in series[0]["points"]],
                         list(result.reconstruction_result.operator.representative_diameters_mm))
        self.assertEqual([point["signed_nv_per_mm3"] for point in series[0]["points"]], list(result.reconstruction_result.signed_nv_per_mm3))
