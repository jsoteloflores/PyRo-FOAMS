from __future__ import annotations

import csv
import json
import shutil
import tempfile
import threading
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from main.core.analysis_runs import (
    SavedAnalysisRun,
    append_run,
    create_run_projection,
    digest,
    json_bytes,
    structured,
    validate_run,
)
from main.core.legacy_exports import write_foams105_export_bundle
from main.core.legacy_summaries import Foams105RealValue
from main.core.run_export_schema import SHAPE_REDUCTIONS, TABLE_COLUMNS, summary_rows
from main.core.run_exports import TABLE_KEYS, _cell, export_selected_run, run_tables
from main.core.sample_analysis import (
    CapturedAnalysisRequest,
    LegacyRunConfig,
    ModernRunConfig,
    analyze_legacy_prepared,
    analyze_modern_prepared,
    prepare_legacy_analysis,
    prepare_modern_analysis,
)
from main.core.sample_project import create_analysis_snapshot, set_analysis_included, set_calibration
from main.core.sample_project_io import SampleProjectIoError, open_project, save_project


class TestSavedAnalysisRuns(unittest.TestCase):
    def test_explicit_shape_reductions_and_membership_match_public_results(self):
        project, _, outcomes = self.completed()
        snapshot = create_analysis_snapshot(project)
        scale = max(group.scale_px_per_mm for group in outcomes["legacy"].prepared.groups)
        minimum = outcomes["legacy"].analysis.shape_result.rows[0].equivalent_diameter_mm * scale
        for mode, threshold in (("initial_analysis", minimum), ("updated_minimum", 100.0)):
            with self.subTest(mode=mode):
                context = prepare_legacy_analysis(snapshot, LegacyRunConfig(minimum_diameter_px=minimum))
                labels = context.prepared.label_preparation_result.bin_labels_mm
                config = LegacyRunConfig(minimum_diameter_px=minimum, selection_mode="manual",
                    manual_ranges=(("group-1", labels[0], labels[-1]),), shape_statistics_mode=mode, shape_minimum_diameter_px=threshold)
                legacy = analyze_legacy_prepared(context, config)
                request = CapturedAnalysisRequest(uuid.uuid4().hex, "session", "Original FOAMS 1.0.5", "run", snapshot, (), (), (), config, None)
                projection = create_run_projection(request, {"legacy": legacy}, project.sample_name)
                tables = run_tables(projection.record)
                metrics = {row["metric_id"]: row for row in tables["summary_metrics"]}
                public = legacy.analysis.shape_statistics_result
                if mode == "initial_analysis":
                    self.assertNotEqual(public.summary_population_identities, public.threshold_identities)
                for statistic in public.statistic_rows:
                    for reduction in SHAPE_REDUCTIONS:
                        row = metrics[f"shape_statistics.{statistic.metric}.{reduction}"]
                        self.assertEqual(row["value"], getattr(statistic, reduction))
                        self.assertEqual(row["unit"], "count" if reduction in ("n", "population_count") else "1")
                        self.assertEqual(row["status"], statistic.status)
                        self.assertEqual(row["reason"], statistic.reason or ("not supplied by source" if getattr(statistic, reduction) is None else ""))
                        self.assertEqual(row["included_identities"], structured(statistic.included_identities))
                        self.assertEqual(row["excluded_identities"], structured(statistic.excluded_identities))
                self.assertEqual(metrics["shape_statistics.maximum_configured_scale_px_per_mm"]["unit"], "px/mm")
                self.assertEqual(metrics["shape_statistics.minimum_diameter_px"]["unit"], "px")
                self.assertEqual(metrics["shape_statistics.minimum_diameter_px"]["kind"], "setting")
                for row in tables["objects"]:
                    identity = (row["image_id"], row["object_id"])
                    self.assertEqual(row["shape_population"], identity in public.summary_population_identities)
                    self.assertEqual(row["shape_threshold_population"], identity in public.threshold_identities)
                    self.assertEqual(row["shape_histogram_population"], identity in legacy.analysis.shape_histogram_result.population_identities)

    def test_undefined_wrapper_is_one_quantity_with_reason(self):
        wrapper = Foams105RealValue(None, reason="division_by_zero", legacy_classification="positive_infinity", operands=(1.0, 0.0))
        rows = summary_rows({"project_id": "project", "run_id": "run", "method_id": "legacy"},
                            {"summary": {"corrected_nv_per_mm3": structured(wrapper)}})
        corrected = next(row for row in rows if row["metric_id"] == "summary.corrected_nv_per_mm3")
        self.assertIsNone(corrected["value"])
        self.assertEqual(corrected["availability"], "unavailable")
        self.assertEqual(corrected["reason"], "division_by_zero")
        self.assertEqual(corrected["classification"], "positive_infinity")
        self.assertFalse(any(row["metric_id"].endswith((".reason", ".operands", ".legacy_classification")) for row in rows))

    def test_schema1_reprojection_preserves_record_and_canonical_bytes(self):
        project, projection, _ = self.completed()
        data = projection.record.data()
        data["schema"] = 1
        data.pop("table_projection_schema")
        data["tables"]["summary_metrics"] = []
        old = SavedAnalysisRun(projection.record.run_id, project.project_id, json_bytes(data))
        project.analysis_runs = [old]
        before = old.payload
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_project(project, root / "project")
            restored = open_project(root / "project")
            loaded = restored.analysis_runs[0]
            self.assertEqual(loaded.payload, before)
            self.assertEqual(loaded.index_entry()["schema"], 1)
            with patch("main.core.sample_analysis.prepare_legacy_analysis", side_effect=AssertionError("migration reran science")):
                tables = run_tables(loaded)
                output = export_selected_run(loaded, restored.analysis_assets, root / "export")
            self.assertTrue(any(row["metric_id"] == "shape_statistics.aspect_ratio.mean" for row in tables["summary_metrics"]))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["source_run_schema"], 1)
            self.assertEqual(manifest["table_projection_schema"], 2)
            self.assertEqual(loaded.payload, before)
            for artifact in data["canonical_original"]:
                self.assertEqual((output / "original_compatible" / artifact["filename"]).read_bytes(), project.analysis_assets[artifact["sha256"]])

    def test_complete_headers_and_explicit_units_on_failed_exports(self):
        project, _, _ = self.completed()
        request = CapturedAnalysisRequest(uuid.uuid4().hex, "session", "Compare methods", "run", create_analysis_snapshot(project), (), (), (), None, None)
        projection = create_run_projection(request, {"legacy_error": ValueError("bad legacy"), "modern_error": ValueError("bad modern")}, project.sample_name)
        append_run(project, projection)
        with tempfile.TemporaryDirectory() as folder:
            output = export_selected_run(projection.record, project.analysis_assets, Path(folder) / "export")
            dictionary = json.loads((output / "data_dictionary.json").read_text())
            for table, declared in TABLE_COLUMNS.items():
                with (output / (table + ".csv")).open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    self.assertTrue(set(declared).issubset(reader.fieldnames))
            self.assertEqual(dictionary["objects.csv"]["units"]["orientation_deg"], "deg")
            self.assertEqual(dictionary["objects.csv"]["units"]["area_px"], "px^2")
            self.assertEqual(dictionary["objects.csv"]["units"]["shape_population_availability"], "text")
            self.assertEqual(dictionary["groups.csv"]["units"]["scale_px_per_mm"], "px/mm")
            self.assertTrue(dictionary["objects.csv"]["foreign_keys"][0]["columns"])
    def test_export_keys_values_artifacts_and_input_assets(self):
        project, projection, _ = self.completed()
        tables = run_tables(projection.record)
        with tempfile.TemporaryDirectory() as folder:
            output = export_selected_run(projection.record, project.analysis_assets, Path(folder) / "bundle")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["run_id"], projection.record.run_id)
            for name, keys in TABLE_KEYS.items():
                with (output / (name + ".csv")).open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), len(tables[name]))
                self.assertEqual(len({tuple(row[key] for key in keys) for row in rows}), len(rows))
            for name, info in manifest["files"].items():
                self.assertEqual(digest((output / name).read_bytes()), info["sha256"])
            for artifact in projection.record.data()["canonical_original"]:
                self.assertEqual((output / "original_compatible" / artifact["filename"]).read_bytes(), project.analysis_assets[artifact["sha256"]])
            for image in projection.record.data()["images"]:
                for checksum in image["asset_ids"].values():
                    self.assertEqual((output / "inputs" / (checksum + ".npy")).read_bytes(), project.analysis_assets[checksum])

    def test_export_failure_and_cancellation_leave_no_bundle(self):
        project, projection, _ = self.completed()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "bundle"
            with patch("main.core.run_exports.os.rename", side_effect=OSError("injected publication failure")):
                with self.assertRaises(OSError):
                    export_selected_run(projection.record, project.analysis_assets, output)
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(folder).iterdir()))
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(InterruptedError):
                export_selected_run(projection.record, project.analysis_assets, output, cancel)
            self.assertFalse(output.exists())
            output.mkdir()
            with self.assertRaises(FileExistsError):
                export_selected_run(projection.record, project.analysis_assets, output)

    def test_decimal_round_trip_missing_zero_and_signed_values(self):
        for value in (0.0, -0.0, -0.12345678901234567, 1.2345678901234567, 1e-250):
            self.assertEqual(float(_cell(value)).hex(), value.hex())
        self.assertEqual(_cell(None), "")
        self.assertNotEqual(_cell(0.0), "")

    def test_partial_and_total_failure_have_no_phantom_native_rows(self):
        project, _, results = self.completed()
        for legacy_failed, modern_failed in ((True, False), (False, True), (True, True)):
            with self.subTest(legacy_failed=legacy_failed, modern_failed=modern_failed):
                snapshot = create_analysis_snapshot(project)
                request = CapturedAnalysisRequest(uuid.uuid4().hex, "session", "Compare methods", "run", snapshot,
                                                  (), (), (), LegacyRunConfig(), ModernRunConfig())
                outcomes = dict(results)
                for method, failed in (("legacy", legacy_failed), ("modern", modern_failed)):
                    if failed:
                        outcomes[method] = None
                        outcomes[method + "_error"] = ValueError("explicit method failure")
                projection = create_run_projection(request, outcomes, project.sample_name,
                    tuple((image.image_id, image.source_image) for image in project.images))
                data = projection.record.data()
                for method, failed in (("legacy", legacy_failed), ("modern", modern_failed)):
                    native = next(item for item in data["methods"] if item["method_id"] == method)
                    self.assertEqual(native["outcome"], "failed" if failed else "success")
                    if failed:
                        self.assertIsNone(native["native"])
                        self.assertFalse([row for row in data["tables"]["objects"] if row["method_id"] == method])
                        self.assertFalse([row for row in data["tables"]["distributions"] if row["method_id"] == method])

    def test_schema2_migrates_to_empty_history(self):
        project, _, _ = self.completed()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_project(project, root)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["schema_version"] = 2
            manifest.pop("analysis_runs")
            manifest.pop("analysis_presentation_errors")
            (root / "manifest.json").write_text(json.dumps(manifest))
            restored = open_project(root)
            self.assertEqual(restored.schema_version, 3)
            self.assertFalse(restored.analysis_runs)
            self.assertFalse(restored.unavailable_history)

    def test_unicode_quotes_newlines_and_no_prepare_history(self):
        project, _, results = self.completed()
        snapshot = create_analysis_snapshot(project)
        request = CapturedAnalysisRequest(uuid.uuid4().hex, "session", "Compare methods", "run", snapshot,
                                          (), (), (), LegacyRunConfig(), ModernRunConfig())
        name = 'Sample, "quoted"\n\u00e9vidence'
        projection = create_run_projection(request, results, name,
            tuple((image.image_id, image.source_image) for image in project.images))
        append_run(project, projection)
        with tempfile.TemporaryDirectory() as folder:
            output = export_selected_run(projection.record, project.analysis_assets, Path(folder) / "bundle")
            with (output / "sample.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual(next(csv.DictReader(handle))["sample_name"], name)
        for action in ("prepare", "suggest", "preview"):
            with self.assertRaisesRegex(ValueError, "Only explicit Run"):
                create_run_projection(replace(request, action=action), results, name)

    def completed(self, mode="Compare methods"):
        from main.tests.test_sample_analysis import TestSampleAnalysisAdapters
        project, _ = TestSampleAnalysisAdapters()._snapshot_with_three_images()
        set_analysis_included(project.images[2], False)
        snapshot = create_analysis_snapshot(project)
        context = prepare_legacy_analysis(snapshot)
        labels = context.prepared.label_preparation_result.bin_labels_mm
        legacy_config = LegacyRunConfig(selection_mode="manual", manual_ranges=(("group-1", labels[0], labels[-1]),))
        modern_config = ModernRunConfig(min_detectable_mm=0.001, bin_edges_mm=(0.001, 0.01, 0.1))
        modern_prepared = prepare_modern_analysis(snapshot, modern_config)
        results = {"legacy": analyze_legacy_prepared(context, legacy_config) if mode != "Modern analysis" else None,
                   "modern": analyze_modern_prepared(snapshot, modern_prepared, modern_config) if not mode.startswith("Original") else None,
                   "legacy_error": None, "modern_error": None, "modern_prepared": modern_prepared}
        request = CapturedAnalysisRequest(uuid.uuid4().hex, "session", mode, "run", snapshot, (project.project_id,),
                                          (("modeVar", mode),), (), legacy_config, modern_config)
        projection = create_run_projection(request, results, project.sample_name,
            tuple((image.image_id, image.source_image) for image in project.images))
        append_run(project, projection)
        return project, projection, results

    def test_native_projection_round_trip_and_portable_history(self):
        for mode in ("Original FOAMS 1.0.5", "Modern analysis", "Compare methods"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                project, projection, _ = self.completed(mode)
                captured = projection.record.payload
                project.sample_name = "renamed"
                set_calibration(project.images[0], 0.004, "mm")
                root = Path(folder) / "first"
                save_project(project, root)
                with patch("main.core.sample_analysis.prepare_legacy_analysis", side_effect=AssertionError("science during load")):
                    restored = open_project(root)
                self.assertEqual(restored.analysis_runs[0].payload, captured)
                self.assertEqual(restored.analysis_runs[0].data()["sample_name"], "Sample")
                self.assertEqual(restored.analysis_assets, project.analysis_assets)
                destination = Path(folder) / "save-as"
                save_project(restored, destination)
                shutil.rmtree(root)
                self.assertEqual(open_project(destination).analysis_runs[0].payload, captured)

    def test_original_artifacts_match_public_writer(self):
        project, projection, results = self.completed()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "reference"
            write_foams105_export_bundle(results["legacy"].analysis, output)
            for artifact in projection.record.data()["canonical_original"]:
                self.assertEqual(project.analysis_assets[artifact["sha256"]], (output / artifact["filename"]).read_bytes())
        native = projection.record.data()["methods"][0]["native"]
        self.assertEqual(sum(item["selected"] for item in native["objects"]), results["legacy"].analysis.raw_selection_result.selected_count)

    def test_interrupted_publication_keeps_last_manifest(self):
        project, _, _ = self.completed()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_project(project, root)
            before = (root / "manifest.json").read_bytes()
            project.sample_name = "new draft name"
            with patch("main.core.sample_project_io._atomic_write_manifest", side_effect=OSError("injected publication failure")):
                with self.assertRaises(OSError):
                    save_project(project, root)
            self.assertEqual((root / "manifest.json").read_bytes(), before)
            self.assertEqual(len(open_project(root).analysis_runs), 1)

    def test_corrupt_or_future_history_keeps_preparation_usable(self):
        project, _, _ = self.completed()
        for failure in ("asset", "schema", "escape"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                save_project(project, root)
                manifest = json.loads((root / "manifest.json").read_text())
                entry = manifest["analysis_runs"][0]
                if failure == "asset":
                    data = project.analysis_runs[0].data()
                    (root / data["assets"][0]["path"]).write_bytes(b"damaged")
                elif failure == "schema":
                    entry["schema"] = 999
                else:
                    entry["path"] = "../escaped.json"
                (root / "manifest.json").write_text(json.dumps(manifest))
                restored = open_project(root)
                self.assertEqual(len(restored.images), 3)
                self.assertTrue(restored.unavailable_history)
                self.assertFalse(restored.analysis_runs)
                with self.assertRaises(SampleProjectIoError):
                    save_project(restored, root)

    def test_run_contents_are_immutable_and_assets_checksummed(self):
        project, projection, _ = self.completed()
        data = projection.record.data()
        data["images"][0]["mm_per_px"] = 100
        self.assertEqual(projection.record.data()["images"][0]["mm_per_px"], 0.003)
        for checksum, payload in project.analysis_assets.items():
            self.assertEqual(digest(payload), checksum)
        validate_run(projection.record)
