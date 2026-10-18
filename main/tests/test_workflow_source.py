from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from main.core.legacy_exports import build_foams105_export_tables
from main.core.legacy_summaries import Foams105RealValue
from main.reference_tools.workflow_compare import (
    REAL_LOG_EXTENSION_POLICY,
    ClassifiedNumber,
    WorkflowComparisonReport,
    WorkflowComparisonRow,
    _assess_real_log_extension,
    _compare_cell,
    _compare_shape_vectors,
    workflow_comparison_json,
    workflow_comparison_markdown,
)
from main.reference_tools.workflow_package import import_workflow_reference_package
from main.reference_tools.workflow_reference import (
    WorkflowReferenceValidationError,
    extract_mapped_source_fragment,
    load_workflow_case_catalog,
    load_workflow_extraction_map,
    run_python_workflow_case,
    verify_workflow_source_checkout,
)
from main.reference_tools.workflow_source import (
    WorkflowSourceValidationError,
    extract_source_fragment,
    source_hashes,
)


class TestWorkflowSourceHashes(unittest.TestCase):
    def test_raw_hash_distinguishes_transport_and_lf_hash_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lf_path = root / "lf.m"
            crlf_path = root / "crlf.m"
            lf_path.write_bytes(b"first\nsecond\n")
            crlf_path.write_bytes(b"first\r\nsecond\r\n")

            lf_raw, lf_content = source_hashes(lf_path)
            crlf_raw, crlf_content = source_hashes(crlf_path)

        self.assertNotEqual(lf_raw, crlf_raw)
        self.assertEqual(lf_content, crlf_content)

    def test_fragment_lines_are_inclusive_and_hash_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.m"
            fragment = b"second = 2;\nthird = 3;\n"
            source_path.write_bytes(b"first = 1;\r\n" + fragment.replace(b"\n", b"\r\n"))
            expected = hashlib.sha256(fragment).hexdigest()

            extracted = extract_source_fragment(source_path, 2, 3, expected_lf_sha256=expected)
            with self.assertRaises(WorkflowSourceValidationError) as caught:
                extract_source_fragment(source_path, 2, 3, expected_lf_sha256="0" * 64)

        self.assertEqual(extracted.text, fragment.decode("ascii"))
        self.assertEqual(extracted.lf_sha256, expected)
        self.assertEqual(caught.exception.code, "fragment_hash_mismatch")

    def test_invalid_fragment_ranges_are_typed(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.m"
            source_path.write_text("one\n", encoding="utf-8")
            with self.assertRaises(WorkflowSourceValidationError) as caught:
                extract_source_fragment(source_path, 0, 1)
        self.assertEqual(caught.exception.code, "invalid_fragment_bounds")


class TestWorkflowSourceMap(unittest.TestCase):
    def test_repository_extraction_map_is_complete_and_pinned(self):
        repository_root = Path(__file__).resolve().parents[2]
        extraction_map = load_workflow_extraction_map(
            repository_root / "reference-tools" / "matlab" / "foams105_workflow_extraction_map.json"
        )
        self.assertEqual(extraction_map["source_commit"], "179663203f2d0f86b2863d5ea7f8f70dadca02f8")
        self.assertGreaterEqual(len(extraction_map["fragments"]), 15)
        self.assertEqual(len(extraction_map["histogram_calls"]), 5)
        self.assertTrue(all(fragment["required_inputs"] for fragment in extraction_map["fragments"]))
        self.assertTrue(all(fragment["produced_variables"] for fragment in extraction_map["fragments"]))

    def test_mapped_fragment_audits_only_explicitly_omitted_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.m"
            source_bytes = b"first = 1;\nwaitbar(0.5)\nsecond = 2;\n"
            source_path.write_bytes(source_bytes)
            fragment_hash = hashlib.sha256(source_bytes).hexdigest()
            extraction_map = {
                "fragments": [
                    {
                        "id": "small",
                        "file": "source.m",
                        "start_line": 1,
                        "end_line": 3,
                        "lf_sha256": fragment_hash,
                        "omitted_calls": [{"call": "waitbar", "reason": "progress only"}],
                    }
                ]
            }
            fragment = extract_mapped_source_fragment(root, extraction_map, "small")

        self.assertEqual(fragment.executable_text, "first = 1;\nsecond = 2;\n")
        self.assertEqual(len(fragment.omitted_lines), 1)
        self.assertEqual(fragment.omitted_lines[0].line_number, 2)
        self.assertEqual(fragment.omitted_lines[0].text, "waitbar(0.5)")

    def test_mapped_fragment_requires_every_declared_statistics_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "newstats.m"
            source_path.write_text("stats_asp_rat = 1;\nstats_ecc = 2;\n", encoding="utf-8")
            source_map = {
                "fragments": [
                    {
                        "id": "updated_stats",
                        "file": "newstats.m",
                        "start_line": 1,
                        "end_line": 2,
                        "lf_sha256": source_hashes(source_path)[1],
                        "required_assignments": ["stats_asp_rat", "stats_ecc"],
                        "omitted_calls": [],
                    }
                ]
            }
            valid = extract_mapped_source_fragment(root, source_map, "updated_stats")
            source_path.write_text("stats_asp_rat = 1;\n", encoding="utf-8")
            source_map["fragments"][0]["end_line"] = 1
            source_map["fragments"][0]["lf_sha256"] = source_hashes(source_path)[1]
            with self.assertRaises(WorkflowReferenceValidationError) as caught:
                extract_mapped_source_fragment(root, source_map, "updated_stats")

        self.assertIn("stats_ecc = 2", valid.executable_text)
        self.assertEqual(caught.exception.code, "missing_source_assignment")

    def test_unmapped_gui_calls_fail_instead_of_being_stripped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.m"
            source_bytes = b"figure(1)\nvalue = 2;\n"
            source_path.write_bytes(source_bytes)
            extraction_map = {
                "fragments": [
                    {
                        "id": "small",
                        "file": "source.m",
                        "start_line": 1,
                        "end_line": 2,
                        "lf_sha256": hashlib.sha256(source_bytes).hexdigest(),
                        "omitted_calls": [],
                    }
                ]
            }
            with self.assertRaises(WorkflowReferenceValidationError) as caught:
                extract_mapped_source_fragment(root, extraction_map, "small")
        self.assertEqual(caught.exception.code, "unmapped_nonnumerical_call")

    def test_source_checkout_requires_exact_commit_and_file_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            source_path = root / "data_meas.m"
            source_bytes = b"function y = f(x)\ny = x;\nend\n"
            source_path.write_bytes(source_bytes)
            subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Workflow Test"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "workflow-test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "data_meas.m"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-m", "source"], check=True, capture_output=True)
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            raw_hash, lf_hash = source_hashes(source_path)
            source_map = {
                "source_commit": commit,
                "source_files": {"data_meas.m": {"raw_sha256": raw_hash, "lf_sha256": lf_hash}},
            }
            verified = verify_workflow_source_checkout(root, source_map)
            self.assertEqual(verified.commit, commit)

            source_path.write_bytes(b"modified source\n")
            with self.assertRaises(WorkflowReferenceValidationError) as caught:
                verify_workflow_source_checkout(root, source_map)
        self.assertEqual(caught.exception.code, "source_hash_mismatch")


class TestWorkflowCaseCatalog(unittest.TestCase):
    catalog_path = Path(__file__).parent / "fixtures" / "19_foams105_workflow_cases.json"

    def test_every_catalog_case_runs_through_public_python_apis(self):
        catalog = load_workflow_case_catalog(self.catalog_path)
        results = {run.case_id: run_python_workflow_case(run) for run in catalog.runs}
        expected_failure = results["empty_autosmart_expected_domain_failure"]
        self.assertIsNone(expected_failure.analysis_result)
        self.assertEqual(expected_failure.failure_stage, "autosmart_cutoffs")
        self.assertEqual(expected_failure.failure_code, "empty_group_support")

        for case_id in (
            "single_manual_unnormalized_initial",
            "single_manual_integrated_2d",
            "single_manual_known_vesicularity",
            "two_scale_autosmart_integrated",
            "two_scale_manual_known",
            "four_slot_manual_slot4_diagnostic",
        ):
            result = results[case_id].analysis_result
            self.assertIsNotNone(result, case_id)
            exports = build_foams105_export_tables(result)
            self.assertTrue(exports.complete, case_id)
            tables = {table.filename: table for table in exports.tables}
            self.assertEqual(len(tables["Shape_out.txt"].rows), result.raw_selection_result.selected_count)
            self.assertEqual(len(tables), 9)

        four_slot = results["four_slot_manual_slot4_diagnostic"].analysis_result
        slot4 = four_slot.bulk_na_plot_data_result.slots_descending[0]
        self.assertEqual(slot4.source_slot, 4)
        self.assertEqual(slot4.count_source_slot, 3)
        self.assertEqual(slot4.denominator_source_slot, 3)

    def test_original_initial_replay_uses_project_minimum(self):
        catalog = load_workflow_case_catalog(self.catalog_path)
        runs = {run.case_id: run for run in catalog.runs}
        initial = run_python_workflow_case(runs["shape_threshold_equality_initial_strict"]).analysis_result
        updated = run_python_workflow_case(runs["shape_threshold_equality_updated_inclusive"]).analysis_result
        self.assertEqual(
            initial.prepared_result.preparation_settings.minimum_diameter_px,
            initial.analysis_settings.shape_minimum_diameter_px,
        )
        self.assertNotEqual(
            updated.prepared_result.preparation_settings.minimum_diameter_px,
            updated.analysis_settings.shape_minimum_diameter_px,
        )

    def test_initial_catalog_rejects_mismatched_project_shape_minima(self):
        payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        case = next(run for run in payload["runs"] if run["case_id"] == "shape_threshold_equality_initial_strict")
        case["analysis"]["shape_minimum_diameter_px"] = 0.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(WorkflowReferenceValidationError) as caught:
                load_workflow_case_catalog(path)
        self.assertEqual(caught.exception.code, "inconsistent_initial_shape_threshold")


class TestWorkflowComparisonNumbers(unittest.TestCase):
    def test_integral_numeric_classes_compare_by_value(self):
        comparison = _compare_cell(
            case_id="case",
            stage="histogram_and_na",
            table_name="NA_mag.txt",
            row_index=0,
            column_index=1,
            column_name="freq_mag1",
            identity=None,
            reference=ClassifiedNumber("finite", 0.0),
            candidate=0,
            unit="count",
        )

        self.assertTrue(comparison.within_tolerance)
        self.assertTrue(comparison.classification_match)

    def test_shape_vectors_compare_source_ordered_values(self):
        rows = []
        _compare_shape_vectors(
            rows,
            case_id="case",
            stage="shape_and_initial_threshold",
            source_shape={
                "area_mm2": {
                    "kind": "numeric_array",
                    "shape": [2, 1],
                    "cells": [
                        {"classification": "finite", "real": "2.0", "imaginary": ""},
                        {"classification": "finite", "real": "3.0", "imaginary": ""},
                    ],
                }
            },
            source_identities=(("image-a", 1), ("image-a", 2)),
            candidate_rows=(SimpleNamespace(area_mm2=2.0), SimpleNamespace(area_mm2=3.0)),
            fields={"area_mm2": ("area_mm2", "mm^2", 1e-10, 1e-12)},
        )

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.within_tolerance for row in rows))
        self.assertEqual(tuple(row.identity for row in rows), (("image-a", 1), ("image-a", 2)))

    def test_undefined_does_not_compare_equal_to_nan(self):
        comparison = _compare_cell(
            case_id="case",
            stage="shape_statistics",
            table_name="Stats_out.txt",
            row_index=0,
            column_index=1,
            column_name="mean",
            identity=None,
            reference=ClassifiedNumber("nan", None),
            candidate=None,
            unit="four-decimal-display",
        )

        self.assertFalse(comparison.within_tolerance)
        self.assertFalse(comparison.classification_match)

    def test_statistics_values_compare_at_display_precision(self):
        comparison = _compare_cell(
            case_id="case",
            stage="shape_statistics",
            table_name="Stats_out.txt",
            row_index=0,
            column_index=4,
            column_name="mean",
            identity=None,
            reference=ClassifiedNumber("finite", 0.35401),
            candidate=0.35404,
            unit="four-decimal-display",
        )

        self.assertTrue(comparison.within_tolerance)
        self.assertTrue(comparison.classification_match)

    def test_reports_serialize_complex_differences_and_mismatch_context(self):
        row = WorkflowComparisonRow(
            case_id="case",
            stage="nv_volume_and_distribution_summary",
            table="Calc_out.txt",
            row_index=1,
            column="Ln(n)",
            identity=None,
            reference_value={"classification": "complex", "value": complex(-2.0, 3.141592653589793)},
            python_value={"classification": "undefined", "value": None, "reason": "log_of_negative"},
            absolute_difference=None,
            relative_difference=None,
            tolerance=(1e-10, 1e-12),
            unit="ln(mm^-4)",
            within_tolerance=False,
            classification_match=False,
            code="value_mismatch",
        )
        report = WorkflowComparisonReport(
            source_commit="pinned",
            matlab_release="R2025b",
            python_revision="test",
            comparison_count=1,
            failure_count=1,
            unavailable_stage_count=0,
            first_divergent_stage=row.stage,
            affected_tables=("Calc_out",),
            known_extensions=(),
            rows=(row,),
        )

        serialized = json.loads(workflow_comparison_json(report))
        markdown = workflow_comparison_markdown(report)

        self.assertEqual(serialized["rows"][0]["reference_value"]["value"]["imaginary"], 3.141592653589793)
        self.assertIn("First divergent stage: `nv_volume_and_distribution_summary`", markdown)
        self.assertIn("log_of_negative", markdown)

    def test_log_zero_extension_requires_matching_source_and_python_operands(self):
        row = WorkflowComparisonRow(
            case_id="case",
            stage="nv_volume_and_distribution_summary",
            table="Calc_out.txt",
            row_index=0,
            column="log(NV>L)",
            identity=None,
            reference_value={"classification": "negative_infinity", "value": None},
            python_value={"classification": "undefined", "value": None, "reason": "log_of_zero"},
            absolute_difference=None,
            relative_difference=None,
            tolerance=(1e-10, 1e-12),
            unit="log10(mm^-3)",
            within_tolerance=False,
            classification_match=False,
        )
        source_case = {
            "volume": {
                "log_inputs": {
                    "log10_cumulative_nv": {
                        "kind": "numeric_array",
                        "shape": [1, 1],
                        "cells": [{"classification": "finite", "real": "0", "imaginary": ""}],
                    },
                    "log10_expression": "Log_Nvl=log10(NV_cumsum);",
                }
            }
        }
        analysis = SimpleNamespace(
            distribution_summary_result=SimpleNamespace(
                rows=(
                    SimpleNamespace(
                        log10_cumulative_nv=Foams105RealValue(
                            None,
                            "log_of_zero",
                            (0.0,),
                            "negative_infinity",
                        ),
                    ),
                )
            )
        )

        assessment = _assess_real_log_extension(row, source_case=source_case, analysis=analysis)
        source_case["volume"]["log_inputs"]["log10_cumulative_nv"]["cells"][0]["real"] = "1"
        wrong_operand_assessment = _assess_real_log_extension(row, source_case=source_case, analysis=analysis)

        self.assertEqual(assessment["policy"], REAL_LOG_EXTENSION_POLICY)
        self.assertEqual(assessment["operation"], "log10")
        self.assertIsNone(wrong_operand_assessment)

    def test_negative_log_extension_rejects_wrong_operation_or_nan(self):
        row = WorkflowComparisonRow(
            case_id="case",
            stage="nv_volume_and_distribution_summary",
            table="Calc_out.txt",
            row_index=0,
            column="Ln(n)",
            identity=None,
            reference_value={"classification": "complex", "value": complex(math.log(2.0), math.pi)},
            python_value={"classification": "undefined", "value": None, "reason": "log_of_negative"},
            absolute_difference=None,
            relative_difference=None,
            tolerance=(1e-10, 1e-12),
            unit="ln(mm^-4)",
            within_tolerance=False,
            classification_match=False,
        )
        source_case = {
            "volume": {
                "log_inputs": {
                    "natural_log_number_density_per_width": {
                        "kind": "numeric_array",
                        "shape": [1, 1],
                        "cells": [{"classification": "finite", "real": "-2", "imaginary": ""}],
                    },
                    "natural_log_expression": "ln_n=log(n_numb);",
                }
            }
        }
        analysis = SimpleNamespace(
            distribution_summary_result=SimpleNamespace(
                rows=(
                    SimpleNamespace(
                        ln_number_density_per_width=Foams105RealValue(
                            None,
                            "log_of_negative",
                            (-2.0,),
                            "complex",
                        ),
                    ),
                )
            )
        )

        assessment = _assess_real_log_extension(row, source_case=source_case, analysis=analysis)
        wrong_result_row = replace(
            row,
            reference_value={"classification": "complex", "value": complex(math.log(2.0), math.pi / 2.0)},
        )
        wrong_result = _assess_real_log_extension(wrong_result_row, source_case=source_case, analysis=analysis)
        unrelated_reason_analysis = SimpleNamespace(
            distribution_summary_result=SimpleNamespace(
                rows=(
                    SimpleNamespace(
                        ln_number_density_per_width=Foams105RealValue(
                            None,
                            "unrelated_undefined",
                            (-2.0,),
                            "complex",
                        ),
                    ),
                )
            )
        )
        unrelated_reason = _assess_real_log_extension(
            row,
            source_case=source_case,
            analysis=unrelated_reason_analysis,
        )
        source_case["volume"]["log_inputs"]["natural_log_expression"] = "ln_n=log10(n_numb);"
        wrong_operation = _assess_real_log_extension(row, source_case=source_case, analysis=analysis)
        source_case["volume"]["log_inputs"]["natural_log_expression"] = "ln_n=log(n_numb);"
        source_case["volume"]["log_inputs"]["natural_log_number_density_per_width"]["cells"][0]["real"] = "-3"
        wrong_source_operand = _assess_real_log_extension(row, source_case=source_case, analysis=analysis)
        source_case["volume"]["log_inputs"]["natural_log_number_density_per_width"]["cells"][0]["real"] = "-2"
        wrong_python_value = SimpleNamespace(
            distribution_summary_result=SimpleNamespace(
                rows=(
                    SimpleNamespace(
                        ln_number_density_per_width=Foams105RealValue(
                            None,
                            "log_of_negative",
                            (-3.0,),
                            "complex",
                        ),
                    ),
                )
            )
        )
        wrong_python_operand = _assess_real_log_extension(
            row,
            source_case=source_case,
            analysis=wrong_python_value,
        )
        row = replace(row, reference_value={"classification": "nan", "value": None})
        nan_assessment = _assess_real_log_extension(row, source_case=source_case, analysis=analysis)

        self.assertEqual(assessment["operation"], "log")
        self.assertIsNone(wrong_result)
        self.assertIsNone(unrelated_reason)
        self.assertIsNone(wrong_operation)
        self.assertIsNone(wrong_source_operand)
        self.assertIsNone(wrong_python_operand)
        self.assertIsNone(nan_assessment)


class TestWorkflowReferencePackage(unittest.TestCase):
    package_path = Path(__file__).parent / "fixtures" / "19a_foams105_reference.json.gz"

    @classmethod
    def setUpClass(cls):
        cls.imported = import_workflow_reference_package(
            cls.package_path,
            repository_root=Path(__file__).resolve().parents[2],
        )

    def _mutated_capture(self):
        capture = copy.deepcopy(self.imported.capture)
        cases = {case["case_id"]: case for case in capture["cases"]}
        return replace(self.imported, capture=capture, cases_by_id=cases)

    def test_saved_real_matlab_reference_compares_without_matlab(self):
        from main.reference_tools.workflow_compare import compare_workflow_reference

        strict = compare_workflow_reference(self.imported)
        report = compare_workflow_reference(self.imported, enable_real_log_extension=True)

        self.assertEqual(len(self.imported.cases_by_id), 5)
        self.assertGreater(report.comparison_count, 3000)
        self.assertGreater(report.failure_count, 0)
        self.assertEqual(report.failure_count, report.extension_eligible_count)
        self.assertEqual(report.unexpected_mismatch_count, 0)
        self.assertEqual(report.unavailable_stage_count, 1)
        self.assertTrue(report.policy_qualified_success)
        self.assertFalse(report.strict_success)
        unavailable = next(row for row in report.rows if row.assessment == "unavailable")
        self.assertEqual(unavailable.reference_value["stage"], "project_image_areas_histograms_and_na_tables")
        self.assertEqual(unavailable.python_value["stage"], "autosmart_cutoffs")
        self.assertEqual(unavailable.python_value["code"], "empty_group_support")
        self.assertEqual(strict.failure_count, 62)
        self.assertEqual(strict.extension_eligible_count, 62)
        self.assertEqual(strict.documented_extension_count, 0)
        self.assertEqual(strict.unexpected_mismatch_count, 0)
        self.assertFalse(strict.strict_success)

    def test_perturbed_finite_reference_cell_fails(self):
        from main.reference_tools.workflow_compare import compare_workflow_reference

        imported = self._mutated_capture()
        case = imported.cases_by_id["single_manual_unnormalized_initial"]
        table = case["tables"]["Calc_out"]["values"]
        row_count = table["shape"][0]
        cell = table["cells"][row_count * 2]
        cell["real"] = str(float(cell["real"]) + 0.125)
        report = compare_workflow_reference(imported, enable_real_log_extension=True)

        self.assertGreater(report.unexpected_mismatch_count, 0)

    def test_perturbed_threshold_settings_fail_before_shape_identity_comparison(self):
        from main.reference_tools.workflow_compare import compare_workflow_reference

        imported = self._mutated_capture()
        case = imported.cases_by_id["shape_threshold_equality_initial_strict"]
        case["threshold_settings"]["statistics_minimum_diameter_px"] += 0.125

        with self.assertRaises(WorkflowReferenceValidationError) as caught:
            compare_workflow_reference(imported)

        self.assertEqual(caught.exception.code, "candidate_threshold_settings_mismatch")

    def test_perturbed_source_identity_fails(self):
        from main.reference_tools.workflow_compare import compare_workflow_reference

        imported = self._mutated_capture()
        case = imported.cases_by_id["single_manual_unnormalized_initial"]
        case["raw_data_identities"][0]["component_id"] += 100
        report = compare_workflow_reference(imported, enable_real_log_extension=True)

        self.assertTrue(any(row.code == "object_identity_or_order_mismatch" for row in report.rows))
        self.assertGreater(report.unexpected_mismatch_count, 0)

    def test_perturbed_log_operand_is_not_policy_qualified(self):
        from main.reference_tools.workflow_compare import compare_workflow_reference

        imported = self._mutated_capture()
        case = imported.cases_by_id["shape_threshold_equality_initial_strict"]
        inputs = case["volume"]["log_inputs"]["log10_cumulative_nv"]
        table = case["tables"]["Calc_out"]["values"]
        row_count, column_count = table["shape"]
        negative_infinity_rows = [
            row
            for row in range(row_count)
            if table["cells"][row_count * 5 + row]["classification"] == "negative_infinity"
        ]
        self.assertTrue(negative_infinity_rows)
        inputs["cells"][negative_infinity_rows[0]]["real"] = "1"
        report = compare_workflow_reference(imported, enable_real_log_extension=True)

        self.assertLess(report.extension_eligible_count, report.failure_count)
        self.assertGreater(report.unexpected_mismatch_count, 0)

    def test_corrupt_compressed_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupted = Path(directory) / self.package_path.name
            compressed = self.package_path.read_bytes()
            corrupted.write_bytes(compressed[:-1] + bytes((compressed[-1] ^ 1,)))
            manifest_source = self.package_path.with_suffix(self.package_path.suffix + ".manifest.json")
            corrupted.with_suffix(corrupted.suffix + ".manifest.json").write_bytes(manifest_source.read_bytes())
            with self.assertRaises(WorkflowReferenceValidationError) as caught:
                import_workflow_reference_package(corrupted, repository_root=Path(__file__).resolve().parents[2])

        self.assertEqual(caught.exception.code, "reference_package_checksum_mismatch")

    def test_package_decompression_and_recompression_are_deterministic(self):
        compressed = self.package_path.read_bytes()
        manifest_path = self.package_path.with_suffix(self.package_path.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = gzip.decompress(compressed)

        self.assertEqual(hashlib.sha256(compressed).hexdigest(), manifest["compressed_sha256"])
        self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest["payload_sha256"])
        self.assertEqual(gzip.compress(payload, compresslevel=9, mtime=0), compressed)


if __name__ == "__main__":
    unittest.main()
