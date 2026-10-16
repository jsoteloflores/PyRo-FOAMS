from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from main.reference_tools.perimeter_reference import prepare_catalog, prepare_matlab_constructor
from main.tests.test_perimeter_reference_tools import (
    FIXTURE_PATH,
    PREP_SCRIPT_PATH,
    RUNNER_PATH,
    _build_valid_transport_run,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "reference-tools" / "perimeter" / "compare_perimeter_reference.py"
PREPARE_CLI_PATH = REPO_ROOT / "reference-tools" / "perimeter" / "prepare_perimeter_reference.py"
IMPORT_CLI_PATH = REPO_ROOT / "reference-tools" / "perimeter" / "import_perimeter_reference.py"


class TestPerimeterReferenceCli(unittest.TestCase):
    def _run_cli(self, *args: str):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        return subprocess.run(
            [sys.executable, str(CLI_PATH), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )

    def _run_script(self, script_path: Path, *args: str):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        return subprocess.run(
            [sys.executable, str(script_path), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )

    def test_direct_wrappers_start_without_pythonpath(self):
        for script_path in (PREPARE_CLI_PATH, IMPORT_CLI_PATH, CLI_PATH):
            with self.subTest(script=script_path.name):
                result = self._run_script(script_path, "--help")
                self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
                self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_within_run_agreement_exit_zero_and_json_rows(self):
        prepared = prepare_catalog(FIXTURE_PATH)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            constructor = root / "constructor.m"
            manifest = root / "manifest.json"
            hashes = prepare_matlab_constructor(
                FIXTURE_PATH,
                constructor,
                manifest,
                runner_path=RUNNER_PATH,
                preparation_path=PREP_SCRIPT_PATH,
            )

            run_dir = root / "run"
            _build_valid_transport_run(
                run_dir,
                prepared,
                expected_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
                per_component_override=lambda case_id, native_id: {
                    "perimeter_px": "0",
                    "perimeter_old_px": "0",
                }
                if case_id == "isolated_pixel_final" and native_id == 1
                else None,
            )

            out_json = root / "report.json"
            result = self._run_cli(
                str(run_dir),
                "--mode",
                "within-run",
                "--catalog",
                str(FIXTURE_PATH),
                "--manifest",
                str(manifest),
                "--output-json",
                str(out_json),
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertTrue(out_json.is_file())
            self.assertNotIn("ModuleNotFoundError", result.stderr)

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "within-run")
            self.assertEqual(payload["counts"]["mismatch_rows"], 0)
            self.assertEqual(payload["counts"]["skipped_rows"], 0)
            self.assertTrue(payload["states"]["numerical_valid"])
            zero_rows = [
                row
                for row in payload["rows"]
                if row["case_id"] == "isolated_pixel_final" and row["native_component_id"] == 1
            ]
            self.assertEqual(len(zero_rows), 1)
            self.assertEqual(zero_rows[0]["relative_difference"], 0.0)
            self.assertIsNone(zero_rows[0]["relative_difference_reason"])
            self.assertEqual(zero_rows[0]["reference_mm"], 0.0)
            self.assertEqual(zero_rows[0]["candidate_mm"], 0.0)

    def test_within_run_999_mismatch_exits_one(self):
        prepared = prepare_catalog(FIXTURE_PATH)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            constructor = root / "constructor.m"
            manifest = root / "manifest.json"
            hashes = prepare_matlab_constructor(
                FIXTURE_PATH,
                constructor,
                manifest,
                runner_path=RUNNER_PATH,
                preparation_path=PREP_SCRIPT_PATH,
            )

            run_dir = root / "run"
            _build_valid_transport_run(
                run_dir,
                prepared,
                expected_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
                per_component_override=lambda case_id, native_id: {
                    "perimeter_px": "999",
                }
                if case_id == "isolated_pixel_final" and native_id == 1
                else None,
            )
            out_json = root / "report.json"
            result = self._run_cli(
                str(run_dir),
                "--mode",
                "within-run",
                "--catalog",
                str(FIXTURE_PATH),
                "--manifest",
                str(manifest),
                "--output-json",
                str(out_json),
            )
            self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)
            self.assertTrue(out_json.is_file())
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertGreaterEqual(payload["counts"]["mismatch_rows"], 1)
            self.assertFalse(payload["states"]["numerical_valid"])
            mismatch_row = next(
                row
                for row in payload["rows"]
                if row["case_id"] == "isolated_pixel_final" and row["native_component_id"] == 1
            )
            self.assertEqual(mismatch_row["candidate_px"], 999.0)
            self.assertFalse(mismatch_row["within_tolerance"])

    def test_within_run_no_comparable_or_partial_rows_exit_two(self):
        prepared = prepare_catalog(FIXTURE_PATH)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            constructor = root / "constructor.m"
            manifest = root / "manifest.json"
            hashes = prepare_matlab_constructor(
                FIXTURE_PATH,
                constructor,
                manifest,
                runner_path=RUNNER_PATH,
                preparation_path=PREP_SCRIPT_PATH,
            )

            no_comp_dir = root / "run_no_comp"
            _build_valid_transport_run(
                no_comp_dir,
                prepared,
                expected_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
                per_component_override=lambda _case_id, _native_id: {
                    "perimeter_old_status": "unsupported",
                    "perimeter_old_px": "",
                    "perimeter_old_error": "MATLAB:images:regionprops:invalidProperty",
                },
            )
            out_json = root / "no_comp.json"
            result = self._run_cli(
                str(no_comp_dir),
                "--mode",
                "within-run",
                "--catalog",
                str(FIXTURE_PATH),
                "--manifest",
                str(manifest),
                "--output-json",
                str(out_json),
            )
            self.assertEqual(result.returncode, 2, msg=result.stdout + result.stderr)
            self.assertTrue(out_json.is_file())
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["counts"]["comparable_rows"], 0)

            partial_dir = root / "run_partial"
            _build_valid_transport_run(
                partial_dir,
                prepared,
                expected_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
                per_component_override=lambda case_id, native_id: {
                    "perimeter_status": "error_nonfinite",
                    "perimeter_px": "",
                    "perimeter_error": "Perimeter_nonfinite",
                }
                if case_id == "isolated_pixel_final" and native_id == 1
                else None,
            )
            out_json = root / "partial.json"
            result = self._run_cli(
                str(partial_dir),
                "--mode",
                "within-run",
                "--catalog",
                str(FIXTURE_PATH),
                "--manifest",
                str(manifest),
                "--output-json",
                str(out_json),
            )
            self.assertEqual(result.returncode, 2, msg=result.stdout + result.stderr)
            self.assertTrue(out_json.is_file())
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertGreater(payload["counts"]["skipped_rows"], 0)
            self.assertGreater(len(payload["skipped_reasons"]), 0)

    def test_invalid_input_exits_two_and_reports_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out_json = root / "invalid.json"
            result = self._run_cli(
                str(root / "missing-run"),
                "--mode",
                "within-run",
                "--catalog",
                str(FIXTURE_PATH),
                "--manifest",
                str(root / "missing-manifest.json"),
                "--output-json",
                str(out_json),
            )
            self.assertEqual(result.returncode, 2)
            self.assertTrue(out_json.is_file())
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertIn("validation_error", " ".join(payload["notes"]))
            self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_run_to_run_mode_uses_run_to_run_exit_status(self):
        prepared = prepare_catalog(FIXTURE_PATH)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            constructor = root / "constructor.m"
            manifest = root / "manifest.json"
            hashes = prepare_matlab_constructor(
                FIXTURE_PATH,
                constructor,
                manifest,
                runner_path=RUNNER_PATH,
                preparation_path=PREP_SCRIPT_PATH,
            )

            run_a_dir = root / "run_a"
            run_b_dir = root / "run_b"
            mismatch_override = lambda case_id, native_id: {
                "perimeter_px": "999",
            } if case_id == "isolated_pixel_final" and native_id == 1 else None
            _build_valid_transport_run(
                run_a_dir,
                prepared,
                expected_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
                per_component_override=mismatch_override,
            )
            _build_valid_transport_run(
                run_b_dir,
                prepared,
                expected_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
                per_component_override=mismatch_override,
            )

            out_json = root / "run_to_run_equal.json"
            result = self._run_cli(
                str(run_a_dir),
                "--mode",
                "run-to-run",
                "--other-run-dir",
                str(run_b_dir),
                "--catalog",
                str(FIXTURE_PATH),
                "--manifest",
                str(manifest),
                "--output-json",
                str(out_json),
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertTrue(out_json.is_file())
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "run-to-run")
            self.assertEqual(payload["counts"]["mismatch_rows"], 0)

            run_c_dir = root / "run_c"
            _build_valid_transport_run(
                run_c_dir,
                prepared,
                expected_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
                per_component_override=lambda case_id, native_id: {
                    "perimeter_px": "999",
                }
                if case_id == "opposing_order_distinguishable_final" and native_id == 1
                else None,
            )
            out_json = root / "run_to_run_mismatch.json"
            result = self._run_cli(
                str(run_a_dir),
                "--mode",
                "run-to-run",
                "--other-run-dir",
                str(run_c_dir),
                "--catalog",
                str(FIXTURE_PATH),
                "--manifest",
                str(manifest),
                "--output-json",
                str(out_json),
            )
            self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)
            self.assertTrue(out_json.is_file())
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertGreater(payload["counts"]["mismatch_rows"], 0)

    def test_documented_prepare_import_compare_sequence_without_pythonpath(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            constructor = root / "generated.m"
            manifest = root / "manifest.json"
            prepare = self._run_script(
                PREPARE_CLI_PATH,
                "--catalog",
                str(FIXTURE_PATH),
                "--runner",
                str(RUNNER_PATH),
                "--constructor",
                str(constructor),
                "--manifest",
                str(manifest),
            )
            self.assertEqual(prepare.returncode, 0, msg=prepare.stdout + prepare.stderr)
            self.assertTrue(constructor.is_file())
            self.assertTrue(manifest.is_file())
            self.assertNotIn("ModuleNotFoundError", prepare.stderr)

            prepared = prepare_catalog(FIXTURE_PATH)
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            run_dir = root / "synthetic-run"
            _build_valid_transport_run(
                run_dir,
                prepared,
                expected_catalog_hash=manifest_payload["case_catalog_hash"],
                expected_preparation_hash=manifest_payload["preparation_hash"],
                expected_runner_hash=manifest_payload["runner_hash"],
            )

            imported = self._run_script(
                IMPORT_CLI_PATH,
                str(run_dir),
                "--catalog",
                str(FIXTURE_PATH),
                "--manifest",
                str(manifest),
            )
            self.assertEqual(imported.returncode, 0, msg=imported.stdout + imported.stderr)
            self.assertIn("case_count:", imported.stdout)
            self.assertNotIn("ModuleNotFoundError", imported.stderr)

            report_path = root / "report.json"
            compared = self._run_cli(
                str(run_dir),
                "--mode",
                "within-run",
                "--catalog",
                str(FIXTURE_PATH),
                "--manifest",
                str(manifest),
                "--output-json",
                str(report_path),
            )
            self.assertEqual(compared.returncode, 0, msg=compared.stdout + compared.stderr)
            self.assertTrue(report_path.is_file())
            self.assertNotIn("ModuleNotFoundError", compared.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertGreater(report["counts"]["expected_rows"], 0)
            self.assertEqual(report["counts"]["comparable_rows"], report["counts"]["expected_rows"])
            self.assertEqual(report["counts"]["mismatch_rows"], 0)
            self.assertTrue(report["states"]["schema_valid"])
            self.assertTrue(report["states"]["topology_valid"])
            self.assertTrue(report["states"]["numerical_valid"])


class TestPerimeterRunnerSourceCompatibility(unittest.TestCase):
    def test_perimeter_old_unsupported_classifier_avoids_contains(self):
        text = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("contains(", text)
        self.assertIn("strfind(", text)
        self.assertIn("has_token(identifier, 'unknown')", text)
        self.assertIn("has_token(messageText, 'property')", text)


if __name__ == "__main__":
    unittest.main()
