from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from main.reference_tools.perimeter_reference import (
    PerimeterReferenceValidationError,
    compare_perimeter_and_old,
    compare_runs,
    import_reference_run,
    prepare_catalog,
    prepare_matlab_constructor,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "16a_foams105_perimeter_cases.json"
RUNNER_PATH = Path("reference-tools") / "matlab" / "foams105_perimeter_capture_tsv.m"
PREP_SCRIPT_PATH = Path("reference-tools") / "perimeter" / "prepare_perimeter_reference.py"


def _write_tsv(path: Path, headers, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _build_valid_transport_run(
    run_dir: Path,
    prepared,
    *,
    expected_catalog_hash: str,
    expected_preparation_hash: str,
    expected_runner_hash: str,
    per_component_override=None,
):
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": "1",
        "source_commit": "179663203f2d0f86b2863d5ea7f8f70dadca02f8",
        "case_catalog_hash": expected_catalog_hash,
        "preparation_hash": expected_preparation_hash,
        "runner_hash": expected_runner_hash,
        "runtime_identity": "synthetic-transport",
        "matlab_version": "R-synthetic",
        "matlab_release": "Rsynthetic",
        "image_processing_toolbox_version": "synthetic",
        "invocation": "synthetic",
        "timestamp_description": "synthetic-local-time",
    }
    with (run_dir / "metadata.tsv").open("w", encoding="utf-8") as handle:
        for key, value in metadata.items():
            handle.write(f"{key}\t{value}\n")

    case_rows = []
    label_rows = []
    component_rows = []
    for case in prepared.cases:
        case_rows.append(
            {
                "case_id": case.case_id,
                "stage": case.stage,
                "height": case.shape[0],
                "width": case.shape[1],
                "scale_px_per_mm": f"{case.scale_px_per_mm:.17g}",
                "foreground_pixel_count": case.foreground_pixel_count,
                "component_count": case.component_count,
                "outcome": "success",
            }
        )
        for row in range(case.shape[0]):
            for col in range(case.shape[1]):
                label_rows.append(
                    {
                        "case_id": case.case_id,
                        "row": row,
                        "column": col,
                        "label": int(case.labels[row, col]),
                    }
                )

        for component in sorted(case.topology, key=lambda item: item.native_component_id):
            row = {
                "case_id": case.case_id,
                "native_component_id": component.native_component_id,
                "canonical_component_id": component.canonical_component_id,
                "first_row": component.first_row,
                "first_column": component.first_column,
                "area_px": component.area_px,
                "perimeter_px": f"{float(component.area_px):.17g}",
                "perimeter_status": "success",
                "perimeter_old_px": f"{float(component.area_px):.17g}",
                "perimeter_old_status": "success",
                "property_error": "",
            }
            if per_component_override is not None:
                updates = per_component_override(case.case_id, component.native_component_id)
                if updates:
                    row.update(updates)
            component_rows.append(row)

    _write_tsv(
        run_dir / "cases.tsv",
        [
            "case_id",
            "stage",
            "height",
            "width",
            "scale_px_per_mm",
            "foreground_pixel_count",
            "component_count",
            "outcome",
        ],
        case_rows,
    )
    _write_tsv(run_dir / "labels.tsv", ["case_id", "row", "column", "label"], label_rows)
    _write_tsv(
        run_dir / "components.tsv",
        [
            "case_id",
            "native_component_id",
            "canonical_component_id",
            "first_row",
            "first_column",
            "area_px",
            "perimeter_px",
            "perimeter_status",
            "perimeter_old_px",
            "perimeter_old_status",
            "property_error",
        ],
        component_rows,
    )
    (run_dir / "completed.marker").write_text("COMPLETE\n", encoding="utf-8")


class TestPerimeterReferenceCatalog(unittest.TestCase):
    def test_prepare_catalog_validates_ordering_counterexample(self):
        catalog = prepare_catalog(FIXTURE_PATH)
        by_id = {case.case_id: case for case in catalog.cases}

        first = by_id["opposing_order_singletons_final"]
        first_rows = tuple(
            (item.first_row, item.first_column)
            for item in sorted(first.topology, key=lambda item: item.canonical_component_id)
        )
        self.assertEqual(first_rows, ((6, 2), (2, 6)))

        second = by_id["opposing_order_distinguishable_final"]
        second_areas = tuple(
            item.area_px for item in sorted(second.topology, key=lambda item: item.canonical_component_id)
        )
        self.assertEqual(second_areas, (2, 1))

    def test_prepare_constructor_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            constructor = root / "generated.m"
            manifest = root / "manifest.json"

            first = prepare_matlab_constructor(
                FIXTURE_PATH,
                constructor,
                manifest,
                runner_path=RUNNER_PATH,
                preparation_path=PREP_SCRIPT_PATH,
            )
            first_text = constructor.read_text(encoding="utf-8")
            second = prepare_matlab_constructor(
                FIXTURE_PATH,
                constructor,
                manifest,
                runner_path=RUNNER_PATH,
                preparation_path=PREP_SCRIPT_PATH,
            )
            second_text = constructor.read_text(encoding="utf-8")

            self.assertEqual(first_text, second_text)
            self.assertEqual(first.catalog_hash, second.catalog_hash)
            self.assertEqual(first.preparation_hash, second.preparation_hash)
            self.assertEqual(first.runner_hash, second.runner_hash)
            self.assertEqual(first.constructor_hash, second.constructor_hash)


class TestPerimeterReferenceImport(unittest.TestCase):
    def test_import_valid_transport_and_zero_perimeter(self):
        prepared = prepare_catalog(FIXTURE_PATH)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "run"
            constructor = root / "constructor.m"
            manifest = root / "manifest.json"
            hashes = prepare_matlab_constructor(
                FIXTURE_PATH,
                constructor,
                manifest,
                runner_path=RUNNER_PATH,
                preparation_path=PREP_SCRIPT_PATH,
            )

            def override(case_id, native_id):
                if case_id == "isolated_pixel_final" and native_id == 1:
                    return {
                        "perimeter_px": "0",
                        "perimeter_status": "success",
                        "perimeter_old_px": "0",
                        "perimeter_old_status": "success",
                    }
                if case_id == "empty_final":
                    return {"perimeter_old_status": "unsupported", "perimeter_old_px": ""}
                return None

            _build_valid_transport_run(
                run_dir,
                prepared,
                expected_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
                per_component_override=override,
            )

            imported = import_reference_run(
                run_dir,
                FIXTURE_PATH,
                expected_case_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
            )
            self.assertEqual(len(imported.cases), len(prepared.cases))

    def test_import_rejects_hash_mismatch_duplicate_component_and_bad_labels(self):
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

            run_hash = root / "run_hash"
            _build_valid_transport_run(
                run_hash,
                prepared,
                expected_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
            )
            with self.assertRaises(PerimeterReferenceValidationError) as caught:
                import_reference_run(
                    run_hash,
                    FIXTURE_PATH,
                    expected_case_catalog_hash="wrong",
                    expected_preparation_hash=hashes.preparation_hash,
                    expected_runner_hash=hashes.runner_hash,
                )
            self.assertEqual(caught.exception.code, "hash_mismatch")

            run_duplicate = root / "run_duplicate"
            _build_valid_transport_run(
                run_duplicate,
                prepared,
                expected_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
            )
            with (run_duplicate / "components.tsv").open("a", encoding="utf-8") as handle:
                handle.write("isolated_pixel_final\t1\t1\t3\t3\t1\t1\tsuccess\t1\tsuccess\t\n")
            with self.assertRaises(PerimeterReferenceValidationError) as caught:
                import_reference_run(
                    run_duplicate,
                    FIXTURE_PATH,
                    expected_case_catalog_hash=hashes.catalog_hash,
                    expected_preparation_hash=hashes.preparation_hash,
                    expected_runner_hash=hashes.runner_hash,
                )
            self.assertEqual(caught.exception.code, "invalid_export")

            run_bad_label = root / "run_bad_label"
            _build_valid_transport_run(
                run_bad_label,
                prepared,
                expected_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
            )
            lines = (run_bad_label / "labels.tsv").read_text(encoding="utf-8").splitlines()
            lines[-1] = lines[-1].rsplit("\t", 1)[0] + "\t99"
            (run_bad_label / "labels.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(PerimeterReferenceValidationError) as caught:
                import_reference_run(
                    run_bad_label,
                    FIXTURE_PATH,
                    expected_case_catalog_hash=hashes.catalog_hash,
                    expected_preparation_hash=hashes.preparation_hash,
                    expected_runner_hash=hashes.runner_hash,
                )
            self.assertEqual(caught.exception.code, "label_mismatch")


class TestPerimeterReferenceComparison(unittest.TestCase):
    def test_comparison_reports_detect_mismatch_and_zero_reference_rule(self):
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
            _build_valid_transport_run(
                run_a_dir,
                prepared,
                expected_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
                per_component_override=lambda case_id, native_id: {
                    "perimeter_px": "0",
                    "perimeter_old_px": "0",
                    "perimeter_status": "success",
                    "perimeter_old_status": "success",
                }
                if case_id == "isolated_pixel_final" and native_id == 1
                else None,
            )
            run_a = import_reference_run(
                run_a_dir,
                FIXTURE_PATH,
                expected_case_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
            )

            old_report = compare_perimeter_and_old(run_a)
            self.assertGreater(old_report.compared_rows, 0)
            self.assertEqual(old_report.mismatch_rows, 0)
            zero_rows = [row for row in old_report.rows if row.reference_value == 0.0]
            self.assertTrue(any(row.relative_difference == 0.0 for row in zero_rows))

            run_b_dir = root / "run_b"
            _build_valid_transport_run(
                run_b_dir,
                prepared,
                expected_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
                per_component_override=lambda case_id, native_id: {
                    "perimeter_px": "999"
                }
                if case_id == "opposing_order_distinguishable_final" and native_id == 1
                else None,
            )
            run_b = import_reference_run(
                run_b_dir,
                FIXTURE_PATH,
                expected_case_catalog_hash=hashes.catalog_hash,
                expected_preparation_hash=hashes.preparation_hash,
                expected_runner_hash=hashes.runner_hash,
            )

            mismatch_report = compare_runs(run_a, run_b)
            self.assertGreater(mismatch_report.compared_rows, 0)
            self.assertGreater(mismatch_report.mismatch_rows, 0)
            self.assertFalse(mismatch_report.numerical_valid)


if __name__ == "__main__":
    unittest.main()
