"""Create perimeter comparison reports from imported MATLAB TSV artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from main.reference_tools.perimeter_reference import (
    ComparisonReport,
    ComparisonRow,
    ImportedReferenceRun,
    PerimeterReferenceValidationError,
    compare_perimeter_and_old,
    compare_runs,
    import_reference_run,
)


def _load_run(run_dir: Path, catalog: Path, manifest: Path):
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return import_reference_run(
        run_dir,
        catalog,
        expected_case_catalog_hash=payload["case_catalog_hash"],
        expected_preparation_hash=payload["preparation_hash"],
        expected_runner_hash=payload["runner_hash"],
    )



def _relative_reason(row: ComparisonRow) -> Optional[str]:
    if row.relative_difference is not None:
        return None
    if row.reference_value == 0.0 and row.candidate_value != 0.0:
        return "reference_zero_candidate_nonzero"
    return "undefined"


def _component_status_details_within_run(run: ImportedReferenceRun) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for component in run.components:
        if component.perimeter_status == "success" and component.perimeter_old_status == "success":
            continue
        reason = (
            f"case={component.case_id} id={component.native_component_id} "
            f"perimeter_status={component.perimeter_status} perimeter_old_status={component.perimeter_old_status}"
        )
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _component_status_details_run_to_run(run_a: ImportedReferenceRun, run_b: ImportedReferenceRun) -> Dict[str, int]:
    index_a = {(item.case_id, item.native_component_id): item for item in run_a.components}
    index_b = {(item.case_id, item.native_component_id): item for item in run_b.components}
    counts: Dict[str, int] = {}
    for identity in sorted(index_a):
        left = index_a[identity]
        right = index_b[identity]
        if left.perimeter_status == "success" and right.perimeter_status == "success":
            continue
        reason = (
            f"case={left.case_id} id={left.native_component_id} "
            f"left_status={left.perimeter_status} right_status={right.perimeter_status}"
        )
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _build_rows_payload(
    report: ComparisonReport,
    run: ImportedReferenceRun,
    *,
    other_run: Optional[ImportedReferenceRun] = None,
) -> List[dict]:
    rows: List[dict] = []
    for row in sorted(report.rows, key=lambda item: (item.case_id, item.native_component_id)):
        scale = run.captured_cases[row.case_id].scale_px_per_mm
        candidate_scale = scale
        if other_run is not None:
            candidate_scale = other_run.captured_cases[row.case_id].scale_px_per_mm
        reference_mm = row.reference_value / scale
        candidate_mm = row.candidate_value / candidate_scale
        row_payload = {
            "case_id": row.case_id,
            "native_component_id": row.native_component_id,
            "reference_field": row.reference_field,
            "candidate_field": row.candidate_field,
            "reference_px": row.reference_value,
            "candidate_px": row.candidate_value,
            "reference_mm": reference_mm,
            "candidate_mm": candidate_mm,
            "scale_px_per_mm": scale,
            "absolute_difference_px": row.absolute_difference,
            "absolute_difference_mm": abs(candidate_mm - reference_mm),
            "relative_difference": row.relative_difference,
            "relative_difference_reason": _relative_reason(row),
            "within_tolerance": row.within_tolerance,
        }
        rows.append(row_payload)
    return rows


def _build_report_payload(
    *,
    mode: str,
    report: ComparisonReport,
    run: ImportedReferenceRun,
    other_run: Optional[ImportedReferenceRun],
) -> dict:
    if mode == "within-run":
        expected_rows = len(run.components)
        skipped_counts = _component_status_details_within_run(run)
    else:
        assert other_run is not None
        expected_rows = len(run.components)
        skipped_counts = _component_status_details_run_to_run(run, other_run)
    comparable_rows = report.compared_rows
    skipped_rows = expected_rows - comparable_rows
    if skipped_rows < 0:
        skipped_rows = 0

    return {
        "mode": mode,
        "states": {
            "schema_valid": report.schema_valid,
            "topology_valid": report.topology_valid,
            "numerical_valid": report.numerical_valid,
            "historical_compatible": report.historical_compatible,
        },
        "counts": {
            "expected_rows": expected_rows,
            "comparable_rows": comparable_rows,
            "skipped_rows": skipped_rows,
            "mismatch_rows": report.mismatch_rows,
        },
        "rows": _build_rows_payload(report, run, other_run=other_run),
        "skipped_reasons": [
            {"count": skipped_counts[key], "reason": key} for key in sorted(skipped_counts)
        ],
        "notes": list(report.notes),
    }


def _print_console_summary(payload: dict) -> None:
    counts = payload["counts"]
    states = payload["states"]
    print("mode:", payload["mode"])
    print("schema_valid:", states["schema_valid"])
    print("topology_valid:", states["topology_valid"])
    print("numerical_valid:", states["numerical_valid"])
    print("historical_compatible:", states["historical_compatible"])
    print("expected_rows:", counts["expected_rows"])
    print("comparable_rows:", counts["comparable_rows"])
    print("skipped_rows:", counts["skipped_rows"])
    print("mismatch_rows:", counts["mismatch_rows"])
    if payload["skipped_reasons"]:
        print("skipped_reasons:")
        for item in payload["skipped_reasons"]:
            print(f"  - [{item['count']}] {item['reason']}")
    if payload["notes"]:
        print("notes:")
        for note in payload["notes"]:
            print(f"  - {note}")


def _determine_exit_code(payload: dict) -> int:
    states = payload["states"]
    counts = payload["counts"]
    if not states["schema_valid"] or not states["topology_valid"]:
        return 2
    if counts["comparable_rows"] == 0:
        return 2
    if counts["skipped_rows"] > 0:
        return 2
    if not states["numerical_valid"] or counts["mismatch_rows"] > 0:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Primary run directory")
    parser.add_argument(
        "--mode",
        choices=("within-run", "run-to-run"),
        required=True,
        help="Comparison mode: within-run (Perimeter vs PerimeterOld) or run-to-run.",
    )
    parser.add_argument("--other-run-dir", help="Second run directory for run-to-run comparison")
    parser.add_argument(
        "--catalog",
        default="main/tests/fixtures/16a_foams105_perimeter_cases.json",
        help="Path to 16a perimeter case catalog JSON.",
    )
    parser.add_argument(
        "--manifest",
        default="reference-tools/perimeter/foams105_perimeter_manifest.json",
        help="Preparation manifest with expected hashes.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path for machine-readable deterministic report JSON.",
    )
    args = parser.parse_args()

    try:
        run = _load_run(Path(args.run_dir), Path(args.catalog), Path(args.manifest))

        if args.mode == "run-to-run":
            if not args.other_run_dir:
                print("error: --other-run-dir is required when --mode=run-to-run")
                return 2
            other = _load_run(Path(args.other_run_dir), Path(args.catalog), Path(args.manifest))
            report = compare_runs(run, other)
            payload = _build_report_payload(mode=args.mode, report=report, run=run, other_run=other)
        else:
            report = compare_perimeter_and_old(run)
            payload = _build_report_payload(mode=args.mode, report=report, run=run, other_run=None)
    except (PerimeterReferenceValidationError, FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as exc:
        code = "invalid_input"
        message = str(exc)
        if isinstance(exc, PerimeterReferenceValidationError):
            code = exc.code
            message = str(exc)
        payload = {
            "mode": args.mode,
            "states": {
                "schema_valid": False,
                "topology_valid": False,
                "numerical_valid": False,
                "historical_compatible": False,
            },
            "counts": {
                "expected_rows": 0,
                "comparable_rows": 0,
                "skipped_rows": 0,
                "mismatch_rows": 0,
            },
            "rows": [],
            "skipped_reasons": [],
            "notes": [f"validation_error[{code}]: {message}"],
        }
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        _print_console_summary(payload)
        return 2

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    _print_console_summary(payload)
    return _determine_exit_code(payload)


if __name__ == "__main__":
    raise SystemExit(main())
