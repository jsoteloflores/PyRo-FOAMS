"""Create perimeter comparison reports from imported MATLAB TSV artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.reference_tools.perimeter_reference import (
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Primary run directory")
    parser.add_argument("--other-run-dir", help="Optional second run for run-to-run comparison")
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
    args = parser.parse_args()

    run = _load_run(Path(args.run_dir), Path(args.catalog), Path(args.manifest))
    old_report = compare_perimeter_and_old(run)
    print("Perimeter vs PerimeterOld")
    print("compared_rows:", old_report.compared_rows)
    print("mismatch_rows:", old_report.mismatch_rows)
    print("numerical_valid:", old_report.numerical_valid)

    if args.other_run_dir:
        other = _load_run(Path(args.other_run_dir), Path(args.catalog), Path(args.manifest))
        run_report = compare_runs(run, other)
        print("Run-to-run perimeter comparison")
        print("compared_rows:", run_report.compared_rows)
        print("mismatch_rows:", run_report.mismatch_rows)
        print("numerical_valid:", run_report.numerical_valid)


if __name__ == "__main__":
    main()
