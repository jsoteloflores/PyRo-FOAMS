"""Compare a completed FOAMS 1.0.5 MATLAB workflow capture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

def main() -> int:
    from main.reference_tools.workflow_compare import (
        compare_workflow_reference,
        import_workflow_reference,
        workflow_comparison_json,
        workflow_comparison_markdown,
    )
    from main.reference_tools.workflow_package import import_workflow_reference_package

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_path", help="Complete MATLAB capture bundle or saved compressed reference package.")
    parser.add_argument(
        "--output-directory",
        help="Directory for comparison-report.json and comparison-report.md; defaults to the capture bundle.",
    )
    outcome = parser.add_mutually_exclusive_group()
    outcome.add_argument(
        "--policy-qualified",
        action="store_true",
        help="Enable the named real-log representation policy; raw differences remain visible in the report.",
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help="Import a saved .json.gz reference package without requiring MATLAB or the original source checkout.",
    )
    outcome.add_argument(
        "--strict",
        action="store_true",
        help="Fail on every raw difference or unavailable stage (the default).",
    )
    args = parser.parse_args()

    imported = (
        import_workflow_reference_package(args.reference_path, repository_root=REPOSITORY_ROOT)
        if args.package
        else import_workflow_reference(args.reference_path, repository_root=REPOSITORY_ROOT)
    )
    report = compare_workflow_reference(imported, enable_real_log_extension=args.policy_qualified)
    output_directory = Path(args.output_directory or (Path(args.reference_path).parent if args.package else args.reference_path)).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "comparison-report.json"
    markdown_path = output_directory / "comparison-report.md"
    json_path.write_text(workflow_comparison_json(report), encoding="utf-8", newline="\n")
    markdown_path.write_text(workflow_comparison_markdown(report), encoding="utf-8", newline="\n")

    print("Comparison report:", markdown_path)
    print("Comparisons:", report.comparison_count)
    print("Exact/toleranced matches:", report.exact_match_count)
    print("Raw differences:", report.failure_count)
    print("Extension-eligible representation differences:", report.extension_eligible_count)
    print("Policy-qualified extension differences:", report.documented_extension_count)
    print("Unexpected mismatches:", report.unexpected_mismatch_count)
    print("First divergent stage:", report.first_divergent_stage or "none")
    print("Affected tables:", ", ".join(report.affected_tables) or "none")
    print("Unavailable stages:", report.unavailable_stage_count)
    print("Candidate Git revision:", report.python_revision)
    print("Catalog label:", report.catalog_label)
    print("Dirty candidate files with hashes:", len(report.candidate_dirty_files))
    print("Policy version:", report.policy_version or "disabled; strict raw comparison")
    if args.policy_qualified:
        print(
            "Policy-qualified supported-comparison status:",
            "PASS" if report.policy_qualified_success else "FAIL",
        )
        print(
            "Overall coverage status:",
            "INCOMPLETE: unavailable source/Python stage present" if report.unavailable_stage_count else "COMPLETE",
        )
        print("Literal raw parity:", "PASS" if report.strict_success else "NOT ESTABLISHED")
        return 0 if report.policy_qualified_success else 1
    print("Strict status:", "PASS" if report.strict_success else "FAIL")
    return 0 if report.strict_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
