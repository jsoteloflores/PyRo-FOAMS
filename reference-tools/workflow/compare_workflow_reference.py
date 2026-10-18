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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", help="Complete external MATLAB capture bundle.")
    parser.add_argument(
        "--output-directory",
        help="Directory for comparison-report.json and comparison-report.md; defaults to the capture bundle.",
    )
    args = parser.parse_args()

    imported = import_workflow_reference(args.capture_directory, repository_root=REPOSITORY_ROOT)
    report = compare_workflow_reference(imported)
    output_directory = Path(args.output_directory or args.capture_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "comparison-report.json"
    markdown_path = output_directory / "comparison-report.md"
    json_path.write_text(workflow_comparison_json(report), encoding="utf-8", newline="\n")
    markdown_path.write_text(workflow_comparison_markdown(report), encoding="utf-8", newline="\n")

    print("Comparison report:", markdown_path)
    print("Comparisons:", report.comparison_count)
    print("Mismatches:", report.failure_count)
    print("First divergent stage:", report.first_divergent_stage or "none")
    print("Affected tables:", ", ".join(report.affected_tables) or "none")
    print("Unavailable stages:", report.unavailable_stage_count)
    return 0 if report.failure_count == 0 and report.unavailable_stage_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
