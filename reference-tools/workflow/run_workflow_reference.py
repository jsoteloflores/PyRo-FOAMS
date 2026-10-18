"""Capture the pinned FOAMS 1.0.5 workflow through MATLAB R2025b."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

def main() -> int:
    from main.reference_tools.workflow_capture import run_foams105_workflow_capture

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_checkout", help="External detached FOAMS-1.0.5 checkout at the pinned source commit.")
    parser.add_argument("output_directory", help="Fresh capture output directory; existing paths are refused.")
    parser.add_argument(
        "--catalog",
        default="main/tests/fixtures/19_foams105_workflow_cases.json",
        help="Versioned input/settings catalog JSON.",
    )
    parser.add_argument(
        "--matlab",
        default=r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe",
        help="MATLAB executable used to execute the pinned original-source replay.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    result = run_foams105_workflow_capture(
        args.source_checkout,
        args.catalog,
        args.output_directory,
        matlab_executable=args.matlab,
        repository_root=REPOSITORY_ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    print("Workflow capture:", result.output_directory)
    print("Process exit code:", result.process_exit_code)
    print("Catalog cases:", result.case_count)
    print("Complete:", result.complete)
    print("Capture SHA-256:", result.capture_sha256 or "unavailable")
    return 0 if result.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
