"""Create a compact, checksummed package from selected MATLAB workflow cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> int:
    from main.reference_tools.workflow_package import (
        DEFAULT_REFERENCE_CASES,
        build_workflow_reference_package,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", help="Complete external MATLAB capture bundle.")
    parser.add_argument(
        "output_package",
        nargs="?",
        default="main/tests/fixtures/19a_foams105_reference.json.gz",
        help="New deterministic gzip package path.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=DEFAULT_REFERENCE_CASES,
        help="Complete capture case IDs to retain; no fields or rows are filtered.",
    )
    args = parser.parse_args()
    manifest = build_workflow_reference_package(
        args.capture_directory,
        args.output_package,
        case_ids=args.cases,
        repository_root=REPOSITORY_ROOT,
    )
    print("Reference package:", Path(args.output_package).resolve())
    print("Parent capture SHA-256:", manifest["parent_capture_sha256"])
    print("Compressed SHA-256:", manifest["compressed_sha256"])
    print("Payload SHA-256:", manifest["payload_sha256"])
    print("Cases:", ", ".join(manifest["case_ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
