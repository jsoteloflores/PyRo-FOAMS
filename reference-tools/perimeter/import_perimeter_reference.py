"""Import and validate MATLAB perimeter TSV export artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from main.reference_tools.perimeter_reference import import_reference_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Directory containing metadata.tsv/cases.tsv/labels.tsv/components.tsv")
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

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    run = import_reference_run(
        Path(args.run_dir),
        Path(args.catalog),
        expected_case_catalog_hash=manifest["case_catalog_hash"],
        expected_preparation_hash=manifest["preparation_hash"],
        expected_runner_hash=manifest["runner_hash"],
    )

    print("Imported perimeter reference run")
    print("run_dir:", run.run_dir)
    print("case_count:", len(run.cases))
    print("component_rows:", len(run.components))
    print("matlab_release:", run.metadata["matlab_release"])
    print("matlab_version:", run.metadata["matlab_version"])
    print("ipt_version:", run.metadata["image_processing_toolbox_version"])


if __name__ == "__main__":
    main()
