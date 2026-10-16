"""Generate deterministic MATLAB perimeter case constructor and manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from main.reference_tools.perimeter_reference import prepare_matlab_constructor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default="main/tests/fixtures/16a_foams105_perimeter_cases.json",
        help="Path to 16a perimeter case catalog JSON.",
    )
    parser.add_argument(
        "--runner",
        default="reference-tools/matlab/foams105_perimeter_capture_tsv.m",
        help="Path to MATLAB TSV capture runner.",
    )
    parser.add_argument(
        "--constructor",
        default="reference-tools/matlab/foams105_perimeter_cases_generated.m",
        help="Output MATLAB constructor .m path.",
    )
    parser.add_argument(
        "--manifest",
        default="reference-tools/perimeter/foams105_perimeter_manifest.json",
        help="Output manifest JSON path.",
    )
    args = parser.parse_args()

    result = prepare_matlab_constructor(
        Path(args.catalog),
        Path(args.constructor),
        Path(args.manifest),
        runner_path=Path(args.runner),
        preparation_path=Path(__file__),
    )
    print("Prepared perimeter reference constructor")
    print("constructor:", result.constructor_path)
    print("manifest:", result.manifest_path)
    print("catalog hash:", result.catalog_hash)
    print("preparation hash:", result.preparation_hash)
    print("runner hash:", result.runner_hash)
    print("constructor hash:", result.constructor_hash)
    print("case count:", result.case_count)


if __name__ == "__main__":
    main()
