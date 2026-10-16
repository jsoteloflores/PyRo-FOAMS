"""Report MATLAB-verified FOAMS 1.0.5 perimeter profile replay on reference cases."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from main.core import (
    FOAMS105_PERIMETER_PROFILE_CURRENT,
    FOAMS105_PERIMETER_PROFILE_OLD,
    measure_foams105_perimeter,
    measure_foams105_phase,
)

SUPPLEMENTARY_CASES_PATH = (
    Path(__file__).parents[1]
    / "tests"
    / "fixtures"
    / "16b_perimeter_evidence"
    / "review_supplementary_cases.json"
)
SUPPLEMENTARY_RESULTS_PATH = SUPPLEMENTARY_CASES_PATH.with_name("review_supplementary.tsv")


def main():
    cases = json.loads(SUPPLEMENTARY_CASES_PATH.read_text(encoding="utf-8"))
    expected = {}
    with SUPPLEMENTARY_RESULTS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.rstrip("\n").split("\t")
            if tokens[0] == "P":
                expected[(int(tokens[1]), int(tokens[2]))] = (float(tokens[3]), float(tokens[4]))

    matched = 0
    profile_differences = 0
    for case in cases:
        case_id = int(case["id"])
        image = np.asarray(case["raw"], dtype=np.uint8)
        measurement = measure_foams105_phase(image, 1, 1.0)
        if not np.array_equal(np.asarray(measurement.label_image, dtype=np.int32), case["expected_labels"]):
            raise ValueError(f"phase-label mismatch for supplementary case {case_id}")

        current = measure_foams105_perimeter(
            f"supp-{case_id}", measurement, profile=FOAMS105_PERIMETER_PROFILE_CURRENT
        )
        old = measure_foams105_perimeter(
            f"supp-{case_id}", measurement, profile=FOAMS105_PERIMETER_PROFILE_OLD
        )
        old_by_id = {component.component_id: component for component in old.components}
        for component in current.components:
            expected_current, expected_old = expected[(case_id, component.component_id)]
            if not math.isclose(component.perimeter_px, expected_current, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"current profile mismatch for {case_id}:{component.component_id}")
            if not math.isclose(
                old_by_id[component.component_id].perimeter_px,
                expected_old,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"old profile mismatch for {case_id}:{component.component_id}")
            if not math.isclose(
                component.perimeter_px,
                old_by_id[component.component_id].perimeter_px,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                profile_differences += 1
            matched += 1

    print("FOAMS 1.0.5 perimeter profile replay")
    print("validated components:", matched)
    print("within-run profile differences:", profile_differences)
    print("note: profile disagreement is expected and does not imply implementation failure")


if __name__ == "__main__":
    main()
