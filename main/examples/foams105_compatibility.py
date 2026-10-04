"""Compare the FOAMS 1.0.5 conversion replay with the modern sphere model.

The workbook comparison covers only the original post-nesting N_A-to-N_V
stage. Its values were extracted from Calc_out.xls; MATLAB was not executed by
this example. The side-by-side three-class calculation explicitly maps legacy
labels to separate modern interval edges and is not full workflow equivalence.
"""

import json
import math
from pathlib import Path

from main.core import (
    FOAMS105_METHOD,
    FOAMS105_SOURCE_COMMIT,
    UPPER_TAIL_ASSUMPTION,
    build_spherical_section_operator,
    convert_foams105_nv,
    create_diameter_bin_spec,
    solve_spherical_number_densities,
)

REFERENCE_PATH = (
    Path(__file__).parents[1]
    / "tests"
    / "fixtures"
    / "06_foams105_nv_reference.json"
)
REFERENCE_ROW_COUNT = 29
REFERENCE_RELATIVE_TOLERANCE = 1e-12


def _reference_errors(candidate_values, expected_values):
    candidates = tuple(candidate_values)
    expected = tuple(expected_values)
    if len(candidates) != len(expected):
        raise ValueError(
            f"Candidate length {len(candidates)} does not match reference length "
            f"{len(expected)}"
        )
    errors = []
    relative_errors = []
    for index, (candidate, reference) in enumerate(zip(candidates, expected)):
        if not math.isfinite(candidate) or not math.isfinite(reference):
            raise ValueError(f"Reference comparison row {index} must be finite")
        error = abs(candidate - reference)
        if reference == 0.0:
            relative_error = 0.0 if candidate == 0.0 else math.inf
            accepted = candidate == 0.0
        else:
            relative_error = error / abs(reference)
            accepted = error <= REFERENCE_RELATIVE_TOLERANCE * abs(reference)
        if not accepted:
            raise ValueError(
                f"Reference comparison row {index} exceeds tolerance: "
                f"candidate={candidate!r}, expected={reference!r}"
            )
        errors.append(error)
        relative_errors.append(relative_error)
    return tuple(errors), tuple(relative_errors)


def workbook_report():
    with REFERENCE_PATH.open(encoding="utf-8") as stream:
        fixture = json.load(stream)
    result = convert_foams105_nv(
        fixture["bin_labels_mm"], fixture["na_per_mm2"]
    )
    lengths = {
        "labels": len(fixture["bin_labels_mm"]),
        "densities": len(fixture["na_per_mm2"]),
        "expected": len(fixture["expected_nv_per_mm3"]),
        "candidate": len(result.signed_nv_per_mm3),
    }
    if any(length != REFERENCE_ROW_COUNT for length in lengths.values()):
        raise ValueError(
            f"Pinned fixture requires {REFERENCE_ROW_COUNT} complete rows; got {lengths}"
        )
    errors, relative_errors = _reference_errors(
        result.signed_nv_per_mm3, fixture["expected_nv_per_mm3"]
    )
    worst_absolute_row = max(range(len(errors)), key=errors.__getitem__)
    worst_relative_row = max(
        range(len(relative_errors)), key=relative_errors.__getitem__
    )
    print("FOAMS 1.0.5 bundled Calc_out fixture report")
    print("method:", result.method)
    print("source commit:", result.source_commit)
    print("compared rows:", len(errors))
    print("maximum absolute error mm^-3:", errors[worst_absolute_row])
    print("worst absolute row (zero-based):", worst_absolute_row)
    print("maximum relative error:", relative_errors[worst_relative_row])
    print("worst relative row (zero-based):", worst_relative_row)
    print(
        "adjacent duplicate label index pairs:",
        result.adjacent_duplicate_label_index_pairs,
    )
    print("boundary: workbook cells were extracted; MATLAB was not run here")


def side_by_side_example():
    labels = (1.0, 2.0, 4.0)
    measured_na = (8.0, 3.0, 1.0)
    legacy = convert_foams105_nv(labels, measured_na)

    modern_edges = (0.5, 1.0, 2.0, 4.0)
    modern = solve_spherical_number_densities(
        build_spherical_section_operator(
            create_diameter_bin_spec(modern_edges)
        ),
        measured_na,
        upper_tail_assumption=UPPER_TAIL_ASSUMPTION,
    )
    if not all(math.isfinite(value) for value in legacy.signed_nv_per_mm3):
        raise RuntimeError("Legacy comparison produced a nonfinite result")
    if not all(math.isfinite(value) for value in modern.signed_nv_per_mm3):
        raise RuntimeError("Modern comparison produced a nonfinite result")

    print("\nExplicit numerical comparison, not workflow equivalence")
    print("input N_A mm^-2:", measured_na)
    print("legacy labels mm:", labels)
    print("legacy representative model: cube-volume midpoint heights")
    print("legacy method:", FOAMS105_METHOD)
    print("legacy signed N_V mm^-3:", legacy.signed_nv_per_mm3)
    print("modern interval edges mm:", modern_edges)
    print("modern representative model: interval upper-edge spheres")
    print("modern method:", modern.method)
    print("modern signed N_V mm^-3:", modern.signed_nv_per_mm3)
    print("source version:", FOAMS105_SOURCE_COMMIT)
    print("differences reflect distinct models and do not alone indicate a bug")


def main():
    workbook_report()
    side_by_side_example()


if __name__ == "__main__":
    main()
