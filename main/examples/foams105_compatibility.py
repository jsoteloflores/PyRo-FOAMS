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


def workbook_report():
    with REFERENCE_PATH.open(encoding="utf-8") as stream:
        fixture = json.load(stream)
    result = convert_foams105_nv(
        fixture["bin_labels_mm"], fixture["na_per_mm2"]
    )
    errors = tuple(
        abs(candidate - expected)
        for candidate, expected in zip(
            result.signed_nv_per_mm3, fixture["expected_nv_per_mm3"]
        )
    )
    relative_errors = tuple(
        error / abs(expected) if expected else 0.0
        for error, expected in zip(errors, fixture["expected_nv_per_mm3"])
    )
    worst_absolute_row = max(range(len(errors)), key=errors.__getitem__)
    worst_relative_row = max(
        range(len(relative_errors)), key=relative_errors.__getitem__
    )
    assert all(
        error <= 1e-12 * abs(expected)
        for error, expected in zip(errors, fixture["expected_nv_per_mm3"])
    )

    print("FOAMS 1.0.5 bundled Calc_out fixture report")
    print("method:", result.method)
    print("source commit:", result.source_commit)
    print("compared rows:", len(errors))
    print("maximum absolute error mm^-3:", errors[worst_absolute_row])
    print("worst absolute row (zero-based):", worst_absolute_row)
    print("maximum relative error:", relative_errors[worst_relative_row])
    print("worst relative row (zero-based):", worst_relative_row)
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
    assert all(math.isfinite(value) for value in legacy.signed_nv_per_mm3)
    assert all(math.isfinite(value) for value in modern.signed_nv_per_mm3)

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
