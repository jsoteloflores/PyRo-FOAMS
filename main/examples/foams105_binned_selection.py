"""Report checked FOAMS 1.0.5 binned selection reconciliation."""

import json
import math
from pathlib import Path

from main.core import (
    Foams105BinnedGroupInput,
    Foams105BinnedRange,
    Foams105SelectionDomainError,
    convert_foams105_nv,
    select_foams105_binned_ranges,
)

FIXTURE_DIR = Path(__file__).parents[1] / "tests" / "fixtures"
SELECTION_CASES_PATH = FIXTURE_DIR / "11_foams105_selection_cases.json"
HISTOGRAM_REFERENCE_PATH = FIXTURE_DIR / "07_foams105_histogram_reference.json"
CONVERSION_REFERENCE_PATH = FIXTURE_DIR / "06_foams105_nv_reference.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def analytical_report():
    fixture = _load(SELECTION_CASES_PATH)
    compared = 0
    expected_errors = 0
    for case in fixture["cases"]:
        groups = tuple(
            Foams105BinnedGroupInput(
                f"slot-{slot}", slot, tuple(densities)
            )
            for slot, densities in enumerate(case["densities"], start=1)
        )
        ranges = tuple(
            None
            if bounds is None
            else Foams105BinnedRange(
                groups[index].group_id, bounds[0], bounds[1]
            )
            for index, bounds in enumerate(case["ranges"])
        )
        try:
            result = select_foams105_binned_ranges(case["labels"], groups, ranges)
        except Foams105SelectionDomainError as exc:
            if exc.code != case.get("expected_error"):
                raise ValueError(
                    f"Case {case['id']} produced unexpected error {exc.code}"
                ) from exc
            expected_errors += 1
            compared += 1
            continue
        if "expected_error" in case:
            raise ValueError(f"Case {case['id']} unexpectedly succeeded")
        expected = (
            tuple(case["expected_labels"]),
            tuple(case["expected_na"]),
            tuple(case["expected_source_slots"]),
            tuple(case["expected_global_indices"]),
        )
        candidate = (
            result.bin_labels_mm,
            result.na_per_mm2,
            tuple(row.source_slot for row in result.rows),
            tuple(row.global_bin_index for row in result.rows),
        )
        if candidate != expected:
            raise ValueError(f"Case {case['id']} differs from fixture")
        compared += 1

    print("FOAMS 1.0.5 checked binned selection report")
    print("analytical cases:", compared)
    print("expected typed rejections:", expected_errors)
    print("analytical mismatches: 0")


def workbook_report():
    histogram = _load(HISTOGRAM_REFERENCE_PATH)
    conversion = _load(CONVERSION_REFERENCE_PATH)
    labels = tuple(histogram["bin_labels_mm"])
    groups = tuple(
        Foams105BinnedGroupInput(
            f"group-{group['group_id']}",
            slot,
            tuple(group["source_na_per_mm2"]),
        )
        for slot, group in enumerate(histogram["groups"], start=1)
    )
    ranges = (
        Foams105BinnedRange("group-1", labels[14], labels[27]),
        None,
        Foams105BinnedRange("group-3", labels[0], labels[14]),
        None,
    )
    selected = select_foams105_binned_ranges(labels, groups, ranges)
    expected_labels = tuple(conversion["bin_labels_mm"])
    expected_na = tuple(conversion["na_per_mm2"])
    if selected.bin_labels_mm != expected_labels or selected.na_per_mm2 != expected_na:
        raise ValueError("Selected workbook rows differ from the pinned fixture")

    converted = convert_foams105_nv(selected.bin_labels_mm, selected.na_per_mm2)
    expected_nv = tuple(conversion["expected_nv_per_mm3"])
    for index, (candidate, reference) in enumerate(
        zip(converted.signed_nv_per_mm3, expected_nv)
    ):
        if not math.isclose(candidate, reference, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"Converted workbook row {index} differs")

    print("\nPinned workbook composition")
    print("selected rows:", selected.output_row_count)
    print("source slots:", tuple(row.source_slot for row in selected.rows))
    print("duplicate output pairs:", selected.adjacent_duplicate_output_index_pairs)
    print("overlapping global indices:", selected.overlapping_global_indices)
    print("converter rows reconciled:", len(expected_nv))
    print("boundary: binned selection only; raw-object selection is not implemented")


def main():
    analytical_report()
    workbook_report()


if __name__ == "__main__":
    main()
