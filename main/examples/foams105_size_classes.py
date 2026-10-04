"""Report bounded FOAMS 1.0.5 size-class preparation checks.

The saved-workbook comparison counts already-thresholded equivalent diameters.
The separate synthetic section demonstrates label generation, filtering, and
normalization without claiming reconstructed acquisition settings or areas.
"""

import json
import math
from pathlib import Path

from main.core import (
    build_foams105_bin_labels,
    count_foams105_histogram,
    filter_foams105_diameters,
    normalize_foams105_counts,
)

REFERENCE_PATH = (
    Path(__file__).parents[1]
    / "tests"
    / "fixtures"
    / "07_foams105_histogram_reference.json"
)
REFERENCE_GROUP_INPUT_COUNTS = (777, 135, 2076, 0)
REFERENCE_LABEL_COUNT = 45
REFERENCE_COMPARED_ROWS = 180


def _count_mismatches(candidate_values, expected_values, group_id):
    candidates = tuple(candidate_values)
    expected = tuple(expected_values)
    if len(candidates) != REFERENCE_LABEL_COUNT:
        raise ValueError(
            f"Group {group_id} candidate length must be {REFERENCE_LABEL_COUNT}; "
            f"got {len(candidates)}"
        )
    if len(expected) != REFERENCE_LABEL_COUNT:
        raise ValueError(
            f"Group {group_id} reference length must be {REFERENCE_LABEL_COUNT}; "
            f"got {len(expected)}"
        )
    return tuple(
        index
        for index, (candidate, reference) in enumerate(zip(candidates, expected))
        if candidate != reference
    )


def workbook_report():
    with REFERENCE_PATH.open(encoding="utf-8") as stream:
        fixture = json.load(stream)
    labels = fixture["bin_labels_mm"]
    groups = fixture["groups"]
    if len(labels) != REFERENCE_LABEL_COUNT:
        raise ValueError(
            f"Fixture label length must be {REFERENCE_LABEL_COUNT}; got {len(labels)}"
        )
    if len(groups) != len(REFERENCE_GROUP_INPUT_COUNTS):
        raise ValueError(
            f"Fixture group count must be {len(REFERENCE_GROUP_INPUT_COUNTS)}; "
            f"got {len(groups)}"
        )

    compared_rows = 0
    total_input = 0
    total_retained = 0
    total_discarded = 0
    mismatches = []
    group_counts = []
    for group, expected_input_count in zip(groups, REFERENCE_GROUP_INPUT_COUNTS):
        group_id = group["group_id"]
        diameters = group["already_thresholded_equivalent_diameters_mm"]
        if len(diameters) != expected_input_count:
            raise ValueError(
                f"Group {group_id} input count must be {expected_input_count}; "
                f"got {len(diameters)}"
            )
        result = count_foams105_histogram(labels, diameters)
        group_mismatches = _count_mismatches(
            result.counts, group["expected_counts"], group_id
        )
        mismatches.extend((group_id, index) for index in group_mismatches)
        compared_rows += len(result.counts)
        total_input += result.input_count
        total_retained += result.retained_count
        total_discarded += result.discarded_upper_count
        group_counts.append(result.input_count)

    if compared_rows != REFERENCE_COMPARED_ROWS:
        raise ValueError(
            f"Compared row count must be {REFERENCE_COMPARED_ROWS}; got {compared_rows}"
        )
    if total_retained + total_discarded != total_input:
        raise ValueError("Fixture histogram conservation failed")
    if total_discarded != 0:
        raise ValueError(
            f"Fixture discarded upper-tail count must be zero; got {total_discarded}"
        )
    if mismatches:
        raise ValueError(f"Fixture count mismatches: {tuple(mismatches)}")

    print("FOAMS 1.0.5 saved-workbook histogram comparison")
    print("group observation counts:", tuple(group_counts))
    print("compared count rows:", compared_rows)
    print("mismatches:", len(mismatches))
    print(
        "conservation (input, retained, discarded upper):",
        (total_input, total_retained, total_discarded),
    )
    print("boundary: supplied diameters were already thresholded in the workbook")


def synthetic_report():
    labels = build_foams105_bin_labels(10, (100,))
    threshold = (10.0 / 100.0) * (10.0 ** -0.1)
    unfiltered = (
        math.nextafter(threshold, 0.0),
        threshold,
        0.5,
        1.0,
        1.5,
        2.0,
        3.9,
        4.0,
        5.0,
    )
    filtered = filter_foams105_diameters(unfiltered, 10, 100)
    analytical = count_foams105_histogram(
        (1.0, 2.0, 4.0), filtered.retained_diameters_mm[1:]
    )
    normalized = normalize_foams105_counts(analytical, 2.0)
    expected_counts = (1, 2, 2)
    expected_na = (0.5, 1.0, 1.0)
    if analytical.counts != expected_counts:
        raise ValueError(
            f"Synthetic counts differ: {analytical.counts} != {expected_counts}"
        )
    if normalized.na_per_mm2 != expected_na:
        raise ValueError(
            f"Synthetic N_A differs: {normalized.na_per_mm2} != {expected_na}"
        )

    print("\nSource-derived analytical checks with synthetic inputs")
    print("generated first five labels mm:", labels.bin_labels_mm[:5])
    print("filter threshold mm:", filtered.threshold_mm)
    print("retained/rejected indices:", filtered.retained_indices, filtered.rejected_below_threshold_indices)
    print("analytical counts/discarded upper:", analytical.counts, analytical.discarded_upper_count)
    print("N_A from caller-supplied area 2 mm^2:", normalized.na_per_mm2)


def main():
    workbook_report()
    synthetic_report()


if __name__ == "__main__":
    main()
