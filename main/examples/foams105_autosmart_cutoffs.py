"""Report synthetic FOAMS 1.0.5 AutoSmart cutoff suggestions.

The fixture is hand-designed from pinned source behavior. It is not saved
MATLAB output and this example does not apply or concatenate the suggestions.
"""

import json
from pathlib import Path

from main.core import (
    Foams105CutoffDomainError,
    Foams105CutoffGroupInput,
    suggest_foams105_cutoffs,
)

REFERENCE_PATH = (
    Path(__file__).parents[1]
    / "tests"
    / "fixtures"
    / "10_foams105_autosmart_cases.json"
)
REFERENCE_CASE_COUNT = 7


def _groups(case):
    return tuple(
        Foams105CutoffGroupInput(
            f"group-{index + 1}",
            scale,
            tuple(densities),
        )
        for index, (scale, densities) in enumerate(
            zip(case["scales_px_per_mm"], case["na_per_mm2"])
        )
    )


def _require_equal(case_id, field, candidate, expected):
    if candidate != expected:
        raise ValueError(
            f"Case {case_id} {field} differs: {candidate!r} != {expected!r}"
        )


def cutoff_report():
    with REFERENCE_PATH.open(encoding="utf-8") as stream:
        fixture = json.load(stream)
    cases = fixture["cases"]
    if len(cases) != REFERENCE_CASE_COUNT:
        raise ValueError(
            f"Fixture must contain {REFERENCE_CASE_COUNT} cases; got {len(cases)}"
        )

    successes = 0
    expected_errors = 0
    for case in cases:
        case_id = case["id"]
        groups = _groups(case)
        if "expected_error" in case:
            try:
                suggest_foams105_cutoffs(case["labels_mm"], groups)
            except Foams105CutoffDomainError as exc:
                _require_equal(case_id, "error code", exc.code, case["expected_error"])
                print(f"{case_id}: expected domain error {exc.code}")
                expected_errors += 1
                continue
            raise ValueError(f"Case {case_id} did not raise its expected domain error")

        result = suggest_foams105_cutoffs(case["labels_mm"], groups)
        transition_indices = tuple(item.selected_index for item in result.transitions)
        expected_transitions = tuple(case["expected_transition_indices"])
        _require_equal(
            case_id,
            "transition indices",
            transition_indices,
            expected_transitions,
        )
        ranges = tuple(
            (item.lower_index, item.upper_index)
            for item in result.suggested_ranges
        )
        expected_ranges = tuple(tuple(value) for value in case["expected_ranges"])
        _require_equal(case_id, "ranges", ranges, expected_ranges)
        for suggested_range in result.suggested_ranges:
            _require_equal(
                case_id,
                "lower label",
                suggested_range.lower_label_mm,
                result.bin_labels_mm[suggested_range.lower_index],
            )
            _require_equal(
                case_id,
                "upper label",
                suggested_range.upper_label_mm,
                result.bin_labels_mm[suggested_range.upper_index],
            )

        print(f"{case_id}: groups {tuple(group.group_id for group in result.groups)}")
        for transition in result.transitions:
            trace = tuple(
                zip(transition.overlap_indices, transition.signed_log_differences)
            )
            print(
                "  transition",
                f"{transition.coarser_group_id}->{transition.finer_group_id}",
                "trace",
                trace,
                "ties",
                transition.tied_indices,
                "selected",
                transition.selected_index,
            )
        print(
            "  ranges",
            tuple(
                (
                    item.group_id,
                    item.lower_index,
                    item.upper_index,
                    item.lower_label_mm,
                    item.upper_label_mm,
                )
                for item in result.suggested_ranges
            ),
        )
        successes += 1

    _require_equal("report", "successful cases", successes, 5)
    _require_equal("report", "expected domain errors", expected_errors, 2)
    print("FOAMS 1.0.5 synthetic AutoSmart suggestion report")
    print("successful suggestions:", successes)
    print("expected domain errors:", expected_errors)
    print("mismatches: 0")
    print("boundary: suggestions only; no range application or concatenation")


def main():
    cutoff_report()


if __name__ == "__main__":
    main()
