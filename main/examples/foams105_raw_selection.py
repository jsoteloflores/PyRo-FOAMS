"""Report FOAMS 1.0.5 raw-object cutoff selection checks."""

import json
import math
from pathlib import Path

import numpy as np

from main.core import (
    Foams105BinnedGroupInput,
    Foams105BinnedRange,
    Foams105RawGroupInput,
    Foams105RawObject,
    Foams105RawSelectionDomainError,
    filter_foams105_diameters,
    measure_foams105_phase,
    select_foams105_binned_ranges,
    select_foams105_raw_objects,
)

FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "tests"
    / "fixtures"
    / "12_foams105_raw_selection_cases.json"
)


def _case_inputs(case):
    groups = tuple(
        Foams105RawGroupInput(
            f"slot-{slot}",
            slot,
            tuple(
                Foams105RawObject(
                    f"image-{slot}",
                    index + 1,
                    diameter,
                    math.pi * diameter * diameter / 4.0,
                )
                for index, diameter in enumerate(diameters)
            ),
        )
        for slot, diameters in enumerate(case["diameters"], start=1)
    )
    ranges = tuple(
        None
        if bounds is None
        else Foams105BinnedRange(groups[index].group_id, *bounds)
        for index, bounds in enumerate(case["ranges"])
    )
    return groups, ranges


def analytical_report():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    successful = 0
    expected_rejections = 0
    for case in fixture["cases"]:
        groups, ranges = _case_inputs(case)
        try:
            result = select_foams105_raw_objects(case["labels"], groups, ranges)
        except Foams105RawSelectionDomainError as exc:
            if exc.code != case.get("expected_error"):
                raise ValueError(
                    f"Case {case['id']} produced unexpected error {exc.code}"
                ) from exc
            expected_rejections += 1
            continue
        if "expected_error" in case:
            raise ValueError(f"Case {case['id']} unexpectedly succeeded")
        expected_indices = tuple(
            tuple(indices) for indices in case["expected_indices"]
        )
        actual_indices = tuple(
            trace.selected_indices for trace in result.group_traces
        )
        actual_lowers = tuple(
            trace.effective_lower_mm for trace in result.group_traces
        )
        if actual_indices != expected_indices:
            raise ValueError(f"Case {case['id']} selected indices differ")
        if actual_lowers != tuple(case["effective_lowers"]):
            raise ValueError(f"Case {case['id']} effective bounds differ")
        for row in result.rows:
            source = result.groups[row.source_slot - 1].objects[
                row.source_object_index
            ]
            if (
                row.image_id,
                row.component_id,
                row.equivalent_diameter_mm,
                row.area_mm2,
            ) != (
                source.image_id,
                source.component_id,
                source.equivalent_diameter_mm,
                source.area_mm2,
            ):
                raise ValueError(f"Case {case['id']} row provenance differs")
        successful += 1

    print("FOAMS 1.0.5 raw-object selection report")
    print("analytical cases:", len(fixture["cases"]))
    print("successful selections:", successful)
    print("expected typed rejections:", expected_rejections)
    print("complete mismatches: 0")


def measurement_filter_selection_report():
    image = np.zeros((14, 14), dtype=np.uint8)
    image[2, 2] = 1
    image[5, 5:7] = 1
    image[8:10, 8:10] = 1
    measured = measure_foams105_phase(image, 1, 10)
    diameters = tuple(
        component.equivalent_diameter_mm for component in measured.components
    )
    filtered = filter_foams105_diameters(diameters, 2, 10)
    objects = tuple(
        Foams105RawObject(
            "synthetic-image",
            measured.components[index].component_id,
            measured.components[index].equivalent_diameter_mm,
            measured.components[index].area_mm2,
        )
        for index in filtered.retained_indices
    )
    result = select_foams105_raw_objects(
        (0.15, 0.3, 1.0),
        (Foams105RawGroupInput("measured", 1, objects),),
        (Foams105BinnedRange("measured", 0.15, 1.0),),
    )
    if filtered.retained_indices != (1, 2):
        raise ValueError("Threshold filtering retained unexpected source indices")
    if result.selected_identities != (
        ("synthetic-image", 2),
        ("synthetic-image", 3),
    ):
        raise ValueError("Component identities were renumbered after filtering")

    print("\nMeasurement -> threshold filter -> raw selection")
    print("retained measurement indices:", filtered.retained_indices)
    print("selected stable identities:", result.selected_identities)
    print("selected areas mm^2:", result.areas_mm2)


def ordering_report():
    raw = select_foams105_raw_objects(
        (1.0, 2.0, 4.0),
        (
            Foams105RawGroupInput("coarse", 1, (Foams105RawObject("c", 1, 2.0, 1.0),)),
            Foams105RawGroupInput("fine", 2, (Foams105RawObject("f", 1, 1.0, 1.0),)),
        ),
        (
            Foams105BinnedRange("coarse", 2.0, 4.0),
            Foams105BinnedRange("fine", 1.0, 1.0),
        ),
    )
    binned = select_foams105_binned_ranges(
        (1.0, 2.0, 4.0),
        (
            Foams105BinnedGroupInput("coarse", 1, (0.0, 1.0, 1.0)),
            Foams105BinnedGroupInput("fine", 2, (1.0, 0.0, 0.0)),
        ),
        (
            Foams105BinnedRange("coarse", 2.0, 4.0),
            Foams105BinnedRange("fine", 1.0, 1.0),
        ),
    )
    raw_slots = tuple(row.source_slot for row in raw.rows)
    binned_slots = tuple(row.source_slot for row in binned.rows)
    if raw_slots != (1, 2) or binned_slots != (2, 1, 1):
        raise ValueError("Raw and binned source-slot order differs from policy")

    print("\nSeparate concatenation policies")
    print("raw object slots, ascending:", raw_slots)
    print("binned rows, descending:", binned_slots)
    print("boundary: no vesicularity, shape, or original-workbook raw oracle")


def main():
    analytical_report()
    measurement_filter_selection_report()
    ordering_report()


if __name__ == "__main__":
    main()
