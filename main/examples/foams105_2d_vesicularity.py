"""Report analytical and synthetic FOAMS 1.0.5 2D vesicularity checks.

The fixture is hand-derived and the public examples use synthetic masks. They
are not MATLAB, workbook-vesicularity, or original-image parity evidence.
"""

import json
import math
from pathlib import Path

import numpy as np

from main.core import (
    DIAGNOSTIC_SOURCE_AREA_REASSIGNED,
    Foams105BinnedRange,
    Foams105RawGroupInput,
    Foams105RawObject,
    Foams105VesicularityGroupInput,
    calculate_foams105_2d_vesicularity,
    calculate_foams105_image_areas,
    filter_foams105_diameters,
    measure_foams105_image,
    select_foams105_raw_objects,
)
from main.core.legacy_vesicularity import (
    Foams105VesicularityDomainError,
    Foams105VesicularityNumericalError,
    _calculate_foams105_vesicularity_numerical,
)

FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "tests"
    / "fixtures"
    / "13_foams105_vesicularity_cases.json"
)


def _require_close(case_id, field, candidate, expected):
    if not math.isclose(candidate, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"Case {case_id} {field} differs: {candidate!r} != {expected!r}"
        )


def _require_close_sequence(case_id, field, candidate, expected):
    if len(candidate) != len(expected):
        raise ValueError(f"Case {case_id} {field} length differs")
    for index, (value, reference) in enumerate(zip(candidate, expected)):
        _require_close(case_id, f"{field}[{index}]", value, reference)


def analytical_report():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if len(fixture["cases"]) != 13:
        raise ValueError("Vesicularity fixture must contain exactly 13 cases")
    successful = 0
    expected_rejections = 0
    for case in fixture["cases"]:
        try:
            result = _calculate_foams105_vesicularity_numerical(
                case["scales"],
                case["image_area2_by_group"],
                case["selected_area_sums"],
                case["unfiltered_area_sums"],
            )
        except (
            Foams105VesicularityDomainError,
            Foams105VesicularityNumericalError,
        ) as exc:
            if exc.code != case.get("expected_error"):
                raise ValueError(
                    f"Case {case['id']} produced unexpected error {exc.code}"
                ) from exc
            expected_rejections += 1
            continue
        if "expected_error" in case:
            raise ValueError(f"Case {case['id']} unexpectedly succeeded")
        _require_close_sequence(
            case["id"],
            "source_denominators",
            result.source_sorted_area2_by_slot_mm2,
            case["expected_source_denominators"],
        )
        _require_close_sequence(
            case["id"],
            "residuals",
            result.residuals_reference_area_mm2,
            case["expected_residuals"],
        )
        _require_close(
            case["id"],
            "integrated_percent",
            result.integrated_vesicularity_percent,
            case["expected_integrated_percent"],
        )
        _require_close_sequence(
            case["id"],
            "group_percent",
            result.group_vesicularity_percent,
            case["expected_group_percent"],
        )
        _require_close(
            case["id"],
            "mean_percent",
            result.mean_group_vesicularity_percent,
            case["expected_mean_percent"],
        )
        if result.reassigned_slots != tuple(
            case.get("expected_reassignment_slots", ())
        ):
            raise ValueError(f"Case {case['id']} reassignment slots differ")
        successful += 1

    if (successful, expected_rejections) != (10, 3):
        raise ValueError("Vesicularity fixture outcome counts differ")
    print("FOAMS 1.0.5 analytical 2D vesicularity report")
    print("fixture cases:", len(fixture["cases"]))
    print("successful calculations:", successful)
    print("expected typed rejections:", expected_rejections)
    print("mismatches: 0")


def _measured_group(group_id, source_slot, scale_px_per_mm, image):
    image_id = f"{group_id}-image"
    measurement = measure_foams105_image(
        image, 1, scale_px_per_mm, (None, None, None, None)
    )
    area = calculate_foams105_image_areas(image_id, measurement)
    objects = tuple(
        Foams105RawObject(
            image_id,
            component.component_id,
            component.equivalent_diameter_mm,
            component.area_mm2,
        )
        for component in measurement.pore_result.components
    )
    context = Foams105VesicularityGroupInput(
        group_id, source_slot, (area,), objects
    )
    return measurement, objects, context


def filtered_public_api_report():
    image = np.zeros((20, 20), dtype=np.uint8)
    image[2, 2] = 1
    image[6, 6:8] = 1
    image[11:13, 11:13] = 1
    measurement, unfiltered_objects, context = _measured_group(
        "measured", 1, 10.0, image
    )
    diameters = tuple(
        component.equivalent_diameter_mm
        for component in measurement.pore_result.components
    )
    filtered = filter_foams105_diameters(diameters, 2.0, 10.0)
    raw_objects = tuple(unfiltered_objects[index] for index in filtered.retained_indices)
    raw = select_foams105_raw_objects(
        (0.1, 0.2, 1.0),
        (Foams105RawGroupInput("measured", 1, raw_objects),),
        (Foams105BinnedRange("measured", 0.2, 1.0),),
    )
    result = calculate_foams105_2d_vesicularity(raw, (context,))

    if filtered.retained_indices != (1, 2):
        raise ValueError("Diameter filter did not retain original indices 1 and 2")
    if filtered.rejected_below_threshold_indices != (0,):
        raise ValueError("Diameter filter did not reject original index 0")
    if raw.selected_identities != (("measured-image", 3),):
        raise ValueError("Raw cutoff did not retain only component 3")
    if result.group_traces[0].unfiltered_identities != (
        ("measured-image", 1),
        ("measured-image", 2),
        ("measured-image", 3),
    ):
        raise ValueError("Unfiltered inventory lost stable component identities")
    _require_close(
        "public", "selected_area", result.selected_area_by_slot_mm2[0], 0.04
    )
    _require_close(
        "public", "unfiltered_area", result.unfiltered_area_by_slot_mm2[0], 0.07
    )

    print("\nMeasurement -> diameter filter -> raw cutoff -> vesicularity")
    print("filter retained original indices:", filtered.retained_indices)
    print("filter rejected original indices:", filtered.rejected_below_threshold_indices)
    print("cutoff-selected identities:", raw.selected_identities)
    print("unfiltered identities:", result.group_traces[0].unfiltered_identities)
    print("selected/unfiltered areas mm^2:", 0.04, 0.07)
    print("integrated percent:", result.integrated_vesicularity_percent)
    print("group percent:", result.group_vesicularity_percent)


def reassignment_report():
    coarse = np.zeros((4, 4), dtype=np.uint8)
    coarse[1, 1] = 1
    fine = np.zeros((30, 30), dtype=np.uint8)
    fine[5, 5] = 1
    _, coarse_objects, coarse_context = _measured_group("coarse", 1, 1.0, coarse)
    _, fine_objects, fine_context = _measured_group("fine", 2, 2.0, fine)
    raw = select_foams105_raw_objects(
        (0.1, 1.0, 10.0),
        (
            Foams105RawGroupInput("coarse", 1, coarse_objects),
            Foams105RawGroupInput("fine", 2, fine_objects),
        ),
        (
            Foams105BinnedRange("coarse", 1.0, 10.0),
            Foams105BinnedRange("fine", 0.1, 10.0),
        ),
    )
    result = calculate_foams105_2d_vesicularity(
        raw, (coarse_context, fine_context)
    )
    reassigned = tuple(
        diagnostic.source_slot
        for diagnostic in result.diagnostics
        if diagnostic.code == DIAGNOSTIC_SOURCE_AREA_REASSIGNED
    )
    if reassigned != (1, 2):
        raise ValueError("Expected source-area reassignment diagnostics for both slots")
    if result.source_sorted_area2_by_slot_mm2 != (225.0, 16.0):
        raise ValueError("Source-sorted denominators differ")
    if result.actual_group_area2_by_slot_mm2 != (16.0, 225.0):
        raise ValueError("Actual-membership denominators differ")

    print("\nIndependent source-area sorting")
    print("source-sorted area2:", result.source_sorted_area2_by_slot_mm2)
    print("actual-membership area2:", result.actual_group_area2_by_slot_mm2)
    print("reassignment diagnostic slots:", reassigned)
    print("boundary: no MATLAB, workbook-vesicularity, or original-image parity")


def main():
    analytical_report()
    filtered_public_api_report()
    reassignment_report()


if __name__ == "__main__":
    main()
