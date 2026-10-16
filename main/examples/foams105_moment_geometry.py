"""Report bounded FOAMS 1.0.5 component moment geometry and shape identity.

Analytical fixture expectations are unit-square pixel-moment calculations, not
MATLAB output. Perimeter compatibility remains a separate pending stage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from main.core import (
    FOAMS105_ORIENTATION_TIE_REASON,
    FOAMS105_PERIMETER_STATUS_PENDING,
    Foams105BinnedRange,
    Foams105RawGroupInput,
    Foams105RawObject,
    filter_foams105_diameters,
    measure_foams105_moment_geometry,
    measure_foams105_phase,
    select_foams105_moment_geometry,
    select_foams105_raw_objects,
)

FIXTURE_PATH = Path(__file__).parents[1] / "tests" / "fixtures" / "16_foams105_moment_cases.json"


def _require_close(case_id, field, candidate, expected):
    if not math.isclose(candidate, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"Case {case_id} {field} differs: {candidate!r} != {expected!r}"
        )


def _build_image(shape, pixels):
    image = np.zeros(shape, dtype=np.uint8)
    for row, column in pixels:
        image[row, column] = 1
    return image


def analytical_report():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    cases = fixture["cases"]
    if len(cases) != 8:
        raise ValueError("Moment fixture must contain exactly eight cases")

    validated_components = 0
    for case in cases:
        image = _build_image(tuple(case["shape"]), case["pixels"])
        measurement = measure_foams105_phase(image, 1, case["scale"])
        result = measure_foams105_moment_geometry(case["id"], measurement)
        if len(result.components) != len(case["components"]):
            raise ValueError(f"Case {case['id']} component count differs")
        for component, expected in zip(result.components, case["components"]):
            if component.component_id != expected["id"] or component.area_px != expected["area"]:
                raise ValueError(f"Case {case['id']} identity differs")
            _require_close(case["id"], "Cxx", component.central_cxx, expected["cxx"])
            _require_close(case["id"], "Cxy", component.central_cxy, expected["cxy"])
            _require_close(case["id"], "Cyy", component.central_cyy, expected["cyy"])
            _require_close(case["id"], "major_px", component.major_axis_px, expected["major_px"])
            _require_close(case["id"], "minor_px", component.minor_axis_px, expected["minor_px"])
            _require_close(case["id"], "eccentricity", component.eccentricity, expected["eccentricity"])
            _require_close(
                case["id"],
                "theta",
                component.orientation_regionprops_profile_deg,
                expected["theta"],
            )
            if component.orientation_defined != expected["defined"]:
                raise ValueError(f"Case {case['id']} orientation-defined status differs")
            if expected["defined"] and component.orientation_tie_reason is not None:
                raise ValueError(f"Case {case['id']} tie reason unexpectedly populated")
            if (not expected["defined"]) and component.orientation_tie_reason != FOAMS105_ORIENTATION_TIE_REASON:
                raise ValueError(f"Case {case['id']} tie reason differs")
            if component.perimeter_status != FOAMS105_PERIMETER_STATUS_PENDING:
                raise ValueError(f"Case {case['id']} perimeter status differs")
            validated_components += 1

    print("FOAMS 1.0.5 analytical moment report")
    print("fixture cases:", len(cases))
    print("validated components:", validated_components)
    print("mismatches: 0")


def public_adapter_report():
    image = np.zeros((12, 12), dtype=np.uint8)
    image[2, 2] = 1
    image[2:4, 6:8] = 1
    image[6:9, 2:5] = 1
    measurement = measure_foams105_phase(image, 1, 1.0)
    geometry = measure_foams105_moment_geometry("example-image", measurement)

    minimum_diameter_px = 2.0
    filter_result = filter_foams105_diameters(
        tuple(component.equivalent_diameter_mm for component in measurement.components),
        minimum_diameter_px,
        1.0,
    )
    retained_objects = tuple(
        Foams105RawObject(
            "example-image",
            measurement.components[index].component_id,
            measurement.components[index].equivalent_diameter_mm,
            measurement.components[index].area_mm2,
        )
        for index in filter_result.retained_indices
    )
    raw = select_foams105_raw_objects(
        (1.0, 2.0, 3.0, 4.0),
        (Foams105RawGroupInput("example-group", 1, retained_objects),),
        (Foams105BinnedRange("example-group", 3.0, 4.0),),
    )
    selected = select_foams105_moment_geometry(raw, (geometry,))

    rejected_threshold_ids = tuple(
        measurement.components[index].component_id
        for index in filter_result.rejected_below_threshold_indices
    )
    cutoff_rejected_ids = tuple(
        raw.groups[0].objects[index].component_id
        for index in raw.group_traces[0].rejected_indices
    )

    if rejected_threshold_ids != (1,):
        raise ValueError("Expected one threshold-rejected component with original ID 1")
    if cutoff_rejected_ids != (3,):
        raise ValueError("Expected one cutoff-rejected retained component with original ID 3")
    if tuple(row.component_id for row in selected.rows) != (2,):
        raise ValueError("Selected moment geometry did not retain original component ID 2")

    print("\nMeasurement -> diameter filter -> raw selection -> moment geometry")
    print("threshold-rejected IDs:", rejected_threshold_ids)
    print("cutoff-rejected IDs:", cutoff_rejected_ids)
    print("selected IDs:", tuple(row.component_id for row in selected.rows))
    print("perimeter status:", selected.rows[0].component_geometry.perimeter_status)
    print("boundary: perimeter and derived shape statistics are not in this stage")


def main():
    analytical_report()
    public_adapter_report()


if __name__ == "__main__":
    main()
