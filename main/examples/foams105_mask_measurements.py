"""Report synthetic FOAMS 1.0.5 mask-topology measurement checks.

The fixture is hand-authored from the pinned source sequence and documented
MATLAB topology semantics. It is not MATLAB output or original acquisition
data, and this report does not claim original-image parity.
"""

import json
import math
from pathlib import Path

import numpy as np

from main.core import (
    count_foams105_histogram,
    filter_foams105_diameters,
    measure_foams105_phase,
)

REFERENCE_PATH = (
    Path(__file__).parents[1]
    / "tests"
    / "fixtures"
    / "08_foams105_measurement_cases.json"
)
REFERENCE_CASE_COUNT = 7


def _require_equal_array(case_id, field, candidate, expected):
    candidate_array = np.asarray(candidate)
    expected_array = np.asarray(expected)
    if candidate_array.shape != expected_array.shape:
        raise ValueError(
            f"Case {case_id} {field} shape differs: "
            f"{candidate_array.shape} != {expected_array.shape}"
        )
    if not np.array_equal(candidate_array, expected_array):
        raise ValueError(f"Case {case_id} {field} differs from the fixture")


def topology_report():
    with REFERENCE_PATH.open(encoding="utf-8") as stream:
        fixture = json.load(stream)
    cases = fixture["cases"]
    if len(cases) != REFERENCE_CASE_COUNT:
        raise ValueError(
            f"Fixture must contain {REFERENCE_CASE_COUNT} cases; got {len(cases)}"
        )

    compared_cases = 0
    compared_arrays = 0
    compared_components = 0
    for case in cases:
        case_id = case["id"]
        result = measure_foams105_phase(
            np.asarray(case["image"]),
            case["foreground_value"],
            case["scale_px_per_mm"],
        )
        arrays = (
            ("removed_border_mask", result.removed_border_mask, case["expected_removed_border_mask"]),
            ("filled_mask", result.filled_mask, case["expected_filled_mask"]),
            ("label_image", result.label_image, case["expected_label_image"]),
        )
        for field, candidate, expected in arrays:
            _require_equal_array(case_id, field, candidate, expected)
            compared_arrays += 1

        expected_areas = tuple(case["expected_area_px"])
        candidate_areas = tuple(component.area_px for component in result.components)
        if candidate_areas != expected_areas:
            raise ValueError(
                f"Case {case_id} component areas differ: "
                f"{candidate_areas} != {expected_areas}"
            )
        for component, area_px in zip(result.components, expected_areas):
            scale = float(case["scale_px_per_mm"])
            expected_area = (float(area_px) / scale) / scale
            expected_diameter = 2.0 * math.sqrt(expected_area / math.pi)
            if not math.isclose(
                component.area_mm2, expected_area, rel_tol=1e-12, abs_tol=0.0
            ):
                raise ValueError(f"Case {case_id} calibrated area differs")
            if not math.isclose(
                component.equivalent_diameter_mm,
                expected_diameter,
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                raise ValueError(f"Case {case_id} equivalent diameter differs")
        if result.selected_pixel_count != (
            result.removed_border_pixel_count
            + result.retained_before_fill_pixel_count
        ):
            raise ValueError(f"Case {case_id} border conservation failed")
        if result.final_foreground_pixel_count != (
            result.retained_before_fill_pixel_count + result.filled_added_pixel_count
        ):
            raise ValueError(f"Case {case_id} fill conservation failed")
        if result.final_foreground_pixel_count != sum(candidate_areas):
            raise ValueError(f"Case {case_id} component conservation failed")
        compared_cases += 1
        compared_components += len(result.components)

    print("FOAMS 1.0.5 synthetic mask-topology report")
    print("compared cases:", compared_cases)
    print("compared complete arrays:", compared_arrays)
    print("compared components:", compared_components)
    print("mismatches: 0")
    print("boundary: analytical masks only; MATLAB and original images were not run")


def composition_report():
    image = np.zeros((14, 14), dtype=np.uint8)
    image[2, 2] = 1
    image[5, 5:7] = 1
    image[8:10, 8:10] = 1
    measurement = measure_foams105_phase(image, 1, 10)
    diameters = tuple(
        component.equivalent_diameter_mm for component in measurement.components
    )
    filtered = filter_foams105_diameters(diameters, 2, 10)
    retained_components = tuple(
        measurement.components[index] for index in filtered.retained_indices
    )
    retained_diameters = tuple(
        component.equivalent_diameter_mm for component in retained_components
    )
    histogram = count_foams105_histogram((0.18, 0.3, 1.0), retained_diameters)
    if tuple(component.area_px for component in retained_components) != (2, 4):
        raise ValueError("Synthetic component filtering differs")
    if histogram.counts != (1, 1, 0) or histogram.discarded_upper_count != 0:
        raise ValueError("Synthetic measurement histogram differs")

    print("\nSynthetic measurement -> diameter filter -> histogram")
    print("component pixel areas:", tuple(component.area_px for component in measurement.components))
    print("retained original indices:", filtered.retained_indices)
    print("retained component pixel areas:", tuple(component.area_px for component in retained_components))
    print("histogram counts/discarded upper:", histogram.counts, histogram.discarded_upper_count)
    print("boundary: no corrected-area normalization or automatic nesting")


def main():
    topology_report()
    composition_report()


if __name__ == "__main__":
    main()
