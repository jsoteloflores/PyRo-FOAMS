"""Report synthetic FOAMS 1.0.5 area-accounting checks.

The fixture contains hand-authored analytical expectations. It is not MATLAB
output or original acquisition data and does not establish workflow parity.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from main.core import (
    aggregate_foams105_group_areas,
    calculate_foams105_image_areas,
    estimate_foams105_binary_area,
    measure_foams105_image,
)

REFERENCE_PATH = (
    Path(__file__).parents[1]
    / "tests"
    / "fixtures"
    / "09_foams105_area_cases.json"
)
WEIGHTED_CASE_COUNT = 6
IMAGE_CASE_COUNT = 2


def _require_equal(case_id, field, candidate, expected):
    if candidate != expected:
        raise ValueError(
            f"Case {case_id} {field} differs: {candidate!r} != {expected!r}"
        )


def area_report():
    with REFERENCE_PATH.open(encoding="utf-8") as stream:
        fixture = json.load(stream)
    weighted_cases = fixture["weighted_area_cases"]
    image_cases = fixture["area_cases"]
    if len(weighted_cases) != WEIGHTED_CASE_COUNT:
        raise ValueError(
            f"Fixture must contain {WEIGHTED_CASE_COUNT} weighted cases"
        )
    if len(image_cases) != IMAGE_CASE_COUNT:
        raise ValueError(f"Fixture must contain {IMAGE_CASE_COUNT} image cases")

    for case in weighted_cases:
        result = estimate_foams105_binary_area(np.asarray(case["mask"]))
        _require_equal(
            case["id"],
            "weighted_area_px",
            result.weighted_area_px,
            case["expected_weighted_area_px"],
        )

    image_results = []
    for case in image_cases:
        image = np.asarray(case["image"], dtype=np.uint8)
        measurement = measure_foams105_image(
            image,
            case["pore_value"],
            case["scale_px_per_mm"],
            tuple(case["excluded_phase_values"]),
        )
        result = calculate_foams105_image_areas(
            case["id"],
            measurement,
            tuple(case["excluded_min_area_mm2"]),
        )
        checks = (
            ("full_area_mm2", result.full_area_mm2, case["expected_full_area_mm2"]),
            (
                "border_weighted_area_px",
                result.border_weighted_area_px,
                case["expected_border_weighted_area_px"],
            ),
            ("border_area_mm2", result.border_area_mm2, case["expected_border_area_mm2"]),
            (
                "excluded_area_mm2_by_slot",
                result.excluded_area_mm2_by_slot,
                tuple(case["expected_excluded_area_mm2_by_slot"]),
            ),
            ("area2_mm2", result.border_corrected_area_mm2, case["expected_area2_mm2"]),
            ("area1_mm2", result.phase_corrected_area_mm2, case["expected_area1_mm2"]),
        )
        for field, candidate, expected in checks:
            _require_equal(case["id"], field, candidate, expected)
        image_results.append(result)

    compatible_images = (
        image_results[0],
        replace(image_results[0], image_id=f"{image_results[0].image_id}-repeat"),
    )
    group = aggregate_foams105_group_areas("fixture-group", compatible_images)
    _require_equal(
        "fixture-group",
        "image_ids",
        group.image_ids,
        tuple(result.image_id for result in compatible_images),
    )

    print("FOAMS 1.0.5 synthetic area-accounting report")
    print("weighted-area cases:", len(weighted_cases))
    print("image-area cases:", len(image_cases))
    print("group area1/area2:", group.phase_corrected_area_mm2, group.border_corrected_area_mm2)
    print("mismatches: 0")
    print("boundary: no MATLAB, original images, vesicularity, nesting, or shape statistics")


def main():
    area_report()


if __name__ == "__main__":
    main()
