"""Report one synthetic FOAMS 1.0.5 two-dimensional vesicularity replay.

This example uses analytical masks and the public measurement, area, raw
selection, and vesicularity APIs. It is not MATLAB or workbook parity evidence.
"""

import math

import numpy as np

from main.core import (
    Foams105BinnedRange,
    Foams105RawGroupInput,
    Foams105RawObject,
    Foams105VesicularityGroupInput,
    calculate_foams105_2d_vesicularity,
    calculate_foams105_image_areas,
    measure_foams105_image,
    select_foams105_raw_objects,
)


def _require_close(field, candidate, expected):
    if not math.isclose(candidate, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{field} differs: {candidate!r} != {expected!r}")


def _group(group_id, source_slot, scale_px_per_mm, image):
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
    return (
        Foams105RawGroupInput(group_id, source_slot, objects),
        Foams105VesicularityGroupInput(group_id, source_slot, (area,), objects),
    )


def vesicularity_report():
    coarse = np.zeros((10, 10), dtype=np.uint8)
    coarse[2, 2] = 1
    coarse[5:7, 5:7] = 1
    fine = np.zeros((10, 10), dtype=np.uint8)
    fine[3, 3] = 1

    coarse_raw, coarse_context = _group("coarse", 1, 1.0, coarse)
    fine_raw, fine_context = _group("fine", 2, 2.0, fine)
    raw = select_foams105_raw_objects(
        (0.1, 1.0, 10.0),
        (coarse_raw, fine_raw),
        (
            Foams105BinnedRange("coarse", 1.0, 10.0),
            Foams105BinnedRange("fine", 0.1, 10.0),
        ),
    )
    result = calculate_foams105_2d_vesicularity(
        raw, (coarse_context, fine_context)
    )

    if result.source_sorted_area2_by_slot_mm2 != (100.0, 25.0):
        raise ValueError("Source denominator assignment differs")
    if result.actual_group_area2_by_slot_mm2 != (100.0, 25.0):
        raise ValueError("Actual image-membership denominators differ")
    _require_close(
        "integrated_vesicularity_percent",
        result.integrated_vesicularity_percent,
        5.95,
    )
    if result.group_vesicularity_percent != (5.0, 1.0):
        raise ValueError("Per-group vesicularity differs")
    _require_close(
        "mean_group_vesicularity_percent",
        result.mean_group_vesicularity_percent,
        3.0,
    )

    print("FOAMS 1.0.5 synthetic 2D vesicularity report")
    print("source area2 denominators:", result.source_sorted_area2_by_slot_mm2)
    print("actual group area2 denominators:", result.actual_group_area2_by_slot_mm2)
    print("selected areas mm^2:", result.selected_area_by_slot_mm2)
    print("unfiltered areas mm^2:", result.unfiltered_area_by_slot_mm2)
    print("integrated vesicularity percent:", result.integrated_vesicularity_percent)
    print("group vesicularity percent:", result.group_vesicularity_percent)
    print("mean group vesicularity percent:", result.mean_group_vesicularity_percent)
    print("mismatches: 0")
    print("boundary: no MATLAB, workbook, original-image, volume, density, or shape parity")


def main():
    vesicularity_report()


if __name__ == "__main__":
    main()
