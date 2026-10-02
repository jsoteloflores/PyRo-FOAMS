"""Headless spherical 3D reconstruction examples.

The assigned pore diameters below are synthetic. The end-to-end path
demonstrates sampling, 2D distribution, manual nesting, and reconstruction;
it is not a segmentation accuracy or legacy FOAMS equivalence benchmark.
"""

import math
from dataclasses import replace

import numpy as np

from main.core import (
    UPPER_TAIL_ASSUMPTION,
    NestingPlan,
    NestingSegment,
    build_sampling_dataset,
    build_spherical_section_operator,
    calculate_2d_number_densities,
    create_diameter_bin_spec,
    create_image_sampling_record,
    measure_labels,
    project_spherical_number_densities,
    reconstruct_spherical_3d,
    solve_spherical_number_densities,
)


def _group_image(image_id, image_index, group_id, width, diameters):
    labels = np.zeros((100, width), dtype=np.int32)
    for label in range(1, len(diameters) + 1):
        labels.flat[width + label] = label
    image = create_image_sampling_record(
        sample_id="example",
        image_id=image_id,
        image_index=image_index,
        magnification_group_id=group_id,
        label_map=labels,
        calibration=0.01,
        calibration_unit="mm",
        min_detectable_diameter=0.1,
        max_reliable_diameter=1.6,
    )
    pores = [
        replace(pore, eq_diam_px=diameters[pore.label - 1] / 0.01)
        for pore in measure_labels(labels, image_index=image_index)
    ]
    return image, pores


def analytical_example():
    operator = build_spherical_section_operator(
        create_diameter_bin_spec((1.0, 2.0, 4.0))
    )
    expected = (
        (math.sqrt(3.0), math.sqrt(15.0) - 2.0 * math.sqrt(3.0)),
        (0.0, 2.0 * math.sqrt(3.0)),
    )
    assert all(
        math.isclose(actual, target, rel_tol=1e-15)
        for actual_row, target_row in zip(operator.coefficients_mm, expected)
        for actual, target in zip(actual_row, target_row)
    )
    projected = project_spherical_number_densities(operator, (2.0, 3.0))
    inverse = solve_spherical_number_densities(
        operator,
        projected,
        upper_tail_assumption=UPPER_TAIL_ASSUMPTION,
    )
    print("Analytical two-class fixture")
    print("operator mm:", operator.coefficients_mm)
    print("projected N_A mm^-2:", projected)
    print("signed N_V mm^-3:", inverse.signed_nv_per_mm3)
    print("status:", inverse.status)
    print("residuals mm^-2:", inverse.residuals_per_mm2)


def end_to_end_example():
    fine_diameters = [0.15] * 20 + [0.3] * 10 + [0.6] * 5 + [1.2] * 2
    coarse_diameters = [0.15] * 180 + [0.3] * 120 + [0.6] * 40 + [1.2] * 10
    fine, fine_pores = _group_image("fine-image", 0, "fine", 100, fine_diameters)
    coarse, coarse_pores = _group_image(
        "coarse-image", 1, "coarse", 1000, coarse_diameters
    )
    dataset = build_sampling_dataset([fine, coarse], fine_pores + coarse_pores)
    source = calculate_2d_number_densities(
        dataset, create_diameter_bin_spec((0.1, 0.2, 0.4, 0.8, 1.6))
    )
    plan = NestingPlan(
        sample_id="example",
        start_bin=0,
        stop_bin=4,
        segments=(
            NestingSegment("fine", 0, 2),
            NestingSegment("coarse", 2, 4),
        ),
    )
    result = reconstruct_spherical_3d(
        source,
        plan,
        upper_tail_assumption=UPPER_TAIL_ASSUMPTION,
    )
    print("\nSampling -> distribution -> nesting -> reconstruction")
    print("input N_A mm^-2:", result.input_na_per_mm2)
    print("signed N_V mm^-3:", result.signed_nv_per_mm3)
    print("status:", result.status)
    print("tail assumption:", result.upper_tail_assumption)
    print("upper range truncated:", result.upper_range_truncated)
    print("residuals mm^-2:", result.residuals_per_mm2)
    print("compatibility:", result.compatibility_status)


def main():
    analytical_example()
    end_to_end_example()


if __name__ == "__main__":
    main()
