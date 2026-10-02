"""Runnable headless example of explicit manual magnification nesting."""

from dataclasses import replace

import numpy as np

from main.core import (
    NestingPlan,
    NestingSegment,
    build_sampling_dataset,
    calculate_2d_number_densities,
    create_diameter_bin_spec,
    create_image_sampling_record,
    measure_labels,
    nest_2d_distribution,
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


def main():
    fine_diameters = [0.15] * 20 + [0.3] * 10 + [0.6] * 5 + [1.2] * 2
    coarse_diameters = [0.15] * 180 + [0.3] * 120 + [0.6] * 40 + [1.2] * 10
    fine, fine_pores = _group_image("fine-image", 0, "fine", 100, fine_diameters)
    coarse, coarse_pores = _group_image(
        "coarse-image", 1, "coarse", 1000, coarse_diameters
    )
    dataset = build_sampling_dataset(
        [fine, coarse], fine_pores + coarse_pores
    )
    source = calculate_2d_number_densities(
        dataset, create_diameter_bin_spec([0.1, 0.2, 0.4, 0.8, 1.6])
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
    composite = nest_2d_distribution(source, plan)

    print("bin | interval mm | group | count | eligible area mm^2 | N_A mm^-2")
    for row in zip(
        composite.selected_bin_indices,
        composite.lower_edges_mm,
        composite.upper_edges_mm,
        composite.source_group_ids,
        composite.counts,
        composite.eligible_areas_mm2,
        composite.number_densities_per_mm2,
    ):
        index, lower, upper, group, count, area, density = row
        print(
            f"{index:>3} | [{lower:g}, {upper:g}] | {group:>6} | "
            f"{count:>5} | {area:>18g} | {density:g}"
        )

    for diagnostic in composite.overlap_diagnostics:
        transition = diagnostic.transition
        print(
            f"transition {transition.left_group_id} -> {transition.right_group_id} "
            f"at edge {transition.edge_index} ({transition.diameter_mm:g} mm)"
        )
        print(
            "flags:",
            {
                "no_shared_supported_bins": diagnostic.no_shared_supported_bins,
                "no_informative_overlap": diagnostic.no_informative_overlap,
                "transition_not_bracketed_by_shared_support": (
                    diagnostic.transition_not_bracketed_by_shared_support
                ),
            },
        )


if __name__ == "__main__":
    main()
