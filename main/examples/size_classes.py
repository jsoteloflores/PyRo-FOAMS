"""Runnable headless example for diameter-bin counts and two-dimensional N_A."""

from dataclasses import replace

import numpy as np

from main.core import (
    build_sampling_dataset,
    calculate_2d_number_densities,
    create_diameter_bin_spec,
    create_image_sampling_record,
    measure_labels,
)


def _image(image_id, image_index, width, minimum, diameters):
    labels = np.zeros((100, width), dtype=np.int32)
    for label in range(1, len(diameters) + 1):
        labels[label, label] = label
    image = create_image_sampling_record(
        sample_id="example",
        image_id=image_id,
        image_index=image_index,
        magnification_group_id="survey",
        label_map=labels,
        calibration=0.01,
        calibration_unit="mm",
        min_detectable_diameter=minimum,
        max_reliable_diameter=0.4,
    )
    pores = [
        replace(pore, eq_diam_px=diameters[pore.label - 1] / 0.01)
        for pore in measure_labels(labels, image_index=image_index)
    ]
    return image, pores


def main():
    one, one_pores = _image("one-mm2", 0, 100, 0.15, [0.15, 0.3])
    three, three_pores = _image("three-mm2", 1, 300, 0.2, [0.3, 0.3])
    dataset = build_sampling_dataset([one, three], one_pores + three_pores)
    result = calculate_2d_number_densities(
        dataset, create_diameter_bin_spec([0.1, 0.2, 0.4])
    )
    group = result.groups[("example", "survey")]
    print("counts:", group.counts)
    print("eligible area (mm^2):", group.eligible_areas_mm2)
    print("N_A (mm^-2):", group.number_densities_per_mm2)
    print("supported:", group.supported)


if __name__ == "__main__":
    main()
