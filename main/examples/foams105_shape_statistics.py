"""Run the accepted measurement-to-shape-statistics path on a small image."""

from __future__ import annotations

import numpy as np

from main.core import (
    Foams105BinnedRange,
    Foams105RawGroupInput,
    Foams105RawObject,
    derive_foams105_shapes,
    measure_foams105_moment_geometry,
    measure_foams105_perimeter,
    measure_foams105_phase,
    select_foams105_raw_objects,
    summarize_foams105_shapes,
)


def main():
    image = np.zeros((12, 12), dtype=np.uint8)
    image[2, 2] = 1
    image[5:7, 5:7] = 1

    measurement = measure_foams105_phase(image, 1, 1.0)
    moments = measure_foams105_moment_geometry("shape-example", measurement)
    perimeter = measure_foams105_perimeter(
        "shape-example",
        measurement,
        profile="matlab_perimeter_r2025b_verified_v1",
    )
    objects = tuple(
        Foams105RawObject(
            "shape-example",
            component.component_id,
            component.equivalent_diameter_mm,
            component.area_mm2,
        )
        for component in measurement.components
    )
    raw = select_foams105_raw_objects(
        (0.1, 1.0, 3.0),
        (Foams105RawGroupInput("sample", 1, objects),),
        (Foams105BinnedRange("sample", 1.0, 3.0),),
    )
    shapes = derive_foams105_shapes(raw, (moments,), (perimeter,))

    initial = summarize_foams105_shapes(
        shapes,
        configured_scales_px_per_mm=(1.0, 2.0),
        minimum_diameter_px=3.0,
        mode="initial_analysis",
    )
    updated = summarize_foams105_shapes(
        shapes,
        configured_scales_px_per_mm=(1.0, 2.0),
        minimum_diameter_px=3.0,
        mode="updated_minimum",
    )

    singleton = next(row for row in shapes.rows if row.area_px == 1)
    if singleton.perimeter_mm != 0.0 or singleton.form_factor.legacy_class != "positive_infinity":
        raise RuntimeError("singleton perimeter/form-factor classification differs")
    if initial.summary_population_count != len(shapes.rows):
        raise RuntimeError("initial statistics must include all post-nesting rows")
    if updated.summary_population_count != updated.threshold_count:
        raise RuntimeError("updated statistics must use the threshold-selected rows")
    if (singleton.image_id, singleton.component_id) in updated.threshold_identities:
        raise RuntimeError("the singleton should be excluded by the configured minimum")

    print("FOAMS 1.0.5 derived shape/statistics example")
    print("perimeter profile:", shapes.perimeter_profile)
    print("threshold mm:", initial.threshold_mm)
    print("all shape identities:", shapes.selected_identities)
    print("strict threshold identities:", initial.threshold_identities)
    print("initial summary population:", initial.summary_population_identities)
    print("updated summary population:", updated.summary_population_identities)
    for row in shapes.rows:
        print(
            "component",
            row.component_id,
            "regularity",
            row.regularity.value,
            "form_factor",
            row.form_factor.status,
            row.form_factor.legacy_class or row.form_factor.value,
            "aspect_ratio",
            row.aspect_ratio.value,
        )
    print("initial aspect-ratio statistics:", initial.statistic_rows[0].display_mean)
    print("updated aspect-ratio statistics:", updated.statistic_rows[0].display_mean)
    print("boundary: shape histograms and Shape/Stats writers are deferred to Brief 18")


if __name__ == "__main__":
    main()
