"""Report bounded FOAMS 1.0.5 equivalent-sphere volume calculations.

The fixture expectations are calculated values supplied for this stage, not
original workbook references. The public example uses one synthetic mask.
"""

import json
import math
from pathlib import Path

import numpy as np

from main.core import (
    Foams105BinnedRange,
    Foams105RawGroupInput,
    Foams105RawObject,
    Foams105VesicularityGroupInput,
    calculate_foams105_2d_vesicularity,
    calculate_foams105_image_areas,
    calculate_foams105_volume_distribution,
    convert_foams105_nv,
    measure_foams105_image,
    select_foams105_raw_objects,
)
from main.core.legacy_volumes import (
    Foams105VolumeDomainError,
    Foams105VolumeNumericalError,
    _calculate_foams105_volume_numerical,
)

FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "tests"
    / "fixtures"
    / "14_foams105_volume_cases.json"
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
    if len(fixture["cases"]) != 10:
        raise ValueError("Volume fixture must contain exactly 10 cases")
    successful = 0
    expected_rejections = 0
    for case in fixture["cases"]:
        try:
            result = _calculate_foams105_volume_numerical(
                case["heights"],
                case["nv"],
                case["mode"],
                target_vesicularity_percent=case.get("target_percent"),
            )
        except (Foams105VolumeDomainError, Foams105VolumeNumericalError) as exc:
            if exc.code != case.get("expected_error"):
                raise ValueError(
                    f"Case {case['id']} produced unexpected error {exc.code}"
                ) from exc
            expected_rejections += 1
            continue
        if "expected_error" in case:
            raise ValueError(f"Case {case['id']} unexpectedly succeeded")
        _require_close_sequence(
            case["id"], "raw", result.raw_volume_fractions, case["raw"]
        )
        _require_close(case["id"], "sum", result.raw_fraction_sum, case["sum"])
        _require_close(
            case["id"], "factor", result.normalization_factor, case["factor"]
        )
        _require_close_sequence(
            case["id"],
            "adjusted",
            result.adjusted_volume_fractions,
            case["adjusted"],
        )
        _require_close_sequence(
            case["id"],
            "cumulative",
            result.cumulative_adjusted_volume_fractions,
            case["cumulative"],
        )
        successful += 1
    if (successful, expected_rejections) != (8, 2):
        raise ValueError("Volume fixture outcome counts differ")

    print("FOAMS 1.0.5 analytical volume report")
    print("fixture cases:", len(fixture["cases"]))
    print("successful calculations:", successful)
    print("expected zero-sum rejections:", expected_rejections)
    print("mismatches: 0")


def _synthetic_vesicularity():
    image = np.zeros((10, 10), dtype=np.uint8)
    image[3:5, 3:5] = 1
    measurement = measure_foams105_image(
        image, 1, 1.0, (None, None, None, None)
    )
    area = calculate_foams105_image_areas("synthetic-image", measurement)
    objects = tuple(
        Foams105RawObject(
            "synthetic-image",
            component.component_id,
            component.equivalent_diameter_mm,
            component.area_mm2,
        )
        for component in measurement.pore_result.components
    )
    raw = select_foams105_raw_objects(
        (0.1, 1.0, 10.0),
        (Foams105RawGroupInput("synthetic", 1, objects),),
        (Foams105BinnedRange("synthetic", 0.1, 10.0),),
    )
    return calculate_foams105_2d_vesicularity(
        raw,
        (Foams105VesicularityGroupInput("synthetic", 1, (area,), objects),),
    )


def public_api_report():
    conversion = convert_foams105_nv(
        (1.0, 2.0, 2.0, 4.0),
        (1.0, 0.0, 2.0, 3.0),
    )
    vesicularity = _synthetic_vesicularity()
    unnormalized = calculate_foams105_volume_distribution(conversion)
    integrated = calculate_foams105_volume_distribution(
        conversion, "integrated_2d", vesicularity_result=vesicularity
    )
    known = calculate_foams105_volume_distribution(
        conversion,
        "known_vesicularity",
        known_vesicularity_percent=40.0,
    )

    if conversion.negative_indices != (1,):
        raise ValueError("Expected one signed negative converter row")
    if conversion.adjacent_duplicate_label_index_pairs != ((1, 2),):
        raise ValueError("Expected one adjacent duplicate-label pair")
    if tuple(row.signed_nv_per_mm3 for row in known.rows) != (
        conversion.signed_nv_per_mm3
    ):
        raise ValueError("Normalization changed canonical converter N_V")
    if unnormalized.adjusted_volume_fractions != unnormalized.raw_volume_fractions:
        raise ValueError("Unnormalized mode changed raw fractions")
    _require_close(
        "public", "integrated target", integrated.adjusted_fraction_sum, 0.04
    )
    _require_close("public", "known target", known.adjusted_fraction_sum, 0.4)

    print("\nPublic normalization modes")
    for result in (unnormalized, integrated, known):
        print(
            result.mode,
            "target percent:",
            result.target_vesicularity_percent,
            "factor:",
            result.normalization_factor,
            "adjusted sum:",
            result.adjusted_fraction_sum,
        )
    print("signed N_V rows:", conversion.negative_indices)
    print("duplicate-label row pairs:", conversion.adjacent_duplicate_label_index_pairs)
    print("boundary: no MATLAB, original-workbook, plotting, or full-workflow parity")


def main():
    analytical_report()
    public_api_report()


if __name__ == "__main__":
    main()
