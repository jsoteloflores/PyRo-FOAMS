"""Report bounded FOAMS 1.0.5 density and size-category calculations.

Analytical expectations are supplied calculations, not original workbook
references. Only the 29 input rows and converter outputs have workbook parity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from main.core import (
    Foams105BinnedRange,
    Foams105CountGroupInput,
    Foams105RawGroupInput,
    Foams105RawObject,
    Foams105VesicularityGroupInput,
    calculate_foams105_2d_vesicularity,
    calculate_foams105_image_areas,
    calculate_foams105_volume_distribution,
    convert_foams105_nv,
    measure_foams105_image,
    select_foams105_raw_objects,
    summarize_foams105_distribution,
    summarize_foams105_size_categories,
)
from main.core.legacy_summaries import _calculate_foams105_distribution_numerical

FIXTURE_DIR = Path(__file__).parents[1] / "tests" / "fixtures"
SUMMARY_FIXTURE_PATH = FIXTURE_DIR / "15_foams105_summary_cases.json"
WORKBOOK_FIXTURE_PATH = FIXTURE_DIR / "06_foams105_nv_reference.json"


def _require_close(case_id, field, candidate, expected):
    if not math.isclose(candidate, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"Case {case_id} {field} differs: {candidate!r} != {expected!r}"
        )


def _require_sequence(case_id, field, candidate, expected):
    if len(candidate) != len(expected):
        raise ValueError(f"Case {case_id} {field} length differs")
    for index, (value, reference) in enumerate(zip(candidate, expected)):
        _require_close(case_id, f"{field}[{index}]", value, reference)


def analytical_report():
    fixture = json.loads(SUMMARY_FIXTURE_PATH.read_text(encoding="utf-8"))
    distribution_cases = fixture["distribution_cases"]
    category_cases = fixture["category_cases"]
    if (len(distribution_cases), len(category_cases)) != (5, 4):
        raise ValueError("Summary fixture must contain five and four cases")

    undefined_values = 0
    for case in distribution_cases:
        result = _calculate_foams105_distribution_numerical(
            case["global_labels"],
            case["labels"],
            case["nv"],
            case["adjusted"],
            case["correction_percent"],
        )
        _require_sequence(case["id"], "lower edges", result.lower_edges_mm, case["lower_edges"])
        _require_sequence(case["id"], "widths", result.widths_mm, case["widths"])
        _require_sequence(case["id"], "midsizes", result.midsizes_mm, case["midsizes"])
        _require_sequence(
            case["id"], "cumulative N_V", result.cumulative_nv_per_mm3, case["cumulative_nv"]
        )
        for record, label in zip(result.log10_labels, case["labels"]):
            _require_close(case["id"], "log10 label", record.value, math.log10(label))
        for fixture_field, records, reasons_field in (
            ("per_width", result.number_density_per_width_mm4, None),
            ("ln_per_width", result.ln_number_density_per_width, "ln_reasons"),
            ("log10_cumulative", result.log10_cumulative_nv, "log10_cumulative_reasons"),
        ):
            if fixture_field not in case:
                continue
            for index, (record, expected) in enumerate(zip(records, case[fixture_field])):
                if expected is None:
                    if record.value is not None:
                        raise ValueError(f"Case {case['id']} {fixture_field}[{index}] is defined")
                else:
                    _require_close(case["id"], f"{fixture_field}[{index}]", record.value, expected)
            if reasons_field in case and tuple(item.reason for item in records) != tuple(case[reasons_field]):
                raise ValueError(f"Case {case['id']} {reasons_field} differs")
        _require_close(case["id"], "total N_V", result.total_nv_per_mm3, case["total_nv"])
        if case["corrected_nv"] is None:
            if result.corrected_nv_per_mm3.reason != case["corrected_reason"]:
                raise ValueError(f"Case {case['id']} corrected status differs")
        else:
            _require_close(
                case["id"], "corrected N_V", result.corrected_nv_per_mm3.value, case["corrected_nv"]
            )
        _require_sequence(
            case["id"],
            "paired fractions",
            tuple(item.adjusted_volume_fraction for item in result.paired_volume_bins),
            case["paired_fractions"],
        )
        if tuple(item.bin_label_mm for item in result.paired_volume_bins) != tuple(case["paired_labels"]):
            raise ValueError(f"Case {case['id']} paired labels differ")
        undefined_values += sum(
            item.value is None
            for values in (
                result.log10_labels,
                result.log10_cumulative_nv,
                result.number_density_per_width_mm4,
                result.ln_number_density_per_width,
                (result.corrected_nv_per_mm3,),
            )
            for item in values
        )

    for case in category_cases:
        groups = tuple(
            Foams105CountGroupInput(f"group-{index}", index, tuple(counts))
            for index, counts in enumerate(case["counts"], 1)
        )
        result = summarize_foams105_size_categories(
            case["global_labels"], groups, case["lower"], case["upper"]
        )
        if result.total_count != case["total"]:
            raise ValueError(f"Case {case['id']} total differs")
        if (
            result.lower_count,
            result.middle_count,
            result.upper_count,
        ) != tuple(case["category_counts"]):
            raise ValueError(f"Case {case['id']} category counts differ")
        if result.boundary_count != case["boundary_count"]:
            raise ValueError(f"Case {case['id']} boundary count differs")
        percentages = (
            result.lower_percentage,
            result.middle_percentage,
            result.upper_percentage,
        )
        for index, (record, expected) in enumerate(zip(percentages, case["percentages"])):
            if expected is None:
                if record.value is not None or record.reason != case["reason"]:
                    raise ValueError(f"Case {case['id']} percentage[{index}] status differs")
                undefined_values += 1
            else:
                _require_close(case["id"], f"percentage[{index}]", record.value, expected)

    print("FOAMS 1.0.5 analytical summary report")
    print("distribution cases:", len(distribution_cases))
    print("category cases:", len(category_cases))
    print("explicit undefined values:", undefined_values)
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
    conversion = convert_foams105_nv((1.0, 2.0, 4.0), (8.0, 3.0, 1.0))
    vesicularity = _synthetic_vesicularity()
    volumes = (
        calculate_foams105_volume_distribution(conversion),
        calculate_foams105_volume_distribution(
            conversion, "integrated_2d", vesicularity_result=vesicularity
        ),
        calculate_foams105_volume_distribution(
            conversion, "known_vesicularity", known_vesicularity_percent=40.0
        ),
    )
    summaries = (
        summarize_foams105_distribution(
            volumes[0], conversion.bin_labels_mm, vesicularity_result=vesicularity
        ),
        summarize_foams105_distribution(volumes[1], conversion.bin_labels_mm),
        summarize_foams105_distribution(volumes[2], conversion.bin_labels_mm),
    )
    if summaries[0].correction_source != "integrated_2d_vesicularity_percent":
        raise ValueError("Unnormalized summary did not acquire integrated 2D correction")
    if tuple(item.volume_result.mode for item in summaries) != (
        "unnormalized",
        "integrated_2d",
        "known_vesicularity",
    ):
        raise ValueError("Public mode sequence differs")

    print("\nPublic conversion -> volume -> density summaries")
    for result in summaries:
        print(
            result.volume_result.mode,
            "correction percent:",
            result.correction_percent,
            "corrected N_V:",
            result.corrected_nv_per_mm3.value,
        )


def workbook_selection_report():
    fixture = json.loads(WORKBOOK_FIXTURE_PATH.read_text(encoding="utf-8"))
    conversion = convert_foams105_nv(
        fixture["bin_labels_mm"], fixture["na_per_mm2"]
    )
    _require_sequence(
        "workbook",
        "converter N_V",
        conversion.signed_nv_per_mm3,
        fixture["expected_nv_per_mm3"],
    )
    volume = calculate_foams105_volume_distribution(
        conversion, "known_vesicularity", known_vesicularity_percent=40.0
    )
    global_labels = tuple(dict.fromkeys(conversion.bin_labels_mm))
    summary = summarize_foams105_distribution(volume, global_labels)
    zero_width_rows = tuple(
        row.output_index for row in summary.rows if row.width_mm == 0.0
    )
    if len(summary.rows) != 29 or zero_width_rows != (15,):
        raise ValueError("Workbook selection duplicate-width status differs")

    print("\nPinned 29-row workbook selection")
    print("converter rows:", len(summary.rows))
    print("adjacent duplicate-label pairs:", conversion.adjacent_duplicate_label_index_pairs)
    print("explicit zero-width rows:", zero_width_rows)
    print("boundary: downstream summaries are calculated, not workbook parity values")


def main():
    analytical_report()
    public_api_report()
    workbook_selection_report()


if __name__ == "__main__":
    main()
