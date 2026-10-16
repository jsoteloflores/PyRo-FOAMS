"""Write a reproducible FOAMS table bundle from synthetic segmented images."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from main.core import (
    FOAMS105_PERIMETER_PROFILE_CURRENT,
    Foams105AnalysisSettings,
    Foams105BinnedRange,
    Foams105PipelineGroupInput,
    Foams105PreparationSettings,
    Foams105SegmentedImageInput,
    Foams105SourceMetadata,
    analyze_foams105_prepared,
    prepare_foams105_analysis,
    write_foams105_export_bundle,
)


def _synthetic_inputs():
    first = np.zeros((20, 20), dtype=np.uint8)
    first[3, 3] = 1
    first[8:10, 8:10] = 1
    first[4, 14] = 2

    second = np.zeros((20, 20), dtype=np.uint8)
    second[6:9, 5:8] = 1

    return (
        Foams105PipelineGroupInput(
            group_id="fine",
            source_slot=1,
            images=(
                Foams105SegmentedImageInput(
                    image_id="synthetic-first",
                    source_image_key="first",
                    segmented_image=first,
                    scale_px_per_mm=1.0,
                    pore_phase_value=1,
                    excluded_phase_values=(2, None, None, None),
                    excluded_min_area_mm2=(0.0, 0.0, 0.0, 0.0),
                ),
                Foams105SegmentedImageInput(
                    image_id="synthetic-second",
                    source_image_key="second",
                    segmented_image=second,
                    scale_px_per_mm=1.0,
                    pore_phase_value=1,
                    excluded_phase_values=(2, None, None, None),
                    excluded_min_area_mm2=(0.0, 0.0, 0.0, 0.0),
                ),
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_directory",
        nargs="?",
        type=Path,
        default=None,
        help="new directory for the deterministic TSV bundle (default: a unique temporary path)",
    )
    args = parser.parse_args()
    output_directory = args.output_directory
    if output_directory is None:
        output_directory = Path(tempfile.mkdtemp(prefix="foams105_export_example_")) / "bundle"

    prepared = prepare_foams105_analysis(
        _synthetic_inputs(),
        Foams105PreparationSettings(
            minimum_diameter_px=0.5,
            perimeter_profile=FOAMS105_PERIMETER_PROFILE_CURRENT,
        ),
        source_metadata=Foams105SourceMetadata(
            density_derived_vesicularity_percent=25.0,
            pyro_revision="synthetic-example",
        ),
    )
    all_labels = prepared.label_preparation_result.bin_labels_mm
    result = analyze_foams105_prepared(
        prepared,
        Foams105AnalysisSettings(
            selection_mode="manual",
            manual_ranges=(Foams105BinnedRange("fine", all_labels[0], all_labels[-1]),),
            volume_normalization_mode="unnormalized",
            known_vesicularity_percent=None,
            size_lower_cutoff_mm=0.5,
            size_upper_cutoff_mm=1.0,
            shape_minimum_diameter_px=2.0,
            shape_statistics_mode="updated_minimum",
        ),
    )
    bundle = write_foams105_export_bundle(result, output_directory)
    if not bundle.complete:
        raise RuntimeError("synthetic example unexpectedly produced an incomplete bundle")

    print("FOAMS 1.0.5 synthetic analysis/export report")
    print("images:", result.image_count)
    print("pre-nesting diameter-filtered objects:", len(result.raw_data_rows))
    print("post-nesting objects:", result.raw_selection_result.selected_count)
    print("shape threshold identities:", result.shape_statistics_result.threshold_identities)
    print("export tables:", bundle.table_filenames)
    print("sidecars:", bundle.sidecar_filenames)
    print("bundle complete:", bundle.complete)
    print("output directory:", bundle.output_directory)
    print("boundary: synthetic segmented masks; no original-image parity")


if __name__ == "__main__":
    main()
