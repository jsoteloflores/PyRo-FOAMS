# main/__init__.py
# PyRo-FOAMS package root
"""
PyRo-FOAMS - Pore/Foam Image Analysis Toolkit

Subpackages:
    core - Pure algorithms (thresholding, watershed, measurements)
    gui  - Tkinter-based user interfaces

Quick start:
    python -m main          # Launch GUI

    # Or use core algorithms directly:
    from main.core import thresholdImageAdvanced, measure_labels
"""

# Re-export commonly used items from core for convenience
from .core import (
    # Calibrated sampling
    CANONICAL_LENGTH_UNIT,
    # Processing
    DEFAULTS,
    DETECTION_ELIGIBILITY_POLICY,
    NESTING_METHOD,
    AnalysisDomainError,
    DiameterBin,
    DiameterBinSpec,
    DistributionDiagnostics,
    DistributionResult,
    DistributionValidationError,
    Group2DDistribution,
    ImageSamplingRecord,
    MeasurementMappingError,
    Nested2DDistribution,
    NestingPlan,
    NestingSegment,
    NestingTransition,
    NestingValidationError,
    OverlapBinComparison,
    # Stereology
    PoreProps,
    SampledPore,
    SamplingDataset,
    SamplingGroupSummary,
    SamplingValidationError,
    TransitionOverlapDiagnostics,
    UnsupportedUnitError,
    # Preprocessing
    applyCropBatch,
    build_sampling_dataset,
    calculate_2d_number_densities,
    clampRectToImage,
    clearBorderTouching,
    colorize_labels,
    create_diameter_bin_spec,
    create_image_sampling_record,
    cropWithMargins,
    cropWithRect,
    fillHoles,
    geometric_diameter_bin_spec,
    labelsToColor,
    length_to_mm,
    loadImage,
    marginsToRect,
    mask_from_labels,
    # Batch
    measure_batch,
    measure_dataset,
    measure_labels,
    nest_2d_distribution,
    normalize_length_unit,
    postSeparateCleanup,
    process_batch_parallel,
    process_batch_sequential,
    rectToMargins,
    removeSmallAreas,
    runSeparationPipeline,
    save_props_csv,
    summarize_sampling_groups,
    threshold_batch,
    thresholdImageAdvanced,
    watershedSeparate,
)

__all__ = [
    # Processing
    "DEFAULTS",
    "thresholdImageAdvanced",
    "fillHoles",
    "removeSmallAreas",
    "clearBorderTouching",
    "watershedSeparate",
    "postSeparateCleanup",
    "labelsToColor",
    "runSeparationPipeline",
    # Stereology
    "PoreProps",
    "colorize_labels",
    "measure_labels",
    "measure_dataset",
    "save_props_csv",
    "mask_from_labels",
    # Calibrated sampling
    "CANONICAL_LENGTH_UNIT",
    "SamplingValidationError",
    "UnsupportedUnitError",
    "AnalysisDomainError",
    "MeasurementMappingError",
    "ImageSamplingRecord",
    "SampledPore",
    "SamplingDataset",
    "SamplingGroupSummary",
    "normalize_length_unit",
    "length_to_mm",
    "create_image_sampling_record",
    "build_sampling_dataset",
    "summarize_sampling_groups",
    # 2D distributions
    "DETECTION_ELIGIBILITY_POLICY",
    "DistributionValidationError",
    "DiameterBin",
    "DiameterBinSpec",
    "DistributionDiagnostics",
    "Group2DDistribution",
    "DistributionResult",
    "create_diameter_bin_spec",
    "geometric_diameter_bin_spec",
    "calculate_2d_number_densities",
    # Manual magnification nesting
    "NESTING_METHOD",
    "NestingValidationError",
    "NestingSegment",
    "NestingPlan",
    "NestingTransition",
    "OverlapBinComparison",
    "TransitionOverlapDiagnostics",
    "Nested2DDistribution",
    "nest_2d_distribution",
    # Preprocessing
    "loadImage",
    "clampRectToImage",
    "rectToMargins",
    "marginsToRect",
    "cropWithRect",
    "cropWithMargins",
    "applyCropBatch",
    # Batch
    "process_batch_parallel",
    "process_batch_sequential",
    "threshold_batch",
    "measure_batch",
]
