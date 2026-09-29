# main/core/__init__.py
# Core algorithms package - pure callables with no GUI dependencies
# Safe for headless testing, multiprocessing, and parallelism

from .batch import (
    measure_batch,
    process_batch_parallel,
    process_batch_sequential,
    threshold_batch,
)
from .distributions import (
    DETECTION_ELIGIBILITY_POLICY,
    DiameterBin,
    DiameterBinSpec,
    DistributionDiagnostics,
    DistributionResult,
    DistributionValidationError,
    Group2DDistribution,
    calculate_2d_number_densities,
    create_diameter_bin_spec,
    geometric_diameter_bin_spec,
)
from .preprocessing import (
    applyCropBatch,
    clampRectToImage,
    cropWithMargins,
    cropWithRect,
    loadImage,
    marginsToRect,
    rectToMargins,
)
from .processing import (
    DEFAULTS,
    clearBorderTouching,
    fillHoles,
    labelsToColor,
    postSeparateCleanup,
    removeSmallAreas,
    runSeparationPipeline,
    thresholdImageAdvanced,
    watershedSeparate,
)
from .sampling import (
    CANONICAL_LENGTH_UNIT,
    AnalysisDomainError,
    ImageSamplingRecord,
    MeasurementMappingError,
    SampledPore,
    SamplingDataset,
    SamplingGroupSummary,
    SamplingValidationError,
    UnsupportedUnitError,
    build_sampling_dataset,
    create_image_sampling_record,
    length_to_mm,
    normalize_length_unit,
    summarize_sampling_groups,
)
from .stereology import (
    PoreProps,
    colorize_labels,
    mask_from_labels,
    measure_dataset,
    measure_labels,
    save_props_csv,
)

__all__ = [
    # processing
    "DEFAULTS",
    "thresholdImageAdvanced",
    "fillHoles",
    "removeSmallAreas",
    "clearBorderTouching",
    "watershedSeparate",
    "postSeparateCleanup",
    "labelsToColor",
    "runSeparationPipeline",
    # stereology
    "PoreProps",
    "colorize_labels",
    "measure_labels",
    "measure_dataset",
    "save_props_csv",
    "mask_from_labels",
    # calibrated sampling
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
    # preprocessing
    "loadImage",
    "clampRectToImage",
    "rectToMargins",
    "marginsToRect",
    "cropWithRect",
    "cropWithMargins",
    "applyCropBatch",
    # batch parallel
    "process_batch_parallel",
    "process_batch_sequential",
    "threshold_batch",
    "measure_batch",
]
