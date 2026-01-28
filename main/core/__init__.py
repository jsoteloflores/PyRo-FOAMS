# main/core/__init__.py
# Core algorithms package - pure callables with no GUI dependencies
# Safe for headless testing, multiprocessing, and parallelism

from .processing import (
    DEFAULTS,
    thresholdImageAdvanced,
    fillHoles,
    removeSmallAreas,
    clearBorderTouching,
    watershedSeparate,
    postSeparateCleanup,
    labelsToColor,
    runSeparationPipeline,
)

from .stereology import (
    PoreProps,
    colorize_labels,
    measure_labels,
    measure_dataset,
    save_props_csv,
    mask_from_labels,
)

from .preprocessing import (
    loadImage,
    clampRectToImage,
    rectToMargins,
    marginsToRect,
    cropWithRect,
    cropWithMargins,
    applyCropBatch,
)

from .batch import (
    process_batch_parallel,
    process_batch_sequential,
    threshold_batch,
    measure_batch,
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
