# main/core/__init__.py
# Core algorithms package - pure callables with no GUI dependencies
# Safe for headless testing, multiprocessing, and parallelism

from .batch import (
    measure_batch,
    process_batch_parallel,
    process_batch_sequential,
    threshold_batch,
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
