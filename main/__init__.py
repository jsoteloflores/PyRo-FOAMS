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
    # Processing
    DEFAULTS,
    # Stereology
    PoreProps,
    applyCropBatch,
    clampRectToImage,
    clearBorderTouching,
    colorize_labels,
    cropWithMargins,
    cropWithRect,
    fillHoles,
    labelsToColor,
    # Preprocessing
    loadImage,
    marginsToRect,
    mask_from_labels,
    measure_batch,
    measure_dataset,
    measure_labels,
    postSeparateCleanup,
    # Batch
    process_batch_parallel,
    process_batch_sequential,
    rectToMargins,
    removeSmallAreas,
    runSeparationPipeline,
    save_props_csv,
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
