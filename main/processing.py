# processing.py
# Re-export shim for backward compatibility
# Actual implementation lives in core/processing.py
#
# GUI modules and tests can continue importing from here,
# but the core algorithms are now in a pure module safe for
# headless testing and multiprocessing.

from core.processing import (
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

__all__ = [
    "DEFAULTS",
    "thresholdImageAdvanced",
    "fillHoles",
    "removeSmallAreas",
    "clearBorderTouching",
    "watershedSeparate",
    "postSeparateCleanup",
    "labelsToColor",
    "runSeparationPipeline",
]
