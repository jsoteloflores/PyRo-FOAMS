# preprocessing.py
# Re-export shim for backward compatibility
# Actual implementation lives in core/preprocessing.py

from core.preprocessing import (
    loadImage,
    clampRectToImage,
    rectToMargins,
    marginsToRect,
    cropWithRect,
    cropWithMargins,
    applyCropBatch,
)

# For backward compatibility with GUI that expects processing_gui import
# (The import statement in old code was likely unused or circular)

__all__ = [
    "loadImage",
    "clampRectToImage",
    "rectToMargins",
    "marginsToRect",
    "cropWithRect",
    "cropWithMargins",
    "applyCropBatch",
]
