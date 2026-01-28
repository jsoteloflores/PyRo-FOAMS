# stereology.py
# Re-export shim for backward compatibility
# Actual implementation lives in core/stereology.py

from core.stereology import (
    PoreProps,
    colorize_labels,
    measure_labels,
    measure_dataset,
    save_props_csv,
    mask_from_labels,
)

__all__ = [
    "PoreProps",
    "colorize_labels",
    "measure_labels",
    "measure_dataset",
    "save_props_csv",
    "mask_from_labels",
]
