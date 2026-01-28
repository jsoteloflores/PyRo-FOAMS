# main/gui/__init__.py
# GUI package - all tkinter-based user interfaces
"""
PyRo-FOAMS GUI modules.

Submodules:
    preprocessing - Image loading, cropping, batch operations
    processing    - Thresholding and watershed separation
    stereology    - Measurements, histograms, CSV export
    postprocessing - Mask editing and refinement
    widgets       - Shared GUI utilities (debounce, BusyDialog, etc.)
"""

from .widgets import BusyDialog, debounce, ensure_labels_int32, ensure_mask_uint8

__all__ = [
    "debounce",
    "ensure_mask_uint8",
    "ensure_labels_int32",
    "BusyDialog",
]
