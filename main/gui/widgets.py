# widgets.py
# Shared GUI utilities for PyRo-FOAMS (debounce, busy dialogs, dtype helpers, etc.)

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional, Tuple

import numpy as np

# ============================================================================
# Dtype Normalization Helpers (R_dtype_contracts)
# ============================================================================

def ensure_mask_uint8(mask: Optional[np.ndarray], shape_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Normalize a mask to canonical uint8 {0, 255} format for OpenCV compatibility.

    Handles:
    - None → zeros
    - bool → uint8 * 255
    - uint8 with max <= 1 → * 255
    - any other dtype → threshold at nonzero → uint8 * 255
    - shape mismatch → resize with INTER_NEAREST

    Args:
        mask: Input mask (bool, uint8, or other)
        shape_hw: Target (H, W) shape. If None, keeps original shape.

    Returns:
        np.ndarray of dtype uint8, values in {0, 255}
    """
    import cv2  # local import to avoid circular deps

    if mask is None:
        if shape_hw is None:
            raise ValueError("ensure_mask_uint8: must provide shape_hw when mask is None")
        return np.zeros(shape_hw, dtype=np.uint8)

    m = mask

    # Convert to uint8 {0, 255}
    if m.dtype == np.bool_:
        m = m.astype(np.uint8) * 255
    elif m.dtype == np.uint8:
        if m.max() <= 1:
            m = m * 255
        # else already 0–255, assume threshold at 128 for "on"
        # (keep as-is; downstream should treat >0 as foreground)
    else:
        # Any other dtype: treat nonzero as foreground
        m = (m != 0).astype(np.uint8) * 255

    # Resize if shape mismatch
    if shape_hw is not None and m.shape[:2] != shape_hw:
        h, w = shape_hw
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

    return m


def ensure_labels_int32(labels: Optional[np.ndarray], shape_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Normalize labels to canonical int32 format with background=0.

    Handles:
    - None → zeros
    - Any int dtype → int32
    - Negative values (watershed boundaries) → set to 0
    - Shape mismatch → resize with INTER_NEAREST

    Args:
        labels: Input label image
        shape_hw: Target (H, W) shape. If None, keeps original shape.

    Returns:
        np.ndarray of dtype int32, background=0, objects=1..N
    """
    import cv2

    if labels is None:
        if shape_hw is None:
            raise ValueError("ensure_labels_int32: must provide shape_hw when labels is None")
        return np.zeros(shape_hw, dtype=np.int32)

    lab = labels.astype(np.int32, copy=False)

    # Normalize watershed boundaries (-1) to background
    lab[lab < 0] = 0

    # Resize if needed
    if shape_hw is not None and lab.shape[:2] != shape_hw:
        h, w = shape_hw
        lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)

    return lab


# ============================================================================
# Debounce Helper (R_debounce_ui)
# ============================================================================


def debounce(widget: tk.Misc, delay_ms: int = 150) -> Callable:
    """
    Decorator factory that debounces a function call using Tk's after().

    Collapses rapid successive calls (e.g., from <Configure> during resize)
    into a single delayed invocation. Each new call cancels any pending one.

    Usage:
        @debounce(self.canvas, 150)
        def _on_resize(self, event):
            ...

        # Or inline:
        canvas.bind("<Configure>", debounce(canvas, 150)(self._render))
    """
    def _decorator(fn: Callable) -> Callable:
        timer_id_attr = f"_debounce_{id(fn)}"

        def _wrapper(*args: Any, **kwargs: Any) -> None:
            # Cancel any pending call
            prev_id = getattr(widget, timer_id_attr, None)
            if prev_id is not None:
                try:
                    widget.after_cancel(prev_id)
                except Exception:
                    pass
            # Schedule the new call
            new_id = widget.after(delay_ms, lambda: fn(*args, **kwargs))
            setattr(widget, timer_id_attr, new_id)

        return _wrapper
    return _decorator


class BusyDialog(tk.Toplevel):
    """
    Modal progress dialog to block UI during long operations.

    Usage:
        dlg = BusyDialog(parent, "Processing...", mode="determinate", maximum=100)
        for i, item in enumerate(items):
            process(item)
            dlg.set_progress(i + 1)
        dlg.close()
    """
    def __init__(self, parent: tk.Misc, title: str = "Working…",
                 mode: str = "indeterminate", maximum: int = 100):
        if not isinstance(parent, tk.Misc):
            parent = tk._get_default_root()
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: None)

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=title).pack(anchor="w", pady=(0, 8))
        self.pb = ttk.Progressbar(frm, orient="horizontal", length=360,
                                  mode=mode, maximum=maximum)
        self.pb.pack(fill="x")
        if mode == "indeterminate":
            self.pb.start(10)
        self.update_idletasks()

    def set_progress(self, value: int) -> None:
        """Update determinate progress value."""
        try:
            self.pb["value"] = value
            self.update_idletasks()
        except Exception:
            pass

    def close(self) -> None:
        """Stop progressbar, release grab, destroy dialog."""
        try:
            if str(self.pb["mode"]) == "indeterminate":
                self.pb.stop()
        except Exception:
            pass
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()
