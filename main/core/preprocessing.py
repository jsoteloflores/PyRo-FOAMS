# main/core/preprocessing.py
# Preprocessing utilities - pure callables with no GUI dependencies
# Safe for headless testing, multiprocessing, and parallelism

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Union

# Type aliases for clarity
Rect = Tuple[int, int, int, int]  # (x0, y0, x1, y1) inclusive-exclusive style
Margins = Tuple[int, int, int, int]  # (left, top, right, bottom)
ImageArray = np.ndarray  # np.uint8, shape (H,W) grayscale or (H,W,3) BGR
ShapeHW = Tuple[int, int]  # (height, width)


def loadImage(path: str, asGray: bool = True) -> ImageArray:
    """Load an image with OpenCV. Returns np.uint8.
    If asGray=True, loads grayscale; else returns BGR color."""
    flag = cv2.IMREAD_GRAYSCALE if asGray else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def clampRectToImage(rect: Rect, shape: ShapeHW) -> Optional[Rect]:
    """Clamp (x0,y0,x1,y1) to image bounds. Returns None if rect is degenerate."""
    h, w = shape[:2]
    x0, y0, x1, y1 = rect
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(1, min(x1, w))
    y1 = max(1, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def rectToMargins(rect: Rect, shape: ShapeHW) -> Margins:
    """Convert an absolute rect to margins (left, top, right, bottom) for the given image shape."""
    h, w = shape[:2]
    x0, y0, x1, y1 = rect
    left = x0
    top = y0
    right = w - x1
    bottom = h - y1
    left = int(max(0, left))
    top = int(max(0, top))
    right = int(max(0, right))
    bottom = int(max(0, bottom))
    return (left, top, right, bottom)


def marginsToRect(margins: Margins, shape: ShapeHW) -> Optional[Rect]:
    """Convert margins back to a rect for the given image shape."""
    h, w = shape[:2]
    left, top, right, bottom = margins
    x0 = int(max(0, left))
    y0 = int(max(0, top))
    x1 = int(min(w, w - max(0, right)))
    y1 = int(min(h, h - max(0, bottom)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def cropWithRect(img: ImageArray, rect: Rect) -> ImageArray:
    """Crop using absolute rect (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = rect
    return img[y0:y1, x0:x1].copy()


def cropWithMargins(img: ImageArray, margins: Margins) -> Optional[ImageArray]:
    """Crop using margins. Returns None if margins invalid for this size."""
    rect = marginsToRect(margins, img.shape)
    if rect is None:
        return None
    return cropWithRect(img, rect)


def applyCropBatch(
    images: List[ImageArray],
    rect: Optional[Rect] = None,
    margins: Optional[Margins] = None,
    useMargins: bool = True
) -> List[Optional[ImageArray]]:
    """Apply crop across a batch. If useMargins=True, rect is converted to margins from the
    first image, then applied to all (robust to varying sizes). If useMargins=False, rect is
    clamped per image."""
    if not images:
        return []
    if rect is None and margins is None:
        raise ValueError("applyCropBatch: either rect or margins must be provided")

    out: List[Optional[np.ndarray]] = []
    # Build margins if only rect provided
    if useMargins:
        if margins is None:
            if rect is None:
                raise ValueError("applyCropBatch: need rect or margins")
            baseRect = clampRectToImage(rect, images[0].shape)
            if baseRect is None:
                return [None for _ in images]
            margins = rectToMargins(baseRect, images[0].shape)
        for img in images:
            out.append(cropWithMargins(img, margins))
    else:
        # Apply rect per image with clamping
        if rect is None:
            # If only margins provided but useMargins=False, convert margins per image
            for img in images:
                r = marginsToRect(margins, img.shape)  # type: ignore
                out.append(cropWithRect(img, r) if r else None)
        else:
            for img in images:
                r = clampRectToImage(rect, img.shape)
                out.append(cropWithRect(img, r) if r else None)
    return out
