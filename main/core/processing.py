# main/core/processing.py
# Core image processing algorithms - pure callables with no GUI dependencies
# Safe for headless testing, multiprocessing, and parallelism

from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List

# --------------------- Defaults (edit freely) ---------------------
DEFAULTS: Dict[str, Dict[str, float | int | bool | str]] = {
    "common": {
        "useCLAHE": False,
        "claheClip": 2.0,
        "claheTile": 8,
        "medianK": 3,            # 0 disables
        "gaussianK": 0,          # 0 disables
        "polarity": "auto",      # "auto" | "poresDarker" | "poresBrighter"
        "applyOpenClose": False,
        "morphK": 3,
    },
    "otsu": {},
    "adaptive": {
        "adaptiveBlock": 51,     # odd
        "adaptiveC": 2,
    },
    "percentile": {
        "percentile": 50.0,      # 0..100
    },
    "pick": {
        "pickTolerance": 10,     # +/- gray
        # pickValue chosen by eyedropper at runtime
    },
    # Separation defaults
    "separation": {
        "method": "watershed",   # "none" | "watershed"
        "fillHoles": True,       # fill internal holes before separation
        "minAreaPx": 30,         # remove specks < this area
        "distanceBlurK": 3,      # 0=off; smoothing for distance
        "peakMinDistance": 9,    # local-max suppression radius (px)
        "peakRelThreshold": 0.2, # 0..1 relative to max distance
        "connectivity": 8,       # 4 or 8 for CC and neighbor checks
        "clearBorder": False,    # drop objects touching image border
        "overlayAlpha": 0.45,    # overlay strength for previews (GUI)
    }
}
# ------------------------------------------------------------------

# --------------------- Internal helpers ---------------------------

def _forceOdd(k: int) -> int:
    k = int(max(1, k))
    return k if k % 2 == 1 else k + 1


def _prepGray(src) -> np.ndarray:
    img = src
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def _isDarkerForeground(img, binary):
    fg = img[binary == 255]; bg = img[binary == 0]
    if fg.size == 0 or bg.size == 0: return False
    return float(np.mean(fg)) < float(np.mean(bg))


def _contrastGap(img, binary):
    fg = img[binary == 255]; bg = img[binary == 0]
    if fg.size == 0 or bg.size == 0: return 0.0
    return abs(float(np.mean(fg)) - float(np.mean(bg)))


def _pickDarkerForeground(img, binA, binB):
    return binA if _isDarkerForeground(img, binA) else binB


def _pickBrighterForeground(img, binA, binB):
    return binA if not _isDarkerForeground(img, binA) else binB


def _choosePolarity(img, binA, binB, polarity: str):
    if polarity == "poresDarker":  return _pickDarkerForeground(img, binA, binB)
    if polarity == "poresBrighter":return _pickBrighterForeground(img, binA, binB)
    A_dark, B_dark = _isDarkerForeground(img, binA), _isDarkerForeground(img, binB)
    if A_dark and not B_dark: return binA
    if B_dark and not A_dark: return binB
    return binA if _contrastGap(img, binA) >= _contrastGap(img, binB) else binB


def _localMaxima(dist: np.ndarray, minDist: int, minVal: float) -> np.ndarray:
    """Binary map of local maxima using dilation-and-compare with min distance suppression."""
    if minDist < 1:
        minDist = 1
    k = int(minDist)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(dist, kernel)
    peaks = (dist == dil) & (dist >= minVal)
    return peaks.astype(np.uint8)


# --------------------- Thresholding -------------------------------

def thresholdImageAdvanced(
    gray,
    method: str = "otsu",              # "otsu" | "adaptive" | "percentile" | "pick"
    polarity: str = "auto",            # "auto" | "poresDarker" | "poresBrighter"
    useCLAHE: bool = False,
    claheClip: float = 2.0,
    claheTile: int = 8,
    medianK: int = 3,
    gaussianK: int = 0,
    adaptiveBlock: int = 51,
    adaptiveC: int = 2,
    percentile: float = 50.0,
    pickValue: Optional[int] = None,
    pickTolerance: int = 10,
    applyOpenClose: bool = False,
    morphK: int = 3
) -> Tuple[np.ndarray, Dict]:
    """
    Robust thresholding with preprocessing and polarity control.
    Returns (binary_uint8, meta_dict). Foreground (vesicles) = 255.
    """
    img = _prepGray(gray)

    # Preprocess
    if medianK and medianK >= 3:
        img = cv2.medianBlur(img, _forceOdd(medianK))
    if gaussianK and gaussianK >= 3:
        img = cv2.GaussianBlur(img, (_forceOdd(gaussianK), _forceOdd(gaussianK)), 0)
    if useCLAHE:
        clahe = cv2.createCLAHE(clipLimit=float(claheClip), tileGridSize=(int(claheTile), int(claheTile)))
        img = clahe.apply(img)

    meta = {"method": method, "polarity": polarity}

    if method == "otsu":
        _, binA = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binB = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = _choosePolarity(img, binA, binB, polarity)
    elif method == "adaptive":
        blk = max(3, _forceOdd(int(adaptiveBlock)))
        binA = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, blk, int(adaptiveC))
        binB = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, blk, int(adaptiveC))
        binary = _choosePolarity(img, binA, binB, polarity)
        meta["block"] = blk; meta["C"] = int(adaptiveC)
    elif method == "percentile":
        t = float(np.clip(percentile, 0, 100))
        thresh = float(np.percentile(img, t))
        _, binA = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        _, binB = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
        binary = _choosePolarity(img, binA, binB, polarity)
        meta["percentile"] = float(t); meta["thresh"] = float(thresh)
    elif method == "pick":
        if pickValue is None:
            raise ValueError("Color-pick method requires pickValue.")
        tol = int(max(0, pickTolerance))
        lo = max(0, int(pickValue) - tol)
        hi = min(255, int(pickValue) + tol)
        mask = (img >= lo) & (img <= hi)
        binary = (mask.astype(np.uint8) * 255)
        meta["pickValue"] = int(pickValue); meta["pickTolerance"] = tol
    else:
        raise ValueError(f"Unknown method: {method}")

    if applyOpenClose:
        k = _forceOdd(int(morphK))
        krn = np.ones((k, k), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, krn)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, krn)

    return binary.astype(np.uint8), meta


# -------------------------- Cleanup -------------------------------

def fillHoles(binary: np.ndarray) -> np.ndarray:
    """Fill internal holes in a binary mask (255=FG)."""
    mask = (binary > 0).astype(np.uint8)
    h, w = mask.shape
    flood = mask.copy()
    pad = cv2.copyMakeBorder(flood, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(pad, None, (0, 0), 1)  # fill from padded corner
    bg = pad[1:h+1, 1:w+1]
    holes = (bg == 0) & (mask == 0)
    out = mask.copy()
    out[holes] = 1
    return (out * 255).astype(np.uint8)


def removeSmallAreas(binary: np.ndarray, minArea: int, connectivity: int = 8) -> np.ndarray:
    """Remove connected components smaller than minArea. Keeps FG=255.
    
    Vectorized: uses connectedComponentsWithStats to get areas in O(n) instead
    of per-label loops, then applies a LUT-based remap.
    """
    if minArea <= 1:
        return binary
    mask = (binary > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    if num <= 1:
        return binary
    # stats[:,cv2.CC_STAT_AREA] gives area for each label (index 0 = background)
    areas = stats[:, cv2.CC_STAT_AREA]
    # Build a LUT: keep[lab] = 1 if area >= minArea, else 0
    keep_lut = (areas >= minArea).astype(np.uint8)
    keep_lut[0] = 0  # background always 0
    # Apply LUT to labels
    out = keep_lut[labels]
    return (out * 255).astype(np.uint8)


def clearBorderTouching(binary: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Remove components that touch the border.
    
    Vectorized: collect border labels via unique on border slices, then use
    np.isin for O(n) removal instead of per-label loops.
    """
    mask = (binary > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(mask, connectivity=connectivity)
    if num <= 1:
        return binary
    h, w = mask.shape
    # Collect labels touching any border edge (single vectorized gather)
    border_labels = np.unique(np.concatenate([
        labels[0, :].ravel(),
        labels[-1, :].ravel(),
        labels[:, 0].ravel(),
        labels[:, -1].ravel()
    ]))
    # Build a keep LUT: default keep=1, set border labels to 0
    keep_lut = np.ones(num, dtype=np.uint8)
    keep_lut[0] = 0  # background always excluded
    keep_lut[border_labels] = 0
    # Apply LUT
    out = keep_lut[labels]
    return (out * 255).astype(np.uint8)


# -------------------------- Separation ----------------------------

def watershedSeparate(
    binary: np.ndarray,
    distanceBlurK: int = 3,
    peakMinDistance: int = 9,
    peakRelThreshold: float = 0.2,
    connectivity: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Watershed separation on binary FG. Returns (labels, distance8u_for_debug).
    labels: int32, 0=background, 1..N objects.
    """
    # Distance transform inside FG
    fg = (binary > 0).astype(np.uint8)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    if distanceBlurK and distanceBlurK >= 3:
        dist = cv2.GaussianBlur(dist, (_forceOdd(distanceBlurK), _forceOdd(distanceBlurK)), 0)

    # Handle NaN/Inf and normalize for thresholds
    dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    dist_max = float(dist.max())
    if dist_max > 1e-9:
        distNorm = dist / dist_max
    else:
        distNorm = np.zeros_like(dist, dtype=np.float32)

    minVal = float(np.clip(peakRelThreshold, 0.0, 1.0)) * max(dist_max, 1e-9)
    peaks = _localMaxima(dist, minDist=max(1, int(peakMinDistance)), minVal=minVal)

    # Markers from peaks
    num, markers = cv2.connectedComponents(peaks, connectivity=connectivity)
    if num <= 1:
        # No peaks => return each CC as one region
        _, markers = cv2.connectedComponents(fg, connectivity=connectivity)
        labels = markers.astype(np.int32)
        labels[fg == 0] = 0
        return labels, (distNorm * 255).astype(np.uint8)

    # Prepare image for watershed: use inverted normalized distance so ridges are dark
    # cv2.watershed expects 3-channel 8-bit image
    invDist8u = (255 - (distNorm * 255).astype(np.uint8))
    wsImage = cv2.cvtColor(invDist8u, cv2.COLOR_GRAY2BGR)

    # Constrain to FG mask: set markers outside FG to 0
    markers = markers.astype(np.int32)
    markers[fg == 0] = 0

    cv2.watershed(wsImage, markers)  # modifies markers in place; -1 are boundaries

    labels = markers.copy()
    labels[labels < 0] = 0  # set boundaries to background (or could keep -1 for explicit)

    # Keep labels only inside FG
    labels[fg == 0] = 0
    return labels.astype(np.int32), invDist8u


def postSeparateCleanup(
    labels: np.ndarray,
    minAreaPx: int = 0,
    connectivity: int = 8,
    clearBorder: bool = False,
    shape: Optional[tuple[int, int]] = None
) -> np.ndarray:
    """Optionally filter labels by area and border touching.
    
    Vectorized: uses np.bincount for area histogram and LUT-based remapping
    instead of per-label loops over the full frame.
    """
    if labels.size == 0:
        return labels
    h, w = labels.shape
    labs = labels.astype(np.int32, copy=False)
    num = int(labs.max()) + 1  # label range is 0..max

    # Build a keep LUT (default: keep all labels)
    keep_lut = np.ones(num, dtype=bool)
    keep_lut[0] = False  # background never kept

    if clearBorder:
        # Collect labels touching any border edge
        border_labels = np.unique(np.concatenate([
            labs[0, :].ravel(),
            labs[-1, :].ravel(),
            labs[:, 0].ravel(),
            labs[:, -1].ravel()
        ]))
        keep_lut[border_labels] = False

    if minAreaPx > 1:
        # Compute area per label using bincount (O(n) in pixel count)
        areas = np.bincount(labs.ravel(), minlength=num)
        keep_lut &= (areas >= minAreaPx)
        keep_lut[0] = False  # ensure background stays excluded

    # Build relabel LUT: old label -> new sequential label (or 0 if removed)
    relabel_lut = np.zeros(num, dtype=np.int32)
    new_label = 1
    for old in range(1, num):
        if keep_lut[old]:
            relabel_lut[old] = new_label
            new_label += 1

    # Apply LUT to produce output
    return relabel_lut[labs]


# -------------------------- Visualization -------------------------

def labelsToColor(labels: np.ndarray, bgGray: Optional[np.ndarray] = None, alpha: float = 0.45) -> np.ndarray:
    """
    Colorize label map. If bgGray provided (uint8), alpha-blend color on top.
    Returns BGR uint8 for display.
    """
    if labels.dtype != np.int32:
        labels = labels.astype(np.int32)
    h, w = labels.shape
    n = int(labels.max())
    if n == 0:
        # Just return gray bg if provided, else a black image
        if bgGray is not None:
            if bgGray.ndim == 2:
                return cv2.cvtColor(bgGray, cv2.COLOR_GRAY2BGR)
            return bgGray.copy()
        return np.zeros((h, w, 3), dtype=np.uint8)

    # Stable random palette
    rng = np.random.default_rng(12345)
    palette = (rng.random((n+1, 3)) * 255).astype(np.uint8)
    palette[0] = (0, 0, 0)

    color = palette[labels]
    if bgGray is not None:
        if bgGray.ndim == 2:
            bg = cv2.cvtColor(bgGray, cv2.COLOR_GRAY2BGR)
        else:
            bg = bgGray
        a = float(np.clip(alpha, 0.0, 1.0))
        out = cv2.addWeighted(bg, 1.0 - a, color, a, 0.0)
        return out
    return color


# -------------------------- Pipeline ------------------------------

def runSeparationPipeline(
    grayOrBgr: np.ndarray,
    threshParams: Dict[str, float | int | bool | str],
    sepParams: Dict[str, float | int | bool | str]
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Performs threshold -> (optional fill holes/cleanup) -> separation (if enabled).
    Returns (binary_uint8, labels_int32_or_None, meta).
    """
    # Threshold
    binary, meta = thresholdImageAdvanced(gray=grayOrBgr, **threshParams)  # type: ignore

    # Optional hole fill and speck cleanup before watershed
    if bool(sepParams.get("fillHoles", True)):
        binary = fillHoles(binary)
    minArea = int(sepParams.get("minAreaPx", 0))
    if minArea > 1:
        binary = removeSmallAreas(binary, minArea=minArea, connectivity=int(sepParams.get("connectivity", 8)))
    if bool(sepParams.get("clearBorder", False)):
        binary = clearBorderTouching(binary, connectivity=int(sepParams.get("connectivity", 8)))

    # Separation
    method = str(sepParams.get("method", "none"))
    labels = None
    if method == "watershed":
        labels, _ = watershedSeparate(
            binary=binary,
            distanceBlurK=int(sepParams.get("distanceBlurK", 3)),
            peakMinDistance=int(sepParams.get("peakMinDistance", 9)),
            peakRelThreshold=float(sepParams.get("peakRelThreshold", 0.2)),
            connectivity=int(sepParams.get("connectivity", 8))
        )
        labels = postSeparateCleanup(
            labels,
            minAreaPx=int(sepParams.get("minAreaPx", 0)),
            connectivity=int(sepParams.get("connectivity", 8)),
            clearBorder=bool(sepParams.get("clearBorder", False))
        )
    elif method == "none":
        labels = None
    else:
        raise ValueError(f"Unknown separation method: {method}")

    return binary, labels, meta
