# main/core/stereology.py
# Per-pore measurements + label coloring + CSV export
# Pure callables with no GUI dependencies - safe for headless testing and parallelism

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Iterable, Dict
import csv
import math
import numpy as np
import cv2

# ---------- Data Model ----------

@dataclass
class PoreProps:
    # identity
    image_index: int
    label: int

    # pixel-space geometry
    area_px: int
    perimeter_px: float
    centroid_x: float
    centroid_y: float
    bbox_x0: int
    bbox_y0: int
    bbox_x1: int  # exclusive
    bbox_y1: int  # exclusive
    touches_border: bool

    # shape descriptors (pixels)
    eq_diam_px: float                 # sqrt(4A/pi)
    circularity: float                # 4*pi*A / P^2  (1.0 => circle)
    major_axis_px: Optional[float]    # from fitEllipse if available
    minor_axis_px: Optional[float]
    aspect_ratio: Optional[float]     # major/minor
    orientation_deg: Optional[float]  # ellipse major-axis angle
    feret_max_px: Optional[float]     # max caliper (approx via hull)
    feret_min_px: Optional[float]     # min caliper (approx via minAreaRect)

    # scaled (optional)
    units_per_px: Optional[float] = None
    area_units2: Optional[float] = None
    eq_diam_units: Optional[float] = None
    major_axis_units: Optional[float] = None
    minor_axis_units: Optional[float] = None
    feret_max_units: Optional[float] = None
    feret_min_units: Optional[float] = None
    unit_name: Optional[str] = None


# ---------- Public API ----------

def colorize_labels(
    labels: np.ndarray,
    seed: int = 123,
    bg_gray: Optional[np.ndarray] = None,
    alpha: float = 0.45
) -> np.ndarray:
    """
    Assign a distinct (pseudo-random) color to each label>0.
    If bg_gray is provided (H,W) uint8, overlay colors over the grayscale.
    Returns BGR uint8 image.
    """
    assert labels.ndim == 2, "labels must be HxW"
    h, w = labels.shape
    lab = labels.astype(np.int32, copy=False)

    # build a color lookup table (LUT)
    u = np.unique(lab)
    u = u[u > 0]
    rng = np.random.default_rng(seed)
    # evenly spaced hues, then permute for variety
    n = len(u)
    hues = (np.linspace(0, 179, num=max(n,1), endpoint=False).astype(np.uint8)
            if n > 0 else np.array([], np.uint8))
    rng.shuffle(hues)
    sat = np.full_like(hues, 200, dtype=np.uint8)
    val = np.full_like(hues, 255, dtype=np.uint8)
    hsv = np.stack([hues, sat, val], axis=1).reshape(-1,1,3)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1,3)

    lut = {int(lbl): tuple(int(c) for c in bgr[i]) for i, lbl in enumerate(u)}

    out = np.zeros((h, w, 3), np.uint8)
    for lbl, col in lut.items():
        out[lab == lbl] = col

    if bg_gray is not None:
        if bg_gray.ndim == 3:
            base = bg_gray.copy()
        else:
            base = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        mask = (lab > 0).astype(np.uint8) * 255
        # blend only where labels exist
        overlay = out
        blended = base.copy()
        roi = mask.astype(bool)
        blended[roi] = cv2.addWeighted(overlay[roi], alpha, base[roi], 1.0 - alpha, 0)
        return blended
    return out


def measure_labels(
    labels: np.ndarray,
    image_index: int = 0,
    scale: Optional[Dict[str, float | str]] = None
) -> List[PoreProps]:
    """
    Compute pore-wise metrics for a labeled image.
    - labels: HxW int labels, 0 = background, >0 = pore id
    - scale: optionally {"unitsPerPx": float, "unitName": str}
    Returns a list of PoreProps (one per label>0).
    """
    if labels is None:
        return []

    assert labels.ndim == 2, "labels must be HxW"
    H, W = labels.shape
    units_per_px = None
    unit_name = None
    if isinstance(scale, dict) and "unitsPerPx" in scale:
        try:
            units_per_px = float(scale["unitsPerPx"])
        except Exception:
            units_per_px = None
        unit_name = str(scale.get("unitName", "") or "")

    props: List[PoreProps] = []
    labs = np.unique(labels)
    labs = labs[(labs > 0)]
    if labs.size == 0:
        return props

    # pre-alloc a uint8 scratch for contours
    scratch = np.zeros_like(labels, dtype=np.uint8)

    for lbl in labs:
        # mask
        np.equal(labels, lbl, out=scratch)       # scratch is 0/1
        area = int(np.count_nonzero(scratch))
        if area == 0:
            continue

        # perimeter via external contours (holes ignored by default)
        cnts, _ = cv2.findContours(scratch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = float(sum(cv2.arcLength(c, True) for c in cnts))

        # centroid via moments
        m = cv2.moments(scratch, binaryImage=True)
        if m["m00"] <= 0:
            # extremely tiny / degenerate
            cx = cy = float("nan")
        else:
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])

        # bbox
        ys, xs = np.nonzero(scratch)
        x0 = int(xs.min()); x1 = int(xs.max()) + 1  # exclusive
        y0 = int(ys.min()); y1 = int(ys.max()) + 1
        touches_border = (x0 == 0) or (y0 == 0) or (x1 >= W) or (y1 >= H)

        # basic shape scalars
        eq_d = math.sqrt((4.0 * area) / math.pi) if area > 0 else float("nan")
        circ = (4.0 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else float("nan")

        # ellipse + orientation (best-effort)
        major = minor = orient = None
        flat_pts = None
        if len(cnts) > 0:
            flat_pts = np.vstack(cnts).squeeze(1)  # (N,2)
            if flat_pts.shape[0] >= 5:
                (ex, ey), (w, h), angle = cv2.fitEllipse(flat_pts)
                # ensure major>=minor
                if w >= h:
                    major, minor = float(w), float(h)
                    orient = float(angle)
                else:
                    major, minor = float(h), float(w)
                    orient = float((angle + 90.0) % 180.0)
            else:
                # PCA fallback
                pts = flat_pts.astype(np.float32)
                mean, eigvecs, eigvals = cv2.PCACompute2(pts, mean=None)
                # Not a physical axis length, but gives relative spread; scale by 4 for rough diameter-like quantity
                try:
                    d1 = 4.0 * float(np.sqrt(max(eigvals[0,0], 0.0)))
                    d2 = 4.0 * float(np.sqrt(max(eigvals[1,0], 0.0)))
                    major = max(d1, d2); minor = min(d1, d2)
                    vx, vy = eigvecs[0]
                    orient = float((math.degrees(math.atan2(vy, vx)) + 360.0) % 180.0)
                except Exception:
                    major = minor = orient = None

        aspect = float(major / minor) if (major and minor and minor > 0) else None

        # Feret diameters
        feret_max = feret_min = None
        if flat_pts is not None and flat_pts.shape[0] >= 2:
            hull = cv2.convexHull(flat_pts)
            Hpts = hull.reshape(-1, 2)
            # min caliper via minAreaRect
            rect = cv2.minAreaRect(Hpts)
            w_rect, h_rect = rect[1]
            if (w_rect is not None) and (h_rect is not None):
                a = float(w_rect); b = float(h_rect)
                feret_min = float(min(a, b))
                # max feret often approximated as hull pairwise max distance (O(n^2) but hull is small)
                # if hull is huge, sub-sample for speed
                if len(Hpts) > 1200:
                    idx = np.linspace(0, len(Hpts)-1, 600, dtype=int)
                    Hsub = Hpts[idx]
                else:
                    Hsub = Hpts
                diffs = Hsub[None, :, :] - Hsub[:, None, :]
                d2 = (diffs[:, :, 0] ** 2 + diffs[:, :, 1] ** 2)
                feret_max = float(np.sqrt(d2.max()))
            else:
                feret_min = None
                feret_max = None

        # build record
        rec = PoreProps(
            image_index=image_index,
            label=int(lbl),
            area_px=area,
            perimeter_px=perimeter,
            centroid_x=cx, centroid_y=cy,
            bbox_x0=x0, bbox_y0=y0, bbox_x1=x1, bbox_y1=y1,
            touches_border=bool(touches_border),
            eq_diam_px=eq_d,
            circularity=circ,
            major_axis_px=major, minor_axis_px=minor,
            aspect_ratio=aspect,
            orientation_deg=orient,
            feret_max_px=feret_max,
            feret_min_px=feret_min,
            units_per_px=units_per_px,
            unit_name=unit_name
        )

        # apply units (optional)
        if units_per_px and units_per_px > 0:
            upx = float(units_per_px)
            rec.area_units2 = float(area) * (upx ** 2)
            rec.eq_diam_units = rec.eq_diam_px * upx
            if rec.major_axis_px:  rec.major_axis_units  = rec.major_axis_px  * upx
            if rec.minor_axis_px:  rec.minor_axis_units  = rec.minor_axis_px  * upx
            if rec.feret_max_px:   rec.feret_max_units   = rec.feret_max_px   * upx
            if rec.feret_min_px:   rec.feret_min_units   = rec.feret_min_px   * upx

        props.append(rec)

    return props


def measure_dataset(
    labels_list: List[Optional[np.ndarray]],
    scales_list: Optional[List[Optional[Dict[str, float | str]]]] = None
) -> List[PoreProps]:
    """
    Measure a whole set of images (some labels may be None).
    scales_list is optional; when present, matched 1:1 with labels_list.
    """
    all_props: List[PoreProps] = []
    n = len(labels_list)
    for i in range(n):
        L = labels_list[i]
        if L is None:
            continue
        scale = None
        if scales_list and i < len(scales_list):
            scale = scales_list[i]
        all_props.extend(measure_labels(L, image_index=i, scale=scale))
    return all_props


def save_props_csv(path: str, props: Iterable[PoreProps]) -> None:
    """
    Write all properties to a CSV (flat columns, no nesting).
    """
    rows = [asdict(p) for p in props]
    if not rows:
        # create empty file with header for reproducibility
        header = [f.name for f in PoreProps.__dataclass_fields__.values()]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        return

    header = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def mask_from_labels(labels: np.ndarray) -> np.ndarray:
    """
    Utility: binary mask (uint8 0/255) from labels (0=background).
    """
    m = (labels.astype(np.int32, copy=False) > 0).astype(np.uint8) * 255
    return m
