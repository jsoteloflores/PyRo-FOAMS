from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_SCHEMA_VERSION = 1

SUPPORTED_CALIBRATION_UNITS_TO_MM = {
    "m": 1000.0,
    "cm": 10.0,
    "mm": 1.0,
    "um": 0.001,
    "µm": 0.001,
    "μm": 0.001,
    "micron": 0.001,
    "microns": 0.001,
    "micrometer": 0.001,
    "micrometers": 0.001,
    "nm": 0.000001,
    "nanometer": 0.000001,
    "nanometers": 0.000001,
}


class SampleProjectError(ValueError):
    """Raised when sample project state or persistence is invalid."""


@dataclass
class Calibration:
    units_per_px: Optional[float] = None
    unit_name: Optional[str] = None

    @property
    def is_calibrated(self) -> bool:
        if self.units_per_px is None or self.unit_name is None:
            return False
        if isinstance(self.units_per_px, (bool, np.bool_)):
            return False
        return (
            math.isfinite(self.units_per_px)
            and self.units_per_px > 0
            and self.unit_name.lower() in SUPPORTED_CALIBRATION_UNITS_TO_MM
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "units_per_px": self.units_per_px,
            "unit_name": self.unit_name,
            "is_calibrated": self.is_calibrated,
        }


@dataclass
class AcceptedSnapshot:
    snapshot_id: str
    checksum_sha256: str
    source_checksum_sha256: str
    working_image_checksum_sha256: str
    processing_roi: Tuple[int, int, int, int]
    source_revision: int
    draft_revision: int
    calibration_revision: int
    source_kind: str
    source_details: Dict[str, Any]
    created_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "checksum_sha256": self.checksum_sha256,
            "source_checksum_sha256": self.source_checksum_sha256,
            "working_image_checksum_sha256": self.working_image_checksum_sha256,
            "processing_roi": list(self.processing_roi),
            "source_revision": self.source_revision,
            "draft_revision": self.draft_revision,
            "calibration_revision": self.calibration_revision,
            "source_kind": self.source_kind,
            "source_details": dict(self.source_details),
            "created_at_utc": self.created_at_utc,
        }


@dataclass
class ProjectImage:
    image_id: str
    display_name: str
    source_filename: str
    source_checksum_sha256: str
    width: int
    height: int
    source_revision: int = 0
    calibration_revision: int = 0
    draft_revision: int = 0
    accepted_revision: int = 0
    analysis_included: bool = False
    calibration: Calibration = field(default_factory=Calibration)
    draft_processing_settings: Dict[str, Any] = field(default_factory=dict)
    draft_source_kind: str = "processing"
    draft_source_details: Dict[str, Any] = field(default_factory=dict)
    accepted_snapshot: Optional[AcceptedSnapshot] = None

    # Runtime arrays (not serialized inline in JSON manifest)
    source_image: Optional[np.ndarray] = None
    working_image: Optional[np.ndarray] = None
    processing_roi: Optional[Tuple[int, int, int, int]] = None
    draft_mask: Optional[np.ndarray] = None
    draft_labels: Optional[np.ndarray] = None
    accepted_mask: Optional[np.ndarray] = None
    accepted_labels: Optional[np.ndarray] = None

    source_asset_path: Optional[str] = None
    working_asset_path: Optional[str] = None
    draft_mask_asset_path: Optional[str] = None
    draft_labels_asset_path: Optional[str] = None
    accepted_mask_asset_path: Optional[str] = None
    accepted_labels_asset_path: Optional[str] = None

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "display_name": self.display_name,
            "source_filename": self.source_filename,
            "source_checksum_sha256": self.source_checksum_sha256,
            "width": self.width,
            "height": self.height,
            "processing_roi": list(self.processing_roi) if self.processing_roi is not None else None,
            "source_asset_path": self.source_asset_path,
            "working_asset_path": self.working_asset_path,
            "draft_mask_asset_path": self.draft_mask_asset_path,
            "draft_labels_asset_path": self.draft_labels_asset_path,
            "accepted_mask_asset_path": self.accepted_mask_asset_path,
            "accepted_labels_asset_path": self.accepted_labels_asset_path,
            "analysis_included": self.analysis_included,
            "revisions": {
                "source": self.source_revision,
                "calibration": self.calibration_revision,
                "draft": self.draft_revision,
                "accepted": self.accepted_revision,
            },
            "calibration": self.calibration.to_dict(),
            "draft_processing_settings": dict(self.draft_processing_settings),
            "draft_source_kind": self.draft_source_kind,
            "draft_source_details": dict(self.draft_source_details),
            "accepted_snapshot": self.accepted_snapshot.to_dict() if self.accepted_snapshot else None,
        }


@dataclass
class SampleProject:
    project_id: str
    sample_name: str
    notes: str = ""
    schema_version: int = PROJECT_SCHEMA_VERSION
    images: List[ProjectImage] = field(default_factory=list)
    analysis_revision: int = 0

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "sample_name": self.sample_name,
            "notes": self.notes,
            "analysis_revision": self.analysis_revision,
            "images": [image.to_manifest_dict() for image in self.images],
        }



def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")



def generate_stable_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"



def checksum_ndarray(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()



def normalize_mask(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    expected_h, expected_w = shape
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.shape != (expected_h, expected_w):
        raise SampleProjectError("mask dimensions do not match source image dimensions")
    out = np.where(arr > 0, 255, 0).astype(np.uint8)
    return out



def labels_from_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 1, 0).astype(np.uint8)
    _num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    return labels.astype(np.int32)



def create_project(sample_name: str, notes: str = "") -> SampleProject:
    name = sample_name.strip() or "Untitled Sample"
    return SampleProject(project_id=generate_stable_id("sample"), sample_name=name, notes=notes)



def add_image(project: SampleProject, image: np.ndarray, source_filename: str, display_name: Optional[str] = None) -> ProjectImage:
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise SampleProjectError("source image must be grayscale or color")
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = arr.shape[:2]
    project_image = ProjectImage(
        image_id=generate_stable_id("img"),
        display_name=(display_name or source_filename).strip() or source_filename,
        source_filename=source_filename,
        source_checksum_sha256=checksum_ndarray(arr),
        width=w,
        height=h,
        source_image=arr.copy(),
        working_image=arr.copy(),
        processing_roi=(0, 0, w, h),
    )
    project.images.append(project_image)
    return project_image



def set_calibration(image: ProjectImage, units_per_px: Optional[float], unit_name: Optional[str]) -> None:
    if units_per_px is None or unit_name is None:
        image.calibration = Calibration()
    else:
        if isinstance(units_per_px, (bool, np.bool_)):
            raise SampleProjectError("calibration value must be numeric, not Boolean")
        value = float(units_per_px)
        if not math.isfinite(value) or value <= 0:
            raise SampleProjectError("calibration value must be finite and positive")
        image.calibration = Calibration(units_per_px=value, unit_name=str(unit_name))
    image.calibration_revision += 1



def update_draft(
    image: ProjectImage,
    *,
    mask: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    settings: Optional[Dict[str, Any]] = None,
    source_kind: Optional[str] = None,
    source_details: Optional[Dict[str, Any]] = None,
) -> None:
    if image.working_image is None:
        raise SampleProjectError("working image array is unavailable")
    shape = image.working_image.shape[:2]
    normalized_mask = normalize_mask(mask, shape)
    image.draft_mask = normalized_mask.copy() if normalized_mask is not None else None
    if labels is not None:
        labels_array = np.asarray(labels)
        if labels_array.shape != shape:
            raise SampleProjectError("label dimensions do not match source image dimensions")
        normalized_labels = labels_array.astype(np.int32, copy=True)
        labels_match_mask = (
            normalized_mask is not None
            and np.array_equal(normalized_labels > 0, normalized_mask > 0)
        )
        image.draft_labels = normalized_labels if labels_match_mask else (
            labels_from_mask(normalized_mask) if normalized_mask is not None else None
        )
    elif normalized_mask is not None:
        image.draft_labels = labels_from_mask(normalized_mask)
    else:
        image.draft_labels = None
    if settings is not None:
        image.draft_processing_settings = dict(settings)
    if source_kind is not None:
        image.draft_source_kind = str(source_kind)
    if source_details is not None:
        image.draft_source_details = dict(source_details)
    image.draft_revision += 1



def accept_draft(
    image: ProjectImage,
    *,
    source_kind: Optional[str] = None,
    source_details: Optional[Dict[str, Any]] = None,
) -> AcceptedSnapshot:
    if image.draft_mask is None:
        raise SampleProjectError("cannot accept without a draft mask")
    if image.working_image is None:
        raise SampleProjectError("working image array is unavailable")
    accepted_mask = image.draft_mask.copy()
    current_labels = image.draft_labels
    accepted_labels = (
        current_labels.copy()
        if current_labels is not None
        and current_labels.shape == accepted_mask.shape
        and np.array_equal(current_labels > 0, accepted_mask > 0)
        else labels_from_mask(accepted_mask)
    )
    snapshot = AcceptedSnapshot(
        snapshot_id=generate_stable_id("snap"),
        checksum_sha256=checksum_ndarray(accepted_mask),
        source_checksum_sha256=image.source_checksum_sha256,
        working_image_checksum_sha256=checksum_ndarray(image.working_image),
        processing_roi=image.processing_roi or (0, 0, image.width, image.height),
        source_revision=image.source_revision,
        draft_revision=image.draft_revision,
        calibration_revision=image.calibration_revision,
        source_kind=source_kind or image.draft_source_kind,
        source_details={
            "draft_source_details": dict(image.draft_source_details),
            "processing_settings": dict(image.draft_processing_settings),
            **dict(source_details or {}),
        },
        created_at_utc=now_utc_iso(),
    )
    image.accepted_mask = accepted_mask
    image.accepted_labels = accepted_labels
    image.accepted_snapshot = snapshot
    image.accepted_revision += 1
    return snapshot



def replace_source_image(image: ProjectImage, new_source: np.ndarray, source_filename: str) -> None:
    arr = np.asarray(new_source)
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = arr.shape[:2]
    image.source_image = arr.copy()
    image.working_image = arr.copy()
    image.source_filename = source_filename
    image.source_checksum_sha256 = checksum_ndarray(arr)
    image.width = w
    image.height = h
    image.processing_roi = (0, 0, w, h)
    image.source_revision += 1

    # Source replacement invalidates accepted and inclusion unless explicit migration exists.
    image.draft_mask = None
    image.draft_labels = None
    image.accepted_mask = None
    image.accepted_labels = None
    image.accepted_snapshot = None
    image.analysis_included = False



def replace_working_image(
    image: ProjectImage,
    working_image: np.ndarray,
    source_roi: Tuple[int, int, int, int],
) -> None:
    if image.source_image is None:
        raise SampleProjectError("source image array is unavailable")
    arr = np.asarray(working_image)
    if arr.ndim not in (2, 3) or arr.dtype != np.uint8:
        raise SampleProjectError("working image must be a uint8 grayscale or color image")
    x0, y0, x1, y1 = (int(value) for value in source_roi)
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise SampleProjectError("processing ROI is outside the original source image")
    if arr.shape[:2] != (y1 - y0, x1 - x0):
        raise SampleProjectError("working image dimensions do not match its source-coordinate ROI")
    image.working_image = arr.copy()
    image.processing_roi = (x0, y0, x1, y1)
    image.draft_mask = None
    image.draft_labels = None
    image.draft_revision += 1
    image.accepted_mask = None
    image.accepted_labels = None
    image.accepted_snapshot = None
    image.accepted_revision += 1
    image.analysis_included = False


def set_analysis_included(image: ProjectImage, included: bool) -> None:
    if included and image.accepted_snapshot is None:
        raise SampleProjectError("analysis inclusion requires an accepted segmentation")
    if image.analysis_included != bool(included):
        image.analysis_included = bool(included)



def calibration_mm_per_px(image: ProjectImage) -> Optional[float]:
    calibration = image.calibration
    if not calibration.is_calibrated or calibration.units_per_px is None or calibration.unit_name is None:
        return None
    factor = SUPPORTED_CALIBRATION_UNITS_TO_MM[calibration.unit_name.lower()]
    return calibration.units_per_px * factor



def included_images(project: SampleProject) -> List[ProjectImage]:
    return [image for image in project.images if image.analysis_included]



def included_images_missing_calibration(project: SampleProject) -> List[ProjectImage]:
    return [image for image in included_images(project) if not image.calibration.is_calibrated]
