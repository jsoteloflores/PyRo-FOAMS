"""FOAMS 1.0.5 perimeter replay profiles with validated raw-selection joins."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .legacy_measurements import (
    FOAMS105_BORDER_CONNECTIVITY,
    FOAMS105_COMPONENT_CONNECTIVITY,
    FOAMS105_COMPONENT_ORDER_POLICY,
    FOAMS105_HOLE_BACKGROUND_CONNECTIVITY,
    FOAMS105_MEASUREMENT_CALIBRATION_SOURCE_FILE,
    FOAMS105_MEASUREMENT_METHOD,
    FOAMS105_MEASUREMENT_SCOPE,
    FOAMS105_MEASUREMENT_SOURCE_COMMIT,
    FOAMS105_MEASUREMENT_SOURCE_FILE,
    FOAMS105_MEASUREMENT_SOURCE_REPOSITORY,
    Foams105ComponentMeasurement,
    Foams105PhaseMeasurementResult,
)
from .legacy_raw_selection import (
    FOAMS105_RAW_SELECTION_METHOD,
    Foams105RawSelectionResult,
    select_foams105_raw_objects,
)

FOAMS105_PERIMETER_METHOD = "foams_1_0_5_perimeter_replay_v1"
FOAMS105_PERIMETER_SELECTION_METHOD = "foams_1_0_5_perimeter_selection_replay_v1"
FOAMS105_PERIMETER_SOURCE_REPOSITORY = FOAMS105_MEASUREMENT_SOURCE_REPOSITORY
FOAMS105_PERIMETER_SOURCE_COMMIT = FOAMS105_MEASUREMENT_SOURCE_COMMIT
FOAMS105_PERIMETER_SOURCE_FILES = ("data_meas.m", "project.m")
FOAMS105_PERIMETER_SCOPE = "component_perimeter_only"
FOAMS105_PERIMETER_SELECTION_SCOPE = "raw_identity_perimeter_selection_only"
FOAMS105_PERIMETER_VERIFICATION_LEVEL = "matlab_r2025b_reference_profiles"
FOAMS105_PERIMETER_PROFILE_OLD = "matlab_perimeter_old_r2025b_verified_v1"
FOAMS105_PERIMETER_PROFILE_CURRENT = "matlab_perimeter_r2025b_verified_v1"
FOAMS105_PERIMETER_SUPPORTED_PROFILES = (
    FOAMS105_PERIMETER_PROFILE_OLD,
    FOAMS105_PERIMETER_PROFILE_CURRENT,
)
FOAMS105_PERIMETER_REFERENCE_RUNTIME = "MATLAB 25.2.0.2998904 (R2025b)"
FOAMS105_PERIMETER_REFERENCE_TOOLBOX = "Image Processing Toolbox 25.2"
FOAMS105_PERIMETER_EVIDENCE_VERSION = "16b_perimeter_evidence_v1"
FOAMS105_PERIMETER_EVIDENCE_HASH = "sha256:0738e29e117bf61ca008688c8db3f8a8c7c9c8f29d29efa5d5c669765576f8d1"
FOAMS105_PERIMETER_SELECTION_RELATIONSHIP_POLICY = (
    "raw_selection_objects_must_match_perimeter_identity_and_metadata"
)

_PROFILE_COEFFICIENTS = {
    FOAMS105_PERIMETER_PROFILE_OLD: (1.0, math.sqrt(2.0), 0.0),
    FOAMS105_PERIMETER_PROFILE_CURRENT: (0.980, 1.406, -0.091),
}


class Foams105PerimeterValidationError(ValueError):
    """Raised when perimeter inputs violate the bounded replay contract."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_input",
        context: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = MappingProxyType(dict(context or {}))


class Foams105PerimeterNumericalError(ArithmeticError):
    """Raised when perimeter tracing or arithmetic cannot be represented."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        context: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = MappingProxyType(dict(context or {}))


@dataclass(frozen=True)
class Foams105PerimeterComponent:
    image_id: str
    component_id: int
    first_pixel_rc: Tuple[int, int]
    area_px: int
    area_mm2: float
    equivalent_diameter_mm: float
    scale_px_per_mm: float
    axial_step_count: int
    diagonal_step_count: int
    corner_class_transition_count: int
    perimeter_px: float
    perimeter_mm: float
    profile: str
    method: str = FOAMS105_PERIMETER_METHOD
    verification_level: str = FOAMS105_PERIMETER_VERIFICATION_LEVEL
    reference_runtime: str = FOAMS105_PERIMETER_REFERENCE_RUNTIME
    reference_toolbox: str = FOAMS105_PERIMETER_REFERENCE_TOOLBOX
    evidence_version: str = FOAMS105_PERIMETER_EVIDENCE_VERSION
    evidence_hash: str = FOAMS105_PERIMETER_EVIDENCE_HASH


@dataclass(frozen=True)
class Foams105ImagePerimeterResult:
    image_id: str
    phase_measurement_result: Foams105PhaseMeasurementResult
    profile: str
    components: Tuple[Foams105PerimeterComponent, ...]
    component_count: int
    foreground_pixel_count: int
    scale_px_per_mm: float
    method: str = FOAMS105_PERIMETER_METHOD
    source_repository: str = FOAMS105_PERIMETER_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_PERIMETER_SOURCE_COMMIT
    source_files: Tuple[str, str] = FOAMS105_PERIMETER_SOURCE_FILES
    scope: str = FOAMS105_PERIMETER_SCOPE
    connectivity: int = FOAMS105_COMPONENT_CONNECTIVITY
    verification_level: str = FOAMS105_PERIMETER_VERIFICATION_LEVEL
    reference_runtime: str = FOAMS105_PERIMETER_REFERENCE_RUNTIME
    reference_toolbox: str = FOAMS105_PERIMETER_REFERENCE_TOOLBOX
    evidence_version: str = FOAMS105_PERIMETER_EVIDENCE_VERSION
    evidence_hash: str = FOAMS105_PERIMETER_EVIDENCE_HASH
    length_unit: str = "mm"
    area_unit: str = "mm^2"
    pixel_length_unit: str = "pixel"


@dataclass(frozen=True)
class Foams105SelectedPerimeterRow:
    output_index: int
    raw_output_index: int
    group_id: str
    source_slot: int
    source_object_index: int
    image_id: str
    component_id: int
    area_mm2: float
    equivalent_diameter_mm: float
    scale_px_per_mm: float
    perimeter_px: float
    perimeter_mm: float
    axial_step_count: int
    diagonal_step_count: int
    corner_class_transition_count: int
    profile: str
    component_perimeter: Foams105PerimeterComponent


@dataclass(frozen=True)
class Foams105SelectedPerimeterResult:
    raw_selection_result: Foams105RawSelectionResult
    image_perimeter_results: Tuple[Foams105ImagePerimeterResult, ...]
    profile: str
    rows: Tuple[Foams105SelectedPerimeterRow, ...]
    selected_identities: Tuple[Tuple[str, int], ...]
    selected_count: int
    method: str = FOAMS105_PERIMETER_SELECTION_METHOD
    source_repository: str = FOAMS105_PERIMETER_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_PERIMETER_SOURCE_COMMIT
    source_files: Tuple[str, str] = FOAMS105_PERIMETER_SOURCE_FILES
    scope: str = FOAMS105_PERIMETER_SELECTION_SCOPE
    verification_level: str = FOAMS105_PERIMETER_VERIFICATION_LEVEL
    relationship_policy: str = FOAMS105_PERIMETER_SELECTION_RELATIONSHIP_POLICY
    length_unit: str = "mm"
    area_unit: str = "mm^2"


def _validate_image_id(image_id: object) -> str:
    if not isinstance(image_id, str) or not image_id.strip():
        raise Foams105PerimeterValidationError(
            "image_id must be a nonempty string",
            context={"field": "image_id"},
        )
    return image_id


def _strict_int(
    value: object,
    field: str,
    *,
    image_id: str,
    component_id: Optional[int] = None,
    minimum: int = 0,
) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise Foams105PerimeterValidationError(
            f"{field} must be an integer scalar",
            code="inconsistent_measurement_result",
            context={"image_id": image_id, "component_id": component_id, "field": field},
        )
    result = int(value)
    if result < minimum:
        raise Foams105PerimeterValidationError(
            f"{field} must be >= {minimum}",
            code="inconsistent_measurement_result",
            context={"image_id": image_id, "component_id": component_id, "field": field, "value": result},
        )
    return result


def _strict_finite_real(
    value: object,
    field: str,
    *,
    image_id: str,
    component_id: Optional[int] = None,
) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise Foams105PerimeterValidationError(
            f"{field} must be a finite real scalar",
            code="inconsistent_measurement_result",
            context={"image_id": image_id, "component_id": component_id, "field": field},
        )
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise Foams105PerimeterValidationError(
            f"{field} must be representable as a finite float64",
            code="inconsistent_measurement_result",
            context={"image_id": image_id, "component_id": component_id, "field": field},
        ) from exc
    if not math.isfinite(result):
        raise Foams105PerimeterValidationError(
            f"{field} must be finite",
            code="inconsistent_measurement_result",
            context={"image_id": image_id, "component_id": component_id, "field": field},
        )
    return result


def _finite_positive_real(value: object, field: str, *, image_id: str) -> float:
    result = _strict_finite_real(value, field, image_id=image_id)
    if result <= 0.0:
        raise Foams105PerimeterValidationError(
            f"{field} must be positive",
            code="inconsistent_measurement_result",
            context={"image_id": image_id, "field": field},
        )
    return result


def _owned_immutable_array(values: object, dtype: np.dtype, field: str, *, image_id: str) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.ndim != 2 or values.size == 0:
        raise Foams105PerimeterValidationError(
            f"{field} must be a nonempty two-dimensional NumPy array",
            code="inconsistent_measurement_result",
            context={"image_id": image_id, "field": field},
        )
    contiguous = np.ascontiguousarray(values, dtype=dtype)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=dtype)
    return immutable.reshape(contiguous.shape)


def _owned_phase_measurement_result(measurement: object, *, image_id: str) -> Foams105PhaseMeasurementResult:
    if not isinstance(measurement, Foams105PhaseMeasurementResult):
        raise Foams105PerimeterValidationError(
            "phase_measurement_result must be a Foams105PhaseMeasurementResult",
            code="incompatible_measurement_result",
            context={"image_id": image_id},
        )
    if not isinstance(measurement.components, tuple):
        raise Foams105PerimeterValidationError(
            "phase_measurement_result components must be a tuple",
            code="inconsistent_measurement_result",
            context={"image_id": image_id},
        )
    for index, component in enumerate(measurement.components):
        if not isinstance(component, Foams105ComponentMeasurement):
            raise Foams105PerimeterValidationError(
                "phase_measurement_result components contain an invalid record",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_index": index},
            )
    expected_dtypes = (
        (measurement.removed_border_mask, np.dtype(np.bool_), "removed_border_mask"),
        (measurement.filled_mask, np.dtype(np.bool_), "filled_mask"),
        (measurement.label_image, np.dtype(np.int32), "label_image"),
    )
    for values, expected_dtype, field in expected_dtypes:
        if not isinstance(values, np.ndarray) or values.dtype != expected_dtype:
            raise Foams105PerimeterValidationError(
                f"phase_measurement_result.{field} has an incompatible dtype",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "field": field},
            )
    return replace(
        measurement,
        components=tuple(measurement.components),
        removed_border_mask=_owned_immutable_array(
            measurement.removed_border_mask,
            np.dtype(np.bool_),
            "phase_measurement_result.removed_border_mask",
            image_id=image_id,
        ),
        filled_mask=_owned_immutable_array(
            measurement.filled_mask,
            np.dtype(np.bool_),
            "phase_measurement_result.filled_mask",
            image_id=image_id,
        ),
        label_image=_owned_immutable_array(
            measurement.label_image,
            np.dtype(np.int32),
            "phase_measurement_result.label_image",
            image_id=image_id,
        ),
    )


def _validate_profile(profile: str) -> str:
    if not isinstance(profile, str) or not profile.strip():
        raise Foams105PerimeterValidationError("profile must be a nonempty string")
    if profile not in FOAMS105_PERIMETER_SUPPORTED_PROFILES:
        raise Foams105PerimeterValidationError(
            "profile is unsupported",
            code="unsupported_profile",
            context={"profile": profile, "supported_profiles": FOAMS105_PERIMETER_SUPPORTED_PROFILES},
        )
    return profile


def _validate_phase_measurement(image_id: str, result: object) -> Foams105PhaseMeasurementResult:
    measurement = _owned_phase_measurement_result(result, image_id=image_id)
    if measurement.method != FOAMS105_MEASUREMENT_METHOD:
        raise Foams105PerimeterValidationError(
            "phase_measurement_result method is incompatible",
            code="incompatible_measurement_result",
            context={"image_id": image_id},
        )
    if measurement.source_repository != FOAMS105_MEASUREMENT_SOURCE_REPOSITORY:
        raise Foams105PerimeterValidationError(
            "phase_measurement_result source repository is incompatible",
            code="incompatible_measurement_result",
            context={"image_id": image_id},
        )
    if measurement.source_commit != FOAMS105_MEASUREMENT_SOURCE_COMMIT:
        raise Foams105PerimeterValidationError(
            "phase_measurement_result source commit is incompatible",
            code="incompatible_measurement_result",
            context={"image_id": image_id},
        )
    if (
        measurement.source_file != FOAMS105_MEASUREMENT_SOURCE_FILE
        or measurement.calibration_source_file != FOAMS105_MEASUREMENT_CALIBRATION_SOURCE_FILE
        or measurement.scope != FOAMS105_MEASUREMENT_SCOPE
    ):
        raise Foams105PerimeterValidationError(
            "phase_measurement_result source metadata is incompatible",
            code="incompatible_measurement_result",
            context={"image_id": image_id},
        )
    if (
        measurement.length_unit != "mm"
        or measurement.area_unit != "mm^2"
        or measurement.pixel_area_unit != "pixels"
        or measurement.scale_unit != "pixels/mm"
    ):
        raise Foams105PerimeterValidationError(
            "phase_measurement_result units are incompatible",
            code="incompatible_measurement_result",
            context={"image_id": image_id},
        )
    if (
        measurement.border_connectivity != FOAMS105_BORDER_CONNECTIVITY
        or measurement.hole_background_connectivity != FOAMS105_HOLE_BACKGROUND_CONNECTIVITY
        or measurement.component_connectivity != FOAMS105_COMPONENT_CONNECTIVITY
        or measurement.component_order_policy != FOAMS105_COMPONENT_ORDER_POLICY
    ):
        raise Foams105PerimeterValidationError(
            "phase_measurement_result topology metadata is incompatible",
            code="incompatible_measurement_result",
            context={"image_id": image_id},
        )
    return measurement


def _validate_labels_and_masks(
    image_id: str,
    measurement: Foams105PhaseMeasurementResult,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not isinstance(measurement.image_shape, tuple) or len(measurement.image_shape) != 2:
        raise Foams105PerimeterValidationError(
            "measurement image_shape must be a (height, width) tuple",
            code="inconsistent_measurement_result",
            context={"image_id": image_id},
        )
    expected_height = _strict_int(
        measurement.image_shape[0],
        "phase_measurement_result.image_shape[0]",
        image_id=image_id,
        minimum=1,
    )
    expected_width = _strict_int(
        measurement.image_shape[1],
        "phase_measurement_result.image_shape[1]",
        image_id=image_id,
        minimum=1,
    )

    filled = measurement.filled_mask
    labels = measurement.label_image
    if not isinstance(filled, np.ndarray) or not isinstance(labels, np.ndarray):
        raise Foams105PerimeterValidationError(
            "filled_mask and label_image must be NumPy arrays",
            code="inconsistent_measurement_result",
            context={"image_id": image_id},
        )
    if filled.ndim != 2 or labels.ndim != 2 or filled.size == 0 or labels.size == 0:
        raise Foams105PerimeterValidationError(
            "filled_mask and label_image must be nonempty two-dimensional arrays",
            code="inconsistent_measurement_result",
            context={"image_id": image_id},
        )
    if filled.shape != labels.shape or filled.shape != (expected_height, expected_width):
        raise Foams105PerimeterValidationError(
            "filled_mask, label_image, and image_shape are inconsistent",
            code="inconsistent_measurement_result",
            context={"image_id": image_id},
        )
    if labels.dtype.kind not in "iu":
        raise Foams105PerimeterValidationError(
            "label_image must contain integer labels",
            code="inconsistent_measurement_result",
            context={"image_id": image_id},
        )

    labels_int = np.asarray(labels, dtype=np.int64)
    if labels_int.min(initial=0) < 0:
        raise Foams105PerimeterValidationError(
            "label_image must use nonnegative labels",
            code="inconsistent_measurement_result",
            context={"image_id": image_id},
        )

    filled_bool = np.asarray(filled, dtype=bool)
    foreground = labels_int > 0
    if not np.array_equal(foreground, filled_bool):
        raise Foams105PerimeterValidationError(
            "filled_mask and label_image foreground are inconsistent",
            code="inconsistent_measurement_result",
            context={"image_id": image_id},
        )
    foreground_count = int(np.count_nonzero(foreground))
    removed_border = np.asarray(measurement.removed_border_mask, dtype=bool)
    if removed_border.shape != (expected_height, expected_width):
        raise Foams105PerimeterValidationError(
            "removed_border_mask shape is inconsistent",
            code="inconsistent_measurement_result",
            context={"image_id": image_id},
        )
    removed_count = int(np.count_nonzero(removed_border))
    selected_count = _strict_int(
        measurement.selected_pixel_count,
        "phase_measurement_result.selected_pixel_count",
        image_id=image_id,
        minimum=0,
    )
    reported_removed_count = _strict_int(
        measurement.removed_border_pixel_count,
        "phase_measurement_result.removed_border_pixel_count",
        image_id=image_id,
        minimum=0,
    )
    retained_count = _strict_int(
        measurement.retained_before_fill_pixel_count,
        "phase_measurement_result.retained_before_fill_pixel_count",
        image_id=image_id,
        minimum=0,
    )
    filled_added_count = _strict_int(
        measurement.filled_added_pixel_count,
        "phase_measurement_result.filled_added_pixel_count",
        image_id=image_id,
        minimum=0,
    )
    reported = _strict_int(
        measurement.final_foreground_pixel_count,
        "phase_measurement_result.final_foreground_pixel_count",
        image_id=image_id,
        minimum=0,
    )
    if (
        foreground_count != reported
        or removed_count != reported_removed_count
        or selected_count != removed_count + retained_count
        or foreground_count != retained_count + filled_added_count
    ):
        raise Foams105PerimeterValidationError(
            "measurement pixel counts are inconsistent",
            code="inconsistent_measurement_result",
            context={
                "image_id": image_id,
                "foreground_count": foreground_count,
                "reported_foreground_count": reported,
                "removed_border_count": removed_count,
                "reported_removed_border_count": reported_removed_count,
                "selected_count": selected_count,
                "retained_count": retained_count,
                "filled_added_count": filled_added_count,
            },
        )
    return filled_bool, labels_int, foreground, foreground_count


def _validate_components(
    image_id: str,
    measurement: Foams105PhaseMeasurementResult,
    labels_int: np.ndarray,
    foreground: np.ndarray,
    foreground_count: int,
    scale_px_per_mm: float,
) -> Tuple[Tuple[Foams105ComponentMeasurement, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    components = measurement.components
    component_count = len(components)
    if component_count == 0:
        if foreground_count != 0:
            raise Foams105PerimeterValidationError(
                "nonempty foreground requires declared components",
                code="inconsistent_measurement_result",
                context={"image_id": image_id},
            )
        empty = np.zeros(1, dtype=np.int64)
        return components, empty, empty, empty, empty

    max_label = int(labels_int.max(initial=0))
    if max_label > component_count:
        raise Foams105PerimeterValidationError(
            "label range exceeds declared component count",
            code="inconsistent_measurement_result",
            context={"image_id": image_id, "max_label": max_label, "component_count": component_count},
        )

    ids = labels_int[foreground]
    rows, cols = np.nonzero(foreground)
    counts = np.bincount(ids, minlength=component_count + 1)
    height = labels_int.shape[0]
    flat_column_major = rows.astype(np.int64) + cols.astype(np.int64) * int(height)
    minima = np.full(component_count + 1, labels_int.size, dtype=np.int64)
    if ids.size:
        np.minimum.at(minima, ids, flat_column_major)

    seen_ids = set()
    for index, component in enumerate(components):
        if not isinstance(component, Foams105ComponentMeasurement):
            raise Foams105PerimeterValidationError(
                "measurement component record type is invalid",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_index": index},
            )
        component_id = _strict_int(
            component.component_id,
            "phase_measurement_result.components.component_id",
            image_id=image_id,
            minimum=1,
        )
        if component_id in seen_ids:
            raise Foams105PerimeterValidationError(
                "measurement component IDs must be unique",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_id": component_id},
            )
        seen_ids.add(component_id)

        if component_id > component_count:
            raise Foams105PerimeterValidationError(
                "measurement component ID exceeds component count",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_id": component_id},
            )

        pixel_count = int(counts[component_id])
        if pixel_count == 0:
            raise Foams105PerimeterValidationError(
                "measurement component ID missing from labels",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_id": component_id},
            )

        declared_area_px = _strict_int(
            component.area_px,
            "phase_measurement_result.components.area_px",
            image_id=image_id,
            component_id=component_id,
            minimum=1,
        )
        if declared_area_px != pixel_count:
            raise Foams105PerimeterValidationError(
                "measurement component area_px is inconsistent",
                code="inconsistent_measurement_result",
                context={
                    "image_id": image_id,
                    "component_id": component_id,
                    "declared_area_px": declared_area_px,
                    "label_area_px": pixel_count,
                },
            )

        first_pixel = component.first_pixel_rc
        if not isinstance(first_pixel, tuple) or len(first_pixel) != 2:
            raise Foams105PerimeterValidationError(
                "measurement first_pixel_rc must be a (row, column) tuple",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_index": index, "component_id": component_id},
            )
        first_row = _strict_int(
            first_pixel[0],
            "phase_measurement_result.components.first_pixel_rc[0]",
            image_id=image_id,
            component_id=component_id,
            minimum=0,
        )
        first_column = _strict_int(
            first_pixel[1],
            "phase_measurement_result.components.first_pixel_rc[1]",
            image_id=image_id,
            component_id=component_id,
            minimum=0,
        )

        minimum = int(minima[component_id])
        first_pixel_from_labels = (minimum % height, minimum // height)
        if (first_row, first_column) != first_pixel_from_labels:
            raise Foams105PerimeterValidationError(
                "measurement first-pixel identity is inconsistent",
                code="inconsistent_measurement_result",
                context={
                    "image_id": image_id,
                    "component_id": component_id,
                    "declared_first_pixel_rc": (first_row, first_column),
                    "label_first_pixel_rc": first_pixel_from_labels,
                },
            )

        area_mm2 = _strict_finite_real(
            component.area_mm2,
            "phase_measurement_result.components.area_mm2",
            image_id=image_id,
            component_id=component_id,
        )
        if area_mm2 <= 0.0:
            raise Foams105PerimeterValidationError(
                "measurement component area_mm2 must be positive",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_id": component_id},
            )
        equivalent_diameter_mm = _strict_finite_real(
            component.equivalent_diameter_mm,
            "phase_measurement_result.components.equivalent_diameter_mm",
            image_id=image_id,
            component_id=component_id,
        )
        if equivalent_diameter_mm <= 0.0:
            raise Foams105PerimeterValidationError(
                "measurement component equivalent_diameter_mm must be positive",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_id": component_id},
            )
        expected_area_mm2 = (float(pixel_count) / scale_px_per_mm) / scale_px_per_mm
        expected_diameter_mm = 2.0 * math.sqrt(expected_area_mm2 / math.pi)
        if area_mm2 != expected_area_mm2 or equivalent_diameter_mm != expected_diameter_mm:
            raise Foams105PerimeterValidationError(
                "measurement component calibration is inconsistent",
                code="inconsistent_measurement_result",
                context={
                    "image_id": image_id,
                    "component_id": component_id,
                    "expected_area_mm2": expected_area_mm2,
                    "actual_area_mm2": area_mm2,
                    "expected_equivalent_diameter_mm": expected_diameter_mm,
                    "actual_equivalent_diameter_mm": equivalent_diameter_mm,
                },
            )

    if len(seen_ids) != component_count:
        raise Foams105PerimeterValidationError(
            "measurement components must define contiguous IDs 1..N",
            code="inconsistent_measurement_result",
            context={"image_id": image_id, "component_count": component_count, "unique_ids": len(seen_ids)},
        )

    return components, rows.astype(np.int32), cols.astype(np.int32), ids.astype(np.int32), counts.astype(np.int32)


def _group_component_points(
    rows: np.ndarray,
    cols: np.ndarray,
    ids: np.ndarray,
    component_count: int,
) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]:
    grouped: list[list[Tuple[int, int]]] = [[] for _ in range(component_count + 1)]
    for row, col, component_id in zip(rows.tolist(), cols.tolist(), ids.tolist()):
        grouped[int(component_id)].append((int(row), int(col)))
    points: list[Tuple[Tuple[int, int], ...]] = [tuple()] * (component_count + 1)
    starts: list[Tuple[int, int]] = [(0, 0)] * (component_count + 1)
    for component_id in range(1, component_count + 1):
        component_points = tuple(grouped[component_id])
        points[component_id] = component_points
        starts[component_id] = component_points[0]
    return tuple(points), tuple(starts)


def _validate_connected_filled_component(
    image_id: str,
    component_id: int,
    points: Tuple[Tuple[int, int], ...],
) -> None:
    if not points:
        raise Foams105PerimeterValidationError(
            "component has no pixels",
            code="inconsistent_measurement_result",
            context={"image_id": image_id, "component_id": component_id},
        )

    point_array = np.asarray(points, dtype=np.intp)
    row_min = int(point_array[:, 0].min())
    col_min = int(point_array[:, 1].min())
    row_max = int(point_array[:, 0].max())
    col_max = int(point_array[:, 1].max())
    height = row_max - row_min + 1
    width = col_max - col_min + 1
    padded_bbox_area = (height + 2) * (width + 2)
    if padded_bbox_area > 8 * len(points):
        _validate_sparse_component_topology(image_id, component_id, points)
        return

    component_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )
    local_points = point_array - np.array((row_min, col_min), dtype=np.intp)
    component_mask[local_points[:, 0], local_points[:, 1]] = 1

    component_count = cv2.connectedComponents(component_mask, connectivity=4)[0] - 1
    if component_count != 1:
        raise Foams105PerimeterValidationError(
            "component foreground is not four-connected",
            code="invalid_component_topology",
            context={"image_id": image_id, "component_id": component_id},
        )

    padded_background = np.pad(component_mask, 1, mode="constant", constant_values=0)
    cv2.floodFill(padded_background, None, (0, 0), 2, flags=4)
    if np.any(padded_background[1:-1, 1:-1] == 0):
        _validate_sparse_component_topology(image_id, component_id, points)


def _validate_sparse_component_topology(
    image_id: str,
    component_id: int,
    points: Tuple[Tuple[int, int], ...],
) -> None:
    point_set = set(points)
    stack = [next(iter(point_set))]
    visited = set()
    for_connected_neighbors = ((-1, 0), (0, 1), (1, 0), (0, -1))
    while stack:
        row, col = stack.pop()
        if (row, col) in visited:
            continue
        visited.add((row, col))
        for row_delta, col_delta in for_connected_neighbors:
            neighbor = (row + row_delta, col + col_delta)
            if neighbor in point_set and neighbor not in visited:
                stack.append(neighbor)
    if len(visited) != len(point_set):
        raise Foams105PerimeterValidationError(
            "component foreground is not four-connected",
            code="invalid_component_topology",
            context={"image_id": image_id, "component_id": component_id},
        )

    vertices = set()
    shared_edges = 0
    for row, col in points:
        vertices.update(((row, col), (row + 1, col), (row, col + 1), (row + 1, col + 1)))
        shared_edges += (row - 1, col) in point_set
        shared_edges += (row, col - 1) in point_set
    euler_characteristic = len(vertices) - (3 * len(points)) + shared_edges
    if euler_characteristic != 1:
        raise Foams105PerimeterValidationError(
            "component foreground contains enclosed holes",
            code="invalid_component_topology",
            context={
                "image_id": image_id,
                "component_id": component_id,
                "euler_characteristic": euler_characteristic,
            },
        )


def _trace_component_boundary(
    points: Tuple[Tuple[int, int], ...],
    *,
    image_id: str,
    component_id: int,
) -> Tuple[int, int, int, float]:
    point_set = set(points)
    start = min(point_set)
    current = start
    backtrack = (start[0], start[1] - 1)
    directions = (
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    )

    first_edge: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    boundary = [start]
    closed = False

    for _ in range((8 * len(points)) + 8):
        vector = (backtrack[0] - current[0], backtrack[1] - current[1])
        if vector not in directions:
            raise Foams105PerimeterNumericalError(
                "trace backtrack vector is invalid",
                code="invalid_trace_state",
                context={"image_id": image_id, "component_id": component_id},
            )
        start_index = directions.index(vector)

        found: Optional[Tuple[int, Tuple[int, int]]] = None
        for offset in range(8):
            index = (start_index + offset) % 8
            dr, dc = directions[index]
            candidate = (current[0] + dr, current[1] + dc)
            if candidate in point_set:
                found = (index, candidate)
                break

        if found is None:
            # Isolated pixel.
            return 0, 0, 0, 0.0

        index, nxt = found
        edge = (current, nxt)
        if first_edge is None:
            first_edge = edge
        elif edge == first_edge:
            closed = True
            break

        previous = directions[(index - 1) % 8]
        backtrack = (current[0] + previous[0], current[1] + previous[1])
        current = nxt
        boundary.append(current)

    if not closed:
        raise Foams105PerimeterNumericalError(
            "component boundary trace did not close within finite bound",
            code="nonclosing_boundary_trace",
            context={
                "image_id": image_id,
                "component_id": component_id,
                "trace_bound": (8 * len(points)) + 8,
            },
        )

    step_classes = []
    old_perimeter = 0.0
    axial = 0
    diagonal = 0
    for source, target in zip(boundary, boundary[1:]):
        dr = target[0] - source[0]
        dc = target[1] - source[1]
        step_class = (dr * dr, dc * dc)
        if step_class not in ((1, 0), (0, 1), (1, 1)):
            raise Foams105PerimeterNumericalError(
                "component boundary has unsupported step class",
                code="invalid_trace_step",
                context={
                    "image_id": image_id,
                    "component_id": component_id,
                    "source": source,
                    "target": target,
                },
            )
        step_classes.append(step_class)
        if step_class == (1, 1):
            diagonal += 1
            old_perimeter += math.sqrt(2.0)
        else:
            axial += 1
            old_perimeter += 1.0

    transition_count = 0
    if step_classes:
        for index, step_class in enumerate(step_classes):
            if step_class != step_classes[(index + 1) % len(step_classes)]:
                transition_count += 1

    return axial, diagonal, transition_count, old_perimeter


def _perimeter_for_profile(
    profile: str,
    *,
    axial: int,
    diagonal: int,
    transitions: int,
    old_perimeter: float,
    image_id: str,
    component_id: int,
) -> float:
    if axial == 0 and diagonal == 0:
        return 0.0
    if profile == FOAMS105_PERIMETER_PROFILE_OLD:
        perimeter = old_perimeter
    else:
        perimeter = (0.980 * float(axial)) + (1.406 * float(diagonal)) - (0.091 * float(transitions))

    if not math.isfinite(perimeter):
        raise Foams105PerimeterNumericalError(
            "perimeter arithmetic is nonfinite",
            code="nonfinite_perimeter",
            context={"image_id": image_id, "component_id": component_id, "profile": profile},
        )
    if perimeter < 0.0:
        raise Foams105PerimeterNumericalError(
            "perimeter arithmetic is negative",
            code="negative_perimeter",
            context={"image_id": image_id, "component_id": component_id, "profile": profile, "perimeter_px": perimeter},
        )
    if perimeter == 0.0:
        raise Foams105PerimeterNumericalError(
            "perimeter underflow produced zero for non-singleton component",
            code="underflow_perimeter",
            context={"image_id": image_id, "component_id": component_id, "profile": profile},
        )
    return perimeter


def measure_foams105_perimeter(
    image_id: str,
    phase_measurement_result: Foams105PhaseMeasurementResult,
    *,
    profile: str,
) -> Foams105ImagePerimeterResult:
    """Measure component perimeter using one explicit MATLAB-verified profile."""

    normalized_image_id = _validate_image_id(image_id)
    normalized_profile = _validate_profile(profile)
    measurement = _validate_phase_measurement(normalized_image_id, phase_measurement_result)
    scale = _finite_positive_real(
        measurement.scale_px_per_mm,
        "phase_measurement_result.scale_px_per_mm",
        image_id=normalized_image_id,
    )

    _, labels_int, foreground, foreground_count = _validate_labels_and_masks(
        normalized_image_id,
        measurement,
    )
    components, rows, cols, ids, counts = _validate_components(
        normalized_image_id,
        measurement,
        labels_int,
        foreground,
        foreground_count,
        scale,
    )

    if not components:
        return Foams105ImagePerimeterResult(
            image_id=normalized_image_id,
            phase_measurement_result=measurement,
            profile=normalized_profile,
            components=(),
            component_count=0,
            foreground_pixel_count=0,
            scale_px_per_mm=scale,
        )

    grouped_points, _ = _group_component_points(rows, cols, ids, len(components))

    result_components = []
    for component in components:
        component_id = int(component.component_id)
        points = grouped_points[component_id]
        _validate_connected_filled_component(normalized_image_id, component_id, points)
        axial, diagonal, transitions, old_perimeter = _trace_component_boundary(
            points,
            image_id=normalized_image_id,
            component_id=component_id,
        )
        perimeter_px = _perimeter_for_profile(
            normalized_profile,
            axial=axial,
            diagonal=diagonal,
            transitions=transitions,
            old_perimeter=old_perimeter,
            image_id=normalized_image_id,
            component_id=component_id,
        )

        perimeter_mm = perimeter_px / scale
        if not math.isfinite(perimeter_mm) or perimeter_mm < 0.0:
            raise Foams105PerimeterNumericalError(
                "calibrated perimeter is invalid",
                code="invalid_calibrated_perimeter",
                context={
                    "image_id": normalized_image_id,
                    "component_id": component_id,
                    "perimeter_px": perimeter_px,
                    "scale_px_per_mm": scale,
                },
            )
        if perimeter_px > 0.0 and perimeter_mm == 0.0:
            raise Foams105PerimeterNumericalError(
                "calibrated perimeter underflowed to zero",
                code="underflow_calibrated_perimeter",
                context={
                    "image_id": normalized_image_id,
                    "component_id": component_id,
                    "perimeter_px": perimeter_px,
                    "scale_px_per_mm": scale,
                },
            )

        result_components.append(
            Foams105PerimeterComponent(
                image_id=normalized_image_id,
                component_id=component_id,
                first_pixel_rc=component.first_pixel_rc,
                area_px=int(counts[component_id]),
                area_mm2=float(component.area_mm2),
                equivalent_diameter_mm=float(component.equivalent_diameter_mm),
                scale_px_per_mm=scale,
                axial_step_count=axial,
                diagonal_step_count=diagonal,
                corner_class_transition_count=transitions,
                perimeter_px=float(perimeter_px),
                perimeter_mm=float(perimeter_mm),
                profile=normalized_profile,
            )
        )

    return Foams105ImagePerimeterResult(
        image_id=normalized_image_id,
        phase_measurement_result=measurement,
        profile=normalized_profile,
        components=tuple(result_components),
        component_count=len(result_components),
        foreground_pixel_count=foreground_count,
        scale_px_per_mm=scale,
    )


def _canonical_raw_selection(result: object) -> Foams105RawSelectionResult:
    if not isinstance(result, Foams105RawSelectionResult):
        raise Foams105PerimeterValidationError("raw_selection_result must be a Foams105RawSelectionResult")
    if result.method != FOAMS105_RAW_SELECTION_METHOD:
        raise Foams105PerimeterValidationError(
            "raw_selection_result method is incompatible",
            code="incompatible_raw_selection_result",
            context={"expected_method": FOAMS105_RAW_SELECTION_METHOD, "actual_method": result.method},
        )
    canonical = select_foams105_raw_objects(
        result.bin_labels_mm,
        result.groups,
        result.requested_ranges,
    )
    if canonical != result:
        raise Foams105PerimeterValidationError(
            "raw_selection_result is not canonical for supplied groups and ranges",
            code="inconsistent_raw_selection_result",
        )
    return canonical


def _validate_perimeter_result_contract(
    supplied: object,
    *,
    index: int,
) -> Foams105ImagePerimeterResult:
    if not isinstance(supplied, Foams105ImagePerimeterResult):
        raise Foams105PerimeterValidationError(
            "image_perimeter_results contains an invalid result record",
            code="inconsistent_perimeter_result",
            context={"result_index": index},
        )
    profile = _validate_profile(supplied.profile)
    canonical = measure_foams105_perimeter(
        supplied.image_id,
        supplied.phase_measurement_result,
        profile=profile,
    )
    supplied_measurement = supplied.phase_measurement_result
    canonical_measurement = canonical.phase_measurement_result
    measurements_match = (
        supplied_measurement.image_shape == canonical_measurement.image_shape
        and supplied_measurement.phase_value == canonical_measurement.phase_value
        and supplied_measurement.scale_px_per_mm == canonical_measurement.scale_px_per_mm
        and supplied_measurement.components == canonical_measurement.components
        and supplied_measurement.selected_pixel_count
        == canonical_measurement.selected_pixel_count
        and supplied_measurement.removed_border_pixel_count
        == canonical_measurement.removed_border_pixel_count
        and supplied_measurement.retained_before_fill_pixel_count
        == canonical_measurement.retained_before_fill_pixel_count
        and supplied_measurement.filled_added_pixel_count
        == canonical_measurement.filled_added_pixel_count
        and supplied_measurement.final_foreground_pixel_count
        == canonical_measurement.final_foreground_pixel_count
        and supplied_measurement.method == canonical_measurement.method
        and supplied_measurement.source_repository == canonical_measurement.source_repository
        and supplied_measurement.source_commit == canonical_measurement.source_commit
        and supplied_measurement.source_file == canonical_measurement.source_file
        and supplied_measurement.calibration_source_file
        == canonical_measurement.calibration_source_file
        and supplied_measurement.scope == canonical_measurement.scope
        and supplied_measurement.border_connectivity == canonical_measurement.border_connectivity
        and supplied_measurement.hole_background_connectivity
        == canonical_measurement.hole_background_connectivity
        and supplied_measurement.component_connectivity
        == canonical_measurement.component_connectivity
        and supplied_measurement.component_order_policy
        == canonical_measurement.component_order_policy
        and supplied_measurement.pixel_area_unit
        == canonical_measurement.pixel_area_unit
        and supplied_measurement.length_unit == canonical_measurement.length_unit
        and supplied_measurement.area_unit == canonical_measurement.area_unit
        and supplied_measurement.scale_unit == canonical_measurement.scale_unit
        and np.array_equal(
            np.asarray(supplied_measurement.removed_border_mask, dtype=np.bool_),
            np.asarray(canonical_measurement.removed_border_mask, dtype=np.bool_),
        )
        and np.array_equal(
            np.asarray(supplied_measurement.filled_mask, dtype=np.bool_),
            np.asarray(canonical_measurement.filled_mask, dtype=np.bool_),
        )
        and np.array_equal(
            np.asarray(supplied_measurement.label_image, dtype=np.int32),
            np.asarray(canonical_measurement.label_image, dtype=np.int32),
        )
    )
    result_match = (
        supplied.image_id == canonical.image_id
        and supplied.profile == canonical.profile
        and supplied.components == canonical.components
        and supplied.component_count == canonical.component_count
        and supplied.foreground_pixel_count == canonical.foreground_pixel_count
        and supplied.scale_px_per_mm == canonical.scale_px_per_mm
        and supplied.method == canonical.method
        and supplied.source_repository == canonical.source_repository
        and supplied.source_commit == canonical.source_commit
        and supplied.source_files == canonical.source_files
        and supplied.scope == canonical.scope
        and supplied.connectivity == canonical.connectivity
        and supplied.verification_level == canonical.verification_level
        and supplied.reference_runtime == canonical.reference_runtime
        and supplied.reference_toolbox == canonical.reference_toolbox
        and supplied.evidence_version == canonical.evidence_version
        and supplied.evidence_hash == canonical.evidence_hash
        and supplied.length_unit == canonical.length_unit
        and supplied.area_unit == canonical.area_unit
        and supplied.pixel_length_unit == canonical.pixel_length_unit
        and measurements_match
    )
    if not result_match:
        raise Foams105PerimeterValidationError(
            "image_perimeter_result is not canonical for supplied measurement and profile",
            code="inconsistent_perimeter_result",
            context={"image_id": supplied.image_id, "result_index": index},
        )
    return canonical


def _validated_perimeter_results(
    values: object,
) -> Tuple[Tuple[Foams105ImagePerimeterResult, ...], Mapping[Tuple[str, int], Foams105PerimeterComponent], str]:
    if isinstance(values, (str, bytes)):
        raise Foams105PerimeterValidationError("image_perimeter_results must be a sequence")
    try:
        raw_results = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105PerimeterValidationError("image_perimeter_results must be a sequence") from exc

    image_ids = set()
    identity_map: Dict[Tuple[str, int], Foams105PerimeterComponent] = {}
    normalized = []
    profile: Optional[str] = None
    for index, supplied in enumerate(raw_results):
        canonical = _validate_perimeter_result_contract(supplied, index=index)
        if canonical.image_id in image_ids:
            raise Foams105PerimeterValidationError(
                "duplicate image_id in image_perimeter_results",
                code="duplicate_image_id",
                context={"image_id": canonical.image_id},
            )
        image_ids.add(canonical.image_id)
        if profile is None:
            profile = canonical.profile
        elif profile != canonical.profile:
            raise Foams105PerimeterValidationError(
                "image_perimeter_results include mixed profiles",
                code="mixed_perimeter_profiles",
                context={"first_profile": profile, "current_profile": canonical.profile},
            )
        for component in canonical.components:
            identity = (component.image_id, component.component_id)
            if identity in identity_map:
                raise Foams105PerimeterValidationError(
                    "duplicate component identity in image_perimeter_results",
                    code="duplicate_component_identity",
                    context={"identity": identity},
                )
            identity_map[identity] = component
        normalized.append(canonical)

    return tuple(normalized), MappingProxyType(identity_map), profile or ""


def select_foams105_perimeter(
    raw_selection: Foams105RawSelectionResult,
    image_perimeter_results: Sequence[Foams105ImagePerimeterResult],
) -> Foams105SelectedPerimeterResult:
    """Join raw-selection identities to perimeter measurements in output order."""

    canonical_raw = _canonical_raw_selection(raw_selection)
    normalized_results, identity_map, profile = _validated_perimeter_results(image_perimeter_results)

    for group in canonical_raw.groups:
        for object_index, object_ in enumerate(group.objects):
            identity = (object_.image_id, object_.component_id)
            component = identity_map.get(identity)
            if component is None:
                raise Foams105PerimeterValidationError(
                    "raw-selection object identity is missing from perimeter results",
                    code="missing_perimeter_identity",
                    context={
                        "group_id": group.group_id,
                        "source_slot": group.source_slot,
                        "source_object_index": object_index,
                        "identity": identity,
                    },
                )
            if (
                component.area_mm2 != object_.area_mm2
                or component.equivalent_diameter_mm != object_.equivalent_diameter_mm
            ):
                raise Foams105PerimeterValidationError(
                    "raw-selection perimeter metadata changed for identity",
                    code="changed_perimeter_identity",
                    context={"identity": identity},
                )

    rows = []
    for output_index, row in enumerate(canonical_raw.rows):
        identity = (row.image_id, row.component_id)
        component = identity_map.get(identity)
        if component is None:
            raise Foams105PerimeterValidationError(
                "selected row identity is missing from perimeter results",
                code="missing_perimeter_identity",
                context={"identity": identity, "raw_output_index": row.output_index},
            )
        if (
            component.area_mm2 != row.area_mm2
            or component.equivalent_diameter_mm != row.equivalent_diameter_mm
        ):
            raise Foams105PerimeterValidationError(
                "selected-row perimeter metadata changed for identity",
                code="changed_perimeter_identity",
                context={"identity": identity, "raw_output_index": row.output_index},
            )
        rows.append(
            Foams105SelectedPerimeterRow(
                output_index=output_index,
                raw_output_index=row.output_index,
                group_id=row.group_id,
                source_slot=row.source_slot,
                source_object_index=row.source_object_index,
                image_id=row.image_id,
                component_id=row.component_id,
                area_mm2=row.area_mm2,
                equivalent_diameter_mm=row.equivalent_diameter_mm,
                scale_px_per_mm=component.scale_px_per_mm,
                perimeter_px=component.perimeter_px,
                perimeter_mm=component.perimeter_mm,
                axial_step_count=component.axial_step_count,
                diagonal_step_count=component.diagonal_step_count,
                corner_class_transition_count=component.corner_class_transition_count,
                profile=component.profile,
                component_perimeter=component,
            )
        )

    return Foams105SelectedPerimeterResult(
        raw_selection_result=canonical_raw,
        image_perimeter_results=normalized_results,
        profile=profile,
        rows=tuple(rows),
        selected_identities=tuple((row.image_id, row.component_id) for row in rows),
        selected_count=len(rows),
    )
