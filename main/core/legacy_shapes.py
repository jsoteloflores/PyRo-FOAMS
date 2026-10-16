"""FOAMS 1.0.5 component moment geometry and shape-identity replay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .legacy_measurements import (
    FOAMS105_COMPONENT_CONNECTIVITY,
    FOAMS105_MEASUREMENT_METHOD,
    FOAMS105_MEASUREMENT_SOURCE_COMMIT,
    FOAMS105_MEASUREMENT_SOURCE_REPOSITORY,
    Foams105ComponentMeasurement,
    Foams105PhaseMeasurementResult,
)
from .legacy_raw_selection import (
    FOAMS105_RAW_SELECTION_METHOD,
    Foams105RawSelectionResult,
    select_foams105_raw_objects,
)

FOAMS105_MOMENT_METHOD = "foams_1_0_5_moment_geometry_replay_v1"
FOAMS105_MOMENT_SELECTION_METHOD = "foams_1_0_5_moment_geometry_selection_replay_v1"
FOAMS105_MOMENT_SOURCE_REPOSITORY = FOAMS105_MEASUREMENT_SOURCE_REPOSITORY
FOAMS105_MOMENT_SOURCE_COMMIT = FOAMS105_MEASUREMENT_SOURCE_COMMIT
FOAMS105_MOMENT_SOURCE_FILES = ("data_meas.m", "project.m")
FOAMS105_MOMENT_SCOPE = "component_moment_geometry_only"
FOAMS105_MOMENT_SELECTION_SCOPE = "raw_identity_shape_selection_only"
FOAMS105_MOMENT_VERIFICATION_LEVEL = "analytical_pixel_moment_profile"
FOAMS105_PERIMETER_STATUS_PENDING = "pending_historical_profile"
FOAMS105_ORIENTATION_TIE_REASON = "isotropic_second_moments"
FOAMS105_PROJECT_ORIENTATION_OFFSET_DEG = 90.0
FOAMS105_SELECTION_RELATIONSHIP_POLICY = (
    "raw_selection_objects_must_match_geometry_identity_and_metadata"
)


class Foams105MomentValidationError(ValueError):
    """Raised when moment-geometry inputs violate the bounded contract."""

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


class Foams105MomentNumericalError(ArithmeticError):
    """Raised when finite moment arithmetic becomes nonphysical or nonfinite."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        context: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = MappingProxyType(dict(context or {}))


@dataclass(frozen=True)
class Foams105MomentComponentGeometry:
    """One measured component with unit-square central-moment geometry."""

    image_id: str
    component_id: int
    first_pixel_rc: Tuple[int, int]
    pixel_count: int
    area_px: int
    area_mm2: float
    equivalent_diameter_mm: float
    centroid_x_px: float
    centroid_y_px: float
    central_cxx: float
    central_cxy: float
    central_cyy: float
    lambda_major: float
    lambda_minor: float
    major_axis_px: float
    minor_axis_px: float
    major_axis_mm: float
    minor_axis_mm: float
    eccentricity: float
    orientation_regionprops_profile_deg: float
    orientation_project_deg: float
    orientation_defined: bool
    orientation_tie_reason: Optional[str]
    perimeter_status: str = FOAMS105_PERIMETER_STATUS_PENDING


@dataclass(frozen=True)
class Foams105ImageMomentGeometryResult:
    """Immutable moment geometry for one accepted measured phase."""

    image_id: str
    phase_measurement_result: Foams105PhaseMeasurementResult
    components: Tuple[Foams105MomentComponentGeometry, ...]
    component_count: int
    foreground_pixel_count: int
    scale_px_per_mm: float
    method: str = FOAMS105_MOMENT_METHOD
    source_repository: str = FOAMS105_MOMENT_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_MOMENT_SOURCE_COMMIT
    source_files: Tuple[str, str] = FOAMS105_MOMENT_SOURCE_FILES
    scope: str = FOAMS105_MOMENT_SCOPE
    connectivity: int = FOAMS105_COMPONENT_CONNECTIVITY
    verification_level: str = FOAMS105_MOMENT_VERIFICATION_LEVEL
    perimeter_status: str = FOAMS105_PERIMETER_STATUS_PENDING
    orientation_tie_reason: str = FOAMS105_ORIENTATION_TIE_REASON
    orientation_project_offset_deg: float = FOAMS105_PROJECT_ORIENTATION_OFFSET_DEG
    length_unit: str = "mm"
    pixel_length_unit: str = "pixel"
    area_unit: str = "mm^2"
    pixel_area_unit: str = "pixel-area"
    angle_unit: str = "degree"


@dataclass(frozen=True)
class Foams105SelectedMomentGeometryRow:
    """One raw-selected object joined with its moment-geometry identity."""

    output_index: int
    raw_output_index: int
    group_id: str
    source_slot: int
    source_object_index: int
    image_id: str
    component_id: int
    area_mm2: float
    equivalent_diameter_mm: float
    major_axis_px: float
    minor_axis_px: float
    major_axis_mm: float
    minor_axis_mm: float
    eccentricity: float
    orientation_regionprops_profile_deg: float
    orientation_project_deg: float
    orientation_defined: bool
    orientation_tie_reason: Optional[str]
    component_geometry: Foams105MomentComponentGeometry


@dataclass(frozen=True)
class Foams105SelectedMomentGeometryResult:
    """Immutable selected moment-geometry rows in raw-selection output order."""

    raw_selection_result: Foams105RawSelectionResult
    image_geometry_results: Tuple[Foams105ImageMomentGeometryResult, ...]
    rows: Tuple[Foams105SelectedMomentGeometryRow, ...]
    selected_identities: Tuple[Tuple[str, int], ...]
    selected_count: int
    method: str = FOAMS105_MOMENT_SELECTION_METHOD
    source_repository: str = FOAMS105_MOMENT_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_MOMENT_SOURCE_COMMIT
    source_files: Tuple[str, str] = FOAMS105_MOMENT_SOURCE_FILES
    scope: str = FOAMS105_MOMENT_SELECTION_SCOPE
    verification_level: str = FOAMS105_MOMENT_VERIFICATION_LEVEL
    perimeter_status: str = FOAMS105_PERIMETER_STATUS_PENDING
    relationship_policy: str = FOAMS105_SELECTION_RELATIONSHIP_POLICY
    length_unit: str = "mm"
    area_unit: str = "mm^2"
    angle_unit: str = "degree"


def _finite_positive_real(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise Foams105MomentValidationError(
            f"{field} must be a finite positive real scalar",
            context={"field": field},
        )
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise Foams105MomentValidationError(
            f"{field} must be representable as finite float64",
            context={"field": field},
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise Foams105MomentValidationError(
            f"{field} must be finite and positive",
            context={"field": field},
        )
    return result


def _validate_image_id(image_id: object) -> str:
    if not isinstance(image_id, str) or not image_id.strip():
        raise Foams105MomentValidationError(
            "image_id must be a nonempty string",
            context={"field": "image_id"},
        )
    return image_id


def _numerical_error(
    code: str,
    stage: str,
    *,
    image_id: str,
    component_id: Optional[int] = None,
    operands: Sequence[float] = (),
) -> Foams105MomentNumericalError:
    return Foams105MomentNumericalError(
        code,
        f"{stage} produced an invalid float64 result",
        context={
            "stage": stage,
            "image_id": image_id,
            "component_id": component_id,
            "operands": tuple(operands),
        },
    )


def _checked_divide(
    numerator: float,
    denominator: float,
    *,
    stage: str,
    image_id: str,
    component_id: Optional[int] = None,
) -> float:
    result = numerator / denominator
    if not math.isfinite(result):
        raise _numerical_error(
            "nonfinite_shape_arithmetic",
            stage,
            image_id=image_id,
            component_id=component_id,
            operands=(numerator, denominator),
        )
    if numerator != 0.0 and result == 0.0:
        raise _numerical_error(
            "nonzero_underflow",
            stage,
            image_id=image_id,
            component_id=component_id,
            operands=(numerator, denominator),
        )
    return result


def _checked_sqrt(
    value: float,
    *,
    stage: str,
    image_id: str,
    component_id: int,
) -> float:
    try:
        result = math.sqrt(value)
    except ValueError as exc:
        raise _numerical_error(
            "invalid_shape_geometry",
            stage,
            image_id=image_id,
            component_id=component_id,
            operands=(value,),
        ) from exc
    if not math.isfinite(result):
        raise _numerical_error(
            "nonfinite_shape_arithmetic",
            stage,
            image_id=image_id,
            component_id=component_id,
            operands=(value,),
        )
    return result


def _validate_phase_measurement(
    image_id: str,
    result: object,
) -> Foams105PhaseMeasurementResult:
    if not isinstance(result, Foams105PhaseMeasurementResult):
        raise Foams105MomentValidationError(
            "phase_measurement_result must be a Foams105PhaseMeasurementResult",
            context={"image_id": image_id},
        )
    if result.method != FOAMS105_MEASUREMENT_METHOD:
        raise Foams105MomentValidationError(
            "phase_measurement_result method is incompatible",
            code="incompatible_measurement_result",
            context={
                "image_id": image_id,
                "expected_method": FOAMS105_MEASUREMENT_METHOD,
                "actual_method": result.method,
            },
        )
    if result.source_repository != FOAMS105_MOMENT_SOURCE_REPOSITORY:
        raise Foams105MomentValidationError(
            "phase_measurement_result source repository is incompatible",
            code="incompatible_measurement_result",
            context={"image_id": image_id},
        )
    if result.source_commit != FOAMS105_MOMENT_SOURCE_COMMIT:
        raise Foams105MomentValidationError(
            "phase_measurement_result source commit is incompatible",
            code="incompatible_measurement_result",
            context={"image_id": image_id},
        )
    if result.length_unit != "mm" or result.area_unit != "mm^2":
        raise Foams105MomentValidationError(
            "phase_measurement_result units are incompatible",
            code="incompatible_measurement_result",
            context={"image_id": image_id},
        )
    if result.component_connectivity != FOAMS105_COMPONENT_CONNECTIVITY:
        raise Foams105MomentValidationError(
            "phase_measurement_result connectivity is incompatible",
            code="incompatible_measurement_result",
            context={"image_id": image_id},
        )
    return result


def _validate_labels_and_masks(
    image_id: str,
    measurement: Foams105PhaseMeasurementResult,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    filled = measurement.filled_mask
    labels = measurement.label_image
    if not isinstance(filled, np.ndarray) or filled.ndim != 2:
        raise Foams105MomentValidationError(
            "filled_mask must be a nonempty two-dimensional array",
            context={"image_id": image_id},
        )
    if not isinstance(labels, np.ndarray) or labels.ndim != 2:
        raise Foams105MomentValidationError(
            "label_image must be a nonempty two-dimensional array",
            context={"image_id": image_id},
        )
    if filled.size == 0 or labels.size == 0 or filled.shape != labels.shape:
        raise Foams105MomentValidationError(
            "filled_mask and label_image must have equal nonempty shapes",
            context={"image_id": image_id},
        )
    if filled.shape != measurement.image_shape:
        raise Foams105MomentValidationError(
            "measurement image_shape and array shapes are inconsistent",
            context={"image_id": image_id},
        )
    if labels.dtype.kind not in "iu":
        raise Foams105MomentValidationError(
            "label_image must contain integer labels",
            context={"image_id": image_id},
        )
    labels_int = np.asarray(labels, dtype=np.int64)
    if labels_int.min(initial=0) < 0:
        raise Foams105MomentValidationError(
            "label_image must use nonnegative labels",
            context={"image_id": image_id},
        )
    filled_bool = np.asarray(filled, dtype=bool)
    foreground = labels_int > 0
    if not bool(np.array_equal(foreground, filled_bool)):
        raise Foams105MomentValidationError(
            "filled_mask and label_image foreground are inconsistent",
            code="inconsistent_measurement_result",
            context={"image_id": image_id},
        )
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count != measurement.final_foreground_pixel_count:
        raise Foams105MomentValidationError(
            "measurement foreground count is inconsistent",
            code="inconsistent_measurement_result",
            context={
                "image_id": image_id,
                "label_foreground_count": foreground_count,
                "reported_foreground_count": measurement.final_foreground_pixel_count,
            },
        )
    return filled_bool, labels_int, foreground, foreground_count


def _validate_components(
    image_id: str,
    measurement: Foams105PhaseMeasurementResult,
    labels_int: np.ndarray,
    foreground: np.ndarray,
    foreground_count: int,
) -> Tuple[Tuple[Foams105ComponentMeasurement, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    components = measurement.components
    if not isinstance(components, tuple):
        raise Foams105MomentValidationError(
            "measurement components must be an immutable tuple",
            context={"image_id": image_id},
        )
    component_count = len(components)
    if component_count == 0:
        if foreground_count != 0:
            raise Foams105MomentValidationError(
                "nonempty foreground requires declared components",
                code="inconsistent_measurement_result",
                context={"image_id": image_id},
            )
        empty = np.zeros(1, dtype=np.float64)
        return components, empty, empty, np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)

    max_label = int(labels_int.max(initial=0))
    if max_label > component_count:
        raise Foams105MomentValidationError(
            "label range exceeds declared component count",
            code="inconsistent_measurement_result",
            context={
                "image_id": image_id,
                "max_label": max_label,
                "component_count": component_count,
            },
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
            raise Foams105MomentValidationError(
                "measurement components contain an invalid record",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_index": index},
            )
        if component.component_id in seen_ids:
            raise Foams105MomentValidationError(
                "measurement components contain duplicate IDs",
                code="inconsistent_measurement_result",
                context={
                    "image_id": image_id,
                    "component_id": component.component_id,
                },
            )
        seen_ids.add(component.component_id)
        component_id = int(component.component_id)
        if component_id <= 0 or component_id > component_count:
            raise Foams105MomentValidationError(
                "measurement component IDs must be contiguous positive indices",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_id": component_id},
            )
        pixel_count = int(counts[component_id])
        if pixel_count <= 0:
            raise Foams105MomentValidationError(
                "measurement component is missing from the label image",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_id": component_id},
            )
        if component.area_px != pixel_count:
            raise Foams105MomentValidationError(
                "measurement component area_px is inconsistent",
                code="inconsistent_measurement_result",
                context={
                    "image_id": image_id,
                    "component_id": component_id,
                    "declared_area_px": component.area_px,
                    "label_area_px": pixel_count,
                },
            )
        minimum = int(minima[component_id])
        first_pixel = (minimum % height, minimum // height)
        if component.first_pixel_rc != first_pixel:
            raise Foams105MomentValidationError(
                "measurement first-pixel identity is inconsistent",
                code="inconsistent_measurement_result",
                context={
                    "image_id": image_id,
                    "component_id": component_id,
                    "declared_first_pixel_rc": component.first_pixel_rc,
                    "label_first_pixel_rc": first_pixel,
                },
            )
        expected_area_mm2 = _checked_divide(
            _checked_divide(
                float(pixel_count),
                measurement.scale_px_per_mm,
                stage="calibrated_area_divide_scale",
                image_id=image_id,
                component_id=component_id,
            ),
            measurement.scale_px_per_mm,
            stage="calibrated_area_divide_scale_again",
            image_id=image_id,
            component_id=component_id,
        )
        if not math.isclose(component.area_mm2, expected_area_mm2, rel_tol=1e-12, abs_tol=0.0):
            raise Foams105MomentValidationError(
                "measurement area_mm2 is inconsistent with label area",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_id": component_id},
            )
        expected_diameter = 2.0 * math.sqrt(component.area_mm2 / math.pi)
        if not math.isclose(
            component.equivalent_diameter_mm,
            expected_diameter,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise Foams105MomentValidationError(
                "measurement equivalent diameter is inconsistent",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_id": component_id},
            )

    return components, rows.astype(np.float64), cols.astype(np.float64), ids, counts


def _orientation(
    cxx: float,
    cxy: float,
    cyy: float,
) -> Tuple[float, bool, Optional[str]]:
    if cxx == cyy and cxy == 0.0:
        return 0.0, False, FOAMS105_ORIENTATION_TIE_REASON
    theta = math.degrees(0.5 * math.atan2(2.0 * cxy, cxx - cyy))
    if cxy == 0.0 and cxx < cyy:
        theta = 90.0
    if theta == -90.0:
        theta = 90.0
    return theta, True, None


def _eigenvalues(
    cxx: float,
    cxy: float,
    cyy: float,
    *,
    image_id: str,
    component_id: int,
) -> Tuple[float, float]:
    trace = cxx + cyy
    delta = math.hypot(cxx - cyy, 2.0 * cxy)
    lambda_major = (trace + delta) / 2.0
    lambda_minor = (trace - delta) / 2.0
    if lambda_minor <= 0.0 or not math.isfinite(lambda_minor) or not math.isfinite(lambda_major):
        matrix = np.array(((cxx, cxy), (cxy, cyy)), dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(matrix)
        lambda_minor = float(eigenvalues[0])
        lambda_major = float(eigenvalues[1])
    if not math.isfinite(lambda_major) or not math.isfinite(lambda_minor):
        raise _numerical_error(
            "nonfinite_shape_arithmetic",
            "eigenvalues",
            image_id=image_id,
            component_id=component_id,
            operands=(cxx, cxy, cyy),
        )
    if lambda_major <= 0.0 or lambda_minor <= 0.0 or lambda_minor > lambda_major:
        raise _numerical_error(
            "invalid_shape_geometry",
            "eigenvalues",
            image_id=image_id,
            component_id=component_id,
            operands=(lambda_major, lambda_minor),
        )
    return lambda_major, lambda_minor


def measure_foams105_moment_geometry(
    image_id: str,
    phase_measurement_result: Foams105PhaseMeasurementResult,
) -> Foams105ImageMomentGeometryResult:
    """Measure unit-square central-moment geometry on accepted labeled components."""
    normalized_image_id = _validate_image_id(image_id)
    measurement = _validate_phase_measurement(normalized_image_id, phase_measurement_result)
    scale = _finite_positive_real(
        measurement.scale_px_per_mm,
        "phase_measurement_result.scale_px_per_mm",
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
    )

    if not components:
        return Foams105ImageMomentGeometryResult(
            image_id=normalized_image_id,
            phase_measurement_result=measurement,
            components=(),
            component_count=0,
            foreground_pixel_count=0,
            scale_px_per_mm=scale,
        )

    x = cols
    y = -rows

    component_count = len(components)
    sum_x = np.bincount(ids, weights=x, minlength=component_count + 1)
    sum_y = np.bincount(ids, weights=y, minlength=component_count + 1)
    mean_x = np.zeros(component_count + 1, dtype=np.float64)
    mean_y = np.zeros(component_count + 1, dtype=np.float64)
    positive_ids = counts > 0
    mean_x[positive_ids] = sum_x[positive_ids] / counts[positive_ids]
    mean_y[positive_ids] = sum_y[positive_ids] / counts[positive_ids]

    dx = x - mean_x[ids]
    dy = y - mean_y[ids]
    sum_dx2 = np.bincount(ids, weights=dx * dx, minlength=component_count + 1)
    sum_dy2 = np.bincount(ids, weights=dy * dy, minlength=component_count + 1)
    sum_dxdy = np.bincount(ids, weights=dx * dy, minlength=component_count + 1)

    results = []
    pixel_interval_variance = 1.0 / 12.0
    machine_tolerance = 32.0 * np.finfo(np.float64).eps
    for component in components:
        component_id = int(component.component_id)
        pixel_count = int(counts[component_id])
        inverse_count = _checked_divide(
            1.0,
            float(pixel_count),
            stage="inverse_pixel_count",
            image_id=normalized_image_id,
            component_id=component_id,
        )
        cxx = (sum_dx2[component_id] * inverse_count) + pixel_interval_variance
        cyy = (sum_dy2[component_id] * inverse_count) + pixel_interval_variance
        cxy = sum_dxdy[component_id] * inverse_count

        lambda_major, lambda_minor = _eigenvalues(
            cxx,
            cxy,
            cyy,
            image_id=normalized_image_id,
            component_id=component_id,
        )
        major_axis_px = 4.0 * _checked_sqrt(
            lambda_major,
            stage="major_axis_px",
            image_id=normalized_image_id,
            component_id=component_id,
        )
        minor_axis_px = 4.0 * _checked_sqrt(
            lambda_minor,
            stage="minor_axis_px",
            image_id=normalized_image_id,
            component_id=component_id,
        )
        major_axis_mm = _checked_divide(
            major_axis_px,
            scale,
            stage="major_axis_mm",
            image_id=normalized_image_id,
            component_id=component_id,
        )
        minor_axis_mm = _checked_divide(
            minor_axis_px,
            scale,
            stage="minor_axis_mm",
            image_id=normalized_image_id,
            component_id=component_id,
        )

        ratio = _checked_divide(
            lambda_minor,
            lambda_major,
            stage="eccentricity_ratio",
            image_id=normalized_image_id,
            component_id=component_id,
        )
        radicand = 1.0 - ratio
        if radicand < 0.0 and radicand >= -machine_tolerance:
            radicand = 0.0
        elif radicand > 1.0 and radicand <= 1.0 + machine_tolerance:
            radicand = 1.0
        elif radicand < 0.0 or radicand > 1.0:
            raise _numerical_error(
                "invalid_shape_geometry",
                "eccentricity_radicand",
                image_id=normalized_image_id,
                component_id=component_id,
                operands=(radicand, ratio),
            )
        eccentricity = _checked_sqrt(
            radicand,
            stage="eccentricity",
            image_id=normalized_image_id,
            component_id=component_id,
        )

        orientation, orientation_defined, tie_reason = _orientation(cxx, cxy, cyy)
        orientation_project = orientation + FOAMS105_PROJECT_ORIENTATION_OFFSET_DEG

        results.append(
            Foams105MomentComponentGeometry(
                image_id=normalized_image_id,
                component_id=component_id,
                first_pixel_rc=component.first_pixel_rc,
                pixel_count=pixel_count,
                area_px=component.area_px,
                area_mm2=component.area_mm2,
                equivalent_diameter_mm=component.equivalent_diameter_mm,
                centroid_x_px=float(mean_x[component_id]),
                centroid_y_px=float(mean_y[component_id]),
                central_cxx=float(cxx),
                central_cxy=float(cxy),
                central_cyy=float(cyy),
                lambda_major=float(lambda_major),
                lambda_minor=float(lambda_minor),
                major_axis_px=float(major_axis_px),
                minor_axis_px=float(minor_axis_px),
                major_axis_mm=float(major_axis_mm),
                minor_axis_mm=float(minor_axis_mm),
                eccentricity=float(eccentricity),
                orientation_regionprops_profile_deg=float(orientation),
                orientation_project_deg=float(orientation_project),
                orientation_defined=orientation_defined,
                orientation_tie_reason=tie_reason,
            )
        )

    return Foams105ImageMomentGeometryResult(
        image_id=normalized_image_id,
        phase_measurement_result=measurement,
        components=tuple(results),
        component_count=len(results),
        foreground_pixel_count=foreground_count,
        scale_px_per_mm=scale,
    )


def _canonical_raw_selection(
    result: object,
) -> Foams105RawSelectionResult:
    if not isinstance(result, Foams105RawSelectionResult):
        raise Foams105MomentValidationError(
            "raw_selection_result must be a Foams105RawSelectionResult"
        )
    if result.method != FOAMS105_RAW_SELECTION_METHOD:
        raise Foams105MomentValidationError(
            "raw_selection_result method is incompatible",
            code="incompatible_raw_selection_result",
            context={
                "expected_method": FOAMS105_RAW_SELECTION_METHOD,
                "actual_method": result.method,
            },
        )
    canonical = select_foams105_raw_objects(
        result.bin_labels_mm,
        result.groups,
        result.requested_ranges,
    )
    if canonical != result:
        raise Foams105MomentValidationError(
            "raw_selection_result is not canonical for supplied groups and ranges",
            code="inconsistent_raw_selection_result",
        )
    return canonical


def _validated_geometry_results(
    values: object,
) -> Tuple[
    Tuple[Foams105ImageMomentGeometryResult, ...],
    Mapping[Tuple[str, int], Foams105MomentComponentGeometry],
]:
    if isinstance(values, (str, bytes)):
        raise Foams105MomentValidationError(
            "image_geometry_results must be a sequence"
        )
    try:
        raw_results = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105MomentValidationError(
            "image_geometry_results must be a sequence"
        ) from exc

    image_ids = set()
    identity_map = {}
    normalized = []
    for index, result in enumerate(raw_results):
        if not isinstance(result, Foams105ImageMomentGeometryResult):
            raise Foams105MomentValidationError(
                "image_geometry_results contains an invalid record",
                context={"index": index},
            )
        if result.method != FOAMS105_MOMENT_METHOD:
            raise Foams105MomentValidationError(
                "image_geometry_results method is incompatible",
                code="incompatible_geometry_result",
                context={"image_id": result.image_id},
            )
        if result.verification_level != FOAMS105_MOMENT_VERIFICATION_LEVEL:
            raise Foams105MomentValidationError(
                "image_geometry_results verification level is incompatible",
                code="incompatible_geometry_result",
                context={"image_id": result.image_id},
            )
        if result.perimeter_status != FOAMS105_PERIMETER_STATUS_PENDING:
            raise Foams105MomentValidationError(
                "image_geometry_results perimeter status is incompatible",
                code="incompatible_geometry_result",
                context={"image_id": result.image_id},
            )
        if result.image_id in image_ids:
            raise Foams105MomentValidationError(
                "duplicate image_id in image_geometry_results",
                code="duplicate_image_id",
                context={"image_id": result.image_id},
            )
        image_ids.add(result.image_id)
        for component in result.components:
            identity = (component.image_id, component.component_id)
            if identity in identity_map:
                raise Foams105MomentValidationError(
                    "duplicate component identity in image_geometry_results",
                    code="duplicate_component_identity",
                    context={"identity": identity},
                )
            identity_map[identity] = component
        normalized.append(result)
    return tuple(normalized), MappingProxyType(identity_map)


def select_foams105_moment_geometry(
    raw_selection_result: Foams105RawSelectionResult,
    image_geometry_results: Sequence[Foams105ImageMomentGeometryResult],
) -> Foams105SelectedMomentGeometryResult:
    """Join raw-selection identities to measured moment geometry in row order."""
    canonical_raw = _canonical_raw_selection(raw_selection_result)
    geometry_results, identity_map = _validated_geometry_results(image_geometry_results)

    for group in canonical_raw.groups:
        for object_index, object_ in enumerate(group.objects):
            identity = (object_.image_id, object_.component_id)
            component = identity_map.get(identity)
            if component is None:
                raise Foams105MomentValidationError(
                    "raw-selection object identity is missing from geometry",
                    code="missing_geometry_identity",
                    context={
                        "group_id": group.group_id,
                        "source_slot": group.source_slot,
                        "source_object_index": object_index,
                        "identity": identity,
                    },
                )
            if (
                component.area_mm2 != object_.area_mm2
                or component.equivalent_diameter_mm
                != object_.equivalent_diameter_mm
            ):
                raise Foams105MomentValidationError(
                    "raw-selection geometry metadata changed for identity",
                    code="changed_geometry_identity",
                    context={"identity": identity},
                )

    rows = []
    for output_index, row in enumerate(canonical_raw.rows):
        identity = (row.image_id, row.component_id)
        component = identity_map.get(identity)
        if component is None:
            raise Foams105MomentValidationError(
                "selected row identity is missing from geometry",
                code="missing_geometry_identity",
                context={"identity": identity, "raw_output_index": row.output_index},
            )
        if (
            component.area_mm2 != row.area_mm2
            or component.equivalent_diameter_mm != row.equivalent_diameter_mm
        ):
            raise Foams105MomentValidationError(
                "selected-row geometry metadata changed for identity",
                code="changed_geometry_identity",
                context={"identity": identity, "raw_output_index": row.output_index},
            )
        rows.append(
            Foams105SelectedMomentGeometryRow(
                output_index=output_index,
                raw_output_index=row.output_index,
                group_id=row.group_id,
                source_slot=row.source_slot,
                source_object_index=row.source_object_index,
                image_id=row.image_id,
                component_id=row.component_id,
                area_mm2=row.area_mm2,
                equivalent_diameter_mm=row.equivalent_diameter_mm,
                major_axis_px=component.major_axis_px,
                minor_axis_px=component.minor_axis_px,
                major_axis_mm=component.major_axis_mm,
                minor_axis_mm=component.minor_axis_mm,
                eccentricity=component.eccentricity,
                orientation_regionprops_profile_deg=(
                    component.orientation_regionprops_profile_deg
                ),
                orientation_project_deg=component.orientation_project_deg,
                orientation_defined=component.orientation_defined,
                orientation_tie_reason=component.orientation_tie_reason,
                component_geometry=component,
            )
        )

    return Foams105SelectedMomentGeometryResult(
        raw_selection_result=canonical_raw,
        image_geometry_results=geometry_results,
        rows=tuple(rows),
        selected_identities=tuple((row.image_id, row.component_id) for row in rows),
        selected_count=len(rows),
    )
