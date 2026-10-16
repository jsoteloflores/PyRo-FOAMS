"""FOAMS 1.0.5 component moment geometry and shape-identity replay."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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
_FLOAT_REL_TOL = 1e-12
_FLOAT_ABS_TOL = 1e-12


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


def _strict_int(
    value: object,
    field: str,
    *,
    image_id: str,
    component_id: Optional[int] = None,
    minimum: int = 0,
    code: str = "inconsistent_geometry_result",
) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
    ):
        raise Foams105MomentValidationError(
            f"{field} must be an integer scalar",
            code=code,
            context={
                "image_id": image_id,
                "component_id": component_id,
                "field": field,
            },
        )
    normalized = int(value)
    if normalized < minimum:
        raise Foams105MomentValidationError(
            f"{field} must be >= {minimum}",
            code=code,
            context={
                "image_id": image_id,
                "component_id": component_id,
                "field": field,
                "value": normalized,
            },
        )
    return normalized


def _strict_finite_real(
    value: object,
    field: str,
    *,
    image_id: str,
    component_id: Optional[int] = None,
    code: str = "inconsistent_geometry_result",
) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise Foams105MomentValidationError(
            f"{field} must be a finite real scalar",
            code=code,
            context={
                "image_id": image_id,
                "component_id": component_id,
                "field": field,
            },
        )
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise Foams105MomentValidationError(
            f"{field} must be representable as finite float64",
            code=code,
            context={
                "image_id": image_id,
                "component_id": component_id,
                "field": field,
            },
        ) from exc
    if not math.isfinite(normalized):
        raise Foams105MomentValidationError(
            f"{field} must be finite",
            code=code,
            context={
                "image_id": image_id,
                "component_id": component_id,
                "field": field,
                "value": normalized,
            },
        )
    return normalized


def _compare_finite_close(
    candidate: object,
    expected: float,
    field: str,
    *,
    image_id: str,
    component_id: Optional[int] = None,
) -> None:
    normalized = _strict_finite_real(
        candidate,
        field,
        image_id=image_id,
        component_id=component_id,
    )
    if not math.isclose(
        normalized,
        expected,
        rel_tol=_FLOAT_REL_TOL,
        abs_tol=_FLOAT_ABS_TOL,
    ):
        raise Foams105MomentValidationError(
            f"{field} does not match canonical geometry",
            code="inconsistent_geometry_result",
            context={
                "image_id": image_id,
                "component_id": component_id,
                "field": field,
                "expected": expected,
                "actual": normalized,
            },
        )


def _owned_immutable_array(
    values: object,
    dtype: np.dtype,
    field: str,
    *,
    image_id: str,
) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise Foams105MomentValidationError(
            f"{field} must be a two-dimensional NumPy array",
            code="inconsistent_geometry_result",
            context={"image_id": image_id, "field": field},
        )
    if values.ndim != 2 or values.size == 0:
        raise Foams105MomentValidationError(
            f"{field} must be a nonempty two-dimensional array",
            code="inconsistent_geometry_result",
            context={"image_id": image_id, "field": field},
        )
    contiguous = np.ascontiguousarray(values, dtype=dtype)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=dtype)
    return immutable.reshape(contiguous.shape)


def _owned_phase_measurement_result(
    measurement: object,
    *,
    image_id: str,
) -> Foams105PhaseMeasurementResult:
    if not isinstance(measurement, Foams105PhaseMeasurementResult):
        raise Foams105MomentValidationError(
            "phase_measurement_result must be a Foams105PhaseMeasurementResult",
            code="inconsistent_geometry_result",
            context={"image_id": image_id},
        )
    if not isinstance(measurement.components, tuple):
        raise Foams105MomentValidationError(
            "phase_measurement_result components must be a tuple",
            code="inconsistent_geometry_result",
            context={"image_id": image_id},
        )
    for index, component in enumerate(measurement.components):
        if not isinstance(component, Foams105ComponentMeasurement):
            raise Foams105MomentValidationError(
                "phase_measurement_result components contain an invalid record",
                code="inconsistent_geometry_result",
                context={"image_id": image_id, "component_index": index},
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


def _validate_component_contract(
    component: object,
    *,
    image_id: str,
    component_index: int,
) -> Foams105MomentComponentGeometry:
    if not isinstance(component, Foams105MomentComponentGeometry):
        raise Foams105MomentValidationError(
            "image_geometry_results contains an invalid component record",
            code="inconsistent_geometry_result",
            context={"image_id": image_id, "component_index": component_index},
        )
    component_id = _strict_int(
        component.component_id,
        "component_id",
        image_id=image_id,
        minimum=1,
    )
    if component.image_id != image_id:
        raise Foams105MomentValidationError(
            "component image_id must match parent image_id",
            code="inconsistent_geometry_result",
            context={
                "image_id": image_id,
                "component_id": component_id,
                "component_image_id": component.image_id,
            },
        )
    if (
        not isinstance(component.first_pixel_rc, tuple)
        or len(component.first_pixel_rc) != 2
    ):
        raise Foams105MomentValidationError(
            "first_pixel_rc must be a (row, column) tuple",
            code="inconsistent_geometry_result",
            context={"image_id": image_id, "component_id": component_id},
        )
    _strict_int(
        component.first_pixel_rc[0],
        "first_pixel_rc.row",
        image_id=image_id,
        component_id=component_id,
        minimum=0,
    )
    _strict_int(
        component.first_pixel_rc[1],
        "first_pixel_rc.column",
        image_id=image_id,
        component_id=component_id,
        minimum=0,
    )
    pixel_count = _strict_int(
        component.pixel_count,
        "pixel_count",
        image_id=image_id,
        component_id=component_id,
        minimum=1,
    )
    area_px = _strict_int(
        component.area_px,
        "area_px",
        image_id=image_id,
        component_id=component_id,
        minimum=1,
    )
    if area_px != pixel_count:
        raise Foams105MomentValidationError(
            "component area_px must equal pixel_count",
            code="inconsistent_geometry_result",
            context={"image_id": image_id, "component_id": component_id},
        )

    area_mm2 = _strict_finite_real(
        component.area_mm2,
        "area_mm2",
        image_id=image_id,
        component_id=component_id,
    )
    diameter_mm = _strict_finite_real(
        component.equivalent_diameter_mm,
        "equivalent_diameter_mm",
        image_id=image_id,
        component_id=component_id,
    )
    lambda_major = _strict_finite_real(
        component.lambda_major,
        "lambda_major",
        image_id=image_id,
        component_id=component_id,
    )
    lambda_minor = _strict_finite_real(
        component.lambda_minor,
        "lambda_minor",
        image_id=image_id,
        component_id=component_id,
    )
    major_axis_px = _strict_finite_real(
        component.major_axis_px,
        "major_axis_px",
        image_id=image_id,
        component_id=component_id,
    )
    minor_axis_px = _strict_finite_real(
        component.minor_axis_px,
        "minor_axis_px",
        image_id=image_id,
        component_id=component_id,
    )
    major_axis_mm = _strict_finite_real(
        component.major_axis_mm,
        "major_axis_mm",
        image_id=image_id,
        component_id=component_id,
    )
    minor_axis_mm = _strict_finite_real(
        component.minor_axis_mm,
        "minor_axis_mm",
        image_id=image_id,
        component_id=component_id,
    )
    eccentricity = _strict_finite_real(
        component.eccentricity,
        "eccentricity",
        image_id=image_id,
        component_id=component_id,
    )
    if (
        area_mm2 <= 0.0
        or diameter_mm <= 0.0
        or lambda_major <= 0.0
        or lambda_minor <= 0.0
        or major_axis_px <= 0.0
        or minor_axis_px <= 0.0
        or major_axis_mm <= 0.0
        or minor_axis_mm <= 0.0
    ):
        raise Foams105MomentValidationError(
            "component geometric lengths and areas must be positive",
            code="inconsistent_geometry_result",
            context={"image_id": image_id, "component_id": component_id},
        )
    if lambda_minor > lambda_major:
        raise Foams105MomentValidationError(
            "lambda_minor must be <= lambda_major",
            code="inconsistent_geometry_result",
            context={"image_id": image_id, "component_id": component_id},
        )
    if minor_axis_px > major_axis_px or minor_axis_mm > major_axis_mm:
        raise Foams105MomentValidationError(
            "minor axis must be <= major axis",
            code="inconsistent_geometry_result",
            context={"image_id": image_id, "component_id": component_id},
        )
    if eccentricity < 0.0 or eccentricity > 1.0:
        raise Foams105MomentValidationError(
            "eccentricity must be within [0, 1]",
            code="inconsistent_geometry_result",
            context={"image_id": image_id, "component_id": component_id},
        )

    _strict_finite_real(
        component.centroid_x_px,
        "centroid_x_px",
        image_id=image_id,
        component_id=component_id,
    )
    _strict_finite_real(
        component.centroid_y_px,
        "centroid_y_px",
        image_id=image_id,
        component_id=component_id,
    )
    _strict_finite_real(
        component.central_cxx,
        "central_cxx",
        image_id=image_id,
        component_id=component_id,
    )
    _strict_finite_real(
        component.central_cxy,
        "central_cxy",
        image_id=image_id,
        component_id=component_id,
    )
    _strict_finite_real(
        component.central_cyy,
        "central_cyy",
        image_id=image_id,
        component_id=component_id,
    )
    orientation = _strict_finite_real(
        component.orientation_regionprops_profile_deg,
        "orientation_regionprops_profile_deg",
        image_id=image_id,
        component_id=component_id,
    )
    orientation_project = _strict_finite_real(
        component.orientation_project_deg,
        "orientation_project_deg",
        image_id=image_id,
        component_id=component_id,
    )
    if not isinstance(component.orientation_defined, bool):
        raise Foams105MomentValidationError(
            "orientation_defined must be a bool",
            code="inconsistent_geometry_result",
            context={"image_id": image_id, "component_id": component_id},
        )
    if component.orientation_defined:
        if component.orientation_tie_reason is not None:
            raise Foams105MomentValidationError(
                "orientation_tie_reason must be None when orientation is defined",
                code="inconsistent_geometry_result",
                context={"image_id": image_id, "component_id": component_id},
            )
    elif component.orientation_tie_reason != FOAMS105_ORIENTATION_TIE_REASON:
        raise Foams105MomentValidationError(
            "orientation_tie_reason is incompatible",
            code="incompatible_geometry_result",
            context={"image_id": image_id, "component_id": component_id},
        )
    if component.perimeter_status != FOAMS105_PERIMETER_STATUS_PENDING:
        raise Foams105MomentValidationError(
            "component perimeter status is incompatible",
            code="incompatible_geometry_result",
            context={"image_id": image_id, "component_id": component_id},
        )
    if not math.isclose(
        orientation_project,
        orientation + FOAMS105_PROJECT_ORIENTATION_OFFSET_DEG,
        rel_tol=_FLOAT_REL_TOL,
        abs_tol=_FLOAT_ABS_TOL,
    ):
        raise Foams105MomentValidationError(
            "orientation_project_deg must equal source orientation plus project offset",
            code="inconsistent_geometry_result",
            context={"image_id": image_id, "component_id": component_id},
        )
    return component


def _assert_component_matches_canonical(
    candidate: Foams105MomentComponentGeometry,
    canonical: Foams105MomentComponentGeometry,
    *,
    image_id: str,
    component_index: int,
) -> None:
    component_id = canonical.component_id
    if candidate.image_id != canonical.image_id:
        raise Foams105MomentValidationError(
            "component image_id does not match canonical geometry",
            code="inconsistent_geometry_result",
            context={
                "image_id": image_id,
                "component_id": component_id,
                "component_index": component_index,
            },
        )
    if candidate.component_id != canonical.component_id:
        raise Foams105MomentValidationError(
            "component_id does not match canonical geometry",
            code="inconsistent_geometry_result",
            context={
                "image_id": image_id,
                "component_id": component_id,
                "component_index": component_index,
            },
        )
    if candidate.first_pixel_rc != canonical.first_pixel_rc:
        raise Foams105MomentValidationError(
            "first_pixel_rc does not match canonical geometry",
            code="inconsistent_geometry_result",
            context={
                "image_id": image_id,
                "component_id": component_id,
                "component_index": component_index,
            },
        )
    for field in ("pixel_count", "area_px"):
        if getattr(candidate, field) != getattr(canonical, field):
            raise Foams105MomentValidationError(
                f"{field} does not match canonical geometry",
                code="inconsistent_geometry_result",
                context={
                    "image_id": image_id,
                    "component_id": component_id,
                    "component_index": component_index,
                    "field": field,
                },
            )
    for field in (
        "area_mm2",
        "equivalent_diameter_mm",
        "centroid_x_px",
        "centroid_y_px",
        "central_cxx",
        "central_cxy",
        "central_cyy",
        "lambda_major",
        "lambda_minor",
        "major_axis_px",
        "minor_axis_px",
        "major_axis_mm",
        "minor_axis_mm",
        "eccentricity",
        "orientation_regionprops_profile_deg",
        "orientation_project_deg",
    ):
        _compare_finite_close(
            getattr(candidate, field),
            getattr(canonical, field),
            field,
            image_id=image_id,
            component_id=component_id,
        )
    if candidate.orientation_defined != canonical.orientation_defined:
        raise Foams105MomentValidationError(
            "orientation_defined does not match canonical geometry",
            code="inconsistent_geometry_result",
            context={
                "image_id": image_id,
                "component_id": component_id,
                "component_index": component_index,
            },
        )
    if candidate.orientation_tie_reason != canonical.orientation_tie_reason:
        raise Foams105MomentValidationError(
            "orientation_tie_reason does not match canonical geometry",
            code="inconsistent_geometry_result",
            context={
                "image_id": image_id,
                "component_id": component_id,
                "component_index": component_index,
            },
        )
    if candidate.perimeter_status != canonical.perimeter_status:
        raise Foams105MomentValidationError(
            "component perimeter status does not match canonical geometry",
            code="inconsistent_geometry_result",
            context={
                "image_id": image_id,
                "component_id": component_id,
                "component_index": component_index,
            },
        )


def _canonical_owned_geometry_result(
    supplied: object,
    *,
    index: int,
) -> Foams105ImageMomentGeometryResult:
    if not isinstance(supplied, Foams105ImageMomentGeometryResult):
        raise Foams105MomentValidationError(
            "image_geometry_results contains an invalid record",
            context={"index": index},
        )
    image_id = _validate_image_id(supplied.image_id)
    if supplied.method != FOAMS105_MOMENT_METHOD:
        raise Foams105MomentValidationError(
            "image_geometry_results method is incompatible",
            code="incompatible_geometry_result",
            context={"image_id": image_id},
        )
    if supplied.source_repository != FOAMS105_MOMENT_SOURCE_REPOSITORY:
        raise Foams105MomentValidationError(
            "image_geometry_results source repository is incompatible",
            code="incompatible_geometry_result",
            context={"image_id": image_id},
        )
    if supplied.source_commit != FOAMS105_MOMENT_SOURCE_COMMIT:
        raise Foams105MomentValidationError(
            "image_geometry_results source commit is incompatible",
            code="incompatible_geometry_result",
            context={"image_id": image_id},
        )
    if supplied.source_files != FOAMS105_MOMENT_SOURCE_FILES:
        raise Foams105MomentValidationError(
            "image_geometry_results source files are incompatible",
            code="incompatible_geometry_result",
            context={"image_id": image_id},
        )
    if supplied.scope != FOAMS105_MOMENT_SCOPE:
        raise Foams105MomentValidationError(
            "image_geometry_results scope is incompatible",
            code="incompatible_geometry_result",
            context={"image_id": image_id},
        )
    if supplied.connectivity != FOAMS105_COMPONENT_CONNECTIVITY:
        raise Foams105MomentValidationError(
            "image_geometry_results connectivity is incompatible",
            code="incompatible_geometry_result",
            context={"image_id": image_id},
        )
    if supplied.verification_level != FOAMS105_MOMENT_VERIFICATION_LEVEL:
        raise Foams105MomentValidationError(
            "image_geometry_results verification level is incompatible",
            code="incompatible_geometry_result",
            context={"image_id": image_id},
        )
    if supplied.perimeter_status != FOAMS105_PERIMETER_STATUS_PENDING:
        raise Foams105MomentValidationError(
            "image_geometry_results perimeter status is incompatible",
            code="incompatible_geometry_result",
            context={"image_id": image_id},
        )
    if supplied.orientation_tie_reason != FOAMS105_ORIENTATION_TIE_REASON:
        raise Foams105MomentValidationError(
            "image_geometry_results orientation tie reason is incompatible",
            code="incompatible_geometry_result",
            context={"image_id": image_id},
        )
    _compare_finite_close(
        supplied.orientation_project_offset_deg,
        FOAMS105_PROJECT_ORIENTATION_OFFSET_DEG,
        "orientation_project_offset_deg",
        image_id=image_id,
    )
    if (
        supplied.length_unit != "mm"
        or supplied.pixel_length_unit != "pixel"
        or supplied.area_unit != "mm^2"
        or supplied.pixel_area_unit != "pixel-area"
        or supplied.angle_unit != "degree"
    ):
        raise Foams105MomentValidationError(
            "image_geometry_results units are incompatible",
            code="incompatible_geometry_result",
            context={"image_id": image_id},
        )
    _strict_int(
        supplied.component_count,
        "component_count",
        image_id=image_id,
        minimum=0,
    )
    _strict_int(
        supplied.foreground_pixel_count,
        "foreground_pixel_count",
        image_id=image_id,
        minimum=0,
    )
    _finite_positive_real(supplied.scale_px_per_mm, "scale_px_per_mm")
    if not isinstance(supplied.components, tuple):
        raise Foams105MomentValidationError(
            "image_geometry_results components must be an immutable tuple",
            code="inconsistent_geometry_result",
            context={"image_id": image_id},
        )
    for component_index, component in enumerate(supplied.components):
        _validate_component_contract(
            component,
            image_id=image_id,
            component_index=component_index,
        )

    canonical = measure_foams105_moment_geometry(image_id, supplied.phase_measurement_result)
    owned_measurement = _owned_phase_measurement_result(
        canonical.phase_measurement_result,
        image_id=image_id,
    )
    canonical = replace(canonical, phase_measurement_result=owned_measurement)

    if supplied.component_count != canonical.component_count:
        raise Foams105MomentValidationError(
            "component_count does not match canonical geometry",
            code="inconsistent_geometry_result",
            context={
                "image_id": image_id,
                "expected": canonical.component_count,
                "actual": supplied.component_count,
            },
        )
    if supplied.foreground_pixel_count != canonical.foreground_pixel_count:
        raise Foams105MomentValidationError(
            "foreground_pixel_count does not match canonical geometry",
            code="inconsistent_geometry_result",
            context={
                "image_id": image_id,
                "expected": canonical.foreground_pixel_count,
                "actual": supplied.foreground_pixel_count,
            },
        )
    if supplied.foreground_pixel_count != canonical.phase_measurement_result.final_foreground_pixel_count:
        raise Foams105MomentValidationError(
            "foreground_pixel_count must match retained label foreground",
            code="inconsistent_geometry_result",
            context={"image_id": image_id},
        )
    _compare_finite_close(
        supplied.scale_px_per_mm,
        canonical.scale_px_per_mm,
        "scale_px_per_mm",
        image_id=image_id,
    )
    if len(supplied.components) != len(canonical.components):
        raise Foams105MomentValidationError(
            "components length does not match canonical geometry",
            code="inconsistent_geometry_result",
            context={"image_id": image_id},
        )
    for component_index, (candidate, expected) in enumerate(
        zip(supplied.components, canonical.components)
    ):
        _assert_component_matches_canonical(
            candidate,
            expected,
            image_id=image_id,
            component_index=component_index,
        )
    return canonical


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
    if (
        not isinstance(measurement.image_shape, tuple)
        or len(measurement.image_shape) != 2
    ):
        raise Foams105MomentValidationError(
            "measurement image_shape must be a (height, width) tuple",
            code="inconsistent_measurement_result",
            context={"image_id": image_id},
        )
    expected_height = _strict_int(
        measurement.image_shape[0],
        "phase_measurement_result.image_shape[0]",
        image_id=image_id,
        minimum=1,
        code="inconsistent_measurement_result",
    )
    expected_width = _strict_int(
        measurement.image_shape[1],
        "phase_measurement_result.image_shape[1]",
        image_id=image_id,
        minimum=1,
        code="inconsistent_measurement_result",
    )

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
    if filled.shape != (expected_height, expected_width):
        raise Foams105MomentValidationError(
            "measurement image_shape and array shapes are inconsistent",
            code="inconsistent_measurement_result",
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
    reported_foreground_count = _strict_int(
        measurement.final_foreground_pixel_count,
        "phase_measurement_result.final_foreground_pixel_count",
        image_id=image_id,
        minimum=0,
        code="inconsistent_measurement_result",
    )
    if foreground_count != reported_foreground_count:
        raise Foams105MomentValidationError(
            "measurement foreground count is inconsistent",
            code="inconsistent_measurement_result",
            context={
                "image_id": image_id,
                "label_foreground_count": foreground_count,
                "reported_foreground_count": reported_foreground_count,
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
        component_id = _strict_int(
            component.component_id,
            "phase_measurement_result.components.component_id",
            image_id=image_id,
            minimum=1,
            code="inconsistent_measurement_result",
        )
        if component_id in seen_ids:
            raise Foams105MomentValidationError(
                "measurement components contain duplicate IDs",
                code="inconsistent_measurement_result",
                context={
                    "image_id": image_id,
                    "component_id": component_id,
                },
            )
        seen_ids.add(component_id)
        if component_id > component_count:
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
        declared_area_px = _strict_int(
            component.area_px,
            "phase_measurement_result.components.area_px",
            image_id=image_id,
            component_id=component_id,
            minimum=1,
            code="inconsistent_measurement_result",
        )
        if declared_area_px != pixel_count:
            raise Foams105MomentValidationError(
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
            raise Foams105MomentValidationError(
                "measurement first_pixel_rc must be a (row, column) tuple",
                code="inconsistent_measurement_result",
                context={
                    "image_id": image_id,
                    "component_index": index,
                    "component_id": component_id,
                },
            )
        first_row = _strict_int(
            first_pixel[0],
            "phase_measurement_result.components.first_pixel_rc[0]",
            image_id=image_id,
            component_id=component_id,
            minimum=0,
            code="inconsistent_measurement_result",
        )
        first_column = _strict_int(
            first_pixel[1],
            "phase_measurement_result.components.first_pixel_rc[1]",
            image_id=image_id,
            component_id=component_id,
            minimum=0,
            code="inconsistent_measurement_result",
        )

        minimum = int(minima[component_id])
        first_pixel_from_labels = (minimum % height, minimum // height)
        if (first_row, first_column) != first_pixel_from_labels:
            raise Foams105MomentValidationError(
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
            code="inconsistent_measurement_result",
        )
        if area_mm2 <= 0.0:
            raise Foams105MomentValidationError(
                "measurement component area_mm2 must be positive",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_id": component_id},
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
        if not math.isclose(area_mm2, expected_area_mm2, rel_tol=1e-12, abs_tol=0.0):
            raise Foams105MomentValidationError(
                "measurement area_mm2 is inconsistent with label area",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_id": component_id},
            )
        equivalent_diameter_mm = _strict_finite_real(
            component.equivalent_diameter_mm,
            "phase_measurement_result.components.equivalent_diameter_mm",
            image_id=image_id,
            component_id=component_id,
            code="inconsistent_measurement_result",
        )
        if equivalent_diameter_mm <= 0.0:
            raise Foams105MomentValidationError(
                "measurement equivalent diameter must be positive",
                code="inconsistent_measurement_result",
                context={"image_id": image_id, "component_id": component_id},
            )
        expected_diameter = 2.0 * math.sqrt(area_mm2 / math.pi)
        if not math.isclose(
            equivalent_diameter_mm,
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
    for index, supplied in enumerate(raw_results):
        canonical = _canonical_owned_geometry_result(supplied, index=index)
        if canonical.image_id in image_ids:
            raise Foams105MomentValidationError(
                "duplicate image_id in image_geometry_results",
                code="duplicate_image_id",
                context={"image_id": canonical.image_id},
            )
        image_ids.add(canonical.image_id)
        for component in canonical.components:
            identity = (component.image_id, component.component_id)
            if identity in identity_map:
                raise Foams105MomentValidationError(
                    "duplicate component identity in image_geometry_results",
                    code="duplicate_component_identity",
                    context={"identity": identity},
                )
            identity_map[identity] = component
        normalized.append(canonical)
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
