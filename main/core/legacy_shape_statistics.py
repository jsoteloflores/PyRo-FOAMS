"""FOAMS 1.0.5 derived shape measures and source-style statistics."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping as MappingABC
from collections.abc import Set as SetABC
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .legacy_measurements import (
    FOAMS105_MEASUREMENT_SOURCE_COMMIT,
    FOAMS105_MEASUREMENT_SOURCE_REPOSITORY,
    Foams105PhaseMeasurementResult,
)
from .legacy_perimeter import (
    FOAMS105_PERIMETER_METHOD,
    FOAMS105_PERIMETER_SCOPE,
    FOAMS105_PERIMETER_SELECTION_METHOD,
    FOAMS105_PERIMETER_SELECTION_SCOPE,
    FOAMS105_PERIMETER_SOURCE_COMMIT,
    FOAMS105_PERIMETER_SOURCE_FILES,
    FOAMS105_PERIMETER_SOURCE_REPOSITORY,
    FOAMS105_PERIMETER_SUPPORTED_PROFILES,
    FOAMS105_PERIMETER_VERIFICATION_LEVEL,
    Foams105ImagePerimeterResult,
    Foams105PerimeterComponent,
    Foams105PerimeterNumericalError,
    Foams105PerimeterValidationError,
    select_foams105_perimeter,
)
from .legacy_raw_selection import (
    Foams105RawSelectionDomainError,
    Foams105RawSelectionNumericalError,
    Foams105RawSelectionResult,
    Foams105RawSelectionValidationError,
    select_foams105_raw_objects,
)
from .legacy_shapes import (
    FOAMS105_MOMENT_METHOD,
    FOAMS105_MOMENT_SCOPE,
    FOAMS105_MOMENT_SELECTION_METHOD,
    FOAMS105_MOMENT_SELECTION_SCOPE,
    FOAMS105_MOMENT_SOURCE_COMMIT,
    FOAMS105_MOMENT_SOURCE_FILES,
    FOAMS105_MOMENT_SOURCE_REPOSITORY,
    FOAMS105_MOMENT_VERIFICATION_LEVEL,
    Foams105ImageMomentGeometryResult,
    Foams105MomentComponentGeometry,
    Foams105MomentNumericalError,
    Foams105MomentValidationError,
    select_foams105_moment_geometry,
)

FOAMS105_SHAPE_METHOD = "foams_1_0_5_derived_shape_measures_v1"
FOAMS105_SHAPE_STATISTICS_METHOD = "foams_1_0_5_shape_statistics_v1"
FOAMS105_SHAPE_SOURCE_REPOSITORY = FOAMS105_MEASUREMENT_SOURCE_REPOSITORY
FOAMS105_SHAPE_SOURCE_COMMIT = FOAMS105_MEASUREMENT_SOURCE_COMMIT
FOAMS105_SHAPE_SOURCE_FILES = ("analysis.m", "newstats.m", "basicstats.m", "savestats.m", "project.m")
FOAMS105_SHAPE_SCOPE = "derived_per_object_shape_measures_and_statistics"
FOAMS105_SHAPE_MODES = ("initial_analysis", "updated_minimum")
FOAMS105_SHAPE_MINIMUM_POLICY = "minimum_diameter_px_over_max_configured_scale"
FOAMS105_SHAPE_INITIAL_MEMBERSHIP_POLICY = "equivalent_diameter_mm_strictly_greater_than_threshold"
FOAMS105_SHAPE_UPDATED_MEMBERSHIP_POLICY = "equivalent_diameter_mm_greater_than_or_equal_to_threshold"
FOAMS105_SHAPE_STATISTIC_REDUCTION_ORDER = "raw_selection_row_order_sequential_float64"
FOAMS105_SHAPE_ROUNDING_POLICY = "four_decimals_half_away_from_zero"
FOAMS105_SHAPE_EMPTY_POLICY = "empty_population_with_explicit_unavailable_statistics"
FOAMS105_SHAPE_VERIFICATION_BOUNDARY = (
    "analytical_moment_geometry_plus_matlab_r2025b_perimeter_reference_profiles"
)


class Foams105ShapeStatisticsValidationError(ValueError):
    """Raised when shape derivation or statistics inputs violate the contract."""

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


class Foams105ShapeStatisticsNumericalError(ArithmeticError):
    """Raised when a derived shape expression cannot be represented safely."""

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
class Foams105ShapeValue:
    """Serializable value/status pair for a derived object measure."""

    value: Optional[float]
    status: str
    reason: Optional[str] = None
    legacy_class: Optional[str] = None


@dataclass(frozen=True)
class Foams105ComponentShapeMeasures:
    """Cached image-local measures independent of later selected populations."""

    moment_component: Foams105MomentComponentGeometry
    perimeter_component: Foams105PerimeterComponent
    ellipse_area_mm2: float
    regularity: Foams105ShapeValue
    form_factor: Foams105ShapeValue
    elongation: Foams105ShapeValue
    aspect_ratio: Foams105ShapeValue
    fitted_ellipse_perimeter_mm: Foams105ShapeValue


@dataclass(frozen=True)
class Foams105ShapeRow:
    """One post-nesting object with source-derived shape measures."""

    output_index: int
    raw_output_index: int
    group_id: str
    source_slot: int
    source_object_index: int
    image_id: str
    component_id: int
    first_pixel_rc: Tuple[int, int]
    pixel_count: int
    area_px: int
    area_mm2: float
    equivalent_diameter_mm: float
    scale_px_per_mm: float
    major_axis_px: float
    minor_axis_px: float
    major_axis_mm: float
    minor_axis_mm: float
    perimeter_px: float
    perimeter_mm: float
    ellipse_area_mm2: float
    regularity: Foams105ShapeValue
    form_factor: Foams105ShapeValue
    elongation: Foams105ShapeValue
    aspect_ratio: Foams105ShapeValue
    fitted_ellipse_perimeter_mm: Foams105ShapeValue
    eccentricity: float
    orientation_regionprops_profile_deg: float
    orientation_project_deg: float
    orientation_defined: bool
    orientation_tie_reason: Optional[str]
    moment_component: Foams105MomentComponentGeometry
    perimeter_component: Foams105PerimeterComponent


@dataclass(frozen=True)
class Foams105ShapeResult:
    """Owned per-object derivation and the validated upstream snapshots."""

    raw_selection_result: Foams105RawSelectionResult
    image_geometry_results: Tuple[Foams105ImageMomentGeometryResult, ...]
    image_perimeter_results: Tuple[Foams105ImagePerimeterResult, ...]
    rows: Tuple[Foams105ShapeRow, ...]
    selected_identities: Tuple[Tuple[str, int], ...]
    selected_count: int
    perimeter_profile: Optional[str]
    perimeter_reference_runtime: Optional[str]
    perimeter_reference_toolbox: Optional[str]
    perimeter_evidence_version: Optional[str]
    perimeter_evidence_hash: Optional[str]
    integrity_sha256: str
    method: str = FOAMS105_SHAPE_METHOD
    source_repository: str = FOAMS105_SHAPE_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_SHAPE_SOURCE_COMMIT
    source_files: Tuple[str, ...] = FOAMS105_SHAPE_SOURCE_FILES
    scope: str = FOAMS105_SHAPE_SCOPE
    moment_method: str = FOAMS105_MOMENT_METHOD
    moment_selection_method: str = FOAMS105_MOMENT_SELECTION_METHOD
    moment_source_repository: str = FOAMS105_MOMENT_SOURCE_REPOSITORY
    moment_source_commit: str = FOAMS105_MOMENT_SOURCE_COMMIT
    moment_source_files: Tuple[str, ...] = FOAMS105_MOMENT_SOURCE_FILES
    moment_scope: str = FOAMS105_MOMENT_SCOPE
    moment_selection_scope: str = FOAMS105_MOMENT_SELECTION_SCOPE
    moment_verification_level: str = FOAMS105_MOMENT_VERIFICATION_LEVEL
    perimeter_method: str = FOAMS105_PERIMETER_METHOD
    perimeter_selection_method: str = FOAMS105_PERIMETER_SELECTION_METHOD
    perimeter_source_repository: str = FOAMS105_PERIMETER_SOURCE_REPOSITORY
    perimeter_source_commit: str = FOAMS105_PERIMETER_SOURCE_COMMIT
    perimeter_source_files: Tuple[str, ...] = FOAMS105_PERIMETER_SOURCE_FILES
    perimeter_scope: str = FOAMS105_PERIMETER_SCOPE
    perimeter_selection_scope: str = FOAMS105_PERIMETER_SELECTION_SCOPE
    perimeter_verification_level: str = FOAMS105_PERIMETER_VERIFICATION_LEVEL
    verification_boundary: str = FOAMS105_SHAPE_VERIFICATION_BOUNDARY
    length_unit: str = "mm"
    area_unit: str = "mm^2"
    angle_unit: str = "degree"


@dataclass(frozen=True)
class Foams105ShapeStatisticsRow:
    """One source-order statistic row, including raw and display values."""

    metric: str
    status: str
    reason: Optional[str]
    included_identities: Tuple[Tuple[str, int], ...]
    excluded_identities: Tuple[Tuple[str, int], ...]
    population_count: int
    n: int
    maximum: Optional[float]
    minimum: Optional[float]
    mean: Optional[float]
    standard_deviation: Optional[float]
    median: Optional[float]
    mode: Optional[float]
    display_maximum: Optional[float]
    display_minimum: Optional[float]
    display_mean: Optional[float]
    display_standard_deviation: Optional[float]
    display_median: Optional[float]
    display_mode: Optional[float]


@dataclass(frozen=True)
class Foams105ShapeStatisticsResult:
    """Threshold view and initial or updated summary populations."""

    shape_result: Foams105ShapeResult
    mode: str
    configured_scales_px_per_mm: Tuple[float, ...]
    maximum_configured_scale_px_per_mm: float
    minimum_diameter_px: float
    threshold_mm: float
    threshold_identities: Tuple[Tuple[str, int], ...]
    threshold_count: int
    summary_population_identities: Tuple[Tuple[str, int], ...]
    summary_population_count: int
    statistic_rows: Tuple[Foams105ShapeStatisticsRow, ...]
    method: str = FOAMS105_SHAPE_STATISTICS_METHOD
    source_repository: str = FOAMS105_SHAPE_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_SHAPE_SOURCE_COMMIT
    source_files: Tuple[str, ...] = FOAMS105_SHAPE_SOURCE_FILES
    scope: str = "shape_statistics_only"
    threshold_policy: str = FOAMS105_SHAPE_MINIMUM_POLICY
    initial_membership_policy: str = FOAMS105_SHAPE_INITIAL_MEMBERSHIP_POLICY
    updated_membership_policy: str = FOAMS105_SHAPE_UPDATED_MEMBERSHIP_POLICY
    reduction_order: str = FOAMS105_SHAPE_STATISTIC_REDUCTION_ORDER
    rounding_policy: str = FOAMS105_SHAPE_ROUNDING_POLICY
    empty_population_policy: str = FOAMS105_SHAPE_EMPTY_POLICY
    length_unit: str = "mm"
    scale_unit: str = "pixels/mm"


def _shape_error(
    message: str,
    *,
    code: str,
    image_id: Optional[str] = None,
    component_id: Optional[int] = None,
    expression: Optional[str] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> Foams105ShapeStatisticsNumericalError:
    context = dict(extra or {})
    context.update({"image_id": image_id, "component_id": component_id, "expression": expression})
    return Foams105ShapeStatisticsNumericalError(message, code=code, context=context)


def _finite(value: float, expression: str, image_id: str, component_id: int) -> float:
    if not math.isfinite(value):
        raise _shape_error(
            f"{expression} produced a nonfinite result",
            code="nonfinite_shape_arithmetic",
            image_id=image_id,
            component_id=component_id,
            expression=expression,
        )
    return value


def _multiply(left: float, right: float, expression: str, image_id: str, component_id: int) -> float:
    result = left * right
    _finite(result, expression, image_id, component_id)
    if left != 0.0 and right != 0.0 and result == 0.0:
        raise _shape_error(
            f"{expression} underflowed to zero",
            code="nonzero_underflow",
            image_id=image_id,
            component_id=component_id,
            expression=expression,
        )
    return result


def _divide(numerator: float, denominator: float, expression: str, image_id: str, component_id: int) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise _shape_error(
            f"{expression} received a nonfinite operand",
            code="nonfinite_shape_arithmetic",
            image_id=image_id,
            component_id=component_id,
            expression=expression,
        )
    if denominator == 0.0:
        raise _shape_error(
            f"{expression} has a zero denominator",
            code="zero_denominator",
            image_id=image_id,
            component_id=component_id,
            expression=expression,
        )
    result = numerator / denominator
    _finite(result, expression, image_id, component_id)
    if numerator != 0.0 and result == 0.0:
        raise _shape_error(
            f"{expression} underflowed to zero",
            code="nonzero_underflow",
            image_id=image_id,
            component_id=component_id,
            expression=expression,
        )
    return result


def _derived_value(value: float) -> Foams105ShapeValue:
    return Foams105ShapeValue(value=value, status="defined")


def _derive_measures(
    moment: Foams105MomentComponentGeometry,
    perimeter: Foams105PerimeterComponent,
) -> Tuple[float, Foams105ShapeValue, Foams105ShapeValue, Foams105ShapeValue, Foams105ShapeValue, Foams105ShapeValue]:
    image_id = moment.image_id
    component_id = moment.component_id
    area = float(moment.area_mm2)
    major = float(moment.major_axis_mm)
    minor = float(moment.minor_axis_mm)
    perimeter_mm = float(perimeter.perimeter_mm)

    if area <= 0.0 or major <= 0.0 or minor <= 0.0 or minor > major:
        raise Foams105ShapeStatisticsValidationError(
            "shape inputs require positive area and ordered positive axes",
            code="invalid_shape_geometry",
            context={"image_id": image_id, "component_id": component_id},
        )
    if perimeter_mm < 0.0 or not math.isfinite(perimeter_mm):
        raise Foams105ShapeStatisticsValidationError(
            "perimeter must be finite and nonnegative",
            code="invalid_perimeter",
            context={"image_id": image_id, "component_id": component_id},
        )

    half_major = major / 2.0
    half_minor = minor / 2.0
    if (major > 0.0 and half_major == 0.0) or (minor > 0.0 and half_minor == 0.0):
        raise _shape_error(
            "ellipse semi-axis underflowed to zero",
            code="nonzero_underflow",
            image_id=image_id,
            component_id=component_id,
            expression="ellipse_area_mm2.semi_axes",
        )
    ellipse_area = _multiply(half_major, half_minor, "ellipse_area_mm2.axes_product", image_id, component_id)
    ellipse_area = _multiply(ellipse_area, math.pi, "ellipse_area_mm2.pi_product", image_id, component_id)
    regularity = _divide(area, ellipse_area, "regularity", image_id, component_id)

    numerator = _multiply(4.0, math.pi, "form_factor.numerator.4pi", image_id, component_id)
    numerator = _multiply(numerator, area, "form_factor.numerator.area", image_id, component_id)
    if perimeter_mm == 0.0:
        form_factor = Foams105ShapeValue(
            value=None,
            status="undefined",
            reason="zero_perimeter",
            legacy_class="positive_infinity",
        )
    else:
        denominator = _multiply(perimeter_mm, perimeter_mm, "form_factor.perimeter_squared", image_id, component_id)
        form_factor = _derived_value(_divide(numerator, denominator, "form_factor", image_id, component_id))

    axis_difference = major - minor
    axis_sum = _finite(major + minor, "elongation.axis_sum", image_id, component_id)
    elongation = _divide(axis_difference, axis_sum, "elongation", image_id, component_id)
    aspect_ratio = _divide(minor, major, "aspect_ratio", image_id, component_id)

    ratio = minor / major
    ratio_power = ratio**1.456
    _finite(ratio_power, "fitted_ellipse_perimeter_mm.ratio_power", image_id, component_id)
    if ratio > 0.0 and ratio_power == 0.0:
        raise _shape_error(
            "fitted ellipse ratio power underflowed to zero",
            code="nonzero_underflow",
            image_id=image_id,
            component_id=component_id,
            expression="fitted_ellipse_perimeter_mm.ratio_power",
        )
    correction = _multiply(math.pi - 2.0, ratio_power, "fitted_ellipse_perimeter_mm.correction", image_id, component_id)
    bracket = _finite(2.0 + correction, "fitted_ellipse_perimeter_mm.bracket", image_id, component_id)
    leading = _multiply(2.0, major, "fitted_ellipse_perimeter_mm.leading_product", image_id, component_id)
    fitted_perimeter = _multiply(leading, bracket, "fitted_ellipse_perimeter_mm", image_id, component_id)

    return (
        ellipse_area,
        _derived_value(regularity),
        form_factor,
        _derived_value(elongation),
        _derived_value(aspect_ratio),
        _derived_value(fitted_perimeter),
    )


def calculate_foams105_component_shape_measures(
    moment_component: Foams105MomentComponentGeometry,
    perimeter_component: Foams105PerimeterComponent,
) -> Foams105ComponentShapeMeasures:
    """Calculate one matched component once for reuse across later selections."""

    if not isinstance(moment_component, Foams105MomentComponentGeometry):
        raise Foams105ShapeStatisticsValidationError(
            "moment_component has an invalid type",
            code="invalid_moment_component",
        )
    if not isinstance(perimeter_component, Foams105PerimeterComponent):
        raise Foams105ShapeStatisticsValidationError(
            "perimeter_component has an invalid type",
            code="invalid_perimeter_component",
        )
    identity = (moment_component.image_id, moment_component.component_id)
    inferred_scale = moment_component.major_axis_px / moment_component.major_axis_mm
    if (
        perimeter_component.image_id != moment_component.image_id
        or perimeter_component.component_id != moment_component.component_id
        or perimeter_component.first_pixel_rc != moment_component.first_pixel_rc
        or perimeter_component.area_px != moment_component.area_px
        or perimeter_component.area_mm2 != moment_component.area_mm2
        or perimeter_component.equivalent_diameter_mm != moment_component.equivalent_diameter_mm
        or not math.isclose(perimeter_component.scale_px_per_mm, inferred_scale, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise Foams105ShapeStatisticsValidationError(
            "moment and perimeter components do not share a measurement identity",
            code="component_geometry_mismatch",
            context={"identity": identity},
        )
    (
        ellipse_area,
        regularity,
        form_factor,
        elongation,
        aspect_ratio,
        fitted_perimeter,
    ) = _derive_measures(moment_component, perimeter_component)
    return Foams105ComponentShapeMeasures(
        moment_component=moment_component,
        perimeter_component=perimeter_component,
        ellipse_area_mm2=ellipse_area,
        regularity=regularity,
        form_factor=form_factor,
        elongation=elongation,
        aspect_ratio=aspect_ratio,
        fitted_ellipse_perimeter_mm=fitted_perimeter,
    )


def _phase_measurements_match(
    geometry_measurement: Foams105PhaseMeasurementResult,
    perimeter_measurement: Foams105PhaseMeasurementResult,
) -> bool:
    for field in fields(Foams105PhaseMeasurementResult):
        left = getattr(geometry_measurement, field.name)
        right = getattr(perimeter_measurement, field.name)
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
                return False
            if not np.array_equal(left, right):
                return False
        elif left != right:
            return False
    return True


def _fingerprint_value(value: object) -> object:
    if is_dataclass(value):
        return {field.name: _fingerprint_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, float):
        return {"float64": value.hex()}
    if isinstance(value, tuple):
        return [_fingerprint_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported shape fingerprint value: {type(value).__name__}")


def _rows_integrity_sha256(rows: Tuple[Foams105ShapeRow, ...]) -> str:
    payload = json.dumps(
        _fingerprint_value(rows),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(payload).hexdigest()


def _selection_rows_match(moment_rows: tuple, perimeter_rows: tuple) -> bool:
    if len(moment_rows) != len(perimeter_rows):
        return False
    fields_to_match = (
        "output_index",
        "raw_output_index",
        "group_id",
        "source_slot",
        "source_object_index",
        "image_id",
        "component_id",
        "area_mm2",
        "equivalent_diameter_mm",
    )
    return all(
        all(getattr(moment_row, field) == getattr(perimeter_row, field) for field in fields_to_match)
        for moment_row, perimeter_row in zip(moment_rows, perimeter_rows)
    )


def _shape_row_from_cached_measure(
    raw_row: object,
    cached: Foams105ComponentShapeMeasures,
    scale_px_per_mm: float,
) -> Foams105ShapeRow:
    moment = cached.moment_component
    perimeter = cached.perimeter_component
    return Foams105ShapeRow(
        output_index=raw_row.output_index,
        raw_output_index=raw_row.output_index,
        group_id=raw_row.group_id,
        source_slot=raw_row.source_slot,
        source_object_index=raw_row.source_object_index,
        image_id=raw_row.image_id,
        component_id=raw_row.component_id,
        first_pixel_rc=moment.first_pixel_rc,
        pixel_count=moment.pixel_count,
        area_px=moment.area_px,
        area_mm2=moment.area_mm2,
        equivalent_diameter_mm=moment.equivalent_diameter_mm,
        scale_px_per_mm=scale_px_per_mm,
        major_axis_px=moment.major_axis_px,
        minor_axis_px=moment.minor_axis_px,
        major_axis_mm=moment.major_axis_mm,
        minor_axis_mm=moment.minor_axis_mm,
        perimeter_px=perimeter.perimeter_px,
        perimeter_mm=perimeter.perimeter_mm,
        ellipse_area_mm2=cached.ellipse_area_mm2,
        regularity=cached.regularity,
        form_factor=cached.form_factor,
        elongation=cached.elongation,
        aspect_ratio=cached.aspect_ratio,
        fitted_ellipse_perimeter_mm=cached.fitted_ellipse_perimeter_mm,
        eccentricity=moment.eccentricity,
        orientation_regionprops_profile_deg=moment.orientation_regionprops_profile_deg,
        orientation_project_deg=moment.orientation_project_deg,
        orientation_defined=moment.orientation_defined,
        orientation_tie_reason=moment.orientation_tie_reason,
        moment_component=moment,
        perimeter_component=perimeter,
    )


def _assemble_foams105_shapes_from_cached_components(
    raw_selection: Foams105RawSelectionResult,
    image_geometry_results: Sequence[Foams105ImageMomentGeometryResult],
    image_perimeter_results: Sequence[Foams105ImagePerimeterResult],
    component_shape_measures: Sequence[Foams105ComponentShapeMeasures],
) -> Foams105ShapeResult:
    """Join prepared shape values to canonical raw identities without image replay."""

    if not isinstance(raw_selection, Foams105RawSelectionResult):
        raise Foams105ShapeStatisticsValidationError("raw_selection has an invalid type", code="invalid_raw_selection")
    try:
        canonical_raw = select_foams105_raw_objects(
            raw_selection.bin_labels_mm,
            raw_selection.groups,
            raw_selection.requested_ranges,
        )
    except (Foams105RawSelectionValidationError, Foams105RawSelectionDomainError, Foams105RawSelectionNumericalError) as exc:
        raise Foams105ShapeStatisticsValidationError(
            "raw_selection cannot be replayed from its source inputs",
            code="invalid_raw_selection",
            context={"cause_code": getattr(exc, "code", None)},
        ) from exc
    if canonical_raw != raw_selection:
        raise Foams105ShapeStatisticsValidationError(
            "raw_selection differs from its canonical replay",
            code="stale_raw_selection",
        )

    moments = tuple(image_geometry_results)
    perimeters = tuple(image_perimeter_results)
    cached_measures = tuple(component_shape_measures)
    if any(not isinstance(item, Foams105ImageMomentGeometryResult) for item in moments):
        raise Foams105ShapeStatisticsValidationError("image_geometry_results contain an invalid record", code="invalid_geometry_cache")
    if any(not isinstance(item, Foams105ImagePerimeterResult) for item in perimeters):
        raise Foams105ShapeStatisticsValidationError("image_perimeter_results contain an invalid record", code="invalid_perimeter_cache")
    if any(not isinstance(item, Foams105ComponentShapeMeasures) for item in cached_measures):
        raise Foams105ShapeStatisticsValidationError("component_shape_measures contain an invalid record", code="invalid_shape_cache")

    moment_by_image = {item.image_id: item for item in moments}
    perimeter_by_image = {item.image_id: item for item in perimeters}
    if (
        len(moment_by_image) != len(moments)
        or len(perimeter_by_image) != len(perimeters)
        or set(moment_by_image) != set(perimeter_by_image)
    ):
        raise Foams105ShapeStatisticsValidationError(
            "geometry image snapshots are inconsistent",
            code="geometry_image_set_mismatch",
        )
    profiles = {item.profile for item in perimeters}
    if len(profiles) > 1:
        raise Foams105ShapeStatisticsValidationError("perimeter cache mixes profiles", code="mixed_perimeter_profiles")
    for image_id, moment_image in moment_by_image.items():
        perimeter_image = perimeter_by_image[image_id]
        if (
            moment_image.scale_px_per_mm != perimeter_image.scale_px_per_mm
            or moment_image.component_count != perimeter_image.component_count
            or moment_image.foreground_pixel_count != perimeter_image.foreground_pixel_count
        ):
            raise Foams105ShapeStatisticsValidationError(
                "moment and perimeter snapshots do not describe one phase measurement",
                code="phase_measurement_mismatch",
                context={"image_id": image_id},
            )

    moment_components = {
        (component.image_id, component.component_id): component
        for image in moments
        for component in image.components
    }
    perimeter_components = {
        (component.image_id, component.component_id): component
        for image in perimeters
        for component in image.components
    }
    measures_by_identity = {}
    for cached in cached_measures:
        identity = (cached.moment_component.image_id, cached.moment_component.component_id)
        if identity in measures_by_identity:
            raise Foams105ShapeStatisticsValidationError(
                "cached shape component identity is duplicated",
                code="duplicate_shape_identity",
                context={"identity": identity},
            )
        if (
            moment_components.get(identity) != cached.moment_component
            or perimeter_components.get(identity) != cached.perimeter_component
        ):
            raise Foams105ShapeStatisticsValidationError(
                "cached shape values refer to altered geometry records",
                code="stale_shape_cache",
                context={"identity": identity},
            )
        measures_by_identity[identity] = cached

    seen_raw_identities = set()
    for group in canonical_raw.groups:
        for object_index, object_ in enumerate(group.objects):
            identity = (object_.image_id, object_.component_id)
            if identity in seen_raw_identities:
                raise Foams105ShapeStatisticsValidationError(
                    "raw-selection contains a duplicate component identity",
                    code="duplicate_raw_identity",
                    context={"identity": identity, "group_id": group.group_id, "source_slot": group.source_slot},
                )
            seen_raw_identities.add(identity)
            cached = measures_by_identity.get(identity)
            if cached is None:
                raise Foams105ShapeStatisticsValidationError(
                    "raw object lacks cached component geometry",
                    code="missing_geometry_identity",
                    context={
                        "identity": identity,
                        "group_id": group.group_id,
                        "source_slot": group.source_slot,
                        "source_object_index": object_index,
                    },
                )
            component = cached.moment_component
            if component.area_mm2 != object_.area_mm2 or component.equivalent_diameter_mm != object_.equivalent_diameter_mm:
                raise Foams105ShapeStatisticsValidationError(
                    "raw object metadata differs from cached geometry",
                    code="changed_geometry_identity",
                    context={"identity": identity},
                )

    rows = tuple(
        _shape_row_from_cached_measure(
            row,
            measures_by_identity[(row.image_id, row.component_id)],
            moment_by_image[row.image_id].scale_px_per_mm,
        )
        for row in canonical_raw.rows
    )
    first_perimeter = perimeters[0] if perimeters else None
    return Foams105ShapeResult(
        raw_selection_result=canonical_raw,
        image_geometry_results=moments,
        image_perimeter_results=perimeters,
        rows=rows,
        selected_identities=tuple((row.image_id, row.component_id) for row in rows),
        selected_count=len(rows),
        perimeter_profile=(next(iter(profiles)) if profiles else None),
        perimeter_reference_runtime=(first_perimeter.reference_runtime if first_perimeter else None),
        perimeter_reference_toolbox=(first_perimeter.reference_toolbox if first_perimeter else None),
        perimeter_evidence_version=(first_perimeter.evidence_version if first_perimeter else None),
        perimeter_evidence_hash=(first_perimeter.evidence_hash if first_perimeter else None),
        integrity_sha256=_rows_integrity_sha256(rows),
    )


def derive_foams105_shapes(
    raw_selection: Foams105RawSelectionResult,
    image_geometry_results: Sequence[Foams105ImageMomentGeometryResult],
    image_perimeter_results: Sequence[Foams105ImagePerimeterResult],
) -> Foams105ShapeResult:
    """Derive FOAMS shape measures from canonical moment and perimeter results."""

    try:
        selected_moments = select_foams105_moment_geometry(raw_selection, image_geometry_results)
    except Foams105MomentNumericalError as exc:
        raise Foams105ShapeStatisticsNumericalError(
            "moment geometry replay failed during shape derivation",
            code="moment_replay_failure",
            context={"cause_code": exc.code, "cause_context": dict(exc.context)},
        ) from exc
    except Foams105MomentValidationError as exc:
        raise Foams105ShapeStatisticsValidationError(
            "moment geometry validation failed during shape derivation",
            code="invalid_moment_input",
            context={"cause_code": exc.code, "cause_context": dict(exc.context)},
        ) from exc

    try:
        selected_perimeters = select_foams105_perimeter(raw_selection, image_perimeter_results)
    except Foams105PerimeterNumericalError as exc:
        raise Foams105ShapeStatisticsNumericalError(
            "perimeter replay failed during shape derivation",
            code="perimeter_replay_failure",
            context={"cause_code": exc.code, "cause_context": dict(exc.context)},
        ) from exc
    except Foams105PerimeterValidationError as exc:
        raise Foams105ShapeStatisticsValidationError(
            "perimeter validation failed during shape derivation",
            code="invalid_perimeter_input",
            context={"cause_code": exc.code, "cause_context": dict(exc.context)},
        ) from exc

    moment_by_image = {result.image_id: result for result in selected_moments.image_geometry_results}
    perimeter_by_image = {result.image_id: result for result in selected_perimeters.image_perimeter_results}
    if set(moment_by_image) != set(perimeter_by_image):
        raise Foams105ShapeStatisticsValidationError(
            "moment and perimeter inputs must contain the same image identities",
            code="geometry_image_set_mismatch",
            context={
                "moment_only_image_ids": tuple(sorted(set(moment_by_image) - set(perimeter_by_image))),
                "perimeter_only_image_ids": tuple(sorted(set(perimeter_by_image) - set(moment_by_image))),
            },
        )
    for image_id, moment_result in moment_by_image.items():
        perimeter_result = perimeter_by_image[image_id]
        if (
            moment_result.scale_px_per_mm != perimeter_result.scale_px_per_mm
            or moment_result.component_count != perimeter_result.component_count
            or moment_result.foreground_pixel_count != perimeter_result.foreground_pixel_count
            or not _phase_measurements_match(
                moment_result.phase_measurement_result,
                perimeter_result.phase_measurement_result,
            )
        ):
            raise Foams105ShapeStatisticsValidationError(
                "moment and perimeter results do not describe the same phase measurement",
                code="phase_measurement_mismatch",
                context={"image_id": image_id},
            )

    if not _selection_rows_match(selected_moments.rows, selected_perimeters.rows):
        raise Foams105ShapeStatisticsValidationError(
            "moment and perimeter selections do not preserve the same raw identities",
            code="selection_identity_mismatch",
        )

    raw_identities = set()
    for group in selected_moments.raw_selection_result.groups:
        for object_index, object_ in enumerate(group.objects):
            identity = (object_.image_id, object_.component_id)
            if identity in raw_identities:
                raise Foams105ShapeStatisticsValidationError(
                    "raw selection contains a duplicate component identity",
                    code="duplicate_raw_identity",
                    context={
                        "identity": identity,
                        "group_id": group.group_id,
                        "source_slot": group.source_slot,
                        "source_object_index": object_index,
                    },
                )
            raw_identities.add(identity)

    moment_components = {
        (component.image_id, component.component_id): component
        for image_result in selected_moments.image_geometry_results
        for component in image_result.components
    }
    perimeter_components = {
        (component.image_id, component.component_id): component
        for image_result in selected_perimeters.image_perimeter_results
        for component in image_result.components
    }
    rows = []
    for moment_row, perimeter_row in zip(selected_moments.rows, selected_perimeters.rows):
        identity = (moment_row.image_id, moment_row.component_id)
        moment = moment_components[identity]
        perimeter = perimeter_components[identity]
        (
            ellipse_area,
            regularity,
            form_factor,
            elongation,
            aspect_ratio,
            fitted_perimeter,
        ) = _derive_measures(moment, perimeter)
        rows.append(
            Foams105ShapeRow(
                output_index=moment_row.output_index,
                raw_output_index=moment_row.raw_output_index,
                group_id=moment_row.group_id,
                source_slot=moment_row.source_slot,
                source_object_index=moment_row.source_object_index,
                image_id=moment_row.image_id,
                component_id=moment_row.component_id,
                first_pixel_rc=moment.first_pixel_rc,
                pixel_count=moment.pixel_count,
                area_px=moment.area_px,
                area_mm2=moment.area_mm2,
                equivalent_diameter_mm=moment.equivalent_diameter_mm,
                scale_px_per_mm=moment_by_image[identity[0]].scale_px_per_mm,
                major_axis_px=moment.major_axis_px,
                minor_axis_px=moment.minor_axis_px,
                major_axis_mm=moment.major_axis_mm,
                minor_axis_mm=moment.minor_axis_mm,
                perimeter_px=perimeter.perimeter_px,
                perimeter_mm=perimeter.perimeter_mm,
                ellipse_area_mm2=ellipse_area,
                regularity=regularity,
                form_factor=form_factor,
                elongation=elongation,
                aspect_ratio=aspect_ratio,
                fitted_ellipse_perimeter_mm=fitted_perimeter,
                eccentricity=moment.eccentricity,
                orientation_regionprops_profile_deg=moment.orientation_regionprops_profile_deg,
                orientation_project_deg=moment.orientation_project_deg,
                orientation_defined=moment.orientation_defined,
                orientation_tie_reason=moment.orientation_tie_reason,
                moment_component=moment,
                perimeter_component=perimeter,
            )
        )

    perimeter_results = selected_perimeters.image_perimeter_results
    profile = selected_perimeters.profile or None
    first_perimeter = perimeter_results[0] if perimeter_results else None
    return Foams105ShapeResult(
        raw_selection_result=selected_moments.raw_selection_result,
        image_geometry_results=selected_moments.image_geometry_results,
        image_perimeter_results=perimeter_results,
        rows=tuple(rows),
        selected_identities=tuple((row.image_id, row.component_id) for row in rows),
        selected_count=len(rows),
        perimeter_profile=profile,
        perimeter_reference_runtime=(first_perimeter.reference_runtime if first_perimeter else None),
        perimeter_reference_toolbox=(first_perimeter.reference_toolbox if first_perimeter else None),
        perimeter_evidence_version=(first_perimeter.evidence_version if first_perimeter else None),
        perimeter_evidence_hash=(first_perimeter.evidence_hash if first_perimeter else None),
        integrity_sha256=_rows_integrity_sha256(tuple(rows)),
    )


def _real_scalar(value: object, field: str, *, nonnegative: bool = False) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise Foams105ShapeStatisticsValidationError(
            f"{field} must be a finite real scalar",
            code="invalid_scalar",
            context={"field": field},
        )
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise Foams105ShapeStatisticsValidationError(
            f"{field} must be representable as float64",
            code="invalid_scalar",
            context={"field": field},
        ) from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise Foams105ShapeStatisticsValidationError(
            f"{field} must be finite" + (" and nonnegative" if nonnegative else ""),
            code="invalid_scalar",
            context={"field": field},
        )
    return result


def _configured_scales(values: object) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes, MappingABC, SetABC)):
        raise Foams105ShapeStatisticsValidationError(
            "configured_scales_px_per_mm must be a nonempty numeric sequence",
            code="invalid_scales",
        )
    try:
        source_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105ShapeStatisticsValidationError(
            "configured_scales_px_per_mm must be a nonempty numeric sequence",
            code="invalid_scales",
        ) from exc
    if not source_values:
        raise Foams105ShapeStatisticsValidationError(
            "configured_scales_px_per_mm must be nonempty",
            code="invalid_scales",
        )
    scales = tuple(
        _real_scalar(value, f"configured_scales_px_per_mm[{index}]")
        for index, value in enumerate(source_values)
    )
    if any(scale <= 0.0 for scale in scales):
        raise Foams105ShapeStatisticsValidationError(
            "configured scales must be positive",
            code="invalid_scales",
            context={"configured_scales_px_per_mm": scales},
        )
    return scales


def _round_half_away_4(value: float, *, metric: str) -> float:
    scaled = value * 10000.0
    if not math.isfinite(scaled):
        raise Foams105ShapeStatisticsNumericalError(
            "four-decimal presentation scaling produced a nonfinite value",
            code="nonfinite_rounding",
            context={"metric": metric, "value": value, "stage": "scale", "scale_factor": 10000.0},
        )
    if scaled == 0.0:
        return 0.0

    negative = scaled < 0.0
    absolute_scaled = abs(scaled)
    lower = math.floor(absolute_scaled)
    fraction = absolute_scaled - lower
    rounded_integer = lower + (1 if fraction >= 0.5 else 0)
    try:
        rounded_magnitude = rounded_integer / 10000.0
    except OverflowError as exc:
        raise Foams105ShapeStatisticsNumericalError(
            "four-decimal presentation division could not be represented",
            code="nonfinite_rounding",
            context={
                "metric": metric,
                "value": value,
                "stage": "divide",
                "rounded_integer": rounded_integer,
            },
        ) from exc
    result = -rounded_magnitude if negative else rounded_magnitude
    if not math.isfinite(result):
        raise Foams105ShapeStatisticsNumericalError(
            "four-decimal presentation division produced a nonfinite value",
            code="nonfinite_rounding",
            context={"metric": metric, "value": value, "stage": "divide"},
        )
    return result


def _metric_value(row: Foams105ShapeRow, metric: str) -> Optional[float]:
    if metric == "aspect_ratio":
        return row.aspect_ratio.value
    if metric == "elongation":
        return row.elongation.value
    if metric == "regularity":
        return row.regularity.value
    return row.eccentricity


def _statistics_row(metric: str, population: Tuple[Foams105ShapeRow, ...]) -> Foams105ShapeStatisticsRow:
    included = []
    excluded = []
    values = []
    for row in population:
        value = _metric_value(row, metric)
        identity = (row.image_id, row.component_id)
        if metric == "eccentricity" and (value is None or not math.isfinite(value)):
            raise Foams105ShapeStatisticsValidationError(
                "eccentricity must remain finite after geometry validation",
                code="invalid_eccentricity",
                context={"identity": identity},
            )
        if value is None or not math.isfinite(value):
            excluded.append(identity)
        else:
            included.append(identity)
            values.append(float(value))

    n = len(values)
    if n == 0:
        return Foams105ShapeStatisticsRow(
            metric=metric,
            status="unavailable",
            reason="empty_population",
            included_identities=tuple(included),
            excluded_identities=tuple(excluded),
            population_count=len(population),
            n=0,
            maximum=None,
            minimum=None,
            mean=None,
            standard_deviation=None,
            median=None,
            mode=None,
            display_maximum=None,
            display_minimum=None,
            display_mean=None,
            display_standard_deviation=None,
            display_median=None,
            display_mode=None,
        )

    total = 0.0
    for value in values:
        total += value
        if not math.isfinite(total):
            raise Foams105ShapeStatisticsNumericalError(
                "statistics mean reduction became nonfinite",
                code="nonfinite_statistic",
                context={"metric": metric, "stage": "mean", "n": n},
            )
    mean = total / n
    if total != 0.0 and mean == 0.0:
        raise Foams105ShapeStatisticsNumericalError(
            "statistics mean underflowed to zero",
            code="nonzero_underflow_statistic",
            context={"metric": metric, "stage": "mean", "n": n},
        )
    ordered = sorted(values)
    maximum = ordered[-1]
    minimum = ordered[0]
    if n % 2:
        median = ordered[n // 2]
    else:
        middle_sum = ordered[(n // 2) - 1] + ordered[n // 2]
        if not math.isfinite(middle_sum):
            raise Foams105ShapeStatisticsNumericalError(
                "statistics median intermediate became nonfinite",
                code="nonfinite_statistic",
                context={"metric": metric, "stage": "median", "n": n},
            )
        median = middle_sum / 2.0
        if middle_sum != 0.0 and median == 0.0:
            raise Foams105ShapeStatisticsNumericalError(
                "statistics median underflowed to zero",
                code="nonzero_underflow_statistic",
                context={"metric": metric, "stage": "median", "n": n},
            )
    counts = Counter(values)
    highest_count = max(counts.values())
    mode = min(value for value, count in counts.items() if count == highest_count)

    if n == 1:
        standard_deviation = 0.0
    else:
        squared_deviation_total = 0.0
        for value in values:
            delta = value - mean
            squared_deviation = delta * delta
            if not math.isfinite(squared_deviation):
                raise Foams105ShapeStatisticsNumericalError(
                    "statistics standard-deviation intermediate became nonfinite",
                    code="nonfinite_statistic",
                    context={"metric": metric, "stage": "standard_deviation", "n": n},
                )
            squared_deviation_total += squared_deviation
            if not math.isfinite(squared_deviation_total):
                raise Foams105ShapeStatisticsNumericalError(
                    "statistics standard-deviation reduction became nonfinite",
                    code="nonfinite_statistic",
                    context={"metric": metric, "stage": "standard_deviation", "n": n},
                )
        variance = squared_deviation_total / (n - 1)
        if squared_deviation_total > 0.0 and variance == 0.0:
            raise Foams105ShapeStatisticsNumericalError(
                "statistics variance underflowed to zero",
                code="nonzero_underflow_statistic",
                context={"metric": metric, "stage": "standard_deviation", "n": n},
            )
        standard_deviation = math.sqrt(variance)

    raw = (maximum, minimum, mean, standard_deviation, median, mode)
    displays = tuple(_round_half_away_4(value, metric=metric) for value in raw)
    return Foams105ShapeStatisticsRow(
        metric=metric,
        status="defined",
        reason=None,
        included_identities=tuple(included),
        excluded_identities=tuple(excluded),
        population_count=len(population),
        n=n,
        maximum=maximum,
        minimum=minimum,
        mean=mean,
        standard_deviation=standard_deviation,
        median=median,
        mode=mode,
        display_maximum=displays[0],
        display_minimum=displays[1],
        display_mean=displays[2],
        display_standard_deviation=displays[3],
        display_median=displays[4],
        display_mode=displays[5],
    )


def summarize_foams105_shapes(
    shape_result: Foams105ShapeResult,
    *,
    configured_scales_px_per_mm: Sequence[float],
    minimum_diameter_px: float,
    mode: str,
) -> Foams105ShapeStatisticsResult:
    """Apply the source's initial or updated minimum and four statistics rows."""

    if not isinstance(shape_result, Foams105ShapeResult):
        raise Foams105ShapeStatisticsValidationError(
            "shape_result must be a Foams105ShapeResult",
            code="invalid_shape_result",
        )
    _validate_cached_shape_result(shape_result)
    if not isinstance(mode, str) or mode not in FOAMS105_SHAPE_MODES:
        raise Foams105ShapeStatisticsValidationError(
            "mode must be initial_analysis or updated_minimum",
            code="invalid_mode",
            context={"mode": mode, "supported_modes": FOAMS105_SHAPE_MODES},
        )
    scales = _configured_scales(configured_scales_px_per_mm)
    minimum = _real_scalar(minimum_diameter_px, "minimum_diameter_px", nonnegative=True)
    maximum_scale = max(scales)
    threshold = minimum / maximum_scale
    if not math.isfinite(threshold):
        raise Foams105ShapeStatisticsNumericalError(
            "diameter threshold is nonfinite",
            code="nonfinite_threshold",
            context={"minimum_diameter_px": minimum, "maximum_scale_px_per_mm": maximum_scale},
        )
    if minimum > 0.0 and threshold == 0.0:
        raise Foams105ShapeStatisticsNumericalError(
            "positive diameter threshold underflowed to zero",
            code="threshold_underflow",
            context={"minimum_diameter_px": minimum, "maximum_scale_px_per_mm": maximum_scale},
        )

    rows = shape_result.rows
    if not isinstance(rows, tuple):
        raise Foams105ShapeStatisticsValidationError(
            "shape_result rows must be an immutable tuple",
            code="invalid_shape_result",
        )
    if mode == "initial_analysis":
        threshold_rows = tuple(row for row in rows if row.equivalent_diameter_mm > threshold)
        summary_population = rows
    else:
        threshold_rows = tuple(row for row in rows if row.equivalent_diameter_mm >= threshold)
        summary_population = threshold_rows

    statistic_rows = tuple(
        _statistics_row(metric, summary_population)
        for metric in ("aspect_ratio", "elongation", "regularity", "eccentricity")
    )
    return Foams105ShapeStatisticsResult(
        shape_result=shape_result,
        mode=mode,
        configured_scales_px_per_mm=scales,
        maximum_configured_scale_px_per_mm=maximum_scale,
        minimum_diameter_px=minimum,
        threshold_mm=threshold,
        threshold_identities=tuple((row.image_id, row.component_id) for row in threshold_rows),
        threshold_count=len(threshold_rows),
        summary_population_identities=tuple((row.image_id, row.component_id) for row in summary_population),
        summary_population_count=len(summary_population),
        statistic_rows=statistic_rows,
    )


def _validate_cached_shape_result(shape_result: Foams105ShapeResult) -> None:
    for field, expected in (
        ("length_unit", "mm"),
        ("area_unit", "mm^2"),
        ("angle_unit", "degree"),
    ):
        actual = getattr(shape_result, field)
        if not isinstance(actual, str) or actual != expected:
            raise Foams105ShapeStatisticsValidationError(
                f"shape_result {field} is incompatible",
                code="incompatible_shape_result_unit",
                context={"field": field, "expected": expected, "actual": actual},
            )
    if (
        shape_result.method != FOAMS105_SHAPE_METHOD
        or shape_result.source_repository != FOAMS105_SHAPE_SOURCE_REPOSITORY
        or shape_result.source_commit != FOAMS105_SHAPE_SOURCE_COMMIT
        or shape_result.source_files != FOAMS105_SHAPE_SOURCE_FILES
        or shape_result.scope != FOAMS105_SHAPE_SCOPE
        or shape_result.moment_method != FOAMS105_MOMENT_METHOD
        or shape_result.moment_selection_method != FOAMS105_MOMENT_SELECTION_METHOD
        or shape_result.moment_source_repository != FOAMS105_MOMENT_SOURCE_REPOSITORY
        or shape_result.moment_source_commit != FOAMS105_MOMENT_SOURCE_COMMIT
        or shape_result.moment_source_files != FOAMS105_MOMENT_SOURCE_FILES
        or shape_result.moment_scope != FOAMS105_MOMENT_SCOPE
        or shape_result.moment_selection_scope != FOAMS105_MOMENT_SELECTION_SCOPE
        or shape_result.moment_verification_level != FOAMS105_MOMENT_VERIFICATION_LEVEL
        or shape_result.perimeter_method != FOAMS105_PERIMETER_METHOD
        or shape_result.perimeter_selection_method != FOAMS105_PERIMETER_SELECTION_METHOD
        or shape_result.perimeter_source_repository != FOAMS105_PERIMETER_SOURCE_REPOSITORY
        or shape_result.perimeter_source_commit != FOAMS105_PERIMETER_SOURCE_COMMIT
        or shape_result.perimeter_source_files != FOAMS105_PERIMETER_SOURCE_FILES
        or shape_result.perimeter_scope != FOAMS105_PERIMETER_SCOPE
        or shape_result.perimeter_selection_scope != FOAMS105_PERIMETER_SELECTION_SCOPE
        or shape_result.perimeter_verification_level != FOAMS105_PERIMETER_VERIFICATION_LEVEL
        or shape_result.verification_boundary != FOAMS105_SHAPE_VERIFICATION_BOUNDARY
    ):
        raise Foams105ShapeStatisticsValidationError(
            "shape_result provenance metadata is incompatible",
            code="invalid_shape_result_metadata",
        )
    if (
        not isinstance(shape_result.rows, tuple)
        or not isinstance(shape_result.selected_count, int)
        or isinstance(shape_result.selected_count, bool)
        or shape_result.selected_count != len(shape_result.rows)
        or any(not isinstance(row, Foams105ShapeRow) for row in shape_result.rows)
    ):
        raise Foams105ShapeStatisticsValidationError(
            "shape_result row count or ownership is inconsistent",
            code="invalid_shape_result_rows",
        )
    identities = tuple((row.image_id, row.component_id) for row in shape_result.rows)
    if any(
        not isinstance(row.image_id, str)
        or not row.image_id
        or not isinstance(row.component_id, int)
        or isinstance(row.component_id, bool)
        or row.component_id < 1
        for row in shape_result.rows
    ):
        raise Foams105ShapeStatisticsValidationError(
            "shape_result contains malformed component identities",
            code="invalid_shape_result_identities",
        )
    if identities != shape_result.selected_identities or len(set(identities)) != len(identities):
        raise Foams105ShapeStatisticsValidationError(
            "shape_result identities are altered or duplicated",
            code="invalid_shape_result_identities",
        )
    if _rows_integrity_sha256(shape_result.rows) != shape_result.integrity_sha256:
        raise Foams105ShapeStatisticsValidationError(
            "shape_result rows do not match their integrity fingerprint",
            code="forged_shape_result",
        )

    if not isinstance(shape_result.raw_selection_result, Foams105RawSelectionResult):
        raise Foams105ShapeStatisticsValidationError(
            "shape_result raw selection snapshot has an invalid type",
            code="invalid_shape_result_raw_selection",
        )
    try:
        canonical_raw = select_foams105_raw_objects(
            shape_result.raw_selection_result.bin_labels_mm,
            shape_result.raw_selection_result.groups,
            shape_result.raw_selection_result.requested_ranges,
        )
    except (
        Foams105RawSelectionValidationError,
        Foams105RawSelectionDomainError,
        Foams105RawSelectionNumericalError,
    ) as exc:
        raise Foams105ShapeStatisticsValidationError(
            "shape_result raw selection is invalid",
            code="invalid_shape_result_raw_selection",
            context={"cause": type(exc).__name__},
        ) from exc
    if canonical_raw != shape_result.raw_selection_result or len(canonical_raw.rows) != len(shape_result.rows):
        raise Foams105ShapeStatisticsValidationError(
            "shape_result raw selection snapshot is stale or altered",
            code="stale_shape_result_selection",
        )

    raw_fields = (
        "output_index",
        "group_id",
        "source_slot",
        "source_object_index",
        "image_id",
        "component_id",
        "area_mm2",
        "equivalent_diameter_mm",
    )
    for shape_row, raw_row in zip(shape_result.rows, canonical_raw.rows):
        if any(getattr(shape_row, field) != getattr(raw_row, field) for field in raw_fields):
            raise Foams105ShapeStatisticsValidationError(
                "shape_result row no longer matches raw-selection identity",
                code="stale_shape_result_selection",
                context={"image_id": shape_row.image_id, "component_id": shape_row.component_id},
            )

    moment_results = shape_result.image_geometry_results
    perimeter_results = shape_result.image_perimeter_results
    if not isinstance(moment_results, tuple) or not isinstance(perimeter_results, tuple):
        raise Foams105ShapeStatisticsValidationError(
            "shape_result image snapshots must be immutable tuples",
            code="invalid_shape_result_snapshots",
        )
    if any(not isinstance(result, Foams105ImageMomentGeometryResult) for result in moment_results) or any(
        not isinstance(result, Foams105ImagePerimeterResult) for result in perimeter_results
    ):
        raise Foams105ShapeStatisticsValidationError(
            "shape_result contains an invalid image snapshot record",
            code="invalid_shape_result_snapshots",
        )
    moment_by_image = {result.image_id: result for result in moment_results}
    perimeter_by_image = {result.image_id: result for result in perimeter_results}
    if len(moment_by_image) != len(moment_results) or len(perimeter_by_image) != len(perimeter_results):
        raise Foams105ShapeStatisticsValidationError(
            "shape_result contains duplicate image snapshots",
            code="duplicate_shape_image",
        )
    if set(moment_by_image) != set(perimeter_by_image):
        raise Foams105ShapeStatisticsValidationError(
            "shape_result moment and perimeter image snapshots differ",
            code="geometry_image_set_mismatch",
        )
    for image_id, result in moment_by_image.items():
        if result.method != FOAMS105_MOMENT_METHOD:
            raise Foams105ShapeStatisticsValidationError(
                "shape_result contains an incompatible moment snapshot",
                code="invalid_shape_result_snapshots",
                context={"image_id": image_id},
            )
    for image_id, result in perimeter_by_image.items():
        if (
            result.method != FOAMS105_PERIMETER_METHOD
            or result.profile != shape_result.perimeter_profile
            or result.verification_level != shape_result.perimeter_verification_level
            or result.reference_runtime != shape_result.perimeter_reference_runtime
            or result.reference_toolbox != shape_result.perimeter_reference_toolbox
            or result.evidence_version != shape_result.perimeter_evidence_version
            or result.evidence_hash != shape_result.perimeter_evidence_hash
        ):
            raise Foams105ShapeStatisticsValidationError(
                "shape_result contains incompatible perimeter provenance",
                code="invalid_shape_result_snapshots",
                context={"image_id": image_id},
            )
    if shape_result.perimeter_profile is None:
        if moment_results or perimeter_results:
            raise Foams105ShapeStatisticsValidationError(
                "nonempty image snapshots require a perimeter profile",
                code="invalid_shape_result_profile",
            )
    elif shape_result.perimeter_profile not in FOAMS105_PERIMETER_SUPPORTED_PROFILES:
        raise Foams105ShapeStatisticsValidationError(
            "shape_result perimeter profile is unsupported",
            code="invalid_shape_result_profile",
            context={"profile": shape_result.perimeter_profile},
        )

    moment_components = {
        (component.image_id, component.component_id): component
        for result in moment_results
        for component in result.components
    }
    perimeter_components = {
        (component.image_id, component.component_id): component
        for result in perimeter_results
        for component in result.components
    }
    if len(moment_components) != sum(len(result.components) for result in moment_results):
        raise Foams105ShapeStatisticsValidationError(
            "shape_result contains duplicate moment component identities",
            code="duplicate_shape_component",
        )
    if len(perimeter_components) != sum(len(result.components) for result in perimeter_results):
        raise Foams105ShapeStatisticsValidationError(
            "shape_result contains duplicate perimeter component identities",
            code="duplicate_shape_component",
        )
    for row in shape_result.rows:
        identity = (row.image_id, row.component_id)
        moment = moment_components.get(identity)
        perimeter = perimeter_components.get(identity)
        if (
            moment is None
            or perimeter is None
            or moment != row.moment_component
            or perimeter != row.perimeter_component
            or row.scale_px_per_mm != moment_by_image[row.image_id].scale_px_per_mm
            or perimeter.profile != shape_result.perimeter_profile
        ):
            raise Foams105ShapeStatisticsValidationError(
                "shape_result row no longer matches its geometry snapshots",
                code="forged_shape_result",
                context={"image_id": row.image_id, "component_id": row.component_id},
            )
