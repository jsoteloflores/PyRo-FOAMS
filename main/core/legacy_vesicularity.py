"""FOAMS 1.0.5 two-dimensional vesicularity compatibility replay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .legacy_areas import (
    FOAMS105_AREA1_MAPPING,
    FOAMS105_AREA2_MAPPING,
    FOAMS105_AREA_SCOPE,
    FOAMS105_AREA_SOURCE_COMMIT,
    FOAMS105_AREA_SOURCE_FILE,
    FOAMS105_AREA_SOURCE_REPOSITORY,
    FOAMS105_GROUP_AREA_METHOD,
    FOAMS105_IMAGE_AREA_METHOD,
    Foams105AreaNumericalError,
    Foams105AreaValidationError,
    Foams105GroupAreaResult,
    Foams105ImageAreaResult,
    aggregate_foams105_group_areas,
)
from .legacy_raw_selection import (
    Foams105RawObject,
    Foams105RawSelectionResult,
    select_foams105_raw_objects,
)

FOAMS105_VESICULARITY_METHOD = "foams_1_0_5_two_dimensional_vesicularity_replay_v1"
FOAMS105_VESICULARITY_SOURCE_REPOSITORY = "jsoteloflores/FOAMS-1.0.5"
FOAMS105_VESICULARITY_SOURCE_COMMIT = (
    "179663203f2d0f86b2863d5ea7f8f70dadca02f8"
)
FOAMS105_VESICULARITY_SOURCE_FILES = ("analysis.m", "project.m")
FOAMS105_VESICULARITY_SCOPE = "two_dimensional_vesicularity_only"
FOAMS105_INTEGRATED_DENOMINATOR_POLICY = "independent_scale_and_area_sort"
FOAMS105_GROUP_SUMMARY_DENOMINATOR_POLICY = "explicit_image_membership"
FOAMS105_UNFILTERED_POPULATION_POLICY = (
    "all_measured_pore_components_before_diameter_filtering_and_raw_cutoffs"
)
FOAMS105_VESICULARITY_MAX_IMAGES = 20

DIAGNOSTIC_SOURCE_AREA_REASSIGNED = "source_area_reassigned"
DIAGNOSTIC_NEGATIVE_RESIDUAL = "negative_residual"
DIAGNOSTIC_SELECTED_AREA_EXCEEDS_SOURCE_DENOMINATOR = (
    "selected_area_exceeds_source_denominator"
)
DIAGNOSTIC_PERCENTAGE_OUT_OF_RANGE = "percentage_out_of_range"


class Foams105VesicularityValidationError(ValueError):
    """Raised when vesicularity inputs violate the bounded contract."""

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


class Foams105VesicularityDomainError(ValueError):
    """Raised when a source vesicularity denominator is unsupported."""

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


class Foams105VesicularityNumericalError(ArithmeticError):
    """Raised when ordered vesicularity arithmetic is nonfinite or underflows."""

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
class Foams105VesicularityGroupInput:
    """Area contexts and all pre-filter pore objects for one source slot."""

    group_id: str
    source_slot: int
    image_area_results: Tuple[Foams105ImageAreaResult, ...]
    unfiltered_pore_objects: Tuple[Foams105RawObject, ...]


@dataclass(frozen=True)
class Foams105SourceAreaAssignment:
    """One positional pairing from independently sorted scales and area2 values."""

    sorted_position: int
    assigned_source_slot: int
    assigned_group_id: str
    sorted_scale_px_per_mm: float
    area2_mm2: float
    original_flat_image_index: int
    original_source_slot: int
    original_group_id: str
    original_image_id: str


@dataclass(frozen=True)
class Foams105VesicularityDiagnostic:
    """Immutable arithmetic or denominator-mapping diagnostic."""

    code: str
    quantity: str
    source_slot: Optional[int] = None
    source_slots: Tuple[int, ...] = ()
    image_ids: Tuple[str, ...] = ()
    actual_area2_values_mm2: Tuple[float, ...] = ()
    assigned_area2_values_mm2: Tuple[float, ...] = ()
    actual_denominator_mm2: Optional[float] = None
    source_denominator_mm2: Optional[float] = None
    numerator_mm2: Optional[float] = None
    value: Optional[float] = None


@dataclass(frozen=True)
class Foams105VesicularityGroupTrace:
    """Both pore populations, denominator mappings, and arithmetic for one slot."""

    group_id: str
    source_slot: int
    scale_px_per_mm: float
    image_ids: Tuple[str, ...]
    image_area2_mm2: Tuple[float, ...]
    assigned_source_area2_mm2: Tuple[float, ...]
    area_result: Foams105GroupAreaResult
    source_sorted_area2_mm2: float
    actual_group_area2_mm2: float
    phase_corrected_area1_mm2: float
    area_diagnostic_codes: Tuple[str, ...]
    selected_objects: Tuple[Foams105RawObject, ...]
    selected_identities: Tuple[Tuple[str, int], ...]
    selected_area_sum_mm2: float
    selected_count: int
    unfiltered_pore_objects: Tuple[Foams105RawObject, ...]
    unfiltered_identities: Tuple[Tuple[str, int], ...]
    unfiltered_area_sum_mm2: float
    unfiltered_count: int
    residual_reference_area_mm2: float
    multiplier: Optional[float]
    removed_reference_area_mm2: float
    group_vesicularity_percent: float


@dataclass(frozen=True)
class Foams105VesicularityResult:
    """Complete two-dimensional source replay and image-membership summaries."""

    raw_selection: Foams105RawSelectionResult
    groups: Tuple[Foams105VesicularityGroupInput, ...]
    group_traces: Tuple[Foams105VesicularityGroupTrace, ...]
    sorted_scales_px_per_mm: Tuple[float, ...]
    sorted_area2_mm2: Tuple[float, ...]
    source_area_assignments: Tuple[Foams105SourceAreaAssignment, ...]
    source_sorted_area2_by_slot_mm2: Tuple[float, ...]
    actual_group_area2_by_slot_mm2: Tuple[float, ...]
    selected_area_by_slot_mm2: Tuple[float, ...]
    unfiltered_area_by_slot_mm2: Tuple[float, ...]
    residuals_reference_area_mm2: Tuple[float, ...]
    multipliers: Tuple[Optional[float], ...]
    removed_reference_area_mm2: Tuple[float, ...]
    integrated_vesicularity_percent: float
    group_vesicularity_percent: Tuple[float, ...]
    mean_group_vesicularity_percent: float
    diagnostics: Tuple[Foams105VesicularityDiagnostic, ...]
    method: str = FOAMS105_VESICULARITY_METHOD
    source_repository: str = FOAMS105_VESICULARITY_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_VESICULARITY_SOURCE_COMMIT
    source_files: Tuple[str, str] = FOAMS105_VESICULARITY_SOURCE_FILES
    scope: str = FOAMS105_VESICULARITY_SCOPE
    integrated_denominator_policy: str = FOAMS105_INTEGRATED_DENOMINATOR_POLICY
    group_summary_denominator_policy: str = (
        FOAMS105_GROUP_SUMMARY_DENOMINATOR_POLICY
    )
    unfiltered_population_policy: str = FOAMS105_UNFILTERED_POPULATION_POLICY
    area1_mapping: str = FOAMS105_AREA1_MAPPING
    area2_mapping: str = FOAMS105_AREA2_MAPPING
    length_unit: str = "mm"
    area_unit: str = "mm^2"
    scale_unit: str = "pixels/mm"
    percentage_unit: str = "percent"


@dataclass(frozen=True)
class _Foams105VesicularityNumericalResult:
    sorted_scales_px_per_mm: Tuple[float, ...]
    sorted_area2_mm2: Tuple[float, ...]
    assignment_source_slots: Tuple[int, ...]
    assignment_original_flat_indices: Tuple[int, ...]
    source_sorted_area2_by_slot_mm2: Tuple[float, ...]
    actual_group_area2_by_slot_mm2: Tuple[float, ...]
    residuals_reference_area_mm2: Tuple[float, ...]
    multipliers: Tuple[Optional[float], ...]
    removed_reference_area_mm2: Tuple[float, ...]
    integrated_vesicularity_percent: float
    group_vesicularity_percent: Tuple[float, ...]
    mean_group_vesicularity_percent: float
    reassigned_slots: Tuple[int, ...]
    diagnostics: Tuple[Foams105VesicularityDiagnostic, ...]


def _real_scalar(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise Foams105VesicularityValidationError(
            f"{field} must be a finite real scalar",
            context={"field": field},
        )
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise Foams105VesicularityValidationError(
            f"{field} must be representable as finite float64",
            context={"field": field},
        ) from exc
    if not math.isfinite(result):
        raise Foams105VesicularityValidationError(
            f"{field} must be finite",
            context={"field": field},
        )
    return result


def _sequence(values: object, field: str) -> Tuple[object, ...]:
    if isinstance(values, (str, bytes)):
        raise Foams105VesicularityValidationError(f"{field} must be a sequence")
    try:
        return tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105VesicularityValidationError(
            f"{field} must be a sequence"
        ) from exc


def _ordered_sum(
    values: Sequence[float],
    *,
    stage: str,
    source_slot: Optional[int] = None,
) -> float:
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if not math.isfinite(total):
            raise Foams105VesicularityNumericalError(
                "nonfinite_sum",
                f"{stage} ordered sum is nonfinite",
                context={
                    "stage": stage,
                    "source_slot": source_slot,
                    "index": index,
                    "partial_sum": total,
                    "operand": value,
                },
            )
    return total


def _checked_divide(
    numerator: float,
    denominator: float,
    *,
    stage: str,
    source_slot: Optional[int],
    code: str,
) -> float:
    try:
        value = numerator / denominator
    except OverflowError as exc:
        raise Foams105VesicularityNumericalError(
            code,
            f"{stage} division overflowed",
            context={
                "stage": stage,
                "source_slot": source_slot,
                "numerator": numerator,
                "denominator": denominator,
            },
        ) from exc
    if not math.isfinite(value):
        raise Foams105VesicularityNumericalError(
            code,
            f"{stage} division is nonfinite",
            context={
                "stage": stage,
                "source_slot": source_slot,
                "numerator": numerator,
                "denominator": denominator,
                "result": value,
            },
        )
    if numerator != 0.0 and value == 0.0:
        raise Foams105VesicularityNumericalError(
            "nonzero_underflow",
            f"{stage} nonzero division underflowed to zero",
            context={
                "stage": stage,
                "source_slot": source_slot,
                "numerator": numerator,
                "denominator": denominator,
            },
        )
    return value


def _checked_multiply(
    left: float,
    right: float,
    *,
    stage: str,
    source_slot: Optional[int],
    code: str,
) -> float:
    value = left * right
    if not math.isfinite(value):
        raise Foams105VesicularityNumericalError(
            code,
            f"{stage} multiplication is nonfinite",
            context={
                "stage": stage,
                "source_slot": source_slot,
                "left": left,
                "right": right,
                "result": value,
            },
        )
    if left != 0.0 and right != 0.0 and value == 0.0:
        raise Foams105VesicularityNumericalError(
            "nonzero_underflow",
            f"{stage} nonzero multiplication underflowed to zero",
            context={
                "stage": stage,
                "source_slot": source_slot,
                "left": left,
                "right": right,
            },
        )
    return value


def _numeric_group_values(
    values: object,
    field: str,
    group_count: int,
    *,
    nonnegative: bool,
) -> Tuple[float, ...]:
    raw = _sequence(values, field)
    if len(raw) != group_count:
        raise Foams105VesicularityValidationError(
            f"{field} must contain exactly one value per group"
        )
    converted = tuple(
        _real_scalar(value, f"{field}[{index}]")
        for index, value in enumerate(raw)
    )
    if nonnegative:
        for index, value in enumerate(converted):
            if value < 0.0:
                raise Foams105VesicularityValidationError(
                    f"{field}[{index}] must be nonnegative"
                )
    return converted


def _calculate_foams105_vesicularity_numerical(
    scales_px_per_mm: Sequence[float],
    image_area2_by_group_mm2: Sequence[Sequence[float]],
    selected_area_by_slot_mm2: Sequence[float],
    unfiltered_area_by_slot_mm2: Sequence[float],
) -> _Foams105VesicularityNumericalResult:
    """Run the source arithmetic for validated or analytical scalar inputs."""
    raw_scales = _sequence(scales_px_per_mm, "scales_px_per_mm")
    if not 1 <= len(raw_scales) <= 4:
        raise Foams105VesicularityValidationError(
            "scales_px_per_mm must contain one to four values"
        )
    scales = tuple(
        _real_scalar(value, f"scales_px_per_mm[{index}]")
        for index, value in enumerate(raw_scales)
    )
    for index, scale in enumerate(scales):
        if scale <= 0.0:
            raise Foams105VesicularityValidationError(
                f"scales_px_per_mm[{index}] must be positive"
            )
        if index and scale <= scales[index - 1]:
            raise Foams105VesicularityValidationError(
                "scales_px_per_mm must be strictly increasing"
            )

    raw_area_groups = _sequence(
        image_area2_by_group_mm2, "image_area2_by_group_mm2"
    )
    if len(raw_area_groups) != len(scales):
        raise Foams105VesicularityValidationError(
            "image_area2_by_group_mm2 must contain exactly one group per scale"
        )
    area_groups = []
    flat_entries = []
    flat_index = 0
    for group_index, raw_group in enumerate(raw_area_groups):
        values = _sequence(raw_group, f"image_area2_by_group_mm2[{group_index}]")
        if not values:
            raise Foams105VesicularityValidationError(
                f"image_area2_by_group_mm2[{group_index}] must not be empty"
            )
        converted = tuple(
            _real_scalar(
                value, f"image_area2_by_group_mm2[{group_index}][{index}]"
            )
            for index, value in enumerate(values)
        )
        for image_index, area in enumerate(converted):
            if area <= 0.0:
                raise Foams105VesicularityDomainError(
                    "nonpositive_border_corrected_area",
                    "border-corrected image area must be positive",
                    context={
                        "source_slot": group_index + 1,
                        "image_index": image_index,
                        "border_corrected_area_mm2": area,
                    },
                )
            flat_entries.append(
                (flat_index, group_index + 1, scales[group_index], area)
            )
            flat_index += 1
        area_groups.append(converted)
    if flat_index > FOAMS105_VESICULARITY_MAX_IMAGES:
        raise Foams105VesicularityValidationError(
            f"at most {FOAMS105_VESICULARITY_MAX_IMAGES} images are supported"
        )

    selected = _numeric_group_values(
        selected_area_by_slot_mm2,
        "selected_area_by_slot_mm2",
        len(scales),
        nonnegative=True,
    )
    unfiltered = _numeric_group_values(
        unfiltered_area_by_slot_mm2,
        "unfiltered_area_by_slot_mm2",
        len(scales),
        nonnegative=True,
    )

    sorted_scales = tuple(sorted(entry[2] for entry in flat_entries))
    sorted_area_entries = tuple(
        sorted(flat_entries, key=lambda entry: (-entry[3], entry[0]))
    )
    sorted_areas = tuple(entry[3] for entry in sorted_area_entries)
    scale_to_slot = {scale: index + 1 for index, scale in enumerate(scales)}
    assignment_slots = tuple(scale_to_slot[scale] for scale in sorted_scales)
    assignment_original_indices = tuple(entry[0] for entry in sorted_area_entries)
    assigned_by_slot = tuple(
        tuple(
            area
            for area, assigned_slot in zip(sorted_areas, assignment_slots)
            if assigned_slot == source_slot
        )
        for source_slot in range(1, len(scales) + 1)
    )
    source_denominators = tuple(
        _ordered_sum(
            values,
            stage="source_sorted_area2",
            source_slot=index + 1,
        )
        for index, values in enumerate(assigned_by_slot)
    )
    actual_denominators = tuple(
        _ordered_sum(values, stage="actual_group_area2", source_slot=index + 1)
        for index, values in enumerate(area_groups)
    )

    reassigned_slots = tuple(
        index + 1
        for index, (actual, assigned) in enumerate(zip(area_groups, assigned_by_slot))
        if sorted(actual) != sorted(assigned)
    )
    diagnostics = []
    for source_slot in reassigned_slots:
        diagnostics.append(
            Foams105VesicularityDiagnostic(
                code=DIAGNOSTIC_SOURCE_AREA_REASSIGNED,
                quantity="integrated_denominator_mapping",
                source_slot=source_slot,
                source_slots=reassigned_slots,
                actual_area2_values_mm2=area_groups[source_slot - 1],
                assigned_area2_values_mm2=assigned_by_slot[source_slot - 1],
                actual_denominator_mm2=actual_denominators[source_slot - 1],
                source_denominator_mm2=source_denominators[source_slot - 1],
            )
        )

    residuals = []
    multipliers: list[Optional[float]] = [None]
    removed = [selected[0]]
    first_residual = source_denominators[0] - selected[0]
    if not math.isfinite(first_residual):
        raise Foams105VesicularityNumericalError(
            "nonfinite_recurrence",
            "slot 1 residual is nonfinite",
            context={
                "stage": "slot1_residual",
                "source_slot": 1,
                "source_denominator": source_denominators[0],
                "selected_area": selected[0],
            },
        )
    residuals.append(first_residual)
    for index in range(1, len(scales)):
        source_slot = index + 1
        multiplier = _checked_divide(
            residuals[-1],
            source_denominators[index],
            stage="recurrence_multiplier",
            source_slot=source_slot,
            code="nonfinite_recurrence",
        )
        removed_area = _checked_multiply(
            multiplier,
            selected[index],
            stage="recurrence_removed_area",
            source_slot=source_slot,
            code="nonfinite_recurrence",
        )
        residual = residuals[-1] - removed_area
        if not math.isfinite(residual):
            raise Foams105VesicularityNumericalError(
                "nonfinite_recurrence",
                "recurrence residual is nonfinite",
                context={
                    "stage": "recurrence_residual",
                    "source_slot": source_slot,
                    "previous_residual": residuals[-1],
                    "removed_area": removed_area,
                },
            )
        multipliers.append(multiplier)
        removed.append(removed_area)
        residuals.append(residual)

    integrated_ratio = _checked_divide(
        residuals[-1],
        source_denominators[0],
        stage="integrated_residual_ratio",
        source_slot=None,
        code="nonfinite_recurrence",
    )
    integrated = _checked_multiply(
        1.0 - integrated_ratio,
        100.0,
        stage="integrated_percentage",
        source_slot=None,
        code="nonfinite_recurrence",
    )

    group_percentages = []
    for index, (pore_area, denominator) in enumerate(
        zip(unfiltered, actual_denominators)
    ):
        ratio = _checked_divide(
            pore_area,
            denominator,
            stage="group_percentage_ratio",
            source_slot=index + 1,
            code="nonfinite_group_summary",
        )
        percentage = _checked_multiply(
            ratio,
            100.0,
            stage="group_percentage",
            source_slot=index + 1,
            code="nonfinite_group_summary",
        )
        group_percentages.append(percentage)
    percentage_sum = _ordered_sum(
        group_percentages, stage="group_percentage_mean"
    )
    mean_percentage = _checked_divide(
        percentage_sum,
        float(len(group_percentages)),
        stage="group_percentage_mean",
        source_slot=None,
        code="nonfinite_group_summary",
    )

    for index, (denominator, selected_area, residual) in enumerate(
        zip(source_denominators, selected, residuals)
    ):
        source_slot = index + 1
        if selected_area > denominator:
            diagnostics.append(
                Foams105VesicularityDiagnostic(
                    code=DIAGNOSTIC_SELECTED_AREA_EXCEEDS_SOURCE_DENOMINATOR,
                    quantity="selected_area_sum_mm2",
                    source_slot=source_slot,
                    source_denominator_mm2=denominator,
                    numerator_mm2=selected_area,
                    value=selected_area,
                )
            )
        if residual < 0.0:
            diagnostics.append(
                Foams105VesicularityDiagnostic(
                    code=DIAGNOSTIC_NEGATIVE_RESIDUAL,
                    quantity="residual_reference_area_mm2",
                    source_slot=source_slot,
                    value=residual,
                )
            )
    if not 0.0 <= integrated <= 100.0:
        diagnostics.append(
            Foams105VesicularityDiagnostic(
                code=DIAGNOSTIC_PERCENTAGE_OUT_OF_RANGE,
                quantity="integrated_vesicularity_percent",
                value=integrated,
            )
        )
    for index, percentage in enumerate(group_percentages):
        if not 0.0 <= percentage <= 100.0:
            diagnostics.append(
                Foams105VesicularityDiagnostic(
                    code=DIAGNOSTIC_PERCENTAGE_OUT_OF_RANGE,
                    quantity="group_vesicularity_percent",
                    source_slot=index + 1,
                    value=percentage,
                )
            )
    if not 0.0 <= mean_percentage <= 100.0:
        diagnostics.append(
            Foams105VesicularityDiagnostic(
                code=DIAGNOSTIC_PERCENTAGE_OUT_OF_RANGE,
                quantity="mean_group_vesicularity_percent",
                value=mean_percentage,
            )
        )

    return _Foams105VesicularityNumericalResult(
        sorted_scales_px_per_mm=sorted_scales,
        sorted_area2_mm2=sorted_areas,
        assignment_source_slots=assignment_slots,
        assignment_original_flat_indices=assignment_original_indices,
        source_sorted_area2_by_slot_mm2=source_denominators,
        actual_group_area2_by_slot_mm2=actual_denominators,
        residuals_reference_area_mm2=tuple(residuals),
        multipliers=tuple(multipliers),
        removed_reference_area_mm2=tuple(removed),
        integrated_vesicularity_percent=integrated,
        group_vesicularity_percent=tuple(group_percentages),
        mean_group_vesicularity_percent=mean_percentage,
        reassigned_slots=reassigned_slots,
        diagnostics=tuple(diagnostics),
    )


def _validate_area_result(
    image: Foams105ImageAreaResult,
    *,
    group_id: str,
    source_slot: int,
    image_index: int,
) -> None:
    if not isinstance(image, Foams105ImageAreaResult):
        raise Foams105VesicularityValidationError(
            "image_area_results must contain Foams105ImageAreaResult records",
            context={
                "group_id": group_id,
                "source_slot": source_slot,
                "image_index": image_index,
            },
        )
    expected_metadata = (
        ("method", image.method, FOAMS105_IMAGE_AREA_METHOD),
        ("source_repository", image.source_repository, FOAMS105_AREA_SOURCE_REPOSITORY),
        ("source_commit", image.source_commit, FOAMS105_AREA_SOURCE_COMMIT),
        ("source_file", image.source_file, FOAMS105_AREA_SOURCE_FILE),
        ("scope", image.scope, FOAMS105_AREA_SCOPE),
        ("original_area1_mapping", image.original_area1_mapping, FOAMS105_AREA1_MAPPING),
        ("original_area2_mapping", image.original_area2_mapping, FOAMS105_AREA2_MAPPING),
        ("length_unit", image.length_unit, "mm"),
        ("area_unit", image.area_unit, "mm^2"),
        ("scale_unit", image.scale_unit, "pixels/mm"),
    )
    for field, actual, expected in expected_metadata:
        if actual != expected:
            raise Foams105VesicularityValidationError(
                f"image area record has unsupported {field}",
                code="inconsistent_area_provenance",
                context={
                    "group_id": group_id,
                    "source_slot": source_slot,
                    "image_id": image.image_id,
                    "field": field,
                    "actual": actual,
                    "expected": expected,
                },
            )
    numeric_fields = (
        "scale_px_per_mm",
        "full_area_mm2",
        "border_weighted_area_px",
        "border_area_mm2",
        "total_excluded_area_mm2",
        "border_corrected_area_mm2",
        "phase_corrected_area_mm2",
    )
    for field in numeric_fields:
        _real_scalar(getattr(image, field), f"image_area_results[{image_index}].{field}")
    if image.scale_px_per_mm <= 0.0:
        raise Foams105VesicularityValidationError(
            "image scale must be positive",
            context={"image_id": image.image_id, "source_slot": source_slot},
        )
    if image.border_corrected_area_mm2 <= 0.0:
        raise Foams105VesicularityDomainError(
            "nonpositive_border_corrected_area",
            "border-corrected image area must be positive",
            context={
                "group_id": group_id,
                "source_slot": source_slot,
                "image_id": image.image_id,
                "border_corrected_area_mm2": image.border_corrected_area_mm2,
            },
        )


def _validate_unfiltered_object(
    object_: object,
    *,
    group_id: str,
    source_slot: int,
    object_index: int,
) -> Foams105RawObject:
    if not isinstance(object_, Foams105RawObject):
        raise Foams105VesicularityValidationError(
            "unfiltered_pore_objects must contain Foams105RawObject records",
            context={
                "group_id": group_id,
                "source_slot": source_slot,
                "object_index": object_index,
            },
        )
    if not isinstance(object_.image_id, str) or not object_.image_id.strip():
        raise Foams105VesicularityValidationError(
            "unfiltered object image_id must be a nonempty string"
        )
    if (
        not isinstance(object_.component_id, (int, np.integer))
        or isinstance(object_.component_id, (bool, np.bool_))
        or int(object_.component_id) <= 0
    ):
        raise Foams105VesicularityValidationError(
            "unfiltered object component_id must be a positive integer"
        )
    diameter = _real_scalar(
        object_.equivalent_diameter_mm,
        f"unfiltered_pore_objects[{object_index}].equivalent_diameter_mm",
    )
    area = _real_scalar(
        object_.area_mm2, f"unfiltered_pore_objects[{object_index}].area_mm2"
    )
    if diameter <= 0.0 or area <= 0.0:
        raise Foams105VesicularityValidationError(
            "unfiltered object diameter and area must be positive",
            context={
                "group_id": group_id,
                "source_slot": source_slot,
                "object_index": object_index,
            },
        )
    return Foams105RawObject(
        object_.image_id, int(object_.component_id), diameter, area
    )


def calculate_foams105_2d_vesicularity(
    raw_selection: Foams105RawSelectionResult,
    groups: Sequence[Foams105VesicularityGroupInput],
) -> Foams105VesicularityResult:
    """Calculate integrated and actual-membership two-dimensional summaries."""
    if not isinstance(raw_selection, Foams105RawSelectionResult):
        raise Foams105VesicularityValidationError(
            "raw_selection must be a Foams105RawSelectionResult"
        )
    try:
        canonical_selection = select_foams105_raw_objects(
            raw_selection.bin_labels_mm,
            raw_selection.groups,
            raw_selection.requested_ranges,
        )
    except (ValueError, ArithmeticError) as exc:
        raise Foams105VesicularityValidationError(
            "raw_selection cannot be replayed from its inputs",
            code="inconsistent_raw_selection",
        ) from exc
    if canonical_selection != raw_selection:
        raise Foams105VesicularityValidationError(
            "raw_selection differs from its canonical replay",
            code="inconsistent_raw_selection",
        )

    raw_groups = _sequence(groups, "groups")
    if len(raw_groups) != len(canonical_selection.groups):
        raise Foams105VesicularityValidationError(
            "groups must exactly match raw_selection groups",
            context={
                "context_group_count": len(raw_groups),
                "raw_group_count": len(canonical_selection.groups),
            },
        )

    normalized_groups = []
    area_results = []
    all_image_ids = set()
    all_unfiltered_identities = set()
    previous_scale = None
    total_images = 0
    for group_index, (context, raw_group) in enumerate(
        zip(raw_groups, canonical_selection.groups)
    ):
        source_slot = group_index + 1
        if not isinstance(context, Foams105VesicularityGroupInput):
            raise Foams105VesicularityValidationError(
                f"groups[{group_index}] must be a Foams105VesicularityGroupInput"
            )
        if (
            not isinstance(context.source_slot, (int, np.integer))
            or isinstance(context.source_slot, (bool, np.bool_))
        ):
            raise Foams105VesicularityValidationError(
                f"groups[{group_index}].source_slot must be an integer",
                context={
                    "group_index": group_index,
                    "actual_source_slot": context.source_slot,
                },
            )
        normalized_source_slot = int(context.source_slot)
        if (
            context.group_id != raw_group.group_id
            or normalized_source_slot != source_slot
        ):
            raise Foams105VesicularityValidationError(
                "group ID and source slot must exactly match raw_selection",
                context={
                    "group_index": group_index,
                    "expected_group_id": raw_group.group_id,
                    "actual_group_id": context.group_id,
                    "expected_source_slot": source_slot,
                    "actual_source_slot": normalized_source_slot,
                },
            )
        if not isinstance(context.image_area_results, tuple):
            raise Foams105VesicularityValidationError(
                f"groups[{group_index}].image_area_results must be an immutable tuple"
            )
        if not context.image_area_results:
            raise Foams105VesicularityValidationError(
                f"groups[{group_index}] must contain at least one image"
            )
        if not isinstance(context.unfiltered_pore_objects, tuple):
            raise Foams105VesicularityValidationError(
                f"groups[{group_index}].unfiltered_pore_objects must be an immutable tuple"
            )

        images = tuple(context.image_area_results)
        total_images += len(images)
        for image_index, image in enumerate(images):
            _validate_area_result(
                image,
                group_id=context.group_id,
                source_slot=source_slot,
                image_index=image_index,
            )
            if image.image_id in all_image_ids:
                raise Foams105VesicularityValidationError(
                    "image IDs must be globally unique",
                    code="duplicate_image_id",
                    context={"image_id": image.image_id},
                )
            all_image_ids.add(image.image_id)
        if total_images > FOAMS105_VESICULARITY_MAX_IMAGES:
            raise Foams105VesicularityValidationError(
                f"at most {FOAMS105_VESICULARITY_MAX_IMAGES} images are supported"
            )
        scale = images[0].scale_px_per_mm
        if any(image.scale_px_per_mm != scale for image in images[1:]):
            raise Foams105VesicularityValidationError(
                "all images in a group must have identical scales",
                context={"group_id": context.group_id, "source_slot": source_slot},
            )
        if previous_scale is not None and scale <= previous_scale:
            raise Foams105VesicularityValidationError(
                "group scales must be strictly increasing by source slot",
                context={
                    "group_id": context.group_id,
                    "source_slot": source_slot,
                    "scale_px_per_mm": scale,
                    "previous_scale_px_per_mm": previous_scale,
                },
            )
        previous_scale = scale
        try:
            area_result = aggregate_foams105_group_areas(context.group_id, images)
        except Foams105AreaValidationError as exc:
            raise Foams105VesicularityValidationError(
                "group image-area records are incompatible",
                code="inconsistent_area_provenance",
                context={"group_id": context.group_id, "source_slot": source_slot},
            ) from exc
        except Foams105AreaNumericalError as exc:
            raise Foams105VesicularityNumericalError(
                "nonfinite_area_aggregation",
                "group area aggregation failed numerically",
                context={"group_id": context.group_id, "source_slot": source_slot},
            ) from exc
        if (
            area_result.method != FOAMS105_GROUP_AREA_METHOD
            or area_result.source_repository != FOAMS105_AREA_SOURCE_REPOSITORY
            or area_result.source_commit != FOAMS105_AREA_SOURCE_COMMIT
            or area_result.source_file != FOAMS105_AREA_SOURCE_FILE
            or area_result.scope != FOAMS105_AREA_SCOPE
        ):
            raise Foams105VesicularityValidationError(
                "group area aggregation returned unsupported provenance"
            )

        image_ids = {image.image_id for image in images}
        objects = []
        for object_index, object_ in enumerate(context.unfiltered_pore_objects):
            normalized = _validate_unfiltered_object(
                object_,
                group_id=context.group_id,
                source_slot=source_slot,
                object_index=object_index,
            )
            identity = (normalized.image_id, normalized.component_id)
            if identity in all_unfiltered_identities:
                raise Foams105VesicularityValidationError(
                    "unfiltered object identities must be globally unique",
                    code="duplicate_object_identity",
                    context={"identity": identity},
                )
            if normalized.image_id not in image_ids:
                raise Foams105VesicularityValidationError(
                    "unfiltered object image is not a member of its group",
                    code="foreign_image_membership",
                    context={
                        "identity": identity,
                        "group_id": context.group_id,
                        "source_slot": source_slot,
                    },
                )
            all_unfiltered_identities.add(identity)
            objects.append(normalized)

        normalized_context = Foams105VesicularityGroupInput(
            context.group_id, source_slot, images, tuple(objects)
        )
        normalized_groups.append(normalized_context)
        area_results.append(area_result)

    unfiltered_maps = tuple(
        {
            (object_.image_id, object_.component_id): object_
            for object_ in context.unfiltered_pore_objects
        }
        for context in normalized_groups
    )
    for raw_group, inventory in zip(canonical_selection.groups, unfiltered_maps):
        for object_ in raw_group.objects:
            identity = (object_.image_id, object_.component_id)
            if identity not in inventory:
                raise Foams105VesicularityValidationError(
                    "raw-selection object is missing from the unfiltered inventory",
                    code="missing_selected_object",
                    context={"group_id": raw_group.group_id, "identity": identity},
                )
            if inventory[identity] != object_:
                raise Foams105VesicularityValidationError(
                    "raw-selection object geometry differs from unfiltered inventory",
                    code="changed_object_geometry",
                    context={"group_id": raw_group.group_id, "identity": identity},
                )

    selected_objects_by_group = tuple(
        trace.selected_objects for trace in canonical_selection.group_traces
    )
    selected_sums = tuple(
        _ordered_sum(
            tuple(object_.area_mm2 for object_ in objects),
            stage="selected_object_area",
            source_slot=index + 1,
        )
        for index, objects in enumerate(selected_objects_by_group)
    )
    unfiltered_sums = tuple(
        _ordered_sum(
            tuple(object_.area_mm2 for object_ in context.unfiltered_pore_objects),
            stage="unfiltered_object_area",
            source_slot=index + 1,
        )
        for index, context in enumerate(normalized_groups)
    )
    numerical = _calculate_foams105_vesicularity_numerical(
        tuple(context.image_area_results[0].scale_px_per_mm for context in normalized_groups),
        tuple(
            tuple(image.border_corrected_area_mm2 for image in context.image_area_results)
            for context in normalized_groups
        ),
        selected_sums,
        unfiltered_sums,
    )

    flattened = tuple(
        (flat_index, context, image)
        for flat_index, (context, image) in enumerate(
            (context, image)
            for context in normalized_groups
            for image in context.image_area_results
        )
    )
    assigned_values_by_slot = tuple(
        tuple(
            numerical.sorted_area2_mm2[position]
            for position, slot in enumerate(numerical.assignment_source_slots)
            if slot == source_slot
        )
        for source_slot in range(1, len(normalized_groups) + 1)
    )
    assignments = tuple(
        Foams105SourceAreaAssignment(
            sorted_position=position,
            assigned_source_slot=numerical.assignment_source_slots[position],
            assigned_group_id=normalized_groups[
                numerical.assignment_source_slots[position] - 1
            ].group_id,
            sorted_scale_px_per_mm=numerical.sorted_scales_px_per_mm[position],
            area2_mm2=numerical.sorted_area2_mm2[position],
            original_flat_image_index=numerical.assignment_original_flat_indices[position],
            original_source_slot=flattened[
                numerical.assignment_original_flat_indices[position]
            ][1].source_slot,
            original_group_id=flattened[
                numerical.assignment_original_flat_indices[position]
            ][1].group_id,
            original_image_id=flattened[
                numerical.assignment_original_flat_indices[position]
            ][2].image_id,
        )
        for position in range(len(flattened))
    )

    diagnostics = list(numerical.diagnostics)
    enriched_diagnostics = []
    for diagnostic in diagnostics:
        if diagnostic.code != DIAGNOSTIC_SOURCE_AREA_REASSIGNED:
            enriched_diagnostics.append(diagnostic)
            continue
        source_slot = diagnostic.source_slot
        if source_slot is None:
            raise Foams105VesicularityNumericalError(
                "invalid_diagnostic_context",
                "source-area reassignment diagnostic has no source slot",
            )
        enriched_diagnostics.append(
            Foams105VesicularityDiagnostic(
                code=diagnostic.code,
                quantity=diagnostic.quantity,
                source_slot=source_slot,
                source_slots=diagnostic.source_slots,
                image_ids=tuple(
                    image.image_id
                    for image in normalized_groups[source_slot - 1].image_area_results
                ),
                actual_area2_values_mm2=diagnostic.actual_area2_values_mm2,
                assigned_area2_values_mm2=diagnostic.assigned_area2_values_mm2,
                actual_denominator_mm2=diagnostic.actual_denominator_mm2,
                source_denominator_mm2=diagnostic.source_denominator_mm2,
            )
        )

    group_traces = tuple(
        Foams105VesicularityGroupTrace(
            group_id=context.group_id,
            source_slot=context.source_slot,
            scale_px_per_mm=context.image_area_results[0].scale_px_per_mm,
            image_ids=tuple(image.image_id for image in context.image_area_results),
            image_area2_mm2=tuple(
                image.border_corrected_area_mm2 for image in context.image_area_results
            ),
            assigned_source_area2_mm2=assigned_values_by_slot[index],
            area_result=area_results[index],
            source_sorted_area2_mm2=numerical.source_sorted_area2_by_slot_mm2[index],
            actual_group_area2_mm2=numerical.actual_group_area2_by_slot_mm2[index],
            phase_corrected_area1_mm2=area_results[index].phase_corrected_area_mm2,
            area_diagnostic_codes=area_results[index].diagnostic_codes,
            selected_objects=selected_objects_by_group[index],
            selected_identities=tuple(
                (object_.image_id, object_.component_id)
                for object_ in selected_objects_by_group[index]
            ),
            selected_area_sum_mm2=selected_sums[index],
            selected_count=len(selected_objects_by_group[index]),
            unfiltered_pore_objects=context.unfiltered_pore_objects,
            unfiltered_identities=tuple(
                (object_.image_id, object_.component_id)
                for object_ in context.unfiltered_pore_objects
            ),
            unfiltered_area_sum_mm2=unfiltered_sums[index],
            unfiltered_count=len(context.unfiltered_pore_objects),
            residual_reference_area_mm2=numerical.residuals_reference_area_mm2[index],
            multiplier=numerical.multipliers[index],
            removed_reference_area_mm2=numerical.removed_reference_area_mm2[index],
            group_vesicularity_percent=numerical.group_vesicularity_percent[index],
        )
        for index, context in enumerate(normalized_groups)
    )

    return Foams105VesicularityResult(
        raw_selection=canonical_selection,
        groups=tuple(normalized_groups),
        group_traces=group_traces,
        sorted_scales_px_per_mm=numerical.sorted_scales_px_per_mm,
        sorted_area2_mm2=numerical.sorted_area2_mm2,
        source_area_assignments=assignments,
        source_sorted_area2_by_slot_mm2=(
            numerical.source_sorted_area2_by_slot_mm2
        ),
        actual_group_area2_by_slot_mm2=(
            numerical.actual_group_area2_by_slot_mm2
        ),
        selected_area_by_slot_mm2=selected_sums,
        unfiltered_area_by_slot_mm2=unfiltered_sums,
        residuals_reference_area_mm2=numerical.residuals_reference_area_mm2,
        multipliers=numerical.multipliers,
        removed_reference_area_mm2=numerical.removed_reference_area_mm2,
        integrated_vesicularity_percent=(
            numerical.integrated_vesicularity_percent
        ),
        group_vesicularity_percent=numerical.group_vesicularity_percent,
        mean_group_vesicularity_percent=(
            numerical.mean_group_vesicularity_percent
        ),
        diagnostics=tuple(enriched_diagnostics),
    )
