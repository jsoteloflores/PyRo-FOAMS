"""FOAMS 1.0.5 border and excluded-phase area accounting.

This module consumes immutable legacy measurement results. It implements the
documented ``bwarea`` neighborhood estimate, strict excluded-component area
thresholds, two source denominators, and explicit magnification-group sums.
It does not rerun topology or implement vesicularity, nesting, or shape data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .legacy_measurements import (
    FOAMS105_EXCLUSION_SLOT_COUNT,
    FOAMS105_MEASUREMENT_SOURCE_COMMIT,
    FOAMS105_MEASUREMENT_SOURCE_REPOSITORY,
    Foams105ImageMeasurementResult,
    PhaseScalar,
)

FOAMS105_AREA_SOURCE_REPOSITORY = FOAMS105_MEASUREMENT_SOURCE_REPOSITORY
FOAMS105_AREA_SOURCE_COMMIT = FOAMS105_MEASUREMENT_SOURCE_COMMIT
FOAMS105_AREA_SOURCE_FILE = "project.m"
FOAMS105_BINARY_AREA_METHOD = "foams_1_0_5_bwarea_weighted_v1"
FOAMS105_IMAGE_AREA_METHOD = "foams_1_0_5_image_area_accounting_v1"
FOAMS105_GROUP_AREA_METHOD = "foams_1_0_5_group_area_aggregation_v1"
FOAMS105_AREA_SCOPE = "area_accounting_only"
FOAMS105_AREA1_MAPPING = "phase_corrected_area_mm2"
FOAMS105_AREA2_MAPPING = "border_corrected_area_mm2"
FOAMS105_BWAREA_WEIGHT_EIGHTHS = (
    0,
    2,
    2,
    4,
    2,
    4,
    6,
    7,
    2,
    6,
    4,
    7,
    4,
    7,
    7,
    8,
)

DIAGNOSTIC_BORDER_AREA_NONPOSITIVE = "border_corrected_area_nonpositive"
DIAGNOSTIC_PHASE_AREA_NONPOSITIVE = "phase_corrected_area_nonpositive"
DIAGNOSTIC_PHASE_VALUE_COLLISION = "phase_value_collision"
DIAGNOSTIC_CONSTITUENT_BORDER_AREA_NONPOSITIVE = (
    "constituent_border_corrected_area_nonpositive"
)
DIAGNOSTIC_CONSTITUENT_PHASE_AREA_NONPOSITIVE = (
    "constituent_phase_corrected_area_nonpositive"
)
DIAGNOSTIC_CONSTITUENT_PHASE_VALUE_COLLISION = "constituent_phase_value_collision"


class Foams105AreaValidationError(ValueError):
    """Raised when area-accounting inputs violate the bounded contract."""


class Foams105AreaNumericalError(ArithmeticError):
    """Raised when physical area arithmetic becomes nonfinite or underflows."""


@dataclass(frozen=True)
class Foams105BinaryAreaResult:
    """Weighted binary-mask area in pixel-area units."""

    mask_shape: Tuple[int, int]
    weighted_area_px: float
    foreground_pixel_count: int
    method: str = FOAMS105_BINARY_AREA_METHOD
    source_repository: str = FOAMS105_AREA_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_AREA_SOURCE_COMMIT
    source_file: str = FOAMS105_AREA_SOURCE_FILE
    scope: str = FOAMS105_AREA_SCOPE
    input_policy: str = "explicit_binary_zero_or_one"
    boundary_policy: str = "one_pixel_zero_padding"
    weighted_area_unit: str = "pixel-area"


@dataclass(frozen=True)
class Foams105ExcludedAreaSlotResult:
    """One stable excluded-phase slot after strict component-area filtering."""

    slot_index: int
    phase_value: Optional[PhaseScalar]
    enabled: bool
    minimum_area_mm2: float
    selected_component_ids: Tuple[int, ...]
    rejected_component_ids: Tuple[int, ...]
    excluded_area_mm2: float
    threshold_policy: str = "component_area_strictly_greater_than_threshold"


@dataclass(frozen=True)
class Foams105ImageAreaResult:
    """All source area terms and signed denominators for one image."""

    image_id: str
    image_shape: Tuple[int, int]
    scale_px_per_mm: float
    excluded_phase_values: Tuple[
        Optional[PhaseScalar],
        Optional[PhaseScalar],
        Optional[PhaseScalar],
        Optional[PhaseScalar],
    ]
    excluded_min_area_mm2: Tuple[float, float, float, float]
    border_binary_area: Foams105BinaryAreaResult
    full_area_mm2: float
    border_weighted_area_px: float
    border_area_mm2: float
    excluded_slots: Tuple[
        Foams105ExcludedAreaSlotResult,
        Foams105ExcludedAreaSlotResult,
        Foams105ExcludedAreaSlotResult,
        Foams105ExcludedAreaSlotResult,
    ]
    excluded_area_mm2_by_slot: Tuple[float, float, float, float]
    total_excluded_area_mm2: float
    border_corrected_area_mm2: float
    phase_corrected_area_mm2: float
    border_corrected_area_is_positive: bool
    phase_corrected_area_is_positive: bool
    diagnostic_codes: Tuple[str, ...]
    phase_value_collision_slot_pairs: Tuple[Tuple[str, str], ...]
    method: str = FOAMS105_IMAGE_AREA_METHOD
    source_repository: str = FOAMS105_AREA_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_AREA_SOURCE_COMMIT
    source_file: str = FOAMS105_AREA_SOURCE_FILE
    scope: str = FOAMS105_AREA_SCOPE
    original_area1_mapping: str = FOAMS105_AREA1_MAPPING
    original_area2_mapping: str = FOAMS105_AREA2_MAPPING
    length_unit: str = "mm"
    area_unit: str = "mm^2"
    scale_unit: str = "pixels/mm"


@dataclass(frozen=True)
class Foams105GroupAreaResult:
    """Ordered sum of compatible per-image area terms for one group."""

    group_id: str
    image_ids: Tuple[str, ...]
    scale_px_per_mm: float
    excluded_phase_values: Tuple[
        Optional[PhaseScalar],
        Optional[PhaseScalar],
        Optional[PhaseScalar],
        Optional[PhaseScalar],
    ]
    excluded_min_area_mm2: Tuple[float, float, float, float]
    full_area_mm2: float
    border_weighted_area_px: float
    border_area_mm2: float
    excluded_area_mm2_by_slot: Tuple[float, float, float, float]
    total_excluded_area_mm2: float
    border_corrected_area_mm2: float
    phase_corrected_area_mm2: float
    border_corrected_area_is_positive: bool
    phase_corrected_area_is_positive: bool
    nonpositive_border_corrected_image_ids: Tuple[str, ...]
    nonpositive_phase_corrected_image_ids: Tuple[str, ...]
    phase_collision_image_ids: Tuple[str, ...]
    diagnostic_codes: Tuple[str, ...]
    method: str = FOAMS105_GROUP_AREA_METHOD
    source_repository: str = FOAMS105_AREA_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_AREA_SOURCE_COMMIT
    source_file: str = FOAMS105_AREA_SOURCE_FILE
    scope: str = FOAMS105_AREA_SCOPE
    original_area1_mapping: str = FOAMS105_AREA1_MAPPING
    original_area2_mapping: str = FOAMS105_AREA2_MAPPING
    length_unit: str = "mm"
    area_unit: str = "mm^2"
    scale_unit: str = "pixels/mm"


def _validated_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Foams105AreaValidationError(f"{field} must be a nonempty string")
    return value


def _validated_thresholds(values: object) -> Tuple[float, float, float, float]:
    if isinstance(values, (str, bytes)):
        raise Foams105AreaValidationError(
            "excluded_min_area_mm2 must contain exactly four values"
        )
    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105AreaValidationError(
            "excluded_min_area_mm2 must contain exactly four values"
        ) from exc
    if len(raw_values) != FOAMS105_EXCLUSION_SLOT_COUNT:
        raise Foams105AreaValidationError(
            "excluded_min_area_mm2 must contain exactly four values"
        )
    thresholds = []
    for index, value in enumerate(raw_values):
        if (
            not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_))
        ):
            raise Foams105AreaValidationError(
                f"excluded_min_area_mm2[{index}] must be a finite nonnegative real scalar"
            )
        try:
            threshold = float(value)
        except (OverflowError, ValueError) as exc:
            raise Foams105AreaValidationError(
                f"excluded_min_area_mm2[{index}] must be representable as finite float64"
            ) from exc
        if not math.isfinite(threshold) or threshold < 0.0:
            raise Foams105AreaValidationError(
                f"excluded_min_area_mm2[{index}] must be finite and nonnegative"
            )
        thresholds.append(threshold)
    return tuple(thresholds)  # type: ignore[return-value]


def _validated_binary_mask(mask: object) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise Foams105AreaValidationError("mask must be a NumPy array")
    if mask.ndim != 2 or mask.size == 0 or not all(mask.shape):
        raise Foams105AreaValidationError(
            "mask must be a nonempty two-dimensional array"
        )
    if mask.dtype.kind not in "biuf":
        raise Foams105AreaValidationError(
            "mask dtype must be Boolean, integer, or real floating"
        )
    if mask.dtype.kind == "f" and not bool(np.isfinite(mask).all()):
        raise Foams105AreaValidationError("mask must contain only finite values")
    if not bool(np.logical_or(mask == 0, mask == 1).all()):
        raise Foams105AreaValidationError("mask values must be exactly 0 or 1")
    return mask.astype(bool, copy=False)


def estimate_foams105_binary_area(mask: np.ndarray) -> Foams105BinaryAreaResult:
    """Estimate binary area with documented padded 2-by-2 ``bwarea`` weights."""
    binary = _validated_binary_mask(mask)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    codes = (
        padded[:-1, :-1].astype(np.uint8)
        + 2 * padded[:-1, 1:].astype(np.uint8)
        + 4 * padded[1:, :-1].astype(np.uint8)
        + 8 * padded[1:, 1:].astype(np.uint8)
    )
    weight_lookup = np.asarray(FOAMS105_BWAREA_WEIGHT_EIGHTHS, dtype=np.uint8)
    accumulated_eighths = int(weight_lookup[codes].sum(dtype=np.uint64))
    return Foams105BinaryAreaResult(
        mask_shape=(int(binary.shape[0]), int(binary.shape[1])),
        weighted_area_px=accumulated_eighths / 8.0,
        foreground_pixel_count=int(np.count_nonzero(binary)),
    )


def _calibrated_area(
    pixel_area: float,
    scale: float,
    field: str,
    *,
    require_positive: bool,
) -> float:
    if pixel_area == 0.0 and not require_positive:
        return 0.0
    try:
        area = (float(pixel_area) / scale) / scale
    except (OverflowError, ValueError) as exc:
        raise Foams105AreaNumericalError(f"{field} calibration failed") from exc
    if not math.isfinite(area) or (require_positive and area <= 0.0):
        raise Foams105AreaNumericalError(
            f"{field} must be positive and finite after calibration"
        )
    if pixel_area > 0.0 and area <= 0.0:
        raise Foams105AreaNumericalError(
            f"{field} underflowed to zero during calibration"
        )
    return area


def _validate_measurement_consistency(
    measurement: Foams105ImageMeasurementResult,
) -> None:
    if not isinstance(measurement, Foams105ImageMeasurementResult):
        raise Foams105AreaValidationError(
            "measurement_result must be a Foams105ImageMeasurementResult"
        )
    if len(measurement.excluded_phase_values) != FOAMS105_EXCLUSION_SLOT_COUNT:
        raise Foams105AreaValidationError(
            "measurement_result must contain exactly four exclusion values"
        )
    if len(measurement.excluded_phase_results) != FOAMS105_EXCLUSION_SLOT_COUNT:
        raise Foams105AreaValidationError(
            "measurement_result must contain exactly four exclusion results"
        )
    pore = measurement.pore_result
    if pore.image_shape != measurement.image_shape:
        raise Foams105AreaValidationError(
            "measurement_result pore image shape is inconsistent"
        )
    if pore.scale_px_per_mm != measurement.scale_px_per_mm:
        raise Foams105AreaValidationError(
            "measurement_result pore scale is inconsistent"
        )
    if pore.removed_border_mask.shape != measurement.image_shape:
        raise Foams105AreaValidationError(
            "measurement_result pore border mask shape is inconsistent"
        )
    if not math.isfinite(measurement.scale_px_per_mm) or measurement.scale_px_per_mm <= 0.0:
        raise Foams105AreaValidationError(
            "measurement_result scale must be finite and positive"
        )
    for index, (phase_value, result) in enumerate(
        zip(measurement.excluded_phase_values, measurement.excluded_phase_results)
    ):
        if phase_value is None:
            if result is not None:
                raise Foams105AreaValidationError(
                    f"measurement_result exclusion slot {index} is disabled but has a result"
                )
            continue
        if result is None:
            raise Foams105AreaValidationError(
                f"measurement_result exclusion slot {index} is enabled but has no result"
            )
        if result.phase_value != phase_value:
            raise Foams105AreaValidationError(
                f"measurement_result exclusion slot {index} phase value is inconsistent"
            )
        if result.image_shape != measurement.image_shape:
            raise Foams105AreaValidationError(
                f"measurement_result exclusion slot {index} image shape is inconsistent"
            )
        if result.scale_px_per_mm != measurement.scale_px_per_mm:
            raise Foams105AreaValidationError(
                f"measurement_result exclusion slot {index} scale is inconsistent"
            )
        for component in result.components:
            if not math.isfinite(component.area_mm2) or component.area_mm2 <= 0.0:
                raise Foams105AreaValidationError(
                    f"measurement_result exclusion slot {index} component "
                    f"{component.component_id} area must be positive and finite"
                )


def _ordered_sum(values: Sequence[float], field: str) -> float:
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if not math.isfinite(total):
            raise Foams105AreaNumericalError(
                f"{field} ordered sum is nonfinite at index {index}"
            )
    return total


def calculate_foams105_image_areas(
    image_id: str,
    measurement_result: Foams105ImageMeasurementResult,
    excluded_min_area_mm2: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
) -> Foams105ImageAreaResult:
    """Calculate both signed source denominators from one measured image."""
    validated_image_id = _validated_id(image_id, "image_id")
    _validate_measurement_consistency(measurement_result)
    thresholds = _validated_thresholds(excluded_min_area_mm2)
    height, width = measurement_result.image_shape
    scale = measurement_result.scale_px_per_mm

    full_area = _calibrated_area(
        float(height * width), scale, "full_area_mm2", require_positive=True
    )
    border_binary_area = estimate_foams105_binary_area(
        measurement_result.pore_result.removed_border_mask
    )
    border_area = _calibrated_area(
        border_binary_area.weighted_area_px,
        scale,
        "border_area_mm2",
        require_positive=False,
    )

    slot_results = []
    excluded_areas = []
    for index, (phase_value, phase_result, threshold) in enumerate(
        zip(
            measurement_result.excluded_phase_values,
            measurement_result.excluded_phase_results,
            thresholds,
        )
    ):
        if phase_result is None:
            slot_results.append(
                Foams105ExcludedAreaSlotResult(
                    slot_index=index,
                    phase_value=phase_value,
                    enabled=False,
                    minimum_area_mm2=threshold,
                    selected_component_ids=(),
                    rejected_component_ids=(),
                    excluded_area_mm2=0.0,
                )
            )
            excluded_areas.append(0.0)
            continue
        selected_ids = tuple(
            component.component_id
            for component in phase_result.components
            if component.area_mm2 > threshold
        )
        rejected_ids = tuple(
            component.component_id
            for component in phase_result.components
            if component.area_mm2 <= threshold
        )
        selected_areas = tuple(
            component.area_mm2
            for component in phase_result.components
            if component.area_mm2 > threshold
        )
        excluded_area = _ordered_sum(
            selected_areas, f"excluded_area_mm2_by_slot[{index}]"
        )
        slot_results.append(
            Foams105ExcludedAreaSlotResult(
                slot_index=index,
                phase_value=phase_value,
                enabled=True,
                minimum_area_mm2=threshold,
                selected_component_ids=selected_ids,
                rejected_component_ids=rejected_ids,
                excluded_area_mm2=excluded_area,
            )
        )
        excluded_areas.append(excluded_area)

    border_corrected = full_area - border_area
    if not math.isfinite(border_corrected):
        raise Foams105AreaNumericalError("border_corrected_area_mm2 is nonfinite")
    phase_corrected = border_corrected
    for index, excluded_area in enumerate(excluded_areas):
        phase_corrected -= excluded_area
        if not math.isfinite(phase_corrected):
            raise Foams105AreaNumericalError(
                f"phase_corrected_area_mm2 is nonfinite after slot {index}"
            )
    total_excluded = _ordered_sum(excluded_areas, "total_excluded_area_mm2")

    diagnostics = []
    if border_corrected <= 0.0:
        diagnostics.append(DIAGNOSTIC_BORDER_AREA_NONPOSITIVE)
    if phase_corrected <= 0.0:
        diagnostics.append(DIAGNOSTIC_PHASE_AREA_NONPOSITIVE)
    if measurement_result.phase_value_collision_slot_pairs:
        diagnostics.append(DIAGNOSTIC_PHASE_VALUE_COLLISION)

    return Foams105ImageAreaResult(
        image_id=validated_image_id,
        image_shape=measurement_result.image_shape,
        scale_px_per_mm=scale,
        excluded_phase_values=measurement_result.excluded_phase_values,
        excluded_min_area_mm2=thresholds,
        border_binary_area=border_binary_area,
        full_area_mm2=full_area,
        border_weighted_area_px=border_binary_area.weighted_area_px,
        border_area_mm2=border_area,
        excluded_slots=tuple(slot_results),  # type: ignore[arg-type]
        excluded_area_mm2_by_slot=tuple(excluded_areas),  # type: ignore[arg-type]
        total_excluded_area_mm2=total_excluded,
        border_corrected_area_mm2=border_corrected,
        phase_corrected_area_mm2=phase_corrected,
        border_corrected_area_is_positive=border_corrected > 0.0,
        phase_corrected_area_is_positive=phase_corrected > 0.0,
        diagnostic_codes=tuple(diagnostics),
        phase_value_collision_slot_pairs=(
            measurement_result.phase_value_collision_slot_pairs
        ),
    )


def aggregate_foams105_group_areas(
    group_id: str,
    image_area_results: Sequence[Foams105ImageAreaResult],
) -> Foams105GroupAreaResult:
    """Aggregate compatible image area terms in caller-supplied order."""
    validated_group_id = _validated_id(group_id, "group_id")
    if isinstance(image_area_results, (str, bytes)):
        raise Foams105AreaValidationError(
            "image_area_results must be a nonempty sequence"
        )
    try:
        images = tuple(image_area_results)
    except TypeError as exc:
        raise Foams105AreaValidationError(
            "image_area_results must be a nonempty sequence"
        ) from exc
    if not images:
        raise Foams105AreaValidationError(
            "image_area_results must contain at least one image"
        )
    for index, result in enumerate(images):
        if not isinstance(result, Foams105ImageAreaResult):
            raise Foams105AreaValidationError(
                f"image_area_results[{index}] must be a Foams105ImageAreaResult"
            )
    image_ids = tuple(result.image_id for result in images)
    if len(set(image_ids)) != len(image_ids):
        raise Foams105AreaValidationError("image_area_results contain duplicate image IDs")

    reference = images[0]
    for index, result in enumerate(images[1:], start=1):
        if result.scale_px_per_mm != reference.scale_px_per_mm:
            raise Foams105AreaValidationError(
                f"image_area_results[{index}] scale differs from the group"
            )
        if result.excluded_phase_values != reference.excluded_phase_values:
            raise Foams105AreaValidationError(
                f"image_area_results[{index}] exclusion configuration differs from the group"
            )
        if result.excluded_min_area_mm2 != reference.excluded_min_area_mm2:
            raise Foams105AreaValidationError(
                f"image_area_results[{index}] exclusion thresholds differ from the group"
            )

    full_area = _ordered_sum(
        tuple(result.full_area_mm2 for result in images), "full_area_mm2"
    )
    border_weighted = _ordered_sum(
        tuple(result.border_weighted_area_px for result in images),
        "border_weighted_area_px",
    )
    border_area = _ordered_sum(
        tuple(result.border_area_mm2 for result in images), "border_area_mm2"
    )
    slot_totals = tuple(
        _ordered_sum(
            tuple(result.excluded_area_mm2_by_slot[slot] for result in images),
            f"excluded_area_mm2_by_slot[{slot}]",
        )
        for slot in range(FOAMS105_EXCLUSION_SLOT_COUNT)
    )
    total_excluded = _ordered_sum(
        tuple(result.total_excluded_area_mm2 for result in images),
        "total_excluded_area_mm2",
    )
    border_corrected = _ordered_sum(
        tuple(result.border_corrected_area_mm2 for result in images),
        "border_corrected_area_mm2",
    )
    phase_corrected = _ordered_sum(
        tuple(result.phase_corrected_area_mm2 for result in images),
        "phase_corrected_area_mm2",
    )
    nonpositive_border_ids = tuple(
        result.image_id
        for result in images
        if not result.border_corrected_area_is_positive
    )
    nonpositive_phase_ids = tuple(
        result.image_id
        for result in images
        if not result.phase_corrected_area_is_positive
    )
    collision_ids = tuple(
        result.image_id
        for result in images
        if result.phase_value_collision_slot_pairs
    )

    diagnostics = []
    if border_corrected <= 0.0:
        diagnostics.append(DIAGNOSTIC_BORDER_AREA_NONPOSITIVE)
    if phase_corrected <= 0.0:
        diagnostics.append(DIAGNOSTIC_PHASE_AREA_NONPOSITIVE)
    if nonpositive_border_ids:
        diagnostics.append(DIAGNOSTIC_CONSTITUENT_BORDER_AREA_NONPOSITIVE)
    if nonpositive_phase_ids:
        diagnostics.append(DIAGNOSTIC_CONSTITUENT_PHASE_AREA_NONPOSITIVE)
    if collision_ids:
        diagnostics.append(DIAGNOSTIC_CONSTITUENT_PHASE_VALUE_COLLISION)

    return Foams105GroupAreaResult(
        group_id=validated_group_id,
        image_ids=image_ids,
        scale_px_per_mm=reference.scale_px_per_mm,
        excluded_phase_values=reference.excluded_phase_values,
        excluded_min_area_mm2=reference.excluded_min_area_mm2,
        full_area_mm2=full_area,
        border_weighted_area_px=border_weighted,
        border_area_mm2=border_area,
        excluded_area_mm2_by_slot=slot_totals,  # type: ignore[arg-type]
        total_excluded_area_mm2=total_excluded,
        border_corrected_area_mm2=border_corrected,
        phase_corrected_area_mm2=phase_corrected,
        border_corrected_area_is_positive=border_corrected > 0.0,
        phase_corrected_area_is_positive=phase_corrected > 0.0,
        nonpositive_border_corrected_image_ids=nonpositive_border_ids,
        nonpositive_phase_corrected_image_ids=nonpositive_phase_ids,
        phase_collision_image_ids=collision_ids,
        diagnostic_codes=tuple(diagnostics),
    )
