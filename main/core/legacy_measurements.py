"""FOAMS 1.0.5 mask topology and calibrated pixel-area measurements.

This bounded replay selects exact image phases, removes 8-connected border
components, fills holes through 4-connected background topology, and labels
the filled foreground with 4-connectivity in MATLAB column-major order. It
does not implement shape properties, ``bwarea``, corrected areas, or nesting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

FOAMS105_MEASUREMENT_METHOD = "foams_1_0_5_mask_measurement_v1"
FOAMS105_MEASUREMENT_SOURCE_REPOSITORY = "jsoteloflores/FOAMS-1.0.5"
FOAMS105_MEASUREMENT_SOURCE_COMMIT = (
    "179663203f2d0f86b2863d5ea7f8f70dadca02f8"
)
FOAMS105_MEASUREMENT_SOURCE_FILE = "data_meas.m"
FOAMS105_MEASUREMENT_CALIBRATION_SOURCE_FILE = "project.m"
FOAMS105_MEASUREMENT_SCOPE = "mask_topology_area_equivalent_diameter_only"
FOAMS105_BORDER_CONNECTIVITY = 8
FOAMS105_HOLE_BACKGROUND_CONNECTIVITY = 4
FOAMS105_COMPONENT_CONNECTIVITY = 4
FOAMS105_COMPONENT_ORDER_POLICY = "minimum_column_major_flat_index"
FOAMS105_EXCLUSION_SLOT_COUNT = 4

PhaseScalar = Union[bool, int, float]


class Foams105MeasurementValidationError(ValueError):
    """Raised when a legacy mask-measurement input is invalid."""


class Foams105MeasurementNumericalError(ArithmeticError):
    """Raised when calibrated measurement arithmetic is not finite and positive."""


@dataclass(frozen=True)
class Foams105ComponentMeasurement:
    """One source-style connected-component area measurement."""

    component_id: int
    first_pixel_rc: Tuple[int, int]
    area_px: int
    area_mm2: float
    equivalent_diameter_mm: float


@dataclass(frozen=True)
class Foams105PhaseMeasurementResult:
    """Immutable result for one independently selected image phase.

    Array fields are backed by immutable byte buffers. They own their data,
    expose Boolean masks and an int32 label image, and reject ordinary writes.
    """

    image_shape: Tuple[int, int]
    phase_value: PhaseScalar
    scale_px_per_mm: float
    components: Tuple[Foams105ComponentMeasurement, ...]
    removed_border_mask: np.ndarray
    filled_mask: np.ndarray
    label_image: np.ndarray
    selected_pixel_count: int
    removed_border_pixel_count: int
    retained_before_fill_pixel_count: int
    filled_added_pixel_count: int
    final_foreground_pixel_count: int
    method: str = FOAMS105_MEASUREMENT_METHOD
    source_repository: str = FOAMS105_MEASUREMENT_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_MEASUREMENT_SOURCE_COMMIT
    source_file: str = FOAMS105_MEASUREMENT_SOURCE_FILE
    calibration_source_file: str = FOAMS105_MEASUREMENT_CALIBRATION_SOURCE_FILE
    scope: str = FOAMS105_MEASUREMENT_SCOPE
    border_connectivity: int = FOAMS105_BORDER_CONNECTIVITY
    hole_background_connectivity: int = FOAMS105_HOLE_BACKGROUND_CONNECTIVITY
    component_connectivity: int = FOAMS105_COMPONENT_CONNECTIVITY
    component_order_policy: str = FOAMS105_COMPONENT_ORDER_POLICY
    pixel_area_unit: str = "pixels"
    length_unit: str = "mm"
    area_unit: str = "mm^2"
    scale_unit: str = "pixels/mm"


@dataclass(frozen=True)
class Foams105ImageMeasurementResult:
    """Pore result and four stable optional excluded-phase slots."""

    image_shape: Tuple[int, int]
    pore_phase_value: PhaseScalar
    excluded_phase_values: Tuple[
        Optional[PhaseScalar],
        Optional[PhaseScalar],
        Optional[PhaseScalar],
        Optional[PhaseScalar],
    ]
    scale_px_per_mm: float
    pore_result: Foams105PhaseMeasurementResult
    excluded_phase_results: Tuple[
        Optional[Foams105PhaseMeasurementResult],
        Optional[Foams105PhaseMeasurementResult],
        Optional[Foams105PhaseMeasurementResult],
        Optional[Foams105PhaseMeasurementResult],
    ]
    phase_value_collision_slot_pairs: Tuple[Tuple[str, str], ...]
    method: str = FOAMS105_MEASUREMENT_METHOD
    source_repository: str = FOAMS105_MEASUREMENT_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_MEASUREMENT_SOURCE_COMMIT
    source_file: str = FOAMS105_MEASUREMENT_SOURCE_FILE
    calibration_source_file: str = FOAMS105_MEASUREMENT_CALIBRATION_SOURCE_FILE
    scope: str = FOAMS105_MEASUREMENT_SCOPE
    length_unit: str = "mm"
    area_unit: str = "mm^2"
    scale_unit: str = "pixels/mm"
    disabled_exclusion_policy: str = "none_in_exactly_four_positional_slots"


def _validated_image(image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise Foams105MeasurementValidationError("image must be a NumPy array")
    if image.ndim != 2:
        raise Foams105MeasurementValidationError(
            "image must be a nonempty two-dimensional array"
        )
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise Foams105MeasurementValidationError(
            "image must be a nonempty two-dimensional array"
        )
    if image.dtype.kind not in "biuf":
        raise Foams105MeasurementValidationError(
            "image dtype must be Boolean, integer, or real floating"
        )
    if image.dtype.kind == "f" and not bool(np.isfinite(image).all()):
        raise Foams105MeasurementValidationError(
            "image must contain only finite values"
        )
    return image


def _validated_phase_value(
    value: object,
    field: str,
    *,
    boolean_image: bool,
) -> PhaseScalar:
    if isinstance(value, (bool, np.bool_)):
        if not boolean_image:
            raise Foams105MeasurementValidationError(
                f"{field} may be Boolean only for a Boolean image"
            )
        return bool(value)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise Foams105MeasurementValidationError(
            f"{field} must be a finite real scalar"
        )
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as exc:
        raise Foams105MeasurementValidationError(
            f"{field} must be representable as a finite float64"
        ) from exc
    if not math.isfinite(numeric):
        raise Foams105MeasurementValidationError(f"{field} must be finite")
    if isinstance(value, (int, np.integer)):
        return int(value)
    return numeric


def _validated_scale(value: object) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise Foams105MeasurementValidationError(
            "scale_px_per_mm must be a finite positive real scalar"
        )
    try:
        scale = float(value)
    except (OverflowError, ValueError) as exc:
        raise Foams105MeasurementValidationError(
            "scale_px_per_mm must be representable as a finite float64"
        ) from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise Foams105MeasurementValidationError(
            "scale_px_per_mm must be finite and positive"
        )
    return scale


def _immutable_array(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=dtype)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=dtype)
    return immutable.reshape(contiguous.shape)


def _remove_border_components(selected: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    number, labels = cv2.connectedComponents(
        selected.astype(np.uint8),
        connectivity=FOAMS105_BORDER_CONNECTIVITY,
    )
    border_labels = np.unique(
        np.concatenate(
            (
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            )
        )
    )
    keep = np.ones(number, dtype=bool)
    keep[0] = False
    keep[border_labels] = False
    retained = keep[labels]
    return retained, selected & ~retained


def _fill_holes_four_connected(retained: np.ndarray) -> np.ndarray:
    background = ~retained
    number, labels = cv2.connectedComponents(
        background.astype(np.uint8),
        connectivity=FOAMS105_HOLE_BACKGROUND_CONNECTIVITY,
    )
    border_labels = np.unique(
        np.concatenate(
            (
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            )
        )
    )
    exterior_lookup = np.zeros(number, dtype=bool)
    exterior_lookup[border_labels] = True
    exterior_background = background & exterior_lookup[labels]
    holes = background & ~exterior_background
    return retained | holes


def _column_major_labels(filled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    number, labels = cv2.connectedComponents(
        filled.astype(np.uint8),
        connectivity=FOAMS105_COMPONENT_CONNECTIVITY,
    )
    if number <= 1:
        return np.zeros(filled.shape, dtype=np.int32), np.zeros(1, dtype=np.int64)

    column_major_labels = labels.T.ravel()
    positions = np.flatnonzero(column_major_labels)
    positive_labels = column_major_labels[positions]
    minimum_indices = np.full(number, filled.size, dtype=np.int64)
    np.minimum.at(minimum_indices, positive_labels, positions)
    old_ids = np.arange(1, number, dtype=np.int32)
    ordered_old_ids = old_ids[np.argsort(minimum_indices[1:])]
    lookup = np.zeros(number, dtype=np.int32)
    lookup[ordered_old_ids] = np.arange(1, number, dtype=np.int32)
    canonical = lookup[labels]
    canonical_minima = minimum_indices[ordered_old_ids]
    return canonical, np.concatenate((np.zeros(1, dtype=np.int64), canonical_minima))


def _component_measurements(
    labels: np.ndarray,
    column_major_minima: np.ndarray,
    scale: float,
) -> Tuple[Foams105ComponentMeasurement, ...]:
    component_count = len(column_major_minima) - 1
    if component_count == 0:
        return ()
    areas = np.bincount(labels.ravel(), minlength=component_count + 1)
    height = labels.shape[0]
    components = []
    for component_id in range(1, component_count + 1):
        area_px = int(areas[component_id])
        try:
            area_mm2 = (float(area_px) / scale) / scale
            equivalent_diameter_mm = 2.0 * math.sqrt(area_mm2 / math.pi)
        except (OverflowError, ValueError) as exc:
            raise Foams105MeasurementNumericalError(
                f"Calibrated component {component_id} arithmetic failed"
            ) from exc
        if not math.isfinite(area_mm2) or area_mm2 <= 0.0:
            raise Foams105MeasurementNumericalError(
                f"Calibrated area for component {component_id} is not positive and finite"
            )
        if (
            not math.isfinite(equivalent_diameter_mm)
            or equivalent_diameter_mm <= 0.0
        ):
            raise Foams105MeasurementNumericalError(
                f"Equivalent diameter for component {component_id} is not positive and finite"
            )
        flat_index = int(column_major_minima[component_id])
        components.append(
            Foams105ComponentMeasurement(
                component_id=component_id,
                first_pixel_rc=(flat_index % height, flat_index // height),
                area_px=area_px,
                area_mm2=area_mm2,
                equivalent_diameter_mm=equivalent_diameter_mm,
            )
        )
    return tuple(components)


def measure_foams105_phase(
    image: np.ndarray,
    phase_value: PhaseScalar,
    scale_px_per_mm: float,
) -> Foams105PhaseMeasurementResult:
    """Measure one exact image phase through the pinned source topology path."""
    validated_image = _validated_image(image)
    selected_value = _validated_phase_value(
        phase_value,
        "phase_value",
        boolean_image=validated_image.dtype.kind == "b",
    )
    scale = _validated_scale(scale_px_per_mm)

    selected = np.equal(validated_image, selected_value)
    retained, removed = _remove_border_components(selected)
    filled = _fill_holes_four_connected(retained)
    labels, minima = _column_major_labels(filled)
    components = _component_measurements(labels, minima, scale)

    selected_count = int(np.count_nonzero(selected))
    removed_count = int(np.count_nonzero(removed))
    retained_count = int(np.count_nonzero(retained))
    final_count = int(np.count_nonzero(filled))
    filled_added_count = final_count - retained_count
    component_area_sum = sum(component.area_px for component in components)
    if selected_count != removed_count + retained_count:
        raise Foams105MeasurementNumericalError(
            "Selected-pixel border-removal conservation failed"
        )
    if final_count != retained_count + filled_added_count:
        raise Foams105MeasurementNumericalError(
            "Hole-fill foreground conservation failed"
        )
    if final_count != component_area_sum:
        raise Foams105MeasurementNumericalError(
            "Component-area foreground conservation failed"
        )

    return Foams105PhaseMeasurementResult(
        image_shape=(int(validated_image.shape[0]), int(validated_image.shape[1])),
        phase_value=selected_value,
        scale_px_per_mm=scale,
        components=components,
        removed_border_mask=_immutable_array(removed, np.dtype(np.bool_)),
        filled_mask=_immutable_array(filled, np.dtype(np.bool_)),
        label_image=_immutable_array(labels, np.dtype(np.int32)),
        selected_pixel_count=selected_count,
        removed_border_pixel_count=removed_count,
        retained_before_fill_pixel_count=retained_count,
        filled_added_pixel_count=filled_added_count,
        final_foreground_pixel_count=final_count,
    )


def measure_foams105_image(
    image: np.ndarray,
    pore_value: PhaseScalar,
    scale_px_per_mm: float,
    excluded_phase_values: Sequence[Optional[PhaseScalar]] = (
        None,
        None,
        None,
        None,
    ),
) -> Foams105ImageMeasurementResult:
    """Measure a pore phase and exactly four optional independent exclusions."""
    validated_image = _validated_image(image)
    if isinstance(excluded_phase_values, (str, bytes)):
        raise Foams105MeasurementValidationError(
            "excluded_phase_values must contain exactly four positional slots"
        )
    try:
        raw_exclusions = tuple(excluded_phase_values)
    except TypeError as exc:
        raise Foams105MeasurementValidationError(
            "excluded_phase_values must contain exactly four positional slots"
        ) from exc
    if len(raw_exclusions) != FOAMS105_EXCLUSION_SLOT_COUNT:
        raise Foams105MeasurementValidationError(
            "excluded_phase_values must contain exactly four positional slots"
        )

    pore_result = measure_foams105_phase(
        validated_image, pore_value, scale_px_per_mm
    )
    exclusion_results = []
    normalized_exclusions = []
    for index, value in enumerate(raw_exclusions):
        if value is None:
            normalized_exclusions.append(None)
            exclusion_results.append(None)
            continue
        normalized = _validated_phase_value(
            value,
            f"excluded_phase_values[{index}]",
            boolean_image=validated_image.dtype.kind == "b",
        )
        normalized_exclusions.append(normalized)
        exclusion_results.append(
            measure_foams105_phase(validated_image, normalized, scale_px_per_mm)
        )

    named_values = [("pore", pore_result.phase_value)]
    named_values.extend(
        (f"excluded_phase_{index + 1}", value)
        for index, value in enumerate(normalized_exclusions)
        if value is not None
    )
    collisions = tuple(
        (left_name, right_name)
        for left_index, (left_name, left_value) in enumerate(named_values)
        for right_name, right_value in named_values[left_index + 1 :]
        if left_value == right_value
    )

    exclusions_tuple = tuple(normalized_exclusions)
    results_tuple = tuple(exclusion_results)
    return Foams105ImageMeasurementResult(
        image_shape=pore_result.image_shape,
        pore_phase_value=pore_result.phase_value,
        excluded_phase_values=exclusions_tuple,  # type: ignore[arg-type]
        scale_px_per_mm=pore_result.scale_px_per_mm,
        pore_result=pore_result,
        excluded_phase_results=results_tuple,  # type: ignore[arg-type]
        phase_value_collision_slot_pairs=collisions,
    )
