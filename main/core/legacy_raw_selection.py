"""FOAMS 1.0.5 raw-object cutoff selection compatibility replay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .legacy_selection import Foams105BinnedRange
from .legacy_size_classes import (
    FOAMS105_ROUNDING_POLICY,
    Foams105SizeClassNumericalError,
    _round_nonnegative_5_decimals,
)

FOAMS105_RAW_SELECTION_METHOD = "foams_1_0_5_raw_object_selection_replay_v1"
FOAMS105_RAW_SELECTION_SOURCE_REPOSITORY = "jsoteloflores/FOAMS-1.0.5"
FOAMS105_RAW_SELECTION_SOURCE_COMMIT = (
    "179663203f2d0f86b2863d5ea7f8f70dadca02f8"
)
FOAMS105_RAW_SELECTION_SOURCE_FILE = "analysis.m"
FOAMS105_RAW_SELECTION_SCOPE = "raw_object_cutoff_selection_only"
FOAMS105_RAW_SELECTION_PREFIX_POLICY = "active_slots_must_form_prefix_1_through_k"
FOAMS105_RAW_SELECTION_CONCATENATION_POLICY = "ascending_source_slot"
FOAMS105_RAW_SELECTION_BOUNDARY_POLICY = "inclusive_unrounded_numeric_comparison"
FOAMS105_RAW_SELECTION_ROUNDING_PROFILE = FOAMS105_ROUNDING_POLICY
FOAMS105_RAW_SELECTION_MAX_LABELS = 45
FOAMS105_RAW_SELECTION_MAX_GROUPS = 4


class Foams105RawSelectionValidationError(ValueError):
    """Raised when raw-selection inputs violate the bounded contract."""

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


class Foams105RawSelectionDomainError(ValueError):
    """Raised when source predecessor rules are undefined for supplied ranges."""

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


class Foams105RawSelectionNumericalError(ArithmeticError):
    """Raised when rounded lookup arithmetic has a nonfinite intermediate."""

    def __init__(
        self,
        message: str,
        *,
        context: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__(message)
        self.code = "rounding_overflow"
        self.context = MappingProxyType(dict(context or {}))


@dataclass(frozen=True)
class Foams105RawObject:
    """One already-thresholded calibrated component with stable identity."""

    image_id: str
    component_id: int
    equivalent_diameter_mm: float
    area_mm2: float


@dataclass(frozen=True)
class Foams105RawGroupInput:
    """One original magnification slot and its ordered measured objects."""

    group_id: str
    source_slot: int
    objects: Tuple[Foams105RawObject, ...]


@dataclass(frozen=True)
class Foams105RawSelectedRow:
    """One selected object retaining group, source index, and component identity."""

    output_index: int
    group_id: str
    source_slot: int
    source_object_index: int
    image_id: str
    component_id: int
    equivalent_diameter_mm: float
    area_mm2: float


@dataclass(frozen=True)
class Foams105RawSelectionGroupTrace:
    """Complete raw cutoff decision and object partition for one source slot."""

    group_id: str
    source_slot: int
    active: bool
    supplied_range: Optional[Foams105BinnedRange]
    rounded_lower_key: Optional[float]
    matched_global_index: Optional[int]
    matching_global_indices: Tuple[int, ...]
    effective_lower_mm: Optional[float]
    literal_upper_mm: Optional[float]
    lower_policy: str
    selected_indices: Tuple[int, ...]
    rejected_indices: Tuple[int, ...]
    selected_objects: Tuple[Foams105RawObject, ...]
    input_count: int
    selected_count: int


@dataclass(frozen=True)
class Foams105RawSelectionResult:
    """Immutable raw-object selection with aligned projections and provenance."""

    bin_labels_mm: Tuple[float, ...]
    groups: Tuple[Foams105RawGroupInput, ...]
    requested_ranges: Tuple[Optional[Foams105BinnedRange], ...]
    active_slot_count: int
    group_traces: Tuple[Foams105RawSelectionGroupTrace, ...]
    rows: Tuple[Foams105RawSelectedRow, ...]
    selected_objects: Tuple[Foams105RawObject, ...]
    selected_identities: Tuple[Tuple[str, int], ...]
    equivalent_diameters_mm: Tuple[float, ...]
    areas_mm2: Tuple[float, ...]
    selected_count: int
    method: str = FOAMS105_RAW_SELECTION_METHOD
    source_repository: str = FOAMS105_RAW_SELECTION_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_RAW_SELECTION_SOURCE_COMMIT
    source_file: str = FOAMS105_RAW_SELECTION_SOURCE_FILE
    scope: str = FOAMS105_RAW_SELECTION_SCOPE
    supported_prefix_policy: str = FOAMS105_RAW_SELECTION_PREFIX_POLICY
    concatenation_policy: str = FOAMS105_RAW_SELECTION_CONCATENATION_POLICY
    boundary_policy: str = FOAMS105_RAW_SELECTION_BOUNDARY_POLICY
    rounding_profile: str = FOAMS105_RAW_SELECTION_ROUNDING_PROFILE
    length_unit: str = "mm"
    area_unit: str = "mm^2"


def _real_scalar(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise Foams105RawSelectionValidationError(
            f"{field} must be a finite real scalar",
            context={"field": field},
        )
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise Foams105RawSelectionValidationError(
            f"{field} must be representable as finite float64",
            context={"field": field},
        ) from exc
    if not math.isfinite(result):
        raise Foams105RawSelectionValidationError(
            f"{field} must be finite",
            context={"field": field},
        )
    return result


def _validated_labels(values: object) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise Foams105RawSelectionValidationError(
            "bin_labels_mm must be a numeric sequence"
        )
    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105RawSelectionValidationError(
            "bin_labels_mm must be a numeric sequence"
        ) from exc
    labels = tuple(
        _real_scalar(value, f"bin_labels_mm[{index}]")
        for index, value in enumerate(raw_values)
    )
    if not 1 <= len(labels) <= FOAMS105_RAW_SELECTION_MAX_LABELS:
        raise Foams105RawSelectionValidationError(
            f"bin_labels_mm must contain 1 to {FOAMS105_RAW_SELECTION_MAX_LABELS} labels",
            code="unsupported_grid",
            context={"label_count": len(labels)},
        )
    for index, label in enumerate(labels):
        if label <= 0.0:
            raise Foams105RawSelectionValidationError(
                f"bin_labels_mm[{index}] must be positive",
                code="unsupported_grid",
                context={"index": index, "value": label},
            )
        if index and label <= labels[index - 1]:
            raise Foams105RawSelectionValidationError(
                "bin_labels_mm must be strictly increasing",
                code="unsupported_grid",
                context={"index": index, "value": label},
            )
    return labels


def _validated_groups(groups: object) -> Tuple[Foams105RawGroupInput, ...]:
    if isinstance(groups, (str, bytes)):
        raise Foams105RawSelectionValidationError("groups must be a sequence")
    try:
        raw_groups = tuple(groups)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105RawSelectionValidationError("groups must be a sequence") from exc
    if not 1 <= len(raw_groups) <= FOAMS105_RAW_SELECTION_MAX_GROUPS:
        raise Foams105RawSelectionValidationError(
            f"groups must contain 1 to {FOAMS105_RAW_SELECTION_MAX_GROUPS} records",
            context={"group_count": len(raw_groups)},
        )

    normalized = []
    group_ids = set()
    identities = set()
    for group_index, group in enumerate(raw_groups):
        if not isinstance(group, Foams105RawGroupInput):
            raise Foams105RawSelectionValidationError(
                f"groups[{group_index}] must be a Foams105RawGroupInput"
            )
        if not isinstance(group.group_id, str) or not group.group_id.strip():
            raise Foams105RawSelectionValidationError(
                f"groups[{group_index}].group_id must be a nonempty string"
            )
        if group.group_id in group_ids:
            raise Foams105RawSelectionValidationError(
                f"duplicate group_id {group.group_id!r}",
                context={"group_id": group.group_id},
            )
        if (
            not isinstance(group.source_slot, (int, np.integer))
            or isinstance(group.source_slot, (bool, np.bool_))
        ):
            raise Foams105RawSelectionValidationError(
                f"groups[{group_index}].source_slot must be an integer"
            )
        source_slot = int(group.source_slot)
        if source_slot != group_index + 1:
            raise Foams105RawSelectionValidationError(
                "groups must declare contiguous source slots 1..G in ascending order",
                context={
                    "group_id": group.group_id,
                    "source_slot": source_slot,
                    "expected_source_slot": group_index + 1,
                },
            )
        if not isinstance(group.objects, tuple):
            raise Foams105RawSelectionValidationError(
                f"groups[{group_index}].objects must be an immutable tuple"
            )
        objects = []
        for object_index, object_ in enumerate(group.objects):
            field = f"groups[{group_index}].objects[{object_index}]"
            if not isinstance(object_, Foams105RawObject):
                raise Foams105RawSelectionValidationError(
                    f"{field} must be a Foams105RawObject"
                )
            if not isinstance(object_.image_id, str) or not object_.image_id.strip():
                raise Foams105RawSelectionValidationError(
                    f"{field}.image_id must be a nonempty string"
                )
            if (
                not isinstance(object_.component_id, (int, np.integer))
                or isinstance(object_.component_id, (bool, np.bool_))
                or int(object_.component_id) <= 0
            ):
                raise Foams105RawSelectionValidationError(
                    f"{field}.component_id must be a positive integer"
                )
            component_id = int(object_.component_id)
            diameter = _real_scalar(
                object_.equivalent_diameter_mm,
                f"{field}.equivalent_diameter_mm",
            )
            area = _real_scalar(object_.area_mm2, f"{field}.area_mm2")
            if diameter <= 0.0 or area <= 0.0:
                raise Foams105RawSelectionValidationError(
                    f"{field} diameter and area must be positive",
                    context={"group_id": group.group_id, "object_index": object_index},
                )
            identity = (object_.image_id, component_id)
            if identity in identities:
                raise Foams105RawSelectionValidationError(
                    f"duplicate object identity {identity!r}",
                    code="duplicate_object_identity",
                    context={"identity": identity},
                )
            identities.add(identity)
            objects.append(
                Foams105RawObject(object_.image_id, component_id, diameter, area)
            )
        normalized.append(
            Foams105RawGroupInput(group.group_id, source_slot, tuple(objects))
        )
        group_ids.add(group.group_id)
    return tuple(normalized)


def _validated_ranges(
    ranges: object,
    groups: Tuple[Foams105RawGroupInput, ...],
) -> Tuple[Optional[Foams105BinnedRange], ...]:
    if isinstance(ranges, (str, bytes)):
        raise Foams105RawSelectionValidationError("ranges must be a sequence")
    try:
        raw_ranges = tuple(ranges)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105RawSelectionValidationError("ranges must be a sequence") from exc
    if len(raw_ranges) != len(groups):
        raise Foams105RawSelectionValidationError(
            "ranges must contain exactly one positional entry per declared group",
            context={"range_count": len(raw_ranges), "group_count": len(groups)},
        )

    normalized = []
    for index, (group, range_) in enumerate(zip(groups, raw_ranges)):
        if range_ is None:
            if index == 0:
                raise Foams105RawSelectionValidationError(
                    "source slot 1 must be active",
                    context={"group_id": group.group_id, "source_slot": 1},
                )
            normalized.append(None)
            continue
        if not isinstance(range_, Foams105BinnedRange):
            raise Foams105RawSelectionValidationError(
                f"ranges[{index}] must be a Foams105BinnedRange or None"
            )
        if range_.group_id != group.group_id:
            raise Foams105RawSelectionValidationError(
                f"ranges[{index}] must correspond to group {group.group_id!r}",
                context={
                    "expected_group_id": group.group_id,
                    "actual_group_id": range_.group_id,
                },
            )
        lower = _real_scalar(range_.lower_label_mm, f"ranges[{index}].lower_label_mm")
        upper = _real_scalar(range_.upper_label_mm, f"ranges[{index}].upper_label_mm")
        if lower <= 0.0 or upper <= 0.0:
            raise Foams105RawSelectionValidationError(
                f"ranges[{index}] bounds must be positive"
            )
        if lower > upper:
            raise Foams105RawSelectionValidationError(
                f"ranges[{index}] lower bound must not exceed upper bound"
            )
        normalized.append(Foams105BinnedRange(group.group_id, lower, upper))

    active_flags = tuple(range_ is not None for range_ in normalized)
    active_count = sum(active_flags)
    if active_flags != (True,) * active_count + (False,) * (len(groups) - active_count):
        first_gap = active_flags.index(False)
        later_active = tuple(
            index + 1
            for index in range(first_gap + 1, len(active_flags))
            if active_flags[index]
        )
        raise Foams105RawSelectionDomainError(
            "noncontiguous_active_slots",
            "Enabled raw-selection slots must form a prefix from slot 1",
            context={
                "first_disabled_slot": first_gap + 1,
                "later_active_slots": later_active,
            },
        )
    return tuple(normalized)


def _rounded_value(value: float, field: str, index: int) -> float:
    try:
        return _round_nonnegative_5_decimals(value, index)
    except Foams105SizeClassNumericalError as exc:
        raise Foams105RawSelectionNumericalError(
            f"{field} cannot be rounded to five decimals",
            context={"field": field, "index": index, "value": value},
        ) from exc


def _rounded_lookup(labels: Tuple[float, ...]) -> Mapping[float, Tuple[int, ...]]:
    mutable: dict[float, list[int]] = {}
    for index, label in enumerate(labels):
        key = _rounded_value(label, f"bin_labels_mm[{index}]", index)
        mutable.setdefault(key, []).append(index)
    return MappingProxyType(
        {key: tuple(indices) for key, indices in mutable.items()}
    )


def _active_bound(
    labels: Tuple[float, ...],
    lookup: Mapping[float, Tuple[int, ...]],
    group: Foams105RawGroupInput,
    range_: Foams105BinnedRange,
    active_slot_count: int,
) -> Tuple[Optional[float], Optional[int], Tuple[int, ...], float, str]:
    if group.source_slot == 4:
        return None, None, (), 0.0, "zero_for_slot4"

    rounded_lower = _rounded_value(
        range_.lower_label_mm,
        f"ranges[{group.source_slot - 1}].lower_label_mm",
        group.source_slot - 1,
    )
    matching = lookup.get(rounded_lower, ())
    context = {
        "group_id": group.group_id,
        "source_slot": group.source_slot,
        "supplied_lower_mm": range_.lower_label_mm,
        "rounded_lower_key": rounded_lower,
        "matching_global_indices": matching,
    }
    if not matching:
        raise Foams105RawSelectionDomainError(
            "lower_label_not_found",
            f"No rounded global label matches group {group.group_id!r} lower bound",
            context=context,
        )
    if len(matching) > 1:
        raise Foams105RawSelectionDomainError(
            "ambiguous_lower_label",
            f"Multiple rounded global labels match group {group.group_id!r} lower bound",
            context=context,
        )
    matched_index = matching[0]
    if group.source_slot < active_slot_count:
        if matched_index == 0:
            raise Foams105RawSelectionDomainError(
                "missing_predecessor_label",
                f"Group {group.group_id!r} matched the first label and has no predecessor",
                context=context,
            )
        return (
            rounded_lower,
            matched_index,
            matching,
            labels[matched_index - 1],
            "predecessor_label",
        )
    return (
        rounded_lower,
        matched_index,
        matching,
        labels[matched_index],
        "own_label",
    )


def select_foams105_raw_objects(
    bin_labels_mm: Sequence[float],
    groups: Sequence[Foams105RawGroupInput],
    ranges: Sequence[Optional[Foams105BinnedRange]],
) -> Foams105RawSelectionResult:
    """Apply original source-slot raw cutoffs without changing object identity."""
    labels = _validated_labels(bin_labels_mm)
    normalized_groups = _validated_groups(groups)
    normalized_ranges = _validated_ranges(ranges, normalized_groups)
    active_slot_count = sum(range_ is not None for range_ in normalized_ranges)
    lookup = _rounded_lookup(labels)

    traces = []
    rows = []
    for group, range_ in zip(normalized_groups, normalized_ranges):
        if range_ is None:
            traces.append(
                Foams105RawSelectionGroupTrace(
                    group_id=group.group_id,
                    source_slot=group.source_slot,
                    active=False,
                    supplied_range=None,
                    rounded_lower_key=None,
                    matched_global_index=None,
                    matching_global_indices=(),
                    effective_lower_mm=None,
                    literal_upper_mm=None,
                    lower_policy="disabled",
                    selected_indices=(),
                    rejected_indices=tuple(range(len(group.objects))),
                    selected_objects=(),
                    input_count=len(group.objects),
                    selected_count=0,
                )
            )
            continue

        rounded, matched, matching, effective_lower, policy = _active_bound(
            labels, lookup, group, range_, active_slot_count
        )
        selected_indices = tuple(
            index
            for index, object_ in enumerate(group.objects)
            if effective_lower
            <= object_.equivalent_diameter_mm
            <= range_.upper_label_mm
        )
        selected_set = set(selected_indices)
        rejected_indices = tuple(
            index for index in range(len(group.objects)) if index not in selected_set
        )
        selected_objects = tuple(group.objects[index] for index in selected_indices)
        traces.append(
            Foams105RawSelectionGroupTrace(
                group_id=group.group_id,
                source_slot=group.source_slot,
                active=True,
                supplied_range=range_,
                rounded_lower_key=rounded,
                matched_global_index=matched,
                matching_global_indices=matching,
                effective_lower_mm=effective_lower,
                literal_upper_mm=range_.upper_label_mm,
                lower_policy=policy,
                selected_indices=selected_indices,
                rejected_indices=rejected_indices,
                selected_objects=selected_objects,
                input_count=len(group.objects),
                selected_count=len(selected_indices),
            )
        )
        for source_index in selected_indices:
            object_ = group.objects[source_index]
            rows.append(
                Foams105RawSelectedRow(
                    output_index=len(rows),
                    group_id=group.group_id,
                    source_slot=group.source_slot,
                    source_object_index=source_index,
                    image_id=object_.image_id,
                    component_id=object_.component_id,
                    equivalent_diameter_mm=object_.equivalent_diameter_mm,
                    area_mm2=object_.area_mm2,
                )
            )

    row_tuple = tuple(rows)
    selected_objects = tuple(
        Foams105RawObject(
            row.image_id,
            row.component_id,
            row.equivalent_diameter_mm,
            row.area_mm2,
        )
        for row in row_tuple
    )
    return Foams105RawSelectionResult(
        bin_labels_mm=labels,
        groups=normalized_groups,
        requested_ranges=normalized_ranges,
        active_slot_count=active_slot_count,
        group_traces=tuple(traces),
        rows=row_tuple,
        selected_objects=selected_objects,
        selected_identities=tuple(
            (row.image_id, row.component_id) for row in row_tuple
        ),
        equivalent_diameters_mm=tuple(
            row.equivalent_diameter_mm for row in row_tuple
        ),
        areas_mm2=tuple(row.area_mm2 for row in row_tuple),
        selected_count=len(row_tuple),
    )
