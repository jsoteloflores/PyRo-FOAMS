"""Checked FOAMS 1.0.5 binned range selection and concatenation.

This module reproduces the supported aligned subset of the binned selection
stage in ``analysis.m``. It does not select raw objects or run conversion.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .legacy_cutoffs import (
    Foams105CutoffDomainError,
    Foams105CutoffSuggestionResult,
    Foams105CutoffValidationError,
    suggest_foams105_cutoffs,
)

FOAMS105_SELECTION_METHOD = "foams_1_0_5_binned_selection_checked_replay_v1"
FOAMS105_SELECTION_SOURCE_REPOSITORY = "jsoteloflores/FOAMS-1.0.5"
FOAMS105_SELECTION_SOURCE_COMMIT = "179663203f2d0f86b2863d5ea7f8f70dadca02f8"
FOAMS105_SELECTION_SOURCE_FILE = "analysis.m"
FOAMS105_SELECTION_SCOPE = "binned_range_selection_only"
FOAMS105_SELECTION_SOURCE_SLICE_POLICY = (
    "slot1_label_span;slot2_3_first_to_last_nonzero_masked;"
    "slot4_global_zero_to_last_nonzero_masked"
)
FOAMS105_SELECTION_MISALIGNMENT_POLICY = "reject_without_repair"
FOAMS105_SELECTION_CONCATENATION_POLICY = "descending_source_slot"
FOAMS105_SELECTION_BOUNDARY_POLICY = "inclusive_numeric_label_comparison"
FOAMS105_SELECTION_MANUAL_ORIGIN = "explicit_manual"
FOAMS105_SELECTION_AUTOSMART_ORIGIN = "autosmart_suggestion"
FOAMS105_SELECTION_MAX_LABELS = 45
FOAMS105_SELECTION_MAX_GROUPS = 4
FOAMS105_SELECTION_CONVERTER_MAX_ROWS = 45


class Foams105SelectionValidationError(ValueError):
    """Raised when binned-selection inputs violate the bounded contract."""

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


class Foams105SelectionDomainError(ValueError):
    """Raised when source slicing cannot produce aligned selected rows."""

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
class Foams105BinnedGroupInput:
    """One original source slot on the complete shared label grid."""

    group_id: str
    source_slot: int
    na_per_mm2: Tuple[float, ...]


@dataclass(frozen=True)
class Foams105BinnedRange:
    """Literal inclusive label bounds for one enabled group."""

    group_id: str
    lower_label_mm: float
    upper_label_mm: float


@dataclass(frozen=True)
class Foams105SelectedRow:
    """One selected row with retained original slot and global-grid identity."""

    output_index: int
    source_slot: int
    group_id: str
    global_bin_index: int
    label_mm: float
    na_per_mm2: float


@dataclass(frozen=True)
class Foams105SelectionGroupTrace:
    """Requested bounds and source index traces for one declared group."""

    group_id: str
    source_slot: int
    enabled: bool
    requested_lower_label_mm: Optional[float]
    requested_upper_label_mm: Optional[float]
    selected_label_indices: Tuple[int, ...]
    density_source_indices: Tuple[int, ...]
    source_slice_policy: str


@dataclass(frozen=True)
class Foams105BinnedSelectionResult:
    """Immutable checked selection with source identities and diagnostics."""

    input_bin_labels_mm: Tuple[float, ...]
    groups: Tuple[Foams105BinnedGroupInput, ...]
    requested_ranges: Tuple[Optional[Foams105BinnedRange], ...]
    group_traces: Tuple[Foams105SelectionGroupTrace, ...]
    rows: Tuple[Foams105SelectedRow, ...]
    bin_labels_mm: Tuple[float, ...]
    na_per_mm2: Tuple[float, ...]
    disabled_group_ids: Tuple[str, ...]
    adjacent_duplicate_output_index_pairs: Tuple[Tuple[int, int], ...]
    uncovered_global_indices: Tuple[int, ...]
    overlapping_global_indices: Tuple[int, ...]
    output_row_count: int
    converter_length_supported: bool
    range_origin: str
    suggestion_method: Optional[str]
    suggestion_source_repository: Optional[str]
    suggestion_source_commit: Optional[str]
    suggestion_source_file: Optional[str]
    method: str = FOAMS105_SELECTION_METHOD
    source_repository: str = FOAMS105_SELECTION_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_SELECTION_SOURCE_COMMIT
    source_file: str = FOAMS105_SELECTION_SOURCE_FILE
    scope: str = FOAMS105_SELECTION_SCOPE
    source_slice_policy: str = FOAMS105_SELECTION_SOURCE_SLICE_POLICY
    misalignment_policy: str = FOAMS105_SELECTION_MISALIGNMENT_POLICY
    concatenation_policy: str = FOAMS105_SELECTION_CONCATENATION_POLICY
    boundary_policy: str = FOAMS105_SELECTION_BOUNDARY_POLICY
    length_unit: str = "mm"
    density_unit: str = "mm^-2"


def _real_scalar(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise Foams105SelectionValidationError(
            f"{field} must be a finite real scalar",
            context={"field": field},
        )
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise Foams105SelectionValidationError(
            f"{field} must be representable as finite float64",
            context={"field": field},
        ) from exc
    if not math.isfinite(result):
        raise Foams105SelectionValidationError(
            f"{field} must be finite",
            context={"field": field},
        )
    return result


def _numeric_tuple(values: object, field: str) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise Foams105SelectionValidationError(
            f"{field} must be a numeric sequence",
            context={"field": field},
        )
    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105SelectionValidationError(
            f"{field} must be a numeric sequence",
            context={"field": field},
        ) from exc
    return tuple(
        _real_scalar(value, f"{field}[{index}]")
        for index, value in enumerate(raw_values)
    )


def _validated_labels(values: object) -> Tuple[float, ...]:
    labels = _numeric_tuple(values, "bin_labels_mm")
    if not 1 <= len(labels) <= FOAMS105_SELECTION_MAX_LABELS:
        raise Foams105SelectionValidationError(
            f"bin_labels_mm must contain 1 to {FOAMS105_SELECTION_MAX_LABELS} labels",
            code="unsupported_grid",
            context={"label_count": len(labels)},
        )
    for index, label in enumerate(labels):
        if label <= 0.0:
            raise Foams105SelectionValidationError(
                f"bin_labels_mm[{index}] must be positive",
                code="unsupported_grid",
                context={"index": index, "value": label},
            )
        if index and label <= labels[index - 1]:
            relation = "duplicate" if label == labels[index - 1] else "descending"
            raise Foams105SelectionValidationError(
                "bin_labels_mm must be strictly increasing",
                code="unsupported_grid",
                context={"index": index, "relation": relation, "value": label},
            )
    return labels


def _validated_groups(
    groups: object, label_count: int
) -> Tuple[Foams105BinnedGroupInput, ...]:
    if isinstance(groups, (str, bytes)):
        raise Foams105SelectionValidationError("groups must be a sequence")
    try:
        raw_groups = tuple(groups)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105SelectionValidationError("groups must be a sequence") from exc
    if not 1 <= len(raw_groups) <= FOAMS105_SELECTION_MAX_GROUPS:
        raise Foams105SelectionValidationError(
            f"groups must contain 1 to {FOAMS105_SELECTION_MAX_GROUPS} records",
            context={"group_count": len(raw_groups)},
        )

    normalized = []
    seen_ids = set()
    for index, group in enumerate(raw_groups):
        if not isinstance(group, Foams105BinnedGroupInput):
            raise Foams105SelectionValidationError(
                f"groups[{index}] must be a Foams105BinnedGroupInput",
                context={"group_index": index},
            )
        if not isinstance(group.group_id, str) or not group.group_id.strip():
            raise Foams105SelectionValidationError(
                f"groups[{index}].group_id must be a nonempty string",
                context={"group_index": index},
            )
        if group.group_id in seen_ids:
            raise Foams105SelectionValidationError(
                f"duplicate group_id {group.group_id!r}",
                context={"group_id": group.group_id},
            )
        if (
            not isinstance(group.source_slot, (int, np.integer))
            or isinstance(group.source_slot, (bool, np.bool_))
        ):
            raise Foams105SelectionValidationError(
                f"groups[{index}].source_slot must be an integer",
                context={"group_id": group.group_id},
            )
        source_slot = int(group.source_slot)
        expected_slot = index + 1
        if source_slot != expected_slot:
            raise Foams105SelectionValidationError(
                "groups must declare contiguous source slots 1..G in ascending order",
                context={
                    "group_id": group.group_id,
                    "source_slot": source_slot,
                    "expected_source_slot": expected_slot,
                },
            )
        if not isinstance(group.na_per_mm2, tuple):
            raise Foams105SelectionValidationError(
                f"groups[{index}].na_per_mm2 must be an immutable tuple",
                context={"group_id": group.group_id},
            )
        densities = _numeric_tuple(group.na_per_mm2, f"groups[{index}].na_per_mm2")
        if len(densities) != label_count:
            raise Foams105SelectionValidationError(
                f"groups[{index}].na_per_mm2 length must match bin_labels_mm",
                context={
                    "group_id": group.group_id,
                    "density_count": len(densities),
                    "label_count": label_count,
                },
            )
        for density_index, density in enumerate(densities):
            if density < 0.0:
                raise Foams105SelectionValidationError(
                    f"groups[{index}].na_per_mm2[{density_index}] must be nonnegative",
                    context={
                        "group_id": group.group_id,
                        "density_index": density_index,
                        "value": density,
                    },
                )
        normalized.append(
            Foams105BinnedGroupInput(group.group_id, source_slot, densities)
        )
        seen_ids.add(group.group_id)
    return tuple(normalized)


def _validated_ranges(
    ranges: object,
    groups: Tuple[Foams105BinnedGroupInput, ...],
) -> Tuple[Optional[Foams105BinnedRange], ...]:
    if isinstance(ranges, (str, bytes)):
        raise Foams105SelectionValidationError("ranges must be a sequence")
    try:
        raw_ranges = tuple(ranges)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105SelectionValidationError("ranges must be a sequence") from exc
    if len(raw_ranges) != len(groups):
        raise Foams105SelectionValidationError(
            "ranges must contain exactly one positional entry per declared group",
            context={"range_count": len(raw_ranges), "group_count": len(groups)},
        )

    normalized = []
    seen_active_ids = set()
    declared_ids = {group.group_id for group in groups}
    for index, (group, range_) in enumerate(zip(groups, raw_ranges)):
        if range_ is None:
            if group.source_slot == 1:
                raise Foams105SelectionValidationError(
                    "source slot 1 must be enabled",
                    context={"group_id": group.group_id, "source_slot": 1},
                )
            normalized.append(None)
            continue
        if not isinstance(range_, Foams105BinnedRange):
            raise Foams105SelectionValidationError(
                f"ranges[{index}] must be a Foams105BinnedRange or None",
                context={"range_index": index},
            )
        if not isinstance(range_.group_id, str) or not range_.group_id.strip():
            raise Foams105SelectionValidationError(
                f"ranges[{index}].group_id must be a nonempty string",
                context={"range_index": index},
            )
        if range_.group_id not in declared_ids:
            raise Foams105SelectionValidationError(
                f"ranges[{index}] references unknown group_id {range_.group_id!r}",
                context={"group_id": range_.group_id},
            )
        if range_.group_id in seen_active_ids:
            raise Foams105SelectionValidationError(
                f"duplicate active range for group_id {range_.group_id!r}",
                context={"group_id": range_.group_id},
            )
        if range_.group_id != group.group_id:
            raise Foams105SelectionValidationError(
                f"ranges[{index}] must correspond to group {group.group_id!r}",
                context={
                    "range_index": index,
                    "expected_group_id": group.group_id,
                    "actual_group_id": range_.group_id,
                },
            )
        lower = _real_scalar(range_.lower_label_mm, f"ranges[{index}].lower_label_mm")
        upper = _real_scalar(range_.upper_label_mm, f"ranges[{index}].upper_label_mm")
        if lower <= 0.0 or upper <= 0.0:
            raise Foams105SelectionValidationError(
                f"ranges[{index}] bounds must be positive",
                context={"group_id": group.group_id, "lower": lower, "upper": upper},
            )
        if lower > upper:
            raise Foams105SelectionValidationError(
                f"ranges[{index}] lower bound must not exceed upper bound",
                context={"group_id": group.group_id, "lower": lower, "upper": upper},
            )
        normalized.append(Foams105BinnedRange(group.group_id, lower, upper))
        seen_active_ids.add(group.group_id)
    return tuple(normalized)


def _slot_policy(source_slot: int) -> str:
    if source_slot == 1:
        return "slot1_label_index_span"
    if source_slot in (2, 3):
        return "slot2_3_first_to_last_nonzero_masked"
    return "slot4_global_zero_to_last_nonzero_masked"


def _trace_group(
    labels: Tuple[float, ...],
    group: Foams105BinnedGroupInput,
    range_: Optional[Foams105BinnedRange],
) -> Foams105SelectionGroupTrace:
    policy = _slot_policy(group.source_slot)
    if range_ is None:
        return Foams105SelectionGroupTrace(
            group_id=group.group_id,
            source_slot=group.source_slot,
            enabled=False,
            requested_lower_label_mm=None,
            requested_upper_label_mm=None,
            selected_label_indices=(),
            density_source_indices=(),
            source_slice_policy=policy,
        )

    selected = tuple(
        index
        for index, label in enumerate(labels)
        if range_.lower_label_mm <= label <= range_.upper_label_mm
    )
    context = {
        "group_id": group.group_id,
        "source_slot": group.source_slot,
        "lower_label_mm": range_.lower_label_mm,
        "upper_label_mm": range_.upper_label_mm,
    }
    if not selected:
        raise Foams105SelectionDomainError(
            "empty_selected_range",
            f"Range for group {group.group_id!r} selects no labels",
            context=context,
        )

    if group.source_slot == 1:
        density_indices = selected
    else:
        positive = tuple(
            index for index in selected if group.na_per_mm2[index] > 0.0
        )
        if not positive:
            raise Foams105SelectionDomainError(
                "empty_selected_density_support",
                f"Selected range for group {group.group_id!r} has no positive density",
                context={**context, "selected_label_indices": selected},
            )
        if group.source_slot in (2, 3):
            density_indices = tuple(range(positive[0], positive[-1] + 1))
        else:
            density_indices = tuple(range(0, positive[-1] + 1))

    if density_indices != selected:
        raise Foams105SelectionDomainError(
            "source_slice_mismatch",
            f"Source label and density slices differ for group {group.group_id!r}",
            context={
                **context,
                "selected_label_indices": selected,
                "density_source_indices": density_indices,
                "source_slice_policy": policy,
            },
        )
    return Foams105SelectionGroupTrace(
        group_id=group.group_id,
        source_slot=group.source_slot,
        enabled=True,
        requested_lower_label_mm=range_.lower_label_mm,
        requested_upper_label_mm=range_.upper_label_mm,
        selected_label_indices=selected,
        density_source_indices=density_indices,
        source_slice_policy=policy,
    )


def _build_result(
    labels: Tuple[float, ...],
    groups: Tuple[Foams105BinnedGroupInput, ...],
    ranges: Tuple[Optional[Foams105BinnedRange], ...],
    *,
    range_origin: str,
    suggestion_method: Optional[str] = None,
    suggestion_source_repository: Optional[str] = None,
    suggestion_source_commit: Optional[str] = None,
    suggestion_source_file: Optional[str] = None,
) -> Foams105BinnedSelectionResult:
    traces = tuple(
        _trace_group(labels, group, range_)
        for group, range_ in zip(groups, ranges)
    )
    groups_by_id = {group.group_id: group for group in groups}
    rows = []
    for trace in reversed(traces):
        if not trace.enabled:
            continue
        group = groups_by_id[trace.group_id]
        for global_index in trace.selected_label_indices:
            rows.append(
                Foams105SelectedRow(
                    output_index=len(rows),
                    source_slot=trace.source_slot,
                    group_id=trace.group_id,
                    global_bin_index=global_index,
                    label_mm=labels[global_index],
                    na_per_mm2=group.na_per_mm2[global_index],
                )
            )

    for left, right in zip(rows, rows[1:]):
        if right.label_mm < left.label_mm:
            raise Foams105SelectionDomainError(
                "concatenation_not_nondecreasing",
                "Descending labels occur at a source-group concatenation boundary",
                context={
                    "left_row_identity": (
                        left.output_index,
                        left.source_slot,
                        left.group_id,
                        left.global_bin_index,
                    ),
                    "right_row_identity": (
                        right.output_index,
                        right.source_slot,
                        right.group_id,
                        right.global_bin_index,
                    ),
                    "left_label_mm": left.label_mm,
                    "right_label_mm": right.label_mm,
                },
            )

    output_labels = tuple(row.label_mm for row in rows)
    output_densities = tuple(row.na_per_mm2 for row in rows)
    duplicate_pairs = tuple(
        (index - 1, index)
        for index in range(1, len(output_labels))
        if output_labels[index - 1] == output_labels[index]
    )
    selected_global_indices = tuple(row.global_bin_index for row in rows)
    counts = Counter(selected_global_indices)
    if selected_global_indices:
        selected_set = set(selected_global_indices)
        uncovered = tuple(
            index
            for index in range(
                min(selected_global_indices), max(selected_global_indices) + 1
            )
            if index not in selected_set
        )
    else:
        uncovered = ()
    overlapping = tuple(sorted(index for index, count in counts.items() if count > 1))
    row_count = len(rows)
    return Foams105BinnedSelectionResult(
        input_bin_labels_mm=labels,
        groups=groups,
        requested_ranges=ranges,
        group_traces=traces,
        rows=tuple(rows),
        bin_labels_mm=output_labels,
        na_per_mm2=output_densities,
        disabled_group_ids=tuple(
            trace.group_id for trace in traces if not trace.enabled
        ),
        adjacent_duplicate_output_index_pairs=duplicate_pairs,
        uncovered_global_indices=uncovered,
        overlapping_global_indices=overlapping,
        output_row_count=row_count,
        converter_length_supported=(
            1 <= row_count <= FOAMS105_SELECTION_CONVERTER_MAX_ROWS
        ),
        range_origin=range_origin,
        suggestion_method=suggestion_method,
        suggestion_source_repository=suggestion_source_repository,
        suggestion_source_commit=suggestion_source_commit,
        suggestion_source_file=suggestion_source_file,
    )


def select_foams105_binned_ranges(
    bin_labels_mm: Sequence[float],
    groups: Sequence[Foams105BinnedGroupInput],
    ranges: Sequence[Optional[Foams105BinnedRange]],
) -> Foams105BinnedSelectionResult:
    """Select and concatenate explicit inclusive source-slot ranges."""
    labels = _validated_labels(bin_labels_mm)
    normalized_groups = _validated_groups(groups, len(labels))
    normalized_ranges = _validated_ranges(ranges, normalized_groups)
    return _build_result(
        labels,
        normalized_groups,
        normalized_ranges,
        range_origin=FOAMS105_SELECTION_MANUAL_ORIGIN,
    )


def _validated_suggestion(
    suggestion_result: object,
) -> Foams105CutoffSuggestionResult:
    if not isinstance(suggestion_result, Foams105CutoffSuggestionResult):
        raise Foams105SelectionValidationError(
            "suggestion_result must be a Foams105CutoffSuggestionResult"
        )
    if not isinstance(suggestion_result.bin_labels_mm, tuple):
        raise Foams105SelectionValidationError(
            "suggestion_result bin_labels_mm must be an immutable tuple"
        )
    if not isinstance(suggestion_result.groups, tuple):
        raise Foams105SelectionValidationError(
            "suggestion_result groups must be an immutable tuple"
        )
    if not isinstance(suggestion_result.suggested_ranges, tuple):
        raise Foams105SelectionValidationError(
            "suggestion_result suggested_ranges must be an immutable tuple"
        )
    try:
        reconstructed = suggest_foams105_cutoffs(
            suggestion_result.bin_labels_mm, suggestion_result.groups
        )
    except (Foams105CutoffValidationError, Foams105CutoffDomainError) as exc:
        raise Foams105SelectionValidationError(
            "suggestion_result cannot be reconstructed from its labels and groups",
            code="inconsistent_suggestion",
            context={"cutoff_error_code": exc.code},
        ) from exc
    if reconstructed != suggestion_result:
        raise Foams105SelectionValidationError(
            "suggestion_result is inconsistent with reconstructed AutoSmart output",
            code="inconsistent_suggestion",
        )
    return suggestion_result


def select_foams105_suggested_ranges(
    suggestion_result: Foams105CutoffSuggestionResult,
) -> Foams105BinnedSelectionResult:
    """Validate and apply accepted AutoSmart ranges through the manual selector."""
    suggestion = _validated_suggestion(suggestion_result)
    groups = tuple(
        Foams105BinnedGroupInput(
            group_id=group.group_id,
            source_slot=index + 1,
            na_per_mm2=group.na_per_mm2,
        )
        for index, group in enumerate(suggestion.groups)
    )
    ranges = tuple(
        Foams105BinnedRange(
            group_id=range_.group_id,
            lower_label_mm=range_.lower_label_mm,
            upper_label_mm=range_.upper_label_mm,
        )
        for range_ in suggestion.suggested_ranges
    )
    labels = _validated_labels(suggestion.bin_labels_mm)
    normalized_groups = _validated_groups(groups, len(labels))
    normalized_ranges = _validated_ranges(ranges, normalized_groups)
    return _build_result(
        labels,
        normalized_groups,
        normalized_ranges,
        range_origin=FOAMS105_SELECTION_AUTOSMART_ORIGIN,
        suggestion_method=suggestion.method,
        suggestion_source_repository=suggestion.source_repository,
        suggestion_source_commit=suggestion.source_commit,
        suggestion_source_file=suggestion.source_file,
    )
