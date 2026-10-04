"""FOAMS 1.0.5 AutoSmart cutoff suggestions.

This module reproduces the bounded transition and range suggestion behavior
from ``defaultbinrange.m``. It does not apply ranges or concatenate data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

FOAMS105_CUTOFF_METHOD = "foams_1_0_5_autosmart_suggestions_v1"
FOAMS105_CUTOFF_SOURCE_REPOSITORY = "jsoteloflores/FOAMS-1.0.5"
FOAMS105_CUTOFF_SOURCE_COMMIT = "179663203f2d0f86b2863d5ea7f8f70dadca02f8"
FOAMS105_CUTOFF_SOURCE_FILE = "defaultbinrange.m"
FOAMS105_CUTOFF_SCOPE = "cutoff_suggestions_only"
FOAMS105_CUTOFF_ZERO_DIFFERENCE_POLICY = "excluded_from_transition_candidates"
FOAMS105_CUTOFF_SIGN_TIE_POLICY = "equal_magnitude_selects_negative"
FOAMS105_CUTOFF_OCCURRENCE_TIE_POLICY = "last_exact_occurrence"
FOAMS105_CUTOFF_GROUP_ORDER_POLICY = "strictly_ascending_scale_coarsest_to_finest"
FOAMS105_CUTOFF_RANGE_POLICY = "zero_based_inclusive"
FOAMS105_CUTOFF_STATUS = "default_suggestions_only_manual_ranges_remain_allowed"
FOAMS105_CUTOFF_MAX_LABELS = 45
FOAMS105_CUTOFF_MAX_GROUPS = 4


class Foams105CutoffValidationError(ValueError):
    """Raised when cutoff-suggestion inputs violate the bounded contract."""

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


class Foams105CutoffDomainError(ValueError):
    """Raised when source-defined cutoff suggestions cannot be constructed."""

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
class Foams105CutoffGroupInput:
    """One ordered magnification group's complete planar-density vector."""

    group_id: str
    scale_px_per_mm: float
    na_per_mm2: Tuple[float, ...]


@dataclass(frozen=True)
class Foams105CutoffTransition:
    """Transparent signed-log transition trace for one adjacent group pair."""

    coarser_group_id: str
    finer_group_id: str
    overlap_indices: Tuple[int, ...]
    signed_log_differences: Tuple[float, ...]
    smallest_positive_difference: Optional[float]
    largest_negative_difference: Optional[float]
    selected_signed_difference: float
    tied_indices: Tuple[int, ...]
    selected_index: int


@dataclass(frozen=True)
class Foams105SuggestedRange:
    """One source-style inclusive label range for a magnification group."""

    group_id: str
    scale_px_per_mm: float
    lower_index: int
    upper_index: int
    lower_label_mm: float
    upper_label_mm: float


@dataclass(frozen=True)
class Foams105CutoffSuggestionResult:
    """Complete immutable AutoSmart suggestion and transition provenance."""

    bin_labels_mm: Tuple[float, ...]
    groups: Tuple[Foams105CutoffGroupInput, ...]
    transitions: Tuple[Foams105CutoffTransition, ...]
    suggested_ranges: Tuple[Foams105SuggestedRange, ...]
    method: str = FOAMS105_CUTOFF_METHOD
    source_repository: str = FOAMS105_CUTOFF_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_CUTOFF_SOURCE_COMMIT
    source_file: str = FOAMS105_CUTOFF_SOURCE_FILE
    scope: str = FOAMS105_CUTOFF_SCOPE
    zero_difference_policy: str = FOAMS105_CUTOFF_ZERO_DIFFERENCE_POLICY
    sign_tie_policy: str = FOAMS105_CUTOFF_SIGN_TIE_POLICY
    occurrence_tie_policy: str = FOAMS105_CUTOFF_OCCURRENCE_TIE_POLICY
    group_order_policy: str = FOAMS105_CUTOFF_GROUP_ORDER_POLICY
    range_policy: str = FOAMS105_CUTOFF_RANGE_POLICY
    suggestion_status: str = FOAMS105_CUTOFF_STATUS
    length_unit: str = "mm"
    scale_unit: str = "pixels/mm"
    density_unit: str = "mm^-2"


def _real_scalar(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise Foams105CutoffValidationError(
            f"{field} must be a finite real scalar",
            context={"field": field},
        )
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise Foams105CutoffValidationError(
            f"{field} must be representable as finite float64",
            context={"field": field},
        ) from exc
    if not math.isfinite(result):
        raise Foams105CutoffValidationError(
            f"{field} must be finite",
            context={"field": field},
        )
    return result


def _numeric_tuple(values: object, field: str) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise Foams105CutoffValidationError(
            f"{field} must be a numeric sequence",
            context={"field": field},
        )
    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105CutoffValidationError(
            f"{field} must be a numeric sequence",
            context={"field": field},
        ) from exc
    return tuple(
        _real_scalar(value, f"{field}[{index}]")
        for index, value in enumerate(raw_values)
    )


def _validated_labels(values: object) -> Tuple[float, ...]:
    labels = _numeric_tuple(values, "bin_labels_mm")
    if not 1 <= len(labels) <= FOAMS105_CUTOFF_MAX_LABELS:
        raise Foams105CutoffValidationError(
            f"bin_labels_mm must contain 1 to {FOAMS105_CUTOFF_MAX_LABELS} labels",
            code="unsupported_grid",
            context={"label_count": len(labels)},
        )
    for index, label in enumerate(labels):
        if label <= 0.0:
            raise Foams105CutoffValidationError(
                f"bin_labels_mm[{index}] must be positive for this suggestion profile",
                code="unsupported_grid",
                context={"index": index, "value": label},
            )
        if index and label <= labels[index - 1]:
            relation = "duplicate" if label == labels[index - 1] else "descending"
            raise Foams105CutoffValidationError(
                "bin_labels_mm must be strictly increasing for this suggestion profile",
                code="unsupported_grid",
                context={"index": index, "relation": relation, "value": label},
            )
    return labels


def _validated_groups(
    groups: object, label_count: int
) -> Tuple[Foams105CutoffGroupInput, ...]:
    if isinstance(groups, (str, bytes)):
        raise Foams105CutoffValidationError("groups must be a sequence of group records")
    try:
        raw_groups = tuple(groups)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105CutoffValidationError(
            "groups must be a sequence of group records"
        ) from exc
    if not 1 <= len(raw_groups) <= FOAMS105_CUTOFF_MAX_GROUPS:
        raise Foams105CutoffValidationError(
            f"groups must contain 1 to {FOAMS105_CUTOFF_MAX_GROUPS} records",
            context={"group_count": len(raw_groups)},
        )

    normalized = []
    seen_ids = set()
    seen_scales = set()
    previous_scale = None
    for index, group in enumerate(raw_groups):
        if not isinstance(group, Foams105CutoffGroupInput):
            raise Foams105CutoffValidationError(
                f"groups[{index}] must be a Foams105CutoffGroupInput",
                context={"group_index": index},
            )
        if not isinstance(group.group_id, str) or not group.group_id.strip():
            raise Foams105CutoffValidationError(
                f"groups[{index}].group_id must be a nonempty string",
                context={"group_index": index},
            )
        if group.group_id in seen_ids:
            raise Foams105CutoffValidationError(
                f"duplicate group_id {group.group_id!r}",
                context={"group_id": group.group_id},
            )
        if not isinstance(group.na_per_mm2, tuple):
            raise Foams105CutoffValidationError(
                f"groups[{index}].na_per_mm2 must be an immutable tuple",
                context={"group_id": group.group_id},
            )
        scale = _real_scalar(group.scale_px_per_mm, f"groups[{index}].scale_px_per_mm")
        if scale <= 0.0:
            raise Foams105CutoffValidationError(
                f"groups[{index}].scale_px_per_mm must be positive",
                context={"group_id": group.group_id, "scale_px_per_mm": scale},
            )
        if scale in seen_scales:
            raise Foams105CutoffValidationError(
                f"duplicate scale_px_per_mm {scale!r}",
                context={"group_id": group.group_id, "scale_px_per_mm": scale},
            )
        if previous_scale is not None and scale <= previous_scale:
            raise Foams105CutoffValidationError(
                "groups must be ordered by strictly ascending scale_px_per_mm",
                context={
                    "group_id": group.group_id,
                    "previous_scale_px_per_mm": previous_scale,
                    "scale_px_per_mm": scale,
                },
            )
        densities = _numeric_tuple(group.na_per_mm2, f"groups[{index}].na_per_mm2")
        if len(densities) != label_count:
            raise Foams105CutoffValidationError(
                f"groups[{index}].na_per_mm2 length must match bin_labels_mm",
                context={
                    "group_id": group.group_id,
                    "density_count": len(densities),
                    "label_count": label_count,
                },
            )
        for density_index, density in enumerate(densities):
            if density < 0.0:
                raise Foams105CutoffValidationError(
                    f"groups[{index}].na_per_mm2[{density_index}] must be nonnegative",
                    context={
                        "group_id": group.group_id,
                        "density_index": density_index,
                        "value": density,
                    },
                )
        normalized.append(Foams105CutoffGroupInput(group.group_id, scale, densities))
        seen_ids.add(group.group_id)
        seen_scales.add(scale)
        previous_scale = scale
    return tuple(normalized)


def _transition(
    coarser: Foams105CutoffGroupInput,
    finer: Foams105CutoffGroupInput,
) -> Foams105CutoffTransition:
    overlap = tuple(
        index
        for index, (coarse_density, fine_density) in enumerate(
            zip(coarser.na_per_mm2, finer.na_per_mm2)
        )
        if coarse_density > 0.0 and fine_density > 0.0
    )
    pair = (coarser.group_id, finer.group_id)
    if not overlap:
        raise Foams105CutoffDomainError(
            "no_positive_overlap",
            f"Groups {pair!r} have no rows with positive density in both groups",
            context={"pair_group_ids": pair},
        )

    differences = tuple(
        math.log(finer.na_per_mm2[index])
        - math.log(coarser.na_per_mm2[index])
        for index in overlap
    )
    for index, difference in zip(overlap, differences):
        if not math.isfinite(difference):
            raise Foams105CutoffDomainError(
                "nonfinite_log_difference",
                f"Groups {pair!r} produced a nonfinite log difference at index {index}",
                context={"pair_group_ids": pair, "index": index},
            )

    positive = tuple(value for value in differences if value > 0.0)
    negative = tuple(value for value in differences if value < 0.0)
    positive_candidate = min(positive) if positive else None
    negative_candidate = max(negative) if negative else None
    if positive_candidate is None and negative_candidate is None:
        raise Foams105CutoffDomainError(
            "no_nonzero_log_difference",
            f"Groups {pair!r} have only exact-zero log differences in positive overlap",
            context={"pair_group_ids": pair, "overlap_indices": overlap},
        )
    if positive_candidate is None:
        selected = negative_candidate
    elif negative_candidate is None:
        selected = positive_candidate
    elif positive_candidate < abs(negative_candidate):
        selected = positive_candidate
    else:
        selected = negative_candidate
    if selected is None:
        raise RuntimeError("unreachable transition selection state")

    tied_indices = tuple(
        index
        for index, difference in zip(overlap, differences)
        if difference == selected
    )
    return Foams105CutoffTransition(
        coarser_group_id=coarser.group_id,
        finer_group_id=finer.group_id,
        overlap_indices=overlap,
        signed_log_differences=differences,
        smallest_positive_difference=positive_candidate,
        largest_negative_difference=negative_candidate,
        selected_signed_difference=selected,
        tied_indices=tied_indices,
        selected_index=tied_indices[-1],
    )


def _positive_support(group: Foams105CutoffGroupInput) -> Tuple[int, ...]:
    return tuple(
        index for index, density in enumerate(group.na_per_mm2) if density > 0.0
    )


def _range_record(
    group: Foams105CutoffGroupInput,
    labels: Tuple[float, ...],
    lower: int,
    upper: int,
) -> Foams105SuggestedRange:
    if lower < 0 or upper >= len(labels):
        raise Foams105CutoffDomainError(
            "suggested_index_out_of_bounds",
            f"Suggested range for group {group.group_id!r} is outside the label grid",
            context={
                "group_id": group.group_id,
                "proposed_lower_index": lower,
                "proposed_upper_index": upper,
                "label_count": len(labels),
            },
        )
    if lower > upper:
        raise Foams105CutoffDomainError(
            "invalid_suggested_range",
            f"Suggested lower index exceeds upper index for group {group.group_id!r}",
            context={
                "group_id": group.group_id,
                "proposed_lower_index": lower,
                "proposed_upper_index": upper,
            },
        )
    return Foams105SuggestedRange(
        group_id=group.group_id,
        scale_px_per_mm=group.scale_px_per_mm,
        lower_index=lower,
        upper_index=upper,
        lower_label_mm=labels[lower],
        upper_label_mm=labels[upper],
    )


def suggest_foams105_cutoffs(
    bin_labels_mm: Sequence[float],
    groups: Sequence[Foams105CutoffGroupInput],
) -> Foams105CutoffSuggestionResult:
    """Return original-style default cutoff suggestions without applying them."""
    labels = _validated_labels(bin_labels_mm)
    normalized_groups = _validated_groups(groups, len(labels))
    group_count = len(normalized_groups)

    if group_count == 1:
        support = _positive_support(normalized_groups[0])
        if not support:
            raise Foams105CutoffDomainError(
                "empty_group_support",
                f"Group {normalized_groups[0].group_id!r} has no positive density rows",
                context={"group_id": normalized_groups[0].group_id},
            )
        ranges = (
            _range_record(
                normalized_groups[0], labels, support[0], support[-1]
            ),
        )
        return Foams105CutoffSuggestionResult(
            bin_labels_mm=labels,
            groups=normalized_groups,
            transitions=(),
            suggested_ranges=ranges,
        )

    transitions = tuple(
        _transition(normalized_groups[index], normalized_groups[index + 1])
        for index in range(group_count - 1)
    )
    transition_indices = tuple(item.selected_index for item in transitions)
    for index, transition_index in enumerate(transition_indices):
        if transition_index + 1 >= len(labels):
            pair = (
                normalized_groups[index].group_id,
                normalized_groups[index + 1].group_id,
            )
            raise Foams105CutoffDomainError(
                "transition_has_no_successor",
                f"Transition for groups {pair!r} is at the final label",
                context={
                    "pair_group_ids": pair,
                    "transition_index": transition_index,
                },
            )

    coarsest_support = _positive_support(normalized_groups[0])
    proposed_ranges = [
        (transition_indices[0] + 1, coarsest_support[-1])
    ]
    proposed_ranges.extend(
        (transition_indices[index] + 1, transition_indices[index - 1])
        for index in range(1, group_count - 1)
    )
    finest_upper = transition_indices[-1]
    if group_count == 4:
        finest_lower = 0
    else:
        finest_support = _positive_support(normalized_groups[-1])
        finest_lower = finest_support[0]
    proposed_ranges.append((finest_lower, finest_upper))

    ranges = tuple(
        _range_record(group, labels, lower, upper)
        for group, (lower, upper) in zip(normalized_groups, proposed_ranges)
    )
    return Foams105CutoffSuggestionResult(
        bin_labels_mm=labels,
        groups=normalized_groups,
        transitions=transitions,
        suggested_ranges=ranges,
    )
