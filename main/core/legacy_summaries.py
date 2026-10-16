"""FOAMS 1.0.5 density summaries and plotting-quantity replay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .legacy_foams import FOAMS105_MAX_CLASSES
from .legacy_vesicularity import (
    Foams105VesicularityResult,
    calculate_foams105_2d_vesicularity,
)
from .legacy_volumes import (
    Foams105VolumeDistributionResult,
    calculate_foams105_volume_distribution,
)

FOAMS105_SUMMARY_METHOD = "foams_1_0_5_density_summary_replay_v1"
FOAMS105_CATEGORY_METHOD = "foams_1_0_5_size_category_summary_replay_v1"
FOAMS105_SUMMARY_SOURCE_REPOSITORY = "jsoteloflores/FOAMS-1.0.5"
FOAMS105_SUMMARY_SOURCE_COMMIT = "179663203f2d0f86b2863d5ea7f8f70dadca02f8"
FOAMS105_SUMMARY_SOURCE_FILE = "analysis.m"
FOAMS105_SUMMARY_SOURCE_LINES = "445-501,526-540,558-597"
FOAMS105_SUMMARY_SCOPE = "distribution_density_and_plotting_quantities_only"
FOAMS105_CATEGORY_SCOPE = "complete_histogram_size_categories_only"
FOAMS105_CUMULATIVE_POLICY = "inclusive_reverse_sum_in_selected_row_order"
FOAMS105_WIDTH_POLICY = "source_predecessor_or_first_grid_geometric_fallback"
FOAMS105_PAIRING_POLICY = "consecutive_adjusted_volume_rows_with_odd_singleton"
FOAMS105_CORRECTION_POLICY = "integrated_or_known_percent_without_altering_nv"
FOAMS105_CATEGORY_BOUNDARY_POLICY = "strict_categories_exclude_cutoff_labels"
FOAMS105_MAX_COUNT_GROUPS = 4
FOAMS105_MAX_EXACT_COUNT = 2**53 - 1

REASON_ZERO_BIN_WIDTH = "zero_bin_width"
REASON_NEGATIVE_BIN_WIDTH = "negative_bin_width"
REASON_LOG_OF_ZERO = "log_of_zero"
REASON_LOG_OF_NEGATIVE = "log_of_negative"
REASON_ZERO_SOLID_FRACTION = "zero_solid_fraction"
REASON_ZERO_TOTAL_COUNT = "zero_total_count"

LEGACY_POSITIVE_INFINITY = "positive_infinity"
LEGACY_NEGATIVE_INFINITY = "negative_infinity"
LEGACY_INDETERMINATE = "indeterminate_0_over_0"
LEGACY_COMPLEX = "complex"

DIAGNOSTIC_NEGATIVE_NV_ROW = "negative_nv_row"
DIAGNOSTIC_NEGATIVE_CUMULATIVE_NV = "negative_cumulative_nv"
DIAGNOSTIC_NEGATIVE_PER_WIDTH_DENSITY = "negative_per_width_density"
DIAGNOSTIC_NEGATIVE_SOLID_FRACTION = "negative_solid_fraction"
DIAGNOSTIC_ZERO_BIN_WIDTH = "zero_bin_width"


class Foams105SummaryValidationError(ValueError):
    """Raised when summary-stage inputs violate the bounded contract."""

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


class Foams105SummaryNumericalError(ArithmeticError):
    """Raised when finite-domain summary arithmetic becomes invalid."""

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
class Foams105RealValue:
    """A finite real value or an explicit source-domain undefined state."""

    value: Optional[float]
    reason: Optional[str] = None
    operands: Tuple[float, ...] = ()
    legacy_classification: Optional[str] = None
    upstream_reason: Optional[str] = None


@dataclass(frozen=True)
class Foams105SummaryDiagnostic:
    """One inspectable signed or boundary condition."""

    code: str
    quantity: str
    row_index: Optional[int] = None
    value: Optional[float] = None


@dataclass(frozen=True)
class Foams105DistributionSummaryRow:
    """One selected converter row and its source plotting quantities."""

    output_index: int
    bin_label_mm: float
    lower_edge_mm: float
    width_mm: float
    midsize_mm: float
    signed_nv_per_mm3: float
    cumulative_nv_per_mm3: float
    log10_label: Foams105RealValue
    log10_cumulative_nv: Foams105RealValue
    number_density_per_width_mm4: Foams105RealValue
    ln_number_density_per_width: Foams105RealValue
    log10_cumulative_eligible: bool
    log10_cumulative_reason: Optional[str]
    ln_density_eligible: bool
    ln_density_reason: Optional[str]


@dataclass(frozen=True)
class Foams105PairedVolumeBin:
    """One consecutive two-row volume pair or trailing singleton."""

    output_index: int
    source_row_indices: Tuple[int, ...]
    bin_label_mm: float
    adjusted_volume_fraction: float


@dataclass(frozen=True)
class Foams105DistributionSummaryResult:
    """Canonical upstream provenance and source distribution summaries."""

    volume_result: Foams105VolumeDistributionResult
    correction_vesicularity_result: Optional[Foams105VesicularityResult]
    global_bin_labels_mm: Tuple[float, ...]
    correction_source: str
    correction_percent: float
    solid_fraction: float
    total_nv_per_mm3: float
    corrected_nv_per_mm3: Foams105RealValue
    rows: Tuple[Foams105DistributionSummaryRow, ...]
    lower_edges_mm: Tuple[float, ...]
    widths_mm: Tuple[float, ...]
    midsizes_mm: Tuple[float, ...]
    cumulative_nv_per_mm3: Tuple[float, ...]
    log10_labels: Tuple[Foams105RealValue, ...]
    log10_cumulative_nv: Tuple[Foams105RealValue, ...]
    number_density_per_width_mm4: Tuple[Foams105RealValue, ...]
    ln_number_density_per_width: Tuple[Foams105RealValue, ...]
    paired_volume_bins: Tuple[Foams105PairedVolumeBin, ...]
    diagnostics: Tuple[Foams105SummaryDiagnostic, ...]
    method: str = FOAMS105_SUMMARY_METHOD
    source_repository: str = FOAMS105_SUMMARY_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_SUMMARY_SOURCE_COMMIT
    source_file: str = FOAMS105_SUMMARY_SOURCE_FILE
    source_lines: str = FOAMS105_SUMMARY_SOURCE_LINES
    scope: str = FOAMS105_SUMMARY_SCOPE
    cumulative_policy: str = FOAMS105_CUMULATIVE_POLICY
    width_policy: str = FOAMS105_WIDTH_POLICY
    pairing_policy: str = FOAMS105_PAIRING_POLICY
    correction_policy: str = FOAMS105_CORRECTION_POLICY
    length_unit: str = "mm"
    number_density_unit: str = "mm^-3"
    per_width_density_unit: str = "mm^-4"
    fraction_unit: str = "dimensionless"
    percentage_unit: str = "percent"


@dataclass(frozen=True)
class Foams105CountGroupInput:
    """One complete post-minimum-filter histogram count vector."""

    group_id: str
    source_slot: int
    counts: Tuple[int, ...]


@dataclass(frozen=True)
class Foams105SizeCategoryResult:
    """Pooled strict-boundary size-category counts and percentages."""

    global_bin_labels_mm: Tuple[float, ...]
    count_groups: Tuple[Foams105CountGroupInput, ...]
    pooled_counts: Tuple[int, ...]
    lower_cutoff_mm: float
    upper_cutoff_mm: float
    lower_indices: Tuple[int, ...]
    middle_indices: Tuple[int, ...]
    upper_indices: Tuple[int, ...]
    boundary_indices: Tuple[int, ...]
    lower_count: int
    middle_count: int
    upper_count: int
    boundary_count: int
    total_count: int
    lower_percentage: Foams105RealValue
    middle_percentage: Foams105RealValue
    upper_percentage: Foams105RealValue
    method: str = FOAMS105_CATEGORY_METHOD
    source_repository: str = FOAMS105_SUMMARY_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_SUMMARY_SOURCE_COMMIT
    source_file: str = FOAMS105_SUMMARY_SOURCE_FILE
    source_lines: str = FOAMS105_SUMMARY_SOURCE_LINES
    scope: str = FOAMS105_CATEGORY_SCOPE
    boundary_policy: str = FOAMS105_CATEGORY_BOUNDARY_POLICY
    length_unit: str = "mm"
    percentage_unit: str = "percent"


@dataclass(frozen=True)
class _Foams105DistributionNumericalResult:
    lower_edges_mm: Tuple[float, ...]
    widths_mm: Tuple[float, ...]
    midsizes_mm: Tuple[float, ...]
    cumulative_nv_per_mm3: Tuple[float, ...]
    log10_labels: Tuple[Foams105RealValue, ...]
    log10_cumulative_nv: Tuple[Foams105RealValue, ...]
    number_density_per_width_mm4: Tuple[Foams105RealValue, ...]
    ln_number_density_per_width: Tuple[Foams105RealValue, ...]
    total_nv_per_mm3: float
    solid_fraction: float
    corrected_nv_per_mm3: Foams105RealValue
    paired_volume_bins: Tuple[Foams105PairedVolumeBin, ...]
    diagnostics: Tuple[Foams105SummaryDiagnostic, ...]


def _finite_real(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise Foams105SummaryValidationError(
            f"{field} must be a finite real scalar", context={"field": field}
        )
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise Foams105SummaryValidationError(
            f"{field} must be representable as float64", context={"field": field}
        ) from exc
    if not math.isfinite(result):
        raise Foams105SummaryValidationError(
            f"{field} must be finite", context={"field": field}
        )
    return result


def _numeric_tuple(values: object, field: str) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise Foams105SummaryValidationError(f"{field} must be a numeric sequence")
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105SummaryValidationError(
            f"{field} must be a numeric sequence"
        ) from exc
    return tuple(_finite_real(value, f"{field}[{index}]") for index, value in enumerate(raw))


def _validated_global_labels(values: object) -> Tuple[float, ...]:
    labels = _numeric_tuple(values, "global_bin_labels_mm")
    if not 1 <= len(labels) <= FOAMS105_MAX_CLASSES:
        raise Foams105SummaryValidationError(
            f"global_bin_labels_mm must contain 1 to {FOAMS105_MAX_CLASSES} values"
        )
    for index, label in enumerate(labels):
        if label <= 0.0:
            raise Foams105SummaryValidationError(
                f"global_bin_labels_mm[{index}] must be positive"
            )
        if index and label <= labels[index - 1]:
            raise Foams105SummaryValidationError(
                "global_bin_labels_mm must be strictly increasing",
                code="ambiguous_global_grid",
                context={"row_index": index},
            )
    return labels


def _numerical_error(
    stage: str,
    *,
    row_index: Optional[int] = None,
    operands: Sequence[float] = (),
    code: str = "nonfinite_summary_arithmetic",
) -> Foams105SummaryNumericalError:
    return Foams105SummaryNumericalError(
        code,
        f"{stage} produced an invalid float64 result",
        context={
            "stage": stage,
            "row_index": row_index,
            "operands": tuple(operands),
        },
    )


def _checked_add(left: float, right: float, *, stage: str, row_index: int) -> float:
    result = left + right
    if not math.isfinite(result):
        raise _numerical_error(
            stage, row_index=row_index, operands=(left, right)
        )
    return result


def _checked_multiply(
    left: float,
    right: float,
    *,
    stage: str,
    row_index: Optional[int] = None,
) -> float:
    result = left * right
    if not math.isfinite(result):
        raise _numerical_error(stage, row_index=row_index, operands=(left, right))
    if left != 0.0 and right != 0.0 and result == 0.0:
        raise _numerical_error(
            stage,
            row_index=row_index,
            operands=(left, right),
            code="nonzero_underflow",
        )
    return result


def _checked_divide(
    numerator: float,
    denominator: float,
    *,
    stage: str,
    row_index: Optional[int] = None,
) -> float:
    result = numerator / denominator
    if not math.isfinite(result):
        raise _numerical_error(
            stage, row_index=row_index, operands=(numerator, denominator)
        )
    if numerator != 0.0 and result == 0.0:
        raise _numerical_error(
            stage,
            row_index=row_index,
            operands=(numerator, denominator),
            code="nonzero_underflow",
        )
    return result


def _finite_value(value: float) -> Foams105RealValue:
    return Foams105RealValue(value=value)


def _undefined(
    reason: str,
    operands: Sequence[float],
    legacy_classification: Optional[str],
    *,
    upstream_reason: Optional[str] = None,
) -> Foams105RealValue:
    return Foams105RealValue(
        value=None,
        reason=reason,
        operands=tuple(operands),
        legacy_classification=legacy_classification,
        upstream_reason=upstream_reason,
    )


def _division_classification(numerator: float) -> str:
    if numerator > 0.0:
        return LEGACY_POSITIVE_INFINITY
    if numerator < 0.0:
        return LEGACY_NEGATIVE_INFINITY
    return LEGACY_INDETERMINATE


def _log_value(value: float, *, base10: bool) -> Foams105RealValue:
    if value == 0.0:
        return _undefined(REASON_LOG_OF_ZERO, (value,), LEGACY_NEGATIVE_INFINITY)
    if value < 0.0:
        return _undefined(REASON_LOG_OF_NEGATIVE, (value,), LEGACY_COMPLEX)
    result = math.log10(value) if base10 else math.log(value)
    if not math.isfinite(result):
        raise _numerical_error(
            "log10" if base10 else "natural_log", operands=(value,)
        )
    return _finite_value(result)


def _calculate_foams105_distribution_numerical(
    global_bin_labels_mm: Sequence[float],
    selected_labels_mm: Sequence[float],
    signed_nv_per_mm3: Sequence[float],
    adjusted_volume_fractions: Sequence[float],
    correction_percent: float,
) -> _Foams105DistributionNumericalResult:
    """Calculate source-order density quantities for analytical vectors."""
    global_labels = _validated_global_labels(global_bin_labels_mm)
    labels = _numeric_tuple(selected_labels_mm, "selected_labels_mm")
    signed_nv = _numeric_tuple(signed_nv_per_mm3, "signed_nv_per_mm3")
    adjusted = _numeric_tuple(
        adjusted_volume_fractions, "adjusted_volume_fractions"
    )
    correction = _finite_real(correction_percent, "correction_percent")
    if not 1 <= len(labels) <= FOAMS105_MAX_CLASSES:
        raise Foams105SummaryValidationError(
            f"selected vectors must contain 1 to {FOAMS105_MAX_CLASSES} rows"
        )
    if len(labels) != len(signed_nv) or len(labels) != len(adjusted):
        raise Foams105SummaryValidationError(
            "selected labels, N_V, and adjusted fractions must have equal lengths"
        )
    global_indices = {label: index for index, label in enumerate(global_labels)}
    for index, label in enumerate(labels):
        if label <= 0.0:
            raise Foams105SummaryValidationError(
                f"selected_labels_mm[{index}] must be positive"
            )
        if index and label < labels[index - 1]:
            raise Foams105SummaryValidationError(
                "selected labels must be nondecreasing"
            )
        if label not in global_indices:
            raise Foams105SummaryValidationError(
                "every selected label must exactly belong to the global grid",
                code="selected_label_absent_from_global_grid",
                context={"row_index": index, "label_mm": label},
            )

    first_global_index = global_indices[labels[0]]
    if first_global_index:
        first_lower = global_labels[first_global_index - 1]
    else:
        first_lower = _checked_divide(
            global_labels[0],
            10.0**0.1,
            stage="first_lower_geometric_fallback",
            row_index=0,
        )
    lower_edges = tuple(sorted((first_lower, *labels[:-1])))
    widths = []
    midsizes = []
    diagnostics = []
    for index, (label, lower) in enumerate(zip(labels, lower_edges)):
        width = label - lower
        if not math.isfinite(width):
            raise _numerical_error(
                "bin_width", row_index=index, operands=(label, lower)
            )
        if width < 0.0:
            raise Foams105SummaryValidationError(
                "source width calculation produced a negative width",
                code=REASON_NEGATIVE_BIN_WIDTH,
                context={"row_index": index, "label_mm": label, "lower_edge_mm": lower},
            )
        midpoint_sum = _checked_add(
            lower, label, stage="midsize_sum", row_index=index
        )
        midsize = _checked_divide(
            midpoint_sum, 2.0, stage="midsize_divide_two", row_index=index
        )
        widths.append(width)
        midsizes.append(midsize)
        if width == 0.0:
            diagnostics.append(
                Foams105SummaryDiagnostic(
                    DIAGNOSTIC_ZERO_BIN_WIDTH,
                    "width_mm",
                    row_index=index,
                    value=width,
                )
            )

    cumulative = [0.0] * len(labels)
    running = 0.0
    for index in range(len(labels) - 1, -1, -1):
        running = _checked_add(
            running,
            signed_nv[index],
            stage="reverse_cumulative_nv",
            row_index=index,
        )
        cumulative[index] = running

    log10_labels = tuple(_log_value(label, base10=True) for label in labels)
    log10_cumulative = tuple(
        _log_value(value, base10=True) for value in cumulative
    )
    per_width = []
    ln_per_width = []
    for index, (nv_value, width) in enumerate(zip(signed_nv, widths)):
        if width == 0.0:
            density = _undefined(
                REASON_ZERO_BIN_WIDTH,
                (nv_value, width),
                _division_classification(nv_value),
            )
            logged = _undefined(
                REASON_ZERO_BIN_WIDTH,
                (nv_value, width),
                None,
                upstream_reason=REASON_ZERO_BIN_WIDTH,
            )
        else:
            density_value = _checked_divide(
                nv_value,
                width,
                stage="number_density_per_width",
                row_index=index,
            )
            density = _finite_value(density_value)
            logged = _log_value(density_value, base10=False)
            if density_value < 0.0:
                diagnostics.append(
                    Foams105SummaryDiagnostic(
                        DIAGNOSTIC_NEGATIVE_PER_WIDTH_DENSITY,
                        "number_density_per_width_mm4",
                        row_index=index,
                        value=density_value,
                    )
                )
        per_width.append(density)
        ln_per_width.append(logged)
        if nv_value < 0.0:
            diagnostics.append(
                Foams105SummaryDiagnostic(
                    DIAGNOSTIC_NEGATIVE_NV_ROW,
                    "signed_nv_per_mm3",
                    row_index=index,
                    value=nv_value,
                )
            )
        if cumulative[index] < 0.0:
            diagnostics.append(
                Foams105SummaryDiagnostic(
                    DIAGNOSTIC_NEGATIVE_CUMULATIVE_NV,
                    "cumulative_nv_per_mm3",
                    row_index=index,
                    value=cumulative[index],
                )
            )

    total_nv = 0.0
    for index, nv_value in enumerate(signed_nv):
        total_nv = _checked_add(
            total_nv, nv_value, stage="forward_total_nv", row_index=index
        )
    scaled_percent = _checked_multiply(
        0.01, correction, stage="correction_percent_times_0_01"
    )
    solid_fraction = 1.0 - scaled_percent
    if not math.isfinite(solid_fraction):
        raise _numerical_error(
            "solid_fraction_subtraction", operands=(1.0, scaled_percent)
        )
    if solid_fraction == 0.0:
        corrected = _undefined(
            REASON_ZERO_SOLID_FRACTION,
            (total_nv, solid_fraction),
            _division_classification(total_nv),
        )
    else:
        corrected = _finite_value(
            _checked_divide(
                total_nv,
                solid_fraction,
                stage="corrected_nv",
            )
        )
    if solid_fraction < 0.0:
        diagnostics.append(
            Foams105SummaryDiagnostic(
                DIAGNOSTIC_NEGATIVE_SOLID_FRACTION,
                "solid_fraction",
                value=solid_fraction,
            )
        )

    pairs = []
    pair_index = 0
    row_index = 0
    while row_index < len(labels):
        if row_index + 1 < len(labels):
            indices = (row_index, row_index + 1)
            pair_fraction = _checked_add(
                adjusted[row_index],
                adjusted[row_index + 1],
                stage="paired_adjusted_volume_fraction",
                row_index=pair_index,
            )
            pair_label = labels[row_index + 1]
        else:
            indices = (row_index,)
            pair_fraction = adjusted[row_index]
            pair_label = labels[row_index]
        pairs.append(
            Foams105PairedVolumeBin(
                pair_index, indices, pair_label, pair_fraction
            )
        )
        pair_index += 1
        row_index += 2

    return _Foams105DistributionNumericalResult(
        lower_edges_mm=lower_edges,
        widths_mm=tuple(widths),
        midsizes_mm=tuple(midsizes),
        cumulative_nv_per_mm3=tuple(cumulative),
        log10_labels=log10_labels,
        log10_cumulative_nv=log10_cumulative,
        number_density_per_width_mm4=tuple(per_width),
        ln_number_density_per_width=tuple(ln_per_width),
        total_nv_per_mm3=total_nv,
        solid_fraction=solid_fraction,
        corrected_nv_per_mm3=corrected,
        paired_volume_bins=tuple(pairs),
        diagnostics=tuple(diagnostics),
    )


def _canonical_vesicularity(
    result: object,
) -> Foams105VesicularityResult:
    if not isinstance(result, Foams105VesicularityResult):
        raise Foams105SummaryValidationError(
            "vesicularity_result must be a Foams105VesicularityResult",
            code="missing_vesicularity_result",
        )
    try:
        canonical = calculate_foams105_2d_vesicularity(
            result.raw_selection, result.groups
        )
    except (ValueError, ArithmeticError) as exc:
        raise Foams105SummaryValidationError(
            "vesicularity_result cannot be replayed from its inputs",
            code="inconsistent_vesicularity_result",
        ) from exc
    if canonical != result:
        raise Foams105SummaryValidationError(
            "vesicularity_result differs from its canonical replay",
            code="inconsistent_vesicularity_result",
        )
    return canonical


def _canonical_volume(result: object) -> Foams105VolumeDistributionResult:
    if not isinstance(result, Foams105VolumeDistributionResult):
        raise Foams105SummaryValidationError(
            "volume_result must be a Foams105VolumeDistributionResult"
        )
    try:
        if result.mode == "unnormalized":
            canonical = calculate_foams105_volume_distribution(
                result.conversion_result
            )
        elif result.mode == "integrated_2d":
            canonical = calculate_foams105_volume_distribution(
                result.conversion_result,
                "integrated_2d",
                vesicularity_result=result.vesicularity_result,
            )
        elif result.mode == "known_vesicularity":
            canonical = calculate_foams105_volume_distribution(
                result.conversion_result,
                "known_vesicularity",
                known_vesicularity_percent=result.target_vesicularity_percent,
            )
        else:
            raise Foams105SummaryValidationError(
                "volume_result has an unsupported mode",
                code="inconsistent_volume_result",
            )
    except (ValueError, ArithmeticError) as exc:
        raise Foams105SummaryValidationError(
            "volume_result cannot be replayed from its inputs",
            code="inconsistent_volume_result",
        ) from exc
    if canonical != result:
        raise Foams105SummaryValidationError(
            "volume_result differs from its canonical replay",
            code="inconsistent_volume_result",
        )
    return canonical


def summarize_foams105_distribution(
    volume_result: Foams105VolumeDistributionResult,
    global_bin_labels_mm: Sequence[float],
    *,
    vesicularity_result: Optional[Foams105VesicularityResult] = None,
) -> Foams105DistributionSummaryResult:
    """Replay density and plotting quantities from a canonical volume result."""
    canonical_volume = _canonical_volume(volume_result)
    global_labels = _validated_global_labels(global_bin_labels_mm)
    if canonical_volume.mode == "unnormalized":
        correction_vesicularity = _canonical_vesicularity(vesicularity_result)
        correction_source = "integrated_2d_vesicularity_percent"
        correction_percent = correction_vesicularity.integrated_vesicularity_percent
    elif canonical_volume.mode == "integrated_2d":
        if vesicularity_result is not None:
            raise Foams105SummaryValidationError(
                "integrated_2d mode rejects a redundant external result",
                code="unused_vesicularity_result",
            )
        correction_vesicularity = canonical_volume.vesicularity_result
        if correction_vesicularity is None:
            raise Foams105SummaryValidationError(
                "integrated volume result lost its vesicularity provenance",
                code="inconsistent_volume_result",
            )
        correction_source = "integrated_2d_vesicularity_percent"
        correction_percent = correction_vesicularity.integrated_vesicularity_percent
    else:
        if vesicularity_result is not None:
            raise Foams105SummaryValidationError(
                "known_vesicularity mode rejects an external 2D result",
                code="unused_vesicularity_result",
            )
        correction_vesicularity = None
        correction_source = "known_vesicularity_percent"
        if canonical_volume.target_vesicularity_percent is None:
            raise Foams105SummaryValidationError(
                "known volume result lost its target",
                code="inconsistent_volume_result",
            )
        correction_percent = canonical_volume.target_vesicularity_percent

    conversion = canonical_volume.conversion_result
    numerical = _calculate_foams105_distribution_numerical(
        global_labels,
        conversion.bin_labels_mm,
        conversion.signed_nv_per_mm3,
        canonical_volume.adjusted_volume_fractions,
        correction_percent,
    )
    rows = tuple(
        Foams105DistributionSummaryRow(
            output_index=index,
            bin_label_mm=label,
            lower_edge_mm=numerical.lower_edges_mm[index],
            width_mm=numerical.widths_mm[index],
            midsize_mm=numerical.midsizes_mm[index],
            signed_nv_per_mm3=conversion.signed_nv_per_mm3[index],
            cumulative_nv_per_mm3=numerical.cumulative_nv_per_mm3[index],
            log10_label=numerical.log10_labels[index],
            log10_cumulative_nv=numerical.log10_cumulative_nv[index],
            number_density_per_width_mm4=(
                numerical.number_density_per_width_mm4[index]
            ),
            ln_number_density_per_width=(
                numerical.ln_number_density_per_width[index]
            ),
            log10_cumulative_eligible=(
                numerical.log10_cumulative_nv[index].value is not None
            ),
            log10_cumulative_reason=numerical.log10_cumulative_nv[index].reason,
            ln_density_eligible=(
                numerical.ln_number_density_per_width[index].value is not None
            ),
            ln_density_reason=numerical.ln_number_density_per_width[index].reason,
        )
        for index, label in enumerate(conversion.bin_labels_mm)
    )
    return Foams105DistributionSummaryResult(
        volume_result=canonical_volume,
        correction_vesicularity_result=correction_vesicularity,
        global_bin_labels_mm=global_labels,
        correction_source=correction_source,
        correction_percent=correction_percent,
        solid_fraction=numerical.solid_fraction,
        total_nv_per_mm3=numerical.total_nv_per_mm3,
        corrected_nv_per_mm3=numerical.corrected_nv_per_mm3,
        rows=rows,
        lower_edges_mm=numerical.lower_edges_mm,
        widths_mm=numerical.widths_mm,
        midsizes_mm=numerical.midsizes_mm,
        cumulative_nv_per_mm3=numerical.cumulative_nv_per_mm3,
        log10_labels=numerical.log10_labels,
        log10_cumulative_nv=numerical.log10_cumulative_nv,
        number_density_per_width_mm4=numerical.number_density_per_width_mm4,
        ln_number_density_per_width=numerical.ln_number_density_per_width,
        paired_volume_bins=numerical.paired_volume_bins,
        diagnostics=numerical.diagnostics,
    )


def _validated_count_groups(
    groups: object,
    label_count: int,
) -> Tuple[Foams105CountGroupInput, ...]:
    if isinstance(groups, (str, bytes)):
        raise Foams105SummaryValidationError("count_groups must be a sequence")
    try:
        raw_groups = tuple(groups)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105SummaryValidationError(
            "count_groups must be a sequence"
        ) from exc
    if not 1 <= len(raw_groups) <= FOAMS105_MAX_COUNT_GROUPS:
        raise Foams105SummaryValidationError("count_groups must contain 1 to 4 groups")
    owned = []
    seen_ids = set()
    running_total = 0
    for group_index, group in enumerate(raw_groups):
        if not isinstance(group, Foams105CountGroupInput):
            raise Foams105SummaryValidationError(
                "each count group must be a Foams105CountGroupInput"
            )
        if not isinstance(group.group_id, str) or not group.group_id:
            raise Foams105SummaryValidationError("group_id must be a nonempty string")
        if group.group_id in seen_ids:
            raise Foams105SummaryValidationError(
                "group IDs must be unique", code="duplicate_group_id"
            )
        if (
            not isinstance(group.source_slot, (int, np.integer))
            or isinstance(group.source_slot, (bool, np.bool_))
            or int(group.source_slot) != group_index + 1
        ):
            raise Foams105SummaryValidationError(
                "source slots must be contiguous Python/NumPy integers from one"
            )
        try:
            raw_counts = tuple(group.counts)
        except TypeError as exc:
            raise Foams105SummaryValidationError(
                "counts must be an owned-compatible integer sequence"
            ) from exc
        if len(raw_counts) != label_count:
            raise Foams105SummaryValidationError(
                "each count vector must match the global grid length"
            )
        counts = []
        for count_index, count in enumerate(raw_counts):
            if (
                not isinstance(count, (int, np.integer))
                or isinstance(count, (bool, np.bool_))
                or int(count) < 0
            ):
                raise Foams105SummaryValidationError(
                    "counts must be nonnegative Python/NumPy integers",
                    context={"group_index": group_index, "count_index": count_index},
                )
            integer = int(count)
            running_total += integer
            if running_total > FOAMS105_MAX_EXACT_COUNT:
                raise Foams105SummaryValidationError(
                    "pooled total exceeds the exact-integer percentage profile",
                    code="count_total_exceeds_exact_limit",
                )
            counts.append(integer)
        seen_ids.add(group.group_id)
        owned.append(
            Foams105CountGroupInput(group.group_id, int(group.source_slot), tuple(counts))
        )
    return tuple(owned)


def summarize_foams105_size_categories(
    global_bin_labels_mm: Sequence[float],
    count_groups: Sequence[Foams105CountGroupInput],
    lower_cutoff_mm: float,
    upper_cutoff_mm: float,
) -> Foams105SizeCategoryResult:
    """Pool complete histograms into original strict size categories."""
    labels = _validated_global_labels(global_bin_labels_mm)
    groups = _validated_count_groups(count_groups, len(labels))
    lower = _finite_real(lower_cutoff_mm, "lower_cutoff_mm")
    upper = _finite_real(upper_cutoff_mm, "upper_cutoff_mm")
    if lower <= 0.0 or upper <= 0.0 or lower > upper:
        raise Foams105SummaryValidationError(
            "cutoffs must be positive with lower_cutoff_mm <= upper_cutoff_mm"
        )
    pooled = tuple(sum(group.counts[index] for group in groups) for index in range(len(labels)))
    total = sum(pooled)
    if total > FOAMS105_MAX_EXACT_COUNT:
        raise Foams105SummaryValidationError(
            "pooled total exceeds the exact-integer percentage profile",
            code="count_total_exceeds_exact_limit",
        )
    lower_indices = tuple(index for index, label in enumerate(labels) if label < lower)
    middle_indices = tuple(
        index for index, label in enumerate(labels) if lower < label < upper
    )
    upper_indices = tuple(index for index, label in enumerate(labels) if label > upper)
    boundary_indices = tuple(
        index for index, label in enumerate(labels) if label == lower or label == upper
    )
    lower_count = sum(pooled[index] for index in lower_indices)
    middle_count = sum(pooled[index] for index in middle_indices)
    upper_count = sum(pooled[index] for index in upper_indices)
    boundary_count = sum(pooled[index] for index in boundary_indices)
    if lower_count + middle_count + upper_count + boundary_count != total:
        raise Foams105SummaryNumericalError(
            "count_conservation_failed",
            "size-category counts do not conserve the pooled total",
        )

    def percentage(count: int) -> Foams105RealValue:
        if total == 0:
            return _undefined(
                REASON_ZERO_TOTAL_COUNT,
                (float(count), float(total)),
                LEGACY_INDETERMINATE,
            )
        ratio = _checked_divide(float(count), float(total), stage="category_ratio")
        return _finite_value(
            _checked_multiply(ratio, 100.0, stage="category_percentage")
        )

    return Foams105SizeCategoryResult(
        global_bin_labels_mm=labels,
        count_groups=groups,
        pooled_counts=pooled,
        lower_cutoff_mm=lower,
        upper_cutoff_mm=upper,
        lower_indices=lower_indices,
        middle_indices=middle_indices,
        upper_indices=upper_indices,
        boundary_indices=boundary_indices,
        lower_count=lower_count,
        middle_count=middle_count,
        upper_count=upper_count,
        boundary_count=boundary_count,
        total_count=total,
        lower_percentage=percentage(lower_count),
        middle_percentage=percentage(middle_count),
        upper_percentage=percentage(upper_count),
    )
