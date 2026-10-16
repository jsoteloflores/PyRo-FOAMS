"""FOAMS 1.0.5 volume-fraction and vesicularity normalization replay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .legacy_foams import (
    FOAMS105_MAX_CLASSES,
    Foams105ConversionResult,
    convert_foams105_nv,
)
from .legacy_vesicularity import (
    Foams105VesicularityDiagnostic,
    Foams105VesicularityResult,
    calculate_foams105_2d_vesicularity,
)

FOAMS105_VOLUME_METHOD = "foams_1_0_5_volume_fraction_normalization_replay_v1"
FOAMS105_VOLUME_SOURCE_REPOSITORY = "jsoteloflores/FOAMS-1.0.5"
FOAMS105_VOLUME_SOURCE_COMMIT = "179663203f2d0f86b2863d5ea7f8f70dadca02f8"
FOAMS105_VOLUME_SOURCE_FILE = "analysis.m"
FOAMS105_VOLUME_SOURCE_LINES = "391-409"
FOAMS105_VOLUME_SCOPE = "equivalent_sphere_volume_fraction_normalization_only"
FOAMS105_VOLUME_MODES = (
    "unnormalized",
    "integrated_2d",
    "known_vesicularity",
)
FOAMS105_VOLUME_HEIGHT_POLICY = "accepted_converter_mean_projected_heights"
FOAMS105_VOLUME_ADJUSTMENT_POLICY = "multiply_volume_fractions_not_nv"
FOAMS105_VOLUME_CUMULATIVE_POLICY = "ordered_prefix_sum_without_clipping"
FOAMS105_VOLUME_RELATIONSHIP_POLICY = (
    "caller_must_supply_conversion_and_vesicularity_from_same_analysis"
)

DIAGNOSTIC_NEGATIVE_NV_ROW = "negative_nv_row"
DIAGNOSTIC_NEGATIVE_RAW_FRACTION_ROW = "negative_raw_fraction_row"
DIAGNOSTIC_NEGATIVE_ADJUSTED_FRACTION_ROW = "negative_adjusted_fraction_row"
DIAGNOSTIC_NEGATIVE_NORMALIZATION_FACTOR = "negative_normalization_factor"
DIAGNOSTIC_NEGATIVE_RAW_FRACTION_SUM = "negative_raw_fraction_sum"
DIAGNOSTIC_RAW_FRACTION_SUM_ABOVE_ONE = "raw_fraction_sum_above_one"
DIAGNOSTIC_TARGET_PERCENT_OUT_OF_RANGE = "target_percent_out_of_range"
DIAGNOSTIC_CUMULATIVE_DECREASE = "cumulative_adjusted_fraction_decrease"
DIAGNOSTIC_NORMALIZATION_FACTOR_MAGNITUDE_ABOVE_ONE = (
    "normalization_factor_magnitude_above_one"
)


class Foams105VolumeValidationError(ValueError):
    """Raised when volume-stage inputs violate the bounded contract."""

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


class Foams105VolumeDomainError(ValueError):
    """Raised when source normalization is undefined for valid inputs."""

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


class Foams105VolumeNumericalError(ArithmeticError):
    """Raised for nonfinite or prohibited-underflow volume arithmetic."""

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
class Foams105VolumeDiagnostic:
    """One inspectable signed or nonphysical volume-stage condition."""

    code: str
    quantity: str
    row_index: Optional[int] = None
    previous_row_index: Optional[int] = None
    value: Optional[float] = None
    previous_value: Optional[float] = None


@dataclass(frozen=True)
class Foams105VolumeRow:
    """One original converter row with raw and adjusted volume fractions."""

    output_index: int
    bin_label_mm: float
    mean_projected_height_mm: float
    signed_nv_per_mm3: float
    sphere_volume_mm3: float
    raw_volume_fraction: float
    adjusted_volume_fraction: float
    cumulative_adjusted_volume_fraction: float


@dataclass(frozen=True)
class Foams105VolumeDistributionResult:
    """Canonical converter provenance and original volume normalization outputs."""

    conversion_result: Foams105ConversionResult
    vesicularity_result: Optional[Foams105VesicularityResult]
    mode: str
    target_source: Optional[str]
    target_vesicularity_percent: Optional[float]
    target_volume_fraction: Optional[float]
    normalization_factor: float
    rows: Tuple[Foams105VolumeRow, ...]
    sphere_volumes_mm3: Tuple[float, ...]
    raw_volume_fractions: Tuple[float, ...]
    adjusted_volume_fractions: Tuple[float, ...]
    cumulative_adjusted_volume_fractions: Tuple[float, ...]
    raw_fraction_sum: float
    nv_derived_vesicularity_percent: float
    adjusted_fraction_sum: float
    normalization_residual: Optional[float]
    adjacent_duplicate_label_index_pairs: Tuple[Tuple[int, int], ...]
    converter_negative_indices: Tuple[int, ...]
    diagnostics: Tuple[Foams105VolumeDiagnostic, ...]
    upstream_vesicularity_diagnostics: Tuple[Foams105VesicularityDiagnostic, ...]
    method: str = FOAMS105_VOLUME_METHOD
    source_repository: str = FOAMS105_VOLUME_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_VOLUME_SOURCE_COMMIT
    source_file: str = FOAMS105_VOLUME_SOURCE_FILE
    source_lines: str = FOAMS105_VOLUME_SOURCE_LINES
    scope: str = FOAMS105_VOLUME_SCOPE
    supported_modes: Tuple[str, str, str] = FOAMS105_VOLUME_MODES
    height_policy: str = FOAMS105_VOLUME_HEIGHT_POLICY
    adjustment_policy: str = FOAMS105_VOLUME_ADJUSTMENT_POLICY
    cumulative_policy: str = FOAMS105_VOLUME_CUMULATIVE_POLICY
    analysis_relationship_policy: str = FOAMS105_VOLUME_RELATIONSHIP_POLICY
    length_unit: str = "mm"
    volume_unit: str = "mm^3"
    number_density_unit: str = "mm^-3"
    fraction_unit: str = "dimensionless"
    percentage_unit: str = "percent"


@dataclass(frozen=True)
class _Foams105VolumeNumericalResult:
    sphere_volumes_mm3: Tuple[float, ...]
    raw_volume_fractions: Tuple[float, ...]
    raw_fraction_sum: float
    nv_derived_vesicularity_percent: float
    target_volume_fraction: Optional[float]
    normalization_factor: float
    adjusted_volume_fractions: Tuple[float, ...]
    cumulative_adjusted_volume_fractions: Tuple[float, ...]
    adjusted_fraction_sum: float
    normalization_residual: Optional[float]
    diagnostics: Tuple[Foams105VolumeDiagnostic, ...]


def _finite_real(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise Foams105VolumeValidationError(
            f"{field} must be a finite real scalar",
            context={"field": field},
        )
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise Foams105VolumeValidationError(
            f"{field} must be representable as finite float64",
            context={"field": field},
        ) from exc
    if not math.isfinite(result):
        raise Foams105VolumeValidationError(
            f"{field} must be finite",
            context={"field": field},
        )
    return result


def _numeric_tuple(values: object, field: str) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise Foams105VolumeValidationError(f"{field} must be a numeric sequence")
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105VolumeValidationError(
            f"{field} must be a numeric sequence"
        ) from exc
    return tuple(
        _finite_real(value, f"{field}[{index}]")
        for index, value in enumerate(raw)
    )


def _numerical_error(
    code: str,
    stage: str,
    *,
    row_index: Optional[int] = None,
    operands: Sequence[float] = (),
) -> Foams105VolumeNumericalError:
    return Foams105VolumeNumericalError(
        code,
        f"{stage} produced an invalid float64 result",
        context={
            "stage": stage,
            "row_index": row_index,
            "operands": tuple(operands),
        },
    )


def _checked_multiply(
    left: float,
    right: float,
    *,
    stage: str,
    row_index: Optional[int] = None,
) -> float:
    result = left * right
    if not math.isfinite(result):
        raise _numerical_error(
            "nonfinite_volume_arithmetic",
            stage,
            row_index=row_index,
            operands=(left, right),
        )
    if left != 0.0 and right != 0.0 and result == 0.0:
        raise _numerical_error(
            "nonzero_underflow",
            stage,
            row_index=row_index,
            operands=(left, right),
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
            "nonfinite_volume_arithmetic",
            stage,
            row_index=row_index,
            operands=(numerator, denominator),
        )
    if numerator != 0.0 and result == 0.0:
        raise _numerical_error(
            "nonzero_underflow",
            stage,
            row_index=row_index,
            operands=(numerator, denominator),
        )
    return result


def _checked_sum(
    total: float,
    value: float,
    *,
    stage: str,
    row_index: int,
) -> float:
    result = total + value
    if not math.isfinite(result):
        raise _numerical_error(
            "nonfinite_volume_arithmetic",
            stage,
            row_index=row_index,
            operands=(total, value),
        )
    return result


def _calculate_foams105_volume_numerical(
    mean_projected_heights_mm: Sequence[float],
    signed_nv_per_mm3: Sequence[float],
    mode: str = "unnormalized",
    *,
    target_vesicularity_percent: Optional[float] = None,
) -> _Foams105VolumeNumericalResult:
    """Apply source-order volume arithmetic to analytical scalar vectors."""
    if mode not in FOAMS105_VOLUME_MODES:
        raise Foams105VolumeValidationError(
            f"mode must be one of {FOAMS105_VOLUME_MODES!r}",
            code="unsupported_mode",
            context={"mode": mode},
        )
    heights = _numeric_tuple(mean_projected_heights_mm, "mean_projected_heights_mm")
    signed_nv = _numeric_tuple(signed_nv_per_mm3, "signed_nv_per_mm3")
    if (
        len(heights) != len(signed_nv)
        or not 1 <= len(heights) <= FOAMS105_MAX_CLASSES
    ):
        raise Foams105VolumeValidationError(
            "height and N_V vectors must have equal lengths from 1 to "
            f"{FOAMS105_MAX_CLASSES}"
        )
    for index, height in enumerate(heights):
        if height <= 0.0:
            raise Foams105VolumeValidationError(
                f"mean_projected_heights_mm[{index}] must be positive"
            )

    if mode == "unnormalized":
        if target_vesicularity_percent is not None:
            raise Foams105VolumeValidationError(
                "unnormalized mode does not accept a target",
                code="unused_target",
            )
        target_percent = None
    else:
        if target_vesicularity_percent is None:
            raise Foams105VolumeValidationError(
                "normalized modes require a target",
                code="missing_target",
            )
        target_percent = _finite_real(
            target_vesicularity_percent, "target_vesicularity_percent"
        )
        if mode == "known_vesicularity" and not 0.0 <= target_percent <= 100.0:
            raise Foams105VolumeValidationError(
                "known vesicularity must be between 0 and 100 percent",
                code="invalid_known_vesicularity",
                context={"target_vesicularity_percent": target_percent},
            )

    sphere_volumes = []
    raw_fractions = []
    raw_sum = 0.0
    diagnostics = []
    for index, (height, nv_value) in enumerate(zip(heights, signed_nv)):
        try:
            height_cubed = height**3
        except OverflowError as exc:
            raise _numerical_error(
                "nonfinite_volume_arithmetic",
                "height_cube",
                row_index=index,
                operands=(height,),
            ) from exc
        if not math.isfinite(height_cubed):
            raise _numerical_error(
                "nonfinite_volume_arithmetic",
                "height_cube",
                row_index=index,
                operands=(height,),
            )
        if height_cubed == 0.0:
            raise _numerical_error(
                "positive_sphere_volume_underflow",
                "height_cube",
                row_index=index,
                operands=(height,),
            )
        pi_product = _checked_multiply(
            height_cubed, math.pi, stage="sphere_volume_pi", row_index=index
        )
        sphere_volume = _checked_divide(
            pi_product, 6.0, stage="sphere_volume_divide_six", row_index=index
        )
        raw_fraction = _checked_multiply(
            sphere_volume,
            nv_value,
            stage="sphere_volume_times_nv",
            row_index=index,
        )
        raw_sum = _checked_sum(
            raw_sum, raw_fraction, stage="raw_fraction_sum", row_index=index
        )
        sphere_volumes.append(sphere_volume)
        raw_fractions.append(raw_fraction)
        if nv_value < 0.0:
            diagnostics.append(
                Foams105VolumeDiagnostic(
                    DIAGNOSTIC_NEGATIVE_NV_ROW,
                    "signed_nv_per_mm3",
                    row_index=index,
                    value=nv_value,
                )
            )
        if raw_fraction < 0.0:
            diagnostics.append(
                Foams105VolumeDiagnostic(
                    DIAGNOSTIC_NEGATIVE_RAW_FRACTION_ROW,
                    "raw_volume_fraction",
                    row_index=index,
                    value=raw_fraction,
                )
            )

    derived_percent = _checked_multiply(
        raw_sum, 100.0, stage="nv_derived_vesicularity_percent"
    )
    if raw_sum < 0.0:
        diagnostics.append(
            Foams105VolumeDiagnostic(
                DIAGNOSTIC_NEGATIVE_RAW_FRACTION_SUM,
                "raw_fraction_sum",
                value=raw_sum,
            )
        )
    if raw_sum > 1.0:
        diagnostics.append(
            Foams105VolumeDiagnostic(
                DIAGNOSTIC_RAW_FRACTION_SUM_ABOVE_ONE,
                "raw_fraction_sum",
                value=raw_sum,
            )
        )

    if mode == "unnormalized":
        target_fraction = None
        factor = 1.0
    else:
        if raw_sum == 0.0:
            raise Foams105VolumeDomainError(
                "zero_raw_volume_sum",
                "normalized volume fractions require a nonzero raw sum",
                context={"mode": mode, "target_vesicularity_percent": target_percent},
            )
        if target_percent is None:
            raise Foams105VolumeValidationError(
                "normalized mode lost its target",
                code="missing_target",
            )
        target_fraction = _checked_divide(
            target_percent, 100.0, stage="target_percent_divide_100"
        )
        factor = _checked_divide(
            target_fraction, raw_sum, stage="normalization_factor"
        )
        if not 0.0 <= target_percent <= 100.0:
            diagnostics.append(
                Foams105VolumeDiagnostic(
                    DIAGNOSTIC_TARGET_PERCENT_OUT_OF_RANGE,
                    "target_vesicularity_percent",
                    value=target_percent,
                )
            )
    if factor < 0.0:
        diagnostics.append(
            Foams105VolumeDiagnostic(
                DIAGNOSTIC_NEGATIVE_NORMALIZATION_FACTOR,
                "normalization_factor",
                value=factor,
            )
        )
    if abs(factor) > 1.0:
        diagnostics.append(
            Foams105VolumeDiagnostic(
                DIAGNOSTIC_NORMALIZATION_FACTOR_MAGNITUDE_ABOVE_ONE,
                "normalization_factor",
                value=factor,
            )
        )

    adjusted = []
    cumulative = []
    running = 0.0
    for index, raw_fraction in enumerate(raw_fractions):
        adjusted_fraction = _checked_multiply(
            raw_fraction,
            factor,
            stage="adjusted_volume_fraction",
            row_index=index,
        )
        previous = running
        running = _checked_sum(
            running,
            adjusted_fraction,
            stage="cumulative_adjusted_volume_fraction",
            row_index=index,
        )
        adjusted.append(adjusted_fraction)
        cumulative.append(running)
        if adjusted_fraction < 0.0:
            diagnostics.append(
                Foams105VolumeDiagnostic(
                    DIAGNOSTIC_NEGATIVE_ADJUSTED_FRACTION_ROW,
                    "adjusted_volume_fraction",
                    row_index=index,
                    value=adjusted_fraction,
                )
            )
        if running < previous:
            diagnostics.append(
                Foams105VolumeDiagnostic(
                    DIAGNOSTIC_CUMULATIVE_DECREASE,
                    "cumulative_adjusted_volume_fraction",
                    row_index=index,
                    previous_row_index=None if index == 0 else index - 1,
                    value=running,
                    previous_value=previous,
                )
            )

    residual = None
    if target_fraction is not None:
        residual = running - target_fraction
        if not math.isfinite(residual):
            raise _numerical_error(
                "nonfinite_volume_arithmetic",
                "normalization_residual",
                operands=(running, target_fraction),
            )

    return _Foams105VolumeNumericalResult(
        sphere_volumes_mm3=tuple(sphere_volumes),
        raw_volume_fractions=tuple(raw_fractions),
        raw_fraction_sum=raw_sum,
        nv_derived_vesicularity_percent=derived_percent,
        target_volume_fraction=target_fraction,
        normalization_factor=factor,
        adjusted_volume_fractions=tuple(adjusted),
        cumulative_adjusted_volume_fractions=tuple(cumulative),
        adjusted_fraction_sum=running,
        normalization_residual=residual,
        diagnostics=tuple(diagnostics),
    )


def calculate_foams105_volume_distribution(
    conversion_result: Foams105ConversionResult,
    mode: str = "unnormalized",
    *,
    vesicularity_result: Optional[Foams105VesicularityResult] = None,
    known_vesicularity_percent: Optional[float] = None,
) -> Foams105VolumeDistributionResult:
    """Replay original equivalent-sphere volume fractions and normalization."""
    if not isinstance(conversion_result, Foams105ConversionResult):
        raise Foams105VolumeValidationError(
            "conversion_result must be a Foams105ConversionResult"
        )
    try:
        canonical_conversion = convert_foams105_nv(
            conversion_result.bin_labels_mm,
            conversion_result.na_per_mm2,
            length_unit=conversion_result.length_unit,
            input_density_unit=conversion_result.input_density_unit,
        )
    except (ValueError, ArithmeticError) as exc:
        raise Foams105VolumeValidationError(
            "conversion_result cannot be replayed from its inputs",
            code="inconsistent_conversion_result",
        ) from exc
    if canonical_conversion != conversion_result:
        raise Foams105VolumeValidationError(
            "conversion_result differs from its canonical replay",
            code="inconsistent_conversion_result",
        )
    if not isinstance(mode, str) or mode not in FOAMS105_VOLUME_MODES:
        raise Foams105VolumeValidationError(
            f"mode must be one of {FOAMS105_VOLUME_MODES!r}",
            code="unsupported_mode",
            context={"mode": mode},
        )

    canonical_vesicularity = None
    if mode == "unnormalized":
        if vesicularity_result is not None or known_vesicularity_percent is not None:
            raise Foams105VolumeValidationError(
                "unnormalized mode accepts no vesicularity target",
                code="unused_target",
            )
        target_source = None
        target_percent = None
    elif mode == "integrated_2d":
        if known_vesicularity_percent is not None:
            raise Foams105VolumeValidationError(
                "integrated_2d mode does not accept a known target",
                code="unused_target",
            )
        if not isinstance(vesicularity_result, Foams105VesicularityResult):
            raise Foams105VolumeValidationError(
                "integrated_2d mode requires a Foams105VesicularityResult",
                code="missing_target",
            )
        try:
            canonical_vesicularity = calculate_foams105_2d_vesicularity(
                vesicularity_result.raw_selection,
                vesicularity_result.groups,
            )
        except (ValueError, ArithmeticError) as exc:
            raise Foams105VolumeValidationError(
                "vesicularity_result cannot be replayed from its inputs",
                code="inconsistent_vesicularity_result",
            ) from exc
        if canonical_vesicularity != vesicularity_result:
            raise Foams105VolumeValidationError(
                "vesicularity_result differs from its canonical replay",
                code="inconsistent_vesicularity_result",
            )
        target_source = "integrated_2d_vesicularity_percent"
        target_percent = canonical_vesicularity.integrated_vesicularity_percent
    else:
        if vesicularity_result is not None:
            raise Foams105VolumeValidationError(
                "known_vesicularity mode does not accept a 2D result",
                code="unused_target",
            )
        if known_vesicularity_percent is None:
            raise Foams105VolumeValidationError(
                "known_vesicularity mode requires a percentage",
                code="missing_target",
            )
        target_percent = _finite_real(
            known_vesicularity_percent, "known_vesicularity_percent"
        )
        if not 0.0 <= target_percent <= 100.0:
            raise Foams105VolumeValidationError(
                "known_vesicularity_percent must be between 0 and 100",
                code="invalid_known_vesicularity",
                context={"known_vesicularity_percent": target_percent},
            )
        target_source = "caller_known_vesicularity_percent"

    numerical = _calculate_foams105_volume_numerical(
        canonical_conversion.mean_projected_heights_mm,
        canonical_conversion.signed_nv_per_mm3,
        mode,
        target_vesicularity_percent=target_percent,
    )
    rows = tuple(
        Foams105VolumeRow(
            output_index=index,
            bin_label_mm=label,
            mean_projected_height_mm=height,
            signed_nv_per_mm3=nv_value,
            sphere_volume_mm3=numerical.sphere_volumes_mm3[index],
            raw_volume_fraction=numerical.raw_volume_fractions[index],
            adjusted_volume_fraction=numerical.adjusted_volume_fractions[index],
            cumulative_adjusted_volume_fraction=(
                numerical.cumulative_adjusted_volume_fractions[index]
            ),
        )
        for index, (label, height, nv_value) in enumerate(
            zip(
                canonical_conversion.bin_labels_mm,
                canonical_conversion.mean_projected_heights_mm,
                canonical_conversion.signed_nv_per_mm3,
            )
        )
    )
    return Foams105VolumeDistributionResult(
        conversion_result=canonical_conversion,
        vesicularity_result=canonical_vesicularity,
        mode=mode,
        target_source=target_source,
        target_vesicularity_percent=target_percent,
        target_volume_fraction=numerical.target_volume_fraction,
        normalization_factor=numerical.normalization_factor,
        rows=rows,
        sphere_volumes_mm3=numerical.sphere_volumes_mm3,
        raw_volume_fractions=numerical.raw_volume_fractions,
        adjusted_volume_fractions=numerical.adjusted_volume_fractions,
        cumulative_adjusted_volume_fractions=(
            numerical.cumulative_adjusted_volume_fractions
        ),
        raw_fraction_sum=numerical.raw_fraction_sum,
        nv_derived_vesicularity_percent=(
            numerical.nv_derived_vesicularity_percent
        ),
        adjusted_fraction_sum=numerical.adjusted_fraction_sum,
        normalization_residual=numerical.normalization_residual,
        adjacent_duplicate_label_index_pairs=(
            canonical_conversion.adjacent_duplicate_label_index_pairs
        ),
        converter_negative_indices=canonical_conversion.negative_indices,
        diagnostics=numerical.diagnostics,
        upstream_vesicularity_diagnostics=(
            ()
            if canonical_vesicularity is None
            else canonical_vesicularity.diagnostics
        ),
    )
