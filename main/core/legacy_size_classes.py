"""FOAMS 1.0.5 size-class preparation from supplied measurements.

This module replays label preparation, minimum-diameter filtering, ``histc``
class ownership, and normalization by a caller-supplied corrected area. It
does not calculate image measurements, corrected areas, or nesting cutoffs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

FOAMS105_SIZE_SOURCE_REPOSITORY = "jsoteloflores/FOAMS-1.0.5"
FOAMS105_SIZE_SOURCE_COMMIT = "179663203f2d0f86b2863d5ea7f8f70dadca02f8"
FOAMS105_SIZE_SOURCE_FILE = "project.m"
FOAMS105_LABEL_METHOD = "foams_1_0_5_size_label_preparation_v1"
FOAMS105_FILTER_METHOD = "foams_1_0_5_minimum_diameter_filter_v1"
FOAMS105_HISTOGRAM_METHOD = "foams_1_0_5_histc_trimmed_v1"
FOAMS105_NORMALIZATION_METHOD = "foams_1_0_5_supplied_area_normalization_v1"
FOAMS105_LABEL_COUNT = 45
FOAMS105_LABEL_LOG10_STEP = 0.1
FOAMS105_ROUNDING_POLICY = "nearest_5_decimals_exact_scaled_half_ties_upward"
FOAMS105_ROUNDING_VERIFICATION = (
    "documented_profile_not_historical_runtime_benchmarked"
)
FOAMS105_FIRST_CLASS_POLICY = "diameter_below_first_label"
FOAMS105_EDGE_POLICY = "lower_inclusive_upper_exclusive_histc_trimmed"
FOAMS105_UPPER_TAIL_POLICY = "diameter_at_or_above_last_label_discarded"
FOAMS105_AREA_PROVENANCE = "caller_supplied_not_computed_here"


class Foams105SizeClassValidationError(ValueError):
    """Raised when a size-class replay input violates its contract."""


class Foams105SizeClassNumericalError(ArithmeticError):
    """Raised when valid inputs produce a nonfinite intermediate."""


@dataclass(frozen=True)
class Foams105LabelPreparationResult:
    """Immutable original-style 45-label preparation result."""

    minimum_diameter_px: float
    scales_px_per_mm: Tuple[float, ...]
    raw_bin_labels_mm: Tuple[float, ...]
    bin_labels_mm: Tuple[float, ...]
    adjacent_duplicate_label_index_pairs: Tuple[Tuple[int, int], ...]
    zero_label_indices: Tuple[int, ...]
    method: str = FOAMS105_LABEL_METHOD
    rounding_policy: str = FOAMS105_ROUNDING_POLICY
    rounding_verification: str = FOAMS105_ROUNDING_VERIFICATION
    source_repository: str = FOAMS105_SIZE_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_SIZE_SOURCE_COMMIT
    source_file: str = FOAMS105_SIZE_SOURCE_FILE
    length_unit: str = "mm"
    scale_unit: str = "pixels/mm"


@dataclass(frozen=True)
class Foams105DiameterFilterResult:
    """Immutable minimum-equivalent-diameter filtering result."""

    minimum_diameter_px: float
    scale_px_per_mm: float
    threshold_mm: float
    retained_diameters_mm: Tuple[float, ...]
    retained_indices: Tuple[int, ...]
    rejected_below_threshold_indices: Tuple[int, ...]
    input_count: int
    method: str = FOAMS105_FILTER_METHOD
    threshold_policy: str = "minimum_over_scale_times_10_to_negative_0_1"
    length_unit: str = "mm"
    scale_unit: str = "pixels/mm"


@dataclass(frozen=True)
class Foams105HistogramResult:
    """Immutable retained source rows after ``histc`` upper-tail trimming."""

    bin_labels_mm: Tuple[float, ...]
    counts: Tuple[int, ...]
    input_count: int
    retained_count: int
    discarded_upper_count: int
    adjacent_duplicate_label_index_pairs: Tuple[Tuple[int, int], ...]
    method: str = FOAMS105_HISTOGRAM_METHOD
    first_class_policy: str = FOAMS105_FIRST_CLASS_POLICY
    edge_policy: str = FOAMS105_EDGE_POLICY
    upper_tail_policy: str = FOAMS105_UPPER_TAIL_POLICY
    source_repository: str = FOAMS105_SIZE_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_SIZE_SOURCE_COMMIT
    source_file: str = FOAMS105_SIZE_SOURCE_FILE
    length_unit: str = "mm"


@dataclass(frozen=True)
class Foams105NormalizationResult:
    """Immutable division of source-style counts by supplied corrected area."""

    bin_labels_mm: Tuple[float, ...]
    counts: Tuple[int, ...]
    corrected_area_mm2: float
    na_per_mm2: Tuple[float, ...]
    input_count: int
    retained_count: int
    discarded_upper_count: int
    adjacent_duplicate_label_index_pairs: Tuple[Tuple[int, int], ...]
    method: str = FOAMS105_NORMALIZATION_METHOD
    histogram_method: str = FOAMS105_HISTOGRAM_METHOD
    first_class_policy: str = FOAMS105_FIRST_CLASS_POLICY
    edge_policy: str = FOAMS105_EDGE_POLICY
    upper_tail_policy: str = FOAMS105_UPPER_TAIL_POLICY
    area_provenance: str = FOAMS105_AREA_PROVENANCE
    area_unit: str = "mm^2"
    output_density_unit: str = "mm^-2"


def _numeric_scalar(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise Foams105SizeClassValidationError(
            f"{field} must be a finite real scalar"
        )
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as exc:
        raise Foams105SizeClassValidationError(
            f"{field} must be representable as a finite float64"
        ) from exc
    if not math.isfinite(numeric):
        raise Foams105SizeClassValidationError(f"{field} must be finite")
    return numeric


def _numeric_tuple(values: object, field: str) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise Foams105SizeClassValidationError(
            f"{field} must be a numeric sequence"
        )
    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105SizeClassValidationError(
            f"{field} must be a numeric sequence"
        ) from exc
    return tuple(
        _numeric_scalar(value, f"{field}[{index}]")
        for index, value in enumerate(raw_values)
    )


def _duplicate_pairs(values: Tuple[float, ...]) -> Tuple[Tuple[int, int], ...]:
    return tuple(
        (index - 1, index)
        for index in range(1, len(values))
        if values[index - 1] == values[index]
    )


def _round_nonnegative_5_decimals(value: float, index: int) -> float:
    scaled = value * 100000.0
    if not math.isfinite(scaled):
        raise Foams105SizeClassNumericalError(
            f"Label rounding scale is nonfinite at index {index}"
        )
    lower = math.floor(scaled)
    rounded_integer = lower + (1 if scaled - lower >= 0.5 else 0)
    rounded = rounded_integer / 100000.0
    if not math.isfinite(rounded):
        raise Foams105SizeClassNumericalError(
            f"Rounded label is nonfinite at index {index}"
        )
    return rounded


def build_foams105_bin_labels(
    minimum_diameter_px: float,
    scales_px_per_mm: Sequence[float],
) -> Foams105LabelPreparationResult:
    """Build 45 original-style labels using unrounded repeated multiplication."""
    minimum = _numeric_scalar(minimum_diameter_px, "minimum_diameter_px")
    scales = _numeric_tuple(scales_px_per_mm, "scales_px_per_mm")
    if minimum <= 0.0:
        raise Foams105SizeClassValidationError(
            "minimum_diameter_px must be positive"
        )
    if not 1 <= len(scales) <= 4:
        raise Foams105SizeClassValidationError(
            "scales_px_per_mm must contain between 1 and 4 values"
        )
    for index, scale in enumerate(scales):
        if scale <= 0.0:
            raise Foams105SizeClassValidationError(
                f"scales_px_per_mm[{index}] must be positive"
            )

    first = minimum / max(scales)
    if not math.isfinite(first):
        raise Foams105SizeClassNumericalError(
            "Raw label is nonfinite at index 0"
        )
    raw_labels = [first]
    multiplier = 10.0 ** FOAMS105_LABEL_LOG10_STEP
    for index in range(1, FOAMS105_LABEL_COUNT):
        next_label = raw_labels[-1] * multiplier
        if not math.isfinite(next_label):
            raise Foams105SizeClassNumericalError(
                f"Raw label recurrence is nonfinite at index {index}"
            )
        raw_labels.append(next_label)

    raw = tuple(raw_labels)
    labels = tuple(
        _round_nonnegative_5_decimals(value, index)
        for index, value in enumerate(raw)
    )
    return Foams105LabelPreparationResult(
        minimum_diameter_px=minimum,
        scales_px_per_mm=scales,
        raw_bin_labels_mm=raw,
        bin_labels_mm=labels,
        adjacent_duplicate_label_index_pairs=_duplicate_pairs(labels),
        zero_label_indices=tuple(
            index for index, label in enumerate(labels) if label == 0.0
        ),
    )


def filter_foams105_diameters(
    diameters_mm: Sequence[float],
    minimum_diameter_px: float,
    scale_px_per_mm: float,
) -> Foams105DiameterFilterResult:
    """Filter one group at the unrounded original equivalent-diameter threshold."""
    diameters = _numeric_tuple(diameters_mm, "diameters_mm")
    minimum = _numeric_scalar(minimum_diameter_px, "minimum_diameter_px")
    scale = _numeric_scalar(scale_px_per_mm, "scale_px_per_mm")
    if minimum <= 0.0:
        raise Foams105SizeClassValidationError(
            "minimum_diameter_px must be positive"
        )
    if scale <= 0.0:
        raise Foams105SizeClassValidationError("scale_px_per_mm must be positive")
    for index, diameter in enumerate(diameters):
        if diameter <= 0.0:
            raise Foams105SizeClassValidationError(
                f"diameters_mm[{index}] must be positive"
            )

    threshold = (minimum / scale) * (10.0 ** -FOAMS105_LABEL_LOG10_STEP)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise Foams105SizeClassNumericalError(
            "Minimum-diameter threshold must be positive and finite"
        )
    retained_indices = tuple(
        index for index, diameter in enumerate(diameters) if diameter >= threshold
    )
    rejected_indices = tuple(
        index for index, diameter in enumerate(diameters) if diameter < threshold
    )
    return Foams105DiameterFilterResult(
        minimum_diameter_px=minimum,
        scale_px_per_mm=scale,
        threshold_mm=threshold,
        retained_diameters_mm=tuple(diameters[index] for index in retained_indices),
        retained_indices=retained_indices,
        rejected_below_threshold_indices=rejected_indices,
        input_count=len(diameters),
    )


def count_foams105_histogram(
    bin_labels_mm: Sequence[float],
    diameters_mm: Sequence[float],
) -> Foams105HistogramResult:
    """Count already-filtered diameters with source ``histc`` ownership."""
    labels = _numeric_tuple(bin_labels_mm, "bin_labels_mm")
    diameters = _numeric_tuple(diameters_mm, "diameters_mm")
    if not 1 <= len(labels) <= FOAMS105_LABEL_COUNT:
        raise Foams105SizeClassValidationError(
            f"bin_labels_mm must contain between 1 and {FOAMS105_LABEL_COUNT} values"
        )
    for index, label in enumerate(labels):
        if label < 0.0:
            raise Foams105SizeClassValidationError(
                f"bin_labels_mm[{index}] must be nonnegative"
            )
        if index and label < labels[index - 1]:
            raise Foams105SizeClassValidationError(
                f"bin_labels_mm must be nondecreasing; index {index} descends"
            )
    for index, diameter in enumerate(diameters):
        if diameter <= 0.0:
            raise Foams105SizeClassValidationError(
                f"diameters_mm[{index}] must be positive"
            )

    ownership = np.searchsorted(
        np.asarray(labels, dtype=np.float64),
        np.asarray(diameters, dtype=np.float64),
        side="right",
    )
    all_counts = np.bincount(ownership, minlength=len(labels) + 1)
    counts = tuple(int(value) for value in all_counts[: len(labels)])
    discarded_upper_count = int(all_counts[len(labels)])
    retained_count = sum(counts)
    if retained_count + discarded_upper_count != len(diameters):
        raise Foams105SizeClassNumericalError(
            "Histogram conservation failed for supplied diameters"
        )
    return Foams105HistogramResult(
        bin_labels_mm=labels,
        counts=counts,
        input_count=len(diameters),
        retained_count=retained_count,
        discarded_upper_count=discarded_upper_count,
        adjacent_duplicate_label_index_pairs=_duplicate_pairs(labels),
    )


def normalize_foams105_counts(
    histogram_result: Foams105HistogramResult,
    corrected_area_mm2: float,
) -> Foams105NormalizationResult:
    """Divide source-style histogram counts by a supplied corrected area."""
    if not isinstance(histogram_result, Foams105HistogramResult):
        raise Foams105SizeClassValidationError(
            "histogram_result must be a Foams105HistogramResult"
        )
    area = _numeric_scalar(corrected_area_mm2, "corrected_area_mm2")
    if area <= 0.0:
        raise Foams105SizeClassValidationError(
            "corrected_area_mm2 must be positive"
        )
    densities = tuple(count / area for count in histogram_result.counts)
    for index, density in enumerate(densities):
        if not math.isfinite(density):
            raise Foams105SizeClassNumericalError(
                f"Normalized N_A is nonfinite at index {index}"
            )
    return Foams105NormalizationResult(
        bin_labels_mm=histogram_result.bin_labels_mm,
        counts=histogram_result.counts,
        corrected_area_mm2=area,
        na_per_mm2=densities,
        input_count=histogram_result.input_count,
        retained_count=histogram_result.retained_count,
        discarded_upper_count=histogram_result.discarded_upper_count,
        adjacent_duplicate_label_index_pairs=(
            histogram_result.adjacent_duplicate_label_index_pairs
        ),
    )
