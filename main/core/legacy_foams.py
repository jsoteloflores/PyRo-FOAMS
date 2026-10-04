"""FOAMS 1.0.5 post-nesting ``N_A``-to-``N_V`` compatibility replay.

This module reproduces the conversion arithmetic in ``analysis.m`` from the
pinned original source. It does not reproduce original histogram preparation,
area correction, automatic nesting, measurements, or the complete workflow.
Repeated adjacent labels are retained because they occur in the authoritative
workbook fixture; their acquisition cause has not been reconstructed.
Descending labels remain invalid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

FOAMS105_METHOD = "foams_1_0_5_alpha_replay_v1"
FOAMS105_SOURCE_REPOSITORY = "jsoteloflores/FOAMS-1.0.5"
FOAMS105_SOURCE_COMMIT = "179663203f2d0f86b2863d5ea7f8f70dadca02f8"
FOAMS105_SOURCE_FILE = "analysis.m"
FOAMS105_SMALLEST_CLASS_POLICY = "legacy_no_larger_class_subtraction"
FOAMS105_HEIGHT_POLICY = "cube_volume_midpoint_with_zero_first_lower"
FOAMS105_SCOPE = "post_nesting_na_to_nv_only"
FOAMS105_COEFFICIENT_LOG10_STEP = 0.1
FOAMS105_MAX_CLASSES = 45


class Foams105ValidationError(ValueError):
    """Raised when inputs do not satisfy the bounded legacy replay profile."""


class Foams105NumericalError(ArithmeticError):
    """Raised when valid inputs produce a nonfinite numerical intermediate."""


@dataclass(frozen=True)
class Foams105ConversionResult:
    """Immutable source-replay result with coefficients and provenance."""

    bin_labels_mm: Tuple[float, ...]
    na_per_mm2: Tuple[float, ...]
    probabilities: Tuple[float, ...]
    alpha_coefficients: Tuple[float, ...]
    mean_projected_heights_mm: Tuple[float, ...]
    larger_contributions_per_mm2: Tuple[float, ...]
    signed_nv_per_mm3: Tuple[float, ...]
    negative_indices: Tuple[int, ...]
    adjacent_duplicate_label_index_pairs: Tuple[Tuple[int, int], ...]
    method: str = FOAMS105_METHOD
    source_repository: str = FOAMS105_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_SOURCE_COMMIT
    conversion_filename: str = FOAMS105_SOURCE_FILE
    smallest_class_policy: str = FOAMS105_SMALLEST_CLASS_POLICY
    height_policy: str = FOAMS105_HEIGHT_POLICY
    coefficient_log10_step: float = FOAMS105_COEFFICIENT_LOG10_STEP
    scope: str = FOAMS105_SCOPE
    length_unit: str = "mm"
    input_density_unit: str = "mm^-2"
    output_density_unit: str = "mm^-3"


def _numeric_tuple(values: object, field: str) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise Foams105ValidationError(f"{field} must be a numeric sequence")
    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise Foams105ValidationError(f"{field} must be a numeric sequence") from exc

    converted = []
    for index, value in enumerate(raw_values):
        if (
            not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_))
        ):
            raise Foams105ValidationError(
                f"{field}[{index}] must be a finite real scalar"
            )
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise Foams105ValidationError(
                f"{field}[{index}] must be representable as a finite float64"
            ) from exc
        if not math.isfinite(numeric):
            raise Foams105ValidationError(f"{field}[{index}] must be finite")
        converted.append(numeric)
    return tuple(converted)


def _probability_sequence(number_of_classes: int) -> Tuple[float, ...]:
    ratio_step = 10.0 ** -FOAMS105_COEFFICIENT_LOG10_STEP
    previous_ratio = 1.0
    previous_root = 0.0
    probabilities = []
    for lag in range(number_of_classes):
        next_ratio = previous_ratio * ratio_step
        radicand = 1.0 - next_ratio * next_ratio
        if not math.isfinite(radicand) or radicand < 0.0:
            raise Foams105NumericalError(
                f"Invalid probability root at lag {lag}"
            )
        next_root = math.sqrt(radicand)
        probability = next_root - previous_root
        if not math.isfinite(probability) or probability <= 0.0:
            raise Foams105NumericalError(
                f"Probability at lag {lag} is not positive and finite"
            )
        probabilities.append(probability)
        previous_ratio = next_ratio
        previous_root = next_root
    return tuple(probabilities)


def _alpha_sequence(probabilities: Tuple[float, ...]) -> Tuple[float, ...]:
    first_probability = probabilities[0]
    first_alpha = 1.0 / first_probability
    if not math.isfinite(first_alpha):
        raise Foams105NumericalError("Alpha coefficient at lag 0 is nonfinite")
    alpha = [first_alpha]
    for lag in range(1, len(probabilities)):
        convolution = 0.0
        for index in range(1, lag):
            product = alpha[index] * probabilities[lag - index]
            if not math.isfinite(product):
                raise Foams105NumericalError(
                    f"Alpha recurrence product is nonfinite at lag {lag}, index {index}"
                )
            convolution += product
            if not math.isfinite(convolution):
                raise Foams105NumericalError(
                    f"Alpha recurrence sum is nonfinite at lag {lag}"
                )
        leading = first_alpha * probabilities[lag]
        value = (leading - convolution) / first_probability
        if not math.isfinite(leading) or not math.isfinite(value):
            raise Foams105NumericalError(
                f"Alpha coefficient at lag {lag} is nonfinite"
            )
        alpha.append(value)
    return tuple(alpha)


def _mean_projected_heights(labels: Tuple[float, ...]) -> Tuple[float, ...]:
    cube_root_half = 0.5 ** (1.0 / 3.0)
    heights = []
    for index, label in enumerate(labels):
        if index == 0:
            height = label * cube_root_half
        else:
            ratio = labels[index - 1] / label
            height = label * (((1.0 + ratio * ratio * ratio) / 2.0) ** (1.0 / 3.0))
        if not math.isfinite(height) or height <= 0.0:
            raise Foams105NumericalError(
                f"Mean projected height at index {index} is not positive and finite"
            )
        heights.append(height)
    return tuple(heights)


def convert_foams105_nv(
    bin_labels_mm: Sequence[float],
    na_per_mm2: Sequence[float],
    *,
    length_unit: str = "mm",
    input_density_unit: str = "mm^-2",
) -> Foams105ConversionResult:
    """Replay the FOAMS 1.0.5 post-nesting conversion for supplied labels.

    Labels must be positive and nondecreasing. Equal adjacent labels are kept
    and reported without assigning an acquisition cause. This function does
    not certify complete original-workflow equivalence.
    """
    if length_unit != "mm":
        raise Foams105ValidationError("length_unit must be 'mm'")
    if input_density_unit != "mm^-2":
        raise Foams105ValidationError("input_density_unit must be 'mm^-2'")

    labels = _numeric_tuple(bin_labels_mm, "bin_labels_mm")
    densities = _numeric_tuple(na_per_mm2, "na_per_mm2")
    if len(labels) != len(densities):
        raise Foams105ValidationError(
            "bin_labels_mm and na_per_mm2 must have equal lengths"
        )
    if not 1 <= len(labels) <= FOAMS105_MAX_CLASSES:
        raise Foams105ValidationError(
            f"input length must be between 1 and {FOAMS105_MAX_CLASSES}"
        )
    for index, label in enumerate(labels):
        if label <= 0.0:
            raise Foams105ValidationError(
                f"bin_labels_mm[{index}] must be positive"
            )
        if index and label < labels[index - 1]:
            raise Foams105ValidationError(
                f"bin_labels_mm must be nondecreasing; index {index} descends"
            )
    for index, density in enumerate(densities):
        if density < 0.0:
            raise Foams105ValidationError(
                f"na_per_mm2[{index}] must be nonnegative"
            )

    probabilities = _probability_sequence(len(labels))
    alpha = _alpha_sequence(probabilities)
    heights = _mean_projected_heights(labels)
    larger_contributions = [0.0] * len(labels)
    signed_nv = []
    for index in range(len(labels)):
        contribution = 0.0
        if index > 0:
            for larger_index in range(index + 1, len(labels)):
                lag = larger_index - index
                product = alpha[lag] * densities[larger_index]
                if not math.isfinite(product):
                    raise Foams105NumericalError(
                        f"Larger-class product is nonfinite at index {index}, "
                        f"larger index {larger_index}"
                    )
                contribution += product
                if not math.isfinite(contribution):
                    raise Foams105NumericalError(
                        f"Larger-class sum is nonfinite at index {index}"
                    )
        leading = alpha[0] * densities[index]
        numerator = leading - contribution
        value = numerator / heights[index]
        if not all(math.isfinite(item) for item in (leading, numerator, value)):
            raise Foams105NumericalError(
                f"Converted N_V is nonfinite at index {index}"
            )
        larger_contributions[index] = contribution
        signed_nv.append(value)

    signed_values = tuple(signed_nv)
    return Foams105ConversionResult(
        bin_labels_mm=labels,
        na_per_mm2=densities,
        probabilities=probabilities,
        alpha_coefficients=alpha,
        mean_projected_heights_mm=heights,
        larger_contributions_per_mm2=tuple(larger_contributions),
        signed_nv_per_mm3=signed_values,
        negative_indices=tuple(
            index for index, value in enumerate(signed_values) if value < 0.0
        ),
        adjacent_duplicate_label_index_pairs=tuple(
            (index - 1, index)
            for index in range(1, len(labels))
            if labels[index - 1] == labels[index]
        ),
    )
