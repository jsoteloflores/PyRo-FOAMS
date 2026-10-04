"""Spherical thin-section forward model and signed triangular reconstruction.

Each 3D class uses its measured bin's upper edge as a representative sphere
diameter. Inputs and outputs are class-integrated number densities, not values
per unit diameter. This PyRo-FOAMS discretization has not been benchmarked
against the original FOAMS implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .distributions import (
    DiameterBinSpec,
    DistributionResult,
    create_diameter_bin_spec,
)
from .nesting import (
    Nested2DDistribution,
    NestingPlan,
    _validate_bin_spec,
    nest_2d_distribution,
)

RECONSTRUCTION_METHOD = "spherical_upper_edge_triangular_v1"
COMPATIBILITY_STATUS = "not_benchmarked_against_foams"
UPPER_TAIL_ASSUMPTION = "zero_beyond_selected_upper_edge"
SPHERE_MODEL_ASSUMPTION = "spherical objects"
SECTION_MODEL_ASSUMPTION = "random negligibly thin planar sections"

_MISSING = object()
_FLOAT64_EPSILON = float(np.finfo(np.float64).eps)


class ReconstructionValidationError(ValueError):
    """Raised when a reconstruction input or operator is inconsistent."""


class ReconstructionNumericalError(ArithmeticError):
    """Raised when finite inputs produce a nonfinite numerical intermediate."""


@dataclass(frozen=True)
class SphericalSectionOperator:
    """Immutable upper-triangular spherical sectioning operator."""

    edges_mm: Tuple[float, ...]
    representative_diameters_mm: Tuple[float, ...]
    coefficients_mm: Tuple[Tuple[float, ...], ...]
    length_unit: str = "mm"
    coefficient_unit: str = "mm"
    input_density_unit: str = "mm^-3"
    output_density_unit: str = "mm^-2"
    method: str = RECONSTRUCTION_METHOD
    compatibility_status: str = COMPATIBILITY_STATUS
    sphere_assumption: str = SPHERE_MODEL_ASSUMPTION
    section_assumption: str = SECTION_MODEL_ASSUMPTION


@dataclass(frozen=True)
class SphericalInverseResult:
    """Signed inverse with residuals defined as fitted minus input density."""

    input_na_per_mm2: Tuple[float, ...]
    signed_nv_per_mm3: Tuple[float, ...]
    fitted_na_per_mm2: Tuple[float, ...]
    residuals_per_mm2: Tuple[float, ...]
    maximum_absolute_residual_per_mm2: float
    normalized_infinity_residual: float
    forward_consistency_tolerance_per_mm2: float
    forward_consistency_passed: bool
    negative_local_bin_indices: Tuple[int, ...]
    materially_negative_local_bin_indices: Tuple[int, ...]
    roundoff_negative_local_bin_indices: Tuple[int, ...]
    negative_classification_tolerance_per_mm3: float
    status: str
    upper_tail_assumption: str
    input_density_unit: str = "mm^-2"
    output_density_unit: str = "mm^-3"
    method: str = RECONSTRUCTION_METHOD
    compatibility_status: str = COMPATIBILITY_STATUS


@dataclass(frozen=True)
class SphericalReconstructionResult:
    """Auditable 3D reconstruction retaining the complete selected 2D source."""

    nested_distribution: Nested2DDistribution
    operator: SphericalSectionOperator
    input_na_per_mm2: Tuple[float, ...]
    signed_nv_per_mm3: Tuple[float, ...]
    fitted_na_per_mm2: Tuple[float, ...]
    residuals_per_mm2: Tuple[float, ...]
    maximum_absolute_residual_per_mm2: float
    normalized_infinity_residual: float
    forward_consistency_tolerance_per_mm2: float
    forward_consistency_passed: bool
    negative_absolute_bin_indices: Tuple[int, ...]
    materially_negative_absolute_bin_indices: Tuple[int, ...]
    roundoff_negative_absolute_bin_indices: Tuple[int, ...]
    negative_classification_tolerance_per_mm3: float
    status: str
    upper_tail_assumption: str
    upper_range_truncated: bool
    upper_range_advisory: str | None
    input_density_unit: str = "mm^-2"
    output_density_unit: str = "mm^-3"
    method: str = RECONSTRUCTION_METHOD
    compatibility_status: str = COMPATIBILITY_STATUS
    sphere_assumption: str = SPHERE_MODEL_ASSUMPTION
    section_assumption: str = SECTION_MODEL_ASSUMPTION


def _validated_edges(edges: object) -> Tuple[float, ...]:
    if not isinstance(edges, tuple):
        raise ReconstructionValidationError("operator edges_mm must be an immutable tuple")
    if len(edges) < 2:
        raise ReconstructionValidationError("operator requires at least two edges")
    validated = []
    for index, edge in enumerate(edges):
        if (
            not isinstance(edge, (int, float, np.integer, np.floating))
            or isinstance(edge, (bool, np.bool_))
        ):
            raise ReconstructionValidationError(
                f"operator edge {index} must be a positive finite real number"
            )
        value = float(edge)
        if not math.isfinite(value) or value <= 0:
            raise ReconstructionValidationError(
                f"operator edge {index} must be a positive finite real number"
            )
        validated.append(value)
    if any(right <= left for left, right in zip(validated, validated[1:])):
        raise ReconstructionValidationError("operator edges must be strictly increasing")
    return tuple(validated)


def _compute_coefficients(
    edges: Tuple[float, ...],
) -> Tuple[Tuple[float, ...], ...]:
    number_of_bins = len(edges) - 1
    rows = []
    for row_index in range(number_of_bins):
        row = []
        for column_index in range(number_of_bins):
            if row_index > column_index:
                coefficient = 0.0
            else:
                lower = edges[row_index]
                upper = edges[row_index + 1]
                diameter = edges[column_index + 1]
                lower_ratio = lower / diameter
                lower_radicand = (
                    ((diameter - lower) / diameter) * (1.0 + lower_ratio)
                )
                if not math.isfinite(lower_radicand) or lower_radicand <= 0.0:
                    raise ReconstructionNumericalError(
                        f"Invalid lower root at row {row_index}, column {column_index}"
                    )
                lower_root = math.sqrt(lower_radicand)
                if row_index == column_index:
                    coefficient = diameter * lower_root
                else:
                    upper_ratio = upper / diameter
                    upper_radicand = (
                        ((diameter - upper) / diameter) * (1.0 + upper_ratio)
                    )
                    if not math.isfinite(upper_radicand) or upper_radicand < 0.0:
                        raise ReconstructionNumericalError(
                            f"Invalid upper root at row {row_index}, column {column_index}"
                        )
                    upper_root = math.sqrt(upper_radicand)
                    denominator = lower_root + upper_root
                    if not math.isfinite(denominator) or denominator <= 0.0:
                        raise ReconstructionNumericalError(
                            f"Invalid coefficient denominator at row {row_index}, "
                            f"column {column_index}"
                        )
                    coefficient = (upper - lower) * (
                        (lower_ratio + upper_ratio) / denominator
                    )
            if not math.isfinite(coefficient) or (
                row_index <= column_index and coefficient <= 0.0
            ):
                raise ReconstructionNumericalError(
                    f"Positive coefficient is not representable at row {row_index}, "
                    f"column {column_index}"
                )
            row.append(float(coefficient))
        rows.append(tuple(row))
    return tuple(rows)


def _validate_operator(operator: object) -> SphericalSectionOperator:
    if not isinstance(operator, SphericalSectionOperator):
        raise ReconstructionValidationError(
            "operator must be a SphericalSectionOperator"
        )
    edges = _validated_edges(operator.edges_mm)
    number_of_bins = len(edges) - 1
    expected_diameters = edges[1:]
    if not isinstance(operator.representative_diameters_mm, tuple):
        raise ReconstructionValidationError(
            "operator representative_diameters_mm must be an immutable tuple"
        )
    if len(operator.representative_diameters_mm) != number_of_bins:
        raise ReconstructionValidationError(
            f"operator representative_diameters_mm must contain {number_of_bins} values"
        )
    validated_diameters = []
    for index, diameter in enumerate(operator.representative_diameters_mm):
        if (
            not isinstance(diameter, (int, float, np.integer, np.floating))
            or isinstance(diameter, (bool, np.bool_))
        ):
            raise ReconstructionValidationError(
                f"operator representative_diameters_mm[{index}] must be a "
                "positive finite real number"
            )
        value = float(diameter)
        if not math.isfinite(value) or value <= 0.0:
            raise ReconstructionValidationError(
                f"operator representative_diameters_mm[{index}] must be a "
                "positive finite real number"
            )
        validated_diameters.append(value)
    if tuple(validated_diameters) != expected_diameters:
        raise ReconstructionValidationError(
            "operator representative diameters must equal upper bin edges"
        )
    if (
        not isinstance(operator.coefficients_mm, tuple)
        or len(operator.coefficients_mm) != number_of_bins
        or any(
            not isinstance(row, tuple) or len(row) != number_of_bins
            for row in operator.coefficients_mm
        )
    ):
        raise ReconstructionValidationError(
            f"operator coefficients must be an immutable {number_of_bins}x{number_of_bins} matrix"
        )
    expected_coefficients = _compute_coefficients(edges)
    for row_index, (supplied_row, expected_row) in enumerate(
        zip(operator.coefficients_mm, expected_coefficients)
    ):
        for column_index, (supplied, expected) in enumerate(
            zip(supplied_row, expected_row)
        ):
            if (
                not isinstance(supplied, (int, float, np.integer, np.floating))
                or isinstance(supplied, (bool, np.bool_))
                or not math.isfinite(float(supplied))
            ):
                raise ReconstructionValidationError(
                    f"operator coefficient [{row_index},{column_index}] must be finite"
                )
            if row_index > column_index and float(supplied) != 0.0:
                raise ReconstructionValidationError(
                    f"operator coefficient [{row_index},{column_index}] must be exactly zero"
                )
            if float(supplied) != expected:
                raise ReconstructionValidationError(
                    f"operator coefficient [{row_index},{column_index}] "
                    "does not match spherical geometry"
                )
    expected_metadata = {
        "length_unit": "mm",
        "coefficient_unit": "mm",
        "input_density_unit": "mm^-3",
        "output_density_unit": "mm^-2",
        "method": RECONSTRUCTION_METHOD,
        "compatibility_status": COMPATIBILITY_STATUS,
        "sphere_assumption": SPHERE_MODEL_ASSUMPTION,
        "section_assumption": SECTION_MODEL_ASSUMPTION,
    }
    for field, expected in expected_metadata.items():
        if getattr(operator, field) != expected:
            raise ReconstructionValidationError(
                f"operator {field} must be {expected!r}"
            )
    return operator


def build_spherical_section_operator(
    bin_spec: DiameterBinSpec,
) -> SphericalSectionOperator:
    """Build the finite-grid spherical section operator in millimeters."""
    try:
        validated_spec = _validate_bin_spec(bin_spec)
    except ValueError as exc:
        raise ReconstructionValidationError(f"Invalid operator bin grid: {exc}") from exc
    edges = tuple(float(value) for value in validated_spec.edges_mm)
    operator = SphericalSectionOperator(
        edges_mm=edges,
        representative_diameters_mm=edges[1:],
        coefficients_mm=_compute_coefficients(edges),
    )
    return _validate_operator(operator)


def _validated_vector(
    values: object,
    expected_length: int,
    field: str,
    *,
    nonnegative: bool,
) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ReconstructionValidationError(f"{field} must be a numeric sequence")
    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ReconstructionValidationError(f"{field} must be a numeric sequence") from exc
    if len(raw_values) != expected_length:
        raise ReconstructionValidationError(
            f"{field} length {len(raw_values)} does not match operator size {expected_length}"
        )
    result = []
    for index, value in enumerate(raw_values):
        if (
            not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_))
        ):
            raise ReconstructionValidationError(
                f"{field}[{index}] must be a finite real number"
            )
        numeric = float(value)
        if not math.isfinite(numeric) or (nonnegative and numeric < 0):
            qualifier = "finite and nonnegative" if nonnegative else "finite"
            raise ReconstructionValidationError(
                f"{field}[{index}] must be {qualifier}"
            )
        result.append(numeric)
    return tuple(result)


def _signed_project(
    operator: SphericalSectionOperator,
    values: Tuple[float, ...],
) -> Tuple[float, ...]:
    projected = []
    for row_index, row in enumerate(operator.coefficients_mm):
        products = []
        for column_index in range(row_index, len(values)):
            product = row[column_index] * values[column_index]
            if not math.isfinite(product):
                raise ReconstructionNumericalError(
                    f"Nonfinite forward product at row {row_index}, "
                    f"column {column_index}"
                )
            products.append(product)
        try:
            total = math.fsum(products)
        except OverflowError as exc:
            raise ReconstructionNumericalError(
                f"Forward sum overflowed at row {row_index}"
            ) from exc
        if not math.isfinite(total):
            raise ReconstructionNumericalError(
                f"Forward sum is nonfinite at row {row_index}"
            )
        projected.append(total)
    return tuple(projected)


def project_spherical_number_densities(
    operator: SphericalSectionOperator,
    nv_per_mm3: Sequence[float],
) -> Tuple[float, ...]:
    """Project finite nonnegative class-integrated ``N_V`` to ``N_A``."""
    validated_operator = _validate_operator(operator)
    values = _validated_vector(
        nv_per_mm3,
        len(validated_operator.representative_diameters_mm),
        "nv_per_mm3",
        nonnegative=True,
    )
    return _signed_project(validated_operator, values)


def _validated_upper_tail_assumption(value: object) -> str:
    if value is _MISSING:
        raise ReconstructionValidationError(
            "upper_tail_assumption must be supplied explicitly"
        )
    if value != UPPER_TAIL_ASSUMPTION:
        raise ReconstructionValidationError(
            f"upper_tail_assumption must be {UPPER_TAIL_ASSUMPTION!r}"
        )
    return UPPER_TAIL_ASSUMPTION


def solve_spherical_number_densities(
    operator: SphericalSectionOperator,
    na_per_mm2: Sequence[float],
    *,
    upper_tail_assumption: object = _MISSING,
) -> SphericalInverseResult:
    """Solve signed class-integrated ``N_V`` by upper-triangular substitution."""
    validated_operator = _validate_operator(operator)
    assumption = _validated_upper_tail_assumption(upper_tail_assumption)
    number_of_bins = len(validated_operator.representative_diameters_mm)
    input_values = _validated_vector(
        na_per_mm2, number_of_bins, "na_per_mm2", nonnegative=True
    )
    solution = [0.0] * number_of_bins
    for row_index in range(number_of_bins - 1, -1, -1):
        products = []
        for column_index in range(row_index + 1, number_of_bins):
            product = (
                validated_operator.coefficients_mm[row_index][column_index]
                * solution[column_index]
            )
            if not math.isfinite(product):
                raise ReconstructionNumericalError(
                    f"Nonfinite inverse product at row {row_index}, "
                    f"column {column_index}"
                )
            products.append(product)
        try:
            upper_contribution = math.fsum(products)
        except OverflowError as exc:
            raise ReconstructionNumericalError(
                f"Inverse sum overflowed at row {row_index}"
            ) from exc
        numerator = input_values[row_index] - upper_contribution
        solved = numerator / validated_operator.coefficients_mm[row_index][row_index]
        if not math.isfinite(numerator) or not math.isfinite(solved):
            raise ReconstructionNumericalError(
                f"Solved value is nonfinite at row {row_index}"
            )
        solution[row_index] = solved

    signed_solution = tuple(solution)
    fitted = _signed_project(validated_operator, signed_solution)
    # Positive residual means the reconstructed model overpredicts the input.
    residuals = tuple(
        fitted_value - input_value
        for fitted_value, input_value in zip(fitted, input_values)
    )
    if any(not math.isfinite(value) for value in residuals):
        raise ReconstructionNumericalError("Forward residual contains a nonfinite value")
    maximum_residual = max((abs(value) for value in residuals), default=0.0)
    input_scale = max((abs(value) for value in input_values), default=0.0)
    if input_scale == 0.0:
        normalized_residual = 0.0 if maximum_residual == 0.0 else math.inf
        consistency_tolerance = 0.0
        consistency_passed = maximum_residual == 0.0
    else:
        normalized_residual = maximum_residual / input_scale
        consistency_tolerance = (
            128.0 * number_of_bins * _FLOAT64_EPSILON * input_scale
        )
        consistency_passed = maximum_residual <= consistency_tolerance

    solution_scale = max((abs(value) for value in signed_solution), default=0.0)
    negative_tolerance = 64.0 * _FLOAT64_EPSILON * solution_scale
    negative = tuple(index for index, value in enumerate(signed_solution) if value < 0.0)
    material = tuple(
        index for index, value in enumerate(signed_solution) if value < -negative_tolerance
    )
    roundoff = tuple(
        index
        for index, value in enumerate(signed_solution)
        if -negative_tolerance <= value < 0.0
    )
    status = (
        "negative_solution"
        if material
        else "roundoff_negative"
        if roundoff
        else "nonnegative"
    )
    return SphericalInverseResult(
        input_na_per_mm2=input_values,
        signed_nv_per_mm3=signed_solution,
        fitted_na_per_mm2=fitted,
        residuals_per_mm2=residuals,
        maximum_absolute_residual_per_mm2=maximum_residual,
        normalized_infinity_residual=normalized_residual,
        forward_consistency_tolerance_per_mm2=consistency_tolerance,
        forward_consistency_passed=consistency_passed,
        negative_local_bin_indices=negative,
        materially_negative_local_bin_indices=material,
        roundoff_negative_local_bin_indices=roundoff,
        negative_classification_tolerance_per_mm3=negative_tolerance,
        status=status,
        upper_tail_assumption=assumption,
    )


def reconstruct_spherical_3d(
    distribution_result: DistributionResult,
    nesting_plan: NestingPlan,
    *,
    upper_tail_assumption: object = _MISSING,
) -> SphericalReconstructionResult:
    """Nest selected ``N_A`` bins once, then reconstruct signed spherical ``N_V``.

    The required tail assumption states that no 3D classes exist above the
    selected upper edge. A truncated selected range is retained as an explicit
    advisory because the source distribution contains higher measured bins.
    """
    assumption = _validated_upper_tail_assumption(upper_tail_assumption)
    nested = nest_2d_distribution(distribution_result, nesting_plan)
    selected_edges = nested.bin_spec.edges_mm[
        nested.plan.start_bin : nested.plan.stop_bin + 1
    ]
    selected_spec = create_diameter_bin_spec(selected_edges)
    operator = build_spherical_section_operator(selected_spec)
    inverse = solve_spherical_number_densities(
        operator,
        nested.number_densities_per_mm2,
        upper_tail_assumption=assumption,
    )

    def absolute_indices(local_indices: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(nested.selected_bin_indices[index] for index in local_indices)

    upper_range_truncated = nested.plan.stop_bin < len(nested.bin_spec.bins)
    advisory = (
        "Selected range excludes higher source bins; reconstruction assumes "
        "zero 3D number density beyond the selected upper edge."
        if upper_range_truncated
        else None
    )
    return SphericalReconstructionResult(
        nested_distribution=nested,
        operator=operator,
        input_na_per_mm2=inverse.input_na_per_mm2,
        signed_nv_per_mm3=inverse.signed_nv_per_mm3,
        fitted_na_per_mm2=inverse.fitted_na_per_mm2,
        residuals_per_mm2=inverse.residuals_per_mm2,
        maximum_absolute_residual_per_mm2=(
            inverse.maximum_absolute_residual_per_mm2
        ),
        normalized_infinity_residual=inverse.normalized_infinity_residual,
        forward_consistency_tolerance_per_mm2=(
            inverse.forward_consistency_tolerance_per_mm2
        ),
        forward_consistency_passed=inverse.forward_consistency_passed,
        negative_absolute_bin_indices=absolute_indices(
            inverse.negative_local_bin_indices
        ),
        materially_negative_absolute_bin_indices=absolute_indices(
            inverse.materially_negative_local_bin_indices
        ),
        roundoff_negative_absolute_bin_indices=absolute_indices(
            inverse.roundoff_negative_local_bin_indices
        ),
        negative_classification_tolerance_per_mm3=(
            inverse.negative_classification_tolerance_per_mm3
        ),
        status=inverse.status,
        upper_tail_assumption=assumption,
        upper_range_truncated=upper_range_truncated,
        upper_range_advisory=advisory,
    )
