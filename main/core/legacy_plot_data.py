"""FOAMS 1.0.5 plotting data projections from accepted analysis results."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

from .legacy_areas import Foams105GroupAreaResult
from .legacy_cutoffs import Foams105CutoffSuggestionResult
from .legacy_selection import (
    FOAMS105_SELECTION_AUTOSMART_ORIGIN,
    FOAMS105_SELECTION_MANUAL_ORIGIN,
    Foams105BinnedSelectionResult,
    Foams105SelectionDomainError,
    Foams105SelectionValidationError,
    select_foams105_binned_ranges,
    select_foams105_suggested_ranges,
)
from .legacy_shape_statistics import (
    Foams105ShapeStatisticsResult,
    _validate_cached_shape_result,
)
from .legacy_summaries import (
    FOAMS105_MAX_EXACT_COUNT,
    LEGACY_INDETERMINATE,
    LEGACY_NEGATIVE_INFINITY,
    LEGACY_POSITIVE_INFINITY,
    Foams105CountGroupInput,
    Foams105RealValue,
)

FOAMS105_SHAPE_HISTOGRAM_METHOD = "foams_1_0_5_shape_histogram_r2025b_reference_v1"
FOAMS105_HIST_CENTERS = (
    0.0,
    0.05,
    0.1,
    0.15000000000000002,
    0.2,
    0.25,
    0.30000000000000004,
    0.35000000000000003,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.6499999999999999,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
)
FOAMS105_HIST_EDGES = FOAMS105_HIST_CENTERS
FOAMS105_ORIENTATION_HIST_CENTERS = tuple(float(value) for value in range(0, 181, 10))
FOAMS105_SHAPE_HISTOGRAM_FIELDS = (
    "eccentricity",
    "elongation",
    "aspect_ratio",
    "regularity",
    "orientation_project_deg",
)


class Foams105PlotDataValidationError(ValueError):
    """Raised when plot-data inputs are inconsistent with accepted results."""

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


@dataclass(frozen=True)
class Foams105ShapeHistogramSeries:
    """One histogram with exact identities and included/omitted diagnostics."""

    field: str
    method: str
    grid_type: str
    centers_or_edges: Tuple[float, ...]
    counts: Tuple[int, ...]
    population_identities: Tuple[Tuple[str, int], ...]
    included_identities: Tuple[Tuple[str, int], ...]
    omitted_identities: Tuple[Tuple[str, int], ...]
    undefined_identities: Tuple[Tuple[str, int], ...]
    under_range_identities: Tuple[Tuple[str, int], ...]
    over_range_identities: Tuple[Tuple[str, int], ...]
    population_count: int
    included_count: int
    omitted_count: int
    under_range_count: int
    over_range_count: int
    orientation_defined_identities: Tuple[Tuple[str, int], ...] = ()
    orientation_undefined_identities: Tuple[Tuple[str, int], ...] = ()
    diagnostics: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Foams105ShapeHistogramResult:
    """All five source shape histogram arrays from threshold-selected objects."""

    shape_statistics_result: Foams105ShapeStatisticsResult
    population_identities: Tuple[Tuple[str, int], ...]
    eccentricity: Foams105ShapeHistogramSeries
    elongation: Foams105ShapeHistogramSeries
    aspect_ratio: Foams105ShapeHistogramSeries
    regularity: Foams105ShapeHistogramSeries
    orientation_project_deg: Foams105ShapeHistogramSeries
    method: str = FOAMS105_SHAPE_HISTOGRAM_METHOD
    source_repository: str = "jsoteloflores/FOAMS-1.0.5"
    source_commit: str = "179663203f2d0f86b2863d5ea7f8f70dadca02f8"
    source_files: Tuple[str, ...] = ("analysis.m", "miscplots.m")
    scope: str = "shape_histogram_data_only"
    orientation_unit: str = "degree"


@dataclass(frozen=True)
class Foams105BulkNaSlotResult:
    """One source-slot bulk-N_A calculation and its cross-slot provenance."""

    source_slot: int
    group_id: str
    count_source_slot: int
    count_source_group_id: str
    denominator_source_slot: int
    denominator_source_group_id: str
    density_source_indices: Tuple[int, ...]
    numerator: Optional[int]
    denominator_area1_mm2: Optional[float]
    plot_x_area1_mm2: float
    value: Foams105RealValue
    diagnostics: Tuple[str, ...]


@dataclass(frozen=True)
class Foams105BulkNaPlotDataResult:
    """Literal bulk-N_A x/y vectors without repairing source length mismatches."""

    selection_result: Foams105BinnedSelectionResult
    count_groups: Tuple[Foams105CountGroupInput, ...]
    group_area_results: Tuple[Foams105GroupAreaResult, ...]
    slots_descending: Tuple[Foams105BulkNaSlotResult, ...]
    source_x_areas_nonzero_mm2: Tuple[float, ...]
    source_y_values: Tuple[Foams105RealValue, ...]
    plot_ready: bool
    diagnostics: Tuple[str, ...]
    method: str = "foams_1_0_5_bulk_na_plot_data_v1"
    source_file: str = "analysis.m"
    source_commit: str = "179663203f2d0f86b2863d5ea7f8f70dadca02f8"
    area_unit: str = "mm^2"
    density_unit: str = "mm^-2"


def _center_histogram(
    field: str,
    samples: Sequence[Tuple[Tuple[str, int], Optional[float]]],
    centers: Tuple[float, ...],
    *,
    orientation_flags: Optional[Mapping[Tuple[str, int], bool]] = None,
) -> Foams105ShapeHistogramSeries:
    if len(centers) < 2 or any(
        not math.isfinite(value) or (index and value <= centers[index - 1])
        for index, value in enumerate(centers)
    ):
        raise Foams105PlotDataValidationError(
            "center histogram centers must be finite and strictly increasing",
            code="invalid_histogram_grid",
            context={"field": field},
        )

    midpoints = tuple((left + right) / 2.0 for left, right in zip(centers, centers[1:]))
    counts = [0] * len(centers)
    population = []
    included = []
    omitted = []
    undefined = []
    under_range = []
    over_range = []
    defined_orientations = []
    undefined_orientations = []
    for identity, value in samples:
        population.append(identity)
        if orientation_flags is not None:
            (defined_orientations if orientation_flags[identity] else undefined_orientations).append(identity)
        if value is None or not math.isfinite(value):
            omitted.append(identity)
            undefined.append(identity)
            continue
        if value < centers[0]:
            under_range.append(identity)
        elif value > centers[-1]:
            over_range.append(identity)
        # MATLAB hist assigns midpoint ties to the lower center.
        bin_index = bisect.bisect_left(midpoints, value)
        counts[bin_index] += 1
        included.append(identity)

    diagnostics = []
    if undefined:
        diagnostics.append("undefined_values_omitted")
    if under_range:
        diagnostics.append("under_range_values_assigned_to_first_bin")
    if over_range:
        diagnostics.append("over_range_values_assigned_to_last_bin")
    if undefined_orientations:
        diagnostics.append("orientation_undefined_status_retained")
    return Foams105ShapeHistogramSeries(
        field=field,
        method="matlab_hist_centers_r2025b_v1",
        grid_type="centers",
        centers_or_edges=centers,
        counts=tuple(counts),
        population_identities=tuple(population),
        included_identities=tuple(included),
        omitted_identities=tuple(omitted),
        undefined_identities=tuple(undefined),
        under_range_identities=tuple(under_range),
        over_range_identities=tuple(over_range),
        population_count=len(population),
        included_count=len(included),
        omitted_count=len(omitted),
        under_range_count=len(under_range),
        over_range_count=len(over_range),
        orientation_defined_identities=tuple(defined_orientations),
        orientation_undefined_identities=tuple(undefined_orientations),
        diagnostics=tuple(diagnostics),
    )


def _edge_histogram(
    field: str,
    samples: Sequence[Tuple[Tuple[str, int], Optional[float]]],
    edges: Tuple[float, ...],
) -> Foams105ShapeHistogramSeries:
    if len(edges) < 2 or any(
        not math.isfinite(value) or (index and value <= edges[index - 1])
        for index, value in enumerate(edges)
    ):
        raise Foams105PlotDataValidationError(
            "edge histogram edges must be finite and strictly increasing",
            code="invalid_histogram_grid",
            context={"field": field},
        )

    counts = [0] * len(edges)
    population = []
    included = []
    omitted = []
    undefined = []
    under_range = []
    over_range = []
    for identity, value in samples:
        population.append(identity)
        if value is None or not math.isfinite(value):
            omitted.append(identity)
            undefined.append(identity)
            continue
        if value < edges[0]:
            omitted.append(identity)
            under_range.append(identity)
            continue
        if value > edges[-1]:
            omitted.append(identity)
            over_range.append(identity)
            continue
        if value == edges[-1]:
            bin_index = len(edges) - 1
        else:
            # histc uses [edge_i, edge_i+1), with one final exact-edge slot.
            bin_index = bisect.bisect_right(edges, value) - 1
        counts[bin_index] += 1
        included.append(identity)

    diagnostics = []
    if undefined:
        diagnostics.append("undefined_values_omitted")
    if under_range:
        diagnostics.append("under_range_values_omitted")
    if over_range:
        diagnostics.append("over_range_values_omitted")
    return Foams105ShapeHistogramSeries(
        field=field,
        method="matlab_histc_edges_r2025b_v1",
        grid_type="edges",
        centers_or_edges=edges,
        counts=tuple(counts),
        population_identities=tuple(population),
        included_identities=tuple(included),
        omitted_identities=tuple(omitted),
        undefined_identities=tuple(undefined),
        under_range_identities=tuple(under_range),
        over_range_identities=tuple(over_range),
        population_count=len(population),
        included_count=len(included),
        omitted_count=len(omitted),
        under_range_count=len(under_range),
        over_range_count=len(over_range),
        diagnostics=tuple(diagnostics),
    )


def build_foams105_shape_histograms(
    shape_statistics_result: Foams105ShapeStatisticsResult,
) -> Foams105ShapeHistogramResult:
    """Build source ``hist``/``histc`` arrays from threshold identities."""

    if not isinstance(shape_statistics_result, Foams105ShapeStatisticsResult):
        raise Foams105PlotDataValidationError(
            "shape_statistics_result must be a Foams105ShapeStatisticsResult",
            code="invalid_shape_statistics_result",
        )
    shape_result = shape_statistics_result.shape_result
    try:
        _validate_cached_shape_result(shape_result)
    except ValueError as exc:
        raise Foams105PlotDataValidationError(
            "shape_statistics_result contains an invalid cached shape result",
            code="invalid_shape_statistics_result",
            context={"cause": type(exc).__name__, "cause_code": getattr(exc, "code", None)},
        ) from exc
    identities = shape_statistics_result.threshold_identities
    if (
        not isinstance(identities, tuple)
        or shape_statistics_result.threshold_count != len(identities)
        or len(set(identities)) != len(identities)
    ):
        raise Foams105PlotDataValidationError(
            "threshold identities are inconsistent",
            code="invalid_threshold_identities",
        )
    rows_by_identity = {(row.image_id, row.component_id): row for row in shape_result.rows}
    if any(identity not in rows_by_identity for identity in identities):
        raise Foams105PlotDataValidationError(
            "threshold identity is missing from cached shape rows",
            code="missing_threshold_identity",
        )
    rows = tuple(rows_by_identity[identity] for identity in identities)

    def value_of(row, field):
        if field == "eccentricity":
            return row.eccentricity
        if field == "orientation_project_deg":
            return row.orientation_project_deg
        shape_value = getattr(row, field)
        return shape_value.value if shape_value.status == "defined" else None

    identity_value_samples = {
        field: tuple(((row.image_id, row.component_id), value_of(row, field)) for row in rows)
        for field in FOAMS105_SHAPE_HISTOGRAM_FIELDS
    }
    orientation_flags = {
        (row.image_id, row.component_id): row.orientation_defined for row in rows
    }
    center_series = {
        field: _center_histogram(
            field,
            identity_value_samples[field],
            FOAMS105_ORIENTATION_HIST_CENTERS if field == "orientation_project_deg" else FOAMS105_HIST_CENTERS,
            orientation_flags=orientation_flags if field == "orientation_project_deg" else None,
        )
        for field in ("eccentricity", "elongation", "aspect_ratio", "orientation_project_deg")
    }
    regularity = _edge_histogram(
        "regularity",
        identity_value_samples["regularity"],
        FOAMS105_HIST_EDGES,
    )
    return Foams105ShapeHistogramResult(
        shape_statistics_result=shape_statistics_result,
        population_identities=identities,
        eccentricity=center_series["eccentricity"],
        elongation=center_series["elongation"],
        aspect_ratio=center_series["aspect_ratio"],
        regularity=regularity,
        orientation_project_deg=center_series["orientation_project_deg"],
    )


def build_foams105_bulk_na_plot_data(
    selection_result: Foams105BinnedSelectionResult,
    count_groups: Sequence[Foams105CountGroupInput],
    group_area_results: Sequence[Foams105GroupAreaResult],
    *,
    suggestion_result: Optional[Foams105CutoffSuggestionResult] = None,
) -> Foams105BulkNaPlotDataResult:
    """Replay bulk-N_A values, retaining the source slot-4 mismatch literally."""

    if not isinstance(selection_result, Foams105BinnedSelectionResult):
        raise Foams105PlotDataValidationError(
            "selection_result must be a Foams105BinnedSelectionResult",
            code="invalid_binned_selection",
        )
    try:
            if selection_result.range_origin == FOAMS105_SELECTION_MANUAL_ORIGIN:
                if suggestion_result is not None:
                    raise Foams105PlotDataValidationError(
                        "manual selection cannot include a cutoff suggestion",
                        code="contradictory_selection_provenance",
                    )
                canonical_selection = select_foams105_binned_ranges(
                    selection_result.input_bin_labels_mm,
                    selection_result.groups,
                    selection_result.requested_ranges,
                )
            elif selection_result.range_origin == FOAMS105_SELECTION_AUTOSMART_ORIGIN:
                if not isinstance(suggestion_result, Foams105CutoffSuggestionResult):
                    raise Foams105PlotDataValidationError(
                        "autosmart selection requires its accepted suggestion result",
                        code="missing_suggestion_provenance",
                    )
                canonical_selection = select_foams105_suggested_ranges(suggestion_result)
            else:
                raise Foams105PlotDataValidationError(
                    "selection range origin is unsupported",
                    code="invalid_selection_provenance",
                    context={"range_origin": selection_result.range_origin},
                )
    except (Foams105SelectionValidationError, Foams105SelectionDomainError) as exc:
        raise Foams105PlotDataValidationError(
            "selection_result cannot be replayed from its source inputs",
            code="invalid_binned_selection",
            context={"cause_code": getattr(exc, "code", "invalid_input")},
        ) from exc
    if canonical_selection != selection_result:
        raise Foams105PlotDataValidationError(
            "selection_result differs from its canonical replay",
            code="stale_binned_selection",
        )

    try:
        raw_count_groups = tuple(count_groups)
        normalized_areas = tuple(group_area_results)
    except TypeError as exc:
        raise Foams105PlotDataValidationError(
            "count_groups and group_area_results must be sequences",
            code="invalid_bulk_na_inputs",
        ) from exc
    if len(raw_count_groups) != len(canonical_selection.groups):
        raise Foams105PlotDataValidationError(
            "count_groups must match the declared binned-selection groups",
            code="bulk_na_group_mismatch",
        )
    if any(not isinstance(group, Foams105CountGroupInput) for group in raw_count_groups):
        raise Foams105PlotDataValidationError(
            "count_groups contains an invalid record",
            code="invalid_count_group",
        )
    if any(not isinstance(area, Foams105GroupAreaResult) for area in normalized_areas):
        raise Foams105PlotDataValidationError(
            "group_area_results contains an invalid record",
            code="invalid_group_area",
        )

    counts_by_slot = {}
    normalized_counts = []
    total_count = 0
    for expected, counts in zip(canonical_selection.groups, raw_count_groups):
        if (
            counts.group_id != expected.group_id
            or counts.source_slot != expected.source_slot
            or not isinstance(counts.counts, tuple)
            or len(counts.counts) != len(canonical_selection.input_bin_labels_mm)
        ):
            raise Foams105PlotDataValidationError(
                "count group identity, slot, or vector is incompatible",
                code="bulk_na_group_mismatch",
                context={"group_id": expected.group_id, "source_slot": expected.source_slot},
            )
        normalized_values = []
        for index, count in enumerate(counts.counts):
            if not isinstance(count, Integral) or isinstance(count, bool) or count < 0:
                raise Foams105PlotDataValidationError(
                    "count group values must be nonnegative integers",
                    code="invalid_count_group",
                    context={"group_id": expected.group_id, "index": index},
                )
            integer = int(count)
            total_count += integer
            if total_count > FOAMS105_MAX_EXACT_COUNT:
                raise Foams105PlotDataValidationError(
                    "count total exceeds the exact-integer plot profile",
                    code="count_total_exceeds_exact_limit",
                )
            normalized_values.append(integer)
        normalized_group = Foams105CountGroupInput(
            expected.group_id, expected.source_slot, tuple(normalized_values)
        )
        counts_by_slot[expected.source_slot] = normalized_group
        normalized_counts.append(normalized_group)

    areas_by_group = {area.group_id: area for area in normalized_areas}
    if len(areas_by_group) != len(normalized_areas):
        raise Foams105PlotDataValidationError(
            "group_area_results contains duplicate group IDs",
            code="duplicate_group_area",
        )
    for area in normalized_areas:
        area1 = area.phase_corrected_area_mm2
        if (
            not isinstance(area1, (int, float))
            or isinstance(area1, bool)
            or not math.isfinite(float(area1))
        ):
            raise Foams105PlotDataValidationError(
                "group phase-corrected area1 values must be finite real values",
                code="invalid_group_area",
                context={"group_id": area.group_id, "area1": repr(area1)},
            )
    expected_group_ids = {group.group_id for group in canonical_selection.groups}
    if set(areas_by_group) != expected_group_ids:
        raise Foams105PlotDataValidationError(
            "group area records must match configured binned groups",
            code="bulk_na_group_mismatch",
            context={
                "missing_group_ids": tuple(sorted(expected_group_ids - set(areas_by_group))),
                "extra_group_ids": tuple(sorted(set(areas_by_group) - expected_group_ids)),
            },
        )

    groups_by_slot = {group.source_slot: group for group in canonical_selection.groups}
    traces_by_slot = {trace.source_slot: trace for trace in canonical_selection.group_traces}
    slots = []
    for group in reversed(canonical_selection.groups):
        slot = group.source_slot
        trace = traces_by_slot[slot]
        x_area = areas_by_group[group.group_id].phase_corrected_area_mm2
        diagnostics = []
        if slot == 4:
            count_slot = 3
            denominator_slot = 3
            diagnostics.append("source_slot4_uses_slot3_counts_and_area")
        else:
            count_slot = slot
            denominator_slot = slot

        count_group = groups_by_slot.get(count_slot)
        denominator_group = groups_by_slot.get(denominator_slot)
        if count_group is None or denominator_group is None:
            raise Foams105PlotDataValidationError(
                "source bulk-N_A slot references an undeclared source slot",
                code="missing_bulk_na_source_slot",
                context={"source_slot": slot, "count_slot": count_slot, "denominator_slot": denominator_slot},
            )
        indices = trace.density_source_indices
        numerator: Optional[int] = None
        denominator: Optional[float] = areas_by_group[denominator_group.group_id].phase_corrected_area_mm2

        if not trace.enabled or not indices:
            value = Foams105RealValue(
                value=None,
                reason="no_source_slice",
                upstream_reason="disabled_group" if not trace.enabled else "empty_density_source_indices",
            )
            diagnostics.append("bulk_na_unavailable_no_source_slice")
        else:
            source_counts = counts_by_slot[count_slot].counts
            numerator = 0
            for index in indices:
                if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(source_counts):
                    raise Foams105PlotDataValidationError(
                        "density_source_indices contains an invalid source index",
                        code="invalid_density_source_index",
                        context={"source_slot": slot, "count_source_slot": count_slot, "index": index},
                    )
                numerator += source_counts[index]

            if denominator == 0.0:
                classification = (
                    LEGACY_INDETERMINATE
                    if numerator == 0
                    else LEGACY_NEGATIVE_INFINITY
                    if numerator < 0
                    else LEGACY_POSITIVE_INFINITY
                )
                value = Foams105RealValue(
                    value=None,
                    reason="zero_area_denominator",
                    operands=(float(numerator), denominator),
                    legacy_classification=classification,
                )
                diagnostics.append("bulk_na_zero_area_denominator")
            else:
                numeric = float(numerator) / denominator
                if not math.isfinite(numeric):
                    raise Foams105PlotDataValidationError(
                        "bulk N_A calculation produced a nonfinite value",
                        code="nonfinite_bulk_na",
                        context={
                            "source_slot": slot,
                            "numerator": numerator,
                            "denominator": denominator,
                        },
                    )
                value = Foams105RealValue(value=numeric)

        slots.append(
            Foams105BulkNaSlotResult(
                source_slot=slot,
                group_id=group.group_id,
                count_source_slot=count_slot,
                count_source_group_id=count_group.group_id,
                denominator_source_slot=denominator_slot,
                denominator_source_group_id=denominator_group.group_id,
                density_source_indices=indices,
                numerator=numerator,
                denominator_area1_mm2=denominator,
                plot_x_area1_mm2=x_area,
                value=value,
                diagnostics=tuple(diagnostics),
            )
        )

    source_x = tuple(slot.plot_x_area1_mm2 for slot in slots if slot.plot_x_area1_mm2 != 0.0)
    source_y = tuple(slot.value for slot in slots)
    diagnostics = []
    if any("bulk_na_unavailable_no_source_slice" in slot.diagnostics for slot in slots):
        diagnostics.append("bulk_na_source_slice_unavailable")
    if len(source_x) != len(source_y):
        diagnostics.append("bulk_na_source_x_y_length_mismatch")
    if any(value.value is None for value in source_y):
        diagnostics.append("bulk_na_source_y_contains_undefined_values")
    plot_ready = len(source_x) == len(source_y) and all(value.value is not None for value in source_y)
    return Foams105BulkNaPlotDataResult(
        selection_result=canonical_selection,
        count_groups=tuple(normalized_counts),
        group_area_results=normalized_areas,
        slots_descending=tuple(slots),
        source_x_areas_nonzero_mm2=source_x,
        source_y_values=source_y,
        plot_ready=plot_ready,
        diagnostics=tuple(diagnostics),
    )
