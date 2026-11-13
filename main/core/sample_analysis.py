from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .distributions import calculate_2d_number_densities, create_diameter_bin_spec, geometric_diameter_bin_spec
from .legacy_perimeter import FOAMS105_PERIMETER_PROFILE_CURRENT
from .legacy_pipeline import (
    FOAMS105_AREA_OUTPUT_SOURCE_KEYS,
    Foams105AnalysisResult,
    Foams105AnalysisSettings,
    Foams105PipelineGroupInput,
    Foams105PreparationSettings,
    Foams105PreparedAnalysisResult,
    Foams105SegmentedImageInput,
    Foams105SourceMetadata,
    analyze_foams105_prepared,
    prepare_foams105_analysis,
)
from .legacy_selection import Foams105BinnedRange
from .nesting import NestingPlan, NestingSegment, nest_2d_distribution
from .reconstruction import UPPER_TAIL_ASSUMPTION, reconstruct_spherical_3d
from .sample_project import ProjectAnalysisImage, ProjectAnalysisSnapshot
from .sampling import build_sampling_dataset, create_image_sampling_record
from .stereology import measure_labels


class SampleAnalysisError(ValueError):
    """Raised when sample analysis setup or execution is invalid."""


@dataclass(frozen=True)
class SnapshotGroup:
    group_id: str
    mm_per_px: float
    px_per_mm: float
    source_slot: int
    image_indices: Tuple[int, ...]


@dataclass(frozen=True)
class LegacySourceAssignment:
    project_id: str
    image_id: str
    snapshot_id: str
    group_id: str
    source_slot: int
    source_key: str
    input_order: int


@dataclass(frozen=True)
class LegacyRunResult:
    prepared: Foams105PreparedAnalysisResult
    analysis: Foams105AnalysisResult
    source_assignments: Tuple[LegacySourceAssignment, ...]


@dataclass(frozen=True)
class ModernRunResult:
    distribution_result: object
    nesting_result: object
    reconstruction_result: Optional[object]
    bin_edges_mm: Tuple[float, ...]
    group_ids: Tuple[str, ...]


@dataclass(frozen=True)
class CompareRunResult:
    legacy: LegacyRunResult
    modern: ModernRunResult


@dataclass(frozen=True)
class LegacyRunConfig:
    minimum_diameter_px: float = 2.0
    perimeter_profile: str = FOAMS105_PERIMETER_PROFILE_CURRENT
    selection_mode: str = "autosmart"
    manual_ranges: Tuple[Tuple[str, float, float], ...] = ()
    volume_normalization_mode: str = "unnormalized"
    known_vesicularity_percent: Optional[float] = None
    size_lower_cutoff_mm: float = 0.5
    size_upper_cutoff_mm: float = 1.0
    shape_minimum_diameter_px: Optional[float] = None
    shape_statistics_mode: str = "initial_analysis"


@dataclass(frozen=True)
class ModernRunConfig:
    bin_edges_mm: Optional[Tuple[float, ...]] = None
    geometric_start_mm: Optional[float] = None
    geometric_bin_count: int = 8
    geometric_log10_step: float = 0.1
    min_detectable_mm: Optional[float] = None
    max_reliable_mm: Optional[float] = None
    transition_bins: Tuple[int, ...] = ()
    enable_reconstruction: bool = False


def _finite_positive(value: object, field: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)):
        raise SampleAnalysisError(f"{field} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise SampleAnalysisError(f"{field} must be a finite positive number")
    return number


def _normalize_groups(snapshot: ProjectAnalysisSnapshot) -> Tuple[SnapshotGroup, ...]:
    grouped: Dict[float, List[int]] = {}
    for index, image in enumerate(snapshot.images):
        mm_per_px = _finite_positive(image.mm_per_px, f"mm_per_px for image {image.image_id}")
        grouped.setdefault(mm_per_px, []).append(index)

    groups: List[SnapshotGroup] = []
    for slot, mm_per_px in enumerate(sorted(grouped.keys(), key=lambda value: 1.0 / value), start=1):
        px_per_mm = 1.0 / mm_per_px
        if not math.isfinite(px_per_mm) or px_per_mm <= 0.0:
            raise SampleAnalysisError(f"px_per_mm is invalid for calibration {mm_per_px!r}")
        groups.append(
            SnapshotGroup(
                group_id=f"group-{slot}",
                mm_per_px=mm_per_px,
                px_per_mm=px_per_mm,
                source_slot=slot,
                image_indices=tuple(grouped[mm_per_px]),
            )
        )

    if len(groups) > 4:
        raise SampleAnalysisError(
            f"Original FOAMS supports up to 4 resolution groups; found {len(groups)}"
        )
    return tuple(groups)


def _phase_from_mask(mask: np.ndarray) -> np.ndarray:
    phase = (np.asarray(mask) > 0).astype(np.uint8)
    return phase.copy()


def build_legacy_group_inputs(
    snapshot: ProjectAnalysisSnapshot,
) -> Tuple[Tuple[Foams105PipelineGroupInput, ...], Tuple[SnapshotGroup, ...], Tuple[LegacySourceAssignment, ...]]:
    groups = _normalize_groups(snapshot)
    assignments: List[LegacySourceAssignment] = []
    group_inputs: List[Foams105PipelineGroupInput] = []
    source_order = 0

    if len(snapshot.images) > len(FOAMS105_AREA_OUTPUT_SOURCE_KEYS):
        raise SampleAnalysisError(
            "Included image count exceeds supported FOAMS source-key capacity"
        )

    for group in groups:
        images: List[Foams105SegmentedImageInput] = []
        for index in group.image_indices:
            item: ProjectAnalysisImage = snapshot.images[index]
            source_key = FOAMS105_AREA_OUTPUT_SOURCE_KEYS[source_order]
            source_order += 1
            phase = _phase_from_mask(item.accepted_mask)
            images.append(
                Foams105SegmentedImageInput(
                    image_id=item.image_id,
                    source_image_key=source_key,
                    segmented_image=phase,
                    scale_px_per_mm=group.px_per_mm,
                    pore_phase_value=1,
                    excluded_phase_values=(None, None, None, None),
                    excluded_min_area_mm2=(0.0, 0.0, 0.0, 0.0),
                    area_output_selected=True,
                )
            )
            assignments.append(
                LegacySourceAssignment(
                    project_id=snapshot.project_id,
                    image_id=item.image_id,
                    snapshot_id=item.snapshot_id,
                    group_id=group.group_id,
                    source_slot=group.source_slot,
                    source_key=source_key,
                    input_order=source_order - 1,
                )
            )
        group_inputs.append(
            Foams105PipelineGroupInput(
                group_id=group.group_id,
                source_slot=group.source_slot,
                images=tuple(images),
            )
        )

    return tuple(group_inputs), groups, tuple(assignments)


def run_legacy_analysis(
    snapshot: ProjectAnalysisSnapshot,
    config: Optional[LegacyRunConfig] = None,
) -> LegacyRunResult:
    active = config or LegacyRunConfig()
    group_inputs, _groups, assignments = build_legacy_group_inputs(snapshot)
    minimum = _finite_positive(active.minimum_diameter_px, "minimum_diameter_px")
    shape_minimum = minimum if active.shape_minimum_diameter_px is None else _finite_positive(
        active.shape_minimum_diameter_px, "shape_minimum_diameter_px"
    )

    prepared = prepare_foams105_analysis(
        group_inputs,
        Foams105PreparationSettings(
            minimum_diameter_px=minimum,
            perimeter_profile=active.perimeter_profile,
        ),
        source_metadata=Foams105SourceMetadata(pyro_revision="sample-analysis-workspace"),
    )

    if active.selection_mode == "manual":
        if not active.manual_ranges:
            raise SampleAnalysisError("manual selection mode requires at least one manual range")
        by_group = {group_id: Foams105BinnedRange(group_id, lower, upper) for group_id, lower, upper in active.manual_ranges}
        manual_ranges = tuple(by_group.get(group.group_id) for group in prepared.groups)
    else:
        manual_ranges = ()

    analysis = analyze_foams105_prepared(
        prepared,
        Foams105AnalysisSettings(
            selection_mode=active.selection_mode,
            manual_ranges=manual_ranges,
            volume_normalization_mode=active.volume_normalization_mode,
            known_vesicularity_percent=active.known_vesicularity_percent,
            size_lower_cutoff_mm=float(active.size_lower_cutoff_mm),
            size_upper_cutoff_mm=float(active.size_upper_cutoff_mm),
            shape_minimum_diameter_px=shape_minimum,
            shape_statistics_mode=active.shape_statistics_mode,
        ),
    )
    return LegacyRunResult(prepared=prepared, analysis=analysis, source_assignments=assignments)


def _default_edges_from_measurements(measurements: Sequence[object], cfg: ModernRunConfig) -> Tuple[float, ...]:
    diameters_mm: List[float] = []
    for pore in measurements:
        value = getattr(pore, "eq_diam_units", None)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number) and number > 0:
            diameters_mm.append(number)
    if not diameters_mm:
        raise SampleAnalysisError("modern analysis requires at least one measurable accepted object")

    minimum = min(diameters_mm)
    maximum = max(diameters_mm)
    start = cfg.geometric_start_mm if cfg.geometric_start_mm is not None else max(minimum * 0.75, 1e-6)
    stop = max(maximum * 1.25, start * 1.5)
    step = cfg.geometric_log10_step
    count = cfg.geometric_bin_count
    # Pick count/step grid and extend until it covers the observed maximum.
    spec = geometric_diameter_bin_spec(start, count, step)
    while spec.edges_mm[-1] < stop:
        count += 1
        spec = geometric_diameter_bin_spec(start, count, step)
    return spec.edges_mm


def _build_modern_plan(sample_id: str, group_ids: Tuple[str, ...], bin_count: int, transition_bins: Tuple[int, ...]) -> NestingPlan:
    if len(group_ids) == 1:
        return NestingPlan(
            sample_id=sample_id,
            start_bin=0,
            stop_bin=bin_count,
            segments=(NestingSegment(group_ids[0], 0, bin_count),),
        )

    if transition_bins:
        boundaries = [int(value) for value in transition_bins]
    else:
        step = max(1, bin_count // len(group_ids))
        boundaries = [step * index for index in range(1, len(group_ids))]
    boundaries = [value for value in boundaries if 0 < value < bin_count]

    if len(boundaries) != len(group_ids) - 1:
        raise SampleAnalysisError(
            "transition_bins must contain exactly one interior boundary per adjacent group pair"
        )

    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise SampleAnalysisError("transition_bins must be strictly increasing")

    segments: List[NestingSegment] = []
    start = 0
    for group_id, stop in zip(group_ids, boundaries + [bin_count]):
        segments.append(NestingSegment(group_id, start, stop))
        start = stop
    return NestingPlan(
        sample_id=sample_id,
        start_bin=0,
        stop_bin=bin_count,
        segments=tuple(segments),
    )


def run_modern_analysis(
    snapshot: ProjectAnalysisSnapshot,
    config: Optional[ModernRunConfig] = None,
) -> ModernRunResult:
    active = config or ModernRunConfig()
    groups = _normalize_groups(snapshot)

    image_records = []
    all_measurements = []
    for index, item in enumerate(snapshot.images):
        group = next(group for group in groups if index in group.image_indices)
        mm_per_px = _finite_positive(item.mm_per_px, f"mm_per_px for image {item.image_id}")
        min_detectable_mm = active.min_detectable_mm if active.min_detectable_mm is not None else mm_per_px
        image_records.append(
            create_image_sampling_record(
                sample_id=snapshot.project_id,
                image_id=item.image_id,
                image_index=index,
                magnification_group_id=group.group_id,
                label_map=item.accepted_labels,
                calibration=mm_per_px,
                calibration_unit="mm",
                min_detectable_diameter=min_detectable_mm,
                max_reliable_diameter=active.max_reliable_mm,
                source_id=item.snapshot_id,
            )
        )
        all_measurements.extend(
            measure_labels(
                item.accepted_labels,
                image_index=index,
                scale={"unitsPerPx": mm_per_px, "unitName": "mm"},
            )
        )

    dataset = build_sampling_dataset(image_records, all_measurements)
    if active.bin_edges_mm is not None:
        edges = tuple(active.bin_edges_mm)
    else:
        edges = _default_edges_from_measurements(all_measurements, active)
    spec = create_diameter_bin_spec(edges)
    distribution = calculate_2d_number_densities(dataset, spec)

    group_ids = tuple(group.group_id for group in groups)
    plan = _build_modern_plan(snapshot.project_id, group_ids, len(spec.bins), active.transition_bins)
    nested = nest_2d_distribution(distribution, plan)

    reconstruction = None
    if active.enable_reconstruction:
        reconstruction = reconstruct_spherical_3d(
            distribution,
            plan,
            upper_tail_assumption=UPPER_TAIL_ASSUMPTION,
        )

    return ModernRunResult(
        distribution_result=distribution,
        nesting_result=nested,
        reconstruction_result=reconstruction,
        bin_edges_mm=spec.edges_mm,
        group_ids=group_ids,
    )


def run_compare_analysis(
    snapshot: ProjectAnalysisSnapshot,
    legacy_config: Optional[LegacyRunConfig] = None,
    modern_config: Optional[ModernRunConfig] = None,
) -> CompareRunResult:
    legacy = run_legacy_analysis(snapshot, legacy_config)
    modern = run_modern_analysis(snapshot, modern_config)
    return CompareRunResult(legacy=legacy, modern=modern)
