"""Headless FOAMS 1.0.5 preparation and integrated analysis workflow."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .legacy_areas import (
    Foams105GroupAreaResult,
    Foams105ImageAreaResult,
    aggregate_foams105_group_areas,
    calculate_foams105_image_areas,
)
from .legacy_cutoffs import (
    Foams105CutoffGroupInput,
    Foams105CutoffSuggestionResult,
    suggest_foams105_cutoffs,
)
from .legacy_foams import Foams105ConversionResult, convert_foams105_nv
from .legacy_measurements import (
    Foams105ImageMeasurementResult,
    measure_foams105_image,
)
from .legacy_perimeter import (
    FOAMS105_PERIMETER_SUPPORTED_PROFILES,
    Foams105ImagePerimeterResult,
    measure_foams105_perimeter,
)
from .legacy_plot_data import (
    Foams105BulkNaPlotDataResult,
    Foams105ShapeHistogramResult,
    build_foams105_bulk_na_plot_data,
    build_foams105_shape_histograms,
)
from .legacy_raw_selection import (
    Foams105RawGroupInput,
    Foams105RawObject,
    Foams105RawSelectionResult,
    select_foams105_raw_objects,
)
from .legacy_selection import (
    Foams105BinnedGroupInput,
    Foams105BinnedRange,
    Foams105BinnedSelectionResult,
    select_foams105_binned_ranges,
    select_foams105_suggested_ranges,
)
from .legacy_shape_statistics import (
    Foams105ComponentShapeMeasures,
    Foams105ShapeResult,
    Foams105ShapeStatisticsResult,
    assemble_foams105_shapes_from_cached_components,
    calculate_foams105_component_shape_measures,
    summarize_foams105_shapes,
)
from .legacy_shapes import (
    Foams105ImageMomentGeometryResult,
    measure_foams105_moment_geometry,
)
from .legacy_size_classes import (
    Foams105DiameterFilterResult,
    Foams105HistogramResult,
    Foams105LabelPreparationResult,
    Foams105NormalizationResult,
    build_foams105_bin_labels,
    count_foams105_histogram,
    filter_foams105_diameters,
    normalize_foams105_counts,
)
from .legacy_summaries import (
    Foams105CountGroupInput,
    Foams105DistributionSummaryResult,
    Foams105SizeCategoryResult,
    summarize_foams105_distribution,
    summarize_foams105_size_categories,
)
from .legacy_vesicularity import (
    Foams105VesicularityGroupInput,
    Foams105VesicularityResult,
    calculate_foams105_2d_vesicularity,
)
from .legacy_volumes import (
    FOAMS105_VOLUME_MODES,
    Foams105VolumeDistributionResult,
    calculate_foams105_volume_distribution,
)

FOAMS105_PIPELINE_METHOD = "foams_1_0_5_headless_pipeline_v1"
FOAMS105_PIPELINE_SOURCE_REPOSITORY = "jsoteloflores/FOAMS-1.0.5"
FOAMS105_PIPELINE_SOURCE_COMMIT = "179663203f2d0f86b2863d5ea7f8f70dadca02f8"
FOAMS105_PIPELINE_SCOPE = "segmented_phase_analysis_and_output_ready_tables"
FOAMS105_AREA_OUTPUT_SOURCE_KEYS = (
    "mag1",
    "first",
    "second",
    "third",
    "fifteenth",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eigth",
    "ninth",
    "tenth",
    "eleventh",
    "sixteenth",
    "seventeenth",
    "twelth",
    "thirteenth",
    "fourteenth",
    "eighteenth",
    "nineteenth",
)


class Foams105PipelineStageError(RuntimeError):
    """A required pipeline stage failed; no complete analysis was returned."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        cause_type: str,
        cause_code: Optional[str] = None,
        context: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.cause_type = cause_type
        self.cause_code = cause_code
        self.code = "pipeline_stage_failed"
        self.context = MappingProxyType(dict(context or {}))


@dataclass(frozen=True)
class Foams105PipelineDiagnostic:
    code: str
    stage: str
    message: str
    context: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Foams105SourceMetadata:
    """Caller-supplied source fields not inferred from scientific results."""

    density_derived_vesicularity_percent: Optional[float] = None
    pyro_revision: str = "unknown"


@dataclass(frozen=True)
class Foams105SegmentedImageInput:
    """One already segmented, label-valued 2D image and its phase settings."""

    image_id: str
    source_image_key: str
    segmented_image: np.ndarray
    scale_px_per_mm: float
    pore_phase_value: object
    excluded_phase_values: Tuple[Optional[object], Optional[object], Optional[object], Optional[object]]
    excluded_min_area_mm2: Tuple[float, float, float, float]
    area_output_selected: bool = True


@dataclass(frozen=True)
class Foams105PipelineGroupInput:
    group_id: str
    source_slot: int
    images: Tuple[Foams105SegmentedImageInput, ...]


@dataclass(frozen=True)
class Foams105PreparationSettings:
    minimum_diameter_px: float
    perimeter_profile: str


@dataclass(frozen=True)
class Foams105AnalysisSettings:
    selection_mode: str
    manual_ranges: Tuple[Optional[Foams105BinnedRange], ...]
    volume_normalization_mode: str
    known_vesicularity_percent: Optional[float]
    size_lower_cutoff_mm: float
    size_upper_cutoff_mm: float
    shape_minimum_diameter_px: float
    shape_statistics_mode: str


@dataclass(frozen=True)
class Foams105SourceImageMapping:
    source_order: int
    source_image_key: str
    image_id: str
    group_id: str
    source_slot: int
    area_output_selected: bool


@dataclass(frozen=True)
class Foams105PreparedImage:
    image_input: Foams105SegmentedImageInput
    input_sha256: str
    measurement_result: Foams105ImageMeasurementResult
    image_area_result: Foams105ImageAreaResult
    moment_geometry_result: Foams105ImageMomentGeometryResult
    perimeter_result: Foams105ImagePerimeterResult
    component_shape_measures: Tuple[Foams105ComponentShapeMeasures, ...]
    unfiltered_pore_objects: Tuple[Foams105RawObject, ...]


@dataclass(frozen=True)
class Foams105PreparedGroup:
    group_id: str
    source_slot: int
    scale_px_per_mm: float
    images: Tuple[Foams105PreparedImage, ...]
    group_area_result: Foams105GroupAreaResult
    unfiltered_pore_objects: Tuple[Foams105RawObject, ...]
    diameter_filter_result: Foams105DiameterFilterResult
    diameter_filtered_objects: Tuple[Foams105RawObject, ...]
    histogram_result: Foams105HistogramResult
    normalization_result: Optional[Foams105NormalizationResult]
    count_group_input: Foams105CountGroupInput
    diagnostics: Tuple[Foams105PipelineDiagnostic, ...]


@dataclass(frozen=True)
class Foams105PreparedAnalysisResult:
    preparation_settings: Foams105PreparationSettings
    source_metadata: Foams105SourceMetadata
    groups: Tuple[Foams105PreparedGroup, ...]
    label_preparation_result: Foams105LabelPreparationResult
    source_image_mapping: Tuple[Foams105SourceImageMapping, ...]
    image_count: int
    perimeter_profile: str
    method: str = FOAMS105_PIPELINE_METHOD
    source_repository: str = FOAMS105_PIPELINE_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_PIPELINE_SOURCE_COMMIT
    scope: str = FOAMS105_PIPELINE_SCOPE
    diagnostics: Tuple[Foams105PipelineDiagnostic, ...] = ()


@dataclass(frozen=True)
class Foams105RawTableRow:
    output_index: int
    source_slot: int
    group_id: str
    image_id: str
    component_id: int
    source_object_index: int
    equivalent_diameter_mm: float
    area_mm2: float
    perimeter_mm: float
    major_axis_mm: float
    minor_axis_mm: float
    orientation_project_deg: float


@dataclass(frozen=True)
class Foams105AnalysisResult:
    prepared_result: Foams105PreparedAnalysisResult
    analysis_settings: Foams105AnalysisSettings
    cutoff_suggestion_result: Optional[Foams105CutoffSuggestionResult]
    applied_ranges: Tuple[Optional[Foams105BinnedRange], ...]
    binned_selection_result: Foams105BinnedSelectionResult
    raw_selection_result: Foams105RawSelectionResult
    vesicularity_result: Foams105VesicularityResult
    conversion_result: Foams105ConversionResult
    volume_result: Foams105VolumeDistributionResult
    distribution_summary_result: Foams105DistributionSummaryResult
    size_category_result: Foams105SizeCategoryResult
    shape_result: Foams105ShapeResult
    shape_statistics_result: Foams105ShapeStatisticsResult
    shape_histogram_result: Foams105ShapeHistogramResult
    shape_export_histogram_result: Foams105ShapeHistogramResult
    bulk_na_plot_data_result: Foams105BulkNaPlotDataResult
    raw_data_rows: Tuple[Foams105RawTableRow, ...]
    rawcutoffs_data_rows: Tuple[Foams105RawTableRow, ...]
    image_count: int
    method: str = FOAMS105_PIPELINE_METHOD
    source_repository: str = FOAMS105_PIPELINE_SOURCE_REPOSITORY
    source_commit: str = FOAMS105_PIPELINE_SOURCE_COMMIT
    scope: str = FOAMS105_PIPELINE_SCOPE
    diagnostics: Tuple[Foams105PipelineDiagnostic, ...] = ()


def _stage(stage: str, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        context = getattr(exc, "context", {})
        raise Foams105PipelineStageError(
            stage,
            f"FOAMS pipeline stage {stage!r} failed: {exc}",
            cause_type=type(exc).__name__,
            cause_code=getattr(exc, "code", None),
            context=dict(context) if isinstance(context, Mapping) else {},
        ) from exc


def _pipeline_validation(stage: str, code: str, message: str, **context) -> Foams105PipelineStageError:
    return Foams105PipelineStageError(
        stage,
        message,
        cause_type="Foams105PipelineValidationError",
        cause_code=code,
        context=context,
    )


def _finite_real(value: object, field: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise _pipeline_validation("input_validation", "invalid_scalar", f"{field} must be a real scalar", field=field)
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise _pipeline_validation("input_validation", "invalid_scalar", f"{field} is not representable", field=field) from exc
    if not math.isfinite(number) or (positive and number <= 0.0) or (nonnegative and number < 0.0):
        raise _pipeline_validation("input_validation", "invalid_scalar", f"{field} is outside its supported domain", field=field, value=number)
    return number


def _snapshot_image(image: object, image_id: str) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise _pipeline_validation(
            "input_validation", "invalid_segmented_image", "segmented_image must be a nonempty 2D ndarray", image_id=image_id
        )
    if image.dtype.kind not in "biuf" or (image.dtype.kind == "f" and not bool(np.isfinite(image).all())):
        raise _pipeline_validation(
            "input_validation", "invalid_segmented_image", "segmented_image dtype/values are unsupported", image_id=image_id
        )
    contiguous = np.ascontiguousarray(image)
    owned = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(contiguous.shape)
    return owned


def _image_checksum(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(image.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(tuple(int(size) for size in image.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(image.tobytes(order="C"))
    return digest.hexdigest()


def _object_from_component(image_id: str, component) -> Foams105RawObject:
    return Foams105RawObject(
        image_id=image_id,
        component_id=component.component_id,
        equivalent_diameter_mm=component.equivalent_diameter_mm,
        area_mm2=component.area_mm2,
    )


def _normalize_source_metadata(value: Optional[Foams105SourceMetadata]) -> Foams105SourceMetadata:
    metadata = value if value is not None else Foams105SourceMetadata()
    if not isinstance(metadata, Foams105SourceMetadata):
        raise _pipeline_validation("input_validation", "invalid_source_metadata", "source_metadata has the wrong type")
    if not isinstance(metadata.pyro_revision, str) or not metadata.pyro_revision.strip():
        raise _pipeline_validation("input_validation", "invalid_source_metadata", "pyro_revision must be a nonempty string")
    vesicularity = metadata.density_derived_vesicularity_percent
    if vesicularity is not None:
        vesicularity = _finite_real(vesicularity, "density_derived_vesicularity_percent", nonnegative=True)
        if vesicularity > 100.0:
            raise _pipeline_validation(
                "input_validation", "invalid_source_metadata", "density-derived vesicularity must be <= 100", value=vesicularity
            )
    return replace(metadata, density_derived_vesicularity_percent=vesicularity)


def prepare_foams105_analysis(
    groups: Sequence[Foams105PipelineGroupInput],
    settings: Foams105PreparationSettings,
    *,
    source_metadata: Optional[Foams105SourceMetadata] = None,
) -> Foams105PreparedAnalysisResult:
    """Own segmented images and run all image-local/preselection stages once."""

    if not isinstance(settings, Foams105PreparationSettings):
        raise _pipeline_validation("input_validation", "invalid_preparation_settings", "settings has the wrong type")
    minimum = _finite_real(settings.minimum_diameter_px, "minimum_diameter_px", positive=True)
    if settings.perimeter_profile not in FOAMS105_PERIMETER_SUPPORTED_PROFILES:
        raise _pipeline_validation(
            "input_validation", "unsupported_perimeter_profile", "an explicit supported perimeter profile is required",
            profile=settings.perimeter_profile,
        )
    metadata = _normalize_source_metadata(source_metadata)
    if isinstance(groups, (str, bytes)):
        raise _pipeline_validation("input_validation", "invalid_groups", "groups must be an ordered sequence")
    try:
        raw_groups = tuple(groups)
    except TypeError as exc:
        raise _pipeline_validation("input_validation", "invalid_groups", "groups must be an ordered sequence") from exc
    if not 1 <= len(raw_groups) <= 4:
        raise _pipeline_validation("input_validation", "unsupported_group_count", "one to four source groups are required", count=len(raw_groups))

    normalized_groups = []
    seen_group_ids = set()
    seen_image_ids = set()
    seen_source_keys = set()
    previous_scale = None
    source_mappings = []
    source_order = 0
    all_scales = []
    for group_index, group in enumerate(raw_groups):
        if not isinstance(group, Foams105PipelineGroupInput):
            raise _pipeline_validation("input_validation", "invalid_group", "group record has the wrong type", group_index=group_index)
        if not isinstance(group.group_id, str) or not group.group_id.strip() or group.group_id in seen_group_ids:
            raise _pipeline_validation("input_validation", "invalid_group_id", "group IDs must be nonempty and unique", group_index=group_index)
        expected_slot = group_index + 1
        if not isinstance(group.source_slot, int) or isinstance(group.source_slot, bool) or group.source_slot != expected_slot:
            raise _pipeline_validation(
                "input_validation", "invalid_source_slot", "source slots must be contiguous and input-ordered",
                group_id=group.group_id, expected=expected_slot, actual=group.source_slot,
            )
        if not isinstance(group.images, tuple) or not group.images:
            raise _pipeline_validation("input_validation", "invalid_group_images", "each configured group needs an immutable nonempty image tuple", group_id=group.group_id)
        scale = _finite_real(group.images[0].scale_px_per_mm, "scale_px_per_mm", positive=True)
        if previous_scale is not None and scale <= previous_scale:
            raise _pipeline_validation(
                "input_validation", "group_scale_order", "source groups must already be ordered by strictly increasing scale",
                group_id=group.group_id, scale_px_per_mm=scale, previous_scale_px_per_mm=previous_scale,
            )
        previous_scale = scale
        all_scales.append(scale)
        prepared_images = []
        image_areas = []
        group_unfiltered_objects = []
        group_diameters = []
        for image_index, image_input in enumerate(group.images):
            if not isinstance(image_input, Foams105SegmentedImageInput):
                raise _pipeline_validation("input_validation", "invalid_image", "image record has the wrong type", group_id=group.group_id, image_index=image_index)
            if not isinstance(image_input.image_id, str) or not image_input.image_id.strip() or image_input.image_id in seen_image_ids:
                raise _pipeline_validation("input_validation", "invalid_image_id", "image IDs must be globally unique nonempty strings", image_id=image_input.image_id)
            if image_input.source_image_key not in FOAMS105_AREA_OUTPUT_SOURCE_KEYS:
                raise _pipeline_validation(
                    "input_validation", "unknown_source_image_key", "source_image_key is not in the pinned source-key map",
                    image_id=image_input.image_id, source_image_key=image_input.source_image_key,
                )
            if image_input.source_image_key in seen_source_keys:
                raise _pipeline_validation(
                    "input_validation", "duplicate_source_image_key", "source image keys must be unique",
                    source_image_key=image_input.source_image_key,
                )
            image_scale = _finite_real(image_input.scale_px_per_mm, "scale_px_per_mm", positive=True)
            if image_scale != scale:
                raise _pipeline_validation(
                    "input_validation", "group_image_scale_mismatch", "all images in a source group must use the group's scale",
                    group_id=group.group_id, image_id=image_input.image_id, group_scale=scale, image_scale=image_scale,
                )
            if not isinstance(image_input.area_output_selected, bool):
                raise _pipeline_validation(
                    "input_validation", "invalid_area_output_selection", "area_output_selected must be bool",
                    image_id=image_input.image_id,
                )
            snapshot = _snapshot_image(image_input.segmented_image, image_input.image_id)
            owned_input = replace(image_input, segmented_image=snapshot, scale_px_per_mm=image_scale)
            measurement = _stage(
                "phase_measurement",
                measure_foams105_image,
                snapshot,
                image_input.pore_phase_value,
                image_scale,
                image_input.excluded_phase_values,
            )
            area_result = _stage(
                "image_area_accounting",
                calculate_foams105_image_areas,
                image_input.image_id,
                measurement,
                image_input.excluded_min_area_mm2,
            )
            moment_result = _stage(
                "moment_geometry",
                measure_foams105_moment_geometry,
                image_input.image_id,
                measurement.pore_result,
            )
            perimeter_result = _stage(
                "perimeter_geometry",
                measure_foams105_perimeter,
                image_input.image_id,
                measurement.pore_result,
                profile=settings.perimeter_profile,
            )
            if len(moment_result.components) != len(perimeter_result.components):
                raise _pipeline_validation(
                    "component_shape_cache",
                    "geometry_component_count_mismatch",
                    "moment/perimeter component counts differ during preparation",
                    image_id=image_input.image_id,
                )
            cached_shape_values = tuple(
                _stage(
                    "component_shape_measures",
                    calculate_foams105_component_shape_measures,
                    moment_component,
                    perimeter_component,
                )
                for moment_component, perimeter_component in zip(
                    moment_result.components,
                    perimeter_result.components,
                )
            )
            image_objects = tuple(
                _object_from_component(image_input.image_id, component)
                for component in measurement.pore_result.components
            )
            prepared = Foams105PreparedImage(
                image_input=owned_input,
                input_sha256=_image_checksum(snapshot),
                measurement_result=measurement,
                image_area_result=area_result,
                moment_geometry_result=moment_result,
                perimeter_result=perimeter_result,
                component_shape_measures=cached_shape_values,
                unfiltered_pore_objects=image_objects,
            )
            prepared_images.append(prepared)
            image_areas.append(area_result)
            group_unfiltered_objects.extend(image_objects)
            group_diameters.extend(obj.equivalent_diameter_mm for obj in image_objects)
            seen_image_ids.add(image_input.image_id)
            seen_source_keys.add(image_input.source_image_key)
            source_mappings.append(
                Foams105SourceImageMapping(
                    source_order=source_order,
                    source_image_key=image_input.source_image_key,
                    image_id=image_input.image_id,
                    group_id=group.group_id,
                    source_slot=group.source_slot,
                    area_output_selected=image_input.area_output_selected,
                )
            )
            source_order += 1
        group_area = _stage("group_area_accounting", aggregate_foams105_group_areas, group.group_id, tuple(image_areas))
        normalized_groups.append(
            (group, scale, tuple(prepared_images), group_area, tuple(group_unfiltered_objects), tuple(group_diameters))
        )
        seen_group_ids.add(group.group_id)

    label_preparation = _stage("global_size_labels", build_foams105_bin_labels, minimum, tuple(all_scales))
    prepared_groups = []
    for group, scale, images, group_area, unfiltered_objects, diameters in normalized_groups:
        diameter_filter = _stage(
            "minimum_diameter_filter",
            filter_foams105_diameters,
            diameters,
            minimum,
            scale,
        )
        filtered_objects = tuple(unfiltered_objects[index] for index in diameter_filter.retained_indices)
        histogram = _stage(
            "group_histogram",
            count_foams105_histogram,
            label_preparation.bin_labels_mm,
            diameter_filter.retained_diameters_mm,
        )
        normalization = None
        diagnostics = []
        if group_area.phase_corrected_area_mm2 > 0.0:
            normalization = _stage(
                "histogram_normalization",
                normalize_foams105_counts,
                histogram,
                group_area.phase_corrected_area_mm2,
            )
        else:
            diagnostics.append(
                Foams105PipelineDiagnostic(
                    code="nonpositive_area1_normalization_unavailable",
                    stage="histogram_normalization",
                    message="phase-corrected area1 is nonpositive; histograms are retained but N_A normalization is unavailable",
                    context=(("group_id", group.group_id), ("phase_corrected_area1_mm2", repr(group_area.phase_corrected_area_mm2))),
                )
            )
        prepared_groups.append(
            Foams105PreparedGroup(
                group_id=group.group_id,
                source_slot=group.source_slot,
                scale_px_per_mm=scale,
                images=images,
                group_area_result=group_area,
                unfiltered_pore_objects=unfiltered_objects,
                diameter_filter_result=diameter_filter,
                diameter_filtered_objects=filtered_objects,
                histogram_result=histogram,
                normalization_result=normalization,
                count_group_input=Foams105CountGroupInput(group.group_id, group.source_slot, histogram.counts),
                diagnostics=tuple(diagnostics),
            )
        )
    normalized_settings = replace(settings, minimum_diameter_px=minimum)
    return Foams105PreparedAnalysisResult(
        preparation_settings=normalized_settings,
        source_metadata=metadata,
        groups=tuple(prepared_groups),
        label_preparation_result=label_preparation,
        source_image_mapping=tuple(source_mappings),
        image_count=len(source_mappings),
        perimeter_profile=settings.perimeter_profile,
        diagnostics=tuple(diagnostic for group in prepared_groups for diagnostic in group.diagnostics),
    )


def _validate_prepared_result(prepared: object) -> Foams105PreparedAnalysisResult:
    if not isinstance(prepared, Foams105PreparedAnalysisResult):
        raise _pipeline_validation("prepared_validation", "invalid_prepared_result", "prepared result has the wrong type")
    if (
        prepared.method != FOAMS105_PIPELINE_METHOD
        or prepared.source_repository != FOAMS105_PIPELINE_SOURCE_REPOSITORY
        or prepared.source_commit != FOAMS105_PIPELINE_SOURCE_COMMIT
        or prepared.scope != FOAMS105_PIPELINE_SCOPE
    ):
        raise _pipeline_validation("prepared_validation", "invalid_prepared_provenance", "prepared provenance is incompatible")
    if prepared.perimeter_profile not in FOAMS105_PERIMETER_SUPPORTED_PROFILES:
        raise _pipeline_validation("prepared_validation", "invalid_prepared_profile", "prepared perimeter profile is unsupported")
    if not isinstance(prepared.preparation_settings, Foams105PreparationSettings):
        raise _pipeline_validation("prepared_validation", "invalid_preparation_settings", "cached preparation settings have the wrong type")
    if (
        prepared.preparation_settings.perimeter_profile != prepared.perimeter_profile
        or prepared.preparation_settings.minimum_diameter_px != prepared.label_preparation_result.minimum_diameter_px
    ):
        raise _pipeline_validation("prepared_validation", "stale_preparation_settings", "cached preparation settings no longer match their results")
    if not isinstance(prepared.source_metadata, Foams105SourceMetadata):
        raise _pipeline_validation("prepared_validation", "invalid_source_metadata", "cached source metadata has the wrong type")
    if not isinstance(prepared.groups, tuple) or not 1 <= len(prepared.groups) <= 4:
        raise _pipeline_validation("prepared_validation", "invalid_prepared_groups", "prepared group tuple is invalid")
    if not isinstance(prepared.source_image_mapping, tuple):
        raise _pipeline_validation("prepared_validation", "invalid_source_mapping", "source image mapping must be an immutable tuple")
    seen_images = set()
    seen_sources = set()
    expected_mapping = []
    previous_scale = None
    for group_index, group in enumerate(prepared.groups):
        expected_slot = group_index + 1
        if not isinstance(group, Foams105PreparedGroup) or group.source_slot != expected_slot:
            raise _pipeline_validation("prepared_validation", "invalid_prepared_groups", "prepared slots are not contiguous/input-ordered", group_index=group_index)
        if not isinstance(group.group_id, str) or not group.group_id.strip():
            raise _pipeline_validation("prepared_validation", "invalid_prepared_groups", "prepared group ID is invalid", group_index=group_index)
        if (
            not isinstance(group.scale_px_per_mm, (int, float))
            or isinstance(group.scale_px_per_mm, bool)
            or not math.isfinite(float(group.scale_px_per_mm))
            or group.scale_px_per_mm <= 0.0
            or (previous_scale is not None and group.scale_px_per_mm <= previous_scale)
        ):
            raise _pipeline_validation("prepared_validation", "invalid_prepared_scale_order", "prepared scales must remain positive and source-ordered", group_id=group.group_id)
        previous_scale = group.scale_px_per_mm
        if not isinstance(group.images, tuple) or not group.images:
            raise _pipeline_validation("prepared_validation", "invalid_prepared_images", "prepared groups need immutable image tuples", group_id=group.group_id)
        if (
            not isinstance(group.diameter_filter_result, Foams105DiameterFilterResult)
            or not isinstance(group.histogram_result, Foams105HistogramResult)
            or not isinstance(group.count_group_input, Foams105CountGroupInput)
            or not isinstance(group.group_area_result, Foams105GroupAreaResult)
        ):
            raise _pipeline_validation("prepared_validation", "invalid_prepared_group_cache", "a cached group stage record has the wrong type", group_id=group.group_id)
        if group.group_area_result.group_id != group.group_id or group.group_area_result.scale_px_per_mm != group.scale_px_per_mm:
            raise _pipeline_validation("prepared_validation", "stale_group_area", "group area identity or calibration differs from its source group", group_id=group.group_id)
        image_ids_for_group = []
        flattened_objects = []
        for image in group.images:
            if not isinstance(image, Foams105PreparedImage):
                raise _pipeline_validation("prepared_validation", "invalid_prepared_image", "prepared image record has the wrong type", group_id=group.group_id)
            if not isinstance(image.image_input, Foams105SegmentedImageInput) or image.image_input.segmented_image.flags.writeable:
                raise _pipeline_validation("prepared_validation", "mutable_prepared_image", "prepared image snapshot is invalid or writable")
            if image.image_input.image_id in seen_images:
                raise _pipeline_validation("prepared_validation", "duplicate_prepared_image", "prepared image IDs must be unique", image_id=image.image_input.image_id)
            if image.image_input.source_image_key in seen_sources:
                raise _pipeline_validation("prepared_validation", "duplicate_source_image_key", "source image keys must be unique", source_image_key=image.image_input.source_image_key)
            if image.image_input.source_image_key not in FOAMS105_AREA_OUTPUT_SOURCE_KEYS:
                raise _pipeline_validation("prepared_validation", "unknown_source_image_key", "prepared source key is not in the pinned source map", source_image_key=image.image_input.source_image_key)
            if image.image_input.scale_px_per_mm != group.scale_px_per_mm:
                raise _pipeline_validation("prepared_validation", "changed_image_scale", "cached image scale differs from source group", image_id=image.image_input.image_id, group_id=group.group_id)
            if (
                image.measurement_result.pore_phase_value != image.image_input.pore_phase_value
                or image.measurement_result.excluded_phase_values != image.image_input.excluded_phase_values
                or image.measurement_result.scale_px_per_mm != group.scale_px_per_mm
                or image.measurement_result.image_shape != tuple(image.image_input.segmented_image.shape)
                or image.image_area_result.image_id != image.image_input.image_id
                or image.image_area_result.scale_px_per_mm != group.scale_px_per_mm
                or image.image_area_result.excluded_phase_values != image.image_input.excluded_phase_values
                or image.image_area_result.excluded_min_area_mm2 != image.image_input.excluded_min_area_mm2
                or image.moment_geometry_result.image_id != image.image_input.image_id
                or image.moment_geometry_result.scale_px_per_mm != group.scale_px_per_mm
                or image.perimeter_result.image_id != image.image_input.image_id
                or image.perimeter_result.scale_px_per_mm != group.scale_px_per_mm
                or image.perimeter_result.profile != prepared.perimeter_profile
                or not isinstance(image.component_shape_measures, tuple)
                or len(image.component_shape_measures) != len(image.moment_geometry_result.components)
            ):
                raise _pipeline_validation(
                    "prepared_validation", "changed_prepared_phase_settings", "cached phase or calibration metadata differs from its image input",
                    image_id=image.image_input.image_id, group_id=group.group_id,
                )
            for index, (cached, moment_component, perimeter_component) in enumerate(
                zip(
                    image.component_shape_measures,
                    image.moment_geometry_result.components,
                    image.perimeter_result.components,
                )
            ):
                if (
                    cached.moment_component != moment_component
                    or cached.perimeter_component != perimeter_component
                ):
                    raise _pipeline_validation(
                        "prepared_validation", "stale_component_shape_cache", "cached shape values differ from prepared geometry",
                        image_id=image.image_input.image_id, component_index=index,
                    )
            seen_images.add(image.image_input.image_id)
            seen_sources.add(image.image_input.source_image_key)
            image_ids_for_group.append(image.image_input.image_id)
            expected_objects = tuple(
                _object_from_component(image.image_input.image_id, component)
                for component in image.measurement_result.pore_result.components
            )
            if image.unfiltered_pore_objects != expected_objects:
                raise _pipeline_validation("prepared_validation", "stale_unfiltered_objects", "image object snapshot differs from its measured components", image_id=image.image_input.image_id)
            flattened_objects.extend(image.unfiltered_pore_objects)
            expected_mapping.append(
                (len(expected_mapping), image.image_input.source_image_key, image.image_input.image_id, group.group_id, group.source_slot, image.image_input.area_output_selected)
            )
        if group.group_area_result.image_ids != tuple(image_ids_for_group):
            raise _pipeline_validation("prepared_validation", "stale_group_area", "group area image IDs differ from cached images", group_id=group.group_id)
        if group.unfiltered_pore_objects != tuple(flattened_objects):
            raise _pipeline_validation("prepared_validation", "stale_unfiltered_objects", "group unfiltered objects differ from cached image measurements", group_id=group.group_id)
        filter_result = group.diameter_filter_result
        if (
            filter_result.minimum_diameter_px != prepared.preparation_settings.minimum_diameter_px
            or filter_result.scale_px_per_mm != group.scale_px_per_mm
            or filter_result.input_count != len(group.unfiltered_pore_objects)
            or len(filter_result.retained_indices) != len(group.diameter_filtered_objects)
            or group.diameter_filtered_objects != tuple(group.unfiltered_pore_objects[index] for index in filter_result.retained_indices)
        ):
            raise _pipeline_validation("prepared_validation", "stale_diameter_filter", "cached diameter filter does not match group objects", group_id=group.group_id)
        if (
            group.histogram_result.bin_labels_mm != prepared.label_preparation_result.bin_labels_mm
            or group.histogram_result.input_count != len(group.diameter_filtered_objects)
            or group.histogram_result.counts != group.count_group_input.counts
            or group.count_group_input.group_id != group.group_id
            or group.count_group_input.source_slot != group.source_slot
        ):
            raise _pipeline_validation("prepared_validation", "stale_histogram", "cached histogram/count input does not match prepared labels/group", group_id=group.group_id)
        if group.normalization_result is not None and (
            group.normalization_result.bin_labels_mm != group.histogram_result.bin_labels_mm
            or group.normalization_result.counts != group.histogram_result.counts
            or group.normalization_result.corrected_area_mm2 != group.group_area_result.phase_corrected_area_mm2
        ):
            raise _pipeline_validation("prepared_validation", "stale_normalization", "cached N_A normalization does not match histogram/area1", group_id=group.group_id)
    actual_mapping = tuple(
        (mapping.source_order, mapping.source_image_key, mapping.image_id, mapping.group_id, mapping.source_slot, mapping.area_output_selected)
        for mapping in prepared.source_image_mapping
    )
    if actual_mapping != tuple(expected_mapping) or prepared.image_count != len(expected_mapping):
        raise _pipeline_validation("prepared_validation", "stale_source_mapping", "prepared source-image mapping is stale")
    for group in prepared.groups:
        for image in group.images:
            if (
                image.measurement_result.pore_result.scale_px_per_mm != group.scale_px_per_mm
                or image.image_area_result.image_id != image.image_input.image_id
                or image.moment_geometry_result.image_id != image.image_input.image_id
                or image.perimeter_result.image_id != image.image_input.image_id
                or image.perimeter_result.profile != prepared.perimeter_profile
            ):
                raise _pipeline_validation(
                    "prepared_validation", "inconsistent_prepared_image", "prepared stage records disagree on image/group identity",
                    image_id=image.image_input.image_id, group_id=group.group_id,
                )
    return prepared


def _raw_table_row(output_index: int, group: Foams105PreparedGroup, object_index: int, object_: Foams105RawObject, moment, perimeter) -> Foams105RawTableRow:
    return Foams105RawTableRow(
        output_index=output_index,
        source_slot=group.source_slot,
        group_id=group.group_id,
        image_id=object_.image_id,
        component_id=object_.component_id,
        source_object_index=object_index,
        equivalent_diameter_mm=object_.equivalent_diameter_mm,
        area_mm2=object_.area_mm2,
        perimeter_mm=perimeter.perimeter_mm,
        major_axis_mm=moment.major_axis_mm,
        minor_axis_mm=moment.minor_axis_mm,
        orientation_project_deg=moment.orientation_project_deg,
    )


def analyze_foams105_prepared(
    prepared: Foams105PreparedAnalysisResult,
    analysis_settings: Foams105AnalysisSettings,
) -> Foams105AnalysisResult:
    """Analyze cached preparation through accepted selection/summary stages."""

    prepared = _validate_prepared_result(prepared)
    if not isinstance(analysis_settings, Foams105AnalysisSettings):
        raise _pipeline_validation("analysis_settings", "invalid_analysis_settings", "analysis_settings has the wrong type")
    settings = analysis_settings
    if settings.selection_mode not in ("manual", "autosmart"):
        raise _pipeline_validation("analysis_settings", "invalid_selection_mode", "selection_mode must be manual or autosmart", mode=settings.selection_mode)
    if settings.volume_normalization_mode not in FOAMS105_VOLUME_MODES:
        raise _pipeline_validation("analysis_settings", "invalid_volume_mode", "volume_normalization_mode is unsupported", mode=settings.volume_normalization_mode)
    if settings.shape_statistics_mode not in ("initial_analysis", "updated_minimum"):
        raise _pipeline_validation("analysis_settings", "invalid_shape_mode", "shape_statistics_mode is unsupported", mode=settings.shape_statistics_mode)
    lower_cutoff = _finite_real(settings.size_lower_cutoff_mm, "size_lower_cutoff_mm", positive=True)
    upper_cutoff = _finite_real(settings.size_upper_cutoff_mm, "size_upper_cutoff_mm", positive=True)
    if lower_cutoff > upper_cutoff:
        raise _pipeline_validation("analysis_settings", "invalid_size_cutoffs", "size lower cutoff must not exceed upper cutoff")
    shape_minimum = _finite_real(settings.shape_minimum_diameter_px, "shape_minimum_diameter_px", nonnegative=True)

    binned_groups = []
    cutoff_groups = []
    count_groups = tuple(group.count_group_input for group in prepared.groups)
    group_areas = tuple(group.group_area_result for group in prepared.groups)
    for group in prepared.groups:
        if group.normalization_result is None:
            raise Foams105PipelineStageError(
                "binned_selection",
                "cannot analyze a group without positive phase-corrected area1 normalization",
                cause_type="Foams105SizeClassValidationError",
                cause_code="nonpositive_area1_normalization_unavailable",
                context={"group_id": group.group_id, "source_slot": group.source_slot},
            )
        na = group.normalization_result.na_per_mm2
        binned_groups.append(Foams105BinnedGroupInput(group.group_id, group.source_slot, na))
        cutoff_groups.append(Foams105CutoffGroupInput(group.group_id, group.scale_px_per_mm, na))

    suggestion = None
    if settings.selection_mode == "autosmart":
        if settings.manual_ranges:
            raise _pipeline_validation("analysis_settings", "contradictory_selection", "autosmart mode rejects supplied manual_ranges")
        suggestion = _stage(
            "autosmart_cutoffs", suggest_foams105_cutoffs,
            prepared.label_preparation_result.bin_labels_mm, tuple(cutoff_groups),
        )
        binned_selection = _stage("binned_selection", select_foams105_suggested_ranges, suggestion)
    else:
        if not isinstance(settings.manual_ranges, tuple) or len(settings.manual_ranges) != len(prepared.groups):
            raise _pipeline_validation(
                "analysis_settings", "manual_ranges_required", "manual mode requires one ordered range/None per source group",
                expected_count=len(prepared.groups),
            )
        binned_selection = _stage(
            "binned_selection", select_foams105_binned_ranges,
            prepared.label_preparation_result.bin_labels_mm, tuple(binned_groups), settings.manual_ranges,
        )

    raw_groups = tuple(
        Foams105RawGroupInput(group.group_id, group.source_slot, group.diameter_filtered_objects)
        for group in prepared.groups
    )
    raw_selection = _stage(
        "raw_selection", select_foams105_raw_objects,
        prepared.label_preparation_result.bin_labels_mm,
        raw_groups,
        binned_selection.requested_ranges,
    )
    vesicularity_groups = tuple(
        Foams105VesicularityGroupInput(
            group_id=group.group_id,
            source_slot=group.source_slot,
            image_area_results=tuple(image.image_area_result for image in group.images),
            unfiltered_pore_objects=group.unfiltered_pore_objects,
        )
        for group in prepared.groups
    )
    vesicularity = _stage("vesicularity", calculate_foams105_2d_vesicularity, raw_selection, vesicularity_groups)
    conversion = _stage("nv_conversion", convert_foams105_nv, binned_selection.bin_labels_mm, binned_selection.na_per_mm2)
    volume_kwargs = {}
    if settings.volume_normalization_mode == "integrated_2d":
        if settings.known_vesicularity_percent is not None:
            raise _pipeline_validation(
                "analysis_settings", "contradictory_normalization",
                "integrated_2d mode rejects known_vesicularity_percent",
            )
        volume_kwargs["vesicularity_result"] = vesicularity
    elif settings.volume_normalization_mode == "known_vesicularity":
        volume_kwargs["known_vesicularity_percent"] = settings.known_vesicularity_percent
    elif settings.known_vesicularity_percent is not None:
        raise _pipeline_validation("analysis_settings", "contradictory_normalization", "unnormalized mode rejects known_vesicularity_percent")
    volume = _stage(
        "volume_distribution", calculate_foams105_volume_distribution,
        conversion, settings.volume_normalization_mode, **volume_kwargs,
    )
    summary_kwargs = {"vesicularity_result": vesicularity} if settings.volume_normalization_mode == "unnormalized" else {}
    distribution_summary = _stage(
        "distribution_summary", summarize_foams105_distribution,
        volume, prepared.label_preparation_result.bin_labels_mm, **summary_kwargs,
    )
    categories = _stage(
        "size_categories", summarize_foams105_size_categories,
        prepared.label_preparation_result.bin_labels_mm,
        count_groups,
        lower_cutoff,
        upper_cutoff,
    )

    moment_results = tuple(image.moment_geometry_result for group in prepared.groups for image in group.images)
    perimeter_results = tuple(image.perimeter_result for group in prepared.groups for image in group.images)
    cached_shape_measures = tuple(
        measure
        for group in prepared.groups
        for image in group.images
        for measure in image.component_shape_measures
    )
    shape_result = _stage(
        "shape_derivation",
        assemble_foams105_shapes_from_cached_components,
        raw_selection,
        moment_results,
        perimeter_results,
        cached_shape_measures,
    )
    scales = tuple(group.scale_px_per_mm for group in prepared.groups)
    shape_statistics = _stage(
        "shape_statistics", summarize_foams105_shapes,
        shape_result,
        configured_scales_px_per_mm=scales,
        minimum_diameter_px=shape_minimum,
        mode=settings.shape_statistics_mode,
    )
    shape_histograms = _stage("shape_histograms", build_foams105_shape_histograms, shape_statistics)
    shape_all_statistics = _stage(
        "shape_export_population", summarize_foams105_shapes,
        shape_result,
        configured_scales_px_per_mm=scales,
        minimum_diameter_px=0.0,
        mode="updated_minimum",
    )
    shape_export_histograms = _stage("shape_export_histograms", build_foams105_shape_histograms, shape_all_statistics)
    bulk_na = _stage(
        "bulk_na_plot_data",
        build_foams105_bulk_na_plot_data,
        binned_selection,
        count_groups,
        group_areas,
        suggestion_result=suggestion,
    )

    moment_by_identity = {
        (item.image_id, item.component_id): item
        for result in moment_results
        for item in result.components
    }
    perimeter_by_identity = {
        (item.image_id, item.component_id): item
        for result in perimeter_results
        for item in result.components
    }
    raw_rows = []
    raw_output_index = 0
    for group in prepared.groups:
        for filtered_index, object_ in enumerate(group.diameter_filtered_objects):
            identity = (object_.image_id, object_.component_id)
            source_object_index = group.diameter_filter_result.retained_indices[filtered_index]
            raw_rows.append(
                _raw_table_row(
                    raw_output_index,
                    group,
                    source_object_index,
                    object_,
                    moment_by_identity[identity],
                    perimeter_by_identity[identity],
                )
            )
            raw_output_index += 1
    rawcutoffs_rows = []
    for raw_row in raw_selection.rows:
        identity = (raw_row.image_id, raw_row.component_id)
        group = prepared.groups[raw_row.source_slot - 1]
        object_ = next(
            item for item in group.diameter_filtered_objects
            if (item.image_id, item.component_id) == identity
        )
        rawcutoffs_rows.append(
            _raw_table_row(
                raw_row.output_index,
                group,
                raw_row.source_object_index,
                object_,
                moment_by_identity[identity],
                perimeter_by_identity[identity],
            )
        )

    return Foams105AnalysisResult(
        prepared_result=prepared,
        analysis_settings=replace(settings, size_lower_cutoff_mm=lower_cutoff, size_upper_cutoff_mm=upper_cutoff, shape_minimum_diameter_px=shape_minimum),
        cutoff_suggestion_result=suggestion,
        applied_ranges=binned_selection.requested_ranges,
        binned_selection_result=binned_selection,
        raw_selection_result=raw_selection,
        vesicularity_result=vesicularity,
        conversion_result=conversion,
        volume_result=volume,
        distribution_summary_result=distribution_summary,
        size_category_result=categories,
        shape_result=shape_result,
        shape_statistics_result=shape_statistics,
        shape_histogram_result=shape_histograms,
        shape_export_histogram_result=shape_export_histograms,
        bulk_na_plot_data_result=bulk_na,
        raw_data_rows=tuple(raw_rows),
        rawcutoffs_data_rows=tuple(rawcutoffs_rows),
        image_count=prepared.image_count,
    )


def run_foams105_analysis(
    groups: Sequence[Foams105PipelineGroupInput],
    preparation_settings: Foams105PreparationSettings,
    analysis_settings: Foams105AnalysisSettings,
    *,
    source_metadata: Optional[Foams105SourceMetadata] = None,
) -> Foams105AnalysisResult:
    """Convenience headless run; callers may retain preparation for reanalysis."""

    prepared = prepare_foams105_analysis(
        groups,
        preparation_settings,
        source_metadata=source_metadata,
    )
    return analyze_foams105_prepared(prepared, analysis_settings)
