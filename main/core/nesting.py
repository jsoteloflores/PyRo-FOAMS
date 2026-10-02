"""Explicit manual selection of 2D density bins across magnification groups.

Segments use absolute, half-open bin ranges on the source grid. Each selected
bin retains one source group's count, area, density, and contributors without
pooling or interpolation. Overlap comparisons are advisory and never alter the
selection.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .distributions import (
    DETECTION_ELIGIBILITY_POLICY,
    INTERVAL_CONVENTION,
    LENGTH_UNIT,
    DiameterBin,
    DiameterBinSpec,
    DistributionDiagnostics,
    DistributionResult,
    Group2DDistribution,
    ImageIdentity,
    create_diameter_bin_spec,
)
from .sampling import ImageSamplingRecord

DENSITY_REL_TOLERANCE = 1e-12
DENSITY_ABS_TOLERANCE = 1e-15
AREA_REL_TOLERANCE = 1e-12
AREA_ABS_TOLERANCE = 0.0
NESTING_METHOD = "manual_group_selection_v1"


class NestingValidationError(ValueError):
    """Raised when a nesting plan or its source distribution is inconsistent."""


@dataclass(frozen=True)
class NestingSegment:
    """Select one magnification group for ``[start_bin, stop_bin)``."""

    group_id: str
    start_bin: int
    stop_bin: int


@dataclass(frozen=True)
class NestingPlan:
    """Explicit ordered selection over one sample and one source-grid range."""

    sample_id: str
    start_bin: int
    stop_bin: int
    segments: Tuple[NestingSegment, ...]


@dataclass(frozen=True)
class NestingTransition:
    """Boundary where ownership changes to the right-hand group."""

    left_group_id: str
    right_group_id: str
    edge_index: int
    diameter_mm: float


@dataclass(frozen=True)
class OverlapBinComparison:
    """Same-bin density comparison between two adjacent selected groups."""

    bin_index: int
    lower_mm: float
    upper_mm: float
    left_count: int
    right_count: int
    left_eligible_area_mm2: float
    right_eligible_area_mm2: float
    left_density_per_mm2: float
    right_density_per_mm2: float
    left_contributing_image_count: int
    right_contributing_image_count: int
    delta_per_mm2: float
    symmetric_relative_difference: float | None
    both_zero: bool


@dataclass(frozen=True)
class TransitionOverlapDiagnostics:
    """Advisory overlap evidence for one selected transition."""

    transition: NestingTransition
    common_supported_bin_count: int
    common_supported_bin_indices: Tuple[int, ...]
    informative_bin_count: int
    informative_bin_indices: Tuple[int, ...]
    comparisons: Tuple[OverlapBinComparison, ...]
    below_transition_common_supported: bool
    above_transition_common_supported: bool
    no_shared_supported_bins: bool
    no_informative_overlap: bool
    transition_not_bracketed_by_shared_support: bool


@dataclass(frozen=True)
class Nested2DDistribution:
    """Auditable manual composite retaining one complete source row per bin."""

    sample_id: str
    bin_spec: DiameterBinSpec
    plan: NestingPlan
    selected_bin_indices: Tuple[int, ...]
    transitions: Tuple[NestingTransition, ...]
    lower_edges_mm: Tuple[float, ...]
    upper_edges_mm: Tuple[float, ...]
    counts: Tuple[int, ...]
    eligible_image_counts: Tuple[int, ...]
    eligible_areas_mm2: Tuple[float, ...]
    number_densities_per_mm2: Tuple[float, ...]
    source_group_ids: Tuple[str, ...]
    contributing_images: Tuple[Tuple[ImageIdentity, ...], ...]
    source_images: Tuple[ImageSamplingRecord, ...]
    source_diagnostics: DistributionDiagnostics
    overlap_diagnostics: Tuple[TransitionOverlapDiagnostics, ...]
    exclude_border: bool
    detection_eligibility_policy: str
    density_unit: str
    method: str = NESTING_METHOD


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NestingValidationError(f"{field} must be a non-empty string")
    return value


def _index(value: object, field: str, number_of_bins: int) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
    ):
        raise NestingValidationError(f"{field} must be an integer")
    result = int(value)
    if result < 0 or result > number_of_bins:
        raise NestingValidationError(
            f"{field} {result} is outside 0..{number_of_bins}"
        )
    return result


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _validate_bin_spec(bin_spec: object) -> DiameterBinSpec:
    if not isinstance(bin_spec, DiameterBinSpec):
        raise NestingValidationError("distribution bin_spec must be a DiameterBinSpec")
    if not isinstance(bin_spec.edges_mm, tuple):
        raise NestingValidationError("bin_spec.edges_mm must be an immutable tuple")
    if not isinstance(bin_spec.bins, tuple):
        raise NestingValidationError("bin_spec.bins must be an immutable tuple")
    for index, edge in enumerate(bin_spec.edges_mm):
        if not _is_real_number(edge) or not math.isfinite(edge) or edge <= 0:
            raise NestingValidationError(
                f"bin_spec.edges_mm[{index}] must be a positive finite real number"
            )
    for index, bin_ in enumerate(bin_spec.bins):
        if not isinstance(bin_, DiameterBin):
            raise NestingValidationError(
                f"bin_spec.bins[{index}] must be a DiameterBin"
            )
    try:
        expected = create_diameter_bin_spec(bin_spec.edges_mm)
    except ValueError as exc:
        raise NestingValidationError(f"Invalid source bin grid: {exc}") from exc
    if bin_spec.length_unit != LENGTH_UNIT:
        raise NestingValidationError("Source bin grid length_unit must be 'mm'")
    if bin_spec.interval_convention != INTERVAL_CONVENTION:
        raise NestingValidationError("Source bin grid interval convention is unsupported")
    if len(bin_spec.bins) != len(expected.bins):
        raise NestingValidationError("Source bin grid has inconsistent derived bins")
    for index, (supplied, derived) in enumerate(zip(bin_spec.bins, expected.bins)):
        supplied_values = (
            supplied.lower_mm,
            supplied.upper_mm,
            supplied.width_mm,
            supplied.geometric_midpoint_mm,
        )
        expected_values = (
            derived.lower_mm,
            derived.upper_mm,
            derived.width_mm,
            derived.geometric_midpoint_mm,
        )
        if any(
            not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_))
            or not math.isfinite(float(value))
            or not math.isclose(
                float(value), expected_value, rel_tol=1e-12, abs_tol=0.0
            )
            for value, expected_value in zip(supplied_values, expected_values)
        ):
            raise NestingValidationError(
                f"Source bin {index} metadata is inconsistent with its edges"
            )
    return bin_spec


def _validate_plan(plan: object, number_of_bins: int) -> NestingPlan:
    if not isinstance(plan, NestingPlan):
        raise NestingValidationError("plan must be a NestingPlan")
    sample_id = _identifier(plan.sample_id, "plan sample_id")
    start = _index(plan.start_bin, "plan start_bin", number_of_bins)
    stop = _index(plan.stop_bin, "plan stop_bin", number_of_bins)
    if start >= stop:
        raise NestingValidationError(
            f"Plan for sample {sample_id!r} must have start_bin < stop_bin"
        )
    if not isinstance(plan.segments, tuple) or not plan.segments:
        raise NestingValidationError(
            f"Plan for sample {sample_id!r} must contain at least one segment"
        )

    expected_start = start
    seen_groups = set()
    for position, segment in enumerate(plan.segments):
        if not isinstance(segment, NestingSegment):
            raise NestingValidationError(f"Plan segment {position} is not a NestingSegment")
        group_id = _identifier(segment.group_id, f"segment {position} group_id")
        segment_start = _index(
            segment.start_bin, f"segment {position} start_bin", number_of_bins
        )
        segment_stop = _index(
            segment.stop_bin, f"segment {position} stop_bin", number_of_bins
        )
        if segment_start >= segment_stop:
            raise NestingValidationError(
                f"Segment {position} for group {group_id!r} is empty or reversed"
            )
        if segment_start != expected_start:
            relation = "overlaps" if segment_start < expected_start else "leaves a gap before"
            raise NestingValidationError(
                f"Segment {position} for group {group_id!r} {relation} bin {expected_start}"
            )
        if group_id in seen_groups:
            raise NestingValidationError(
                f"Magnification group {group_id!r} is used more than once"
            )
        seen_groups.add(group_id)
        expected_start = segment_stop
    if expected_start != stop:
        raise NestingValidationError(
            f"Plan segments stop at bin {expected_start}, not plan stop_bin {stop}"
        )
    return plan


def _validate_source_image(image: object, sample_id: str, group_id: str) -> None:
    if not isinstance(image, ImageSamplingRecord):
        raise NestingValidationError(
            f"Source provenance for sample {sample_id!r}, group {group_id!r} "
            "contains a non-image record"
        )
    if (
        not _is_real_number(image.analyzed_area_mm2)
        or not math.isfinite(image.analyzed_area_mm2)
        or image.analyzed_area_mm2 <= 0
    ):
        raise NestingValidationError(
            f"Source image {image.image_id!r} analyzed_area_mm2 must be positive and finite"
        )
    for field in ("included_labels", "omitted_labels"):
        labels = getattr(image, field)
        if not isinstance(labels, tuple):
            raise NestingValidationError(
                f"Source image {image.image_id!r} {field} must be an immutable tuple"
            )
        if any(
            not isinstance(label, (int, np.integer))
            or isinstance(label, (bool, np.bool_))
            or label <= 0
            for label in labels
        ):
            raise NestingValidationError(
                f"Source image {image.image_id!r} {field} must contain positive integer labels"
            )
        if len(set(labels)) != len(labels):
            raise NestingValidationError(
                f"Source image {image.image_id!r} {field} contains duplicate labels"
            )
    try:
        image.__post_init__()
    except (TypeError, ValueError) as exc:
        raise NestingValidationError(
            f"Source image {getattr(image, 'image_id', None)!r} is inconsistent: {exc}"
        ) from exc
    if image.sample_id != sample_id or image.magnification_group_id != group_id:
        raise NestingValidationError(
            f"Source image {image.image_id!r} does not match sample {sample_id!r}, "
            f"group {group_id!r}"
        )
    if not isinstance(image.image_id, str) or not image.image_id.strip():
        raise NestingValidationError("Source image_id must be a non-empty string")
    minimum = image.min_detectable_diameter_mm
    maximum = image.max_reliable_diameter_mm
    if minimum is None or not math.isfinite(minimum) or minimum <= 0:
        raise NestingValidationError(
            f"Source image {image.image_id!r} has invalid minimum detection diameter"
        )
    if maximum is not None and (
        not math.isfinite(maximum) or maximum <= minimum
    ):
        raise NestingValidationError(
            f"Source image {image.image_id!r} has invalid maximum detection diameter"
        )


def _validate_source_images(source_images: Tuple[ImageSamplingRecord, ...]) -> None:
    seen_indices = set()
    seen_identities = set()
    for position, image in enumerate(source_images):
        if not isinstance(image, ImageSamplingRecord):
            raise NestingValidationError(
                f"Source provenance position {position} is not an ImageSamplingRecord"
            )
        _validate_source_image(
            image, image.sample_id, image.magnification_group_id
        )
        identity = (image.sample_id, image.image_id)
        if image.image_index in seen_indices:
            raise NestingValidationError(
                f"Duplicate source image_index {image.image_index}"
            )
        if identity in seen_identities:
            raise NestingValidationError(
                f"Duplicate source image identity {identity!r}"
            )
        seen_indices.add(image.image_index)
        seen_identities.add(identity)


def _validate_diagnostics(diagnostics: object) -> DistributionDiagnostics:
    if not isinstance(diagnostics, DistributionDiagnostics):
        raise NestingValidationError(
            "distribution diagnostics must be DistributionDiagnostics"
        )
    for field in (
        "excluded_border",
        "below_grid",
        "above_grid",
        "unsupported_by_image",
        "omitted_domain",
    ):
        identities = getattr(diagnostics, field)
        if not isinstance(identities, tuple):
            raise NestingValidationError(
                f"distribution diagnostics {field} must be an immutable tuple"
            )
        for position, identity in enumerate(identities):
            if (
                not isinstance(identity, tuple)
                or len(identity) != 2
                or any(
                    not isinstance(value, (int, np.integer))
                    or isinstance(value, (bool, np.bool_))
                    for value in identity
                )
            ):
                raise NestingValidationError(
                    f"distribution diagnostics {field}[{position}] must be a "
                    "two-integer pore identity"
                )
    return diagnostics


def _validate_group_keys(groups: Mapping) -> None:
    for key in groups:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(not isinstance(value, str) or not value.strip() for value in key)
        ):
            raise NestingValidationError(
                f"distribution group key {key!r} must be a two-string "
                "(sample_id, magnification_group_id) tuple"
            )


def _validate_group(
    group: object,
    sample_id: str,
    group_id: str,
    bin_spec: DiameterBinSpec,
    source_images: Tuple[ImageSamplingRecord, ...],
) -> Group2DDistribution:
    if not isinstance(group, Group2DDistribution):
        raise NestingValidationError(
            f"Source group {group_id!r} for sample {sample_id!r} has invalid type"
        )
    if group.sample_id != sample_id or group.magnification_group_id != group_id:
        raise NestingValidationError(
            f"Source group {group_id!r} identity does not match sample {sample_id!r}"
        )
    fields = {
        "counts": group.counts,
        "eligible_image_counts": group.eligible_image_counts,
        "eligible_areas_mm2": group.eligible_areas_mm2,
        "number_densities_per_mm2": group.number_densities_per_mm2,
        "supported": group.supported,
        "contributing_images": group.contributing_images,
    }
    number_of_bins = len(bin_spec.bins)
    for field, values in fields.items():
        if not isinstance(values, tuple) or len(values) != number_of_bins:
            raise NestingValidationError(
                f"Source group {group_id!r} field {field} must contain "
                f"{number_of_bins} aligned bins"
            )

    relevant_images = tuple(
        image
        for image in source_images
        if image.sample_id == sample_id
        and image.magnification_group_id == group_id
    )
    seen_image_ids = set()
    for image in relevant_images:
        _validate_source_image(image, sample_id, group_id)
        if image.image_id in seen_image_ids:
            raise NestingValidationError(
                f"Duplicate source image identity {(sample_id, image.image_id)!r}"
            )
        seen_image_ids.add(image.image_id)

    for bin_index, bin_ in enumerate(bin_spec.bins):
        count = group.counts[bin_index]
        eligible_count = group.eligible_image_counts[bin_index]
        area = group.eligible_areas_mm2[bin_index]
        density = group.number_densities_per_mm2[bin_index]
        supported = group.supported[bin_index]
        contributors = group.contributing_images[bin_index]
        prefix = (
            f"Sample {sample_id!r}, group {group_id!r}, bin {bin_index}"
        )
        if (
            not isinstance(count, (int, np.integer))
            or isinstance(count, (bool, np.bool_))
            or count < 0
        ):
            raise NestingValidationError(f"{prefix} count must be a nonnegative integer")
        if (
            not isinstance(eligible_count, (int, np.integer))
            or isinstance(eligible_count, (bool, np.bool_))
            or eligible_count < 0
        ):
            raise NestingValidationError(
                f"{prefix} eligible image count must be a nonnegative integer"
            )
        if not isinstance(supported, (bool, np.bool_)):
            raise NestingValidationError(f"{prefix} supported flag must be Boolean")
        if not isinstance(contributors, tuple):
            raise NestingValidationError(f"{prefix} contributors must be a tuple")
        if any(
            not isinstance(identity, tuple)
            or len(identity) != 2
            or not all(isinstance(value, str) and value.strip() for value in identity)
            for identity in contributors
        ):
            raise NestingValidationError(f"{prefix} contains an invalid contributor identity")
        if len(set(contributors)) != len(contributors):
            raise NestingValidationError(f"{prefix} contains duplicate contributors")
        if int(eligible_count) != len(contributors):
            raise NestingValidationError(
                f"{prefix} eligible image count does not match contributors"
            )

        expected_images = tuple(
            image
            for image in relevant_images
            if image.min_detectable_diameter_mm <= bin_.lower_mm
            and (
                image.max_reliable_diameter_mm is None
                or image.max_reliable_diameter_mm >= bin_.upper_mm
            )
        )
        expected_contributors = tuple(
            (image.sample_id, image.image_id) for image in expected_images
        )
        if contributors != expected_contributors:
            raise NestingValidationError(
                f"{prefix} contributors do not match source-image eligibility"
            )
        try:
            expected_area = math.fsum(
                image.analyzed_area_mm2 for image in expected_images
            )
        except OverflowError as exc:
            raise NestingValidationError(
                f"{prefix} source contributor area accumulation overflowed"
            ) from exc
        area_is_number = _is_real_number(area)
        density_is_number = _is_real_number(density)
        if bool(supported):
            if (
                not expected_images
                or int(eligible_count) <= 0
                or not math.isfinite(expected_area)
                or expected_area <= 0
            ):
                raise NestingValidationError(
                    f"{prefix} supported bin requires positive source support"
                )
            if not area_is_number or not math.isfinite(area) or area <= 0:
                raise NestingValidationError(f"{prefix} supported area must be positive and finite")
            if not math.isclose(
                area,
                expected_area,
                rel_tol=AREA_REL_TOLERANCE,
                abs_tol=AREA_ABS_TOLERANCE,
            ):
                raise NestingValidationError(
                    f"{prefix} eligible area does not match source contributors"
                )
            if not density_is_number or not math.isfinite(density) or density < 0:
                raise NestingValidationError(
                    f"{prefix} supported density must be finite and nonnegative"
                )
            expected_density = int(count) / float(area)
            if not math.isclose(
                density,
                expected_density,
                rel_tol=DENSITY_REL_TOLERANCE,
                abs_tol=DENSITY_ABS_TOLERANCE,
            ):
                raise NestingValidationError(
                    f"{prefix} density does not agree with count / eligible area"
                )
        elif not (
            int(count) == 0
            and int(eligible_count) == 0
            and expected_area == 0.0
            and area_is_number
            and area == 0.0
            and density_is_number
            and math.isnan(density)
        ):
            raise NestingValidationError(
                f"{prefix} unsupported bin must have zero count/area/contributors and NaN density"
            )
    return group


def _symmetric_relative_difference(left: float, right: float) -> float | None:
    scale = max(left, right)
    if scale == 0.0:
        return None
    scaled_left = left / scale
    scaled_right = right / scale
    return 2.0 * (scaled_right - scaled_left) / (scaled_right + scaled_left)


def _overlap_diagnostics(
    transition: NestingTransition,
    left: Group2DDistribution,
    right: Group2DDistribution,
    bin_spec: DiameterBinSpec,
    start_bin: int,
    stop_bin: int,
) -> TransitionOverlapDiagnostics:
    comparisons = []
    informative = []
    common = []
    for bin_index in range(start_bin, stop_bin):
        if not (left.supported[bin_index] and right.supported[bin_index]):
            continue
        left_density = left.number_densities_per_mm2[bin_index]
        right_density = right.number_densities_per_mm2[bin_index]
        both_zero = left_density == 0.0 and right_density == 0.0
        common.append(bin_index)
        if not both_zero:
            informative.append(bin_index)
        bin_ = bin_spec.bins[bin_index]
        comparisons.append(
            OverlapBinComparison(
                bin_index=bin_index,
                lower_mm=bin_.lower_mm,
                upper_mm=bin_.upper_mm,
                left_count=left.counts[bin_index],
                right_count=right.counts[bin_index],
                left_eligible_area_mm2=left.eligible_areas_mm2[bin_index],
                right_eligible_area_mm2=right.eligible_areas_mm2[bin_index],
                left_density_per_mm2=left_density,
                right_density_per_mm2=right_density,
                left_contributing_image_count=left.eligible_image_counts[bin_index],
                right_contributing_image_count=right.eligible_image_counts[bin_index],
                delta_per_mm2=right_density - left_density,
                symmetric_relative_difference=_symmetric_relative_difference(
                    left_density, right_density
                ),
                both_zero=both_zero,
            )
        )
    common_set = set(common)
    below = transition.edge_index - 1 in common_set
    above = transition.edge_index in common_set
    return TransitionOverlapDiagnostics(
        transition=transition,
        common_supported_bin_count=len(common),
        common_supported_bin_indices=tuple(common),
        informative_bin_count=len(informative),
        informative_bin_indices=tuple(informative),
        comparisons=tuple(comparisons),
        below_transition_common_supported=below,
        above_transition_common_supported=above,
        no_shared_supported_bins=not common,
        no_informative_overlap=not informative,
        transition_not_bracketed_by_shared_support=not (below and above),
    )


def nest_2d_distribution(
    distribution_result: DistributionResult,
    plan: NestingPlan,
) -> Nested2DDistribution:
    """Apply an explicit manual group plan without pooling or changing values.

    Source bins follow the original ``[lower, upper)`` convention, with the
    final source bin including its upper edge. Plan ranges are half-open bin
    index ranges. Invalid selected bins raise :class:`NestingValidationError`;
    absent or weak overlap only sets advisory diagnostic flags.
    """
    if not isinstance(distribution_result, DistributionResult):
        raise NestingValidationError("distribution_result must be a DistributionResult")
    bin_spec = _validate_bin_spec(distribution_result.bin_spec)
    validated_plan = _validate_plan(plan, len(bin_spec.bins))
    if not isinstance(distribution_result.groups, Mapping):
        raise NestingValidationError("distribution groups must be a mapping")
    _validate_group_keys(distribution_result.groups)
    if not isinstance(distribution_result.source_images, tuple):
        raise NestingValidationError("distribution source_images must be a tuple")
    _validate_source_images(distribution_result.source_images)
    _validate_diagnostics(distribution_result.diagnostics)
    if not isinstance(distribution_result.exclude_border, bool):
        raise NestingValidationError("distribution exclude_border must be Boolean")
    if distribution_result.detection_eligibility_policy != DETECTION_ELIGIBILITY_POLICY:
        raise NestingValidationError(
            "distribution detection_eligibility_policy must be "
            f"{DETECTION_ELIGIBILITY_POLICY!r}"
        )
    if distribution_result.density_unit != "mm^-2":
        raise NestingValidationError(
            "distribution density_unit must be 'mm^-2'"
        )

    selected_groups = {}
    for segment_index, segment in enumerate(validated_plan.segments):
        key = (validated_plan.sample_id, segment.group_id)
        if key not in distribution_result.groups:
            same_name = [
                group_key
                for group_key in distribution_result.groups
                if group_key[1] == segment.group_id
            ]
            if same_name:
                raise NestingValidationError(
                    f"Segment {segment_index} group {segment.group_id!r} belongs to "
                    f"another sample, not {validated_plan.sample_id!r}"
                )
            raise NestingValidationError(
                f"Segment {segment_index} references missing group "
                f"{segment.group_id!r} for sample {validated_plan.sample_id!r}"
            )
        selected_groups[segment.group_id] = _validate_group(
            distribution_result.groups[key],
            validated_plan.sample_id,
            segment.group_id,
            bin_spec,
            distribution_result.source_images,
        )

    selected_bin_indices = []
    lower_edges = []
    upper_edges = []
    counts = []
    eligible_image_counts = []
    eligible_areas = []
    densities = []
    source_groups = []
    contributors = []
    for segment_index, segment in enumerate(validated_plan.segments):
        group = selected_groups[segment.group_id]
        for bin_index in range(segment.start_bin, segment.stop_bin):
            if not group.supported[bin_index]:
                raise NestingValidationError(
                    f"Sample {validated_plan.sample_id!r}, segment {segment_index}, "
                    f"group {segment.group_id!r} selects unsupported bin {bin_index}"
                )
            selected_bin_indices.append(bin_index)
            lower_edges.append(bin_spec.bins[bin_index].lower_mm)
            upper_edges.append(bin_spec.bins[bin_index].upper_mm)
            counts.append(group.counts[bin_index])
            eligible_image_counts.append(group.eligible_image_counts[bin_index])
            eligible_areas.append(group.eligible_areas_mm2[bin_index])
            densities.append(group.number_densities_per_mm2[bin_index])
            source_groups.append(segment.group_id)
            contributors.append(group.contributing_images[bin_index])

    transitions = tuple(
        NestingTransition(
            left_group_id=left.group_id,
            right_group_id=right.group_id,
            edge_index=right.start_bin,
            diameter_mm=bin_spec.edges_mm[right.start_bin],
        )
        for left, right in zip(validated_plan.segments, validated_plan.segments[1:])
    )
    overlap_diagnostics = tuple(
        _overlap_diagnostics(
            transition,
            selected_groups[transition.left_group_id],
            selected_groups[transition.right_group_id],
            bin_spec,
            validated_plan.start_bin,
            validated_plan.stop_bin,
        )
        for transition in transitions
    )
    return Nested2DDistribution(
        sample_id=validated_plan.sample_id,
        bin_spec=bin_spec,
        plan=validated_plan,
        selected_bin_indices=tuple(selected_bin_indices),
        transitions=transitions,
        lower_edges_mm=tuple(lower_edges),
        upper_edges_mm=tuple(upper_edges),
        counts=tuple(counts),
        eligible_image_counts=tuple(eligible_image_counts),
        eligible_areas_mm2=tuple(eligible_areas),
        number_densities_per_mm2=tuple(densities),
        source_group_ids=tuple(source_groups),
        contributing_images=tuple(contributors),
        source_images=distribution_result.source_images,
        source_diagnostics=distribution_result.diagnostics,
        overlap_diagnostics=overlap_diagnostics,
        exclude_border=distribution_result.exclude_border,
        detection_eligibility_policy=distribution_result.detection_eligibility_policy,
        density_unit=distribution_result.density_unit,
    )
