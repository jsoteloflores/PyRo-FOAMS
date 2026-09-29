"""Diameter classes and area-normalized two-dimensional pore densities.

Bins are left-closed and right-open, except that the final bin includes its
upper edge. An image contributes to a bin only when its detection window
covers that entire bin. ``N_A`` is the bin-integrated count per square
millimeter; it is not divided by bin width.

Example::

    bins = create_diameter_bin_spec([0.1, 0.2, 0.4])
    result = calculate_2d_number_densities(dataset, bins)
    group = result.groups[("sample-1", "survey")]
    print(group.counts, group.eligible_areas_mm2, group.number_densities_per_mm2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .sampling import ImageSamplingRecord, SamplingDataset

DETECTION_ELIGIBILITY_POLICY = "whole-bin detection-window coverage"
PoreIdentity = Tuple[int, int]
ImageIdentity = Tuple[str, str]
GroupIdentity = Tuple[str, str]


class DistributionValidationError(ValueError):
    """Raised when a diameter grid or sampling input is invalid."""


def _validated_real(value: object, message: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise DistributionValidationError(message)
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise DistributionValidationError(message)
    return numeric


@dataclass(frozen=True)
class DiameterBin:
    """One diameter interval in canonical millimeters."""

    lower_mm: float
    upper_mm: float
    width_mm: float
    geometric_midpoint_mm: float


@dataclass(frozen=True)
class DiameterBinSpec:
    """An immutable, explicit diameter grid shared by every result group."""

    edges_mm: Tuple[float, ...]
    bins: Tuple[DiameterBin, ...]
    length_unit: str = "mm"
    interval_convention: str = "[lower, upper), final bin includes upper edge"


def create_diameter_bin_spec(edges_mm: Sequence[float]) -> DiameterBinSpec:
    """Validate and copy explicit diameter edges expressed in millimeters."""
    try:
        raw_edges = tuple(edges_mm)
    except TypeError as exc:
        raise DistributionValidationError("Bin edges must be a one-dimensional sequence") from exc
    if any(isinstance(edge, (bool, np.bool_)) for edge in raw_edges):
        raise DistributionValidationError("Bin edges must be numeric, not Boolean")
    try:
        edges = tuple(float(edge) for edge in raw_edges)
    except (TypeError, ValueError) as exc:
        raise DistributionValidationError("Bin edges must be numeric") from exc
    if len(edges) < 2:
        raise DistributionValidationError("At least two bin edges are required")
    if any(not math.isfinite(edge) or edge <= 0 for edge in edges):
        raise DistributionValidationError("Bin edges must be positive and finite")
    if any(upper <= lower for lower, upper in zip(edges, edges[1:])):
        raise DistributionValidationError("Bin edges must be strictly increasing")
    bins = tuple(
        DiameterBin(
            lower_mm=lower,
            upper_mm=upper,
            width_mm=upper - lower,
            geometric_midpoint_mm=math.sqrt(lower * upper),
        )
        for lower, upper in zip(edges, edges[1:])
    )
    return DiameterBinSpec(edges_mm=edges, bins=bins)


def geometric_diameter_bin_spec(
    start_diameter_mm: float,
    number_of_bins: int,
    log10_step: float = 0.1,
) -> DiameterBinSpec:
    """Create ``edge[k] = start * 10**(k * step)`` for ``k=0..bins``."""
    if isinstance(start_diameter_mm, (bool, np.bool_)) or isinstance(
        log10_step, (bool, np.bool_)
    ):
        raise DistributionValidationError("Grid start and step must be numeric, not Boolean")
    try:
        start = float(start_diameter_mm)
        step = float(log10_step)
    except (TypeError, ValueError) as exc:
        raise DistributionValidationError("Grid start and step must be numeric") from exc
    if not math.isfinite(start) or start <= 0:
        raise DistributionValidationError("start_diameter_mm must be positive and finite")
    if not math.isfinite(step) or step <= 0:
        raise DistributionValidationError("log10_step must be positive and finite")
    if (
        not isinstance(number_of_bins, int)
        or isinstance(number_of_bins, bool)
        or number_of_bins <= 0
    ):
        raise DistributionValidationError("number_of_bins must be a positive integer")

    edges = []
    for index in range(number_of_bins + 1):
        try:
            edge = start * 10.0 ** (index * step)
        except OverflowError as exc:
            raise DistributionValidationError("Generated bin grid overflowed") from exc
        if not math.isfinite(edge):
            raise DistributionValidationError("Generated bin grid contains a nonfinite edge")
        edges.append(edge)
    return create_diameter_bin_spec(edges)


@dataclass(frozen=True)
class DistributionDiagnostics:
    """Mutually exclusive outcomes for every input pore measurement.

    Precedence is excluded border, below grid, above grid, then inside-grid but
    unsupported by that pore's image. Brief 01 domain omissions remain separate.
    """

    excluded_border: Tuple[PoreIdentity, ...] = ()
    below_grid: Tuple[PoreIdentity, ...] = ()
    above_grid: Tuple[PoreIdentity, ...] = ()
    unsupported_by_image: Tuple[PoreIdentity, ...] = ()
    omitted_domain: Tuple[PoreIdentity, ...] = ()

    @property
    def uncounted_input_count(self) -> int:
        return (
            len(self.excluded_border)
            + len(self.below_grid)
            + len(self.above_grid)
            + len(self.unsupported_by_image)
        )


@dataclass(frozen=True)
class Group2DDistribution:
    """Aligned counts, denominators, and ``N_A`` values for one group."""

    sample_id: str
    magnification_group_id: str
    counts: Tuple[int, ...]
    eligible_image_counts: Tuple[int, ...]
    eligible_areas_mm2: Tuple[float, ...]
    number_densities_per_mm2: Tuple[float, ...]
    supported: Tuple[bool, ...]
    contributing_images: Tuple[Tuple[ImageIdentity, ...], ...]


@dataclass(frozen=True)
class DistributionResult:
    """Common-grid two-dimensional distributions and auditable diagnostics."""

    bin_spec: DiameterBinSpec
    groups: Mapping[GroupIdentity, Group2DDistribution]
    diagnostics: DistributionDiagnostics
    exclude_border: bool
    detection_eligibility_policy: str = DETECTION_ELIGIBILITY_POLICY
    density_unit: str = "mm^-2"


def _validate_analysis_inputs(dataset: SamplingDataset, bin_spec: DiameterBinSpec) -> None:
    if not isinstance(bin_spec, DiameterBinSpec):
        raise DistributionValidationError("bin_spec must be a DiameterBinSpec")
    validated_spec = create_diameter_bin_spec(bin_spec.edges_mm)
    if validated_spec.bins != bin_spec.bins:
        raise DistributionValidationError("bin_spec derived bin values are inconsistent")

    seen_indices = set()
    seen_identities = set()
    images_by_index = {}
    for image in dataset.images:
        identity = (image.sample_id, image.image_id)
        if image.image_index in seen_indices:
            raise DistributionValidationError(f"Duplicate image_index {image.image_index}")
        if identity in seen_identities:
            raise DistributionValidationError(f"Duplicate image identity {identity!r}")
        seen_indices.add(image.image_index)
        seen_identities.add(identity)
        images_by_index[image.image_index] = image
        _validated_real(
            image.analyzed_area_mm2,
            f"Image {image.image_id!r} analyzed area must be positive and finite",
        )
        minimum = image.min_detectable_diameter_mm
        maximum = image.max_reliable_diameter_mm
        if minimum is None:
            raise DistributionValidationError(
                f"Minimum detection diameter is unset for image {image.image_id!r}"
            )
        minimum_value = _validated_real(
            minimum,
            f"Image {image.image_id!r} minimum detection diameter must be positive and finite",
        )
        if maximum is not None:
            maximum_value = _validated_real(
                maximum,
                f"Image {image.image_id!r} maximum detection diameter is invalid",
            )
            if maximum_value <= minimum_value:
                raise DistributionValidationError(
                    f"Image {image.image_id!r} maximum detection diameter is invalid"
                )

    seen_pores = set()
    for pore in dataset.pores:
        identity = (pore.measurement.image_index, pore.measurement.label)
        if identity in seen_pores:
            raise DistributionValidationError(f"Duplicate pore identity {identity!r}")
        seen_pores.add(identity)
        image = images_by_index.get(pore.measurement.image_index)
        if image is None or pore.image != image:
            raise DistributionValidationError(
                f"Pore {identity!r} is not associated with its dataset image"
            )
        _validated_real(
            pore.equivalent_diameter_mm,
            f"Pore {identity!r} diameter must be positive and finite",
        )


def _image_covers_bin(image: ImageSamplingRecord, bin_: DiameterBin) -> bool:
    minimum = image.min_detectable_diameter_mm
    maximum = image.max_reliable_diameter_mm
    return minimum <= bin_.lower_mm and (maximum is None or maximum >= bin_.upper_mm)


def _bin_index(diameter_mm: float, edges: Tuple[float, ...]) -> int:
    if diameter_mm == edges[-1]:
        return len(edges) - 2
    return int(np.searchsorted(edges, diameter_mm, side="right") - 1)


def calculate_2d_number_densities(
    dataset: SamplingDataset,
    bin_spec: DiameterBinSpec,
    *,
    exclude_border: bool = False,
) -> DistributionResult:
    """Calculate per-sample/group bin counts and area-weighted ``N_A``.

    Border exclusion, when requested, removes observations but never sampled
    area and is not an unbiased boundary correction.
    """
    if not isinstance(exclude_border, bool):
        raise DistributionValidationError("exclude_border must be Boolean")
    dataset.require_detection_limits()
    _validate_analysis_inputs(dataset, bin_spec)

    group_images: Dict[GroupIdentity, list[ImageSamplingRecord]] = {}
    for image in dataset.images:
        key = (image.sample_id, image.magnification_group_id)
        group_images.setdefault(key, []).append(image)

    group_counts = {key: [0] * len(bin_spec.bins) for key in group_images}
    group_image_counts: Dict[GroupIdentity, list[int]] = {}
    group_areas: Dict[GroupIdentity, list[float]] = {}
    group_contributors: Dict[GroupIdentity, list[Tuple[ImageIdentity, ...]]] = {}
    eligibility: Dict[int, Tuple[bool, ...]] = {}

    for key, images in group_images.items():
        image_counts = []
        areas = []
        contributors = []
        for bin_index, bin_ in enumerate(bin_spec.bins):
            eligible = [image for image in images if _image_covers_bin(image, bin_)]
            for image in images:
                current = list(eligibility.get(image.image_index, (False,) * len(bin_spec.bins)))
                current[bin_index] = image in eligible
                eligibility[image.image_index] = tuple(current)
            area = math.fsum(image.analyzed_area_mm2 for image in eligible)
            if not math.isfinite(area):
                raise DistributionValidationError(f"Accumulated area is nonfinite for group {key!r}")
            image_counts.append(len(eligible))
            areas.append(area)
            contributors.append(tuple((image.sample_id, image.image_id) for image in eligible))
        group_image_counts[key] = image_counts
        group_areas[key] = areas
        group_contributors[key] = contributors

    excluded_border = []
    below_grid = []
    above_grid = []
    unsupported = []
    for pore in dataset.pores:
        identity = (pore.measurement.image_index, pore.measurement.label)
        diameter = pore.equivalent_diameter_mm
        if exclude_border and pore.measurement.touches_border:
            excluded_border.append(identity)
        elif diameter < bin_spec.edges_mm[0]:
            below_grid.append(identity)
        elif diameter > bin_spec.edges_mm[-1]:
            above_grid.append(identity)
        else:
            bin_index = _bin_index(diameter, bin_spec.edges_mm)
            if not eligibility[pore.image.image_index][bin_index]:
                unsupported.append(identity)
            else:
                key = (pore.sample_id, pore.magnification_group_id)
                group_counts[key][bin_index] += 1

    groups = {}
    for key in group_images:
        counts = tuple(group_counts[key])
        areas = tuple(group_areas[key])
        supported = tuple(area > 0 for area in areas)
        densities = tuple(
            count / area if is_supported else math.nan
            for count, area, is_supported in zip(counts, areas, supported)
        )
        if any(not math.isfinite(value) for value, flag in zip(densities, supported) if flag):
            raise DistributionValidationError(f"Calculated density is nonfinite for group {key!r}")
        groups[key] = Group2DDistribution(
            sample_id=key[0],
            magnification_group_id=key[1],
            counts=counts,
            eligible_image_counts=tuple(group_image_counts[key]),
            eligible_areas_mm2=areas,
            number_densities_per_mm2=densities,
            supported=supported,
            contributing_images=tuple(group_contributors[key]),
        )

    diagnostics = DistributionDiagnostics(
        excluded_border=tuple(excluded_border),
        below_grid=tuple(below_grid),
        above_grid=tuple(above_grid),
        unsupported_by_image=tuple(unsupported),
        omitted_domain=tuple(dataset.omitted_pore_identities),
    )
    return DistributionResult(
        bin_spec=bin_spec,
        groups=MappingProxyType(groups),
        diagnostics=diagnostics,
        exclude_border=exclude_border,
    )
