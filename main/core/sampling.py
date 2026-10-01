"""Calibrated image sampling records for physical pore analysis.

The canonical length unit is millimeters. Input calibration and detection
limits retain their declared units, which may be ``mm``, ``um``/``µm``, or
``nm``. Existing pixel-only APIs remain independent of this stricter layer.

Example::

    low = create_image_sampling_record(
        sample_id="sample-1", image_id="low", image_index=0,
        magnification_group_id="survey", label_map=low_labels,
        calibration=0.01, calibration_unit="mm",
        min_detectable_diameter=0.05,
    )
    high = create_image_sampling_record(
        sample_id="sample-1", image_id="high", image_index=1,
        magnification_group_id="survey", label_map=high_labels,
        calibration=10.0, calibration_unit="um",
        min_detectable_diameter=50.0, parent_image_id="low",
    )
    dataset = build_sampling_dataset([low, high], low_props + high_props)
    summary = dataset.summarize_groups()[("sample-1", "survey")]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .stereology import PoreProps

CANONICAL_LENGTH_UNIT = "mm"
DIAMETER_REL_TOLERANCE = 1e-9
DIAMETER_ABS_TOLERANCE_MM = 1e-12

_MM_PER_UNIT = {
    "mm": 1.0,
    "um": 1e-3,
    "µm": 1e-3,
    "μm": 1e-3,
    "nm": 1e-6,
}


class SamplingValidationError(ValueError):
    """Base error for invalid physical sampling data."""


class UnsupportedUnitError(SamplingValidationError):
    """Raised when a physical length unit is unsupported."""


class AnalysisDomainError(SamplingValidationError):
    """Raised when an analysis domain is invalid or cuts a labeled object."""


class MeasurementMappingError(SamplingValidationError):
    """Raised when measurements cannot be joined unambiguously to images."""


def normalize_length_unit(unit: str) -> str:
    """Return the supported spelling of a declared length unit."""
    if not isinstance(unit, str) or unit not in _MM_PER_UNIT:
        raise UnsupportedUnitError(
            f"Unsupported length unit {unit!r}; expected one of mm, um/µm, or nm"
        )
    return unit


def length_to_mm(value: float, unit: str, *, field_name: str = "length") -> float:
    """Validate and convert a positive finite physical length to millimeters."""
    normalized_unit = normalize_length_unit(unit)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise SamplingValidationError(f"{field_name} must be a positive finite number") from exc
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise SamplingValidationError(f"{field_name} must be a positive finite number")
    return numeric_value * _MM_PER_UNIT[normalized_unit]


def _require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SamplingValidationError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ImageSamplingRecord:
    """Validated sampling metadata for one analyzed label-map extent.

    Detection limits are optional so records can be inventoried before those
    limits are known. When supplied, they use ``calibration_unit``.
    """

    sample_id: str
    image_id: str
    image_index: int
    magnification_group_id: str
    height_px: int
    width_px: int
    calibration: float
    calibration_unit: str
    calibration_mm_per_px: float
    analyzed_pixel_count: int
    analyzed_area_mm2: float
    min_detectable_diameter: Optional[float]
    max_reliable_diameter: Optional[float]
    min_detectable_diameter_mm: Optional[float]
    max_reliable_diameter_mm: Optional[float]
    source_id: Optional[str] = None
    parent_image_id: Optional[str] = None
    included_labels: Tuple[int, ...] = ()
    omitted_labels: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.sample_id, "sample_id")
        _require_identifier(self.image_id, "image_id")
        _require_identifier(self.magnification_group_id, "magnification_group_id")
        if not isinstance(self.image_index, int) or isinstance(self.image_index, bool):
            raise SamplingValidationError("image_index must be an integer")
        if (
            not isinstance(self.height_px, int)
            or isinstance(self.height_px, bool)
            or self.height_px <= 0
            or not isinstance(self.width_px, int)
            or isinstance(self.width_px, bool)
            or self.width_px <= 0
        ):
            raise SamplingValidationError("image dimensions must be positive integers")

        expected_calibration_mm = length_to_mm(
            self.calibration, self.calibration_unit, field_name="calibration"
        )
        if not math.isclose(
            self.calibration_mm_per_px,
            expected_calibration_mm,
            rel_tol=DIAMETER_REL_TOLERANCE,
            abs_tol=DIAMETER_ABS_TOLERANCE_MM,
        ):
            raise SamplingValidationError(
                "calibration_mm_per_px is inconsistent with the declared calibration"
            )
        if (
            not isinstance(self.analyzed_pixel_count, int)
            or isinstance(self.analyzed_pixel_count, bool)
            or self.analyzed_pixel_count <= 0
            or self.analyzed_pixel_count > self.height_px * self.width_px
        ):
            raise SamplingValidationError("analyzed_pixel_count is invalid for image dimensions")
        expected_area = self.analyzed_pixel_count * expected_calibration_mm**2
        if not math.isfinite(self.analyzed_area_mm2) or not math.isclose(
            self.analyzed_area_mm2,
            expected_area,
            rel_tol=DIAMETER_REL_TOLERANCE,
            abs_tol=DIAMETER_ABS_TOLERANCE_MM,
        ):
            raise SamplingValidationError(
                "analyzed_area_mm2 is inconsistent with pixel count and calibration"
            )

        expected_minimum = (
            None
            if self.min_detectable_diameter is None
            else length_to_mm(
                self.min_detectable_diameter,
                self.calibration_unit,
                field_name="min_detectable_diameter",
            )
        )
        expected_maximum = (
            None
            if self.max_reliable_diameter is None
            else length_to_mm(
                self.max_reliable_diameter,
                self.calibration_unit,
                field_name="max_reliable_diameter",
            )
        )
        if expected_maximum is not None and expected_minimum is None:
            raise SamplingValidationError(
                "max_reliable_diameter requires min_detectable_diameter"
            )
        if (
            expected_minimum is not None
            and expected_maximum is not None
            and expected_maximum <= expected_minimum
        ):
            raise SamplingValidationError(
                "max_reliable_diameter must be greater than min_detectable_diameter"
            )
        if self.min_detectable_diameter_mm != expected_minimum:
            raise SamplingValidationError("canonical minimum detection diameter is inconsistent")
        if self.max_reliable_diameter_mm != expected_maximum:
            raise SamplingValidationError("canonical maximum detection diameter is inconsistent")
        if set(self.included_labels) & set(self.omitted_labels):
            raise SamplingValidationError("included and omitted labels must be disjoint")

    @property
    def detection_window_is_set(self) -> bool:
        """Whether the required lower detection limit is known."""
        return self.min_detectable_diameter_mm is not None


def create_image_sampling_record(
    *,
    sample_id: str,
    image_id: str,
    image_index: int,
    magnification_group_id: str,
    label_map: np.ndarray,
    calibration: float,
    calibration_unit: str,
    min_detectable_diameter: Optional[float] = None,
    max_reliable_diameter: Optional[float] = None,
    analysis_domain_mask: Optional[np.ndarray] = None,
    source_id: Optional[str] = None,
    parent_image_id: Optional[str] = None,
) -> ImageSamplingRecord:
    """Create a validated image record from its actual analyzed label map.

    The domain mask denotes all eligible sampled space, not pore foreground.
    Labels wholly outside it are recorded as omitted; a boundary-crossing label
    is rejected because this brief defines no partial-object correction.
    """
    sample_id = _require_identifier(sample_id, "sample_id")
    image_id = _require_identifier(image_id, "image_id")
    magnification_group_id = _require_identifier(
        magnification_group_id, "magnification_group_id"
    )
    if not isinstance(image_index, int) or isinstance(image_index, bool):
        raise SamplingValidationError("image_index must be an integer")
    if not isinstance(label_map, np.ndarray) or label_map.ndim != 2:
        raise SamplingValidationError("label_map must be a two-dimensional NumPy array")
    if not np.issubdtype(label_map.dtype, np.integer):
        raise SamplingValidationError("label_map must have an integer dtype")
    height_px, width_px = label_map.shape
    if height_px <= 0 or width_px <= 0:
        raise SamplingValidationError("label_map dimensions must be positive")
    if np.any(label_map < 0):
        raise SamplingValidationError("label_map cannot contain negative labels")

    calibration_mm = length_to_mm(calibration, calibration_unit, field_name="calibration")
    normalized_unit = normalize_length_unit(calibration_unit)
    minimum_mm = (
        None
        if min_detectable_diameter is None
        else length_to_mm(
            min_detectable_diameter,
            normalized_unit,
            field_name="min_detectable_diameter",
        )
    )
    maximum_mm = (
        None
        if max_reliable_diameter is None
        else length_to_mm(
            max_reliable_diameter,
            normalized_unit,
            field_name="max_reliable_diameter",
        )
    )
    if maximum_mm is not None and minimum_mm is None:
        raise SamplingValidationError(
            "max_reliable_diameter requires min_detectable_diameter"
        )
    if minimum_mm is not None and maximum_mm is not None and maximum_mm <= minimum_mm:
        raise SamplingValidationError(
            "max_reliable_diameter must be greater than min_detectable_diameter"
        )

    if analysis_domain_mask is None:
        domain = np.ones(label_map.shape, dtype=bool)
    else:
        if not isinstance(analysis_domain_mask, np.ndarray):
            raise AnalysisDomainError("analysis_domain_mask must be a NumPy array")
        if analysis_domain_mask.shape != label_map.shape:
            raise AnalysisDomainError("analysis_domain_mask must match label_map shape")
        if analysis_domain_mask.dtype != np.bool_:
            raise AnalysisDomainError("analysis_domain_mask must have Boolean dtype")
        domain = analysis_domain_mask

    analyzed_pixel_count = int(np.count_nonzero(domain))
    if analyzed_pixel_count <= 0:
        raise AnalysisDomainError("analysis_domain_mask must contain eligible sampled pixels")

    if analyzed_pixel_count == label_map.size:
        positive_labels = np.unique(label_map)
        positive_labels = positive_labels[positive_labels > 0]
        included_labels = positive_labels.astype(int).tolist()
        omitted_labels = []
    else:
        label_values, compact_labels, object_counts = np.unique(
            label_map, return_inverse=True, return_counts=True
        )
        positive = label_values > 0
        positive_labels = label_values[positive]
        positive_counts = object_counts[positive]
        compact_inside_counts = np.bincount(
            compact_labels.ravel()[domain.ravel()], minlength=label_values.size
        )
        inside_counts = compact_inside_counts[positive]
        partial = (inside_counts > 0) & (inside_counts < positive_counts)
        if np.any(partial):
            cut_label = int(positive_labels[np.flatnonzero(partial)[0]])
            raise AnalysisDomainError(
                f"analysis_domain_mask cuts through label {cut_label} in image {image_id!r}"
            )
        included_labels = positive_labels[
            inside_counts == positive_counts
        ].astype(int).tolist()
        omitted_labels = positive_labels[inside_counts == 0].astype(int).tolist()

    analyzed_area_mm2 = analyzed_pixel_count * calibration_mm**2
    if not math.isfinite(analyzed_area_mm2) or analyzed_area_mm2 <= 0:
        raise SamplingValidationError("analyzed area must be positive and finite")

    return ImageSamplingRecord(
        sample_id=sample_id,
        image_id=image_id,
        image_index=image_index,
        magnification_group_id=magnification_group_id,
        height_px=height_px,
        width_px=width_px,
        calibration=float(calibration),
        calibration_unit=normalized_unit,
        calibration_mm_per_px=calibration_mm,
        analyzed_pixel_count=analyzed_pixel_count,
        analyzed_area_mm2=analyzed_area_mm2,
        min_detectable_diameter=(
            None if min_detectable_diameter is None else float(min_detectable_diameter)
        ),
        max_reliable_diameter=(
            None if max_reliable_diameter is None else float(max_reliable_diameter)
        ),
        min_detectable_diameter_mm=minimum_mm,
        max_reliable_diameter_mm=maximum_mm,
        source_id=source_id,
        parent_image_id=parent_image_id,
        included_labels=tuple(included_labels),
        omitted_labels=tuple(omitted_labels),
    )


@dataclass(frozen=True)
class SampledPore:
    """An existing pore measurement associated with calibrated image metadata."""

    image: ImageSamplingRecord
    measurement: PoreProps
    equivalent_diameter_mm: float

    @property
    def sample_id(self) -> str:
        return self.image.sample_id

    @property
    def image_id(self) -> str:
        return self.image.image_id

    @property
    def magnification_group_id(self) -> str:
        return self.image.magnification_group_id


@dataclass(frozen=True)
class SamplingGroupSummary:
    """Inventory statistics for one sample and magnification group."""

    sample_id: str
    magnification_group_id: str
    image_count: int
    observed_pore_count: int
    total_analyzed_area_mm2: float


@dataclass(frozen=True)
class SamplingDataset:
    """Validated images and their non-destructively adapted pore measurements."""

    images: Tuple[ImageSamplingRecord, ...]
    pores: Tuple[SampledPore, ...]
    omitted_pore_identities: Tuple[Tuple[int, int], ...] = ()

    def summarize_groups(self) -> Dict[Tuple[str, str], SamplingGroupSummary]:
        return summarize_sampling_groups(self)

    @property
    def detection_limits_are_set(self) -> bool:
        """Whether every image has the lower limit needed by later analyses."""
        return all(image.detection_window_is_set for image in self.images)

    def require_detection_limits(self) -> None:
        """Raise when any image still has an unset minimum detection diameter."""
        missing = [image.image_id for image in self.images if not image.detection_window_is_set]
        if missing:
            raise SamplingValidationError(
                "Minimum detection diameter is unset for images: " + ", ".join(missing)
            )


def _validate_measurement_scale(pore: PoreProps, image: ImageSamplingRecord) -> None:
    has_scale = pore.units_per_px is not None
    has_unit = pore.unit_name is not None and pore.unit_name != ""
    if has_scale != has_unit:
        raise MeasurementMappingError(
            f"Pore {(pore.image_index, pore.label)} has incomplete scale information"
        )
    if not has_scale:
        if pore.eq_diam_units is not None:
            raise MeasurementMappingError(
                f"Pore {(pore.image_index, pore.label)} has a calibrated diameter without scale information"
            )
        return

    pore_scale_mm = length_to_mm(
        pore.units_per_px, pore.unit_name, field_name="measurement units_per_px"
    )
    if not math.isclose(
        pore_scale_mm,
        image.calibration_mm_per_px,
        rel_tol=DIAMETER_REL_TOLERANCE,
        abs_tol=DIAMETER_ABS_TOLERANCE_MM,
    ):
        raise MeasurementMappingError(
            f"Pore {(pore.image_index, pore.label)} scale is inconsistent with its image calibration"
        )


def build_sampling_dataset(
    images: Sequence[ImageSamplingRecord], measurements: Iterable[PoreProps]
) -> SamplingDataset:
    """Join existing measurements to images through the explicit image index."""
    images_tuple = tuple(images)
    images_by_index: Dict[int, ImageSamplingRecord] = {}
    image_identities = set()
    for image in images_tuple:
        identity = (image.sample_id, image.image_id)
        if identity in image_identities:
            raise MeasurementMappingError(f"Duplicate image identity {identity!r}")
        image_identities.add(identity)
        if image.image_index in images_by_index:
            raise MeasurementMappingError(
                f"Ambiguous image_index mapping for {image.image_index}"
            )
        images_by_index[image.image_index] = image

    sampled_pores = []
    omitted = []
    pore_identities = set()
    for pore in measurements:
        identity = (pore.image_index, pore.label)
        if identity in pore_identities:
            raise MeasurementMappingError(f"Duplicate pore identity {identity!r}")
        pore_identities.add(identity)
        image = images_by_index.get(pore.image_index)
        if image is None:
            raise MeasurementMappingError(
                f"Pore {identity!r} references unknown image_index {pore.image_index}"
            )
        if pore.label in image.omitted_labels:
            omitted.append(identity)
            continue
        if pore.label not in image.included_labels:
            raise MeasurementMappingError(
                f"Pore {identity!r} does not reference a label in image {image.image_id!r}"
            )
        try:
            diameter_px = float(pore.eq_diam_px)
        except (TypeError, ValueError) as exc:
            raise MeasurementMappingError(
                f"Pore {identity!r} equivalent diameter must be positive and finite"
            ) from exc
        if not math.isfinite(diameter_px) or diameter_px <= 0:
            raise MeasurementMappingError(
                f"Pore {identity!r} equivalent diameter must be positive and finite"
            )

        _validate_measurement_scale(pore, image)
        diameter_mm = diameter_px * image.calibration_mm_per_px
        if pore.eq_diam_units is not None:
            calibrated_mm = length_to_mm(
                pore.eq_diam_units,
                pore.unit_name,
                field_name="measurement eq_diam_units",
            )
            if not math.isclose(
                calibrated_mm,
                diameter_mm,
                rel_tol=DIAMETER_REL_TOLERANCE,
                abs_tol=DIAMETER_ABS_TOLERANCE_MM,
            ):
                raise MeasurementMappingError(
                    f"Pore {identity!r} calibrated diameter is inconsistent with pixel diameter"
                )
        sampled_pores.append(
            SampledPore(
                image=image,
                measurement=pore,
                equivalent_diameter_mm=diameter_mm,
            )
        )

    return SamplingDataset(images_tuple, tuple(sampled_pores), tuple(omitted))


def summarize_sampling_groups(
    dataset: SamplingDataset,
) -> Dict[Tuple[str, str], SamplingGroupSummary]:
    """Return image, pore, and area inventory totals per sample/group."""
    area_and_images: Dict[Tuple[str, str], Tuple[int, float]] = {}
    for image in dataset.images:
        key = (image.sample_id, image.magnification_group_id)
        image_count, area = area_and_images.get(key, (0, 0.0))
        area_and_images[key] = (image_count + 1, area + image.analyzed_area_mm2)

    pore_counts: Dict[Tuple[str, str], int] = {}
    for pore in dataset.pores:
        key = (pore.sample_id, pore.magnification_group_id)
        pore_counts[key] = pore_counts.get(key, 0) + 1

    return {
        key: SamplingGroupSummary(
            sample_id=key[0],
            magnification_group_id=key[1],
            image_count=image_count,
            observed_pore_count=pore_counts.get(key, 0),
            total_analyzed_area_mm2=area,
        )
        for key, (image_count, area) in area_and_images.items()
    }
