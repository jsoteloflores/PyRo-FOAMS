"""Validation and extraction for the pinned FOAMS workflow source replay."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Tuple, Union

import numpy as np

from .workflow_source import (
    SourceFragment,
    WorkflowSourceValidationError,
    extract_source_fragment,
    source_hashes,
)

ORIGINAL_SOURCE_REPOSITORY = "jsoteloflores/FOAMS-1.0.5"
ORIGINAL_SOURCE_COMMIT = "179663203f2d0f86b2863d5ea7f8f70dadca02f8"
DEFAULT_EXTRACTION_MAP = Path("reference-tools/matlab/foams105_workflow_extraction_map.json")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class WorkflowReferenceValidationError(ValueError):
    """Raised when a workflow reference map, source checkout, or fragment is invalid."""

    def __init__(self, message: str, *, code: str, context: Optional[Mapping[str, object]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})


@dataclass(frozen=True)
class WorkflowSourceCheckout:
    root: Path
    commit: str
    raw_file_hashes: Tuple[Tuple[str, str], ...]
    lf_file_hashes: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class OmittedSourceLine:
    line_number: int
    text: str
    reason: str


@dataclass(frozen=True)
class ExecutableSourceFragment:
    fragment_id: str
    source: SourceFragment
    executable_text: str
    omitted_lines: Tuple[OmittedSourceLine, ...]


@dataclass(frozen=True)
class WorkflowCatalogImage:
    image_id: str
    source_data_slot: int
    source_image_key: str
    pore_phase_value: int
    excluded_phase_values: Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]
    excluded_min_area_mm2: Tuple[float, float, float, float]
    area_output_selected: bool
    image: np.ndarray


@dataclass(frozen=True)
class WorkflowCatalogGroup:
    group_id: str
    source_slot: int
    scale_px_per_mm: float
    images: Tuple[WorkflowCatalogImage, ...]


@dataclass(frozen=True)
class WorkflowCatalogScene:
    scene_id: str
    groups: Tuple[WorkflowCatalogGroup, ...]


@dataclass(frozen=True)
class WorkflowCatalogRun:
    case_id: str
    scene: WorkflowCatalogScene
    expected_outcome: str
    expected_stage: Optional[str]
    expected_source_stage: Optional[str]
    expected_source_error: Optional[str]
    preparation: Mapping[str, object]
    source_metadata: Mapping[str, object]
    analysis: Mapping[str, object]


@dataclass(frozen=True)
class WorkflowCaseCatalog:
    path: Path
    raw_sha256: str
    source_commit: str
    images: Mapping[str, WorkflowCatalogImage]
    scenes: Mapping[str, WorkflowCatalogScene]
    runs: Tuple[WorkflowCatalogRun, ...]


@dataclass(frozen=True)
class WorkflowReferencePreparation:
    output_directory: Path
    constructor_path: Path
    manifest_path: Path
    catalog_raw_sha256: str
    extraction_map_raw_sha256: str
    runner_raw_sha256: str
    constructor_raw_sha256: str
    run_count: int


@dataclass(frozen=True)
class WorkflowPythonRunResult:
    case_id: str
    prepared_result: object
    analysis_result: Optional[object]
    failure_stage: Optional[str]
    failure_code: Optional[str]


_SOURCE_DATA_SLOT_MAP = {
    1: (1, "mag1"),
    2: (1, "first"),
    3: (1, "second"),
    4: (1, "fifteenth"),
    5: (1, "third"),
    6: (2, "fourth"),
    7: (2, "fifth"),
    8: (2, "sixth"),
    9: (2, "seventh"),
    10: (2, "eigth"),
    11: (3, "sixteenth"),
    12: (3, "seventeenth"),
    13: (3, "ninth"),
    14: (3, "tenth"),
    15: (3, "eleventh"),
    16: (4, "twelth"),
    17: (4, "thirteenth"),
    18: (4, "fourteenth"),
    19: (4, "eighteenth"),
    20: (4, "nineteenth"),
}


def _catalog_error(message: str, code: str, **context: object) -> WorkflowReferenceValidationError:
    return WorkflowReferenceValidationError(message, code=code, context=context)


def load_workflow_case_catalog(path: Union[Path, str]) -> WorkflowCaseCatalog:
    """Load exact sparse row-run pixels and validate every recorded image hash."""

    catalog_path = Path(path)
    try:
        raw = catalog_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _catalog_error("workflow case catalog cannot be read", "invalid_case_catalog", path=str(catalog_path), cause=str(exc)) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("source_repository") != ORIGINAL_SOURCE_REPOSITORY
        or payload.get("source_commit") != ORIGINAL_SOURCE_COMMIT
        or not isinstance(payload.get("images"), list)
        or not isinstance(payload.get("scenes"), list)
        or not isinstance(payload.get("runs"), list)
    ):
        raise _catalog_error("workflow case catalog header or collections are invalid", "invalid_case_catalog")

    images = {}
    for image_spec in payload["images"]:
        if not isinstance(image_spec, dict):
            raise _catalog_error("catalog image record is not an object", "invalid_catalog_image")
        image_id = image_spec.get("image_id")
        source_data_slot = image_spec.get("source_data_slot")
        source_key = image_spec.get("source_image_key")
        pore_phase = image_spec.get("pore_phase_value")
        excluded_values = image_spec.get("excluded_phase_values")
        excluded_thresholds = image_spec.get("excluded_min_area_mm2")
        area_selected = image_spec.get("area_output_selected")
        shape = image_spec.get("shape")
        runs = image_spec.get("runs")
        if (
            not isinstance(image_id, str)
            or not image_id
            or image_id in images
            or not isinstance(source_data_slot, int)
            or isinstance(source_data_slot, bool)
            or source_data_slot not in _SOURCE_DATA_SLOT_MAP
            or _SOURCE_DATA_SLOT_MAP[source_data_slot][1] != source_key
            or not isinstance(pore_phase, int)
            or isinstance(pore_phase, bool)
            or not isinstance(excluded_values, list)
            or len(excluded_values) != 4
            or any(value is not None and (not isinstance(value, int) or isinstance(value, bool)) for value in excluded_values)
            or pore_phase in excluded_values
            or len(tuple(value for value in excluded_values if value is not None)) != len(set(value for value in excluded_values if value is not None))
            or not isinstance(excluded_thresholds, list)
            or len(excluded_thresholds) != 4
            or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value < 0 for value in excluded_thresholds)
            or not isinstance(area_selected, bool)
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in shape)
            or image_spec.get("dtype") != "uint8"
            or not isinstance(runs, list)
            or not isinstance(image_spec.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", image_spec["sha256"])
        ):
            raise _catalog_error("catalog image identity, dimensions, dtype, or checksum is invalid", "invalid_catalog_image", image_id=image_id)
        array = np.zeros(tuple(shape), dtype=np.uint8)
        occupied = np.zeros(tuple(shape), dtype=np.bool_)
        for run_index, run in enumerate(runs):
            if (
                not isinstance(run, list)
                or len(run) != 4
                or any(not isinstance(value, int) or isinstance(value, bool) for value in run)
            ):
                raise _catalog_error("sparse image run is malformed", "invalid_image_run", image_id=image_id, run_index=run_index)
            row, column_start, column_end, pixel_value = run
            if not (0 <= row < shape[0] and 0 <= column_start < column_end <= shape[1] and 0 <= pixel_value <= 255):
                raise _catalog_error("sparse image run lies outside the declared image", "invalid_image_run", image_id=image_id, run_index=run_index)
            if bool(occupied[row, column_start:column_end].any()):
                raise _catalog_error("sparse image runs overlap", "overlapping_image_runs", image_id=image_id, run_index=run_index)
            array[row, column_start:column_end] = pixel_value
            occupied[row, column_start:column_end] = True
        actual_hash = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        if actual_hash != image_spec["sha256"]:
            raise _catalog_error(
                "expanded image pixels do not match the catalog checksum",
                "image_hash_mismatch",
                image_id=image_id,
                expected=image_spec["sha256"],
                actual=actual_hash,
            )
        owned_image = np.frombuffer(array.tobytes(order="C"), dtype=np.uint8).reshape(array.shape)
        images[image_id] = WorkflowCatalogImage(
            image_id,
            source_data_slot,
            source_key,
            pore_phase,
            tuple(excluded_values),
            tuple(float(value) for value in excluded_thresholds),
            area_selected,
            owned_image,
        )

    scenes = {}
    for scene_spec in payload["scenes"]:
        if not isinstance(scene_spec, dict):
            raise _catalog_error("catalog scene record is not an object", "invalid_catalog_scene")
        scene_id = scene_spec.get("scene_id")
        group_specs = scene_spec.get("groups")
        if not isinstance(scene_id, str) or not scene_id or scene_id in scenes or not isinstance(group_specs, list) or not group_specs:
            raise _catalog_error("scene identity or groups are invalid", "invalid_catalog_scene", scene_id=scene_id)
        groups = []
        scene_image_ids = set()
        scene_data_slots = set()
        previous_scale = None
        for group_index, group_spec in enumerate(group_specs):
            if not isinstance(group_spec, dict):
                raise _catalog_error("catalog group record is not an object", "invalid_catalog_group", scene_id=scene_id)
            group_id = group_spec.get("group_id")
            source_slot = group_spec.get("source_slot")
            scale = group_spec.get("scale_px_per_mm")
            image_ids = group_spec.get("image_ids")
            if (
                not isinstance(group_id, str)
                or not group_id
                or not isinstance(source_slot, int)
                or isinstance(source_slot, bool)
                or source_slot != group_index + 1
                or not isinstance(scale, (int, float))
                or isinstance(scale, bool)
                or not np.isfinite(scale)
                or scale <= 0
                or (previous_scale is not None and scale <= previous_scale)
                or not isinstance(image_ids, list)
                or not image_ids
            ):
                raise _catalog_error("catalog groups must be contiguous, calibrated, ordered, and nonempty", "invalid_catalog_group", scene_id=scene_id, group_id=group_id)
            previous_scale = float(scale)
            group_images = []
            for image_id in image_ids:
                image = images.get(image_id)
                if image is None or image_id in scene_image_ids or image.source_data_slot in scene_data_slots:
                    raise _catalog_error("scene image reference is missing or duplicated", "invalid_scene_image_reference", scene_id=scene_id, image_id=image_id)
                if _SOURCE_DATA_SLOT_MAP[image.source_data_slot][0] != source_slot:
                    raise _catalog_error("original MATLAB data slot is assigned to the wrong source group", "source_slot_mapping_mismatch", scene_id=scene_id, image_id=image_id, source_slot=source_slot)
                scene_image_ids.add(image_id)
                scene_data_slots.add(image.source_data_slot)
                group_images.append(image)
            groups.append(WorkflowCatalogGroup(group_id, source_slot, float(scale), tuple(group_images)))
        scenes[scene_id] = WorkflowCatalogScene(scene_id, tuple(groups))

    runs = []
    run_ids = set()
    for run_spec in payload["runs"]:
        if not isinstance(run_spec, dict):
            raise _catalog_error("catalog run record is not an object", "invalid_catalog_run")
        case_id = run_spec.get("case_id")
        scene = scenes.get(run_spec.get("scene_id"))
        expected = run_spec.get("expected_outcome")
        preparation = run_spec.get("preparation")
        metadata = run_spec.get("source_metadata")
        analysis = run_spec.get("analysis")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in run_ids
            or scene is None
            or expected not in ("success", "expected_domain_failure")
            or not isinstance(preparation, dict)
            or not isinstance(metadata, dict)
            or not isinstance(analysis, dict)
        ):
            raise _catalog_error("catalog run identity, scene, outcome, or settings are invalid", "invalid_catalog_run", case_id=case_id)
        if analysis.get("selection_mode") not in ("manual", "autosmart") or analysis.get("volume_normalization_mode") not in ("unnormalized", "integrated_2d", "known_vesicularity"):
            raise _catalog_error("catalog run uses unsupported analysis modes", "invalid_catalog_settings", case_id=case_id)
        shape_mode = analysis.get("shape_statistics_mode")
        preparation_minimum = preparation.get("minimum_diameter_px")
        shape_minimum = analysis.get("shape_minimum_diameter_px")
        if shape_mode not in ("initial_analysis", "updated_minimum"):
            raise _catalog_error("catalog run uses an unsupported shape-statistics mode", "invalid_catalog_settings", case_id=case_id)
        if (
            not isinstance(preparation_minimum, (int, float))
            or isinstance(preparation_minimum, bool)
            or not np.isfinite(preparation_minimum)
            or preparation_minimum <= 0
            or not isinstance(shape_minimum, (int, float))
            or isinstance(shape_minimum, bool)
            or not np.isfinite(shape_minimum)
            or shape_minimum < 0
        ):
            raise _catalog_error("catalog diameter minima must be finite and valid", "invalid_catalog_settings", case_id=case_id)
        if shape_mode == "initial_analysis" and shape_minimum != preparation_minimum:
            raise _catalog_error(
                "original initial-analysis replay requires one shared project/shape minimum",
                "inconsistent_initial_shape_threshold",
                case_id=case_id,
                preparation_minimum_diameter_px=preparation_minimum,
                shape_minimum_diameter_px=shape_minimum,
            )
        ranges = analysis.get("manual_ranges")
        if not isinstance(ranges, list):
            raise _catalog_error("manual_ranges must be an immutable list", "invalid_catalog_settings", case_id=case_id)
        if analysis["selection_mode"] == "manual":
            if len(ranges) != len(scene.groups):
                raise _catalog_error("manual run must supply one range per ordered group", "invalid_catalog_settings", case_id=case_id)
            if tuple(group.group_id for group, item in zip(scene.groups, ranges) if item is not None) != tuple(item.get("group_id") for item in ranges if isinstance(item, dict)):
                raise _catalog_error("manual ranges must follow source group order", "invalid_catalog_settings", case_id=case_id)
            for group, item in zip(scene.groups, ranges):
                if item is None:
                    continue
                if (
                    not isinstance(item, dict)
                    or item.get("group_id") != group.group_id
                    or not isinstance(item.get("lower_bin_index"), int)
                    or isinstance(item.get("lower_bin_index"), bool)
                    or not isinstance(item.get("upper_bin_index"), int)
                    or isinstance(item.get("upper_bin_index"), bool)
                    or not 0 <= item["lower_bin_index"] <= item["upper_bin_index"] < 45
                ):
                    raise _catalog_error("manual range bin indices are invalid", "invalid_catalog_settings", case_id=case_id)
        elif ranges:
            raise _catalog_error("AutoSmart run must not carry manual ranges", "invalid_catalog_settings", case_id=case_id)
        if expected == "expected_domain_failure" and (
            not isinstance(run_spec.get("expected_stage"), str)
            or not isinstance(run_spec.get("expected_source_stage"), str)
            or not isinstance(run_spec.get("expected_source_error"), str)
        ):
            raise _catalog_error("expected failure must declare Python/source stages and MATLAB exception", "invalid_catalog_run", case_id=case_id)
        run_ids.add(case_id)
        runs.append(
            WorkflowCatalogRun(
                case_id,
                scene,
                expected,
                run_spec.get("expected_stage"),
                run_spec.get("expected_source_stage"),
                run_spec.get("expected_source_error"),
                MappingProxyType(dict(preparation)),
                MappingProxyType(dict(metadata)),
                MappingProxyType(dict(analysis)),
            )
        )
    return WorkflowCaseCatalog(
        catalog_path,
        hashlib.sha256(raw).hexdigest(),
        ORIGINAL_SOURCE_COMMIT,
        MappingProxyType(images),
        MappingProxyType(scenes),
        tuple(runs),
    )


def _matlab_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _matlab_number(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "Inf"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return "NaN" if math.isnan(number) else "Inf" if math.isinf(number) and number > 0 else "-Inf" if math.isinf(number) else format(number, ".17g")
    raise WorkflowReferenceValidationError(
        "catalog value cannot be emitted as a MATLAB scalar",
        code="unsupported_constructor_value",
        context={"type": type(value).__name__},
    )


def _matlab_matrix(image: np.ndarray) -> str:
    rows = [" ".join(str(int(value)) for value in row) for row in image]
    return "uint8([" + "; ".join(rows) + "])"


def _render_workflow_constructor(catalog: WorkflowCaseCatalog) -> str:
    lines = [
        "function bundle = foams105_workflow_cases_generated()",
        "% Generated from the checksum-pinned Brief 19 JSON case catalog.",
        "bundle.schema_version = 1;",
        f"bundle.source_repository = {_matlab_string(ORIGINAL_SOURCE_REPOSITORY)};",
        f"bundle.source_commit = {_matlab_string(ORIGINAL_SOURCE_COMMIT)};",
        f"bundle.catalog_raw_sha256 = {_matlab_string(catalog.raw_sha256)};",
        f"bundle.runs = repmat(struct('case_id','','expected_outcome','','expected_stage','','expected_source_stage','','groups',[],'preparation',[],'source_metadata',[],'analysis',[]), {len(catalog.runs)}, 1);",
    ]
    for run_index, run in enumerate(catalog.runs, start=1):
        prefix = f"bundle.runs({run_index})"
        lines.extend(
            (
                f"{prefix}.case_id = {_matlab_string(run.case_id)};",
                f"{prefix}.expected_outcome = {_matlab_string(run.expected_outcome)};",
                f"{prefix}.expected_stage = {_matlab_string(run.expected_stage or '')};",
                f"{prefix}.expected_source_stage = {_matlab_string(run.expected_source_stage or '')};",
                f"{prefix}.groups = repmat(struct('group_id','','source_slot',0,'scale_px_per_mm',0,'images',[]), {len(run.scene.groups)}, 1);",
            )
        )
        for group_index, group in enumerate(run.scene.groups, start=1):
            group_prefix = f"{prefix}.groups({group_index})"
            lines.extend(
                (
                    f"{group_prefix}.group_id = {_matlab_string(group.group_id)};",
                    f"{group_prefix}.source_slot = {group.source_slot};",
                    f"{group_prefix}.scale_px_per_mm = {_matlab_number(group.scale_px_per_mm)};",
                    f"{group_prefix}.images = repmat(struct('image_id','','source_data_slot',0,'source_image_key','','segmented_image',[],'pore_phase_value',0,'excluded_phase_values',[],'excluded_min_area_mm2',[],'area_output_selected',true), {len(group.images)}, 1);",
                )
            )
            for image_index, image in enumerate(group.images, start=1):
                image_prefix = f"{group_prefix}.images({image_index})"
                excluded_values = ";".join(_matlab_number(value) for value in image.excluded_phase_values)
                thresholds = ";".join(_matlab_number(value) for value in image.excluded_min_area_mm2)
                lines.extend(
                    (
                        f"{image_prefix}.image_id = {_matlab_string(image.image_id)};",
                        f"{image_prefix}.source_data_slot = {image.source_data_slot};",
                        f"{image_prefix}.source_image_key = {_matlab_string(image.source_image_key)};",
                        f"{image_prefix}.segmented_image = {_matlab_matrix(image.image)};",
                        f"{image_prefix}.pore_phase_value = {image.pore_phase_value};",
                        f"{image_prefix}.excluded_phase_values = [{excluded_values}];",
                        f"{image_prefix}.excluded_min_area_mm2 = [{thresholds}];",
                        f"{image_prefix}.area_output_selected = {_matlab_number(image.area_output_selected)};",
                    )
                )
        lines.extend(
            (
                f"{prefix}.preparation.minimum_diameter_px = {_matlab_number(run.preparation['minimum_diameter_px'])};",
                f"{prefix}.preparation.perimeter_profile = {_matlab_string(str(run.preparation['perimeter_profile']))};",
                f"{prefix}.source_metadata.density_derived_vesicularity_percent = {_matlab_number(run.source_metadata.get('density_derived_vesicularity_percent'))};",
                f"{prefix}.source_metadata.pyro_revision = {_matlab_string(str(run.source_metadata.get('pyro_revision', 'unknown')))};",
                f"{prefix}.analysis.selection_mode = {_matlab_string(str(run.analysis['selection_mode']))};",
                f"{prefix}.analysis.manual_range_lower_indices = {_matlab_matrix_vector([item['lower_bin_index'] if item is not None else -1 for item in run.analysis['manual_ranges']])};",
                f"{prefix}.analysis.manual_range_upper_indices = {_matlab_matrix_vector([item['upper_bin_index'] if item is not None else -1 for item in run.analysis['manual_ranges']])};",
                f"{prefix}.analysis.volume_normalization_mode = {_matlab_string(str(run.analysis['volume_normalization_mode']))};",
                f"{prefix}.analysis.known_vesicularity_percent = {_matlab_number(run.analysis.get('known_vesicularity_percent'))};",
                f"{prefix}.analysis.size_lower_cutoff_mm = {_matlab_number(run.analysis['size_lower_cutoff_mm'])};",
                f"{prefix}.analysis.size_upper_cutoff_mm = {_matlab_number(run.analysis['size_upper_cutoff_mm'])};",
                f"{prefix}.analysis.shape_minimum_diameter_px = {_matlab_number(run.analysis['shape_minimum_diameter_px'])};",
                f"{prefix}.analysis.shape_statistics_mode = {_matlab_string(str(run.analysis['shape_statistics_mode']))};",
            )
        )
    lines.append("end")
    return "\n".join(lines) + "\n"


def _matlab_matrix_vector(values) -> str:
    return "[" + " ".join(_matlab_number(value) for value in values) + "]"


def prepare_workflow_reference(
    case_catalog_path: Union[Path, str],
    output_directory: Union[Path, str],
    *,
    extraction_map_path: Union[Path, str] = DEFAULT_EXTRACTION_MAP,
    runner_path: Union[Path, str] = "reference-tools/matlab/foams105_workflow_capture.m",
) -> WorkflowReferencePreparation:
    """Generate a deterministic MATLAB input constructor and pinned manifest."""

    output = Path(output_directory)
    if output.exists():
        raise WorkflowReferenceValidationError(
            "reference preparation output directory already exists",
            code="output_directory_exists",
            context={"output_directory": str(output)},
        )
    catalog = load_workflow_case_catalog(case_catalog_path)
    extraction_map = load_workflow_extraction_map(extraction_map_path)
    map_raw_hash, map_lf_hash = source_hashes(extraction_map_path)
    runner_file = Path(runner_path)
    runner_raw_hash, runner_lf_hash = source_hashes(runner_file)
    constructor = _render_workflow_constructor(catalog)
    constructor_path = output / "foams105_workflow_cases_generated.m"
    manifest_path = output / "preparation_manifest.json"
    output.mkdir(parents=True, exist_ok=False)
    constructor_path.write_text(constructor, encoding="utf-8", newline="\n")
    constructor_raw_hash, constructor_lf_hash = source_hashes(constructor_path)
    manifest = {
        "schema_version": 1,
        "source_repository": ORIGINAL_SOURCE_REPOSITORY,
        "source_commit": ORIGINAL_SOURCE_COMMIT,
        "workflow_classification": extraction_map["workflow_classification"],
        "catalog": {"path": str(Path(case_catalog_path)), "raw_sha256": catalog.raw_sha256},
        "extraction_map": {"path": str(Path(extraction_map_path)), "raw_sha256": map_raw_hash, "lf_sha256": map_lf_hash},
        "runner": {"path": str(runner_file), "raw_sha256": runner_raw_hash, "lf_sha256": runner_lf_hash},
        "constructor": {"path": constructor_path.name, "raw_sha256": constructor_raw_hash, "lf_sha256": constructor_lf_hash},
        "source_files": extraction_map["source_files"],
        "fragment_ids": tuple(fragment["id"] for fragment in extraction_map["fragments"]),
        "histogram_ids": tuple(call["id"] for call in extraction_map["histogram_calls"]),
        "case_ids": tuple(run.case_id for run in catalog.runs),
        "expected_success_count": sum(run.expected_outcome == "success" for run in catalog.runs),
        "expected_failure_count": sum(run.expected_outcome == "expected_domain_failure" for run in catalog.runs),
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return WorkflowReferencePreparation(
        output,
        constructor_path,
        manifest_path,
        catalog.raw_sha256,
        map_raw_hash,
        runner_raw_hash,
        constructor_raw_hash,
        len(catalog.runs),
    )
def run_python_workflow_case(run: WorkflowCatalogRun) -> WorkflowPythonRunResult:
    """Build the public headless Python result for one fully specified catalog run."""

    from main.core.legacy_perimeter import FOAMS105_PERIMETER_SUPPORTED_PROFILES
    from main.core.legacy_pipeline import (
        Foams105AnalysisSettings,
        Foams105PipelineGroupInput,
        Foams105PipelineStageError,
        Foams105PreparationSettings,
        Foams105SegmentedImageInput,
        Foams105SourceMetadata,
        analyze_foams105_prepared,
        prepare_foams105_analysis,
    )
    from main.core.legacy_selection import Foams105BinnedRange

    profile = run.preparation.get("perimeter_profile")
    if profile not in FOAMS105_PERIMETER_SUPPORTED_PROFILES:
        raise _catalog_error("catalog perimeter profile is unsupported", "invalid_catalog_settings", case_id=run.case_id)
    prepared_settings = Foams105PreparationSettings(
        run.preparation.get("minimum_diameter_px"),
        str(profile),
    )
    metadata = Foams105SourceMetadata(
        run.source_metadata.get("density_derived_vesicularity_percent"),
        str(run.source_metadata.get("pyro_revision", "unknown")),
    )
    group_inputs = tuple(
        Foams105PipelineGroupInput(
            group.group_id,
            group.source_slot,
            tuple(
                Foams105SegmentedImageInput(
                    image.image_id,
                    image.source_image_key,
                    image.image,
                    group.scale_px_per_mm,
                    image.pore_phase_value,
                    image.excluded_phase_values,
                    image.excluded_min_area_mm2,
                    image.area_output_selected,
                )
                for image in group.images
            ),
        )
        for group in run.scene.groups
    )
    try:
        prepared = prepare_foams105_analysis(
            group_inputs,
            prepared_settings,
            source_metadata=metadata,
        )
        analysis_spec = run.analysis
        requested_ranges = analysis_spec["manual_ranges"]
        if analysis_spec["selection_mode"] == "manual":
            labels = prepared.label_preparation_result.bin_labels_mm
            manual_ranges = tuple(
                None if range_spec is None else Foams105BinnedRange(
                    group.group_id,
                    labels[range_spec["lower_bin_index"]],
                    labels[range_spec["upper_bin_index"]],
                )
                for group, range_spec in zip(run.scene.groups, requested_ranges)
            )
        else:
            manual_ranges = ()
        settings = Foams105AnalysisSettings(
            selection_mode=analysis_spec["selection_mode"],
            manual_ranges=manual_ranges,
            volume_normalization_mode=analysis_spec["volume_normalization_mode"],
            known_vesicularity_percent=analysis_spec.get("known_vesicularity_percent"),
            size_lower_cutoff_mm=analysis_spec["size_lower_cutoff_mm"],
            size_upper_cutoff_mm=analysis_spec["size_upper_cutoff_mm"],
            shape_minimum_diameter_px=analysis_spec["shape_minimum_diameter_px"],
            shape_statistics_mode=analysis_spec["shape_statistics_mode"],
        )
        analysis = analyze_foams105_prepared(prepared, settings)
    except Foams105PipelineStageError as exc:
        if run.expected_outcome != "expected_domain_failure":
            raise WorkflowReferenceValidationError(
                "Python public workflow failed for a catalog success case",
                code="unexpected_python_stage_failure",
                context={
                    "case_id": run.case_id,
                    "stage": exc.stage,
                    "cause_code": exc.cause_code,
                    "cause_context": dict(exc.context),
                },
            ) from exc
        if exc.stage != run.expected_stage:
            raise WorkflowReferenceValidationError(
                "Python domain failure occurred at an unexpected stage",
                code="wrong_python_failure_stage",
                context={
                    "case_id": run.case_id,
                    "expected_stage": run.expected_stage,
                    "actual_stage": exc.stage,
                    "cause_code": exc.cause_code,
                    "cause_context": dict(exc.context),
                },
            ) from exc
        return WorkflowPythonRunResult(run.case_id, prepared if "prepared" in locals() else None, None, exc.stage, exc.cause_code)
    if run.expected_outcome == "expected_domain_failure":
        raise WorkflowReferenceValidationError(
            "Python public workflow unexpectedly succeeded for an expected failure case",
            code="expected_failure_succeeded",
            context={"case_id": run.case_id, "expected_stage": run.expected_stage},
        )
    return WorkflowPythonRunResult(run.case_id, prepared, analysis, None, None)
def load_workflow_extraction_map(path: Union[Path, str] = DEFAULT_EXTRACTION_MAP) -> Mapping[str, object]:
    map_path = Path(path)
    try:
        payload = json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkflowReferenceValidationError(
            "workflow extraction map cannot be read as UTF-8 JSON",
            code="invalid_extraction_map",
            context={"path": str(map_path), "cause": str(exc)},
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise WorkflowReferenceValidationError(
            "workflow extraction map schema is unsupported",
            code="invalid_extraction_map",
            context={"path": str(map_path)},
        )
    if (
        payload.get("source_repository") != ORIGINAL_SOURCE_REPOSITORY
        or payload.get("source_commit") != ORIGINAL_SOURCE_COMMIT
    ):
        raise WorkflowReferenceValidationError(
            "workflow extraction map does not identify the pinned original source",
            code="wrong_source_identity",
            context={"path": str(map_path)},
        )
    source_files = payload.get("source_files")
    fragments = payload.get("fragments")
    histogram_calls = payload.get("histogram_calls")
    if not isinstance(source_files, dict) or not isinstance(fragments, list) or not isinstance(histogram_calls, list):
        raise WorkflowReferenceValidationError(
            "workflow extraction map is missing source files, fragments, or histogram calls",
            code="invalid_extraction_map",
            context={"path": str(map_path)},
        )
    for relative_name, pin in source_files.items():
        if (
            not isinstance(relative_name, str)
            or not isinstance(pin, dict)
            or not all(_SHA256_PATTERN.fullmatch(pin.get(key, "")) for key in ("raw_sha256", "lf_sha256"))
        ):
            raise WorkflowReferenceValidationError(
                "source file hash pin is malformed",
                code="invalid_extraction_map",
                context={"file": relative_name},
            )
    fragment_ids = set()
    for fragment in fragments:
        if not isinstance(fragment, dict):
            raise WorkflowReferenceValidationError("extraction fragment entry is invalid", code="invalid_extraction_map")
        fragment_id = fragment.get("id")
        if (
            not isinstance(fragment_id, str)
            or not fragment_id
            or fragment_id in fragment_ids
            or fragment.get("file") not in source_files
            or not isinstance(fragment.get("start_line"), int)
            or isinstance(fragment.get("start_line"), bool)
            or not isinstance(fragment.get("end_line"), int)
            or isinstance(fragment.get("end_line"), bool)
            or fragment.get("start_line", 0) < 1
            or fragment.get("end_line", 0) < fragment.get("start_line", 1)
            or not _SHA256_PATTERN.fullmatch(fragment.get("lf_sha256", ""))
            or not isinstance(fragment.get("required_inputs"), list)
            or not isinstance(fragment.get("produced_variables"), list)
            or not isinstance(fragment.get("omitted_calls"), list)
            or not isinstance(fragment.get("required_assignments", []), list)
            or not all(isinstance(name, str) and name for name in fragment.get("required_assignments", []))
            or not set(fragment.get("required_assignments", [])).issubset(set(fragment.get("produced_variables", [])))
        ):
            raise WorkflowReferenceValidationError(
                "extraction fragment identity or contract is invalid",
                code="invalid_extraction_map",
                context={"fragment_id": fragment_id},
            )
        fragment_ids.add(fragment_id)
    histogram_ids = set()
    for call in histogram_calls:
        if (
            not isinstance(call, dict)
            or not isinstance(call.get("id"), str)
            or not call.get("id")
            or call.get("id") in histogram_ids
            or call.get("file") not in source_files
            or not isinstance(call.get("line"), int)
            or isinstance(call.get("line"), bool)
            or call.get("line", 0) < 1
            or not _SHA256_PATTERN.fullmatch(call.get("lf_sha256", ""))
            or not isinstance(call.get("expression"), str)
            or not call.get("expression")
            or call.get("function") not in ("hist", "histc")
            or not isinstance(call.get("grid_expression"), str)
            or not call.get("grid_expression")
            or call.get("grid_kind") not in ("centers", "edges")
        ):
            raise WorkflowReferenceValidationError(
                "histogram call entry is malformed",
                code="invalid_extraction_map",
                context={"histogram_id": call.get("id") if isinstance(call, dict) else None},
            )
        histogram_ids.add(call["id"])
    return payload


def verify_workflow_source_checkout(
    source_root: Union[Path, str],
    extraction_map: Mapping[str, object],
    *,
    repository_root: Optional[Union[Path, str]] = None,
) -> WorkflowSourceCheckout:
    root = Path(source_root).resolve()
    if repository_root is not None:
        repo = Path(repository_root).resolve()
        if root == repo or repo in root.parents:
            raise WorkflowReferenceValidationError(
                "original source checkout must remain outside the PyRo repository",
                code="source_checkout_inside_repository",
                context={"source_root": str(root), "repository_root": str(repo)},
            )
    try:
        process = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise WorkflowReferenceValidationError(
            "git is unavailable for source commit verification",
            code="source_git_unavailable",
            context={"source_root": str(root), "cause": str(exc)},
        ) from exc
    if process.returncode != 0:
        raise WorkflowReferenceValidationError(
            "source_root is not a readable Git checkout",
            code="invalid_source_checkout",
            context={"source_root": str(root), "stderr": process.stderr.strip()},
        )
    commit = process.stdout.strip().lower()
    expected_commit = extraction_map.get("source_commit")
    if commit != expected_commit:
        raise WorkflowReferenceValidationError(
            "source checkout commit differs from the pinned extraction map",
            code="wrong_source_commit",
            context={"expected": expected_commit, "actual": commit},
        )

    pins = extraction_map.get("source_files")
    if not isinstance(pins, dict):
        raise WorkflowReferenceValidationError("source file pins are missing", code="invalid_extraction_map")
    raw_hashes = []
    lf_hashes = []
    for relative_name, pin in pins.items():
        if not isinstance(relative_name, str) or not isinstance(pin, dict):
            raise WorkflowReferenceValidationError("source file pin is malformed", code="invalid_extraction_map")
        source_path = root / relative_name
        try:
            raw_hash, lf_hash = source_hashes(source_path)
        except OSError as exc:
            raise WorkflowReferenceValidationError(
                "pinned original source file is unavailable",
                code="missing_source_file",
                context={"file": relative_name, "cause": str(exc)},
            ) from exc
        if lf_hash != pin.get("lf_sha256"):
            raise WorkflowReferenceValidationError(
                "normalized source file hash differs from the extraction map",
                code="source_hash_mismatch",
                context={
                    "file": relative_name,
                    "expected_lf_sha256": pin.get("lf_sha256"),
                    "actual_lf_sha256": lf_hash,
                    "actual_raw_sha256": raw_hash,
                },
            )
        raw_hashes.append((relative_name, raw_hash))
        lf_hashes.append((relative_name, lf_hash))
    return WorkflowSourceCheckout(root, commit, tuple(raw_hashes), tuple(lf_hashes))


def _omission_reason(line: str, omitted_calls: list) -> Optional[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("%"):
        return None
    for item in omitted_calls:
        if not isinstance(item, dict):
            continue
        call = item.get("call")
        matched = False
        if call == "waitbar":
            matched = re.search(r"^(?:\w+\s*=\s*)?waitbar\s*\(", stripped) is not None
        elif call == "findobj/get":
            matched = "findobj(" in stripped or re.search(r"\bget\s*\(", stripped) is not None
        elif call == "set/findobj":
            matched = "set(" in stripped and "findobj(" in stripped
        elif call == "helpdlg":
            matched = "helpdlg(" in stripped
        elif isinstance(call, str) and call:
            matched = re.search(rf"\b{re.escape(call)}\s*\(", stripped) is not None
        elif isinstance(call, str) and call:
            matched = re.search(rf"\b{re.escape(call)}\s*\(", stripped) is not None
        if matched:
            return str(item.get("reason", "mapped nonnumerical call"))
    return None


def extract_mapped_source_fragment(
    source_root: Union[Path, str],
    extraction_map: Mapping[str, object],
    fragment_id: str,
) -> ExecutableSourceFragment:
    fragments = extraction_map.get("fragments")
    if not isinstance(fragments, list):
        raise WorkflowReferenceValidationError("extraction fragments are missing", code="invalid_extraction_map")
    entry = next((item for item in fragments if isinstance(item, dict) and item.get("id") == fragment_id), None)
    if entry is None:
        raise WorkflowReferenceValidationError(
            "requested source fragment is not in the reviewed extraction map",
            code="unknown_source_fragment",
            context={"fragment_id": fragment_id},
        )
    source_path = Path(source_root) / str(entry["file"])
    try:
        source_fragment = extract_source_fragment(
            source_path,
            entry["start_line"],
            entry["end_line"],
            expected_lf_sha256=entry["lf_sha256"],
        )
    except WorkflowSourceValidationError as exc:
        raise WorkflowReferenceValidationError(
            "source fragment failed extraction-map validation",
            code="invalid_source_fragment",
            context={"fragment_id": fragment_id, "cause_code": exc.code, **exc.context},
        ) from exc

    omissions = entry.get("omitted_calls")
    if not isinstance(omissions, list):
        raise WorkflowReferenceValidationError(
            "fragment omission map is invalid",
            code="invalid_extraction_map",
            context={"fragment_id": fragment_id},
        )
    executable_lines = []
    omitted_lines = []
    for offset, line in enumerate(source_fragment.text.splitlines(keepends=True)):
        reason = _omission_reason(line, omissions)
        if reason is None:
            executable_lines.append(line)
        else:
            omitted_lines.append(
                OmittedSourceLine(entry["start_line"] + offset, line.rstrip("\r\n"), reason)
            )
    executable_text = "".join(executable_lines)
    for variable in entry.get("required_assignments", ()):
        if re.search(rf"(?m)^\s*{re.escape(variable)}\s*=", executable_text) is None:
            raise WorkflowReferenceValidationError(
                "mapped source fragment does not assign a required output",
                code="missing_source_assignment",
                context={"fragment_id": fragment_id, "variable": variable},
            )
    forbidden = re.compile(r"\b(?:waitbar|findobj|helpdlg|uiwait|axes|figure|plot|loglog|semilogy|bar|tblwrite|xlswrite)\s*\(")
    remaining = [
        (entry["start_line"] + offset, line.rstrip("\r\n"))
        for offset, line in enumerate(executable_text.splitlines())
        if not line.lstrip().startswith("%") and forbidden.search(line)
    ]
    if remaining:
        raise WorkflowReferenceValidationError(
            "unmapped GUI, rendering, or file-writing call remains in executable source fragment",
            code="unmapped_nonnumerical_call",
            context={"fragment_id": fragment_id, "statements": tuple(remaining)},
        )
    return ExecutableSourceFragment(fragment_id, source_fragment, executable_text, tuple(omitted_lines))


def extract_mapped_histogram_expression(
    source_root: Union[Path, str],
    extraction_map: Mapping[str, object],
    histogram_id: str,
) -> str:
    calls = extraction_map.get("histogram_calls")
    if not isinstance(calls, list):
        raise WorkflowReferenceValidationError("histogram call map is missing", code="invalid_extraction_map")
    entry = next((item for item in calls if isinstance(item, dict) and item.get("id") == histogram_id), None)
    if entry is None:
        raise WorkflowReferenceValidationError(
            "requested histogram call is not mapped",
            code="unknown_histogram_call",
            context={"histogram_id": histogram_id},
        )
    source_fragment = extract_source_fragment(
        Path(source_root) / str(entry["file"]),
        entry["line"],
        entry["line"],
        expected_lf_sha256=entry["lf_sha256"],
    )
    expression = entry.get("expression")
    if (
        not isinstance(expression, str)
        or expression not in source_fragment.text
        or entry.get("function") not in ("hist", "histc")
        or not isinstance(entry.get("grid_expression"), str)
        or entry.get("grid_kind") not in ("centers", "edges")
    ):
        raise WorkflowReferenceValidationError(
            "mapped histogram expression is absent from its pinned source line",
            code="histogram_expression_mismatch",
            context={"histogram_id": histogram_id, "line": entry["line"]},
        )
    return expression
