"""FOAMS perimeter reference preparation, import, and comparison tooling.

This module is intentionally outside production measurement APIs. It validates
and prepares deterministic reference inputs, imports external MATLAB captures,
and reports numerical comparisons without claiming historical compatibility.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from main.core.legacy_measurements import measure_foams105_phase

ORIGINAL_SOURCE_COMMIT = "179663203f2d0f86b2863d5ea7f8f70dadca02f8"
SCHEMA_VERSION = 1
ABS_TOL = 1e-12
REL_TOL = 1e-12


class PerimeterReferenceValidationError(ValueError):
    """Raised when perimeter reference artifacts violate the tooling contract."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_reference",
        context: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})


@dataclass(frozen=True)
class ComponentTopology:
    case_id: str
    native_component_id: int
    canonical_component_id: int
    first_row: int
    first_column: int
    area_px: int


@dataclass(frozen=True)
class PreparedCase:
    case_id: str
    stage: str
    purpose: str
    shape: Tuple[int, int]
    scale_px_per_mm: float
    labels: np.ndarray
    foreground_pixel_count: int
    component_count: int
    topology: Tuple[ComponentTopology, ...]


@dataclass(frozen=True)
class PreparedCatalog:
    catalog_path: Path
    catalog_hash: str
    cases: Tuple[PreparedCase, ...]


@dataclass(frozen=True)
class PreparedConstructorResult:
    constructor_path: Path
    manifest_path: Path
    catalog_hash: str
    preparation_hash: str
    runner_hash: str
    constructor_hash: str
    case_count: int


@dataclass(frozen=True)
class CapturedComponent:
    case_id: str
    native_component_id: int
    canonical_component_id: int
    first_row: int
    first_column: int
    area_px: int
    perimeter_px: Optional[float]
    perimeter_status: str
    perimeter_old_px: Optional[float]
    perimeter_old_status: str
    property_error: str


@dataclass(frozen=True)
class ImportedReferenceRun:
    run_dir: Path
    metadata: Mapping[str, str]
    cases: Tuple[PreparedCase, ...]
    components: Tuple[CapturedComponent, ...]


@dataclass(frozen=True)
class ComparisonRow:
    case_id: str
    native_component_id: int
    reference_field: str
    candidate_field: str
    reference_value: float
    candidate_value: float
    absolute_difference: float
    relative_difference: Optional[float]
    within_tolerance: bool


@dataclass(frozen=True)
class ComparisonReport:
    schema_valid: bool
    topology_valid: bool
    numerical_valid: bool
    historical_compatible: bool
    compared_rows: int
    mismatch_rows: int
    rows: Tuple[ComparisonRow, ...]
    notes: Tuple[str, ...]


def _hash_file_lf(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _require_str(value: object, field: str, *, case_id: Optional[str] = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PerimeterReferenceValidationError(
            f"{field} must be a nonempty string",
            code="invalid_catalog",
            context={"case_id": case_id, "field": field},
        )
    return value


def _require_int(value: object, field: str, *, case_id: Optional[str] = None, minimum: int = 0) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise PerimeterReferenceValidationError(
            f"{field} must be an integer",
            code="invalid_catalog",
            context={"case_id": case_id, "field": field},
        )
    normalized = int(value)
    if normalized < minimum:
        raise PerimeterReferenceValidationError(
            f"{field} must be >= {minimum}",
            code="invalid_catalog",
            context={"case_id": case_id, "field": field, "value": normalized},
        )
    return normalized


def _require_positive_float(value: object, field: str, *, case_id: Optional[str] = None) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)):
        raise PerimeterReferenceValidationError(
            f"{field} must be a positive finite scalar",
            code="invalid_catalog",
            context={"case_id": case_id, "field": field},
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise PerimeterReferenceValidationError(
            f"{field} must be positive and finite",
            code="invalid_catalog",
            context={"case_id": case_id, "field": field, "value": normalized},
        )
    return normalized


def _literal_matlab_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _matrix_to_matlab_literal(labels: np.ndarray) -> str:
    if labels.size == 0:
        return "int32([])"
    rows = []
    for row in labels:
        rows.append(" ".join(str(int(item)) for item in row))
    return "int32([" + "; ".join(rows) + "])"


def _topology_from_labels(case_id: str, labels: np.ndarray) -> Tuple[ComponentTopology, ...]:
    if labels.ndim != 2:
        raise PerimeterReferenceValidationError(
            "labels must be two-dimensional",
            code="invalid_labels",
            context={"case_id": case_id},
        )
    if labels.dtype.kind not in "iu":
        raise PerimeterReferenceValidationError(
            "labels must use integer dtype",
            code="invalid_labels",
            context={"case_id": case_id},
        )
    labels_int = np.asarray(labels, dtype=np.int64)
    if labels_int.min(initial=0) < 0:
        raise PerimeterReferenceValidationError(
            "labels must be nonnegative",
            code="invalid_labels",
            context={"case_id": case_id},
        )

    native_ids = sorted(int(item) for item in np.unique(labels_int) if int(item) > 0)
    if not native_ids:
        return ()

    height = int(labels_int.shape[0])
    minima: Dict[int, int] = {}
    areas: Dict[int, int] = {}
    for native_id in native_ids:
        rows, cols = np.nonzero(labels_int == native_id)
        if rows.size == 0:
            raise PerimeterReferenceValidationError(
                "native component ID missing from labels",
                code="invalid_labels",
                context={"case_id": case_id, "native_component_id": native_id},
            )
        flat = rows.astype(np.int64) + cols.astype(np.int64) * height
        minima[native_id] = int(np.min(flat))
        areas[native_id] = int(rows.size)

    ordered_native = sorted(native_ids, key=lambda native_id: minima[native_id])
    canonical_lookup = {native_id: index + 1 for index, native_id in enumerate(ordered_native)}
    output: List[ComponentTopology] = []
    for native_id in native_ids:
        first_flat = minima[native_id]
        output.append(
            ComponentTopology(
                case_id=case_id,
                native_component_id=native_id,
                canonical_component_id=canonical_lookup[native_id],
                first_row=int(first_flat % height),
                first_column=int(first_flat // height),
                area_px=areas[native_id],
            )
        )

    canonical_ids = [item.canonical_component_id for item in sorted(output, key=lambda item: item.canonical_component_id)]
    if canonical_ids != list(range(1, len(canonical_ids) + 1)):
        raise PerimeterReferenceValidationError(
            "canonical component IDs must be contiguous",
            code="invalid_labels",
            context={"case_id": case_id},
        )
    return tuple(output)


def _labels_from_case(case: Mapping[str, object]) -> np.ndarray:
    case_id = _require_str(case.get("id"), "case.id")
    stage = _require_str(case.get("stage"), "case.stage", case_id=case_id)
    shape_raw = case.get("shape")
    if not isinstance(shape_raw, list) or len(shape_raw) != 2:
        raise PerimeterReferenceValidationError(
            "case.shape must contain [height, width]",
            code="invalid_catalog",
            context={"case_id": case_id, "field": "shape"},
        )
    height = _require_int(shape_raw[0], "shape[0]", case_id=case_id, minimum=1)
    width = _require_int(shape_raw[1], "shape[1]", case_id=case_id, minimum=1)

    labels = np.zeros((height, width), dtype=np.int32)
    if stage == "raw_phase_mask":
        phase_value = _require_int(case.get("phase_value", 1), "phase_value", case_id=case_id)
        pixels = case.get("foreground_pixels")
        if not isinstance(pixels, list):
            raise PerimeterReferenceValidationError(
                "raw_phase_mask requires foreground_pixels",
                code="invalid_catalog",
                context={"case_id": case_id},
            )
        image = np.zeros((height, width), dtype=np.uint8)
        for pixel in pixels:
            if not isinstance(pixel, list) or len(pixel) != 2:
                raise PerimeterReferenceValidationError(
                    "foreground_pixels must contain [row, column] entries",
                    code="invalid_catalog",
                    context={"case_id": case_id},
                )
            row = _require_int(pixel[0], "foreground_pixels.row", case_id=case_id, minimum=0)
            col = _require_int(pixel[1], "foreground_pixels.column", case_id=case_id, minimum=0)
            if row >= height or col >= width:
                raise PerimeterReferenceValidationError(
                    "foreground pixel out of bounds",
                    code="invalid_catalog",
                    context={"case_id": case_id, "row": row, "column": col},
                )
            image[row, col] = phase_value
        measured = measure_foams105_phase(image, phase_value, 1.0)
        labels = np.asarray(measured.label_image, dtype=np.int32)
    elif stage == "final_label_matrix":
        entries = case.get("label_entries")
        if not isinstance(entries, list):
            raise PerimeterReferenceValidationError(
                "final_label_matrix requires label_entries",
                code="invalid_catalog",
                context={"case_id": case_id},
            )
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 3:
                raise PerimeterReferenceValidationError(
                    "label_entries must contain [row, column, label] entries",
                    code="invalid_catalog",
                    context={"case_id": case_id},
                )
            row = _require_int(entry[0], "label_entries.row", case_id=case_id, minimum=0)
            col = _require_int(entry[1], "label_entries.column", case_id=case_id, minimum=0)
            label = _require_int(entry[2], "label_entries.label", case_id=case_id, minimum=1)
            if row >= height or col >= width:
                raise PerimeterReferenceValidationError(
                    "label entry out of bounds",
                    code="invalid_catalog",
                    context={"case_id": case_id, "row": row, "column": col},
                )
            if labels[row, col] != 0:
                raise PerimeterReferenceValidationError(
                    "duplicate label entry coordinate",
                    code="invalid_catalog",
                    context={"case_id": case_id, "row": row, "column": col},
                )
            labels[row, col] = label
    else:
        raise PerimeterReferenceValidationError(
            "unsupported case stage",
            code="invalid_catalog",
            context={"case_id": case_id, "stage": stage},
        )
    return labels


def _assert_topology(case: Mapping[str, object], topology: Sequence[ComponentTopology]) -> None:
    case_id = _require_str(case.get("id"), "case.id")
    assertions = case.get("topology_assertions")
    if not isinstance(assertions, Mapping):
        raise PerimeterReferenceValidationError(
            "topology_assertions are required",
            code="invalid_catalog",
            context={"case_id": case_id},
        )
    expected_count = _require_int(assertions.get("component_count"), "topology_assertions.component_count", case_id=case_id)
    if expected_count != len(topology):
        raise PerimeterReferenceValidationError(
            "topology component_count does not match prepared labels",
            code="invalid_catalog",
            context={"case_id": case_id, "expected": expected_count, "actual": len(topology)},
        )
    expected_first = assertions.get("first_pixels")
    expected_areas = assertions.get("areas")
    if not isinstance(expected_first, list) or not isinstance(expected_areas, list):
        raise PerimeterReferenceValidationError(
            "topology_assertions must include first_pixels and areas",
            code="invalid_catalog",
            context={"case_id": case_id},
        )
    if len(expected_first) != expected_count or len(expected_areas) != expected_count:
        raise PerimeterReferenceValidationError(
            "topology_assertions lengths must match component_count",
            code="invalid_catalog",
            context={"case_id": case_id},
        )
    ordered = sorted(topology, key=lambda item: item.canonical_component_id)
    for index, component in enumerate(ordered):
        pixel = expected_first[index]
        if not isinstance(pixel, list) or len(pixel) != 2:
            raise PerimeterReferenceValidationError(
                "first_pixels entries must be [row, column]",
                code="invalid_catalog",
                context={"case_id": case_id, "index": index},
            )
        expected_row = _require_int(pixel[0], "topology_assertions.first_pixels.row", case_id=case_id, minimum=0)
        expected_col = _require_int(pixel[1], "topology_assertions.first_pixels.column", case_id=case_id, minimum=0)
        expected_area = _require_int(expected_areas[index], "topology_assertions.areas", case_id=case_id, minimum=1)
        if (component.first_row, component.first_column) != (expected_row, expected_col):
            raise PerimeterReferenceValidationError(
                "topology first pixel does not match assertion",
                code="invalid_catalog",
                context={
                    "case_id": case_id,
                    "canonical_component_id": component.canonical_component_id,
                    "expected": (expected_row, expected_col),
                    "actual": (component.first_row, component.first_column),
                },
            )
        if component.area_px != expected_area:
            raise PerimeterReferenceValidationError(
                "topology area does not match assertion",
                code="invalid_catalog",
                context={
                    "case_id": case_id,
                    "canonical_component_id": component.canonical_component_id,
                    "expected": expected_area,
                    "actual": component.area_px,
                },
            )


def prepare_catalog(catalog_path: Path | str) -> PreparedCatalog:
    path = Path(catalog_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if _require_int(raw.get("schema_version"), "schema_version") != SCHEMA_VERSION:
        raise PerimeterReferenceValidationError(
            "unsupported catalog schema version",
            code="invalid_catalog",
            context={"path": str(path)},
        )
    _require_str(raw.get("catalog_id"), "catalog_id")
    source_commit = _require_str(raw.get("source_commit"), "source_commit")
    if source_commit != ORIGINAL_SOURCE_COMMIT:
        raise PerimeterReferenceValidationError(
            "catalog source_commit is incompatible",
            code="invalid_catalog",
            context={"expected": ORIGINAL_SOURCE_COMMIT, "actual": source_commit},
        )

    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise PerimeterReferenceValidationError(
            "catalog cases must be a nonempty list",
            code="invalid_catalog",
            context={"path": str(path)},
        )

    seen = set()
    prepared_cases: List[PreparedCase] = []
    for case in cases_raw:
        if not isinstance(case, Mapping):
            raise PerimeterReferenceValidationError("each case must be an object", code="invalid_catalog")
        case_id = _require_str(case.get("id"), "case.id")
        if case_id in seen:
            raise PerimeterReferenceValidationError(
                "duplicate case id",
                code="invalid_catalog",
                context={"case_id": case_id},
            )
        seen.add(case_id)
        stage = _require_str(case.get("stage"), "case.stage", case_id=case_id)
        purpose = _require_str(case.get("purpose"), "case.purpose", case_id=case_id)
        shape_raw = case.get("shape")
        if not isinstance(shape_raw, list) or len(shape_raw) != 2:
            raise PerimeterReferenceValidationError(
                "case.shape must contain [height, width]",
                code="invalid_catalog",
                context={"case_id": case_id},
            )
        height = _require_int(shape_raw[0], "shape[0]", case_id=case_id, minimum=1)
        width = _require_int(shape_raw[1], "shape[1]", case_id=case_id, minimum=1)
        scale = _require_positive_float(case.get("scale_px_per_mm"), "scale_px_per_mm", case_id=case_id)

        labels = _labels_from_case(case)
        if labels.shape != (height, width):
            raise PerimeterReferenceValidationError(
                "prepared labels shape does not match case shape",
                code="invalid_catalog",
                context={"case_id": case_id},
            )
        topology = _topology_from_labels(case_id, labels)
        _assert_topology(case, topology)
        prepared_cases.append(
            PreparedCase(
                case_id=case_id,
                stage=stage,
                purpose=purpose,
                shape=(height, width),
                scale_px_per_mm=scale,
                labels=np.asarray(labels, dtype=np.int32),
                foreground_pixel_count=int(np.count_nonzero(labels > 0)),
                component_count=len(topology),
                topology=tuple(topology),
            )
        )

    return PreparedCatalog(
        catalog_path=path,
        catalog_hash=_hash_file_lf(path),
        cases=tuple(prepared_cases),
    )


def _write_constructor(
    prepared: PreparedCatalog,
    constructor_path: Path,
    *,
    preparation_hash: str,
    runner_hash: str,
) -> None:
    lines: List[str] = []
    lines.append("function bundle = foams105_perimeter_cases_generated()")
    lines.append("% Auto-generated by reference perimeter preparation tooling.")
    lines.append(f"bundle.schema_version = {SCHEMA_VERSION};")
    lines.append(f"bundle.source_commit = {_literal_matlab_string(ORIGINAL_SOURCE_COMMIT)};")
    lines.append(f"bundle.case_catalog_hash = {_literal_matlab_string(prepared.catalog_hash)};")
    lines.append(f"bundle.preparation_hash = {_literal_matlab_string(preparation_hash)};")
    lines.append(f"bundle.runner_hash = {_literal_matlab_string(runner_hash)};")
    lines.append(f"bundle.case_count = {len(prepared.cases)};")
    lines.append("cases = struct('id', {}, 'stage', {}, 'purpose', {}, 'height', {}, 'width', {}, 'scale_px_per_mm', {}, 'labels', {}, 'foreground_pixel_count', {}, 'component_count', {}, 'expected_first_rows', {}, 'expected_first_cols', {}, 'expected_areas', {});")
    for index, case in enumerate(prepared.cases, start=1):
        ordered = sorted(case.topology, key=lambda item: item.canonical_component_id)
        first_rows = " ".join(str(item.first_row + 1) for item in ordered)
        first_cols = " ".join(str(item.first_column + 1) for item in ordered)
        areas = " ".join(str(item.area_px) for item in ordered)
        lines.append(f"cases({index}).id = {_literal_matlab_string(case.case_id)};")
        lines.append(f"cases({index}).stage = {_literal_matlab_string(case.stage)};")
        lines.append(f"cases({index}).purpose = {_literal_matlab_string(case.purpose)};")
        lines.append(f"cases({index}).height = {case.shape[0]};")
        lines.append(f"cases({index}).width = {case.shape[1]};")
        lines.append(f"cases({index}).scale_px_per_mm = {case.scale_px_per_mm:.17g};")
        lines.append(f"cases({index}).labels = {_matrix_to_matlab_literal(case.labels)};")
        lines.append(f"cases({index}).foreground_pixel_count = {case.foreground_pixel_count};")
        lines.append(f"cases({index}).component_count = {case.component_count};")
        if ordered:
            lines.append(f"cases({index}).expected_first_rows = [{first_rows}];")
            lines.append(f"cases({index}).expected_first_cols = [{first_cols}];")
            lines.append(f"cases({index}).expected_areas = [{areas}];")
        else:
            lines.append(f"cases({index}).expected_first_rows = [];")
            lines.append(f"cases({index}).expected_first_cols = [];")
            lines.append(f"cases({index}).expected_areas = [];")
    lines.append("bundle.cases = cases;")
    lines.append("end")
    constructor_path.parent.mkdir(parents=True, exist_ok=True)
    constructor_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_matlab_constructor(
    catalog_path: Path | str,
    constructor_path: Path | str,
    manifest_path: Path | str,
    *,
    runner_path: Path | str,
    preparation_path: Optional[Path | str] = None,
) -> PreparedConstructorResult:
    prepared = prepare_catalog(catalog_path)
    constructor = Path(constructor_path)
    manifest = Path(manifest_path)
    runner = Path(runner_path)
    prep_path = Path(preparation_path) if preparation_path is not None else Path(__file__)

    if not runner.exists():
        raise PerimeterReferenceValidationError(
            "runner_path does not exist",
            code="missing_runner",
            context={"runner_path": str(runner)},
        )

    preparation_hash = _hash_file_lf(prep_path)
    runner_hash = _hash_file_lf(runner)
    _write_constructor(prepared, constructor, preparation_hash=preparation_hash, runner_hash=runner_hash)
    constructor_hash = _hash_file_lf(constructor)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_commit": ORIGINAL_SOURCE_COMMIT,
        "catalog_path": str(prepared.catalog_path),
        "constructor_path": str(constructor),
        "runner_path": str(runner),
        "case_count": len(prepared.cases),
        "case_catalog_hash": prepared.catalog_hash,
        "preparation_hash": preparation_hash,
        "runner_hash": runner_hash,
        "constructor_hash": constructor_hash,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return PreparedConstructorResult(
        constructor_path=constructor,
        manifest_path=manifest,
        catalog_hash=prepared.catalog_hash,
        preparation_hash=preparation_hash,
        runner_hash=runner_hash,
        constructor_hash=constructor_hash,
        case_count=len(prepared.cases),
    )


def _parse_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise PerimeterReferenceValidationError(
                "TSV file is missing header",
                code="invalid_export",
                context={"path": str(path)},
            )
        return [dict(row) for row in reader]


def _parse_required_int(value: str, *, field: str, context: Mapping[str, object]) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise PerimeterReferenceValidationError(
            f"{field} must be an integer",
            code="invalid_export",
            context=dict(context),
        ) from exc
    return parsed


def _parse_optional_float(value: str) -> Optional[float]:
    if value == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return parsed
    return parsed


def _parse_metadata(path: Path) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 2:
                raise PerimeterReferenceValidationError(
                    "metadata.tsv rows must have exactly two tab-separated fields",
                    code="invalid_export",
                    context={"path": str(path), "line": line_number},
                )
            key, value = parts
            if key in metadata:
                raise PerimeterReferenceValidationError(
                    "metadata.tsv contains duplicate keys",
                    code="invalid_export",
                    context={"path": str(path), "key": key},
                )
            metadata[key] = value
    return metadata


def import_reference_run(
    run_dir: Path | str,
    catalog_path: Path | str,
    *,
    expected_case_catalog_hash: str,
    expected_preparation_hash: str,
    expected_runner_hash: str,
) -> ImportedReferenceRun:
    run_path = Path(run_dir)
    required_files = {
        "metadata": run_path / "metadata.tsv",
        "cases": run_path / "cases.tsv",
        "labels": run_path / "labels.tsv",
        "components": run_path / "components.tsv",
        "completed": run_path / "completed.marker",
    }
    for name, path in required_files.items():
        if not path.exists():
            raise PerimeterReferenceValidationError(
                "reference export is missing required files",
                code="incomplete_export",
                context={"missing": name, "path": str(path)},
            )
    if required_files["completed"].read_text(encoding="utf-8").strip() != "COMPLETE":
        raise PerimeterReferenceValidationError(
            "reference export is missing completion marker",
            code="incomplete_export",
            context={"path": str(required_files["completed"] )},
        )

    metadata = _parse_metadata(required_files["metadata"])
    for required_key in (
        "schema_version",
        "source_commit",
        "case_catalog_hash",
        "preparation_hash",
        "runner_hash",
        "runtime_identity",
        "matlab_version",
        "matlab_release",
        "image_processing_toolbox_version",
        "invocation",
        "timestamp_description",
    ):
        if required_key not in metadata:
            raise PerimeterReferenceValidationError(
                "metadata.tsv is missing required keys",
                code="invalid_export",
                context={"missing_key": required_key},
            )
    if metadata["schema_version"] != str(SCHEMA_VERSION):
        raise PerimeterReferenceValidationError(
            "metadata schema_version mismatch",
            code="invalid_export",
            context={"actual": metadata["schema_version"]},
        )
    if metadata["source_commit"] != ORIGINAL_SOURCE_COMMIT:
        raise PerimeterReferenceValidationError(
            "metadata source_commit mismatch",
            code="invalid_export",
            context={"actual": metadata["source_commit"]},
        )

    hash_checks = {
        "case_catalog_hash": expected_case_catalog_hash,
        "preparation_hash": expected_preparation_hash,
        "runner_hash": expected_runner_hash,
    }
    for key, expected in hash_checks.items():
        if metadata[key] != expected:
            raise PerimeterReferenceValidationError(
                "metadata hash mismatch",
                code="hash_mismatch",
                context={"field": key, "expected": expected, "actual": metadata[key]},
            )

    prepared = prepare_catalog(catalog_path)
    if prepared.catalog_hash != expected_case_catalog_hash:
        raise PerimeterReferenceValidationError(
            "expected_case_catalog_hash does not match current catalog file",
            code="hash_mismatch",
            context={"expected": expected_case_catalog_hash, "actual": prepared.catalog_hash},
        )

    cases_rows = _parse_tsv(required_files["cases"])
    labels_rows = _parse_tsv(required_files["labels"])
    components_rows = _parse_tsv(required_files["components"])

    expected_case_ids = {case.case_id for case in prepared.cases}
    seen_case_ids = {row.get("case_id", "") for row in cases_rows}
    if seen_case_ids != expected_case_ids:
        raise PerimeterReferenceValidationError(
            "cases.tsv case coverage mismatch",
            code="invalid_export",
            context={"expected": sorted(expected_case_ids), "actual": sorted(seen_case_ids)},
        )

    case_lookup = {case.case_id: case for case in prepared.cases}
    labels_by_case: Dict[str, np.ndarray] = {}
    for case in prepared.cases:
        labels_by_case[case.case_id] = np.full(case.shape, -1, dtype=np.int64)

    for row in labels_rows:
        case_id = row.get("case_id", "")
        if case_id not in case_lookup:
            raise PerimeterReferenceValidationError(
                "labels.tsv contains unknown case",
                code="invalid_export",
                context={"case_id": case_id},
            )
        case = case_lookup[case_id]
        r = _parse_required_int(row.get("row", ""), field="row", context={"case_id": case_id})
        c = _parse_required_int(row.get("column", ""), field="column", context={"case_id": case_id})
        label = _parse_required_int(row.get("label", ""), field="label", context={"case_id": case_id})
        if r < 0 or c < 0 or r >= case.shape[0] or c >= case.shape[1]:
            raise PerimeterReferenceValidationError(
                "labels.tsv contains out-of-bounds coordinates",
                code="invalid_export",
                context={"case_id": case_id, "row": r, "column": c},
            )
        if labels_by_case[case_id][r, c] != -1:
            raise PerimeterReferenceValidationError(
                "labels.tsv contains duplicate coordinates",
                code="invalid_export",
                context={"case_id": case_id, "row": r, "column": c},
            )
        labels_by_case[case_id][r, c] = label

    for case in prepared.cases:
        matrix = labels_by_case[case.case_id]
        if int(np.count_nonzero(matrix == -1)) != 0:
            raise PerimeterReferenceValidationError(
                "labels.tsv does not include the full label matrix",
                code="invalid_export",
                context={"case_id": case.case_id},
            )
        if not np.array_equal(np.asarray(matrix, dtype=np.int32), case.labels):
            raise PerimeterReferenceValidationError(
                "labels.tsv does not match prepared case labels",
                code="label_mismatch",
                context={"case_id": case.case_id},
            )

    parsed_components: List[CapturedComponent] = []
    index_seen = set()
    for row in components_rows:
        case_id = row.get("case_id", "")
        if case_id not in case_lookup:
            raise PerimeterReferenceValidationError(
                "components.tsv contains unknown case",
                code="invalid_export",
                context={"case_id": case_id},
            )
        native_id = _parse_required_int(
            row.get("native_component_id", ""),
            field="native_component_id",
            context={"case_id": case_id},
        )
        canonical_id = _parse_required_int(
            row.get("canonical_component_id", ""),
            field="canonical_component_id",
            context={"case_id": case_id},
        )
        key = (case_id, native_id, canonical_id)
        if key in index_seen:
            raise PerimeterReferenceValidationError(
                "components.tsv contains duplicate component keys",
                code="invalid_export",
                context={"case_id": case_id, "native_component_id": native_id, "canonical_component_id": canonical_id},
            )
        index_seen.add(key)

        first_row = _parse_required_int(row.get("first_row", ""), field="first_row", context={"case_id": case_id})
        first_col = _parse_required_int(row.get("first_column", ""), field="first_column", context={"case_id": case_id})
        area_px = _parse_required_int(row.get("area_px", ""), field="area_px", context={"case_id": case_id})
        perimeter_status = row.get("perimeter_status", "")
        perimeter_old_status = row.get("perimeter_old_status", "")
        if not perimeter_status or not perimeter_old_status:
            raise PerimeterReferenceValidationError(
                "component perimeter statuses must be explicit",
                code="invalid_export",
                context={"case_id": case_id, "native_component_id": native_id},
            )
        perimeter_px = _parse_optional_float(row.get("perimeter_px", ""))
        perimeter_old_px = _parse_optional_float(row.get("perimeter_old_px", ""))

        if perimeter_status == "success":
            if perimeter_px is None:
                raise PerimeterReferenceValidationError(
                    "successful Perimeter requires a numeric value",
                    code="invalid_export",
                    context={"case_id": case_id, "native_component_id": native_id},
                )
            if not math.isfinite(perimeter_px) or perimeter_px < 0.0:
                raise PerimeterReferenceValidationError(
                    "Perimeter must be finite and nonnegative",
                    code="invalid_export",
                    context={"case_id": case_id, "native_component_id": native_id, "value": perimeter_px},
                )
        if perimeter_old_status == "success":
            if perimeter_old_px is None:
                raise PerimeterReferenceValidationError(
                    "successful PerimeterOld requires a numeric value",
                    code="invalid_export",
                    context={"case_id": case_id, "native_component_id": native_id},
                )
            if not math.isfinite(perimeter_old_px) or perimeter_old_px < 0.0:
                raise PerimeterReferenceValidationError(
                    "PerimeterOld must be finite and nonnegative",
                    code="invalid_export",
                    context={"case_id": case_id, "native_component_id": native_id, "value": perimeter_old_px},
                )

        parsed_components.append(
            CapturedComponent(
                case_id=case_id,
                native_component_id=native_id,
                canonical_component_id=canonical_id,
                first_row=first_row,
                first_column=first_col,
                area_px=area_px,
                perimeter_px=perimeter_px,
                perimeter_status=perimeter_status,
                perimeter_old_px=perimeter_old_px,
                perimeter_old_status=perimeter_old_status,
                property_error=row.get("property_error", ""),
            )
        )

    parsed_by_case: Dict[str, List[CapturedComponent]] = {case.case_id: [] for case in prepared.cases}
    for component in parsed_components:
        parsed_by_case[component.case_id].append(component)

    for case in prepared.cases:
        expected_topology = sorted(case.topology, key=lambda item: (item.native_component_id, item.canonical_component_id))
        actual_topology = sorted(parsed_by_case[case.case_id], key=lambda item: (item.native_component_id, item.canonical_component_id))
        if len(expected_topology) != len(actual_topology):
            raise PerimeterReferenceValidationError(
                "components.tsv component count mismatch",
                code="component_mismatch",
                context={"case_id": case.case_id, "expected": len(expected_topology), "actual": len(actual_topology)},
            )
        for expected, actual in zip(expected_topology, actual_topology):
            if (
                expected.native_component_id != actual.native_component_id
                or expected.canonical_component_id != actual.canonical_component_id
                or expected.first_row != actual.first_row
                or expected.first_column != actual.first_column
                or expected.area_px != actual.area_px
            ):
                raise PerimeterReferenceValidationError(
                    "components.tsv topology/identity mismatch",
                    code="component_mismatch",
                    context={
                        "case_id": case.case_id,
                        "expected": expected,
                        "actual": actual,
                    },
                )

    return ImportedReferenceRun(
        run_dir=run_path,
        metadata=metadata,
        cases=prepared.cases,
        components=tuple(parsed_components),
    )


def _relative_difference(reference: float, candidate: float) -> Optional[float]:
    if reference == 0.0:
        if candidate == 0.0:
            return 0.0
        return None
    return abs(candidate - reference) / abs(reference)


def compare_perimeter_and_old(run: ImportedReferenceRun) -> ComparisonReport:
    rows: List[ComparisonRow] = []
    notes: List[str] = []
    mismatches = 0
    for component in run.components:
        if component.perimeter_status == "success" and component.perimeter_old_status == "success":
            assert component.perimeter_px is not None
            assert component.perimeter_old_px is not None
            absolute = abs(component.perimeter_px - component.perimeter_old_px)
            relative = _relative_difference(component.perimeter_old_px, component.perimeter_px)
            within = math.isclose(
                component.perimeter_px,
                component.perimeter_old_px,
                rel_tol=REL_TOL,
                abs_tol=ABS_TOL,
            )
            if not within:
                mismatches += 1
            rows.append(
                ComparisonRow(
                    case_id=component.case_id,
                    native_component_id=component.native_component_id,
                    reference_field="perimeter_old_px",
                    candidate_field="perimeter_px",
                    reference_value=component.perimeter_old_px,
                    candidate_value=component.perimeter_px,
                    absolute_difference=absolute,
                    relative_difference=relative,
                    within_tolerance=within,
                )
            )
        elif component.perimeter_old_status != "success":
            notes.append(
                f"{component.case_id}:{component.native_component_id} perimeter_old_status={component.perimeter_old_status}"
            )

    compared_rows = len(rows)
    numerical_valid = mismatches == 0
    if compared_rows == 0:
        numerical_valid = False
        notes.append("no comparable perimeter/perimeter_old rows")

    return ComparisonReport(
        schema_valid=True,
        topology_valid=True,
        numerical_valid=numerical_valid,
        historical_compatible=False,
        compared_rows=compared_rows,
        mismatch_rows=mismatches,
        rows=tuple(rows),
        notes=tuple(sorted(set(notes))),
    )


def compare_runs(run_a: ImportedReferenceRun, run_b: ImportedReferenceRun) -> ComparisonReport:
    index_a = {(item.case_id, item.native_component_id): item for item in run_a.components}
    index_b = {(item.case_id, item.native_component_id): item for item in run_b.components}
    if set(index_a) != set(index_b):
        raise PerimeterReferenceValidationError(
            "run component identities differ",
            code="run_mismatch",
            context={"left": len(index_a), "right": len(index_b)},
        )

    rows: List[ComparisonRow] = []
    mismatches = 0
    for identity in sorted(index_a):
        left = index_a[identity]
        right = index_b[identity]
        if (
            left.canonical_component_id != right.canonical_component_id
            or left.first_row != right.first_row
            or left.first_column != right.first_column
            or left.area_px != right.area_px
        ):
            raise PerimeterReferenceValidationError(
                "run component topology mismatch",
                code="run_mismatch",
                context={"identity": identity},
            )
        if left.perimeter_status == "success" and right.perimeter_status == "success":
            assert left.perimeter_px is not None
            assert right.perimeter_px is not None
            absolute = abs(right.perimeter_px - left.perimeter_px)
            relative = _relative_difference(left.perimeter_px, right.perimeter_px)
            within = math.isclose(
                left.perimeter_px,
                right.perimeter_px,
                rel_tol=REL_TOL,
                abs_tol=ABS_TOL,
            )
            if not within:
                mismatches += 1
            rows.append(
                ComparisonRow(
                    case_id=left.case_id,
                    native_component_id=left.native_component_id,
                    reference_field="run_a_perimeter_px",
                    candidate_field="run_b_perimeter_px",
                    reference_value=left.perimeter_px,
                    candidate_value=right.perimeter_px,
                    absolute_difference=absolute,
                    relative_difference=relative,
                    within_tolerance=within,
                )
            )

    compared_rows = len(rows)
    numerical_valid = compared_rows > 0 and mismatches == 0
    return ComparisonReport(
        schema_valid=True,
        topology_valid=True,
        numerical_valid=numerical_valid,
        historical_compatible=False,
        compared_rows=compared_rows,
        mismatch_rows=mismatches,
        rows=tuple(rows),
        notes=() if compared_rows > 0 else ("no overlapping successful perimeter rows",),
    )
