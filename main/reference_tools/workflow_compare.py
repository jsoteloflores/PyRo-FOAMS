"""Strict import and comparison for original-source FOAMS workflow captures."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from main.core.legacy_exports import Foams105ExportTableProjection, build_foams105_export_tables
from main.core.legacy_shape_statistics import Foams105ShapeValue
from main.core.legacy_summaries import Foams105RealValue

from .workflow_capture import _verify_captured_cases
from .workflow_reference import (
    ORIGINAL_SOURCE_COMMIT,
    WorkflowCaseCatalog,
    WorkflowPythonRunResult,
    WorkflowReferenceValidationError,
    load_workflow_case_catalog,
    load_workflow_extraction_map,
    run_python_workflow_case,
)
from .workflow_source import source_hashes


@dataclass(frozen=True)
class ClassifiedNumber:
    classification: str
    value: Optional[complex]
    reason: Optional[str] = None


@dataclass(frozen=True)
class ImportedWorkflowCapture:
    directory: Path
    capture: Mapping[str, object]
    manifest: Mapping[str, object]
    catalog: WorkflowCaseCatalog
    cases_by_id: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class WorkflowComparisonRow:
    case_id: str
    stage: str
    table: Optional[str]
    row_index: Optional[int]
    column: Optional[str]
    identity: Optional[Tuple[str, int]]
    reference_value: object
    python_value: object
    absolute_difference: Optional[float]
    relative_difference: Optional[float]
    tolerance: Optional[Tuple[float, float]]
    unit: str
    within_tolerance: bool
    classification_match: bool
    code: Optional[str] = None


@dataclass(frozen=True)
class WorkflowComparisonReport:
    source_commit: str
    matlab_release: str
    python_revision: str
    comparison_count: int
    failure_count: int
    unavailable_stage_count: int
    first_divergent_stage: Optional[str]
    affected_tables: Tuple[str, ...]
    known_extensions: Tuple[str, ...]
    rows: Tuple[WorkflowComparisonRow, ...]


TABLE_STAGE = {
    "NA_mag": "histogram_and_na",
    "Raw_data": "calibration_and_diameter_filter",
    "Rawcutoffs_data": "raw_selection",
    "Calc_out": "nv_volume_and_distribution_summary",
    "Area_out": "image_area",
    "Ves_out": "vesicularity_and_volume_normalization",
    "Misc_out": "category_and_density_summary",
    "Shape_out": "shape_and_initial_threshold",
    "Stats_out": "shape_statistics",
}

TABLE_DOWNSTREAM = {
    "direct_measurement": ("Raw_data", "Rawcutoffs_data", "Area_out", "Ves_out", "Shape_out", "Stats_out"),
    "calibration_and_diameter_filter": ("Raw_data", "Rawcutoffs_data", "NA_mag", "Area_out", "Ves_out", "Shape_out", "Stats_out"),
    "image_area": ("NA_mag", "Ves_out", "Misc_out"),
    "histogram_and_na": ("NA_mag", "Rawcutoffs_data", "Calc_out", "Ves_out", "Misc_out", "Shape_out", "Stats_out"),
    "binned_selection": ("Calc_out", "Ves_out", "Misc_out"),
    "raw_selection": ("Rawcutoffs_data", "Ves_out", "Misc_out", "Shape_out", "Stats_out"),
    "vesicularity_and_volume_normalization": ("Ves_out", "Misc_out", "Calc_out"),
    "shape_and_initial_threshold": ("Shape_out", "Stats_out"),
    "shape_statistics": ("Stats_out",),
}

KNOWN_EXTENSIONS = (
    "Python exposes typed undefined values and serializes them as classified NA cells.",
    "Python defines empty-statistics behavior as an extension; original basicstats is called only on finite nonempty vectors.",
    "Area_out is unavailable when source image-number and nonzero-area vectors are misaligned.",
    "Analytical moment/orientation conventions are not promoted to MATLAB equivalence by this comparison.",
    "A source-domain failure is a recorded outcome, not a passing numerical comparison.",
)


def _validation_error(message: str, code: str, **context: object) -> WorkflowReferenceValidationError:
    return WorkflowReferenceValidationError(message, code=code, context=context)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_number(cell: object, *, context: Mapping[str, object]) -> ClassifiedNumber:
    if not isinstance(cell, dict) or not isinstance(cell.get("classification"), str):
        raise _validation_error("numeric cell is malformed", "malformed_numeric_cell", **dict(context))
    classification = cell["classification"]
    if classification == "finite":
        try:
            value = float(cell["real"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise _validation_error("finite numeric cell is malformed", "malformed_numeric_cell", **dict(context)) from exc
        if not math.isfinite(value):
            raise _validation_error("finite numeric cell contains a nonfinite value", "malformed_numeric_cell", **dict(context))
        return ClassifiedNumber("finite", value)
    if classification == "complex":
        try:
            value = complex(float(cell["real"]), float(cell["imaginary"]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise _validation_error("complex numeric cell is malformed", "malformed_numeric_cell", **dict(context)) from exc
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise _validation_error("complex numeric cell contains a nonfinite value", "malformed_numeric_cell", **dict(context))
        return ClassifiedNumber("complex", value)
    if classification in ("nan", "positive_infinity", "negative_infinity"):
        return ClassifiedNumber(classification, None)
    raise _validation_error("numeric cell classification is unsupported", "malformed_numeric_cell", **dict(context), classification=classification)


def _decode_array(value: object, *, context: Mapping[str, object]) -> Tuple[Tuple[int, ...], Tuple[ClassifiedNumber, ...]]:
    if not isinstance(value, dict) or value.get("kind") != "numeric_array":
        raise _validation_error("numeric array record is malformed", "malformed_numeric_array", **dict(context))
    shape = value.get("shape")
    cells = value.get("cells")
    if (
        not isinstance(shape, list)
        or any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape)
        or not isinstance(cells, list)
    ):
        raise _validation_error("numeric array dimensions or cells are malformed", "malformed_numeric_array", **dict(context))
    expected_count = math.prod(shape)
    if expected_count != len(cells):
        raise _validation_error(
            "numeric array shape does not match flattened cell count",
            "numeric_array_length_mismatch",
            **dict(context),
            shape=tuple(shape),
            cell_count=len(cells),
        )
    decoded = tuple(_decode_number(cell, context={**dict(context), "cell_index": index}) for index, cell in enumerate(cells))
    return tuple(shape), decoded


def _decode_matrix(value: object, *, context: Mapping[str, object]) -> Tuple[Tuple[ClassifiedNumber, ...], ...]:
    shape, cells = _decode_array(value, context=context)
    if len(shape) != 2:
        raise _validation_error("table numeric array must be rank two", "invalid_table_rank", **dict(context), shape=shape)
    rows, columns = shape
    return tuple(tuple(cells[column * rows + row] for column in range(columns)) for row in range(rows))


def import_workflow_reference(
    capture_directory: Union[Path, str],
    *,
    expected_catalog_path: Optional[Union[Path, str]] = None,
    expected_extraction_map_path: Optional[Union[Path, str]] = None,
    repository_root: Optional[Union[Path, str]] = None,
    expected_matlab_release: Optional[str] = "R2025b",
    expected_toolbox_version: Optional[str] = "25.2",
) -> ImportedWorkflowCapture:
    """Import only complete, checksum-valid, source/runtime/settings-matched runs."""

    directory = Path(capture_directory).resolve()
    required_files = (
        "capture.json",
        "manifest.json",
        "checksums.json",
        "completed.marker",
        "input_catalog.json",
        "extraction_map.json",
        "foams105_workflow_cases_generated.m",
        "preparation_manifest.json",
        "foams105_workflow_capture.m",
        "matlab.stdout.log",
        "matlab.stderr.log",
        "process_exit_code.txt",
        "run_metadata.json",
    )
    missing = tuple(name for name in required_files if not (directory / name).is_file())
    if missing:
        raise _validation_error("capture is incomplete or missing required files", "incomplete_capture", missing=missing)
    marker = (directory / "completed.marker").read_text(encoding="ascii").splitlines()
    if len(marker) < 3 or marker[0] != "COMPLETE":
        raise _validation_error("capture lacks a valid completed marker", "incomplete_capture_marker")
    process_exit_code = (directory / "process_exit_code.txt").read_text(encoding="ascii").strip()
    if process_exit_code != "0" or "process_exit_code=0" not in marker:
        raise _validation_error("MATLAB process did not exit successfully", "matlab_process_failed", process_exit_code=process_exit_code)

    try:
        capture = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
        run_metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _validation_error("capture metadata is malformed", "malformed_capture_metadata", cause=str(exc)) from exc

    file_hashes = checksums.get("files") if isinstance(checksums, dict) else None
    if not isinstance(file_hashes, dict):
        raise _validation_error("capture checksums file is malformed", "malformed_checksums")
    for name, expected_hash in file_hashes.items():
        if not isinstance(name, str) or Path(name).name != name or not isinstance(expected_hash, str):
            raise _validation_error("checksum entry is malformed or escapes the capture directory", "malformed_checksums", file=name)
        path = directory / name
        if not path.is_file():
            raise _validation_error("checksummed capture artifact is missing", "missing_checksummed_file", file=name)
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise _validation_error("capture artifact checksum mismatch", "capture_checksum_mismatch", file=name, expected=expected_hash, actual=actual_hash)
    if set(file_hashes) != set(required_files) - {"checksums.json", "completed.marker"}:
        raise _validation_error("capture checksum inventory is incomplete or has unexpected entries", "checksum_inventory_mismatch")

    if run_metadata.get("process_exit_code") != 0 or run_metadata.get("timed_out") is not False:
        raise _validation_error("MATLAB process metadata indicates failure or timeout", "matlab_process_failed")
    if capture.get("schema_version") != 1 or capture.get("source_commit") != ORIGINAL_SOURCE_COMMIT:
        raise _validation_error("capture schema or source identity is incompatible", "wrong_capture_source")
    if marker[1] != f"source_commit={ORIGINAL_SOURCE_COMMIT}":
        raise _validation_error("completion marker source commit is inconsistent", "wrong_capture_source")
    if expected_matlab_release is not None and capture.get("matlab_release") != expected_matlab_release:
        raise _validation_error("capture was produced by an unexpected MATLAB release", "wrong_matlab_runtime", expected=expected_matlab_release, actual=capture.get("matlab_release"))
    if expected_toolbox_version is not None and capture.get("image_processing_toolbox_version") != expected_toolbox_version:
        raise _validation_error("capture was produced by an unexpected Image Processing Toolbox", "wrong_matlab_toolbox", expected=expected_toolbox_version, actual=capture.get("image_processing_toolbox_version"))

    catalog_path = Path(expected_catalog_path) if expected_catalog_path is not None else directory / "input_catalog.json"
    catalog = load_workflow_case_catalog(catalog_path)
    if _sha256(directory / "input_catalog.json") != catalog.raw_sha256 or capture.get("catalog_raw_sha256") != catalog.raw_sha256:
        raise _validation_error("capture catalog bytes differ from the pinned inputs", "wrong_capture_settings")
    extraction_map_path = Path(expected_extraction_map_path) if expected_extraction_map_path is not None else directory / "extraction_map.json"
    extraction_map_raw, _ = source_hashes(extraction_map_path)
    if extraction_map_raw != capture.get("extraction_map_raw_sha256") or extraction_map_raw != _sha256(directory / "extraction_map.json"):
        raise _validation_error("capture extraction map differs from the current pin", "wrong_extraction_map")
    extraction_map = load_workflow_extraction_map(extraction_map_path)
    if capture.get("runner_raw_sha256") != _sha256(directory / "foams105_workflow_capture.m"):
        raise _validation_error("capture runner differs from the recorded source runner", "wrong_capture_runner")
    if capture.get("constructor_raw_sha256") != _sha256(directory / "foams105_workflow_cases_generated.m"):
        raise _validation_error("generated constructor checksum mismatch", "wrong_capture_settings")
    if manifest.get("capture_sha256") != _sha256(directory / "capture.json"):
        raise _validation_error("manifest capture checksum mismatch", "capture_manifest_checksum_mismatch")
    if manifest.get("source_commit") != ORIGINAL_SOURCE_COMMIT or manifest.get("runner_raw_sha256") != capture.get("runner_raw_sha256"):
        raise _validation_error("capture manifest provenance is inconsistent", "wrong_capture_source")

    source_records = capture.get("source_files")
    source_pins = extraction_map.get("source_files")
    if not isinstance(source_records, list) or not isinstance(source_pins, dict):
        raise _validation_error("capture source file provenance is malformed", "wrong_capture_source")
    source_by_path = {record.get("path"): record for record in source_records if isinstance(record, dict)}
    if set(source_by_path) != set(source_pins):
        raise _validation_error("capture source file set differs from the extraction map", "wrong_capture_source")
    for name, pin in source_pins.items():
        if source_by_path[name].get("lf_sha256") != pin["lf_sha256"]:
            raise _validation_error("capture normalized source hash differs from the map", "source_hash_mismatch", file=name)

    _verify_captured_cases(capture, catalog)
    cases = {case["case_id"]: case for case in capture["cases"]}
    if repository_root is not None and Path(capture_directory).resolve() == Path(repository_root).resolve():
        raise _validation_error("capture directory must remain outside the repository source tree", "capture_inside_repository")
    return ImportedWorkflowCapture(directory, capture, manifest, catalog, cases)


def _python_number(value: object) -> ClassifiedNumber:
    if isinstance(value, Foams105ShapeValue):
        if value.value is None:
            return ClassifiedNumber("undefined", None, value.reason)
        value = value.value
    if isinstance(value, Foams105RealValue):
        if value.value is None:
            return ClassifiedNumber("undefined", None, value.reason or value.upstream_reason)
        value = value.value
    if value is None:
        return ClassifiedNumber("unavailable", None)
    if isinstance(value, (bool, np.bool_)):
        return ClassifiedNumber("integer", int(value))
    if isinstance(value, (int, np.integer)):
        return ClassifiedNumber("integer", int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return ClassifiedNumber("nan", None)
        if math.isinf(number):
            return ClassifiedNumber("positive_infinity" if number > 0 else "negative_infinity", None)
        return ClassifiedNumber("finite", number)
    if isinstance(value, str):
        return ClassifiedNumber("text", value)
    raise TypeError(f"unsupported Python comparison value: {type(value).__name__}")


def _reference_matrix(table: Mapping[str, object], table_name: str):
    headers = table.get("headers")
    labels = table.get("row_labels")
    shape, values = _decode_array(table.get("values"), context={"table": table_name})
    if len(shape) != 2:
        raise _validation_error("captured table values are not a rank-two array", "invalid_table_rank", table=table_name)
    rows, columns = shape
    matrix = tuple(tuple(values[column * rows + row] for column in range(columns)) for row in range(rows))
    if not isinstance(headers, list) or not all(isinstance(header, str) for header in headers):
        raise _validation_error("captured table headers are malformed", "invalid_table_headers", table=table_name)
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise _validation_error("captured table row labels are malformed", "invalid_table_labels", table=table_name)
    if len(headers) != columns:
        raise _validation_error("captured table header width differs from its array", "table_header_width_mismatch", table=table_name)
    if table_name == "Stats_out":
        if len(labels) != rows:
            raise _validation_error("Stats_out row labels differ from numeric rows", "table_row_label_count_mismatch", table=table_name)
        headers = [""] + headers
        matrix = tuple((ClassifiedNumber("text", labels[row]),) + matrix[row] for row in range(rows))
    elif labels:
        raise _validation_error("non-Stats table unexpectedly contains row labels", "unexpected_table_row_labels", table=table_name)
    return tuple(headers), matrix


def _candidate_table_rows(table: Foams105ExportTableProjection):
    return tuple(tuple(_python_number(value) for value in row) for row in table.rows)


def _field_tolerance(table_name: str, column_index: int) -> Tuple[float, float, str]:
    if table_name in ("Raw_data.txt", "Rawcutoffs_data.txt"):
        if column_index == 0:
            return 1e-10, 1e-12, "mm"
        if column_index == 1:
            return 1e-10, 1e-12, "mm^2"
        if column_index == 2:
            return 1e-10, 1e-12, "mm"
        if column_index in (3, 4):
            return 1e-10, 1e-12, "mm"
        if column_index == 5:
            return 0.0, 1e-8, "degree"
        return 0.0, 0.0, "magnification-slot"
    if table_name == "NA_mag.txt":
        if column_index == 0:
            return 0.0, 0.0, "mm"
        if column_index in (1, 3, 5, 7):
            return 0.0, 0.0, "count"
        return 1e-10, 1e-12, "mm^-2"
    if table_name == "Calc_out.txt":
        units = ("mm", "mm^-2", "mm^-3", "dimensionless", "dimensionless", "log10(mm^-3)", "log10(mm)", "ln(mm^-4)")
        return 1e-10, 1e-12, units[column_index]
    if table_name == "Area_out.txt":
        return (0.0, 0.0, "image-index") if column_index == 0 else (1e-10, 1e-12, "mm^2")
    if table_name == "Ves_out.txt":
        unit = "dimensionless" if column_index == 6 else "percent"
        return 1e-10, 1e-12, unit
    if table_name == "Misc_out.txt":
        if column_index in (0, 1):
            return 0.0, 0.0, "count"
        if column_index in (2, 3):
            return 1e-10, 1e-12, "mm^-3"
        if column_index in (4, 5):
            return 1e-10, 1e-12, "mm"
        return 1e-10, 1e-12, "percent"
    if table_name == "Shape_out.txt":
        return 1e-10, 1e-12, "dimensionless"
    if table_name == "Stats_out.txt":
        return 0.0, 0.0, "four-decimal-display"
    return 1e-10, 1e-12, "source-unit"


def _compare_cell(
    *,
    case_id: str,
    stage: str,
    table_name: Optional[str],
    row_index: Optional[int],
    column_index: Optional[int],
    column_name: Optional[str],
    identity: Optional[Tuple[str, int]],
    reference: object,
    candidate: object,
    unit: str,
    tolerance_override: Optional[Tuple[float, float]] = None,
) -> WorkflowComparisonRow:
    source_value = reference if isinstance(reference, ClassifiedNumber) else _python_number(reference)
    python_value = candidate if isinstance(candidate, ClassifiedNumber) else _python_number(candidate)
    rtol, atol, declared_unit = _field_tolerance(table_name or "", column_index or 0)
    if tolerance_override is not None:
        rtol, atol = tolerance_override
    numeric_classes = ("finite", "integer", "complex")
    if (
        table_name == "Stats_out.txt"
        and source_value.classification in numeric_classes
        and python_value.classification in numeric_classes
    ):
        reference_display = format(float(source_value.value), ".4f")
        python_display = format(float(python_value.value), ".4f")
        within = reference_display == python_display
        absolute = abs(float(source_value.value) - float(python_value.value))
        relative = absolute / abs(float(source_value.value)) if float(source_value.value) != 0.0 else None
        classification_match = within
        tolerance = (0.0, 0.0)
    elif source_value.classification in numeric_classes and python_value.classification in numeric_classes:
        reference_number = complex(source_value.value)
        python_number = complex(python_value.value)
        absolute = abs(reference_number - python_number)
        scale = abs(reference_number)
        relative = absolute / scale if scale else None
        within = absolute <= atol + rtol * scale
        classification_match = True
        tolerance = (rtol, atol)
    elif source_value.classification != python_value.classification:
        within = False
        absolute = None
        relative = None
        classification_match = False
        tolerance = (rtol, atol)
    elif source_value.classification in ("text", "integer"):
        within = source_value.value == python_value.value
        absolute = 0.0 if within else None
        relative = 0.0 if within else None
        classification_match = within
        tolerance = (0.0, 0.0)
    elif source_value.classification in ("nan", "positive_infinity", "negative_infinity", "undefined", "unavailable"):
        within = source_value.classification == python_value.classification
        if (
            within
            and source_value.classification == "undefined"
            and source_value.reason is not None
            and python_value.reason is not None
        ):
            within = source_value.reason == python_value.reason
        absolute = 0.0 if within else None
        relative = 0.0 if within else None
        classification_match = within
        tolerance = (0.0, 0.0)
    else:
        reference_number = complex(source_value.value)
        python_number = complex(python_value.value)
        absolute = abs(reference_number - python_number)
        scale = abs(reference_number)
        relative = absolute / scale if scale else None
        within = absolute <= atol + rtol * scale
        classification_match = source_value.classification == python_value.classification
        tolerance = (rtol, atol)
    return WorkflowComparisonRow(
        case_id=case_id,
        stage=stage,
        table=table_name,
        row_index=row_index,
        column=column_name,
        identity=identity,
        reference_value={"classification": source_value.classification, "value": source_value.value, "reason": source_value.reason},
        python_value={"classification": python_value.classification, "value": python_value.value, "reason": python_value.reason},
        absolute_difference=absolute,
        relative_difference=relative,
        tolerance=tolerance,
        unit=declared_unit if table_name else unit,
        within_tolerance=within,
        classification_match=classification_match,
        code=None if within else "value_mismatch",
    )


def _matrix_rows_for_python(table_name: str, projection: Foams105ExportTableProjection):
    if table_name == "Stats_out.txt":
        rows = tuple((row[0],) + row[1:] for row in projection.rows)
    else:
        rows = projection.rows
    return tuple(tuple(_python_number(value) for value in row) for row in rows)


def _flatten_encoded_array(value: object, *, context: Mapping[str, object]):
    shape, cells = _decode_array(value, context=context)
    return shape, cells


def _compare_shape_vectors(
    rows: list[WorkflowComparisonRow],
    *,
    case_id: str,
    stage: str,
    source_shape: Mapping[str, object],
    source_identities: Sequence[Tuple[str, int]],
    candidate_rows: Sequence[object],
    fields: Mapping[str, Tuple[str, str, float, float]],
) -> None:
    if len(source_identities) != len(candidate_rows):
        rows.append(
            WorkflowComparisonRow(
                case_id,
                stage,
                "shape_measurements",
                None,
                "identity_count",
                None,
                len(source_identities),
                len(candidate_rows),
                None,
                None,
                (0.0, 0.0),
                "object-count",
                False,
                False,
                "shape_measurement_identity_count_mismatch",
            )
        )
    for source_field, (candidate_field, unit, rtol, atol) in fields.items():
        shape, source_values = _flatten_encoded_array(
            source_shape[source_field],
            context={"case_id": case_id, "stage": stage, "field": source_field},
        )
        if len(shape) != 2 or shape[1] != 1:
            raise _validation_error(
                "captured shape measurement is not a column vector",
                "invalid_shape_measurement_shape",
                case_id=case_id,
                stage=stage,
                field=source_field,
                shape=shape,
            )
        if len(source_values) != len(candidate_rows):
            rows.append(
                WorkflowComparisonRow(
                    case_id,
                    stage,
                    "shape_measurements",
                    None,
                    source_field,
                    None,
                    len(source_values),
                    len(candidate_rows),
                    None,
                    None,
                    (0.0, 0.0),
                    "object-count",
                    False,
                    False,
                    "shape_measurement_count_mismatch",
                )
            )
        for index, (source_value, candidate_row) in enumerate(zip(source_values, candidate_rows)):
            identity = source_identities[index] if index < len(source_identities) else None
            comparison = _compare_cell(
                case_id=case_id,
                stage=stage,
                table_name=None,
                row_index=index,
                column_index=0,
                column_name=source_field,
                identity=identity,
                reference=source_value,
                candidate=getattr(candidate_row, candidate_field),
                unit=unit,
                tolerance_override=(rtol, atol),
            )
            rows.append(comparison)


def compare_workflow_reference(
    imported: ImportedWorkflowCapture,
    *,
    python_results: Optional[Mapping[str, WorkflowPythonRunResult]] = None,
) -> WorkflowComparisonReport:
    """Compare the imported MATLAB replay against production Python results."""

    if python_results is None:
        python_results = {run.case_id: run_python_workflow_case(run) for run in imported.catalog.runs}
    expected_ids = {run.case_id for run in imported.catalog.runs}
    if set(python_results) != expected_ids:
        raise _validation_error(
            "Python comparison results do not match the versioned catalog",
            "python_case_identity_mismatch",
            expected=tuple(sorted(expected_ids)),
            actual=tuple(sorted(python_results)),
        )

    rows = []
    unavailable = 0
    capture_images = {image["image_id"]: image for image in imported.capture["image_evidence"]}
    run_specs = {run.case_id: run for run in imported.catalog.runs}
    comparison_order = (
        "direct_measurement",
        "calibration_and_diameter_filter",
        "image_area",
        "histogram_and_na",
        "binned_selection",
        "raw_selection",
        "shape_and_initial_threshold",
        "vesicularity_and_volume_normalization",
        "nv_volume_and_distribution_summary",
        "shape_statistics",
        "table_projection",
    )
    table_names = (
        "NA_mag",
        "Raw_data",
        "Rawcutoffs_data",
        "Calc_out",
        "Area_out",
        "Ves_out",
        "Misc_out",
        "Shape_out",
        "Stats_out",
    )
    for case_id in sorted(expected_ids):
        source_case = imported.cases_by_id[case_id]
        python_case = python_results[case_id]
        run_spec = run_specs[case_id]
        if run_spec.expected_outcome == "expected_domain_failure":
            unavailable += 1
            if python_case.failure_stage != run_spec.expected_stage:
                rows.append(
                    WorkflowComparisonRow(
                        case_id, "expected_failure", None, None, None, None,
                        {"stage": run_spec.expected_source_stage, "identifier": run_spec.expected_source_error},
                        {"stage": python_case.failure_stage, "code": python_case.failure_code},
                        None, None, None, "stage/error-code", False, False, "failure_stage_mismatch",
                    )
                )
            continue
        analysis = python_case.analysis_result
        if analysis is None:
            unavailable += 1
            rows.append(
                WorkflowComparisonRow(
                    case_id, "python_workflow", None, None, None, None,
                    "source success", {"stage": python_case.failure_stage, "code": python_case.failure_code},
                    None, None, None, "stage", False, False, "python_stage_failure",
                )
            )
            continue

        for source_key, python_rows, stage in (
            ("raw_data_identities", analysis.raw_data_rows, "calibration_and_diameter_filter"),
            ("rawcutoffs_identities", analysis.rawcutoffs_data_rows, "raw_selection"),
        ):
            source_identities = source_case[source_key]
            if len(source_identities) != len(python_rows):
                rows.append(
                    WorkflowComparisonRow(
                        case_id, stage, source_key, None, "identity_count", None,
                        len(source_identities), len(python_rows), None, None, (0.0, 0.0),
                        "object-count", False, False, "identity_count_mismatch",
                    )
                )
            for identity_index, (source_identity, python_row) in enumerate(zip(source_identities, python_rows)):
                source_signature = (
                    int(source_identity["source_slot"]),
                    int(source_identity["source_object_index_original_one_based"]),
                    str(source_identity["image_id"]),
                    int(source_identity["component_id"]),
                )
                python_signature = (
                    int(python_row.source_slot),
                    int(python_row.source_object_index) + 1,
                    str(python_row.image_id),
                    int(python_row.component_id),
                )
                matches = source_signature == python_signature
                rows.append(
                    WorkflowComparisonRow(
                        case_id,
                        stage,
                        source_key,
                        identity_index,
                        "source-order identity",
                        (source_signature[2], source_signature[3]),
                        source_signature,
                        python_signature,
                        0.0 if matches else None,
                        0.0 if matches else None,
                        (0.0, 0.0),
                        "slot/object/image/component",
                        matches,
                        matches,
                        None if matches else "object_identity_or_order_mismatch",
                    )
                )

        updated_shape_view = source_case.get("updated_shape") or {}
        source_shape_view = updated_shape_view or source_case["initial_shape"]
        source_shape_stage = "shape_statistics" if updated_shape_view else "shape_and_initial_threshold"
        source_shape_identities = tuple(
            (str(identity["image_id"]), int(identity["component_id"]))
            for identity in source_shape_view["threshold_identities"]
        )
        python_shape_identities = tuple(analysis.shape_statistics_result.threshold_identities)
        shape_identities_match = source_shape_identities == python_shape_identities
        rows.append(
            WorkflowComparisonRow(
                case_id,
                    source_shape_stage,
                "shape_threshold_identities",
                None,
                "source-order identity",
                None,
                source_shape_identities,
                python_shape_identities,
                0.0 if shape_identities_match else None,
                0.0 if shape_identities_match else None,
                (0.0, 0.0),
                "image/component",
                shape_identities_match,
                shape_identities_match,
                None if shape_identities_match else "shape_identity_or_order_mismatch",
            )
        )
        source_raw_identities = tuple(
            (str(identity["image_id"]), int(identity["component_id"]))
            for identity in source_case["rawcutoffs_identities"]
        )
        _compare_shape_vectors(
            rows,
            case_id=case_id,
            stage="shape_and_initial_threshold",
            source_shape=source_case["initial_shape"],
            source_identities=source_raw_identities,
            candidate_rows=analysis.shape_result.rows,
            fields={
                "equivalent_diameter_mm": ("equivalent_diameter_mm", "mm", 1e-10, 1e-12),
                "area_mm2": ("area_mm2", "mm^2", 1e-10, 1e-12),
                "perimeter_mm": ("perimeter_mm", "mm", 1e-10, 1e-12),
                "major_axis_mm": ("major_axis_mm", "mm", 1e-10, 1e-12),
                "minor_axis_mm": ("minor_axis_mm", "mm", 1e-10, 1e-12),
                "orientation_project_deg": ("orientation_project_deg", "degree", 0.0, 1e-8),
                "eccentricity": ("eccentricity", "dimensionless", 1e-10, 1e-12),
                "regularity": ("regularity", "dimensionless", 1e-10, 1e-12),
                "elongation": ("elongation", "dimensionless", 1e-10, 1e-12),
                "aspect_ratio": ("aspect_ratio", "dimensionless", 1e-10, 1e-12),
                "fitted_ellipse_perimeter_mm": ("fitted_ellipse_perimeter_mm", "mm", 1e-10, 1e-12),
            },
        )
        if updated_shape_view:
            candidate_shape_by_identity = {
                (row.image_id, row.component_id): row
                for row in analysis.shape_result.rows
            }
            updated_candidates = tuple(
                candidate_shape_by_identity[identity]
                for identity in python_shape_identities
                if identity in candidate_shape_by_identity
            )
            _compare_shape_vectors(
                rows,
                case_id=case_id,
                stage="shape_statistics",
                source_shape=updated_shape_view,
                source_identities=source_shape_identities,
                candidate_rows=updated_candidates,
                fields={
                    "equivalent_diameter_mm": ("equivalent_diameter_mm", "mm", 1e-10, 1e-12),
                    "area_mm2": ("area_mm2", "mm^2", 1e-10, 1e-12),
                    "eccentricity": ("eccentricity", "dimensionless", 1e-10, 1e-12),
                    "orientation_project_deg": ("orientation_project_deg", "degree", 0.0, 1e-8),
                    "regularity": ("regularity", "dimensionless", 1e-10, 1e-12),
                    "elongation": ("elongation", "dimensionless", 1e-10, 1e-12),
                    "aspect_ratio": ("aspect_ratio", "dimensionless", 1e-10, 1e-12),
                },
            )

        python_images = {
            image.image_input.image_id: (group, image)
            for group in analysis.prepared_result.groups
            for image in group.images
        }
        source_image_ids = tuple(source_case["direct_image_ids"])
        for image_id in source_image_ids:
            source_image = capture_images.get(image_id)
            python_pair = python_images.get(image_id)
            if source_image is None or python_pair is None:
                rows.append(
                    WorkflowComparisonRow(
                        case_id, "direct_measurement", None, None, None, (image_id, 0),
                        "present", "missing", None, None, (0.0, 0.0), "component-identity", False, False, "missing_image_identity",
                    )
                )
                continue
            group, prepared_image = python_pair
            measurement = prepared_image.measurement_result
            selected_mask = np.equal(
                prepared_image.image_input.segmented_image,
                measurement.pore_phase_value,
            )
            python_masks = {
                "pore_mask": selected_mask,
                "border_removed_mask": selected_mask & ~measurement.pore_result.removed_border_mask,
                "border_mask": measurement.pore_result.removed_border_mask,
                "filled_mask": measurement.pore_result.filled_mask,
                "labels": measurement.pore_result.label_image,
            }
            for field, candidate_array in python_masks.items():
                source_shape, source_cells = _flatten_encoded_array(source_image[field], context={"case_id": case_id, "image_id": image_id, "field": field})
                source_array = np.asarray([cell.value for cell in source_cells]).reshape(source_shape, order="F")
                equal = source_array.shape == candidate_array.shape and np.array_equal(source_array, candidate_array)
                rows.append(
                    WorkflowComparisonRow(
                        case_id,
                        "direct_measurement",
                        None,
                        None,
                        field,
                        (image_id, 0),
                        {"shape": source_array.shape, "sha256": hashlib.sha256(source_array.tobytes(order="C")).hexdigest()},
                        {"shape": candidate_array.shape, "sha256": hashlib.sha256(np.ascontiguousarray(candidate_array).tobytes(order="C")).hexdigest()},
                        0.0 if equal else None,
                        0.0 if equal else None,
                        (0.0, 0.0),
                        "pixels",
                        equal,
                        equal,
                        None if equal else "mask_or_label_mismatch",
                    )
                )
            source_components = source_image["components"]
            python_components = measurement.pore_result.components
            python_moments = {component.component_id: component for component in prepared_image.moment_geometry_result.components}
            python_perimeters = {component.component_id: component for component in prepared_image.perimeter_result.components}
            source_vectors = {}
            for field in (
                "area_px",
                "major_axis_px",
                "minor_axis_px",
                "eccentricity",
                "orientation_regionprops_deg",
                "perimeter_regionprops_px",
            ):
                vector_shape, vector_cells = _flatten_encoded_array(
                    source_image["measurements"][field],
                    context={"case_id": case_id, "image_id": image_id, "field": field},
                )
                source_vectors[field] = vector_cells
                if len(vector_shape) != 2 or vector_shape[1] != 1:
                    raise _validation_error(
                        "direct source measurement is not a column vector",
                        "invalid_direct_measurement_shape",
                        case_id=case_id,
                        image_id=image_id,
                        field=field,
                        shape=vector_shape,
                    )
            if len(source_components) != len(python_components):
                rows.append(
                    WorkflowComparisonRow(
                        case_id, "direct_measurement", None, None, "component_count", (image_id, 0),
                        len(source_components), len(python_components), None, None, (0.0, 0.0), "count", False, False, "component_count_mismatch",
                    )
                )
            for component_index, (source_component, python_component) in enumerate(zip(source_components, python_components)):
                identity = (image_id, int(source_component["component_id"]))
                source_indices = source_component["pixel_linear_indices_column_major_one_based"]
                _, source_index_cells = _flatten_encoded_array(source_indices, context={"case_id": case_id, "identity": identity, "field": "membership"})
                source_membership = tuple(int(cell.value) for cell in source_index_cells)
                python_membership = tuple(
                    int(index + 1)
                    for index in np.flatnonzero(measurement.pore_result.label_image.ravel(order="F") == python_component.component_id)
                )
                match = source_membership == python_membership
                rows.append(
                    WorkflowComparisonRow(
                        case_id, "direct_measurement", None, component_index, "pixel_membership", identity,
                        source_membership, python_membership, 0.0 if match else None, 0.0 if match else None,
                        (0.0, 0.0), "column-major-linear-index", match, match, None if match else "component_identity_mismatch",
                    )
                )
                moment = python_moments.get(python_component.component_id)
                perimeter = python_perimeters.get(python_component.component_id)
                direct_candidates = (
                    ("area_px", python_component.area_px, "pixel-area", 0.0, 0.0),
                    ("major_axis_px", None if moment is None else moment.major_axis_px, "pixel-length", 1e-10, 1e-10),
                    ("minor_axis_px", None if moment is None else moment.minor_axis_px, "pixel-length", 1e-10, 1e-10),
                    ("eccentricity", None if moment is None else moment.eccentricity, "dimensionless", 1e-10, 1e-12),
                    (
                        "orientation_regionprops_deg",
                        None if moment is None else moment.orientation_regionprops_profile_deg,
                        "degree",
                        0.0,
                        1e-8,
                    ),
                    ("perimeter_regionprops_px", None if perimeter is None else perimeter.perimeter_px, "pixel-length", 1e-10, 1e-10),
                )
                for source_field, candidate, unit, rtol, atol in direct_candidates:
                    vector = source_vectors[source_field]
                    if len(vector) != len(source_components):
                        raise _validation_error(
                            "direct source measurement count differs from component identity count",
                            "direct_measurement_count_mismatch",
                            case_id=case_id,
                            image_id=image_id,
                            field=source_field,
                            measurement_count=len(vector),
                            component_count=len(source_components),
                        )
                    source_classified = vector[component_index]
                    candidate_classified = _python_number(candidate)
                    comparison = _compare_cell(
                        case_id=case_id,
                        stage="direct_measurement",
                        table_name=None,
                        row_index=component_index,
                        column_index=0,
                        column_name=source_field,
                        identity=identity,
                        reference=source_classified,
                        candidate=candidate_classified,
                        unit=unit,
                        tolerance_override=(rtol, atol),
                    )
                    rows.append(comparison)

        projections = build_foams105_export_tables(analysis)
        python_tables = {table.filename.removesuffix(".txt"): table for table in projections.tables}
        source_tables = source_case["tables"]
        if tuple(source_tables) != table_names:
            rows.append(
                WorkflowComparisonRow(
                    case_id, "table_projection", None, None, None, None,
                    table_names, tuple(source_tables), None, None, (0.0, 0.0), "table-order", False, False, "table_set_mismatch",
                )
            )
        for table_name in table_names:
            source_table = source_tables[table_name]
            python_projection = python_tables.get(table_name)
            if python_projection is None:
                unavailable += 1
                rows.append(
                    WorkflowComparisonRow(
                        case_id, TABLE_STAGE[table_name], table_name, None, None, None,
                        "available source table", "missing Python projection", None, None, None,
                        "availability", False, False, "python_table_unavailable",
                    )
                )
                continue
            source_headers, source_matrix = _reference_matrix(source_table, table_name)
            python_headers = tuple(python_projection.headers)
            python_matrix = _matrix_rows_for_python(table_name + ".txt", python_projection)
            if source_headers != python_headers:
                rows.append(
                    WorkflowComparisonRow(
                        case_id, TABLE_STAGE[table_name], table_name, None, None, None,
                        source_headers, python_headers, None, None, (0.0, 0.0), "header", False, False, "header_mismatch",
                    )
                )
            if len(source_matrix) != len(python_matrix):
                rows.append(
                    WorkflowComparisonRow(
                        case_id, TABLE_STAGE[table_name], table_name, None, None, None,
                        len(source_matrix), len(python_matrix), None, None, (0.0, 0.0), "row-count", False, False, "table_row_count_mismatch",
                    )
                )
            for row_index, (source_row, python_row) in enumerate(zip(source_matrix, python_matrix)):
                if len(source_row) != len(python_row):
                    rows.append(
                        WorkflowComparisonRow(
                            case_id, TABLE_STAGE[table_name], table_name, row_index, None, None,
                            len(source_row), len(python_row), None, None, (0.0, 0.0), "column-count", False, False, "table_column_count_mismatch",
                        )
                    )
                    continue
                for column_index, (source_value, python_value) in enumerate(zip(source_row, python_row)):
                    column_name = python_headers[column_index] if column_index < len(python_headers) else str(column_index)
                    row_identity = None
                    if table_name in ("Raw_data", "Rawcutoffs_data", "Shape_out"):
                        identities = source_case["raw_data_identities"] if table_name == "Raw_data" else source_case["rawcutoffs_identities"]
                        if table_name == "Shape_out":
                            identities = ()
                        if row_index < len(identities):
                            identity_row = identities[row_index]
                            row_identity = (str(identity_row["image_id"]), int(identity_row["component_id"]))
                    comparison = _compare_cell(
                        case_id=case_id,
                        stage=TABLE_STAGE[table_name],
                        table_name=table_name + ".txt",
                        row_index=row_index,
                        column_index=column_index,
                        column_name=column_name,
                        identity=row_identity,
                        reference=source_value,
                        candidate=python_value,
                        unit="source-unit",
                    )
                    rows.append(comparison)

    stage_rank = {stage: index for index, stage in enumerate(comparison_order)}
    mismatches = [row for row in rows if not row.within_tolerance or not row.classification_match]
    first_stage = min((row.stage for row in mismatches), key=lambda stage: stage_rank.get(stage, len(stage_rank)), default=None)
    affected = set()
    if first_stage:
        affected.update(TABLE_DOWNSTREAM.get(first_stage, ()))
    for mismatch in mismatches:
        if mismatch.table:
            affected.add(mismatch.table.removesuffix(".txt"))
    return WorkflowComparisonReport(
        source_commit=ORIGINAL_SOURCE_COMMIT,
        matlab_release=str(imported.capture["matlab_release"]),
        python_revision=str(imported.catalog.runs[0].source_metadata.get("pyro_revision", "unknown")),
        comparison_count=len(rows),
        failure_count=len(mismatches),
        unavailable_stage_count=unavailable,
        first_divergent_stage=first_stage,
        affected_tables=tuple(table for table in table_names if table in affected),
        known_extensions=KNOWN_EXTENSIONS,
        rows=tuple(rows),
    )


def _json_safe(value: object) -> object:
    if isinstance(value, complex):
        return {"real": value.real, "imaginary": value.imag}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def workflow_comparison_json(report: WorkflowComparisonReport) -> str:
    """Return a deterministic, JSON-safe representation of a comparison report."""

    return json.dumps(_json_safe(asdict(report)), indent=2, sort_keys=True, allow_nan=False) + "\n"


def workflow_comparison_markdown(report: WorkflowComparisonReport) -> str:
    """Render the first divergence and every mismatch as a concise review report."""

    failures = tuple(row for row in report.rows if not row.within_tolerance or not row.classification_match)

    def cell(value: object) -> str:
        text = json.dumps(_json_safe(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "# FOAMS 1.0.5 Workflow Comparison",
        "",
        f"- Source commit: `{report.source_commit}`",
        f"- MATLAB: `{report.matlab_release}`",
        f"- Python revision: `{report.python_revision}`",
        f"- Comparisons: {report.comparison_count}",
        f"- Mismatches: {report.failure_count}",
        f"- Unavailable stages: {report.unavailable_stage_count}",
        f"- First divergent stage: `{report.first_divergent_stage or 'none'}`",
        f"- Affected tables: {', '.join(report.affected_tables) or 'none'}",
        "",
        "## Mismatches",
        "",
        "| Case | Stage | Table | Row | Column | Identity | MATLAB | Python | Tolerance | Unit | Code |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in failures:
        lines.append(
            "| "
            + " | ".join(
                (
                    cell(row.case_id),
                    cell(row.stage),
                    cell(row.table),
                    cell(row.row_index),
                    cell(row.column),
                    cell(row.identity),
                    cell(row.reference_value),
                    cell(row.python_value),
                    cell(row.tolerance),
                    cell(row.unit),
                    cell(row.code),
                )
            )
            + " |"
        )
    lines.extend(("", "## Known Extensions", ""))
    lines.extend(f"- {extension}" for extension in report.known_extensions)
    return "\n".join(lines) + "\n"
