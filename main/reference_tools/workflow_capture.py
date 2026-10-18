"""External MATLAB orchestration for the pinned FOAMS workflow replay."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Union

from .workflow_reference import (
    DEFAULT_EXTRACTION_MAP,
    WorkflowCaseCatalog,
    WorkflowReferenceValidationError,
    load_workflow_case_catalog,
    load_workflow_extraction_map,
    prepare_workflow_reference,
    verify_workflow_source_checkout,
)


@dataclass(frozen=True)
class WorkflowCaptureResult:
    output_directory: Path
    process_exit_code: int
    case_count: int
    complete: bool
    capture_sha256: Optional[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_flushed(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _matlab_string(value: Union[Path, str]) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _verify_captured_cases(capture: Mapping[str, object], catalog: WorkflowCaseCatalog) -> None:
    cases = capture.get("cases")
    if not isinstance(cases, list):
        raise WorkflowReferenceValidationError("capture cases are missing", code="invalid_capture_cases")
    actual = {item.get("case_id"): item for item in cases if isinstance(item, dict)}
    if len(actual) != len(cases) or set(actual) != {run.case_id for run in catalog.runs}:
        raise WorkflowReferenceValidationError(
            "capture case identities differ from the pinned catalog",
            code="capture_case_identity_mismatch",
            context={"expected": tuple(run.case_id for run in catalog.runs), "actual": tuple(actual)},
        )
    image_evidence = capture.get("image_evidence")
    if not isinstance(image_evidence, list):
        raise WorkflowReferenceValidationError("deduplicated direct image evidence is missing", code="missing_image_evidence")
    image_ids = [item.get("image_id") for item in image_evidence if isinstance(item, dict)]
    if len(image_ids) != len(image_evidence) or len(image_ids) != len(set(image_ids)):
        raise WorkflowReferenceValidationError("direct image evidence identities are malformed or duplicated", code="invalid_image_evidence")
    for run in catalog.runs:
        case = actual[run.case_id]
        if run.expected_outcome == "success":
            required = {"direct_image_ids", "project_measurements", "initial_shape", "initial_statistics", "threshold_settings", "tables", "volume", "bulk_na"}
            if case.get("outcome") != "success" or not required.issubset(case):
                raise WorkflowReferenceValidationError(
                    "successful catalog case lacks one or more required stage outputs",
                    code="incomplete_success_case",
                    context={"case_id": run.case_id, "outcome": case.get("outcome"), "missing": tuple(sorted(required - set(case)))},
                )
            scene_ids = tuple(image.image_id for group in run.scene.groups for image in group.images)
            if set(case["direct_image_ids"]) != set(scene_ids) or not set(scene_ids).issubset(image_ids):
                raise WorkflowReferenceValidationError(
                    "successful case direct-image references do not match its input scene",
                    code="missing_image_identity",
                    context={"case_id": run.case_id, "expected": scene_ids, "actual": tuple(case["direct_image_ids"])},
                )
            table_names = {"NA_mag", "Raw_data", "Rawcutoffs_data", "Calc_out", "Area_out", "Ves_out", "Misc_out", "Shape_out", "Stats_out"}
            if set(case["tables"]) != table_names:
                raise WorkflowReferenceValidationError(
                    "successful catalog case lacks exactly nine source table arrays",
                    code="capture_table_set_mismatch",
                    context={"case_id": run.case_id, "tables": tuple(case["tables"])},
                )
            log_inputs = case["volume"].get("log_inputs")
            if not isinstance(log_inputs, dict) or (
                log_inputs.get("log10_expression") != "Log_Nvl=log10(NV_cumsum);"
                or log_inputs.get("natural_log_expression") != "ln_n=log(n_numb);"
            ):
                raise WorkflowReferenceValidationError(
                    "source logarithm expressions or operands are missing",
                    code="missing_log_operand_evidence",
                    context={"case_id": run.case_id},
                )
            thresholds = case["threshold_settings"]
            if not isinstance(thresholds, dict):
                raise WorkflowReferenceValidationError(
                    "captured threshold settings are malformed",
                    code="capture_threshold_settings_mismatch",
                    context={"case_id": run.case_id},
                )
            maximum_scale = max(group.scale_px_per_mm for group in run.scene.groups)
            project_minimum = float(run.preparation["minimum_diameter_px"])
            statistics_minimum = float(run.analysis["shape_minimum_diameter_px"])
            expected_operator = ">" if run.analysis["shape_statistics_mode"] == "initial_analysis" else ">="
            expected_thresholds = {
                "project_minimum_diameter_px": project_minimum,
                "statistics_minimum_diameter_px": statistics_minimum,
                "maximum_scale_px_per_mm": maximum_scale,
                "project_threshold_mm": project_minimum / maximum_scale,
                "statistics_threshold_mm": statistics_minimum / maximum_scale,
                "statistics_operator": expected_operator,
            }
            if any(thresholds.get(key) != value for key, value in expected_thresholds.items()):
                raise WorkflowReferenceValidationError(
                    "captured MATLAB threshold settings differ from the versioned Python run settings",
                    code="capture_threshold_settings_mismatch",
                    context={"case_id": run.case_id, "expected": expected_thresholds, "actual": thresholds},
                )
        else:
            failure = case.get("failure")
            if (
                case.get("outcome") != "expected_source_domain_failure"
                or not isinstance(failure, dict)
                or failure.get("stage") != run.expected_source_stage
                or failure.get("identifier") != run.expected_source_error
            ):
                raise WorkflowReferenceValidationError(
                    "expected source-domain failure differs from its catalog contract",
                    code="unexpected_source_failure",
                    context={"case_id": run.case_id, "failure": failure, "expected_stage": run.expected_source_stage, "expected_identifier": run.expected_source_error},
                )


def run_foams105_workflow_capture(
    source_root: Union[Path, str],
    case_catalog_path: Union[Path, str],
    output_directory: Union[Path, str],
    *,
    matlab_executable: Union[Path, str],
    repository_root: Union[Path, str],
    extraction_map_path: Union[Path, str] = DEFAULT_EXTRACTION_MAP,
    runner_path: Union[Path, str] = "reference-tools/matlab/foams105_workflow_capture.m",
    timeout_seconds: int = 1800,
) -> WorkflowCaptureResult:
    """Run MATLAB, retain logs/status, and write `completed.marker` last."""

    output = Path(output_directory).resolve()
    source = Path(source_root).resolve()
    repository = Path(repository_root).resolve()
    executable = Path(matlab_executable).resolve()
    catalog_file = Path(case_catalog_path)
    if not catalog_file.is_absolute():
        catalog_file = repository / catalog_file
    map_file = Path(extraction_map_path)
    if not map_file.is_absolute():
        map_file = repository / map_file
    runner_file = Path(runner_path)
    if not runner_file.is_absolute():
        runner_file = repository / runner_file
    runner_file = runner_file.resolve()
    if output.exists():
        raise WorkflowReferenceValidationError(
            "capture output directory already exists",
            code="output_directory_exists",
            context={"output_directory": str(output)},
        )
    if not executable.is_file():
        raise WorkflowReferenceValidationError(
            "MATLAB executable is unavailable",
            code="matlab_executable_missing",
            context={"matlab_executable": str(executable)},
        )
    extraction_map = load_workflow_extraction_map(map_file)
    verify_workflow_source_checkout(source, extraction_map, repository_root=repository)
    catalog = load_workflow_case_catalog(catalog_file)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="foams105-workflow-prep-") as temporary:
        preparation_directory = Path(temporary) / "preparation"
        prepared = prepare_workflow_reference(
            catalog_file,
            preparation_directory,
            extraction_map_path=map_file,
            runner_path=runner_file,
        )
        matlab_command = (
            f"cd({_matlab_string(repository)}); "
            f"addpath({_matlab_string(runner_file.parent)}); "
            "foams105_workflow_capture("
            f"{_matlab_string(source)},{_matlab_string(prepared.constructor_path)},"
            f"{_matlab_string(map_file)},{_matlab_string(output)},{_matlab_string(repository)});"
        )
        started = time.perf_counter()
        try:
            process = subprocess.run(
                [str(executable), "-batch", matlab_command],
                cwd=repository,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
            exit_code = int(process.returncode)
            stdout = process.stdout
            stderr = process.stderr
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            stderr += f"\nMATLAB process timed out after {timeout_seconds} seconds.\n"
            timed_out = True
        elapsed_seconds = time.perf_counter() - started

        if not output.exists():
            output.mkdir(parents=True)
        for source_path, target_name in (
            (catalog_file, "input_catalog.json"),
            (map_file, "extraction_map.json"),
            (runner_file, "foams105_workflow_capture.m"),
            (prepared.constructor_path, "foams105_workflow_cases_generated.m"),
            (prepared.manifest_path, "preparation_manifest.json"),
        ):
            shutil.copyfile(source_path, output / target_name)

        _write_flushed(output / "matlab.stdout.log", stdout.encode("utf-8"))
        _write_flushed(output / "matlab.stderr.log", stderr.encode("utf-8"))
        _write_flushed(output / "process_exit_code.txt", f"{exit_code}\n".encode("ascii"))
        run_metadata = {
            "schema_version": 1,
            "source_commit": extraction_map["source_commit"],
            "matlab_executable": str(executable),
            "matlab_batch_command": matlab_command,
            "process_exit_code": exit_code,
            "timed_out": timed_out,
            "elapsed_seconds": elapsed_seconds,
            "matlab_stdout_sha256": _sha256(output / "matlab.stdout.log"),
            "matlab_stderr_sha256": _sha256(output / "matlab.stderr.log"),
        }
        run_metadata_path = output / "run_metadata.json"
        _write_flushed(run_metadata_path, (json.dumps(run_metadata, sort_keys=True, indent=2) + "\n").encode("utf-8"))

        complete = False
        capture_path = output / "capture.json"
        if exit_code == 0 and capture_path.is_file() and (output / "manifest.json").is_file():
            try:
                capture = json.loads(capture_path.read_text(encoding="utf-8"))
                _verify_captured_cases(capture, catalog)
                complete = True
            except (OSError, UnicodeError, json.JSONDecodeError, WorkflowReferenceValidationError):
                complete = False

        inventory = tuple(
            sorted(path.name for path in output.iterdir() if path.is_file() and path.name != "checksums.json")
        )
        checksum_map = {name: _sha256(output / name) for name in inventory}
        _write_flushed(
            output / "checksums.json",
            (json.dumps({"files": checksum_map}, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
        if complete:
            _write_flushed(
                output / "completed.marker",
                (f"COMPLETE\nsource_commit={extraction_map['source_commit']}\nprocess_exit_code=0\n").encode("ascii"),
            )
        return WorkflowCaptureResult(
            output,
            exit_code,
            len(catalog.runs),
            complete,
            checksum_map.get("capture.json"),
        )
