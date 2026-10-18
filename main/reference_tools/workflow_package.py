"""Build and import compact, checksummed real-MATLAB workflow references."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Mapping, Sequence, Union

from .workflow_capture import _verify_captured_cases
from .workflow_compare import (
    ImportedWorkflowCapture,
    compare_workflow_reference,
    import_workflow_reference,
    workflow_comparison_json,
)
from .workflow_reference import (
    WorkflowReferenceValidationError,
    load_workflow_case_catalog,
)

DEFAULT_REFERENCE_CASES = (
    "single_manual_unnormalized_initial",
    "two_scale_manual_known",
    "shape_threshold_equality_initial_strict",
    "shape_threshold_equality_updated_inclusive",
    "empty_autosmart_expected_domain_failure",
)
REFERENCE_COMPRESSION_METHOD = "gzip.GzipFile-filename-empty-mtime-zero-level9-canonical-json-v1"
LEGACY_COMPRESSION_METHOD = "gzip-mtime-zero-canonical-json-v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _gzip_canonical_payload(payload: bytes) -> bytes:
    stream = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=stream,
        compresslevel=9,
        mtime=0,
    ) as compressor:
        compressor.write(payload)
    return stream.getvalue()


def repack_workflow_reference_package_transport_only(
    package_path: Union[Path, str],
    output_path: Union[Path, str],
) -> Mapping[str, object]:
    """Recompress verified package payload bytes without changing scientific/provenance JSON."""

    source = Path(package_path).resolve()
    source_manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    destination = Path(output_path).resolve()
    destination_manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists() or destination_manifest_path.exists():
        raise _error("transport-repack output already exists", "output_directory_exists", path=str(destination))
    try:
        compressed = source.read_bytes()
        manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if _sha256_bytes(compressed) != manifest.get("compressed_sha256"):
            raise _error("source package compressed hash mismatch", "reference_package_checksum_mismatch")
        payload = gzip.decompress(compressed)
        if _sha256_bytes(payload) != manifest.get("payload_sha256"):
            raise _error("source package payload hash mismatch", "reference_package_payload_checksum_mismatch")
    except WorkflowReferenceValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, gzip.BadGzipFile, EOFError) as exc:
        raise _error("source package is unreadable or malformed", "invalid_reference_package", cause=str(exc)) from exc

    new_compressed = _gzip_canonical_payload(payload)
    updated_manifest = dict(manifest)
    updated_manifest["filename"] = destination.name
    updated_manifest["compression"] = REFERENCE_COMPRESSION_METHOD
    updated_manifest["compressed_sha256"] = _sha256_bytes(new_compressed)
    updated_manifest["payload_sha256"] = _sha256_bytes(payload)
    if not isinstance(manifest.get("transport_change"), dict):
        if manifest.get("compression") != REFERENCE_COMPRESSION_METHOD or manifest.get("compressed_sha256") != _sha256_bytes(new_compressed):
            updated_manifest["transport_change"] = {
                "kind": "gzip_representation_only",
                "from": manifest.get("compression", "unknown"),
                "to": REFERENCE_COMPRESSION_METHOD,
                "payload_sha256_preserved": True,
                "parent_capture_sha256_preserved": manifest.get("parent_capture_sha256"),
            }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(new_compressed)
    destination_manifest_path.write_text(
        json.dumps(updated_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return updated_manifest


def _error(message: str, code: str, **context: object) -> WorkflowReferenceValidationError:
    return WorkflowReferenceValidationError(message, code=code, context=context)


def build_workflow_reference_package(
    capture_directory: Union[Path, str],
    output_path: Union[Path, str],
    *,
    case_ids: Sequence[str] = DEFAULT_REFERENCE_CASES,
    repository_root: Union[Path, str],
) -> Mapping[str, object]:
    """Project complete selected cases and all their evidence into deterministic gzip JSON."""

    imported = import_workflow_reference(capture_directory, repository_root=repository_root)
    selected_ids = tuple(case_ids)
    if len(selected_ids) != len(set(selected_ids)) or any(case_id not in imported.cases_by_id for case_id in selected_ids):
        raise _error("reference package case selection is invalid", "invalid_reference_selection", case_ids=selected_ids)
    selected_cases = [imported.cases_by_id[case_id] for case_id in selected_ids]
    if not selected_cases:
        raise _error("reference package must contain at least one case", "empty_reference_selection")
    all_direct_image_ids = {
        image_id
        for case in selected_cases
        for image_id in case.get("direct_image_ids", ())
    }
    selected_images = [
        image for image in imported.capture["image_evidence"]
        if image["image_id"] in all_direct_image_ids
    ]
    if {image["image_id"] for image in selected_images} != all_direct_image_ids:
        raise _error("selected cases reference missing deduplicated image evidence", "reference_image_evidence_missing")

    source_catalog_path = imported.directory / "input_catalog.json"
    catalog_payload = json.loads(source_catalog_path.read_text(encoding="utf-8"))
    selected_run_specs = [run for run in catalog_payload["runs"] if run["case_id"] in selected_ids]
    selected_scene_ids = {run["scene_id"] for run in selected_run_specs}
    selected_scenes = [scene for scene in catalog_payload["scenes"] if scene["scene_id"] in selected_scene_ids]
    selected_catalog_image_ids = {
        image_id
        for scene in selected_scenes
        for group in scene["groups"]
        for image_id in group["image_ids"]
    }
    selected_catalog = dict(catalog_payload)
    selected_catalog["images"] = [image for image in catalog_payload["images"] if image["image_id"] in selected_catalog_image_ids]
    selected_catalog["scenes"] = selected_scenes
    selected_catalog["runs"] = selected_run_specs
    selected_catalog_bytes = _canonical_json(selected_catalog)
    selected_catalog_sha256 = _sha256_bytes(selected_catalog_bytes)

    capture_projection = dict(imported.capture)
    capture_projection["catalog_raw_sha256"] = selected_catalog_sha256
    capture_projection["cases"] = selected_cases
    capture_projection["image_evidence"] = selected_images
    comparison_import = ImportedWorkflowCapture(
        imported.directory,
        capture_projection,
        imported.manifest,
        imported.catalog,
        {case["case_id"]: case for case in selected_cases},
    )
    strict_report = compare_workflow_reference(comparison_import)
    policy_report = compare_workflow_reference(comparison_import, enable_real_log_extension=True)

    source_manifest = json.loads((imported.directory / "manifest.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((imported.directory / "run_metadata.json").read_text(encoding="utf-8"))
    package_payload = {
        "schema_version": 1,
        "package_kind": "foams105_real_matlab_workflow_projection",
        "parent_capture_sha256": source_manifest["capture_sha256"],
        "projection_selection": {
            "case_ids": selected_ids,
            "rules": (
                "Retain every field, vector, identity, table cell, and mismatch from each selected case.",
                "Retain all deduplicated image evidence referenced by the selected cases.",
                "Do not filter numeric rows or cells based on comparison outcome.",
            ),
        },
        "selected_catalog": selected_catalog,
        "selected_catalog_sha256": selected_catalog_sha256,
        "capture": capture_projection,
        "source_manifest": source_manifest,
        "extraction_map": json.loads((imported.directory / "extraction_map.json").read_text(encoding="utf-8")),
        "completion": {
            "status": "COMPLETE",
            "process_exit_code": run_metadata.get("process_exit_code"),
            "timed_out": run_metadata.get("timed_out"),
            "matlab_release": imported.capture["matlab_release"],
            "matlab_version": imported.capture["matlab_version"],
            "image_processing_toolbox_version": imported.capture["image_processing_toolbox_version"],
        },
        "comparison_reports": {
            "strict_raw": json.loads(workflow_comparison_json(strict_report)),
            "real_log_extension_policy": json.loads(workflow_comparison_json(policy_report)),
        },
    }
    raw_payload = _canonical_json(package_payload)
    compressed_payload = _gzip_canonical_payload(raw_payload)
    destination = Path(output_path).resolve()
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists() or manifest_path.exists():
        raise _error("reference package output already exists", "output_directory_exists", path=str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(compressed_payload)
    manifest = {
        "schema_version": 1,
        "filename": destination.name,
        "compression": REFERENCE_COMPRESSION_METHOD,
        "compressed_sha256": _sha256_bytes(compressed_payload),
        "payload_sha256": _sha256_bytes(raw_payload),
        "parent_capture_sha256": source_manifest["capture_sha256"],
        "selected_catalog_sha256": selected_catalog_sha256,
        "case_ids": selected_ids,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def import_workflow_reference_package(
    package_path: Union[Path, str],
    *,
    repository_root: Union[Path, str],
) -> ImportedWorkflowCapture:
    """Verify deterministic gzip/package hashes and load saved MATLAB evidence without MATLAB."""

    path = Path(package_path).resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    try:
        compressed = path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _sha256_bytes(compressed) != manifest.get("compressed_sha256"):
            raise _error("compressed reference package hash mismatch", "reference_package_checksum_mismatch")
        raw_payload = gzip.decompress(compressed)
        if _sha256_bytes(raw_payload) != manifest.get("payload_sha256"):
            raise _error("decompressed reference package hash mismatch", "reference_package_payload_checksum_mismatch")
        payload = json.loads(raw_payload.decode("utf-8"))
    except WorkflowReferenceValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, gzip.BadGzipFile, EOFError) as exc:
        raise _error("reference package is unreadable or malformed", "invalid_reference_package", cause=str(exc)) from exc
    if payload.get("schema_version") != 1 or payload.get("package_kind") != "foams105_real_matlab_workflow_projection":
        raise _error("reference package schema is unsupported", "invalid_reference_package_schema")
    if payload.get("parent_capture_sha256") != manifest.get("parent_capture_sha256"):
        raise _error("reference package parent capture hash is inconsistent", "reference_package_parent_mismatch")
    if payload.get("selected_catalog_sha256") != manifest.get("selected_catalog_sha256"):
        raise _error("reference package catalog hash is inconsistent", "reference_package_catalog_mismatch")
    if manifest.get("compression") not in (REFERENCE_COMPRESSION_METHOD, LEGACY_COMPRESSION_METHOD):
        raise _error("reference package compression policy is unsupported", "reference_package_compression_mismatch")
    if payload.get("completion", {}).get("status") != "COMPLETE" or payload.get("completion", {}).get("process_exit_code") != 0:
        raise _error("reference package does not record successful MATLAB completion", "incomplete_reference_package")
    if payload.get("capture", {}).get("source_commit") != payload.get("source_manifest", {}).get("source_commit"):
        raise _error("reference package source commit metadata differs", "reference_package_source_mismatch")

    with tempfile.TemporaryDirectory(prefix="foams105-reference-package-") as temporary:
        catalog_path = Path(temporary) / "input_catalog.json"
        catalog_bytes = _canonical_json(payload["selected_catalog"])
        if _sha256_bytes(catalog_bytes) != payload["selected_catalog_sha256"]:
            raise _error("selected catalog contents do not match their hash", "reference_package_catalog_mismatch")
        catalog_path.write_bytes(catalog_bytes)
        catalog = load_workflow_case_catalog(catalog_path)
    capture = payload["capture"]
    cases = {case["case_id"]: case for case in capture.get("cases", ())}
    expected = {run.case_id for run in catalog.runs}
    if set(cases) != expected or tuple(manifest.get("case_ids", ())) != tuple(run.case_id for run in catalog.runs):
        raise _error("selected case records do not match the package catalog", "reference_package_case_mismatch")
    if capture.get("source_commit") != catalog.source_commit or capture.get("catalog_raw_sha256") != catalog.raw_sha256:
        raise _error("projected capture source/catalog identity differs", "reference_package_provenance_mismatch")
    _verify_captured_cases(capture, catalog)
    return ImportedWorkflowCapture(path.parent, capture, payload["source_manifest"], catalog, cases)
