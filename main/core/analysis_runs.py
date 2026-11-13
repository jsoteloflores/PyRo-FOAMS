from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .legacy_exports import write_foams105_export_bundle
from .sample_project import checksum_ndarray

RUN_SCHEMA_VERSION = 2


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def structured(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else {"value": None, "availability": "unavailable", "reason": "nonfinite scientific value"}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return {"availability": "not_projected", "reason": "scientific working array; captured inputs stored as assets"}
    if is_dataclass(value):
        return {field.name: structured(getattr(value, field.name)) for field in fields(value) if not field.name.startswith("_")}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: structured(item) for key, item in value.items()}
        return [{"key": structured(key), "value": structured(item)} for key, item in value.items()]
    if isinstance(value, (list, tuple)):
        return [structured(item) for item in value]
    raise ValueError(f"Unsupported run projection value: {type(value).__name__}")


def error_diagnostic(method, error, stage=None):
    return {"method_id": method, "stage": stage or getattr(error, "stage", "calculation"), "severity": "error",
            "code": getattr(error, "cause_code", None) or getattr(error, "code", type(error).__name__),
            "message": str(error), "context": structured(getattr(error, "context", {}))}


@dataclass(frozen=True)
class SavedAnalysisRun:
    run_id: str
    project_id: str
    payload: bytes
    completed_at_utc: str = field(init=False)
    preset: str = field(init=False)
    outcomes: tuple = field(init=False)
    schema: int = field(init=False)

    def __post_init__(self):
        data = self.data()
        object.__setattr__(self, "schema", data["schema"])
        object.__setattr__(self, "completed_at_utc", data["completed_at_utc"])
        object.__setattr__(self, "preset", data["preset"])
        object.__setattr__(self, "outcomes", tuple(method["outcome"] for method in data["methods"]))

    def data(self):
        return json.loads(self.payload)

    def index_entry(self):
        return {"run_id": self.run_id, "project_id": self.project_id, "schema": self.schema,
                "path": f"runs/{self.run_id}-{digest(self.payload)}.json", "sha256": digest(self.payload),
            "completed_at_utc": self.completed_at_utc, "preset": self.preset}


@dataclass(frozen=True)
class RunProjection:
    record: SavedAnalysisRun
    assets: tuple


def validate_run(record):
    if not isinstance(record, SavedAnalysisRun):
        raise ValueError("Run must be a SavedAnalysisRun")
    data = record.data()
    if data.get("schema") not in (1, RUN_SCHEMA_VERSION) or type(data.get("schema")) is not int:
        raise ValueError("Unsupported analysis run schema")
    if type(data.get("table_projection_schema", 1)) is not int or data.get("table_projection_schema", 1) not in (1, 2):
        raise ValueError("Unsupported run table projection schema")
    if data.get("run_id") != record.run_id or data.get("project_id") != record.project_id:
        raise ValueError("Analysis run identity mismatch")
    if not record.run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in record.run_id):
        raise ValueError("Unsafe analysis run ID")
    if data.get("preset") not in ("Original FOAMS 1.0.5", "Modern analysis", "Compare methods"):
        raise ValueError("Invalid recorded method preset")
    datetime.fromisoformat(data["completed_at_utc"].replace("Z", "+00:00"))
    image_ids = [image["image_id"] for image in data["images"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate captured image identities")
    expected_methods = {"legacy", "modern"} if data["preset"] == "Compare methods" else {
        "modern" if data["preset"] == "Modern analysis" else "legacy"}
    if {method["method_id"] for method in data["methods"]} != expected_methods:
        raise ValueError("Recorded method outcomes do not match preset")
    if len(data["methods"]) != len(expected_methods):
        raise ValueError("Duplicate recorded method outcomes")
    for method in data["methods"]:
        if method["outcome"] not in ("success", "failed") or (method["outcome"] == "success") != (method["native"] is not None):
            raise ValueError("Recorded method outcome and native projection disagree")
    assets = data["assets"]
    if len({asset["sha256"] for asset in assets}) != len(assets):
        raise ValueError("Duplicate recorded assets")
    for asset in assets:
        if len(asset["sha256"]) != 64 or any(char not in "0123456789abcdef" for char in asset["sha256"]):
            raise ValueError("Invalid historical asset checksum")
        if asset["path"] != f"assets/history/{asset['sha256']}.bin":
            raise ValueError("Invalid historical asset path")
    known = {asset["sha256"] for asset in assets}
    for image in data["images"]:
        if not set(image["asset_ids"].values()).issubset(known):
            raise ValueError("Missing captured image asset reference")
    for artifact in data["canonical_original"]:
        if artifact["sha256"] not in known or Path(artifact["filename"]).name != artifact["filename"]:
            raise ValueError("Invalid canonical Original artifact reference")
    if "tables" in data:
        from .run_exports import TABLE_KEYS
        tables = data["tables"]
        for name, keys in TABLE_KEYS.items():
            identities = set()
            for row in tables[name]:
                if row.get("project_id") != record.project_id or row.get("run_id") != record.run_id:
                    raise ValueError("Projected row belongs to another run")
                identity = tuple(row.get(key) for key in keys)
                if identity in identities:
                    raise ValueError(f"Duplicate projected key in {name}")
                identities.add(identity)
        captured_images = {(image["image_id"], image["snapshot_id"]) for image in data["images"]}
        groups = {(row["method_id"], row["group_id"]) for row in tables["groups"]}
        for row in tables["objects"] + tables["group_images"]:
            if (row["image_id"], row["snapshot_id"]) not in captured_images:
                raise ValueError("Invalid projected image foreign key")
        for row in tables["group_images"] + tables["distributions"]:
            if (row["method_id"], row["group_id"]) not in groups:
                raise ValueError("Invalid projected group foreign key")
    if json_bytes(data) != record.payload:
        raise ValueError("Run payload is not canonical JSON")
    return data


def application_identity():
    try:
        app_version = version("pyro-foams")
    except PackageNotFoundError:
        app_version = None
    try:
        root = Path(__file__).resolve().parents[2]
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=2)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=2)
        revision_id = revision.stdout.strip() if revision.returncode == 0 else None
        dirty_state = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        revision_id = dirty_state = None
    return {"version": app_version, "revision": revision_id, "dirty": dirty_state,
            "revision_availability": "available" if revision_id else "unavailable"}


def _legacy_native(result):
    analysis = result.analysis
    groups = []
    objects = []
    selected = set(analysis.raw_selection_result.selected_identities)
    statistics = analysis.shape_statistics_result
    shape = set(statistics.summary_population_identities)
    threshold = set(statistics.threshold_identities)
    histogram = set(analysis.shape_histogram_result.population_identities)
    for group in result.prepared.groups:
        retained = {(item.image_id, item.component_id) for item in group.diameter_filtered_objects}
        groups.append({"group_id": group.group_id, "source_slot": group.source_slot, "scale_px_per_mm": group.scale_px_per_mm,
            "images": [image.image_input.image_id for image in group.images], "area": structured(group.group_area_result),
            "histogram": structured(group.histogram_result), "normalization": structured(group.normalization_result)})
        for raw in group.unfiltered_pore_objects:
            item = structured(raw)
            identity = (raw.image_id, raw.component_id)
            item.update(group_id=group.group_id, selected=identity in selected, shape_population=identity in shape,
                        shape_threshold_population=identity in threshold, shape_histogram_population=identity in histogram,
                        shape_included_metrics=[row.metric for row in statistics.statistic_rows if identity in row.included_identities],
                        shape_excluded_metrics=[row.metric for row in statistics.statistic_rows if identity in row.excluded_identities],
                        preparation_retained=identity in retained)
            objects.append(item)
    return {"method": analysis.method, "profile": result.prepared.perimeter_profile, "groups": groups,
            "objects": objects, "assignments": structured(result.source_assignments),
            "selected": structured(analysis.binned_selection_result), "raw_selection": structured(analysis.raw_selection_result),
            "conversion": structured(analysis.conversion_result), "volume": structured(analysis.volume_result),
            "summary": structured(analysis.distribution_summary_result), "categories": structured(analysis.size_category_result),
            "shape_statistics": structured(analysis.shape_statistics_result), "shape_histogram": structured(analysis.shape_histogram_result),
            "raw_rows": structured(analysis.raw_data_rows), "raw_selected_rows": structured(analysis.rawcutoffs_data_rows),
            "diagnostics": structured(analysis.diagnostics)}


def _modern_native(result, prepared):
    groups = [{"group_id": key[1], "distribution": structured(group)} for key, group in result.distribution_result.groups.items()]
    return {"method": result.nesting_result.method, "profile": None, "groups": groups,
            "objects": structured(prepared.measurements), "image_records": structured(prepared.image_records),
            "bin_spec": structured(result.distribution_result.bin_spec), "selected": structured(result.nesting_result),
            "reconstruction": structured(result.reconstruction_result),
            "diagnostics": structured(result.distribution_result.diagnostics)}


def create_run_projection(request, results, sample_name, source_arrays=()):
    if request.action != "run":
        raise ValueError("Only explicit Run analysis requests produce durable runs")
    assets = {}
    descriptors = {}

    def asset(data, kind):
        checksum = digest(data)
        assets.setdefault(checksum, data)
        descriptors.setdefault(checksum, {"sha256": checksum, "path": f"assets/history/{checksum}.bin", "kind": kind})
        return checksum

    sources = dict(source_arrays)
    images = []
    for image in request.snapshot.images:
        metadata = {field.name: structured(getattr(image, field.name)) for field in fields(image)
                    if not isinstance(getattr(image, field.name), np.ndarray)}
        metadata["accepted_labels_checksum_sha256"] = checksum_ndarray(image.accepted_labels)
        metadata["asset_ids"] = {}
        metadata["asset_metadata"] = {}
        arrays = {"working_image": image.working_image, "accepted_mask": image.accepted_mask, "accepted_labels": image.accepted_labels}
        if image.image_id in sources:
            if checksum_ndarray(sources[image.image_id]) != image.source_checksum_sha256:
                raise ValueError("Captured historical source checksum mismatch")
            arrays["source_image"] = sources[image.image_id]
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            metadata["asset_ids"][name] = asset(buffer.getvalue(), "numpy-array")
            metadata["asset_metadata"][name] = {"checksum_sha256": checksum_ndarray(array), "shape": list(array.shape), "dtype": str(array.dtype)}
        images.append(metadata)
    methods = []
    canonical = []
    requested = ("legacy", "modern") if request.mode == "Compare methods" else (
        "modern" if request.mode == "Modern analysis" else "legacy",)
    for method in requested:
        result = results.get(method)
        error = results.get(f"{method}_error")
        native = _legacy_native(result) if result is not None and method == "legacy" else (
            _modern_native(result, results["modern_prepared"]) if result is not None else None)
        diagnostics = [error_diagnostic(method, error)] if error is not None else []
        methods.append({"method_id": method, "outcome": "success" if result is not None else "failed",
                        "settings": structured(request.legacy_config if method == "legacy" else request.modern_config),
                        "diagnostics": diagnostics, "native": native})
        if result is not None and method == "legacy":
            with tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "canonical"
                write_foams105_export_bundle(result.analysis, output)
                for path in sorted(output.iterdir()):
                    canonical.append({"filename": path.name, "sha256": asset(path.read_bytes(), "canonical-original")})
    data = {"schema": RUN_SCHEMA_VERSION, "table_projection_schema": 2, "run_id": request.request_id, "project_id": request.snapshot.project_id,
            "sample_name": sample_name, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "preset": request.mode, "session_id": request.session_id, "application": application_identity(),
            "draft_settings": structured(dict(request.draft_settings)), "method_settings": structured(dict(request.method_settings)),
            "input_signature": structured(request.input_signature), "images": images, "methods": methods,
            "assets": list(descriptors.values()), "canonical_original": canonical}
    record = SavedAnalysisRun(request.request_id, request.snapshot.project_id, json_bytes(data))
    validate_run(record)
    from .run_exports import run_tables
    data["tables"] = run_tables(record)
    record = SavedAnalysisRun(request.request_id, request.snapshot.project_id, json_bytes(data))
    validate_run(record)
    return RunProjection(record, tuple(assets.items()))


def append_run(project, projection):
    validate_run(projection.record)
    if projection.record.project_id != project.project_id:
        raise ValueError("Cannot append a run from a different physical sample")
    if any(run.run_id == projection.record.run_id for run in project.analysis_runs):
        raise ValueError("Duplicate analysis run ID")
    for checksum, data in projection.assets:
        if digest(data) != checksum:
            raise ValueError("Historical asset checksum mismatch")
    project.analysis_assets.update(projection.assets)
    project.analysis_runs.append(projection.record)


def validate_historical_assets(data, assets):
    for image in data["images"]:
        for kind, checksum in image["asset_ids"].items():
            content = assets[checksum]
            if digest(content) != checksum:
                raise ValueError("Historical array file checksum mismatch")
            array = np.load(io.BytesIO(content), allow_pickle=False)
            metadata = image["asset_metadata"][kind]
            if list(array.shape) != metadata["shape"] or str(array.dtype) != metadata["dtype"] or checksum_ndarray(array) != metadata["checksum_sha256"]:
                raise ValueError("Historical array metadata/checksum mismatch")
