from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .analysis_runs import RUN_SCHEMA_VERSION, SavedAnalysisRun, digest, validate_historical_assets, validate_run
from .sample_project import (
    PROJECT_SCHEMA_VERSION,
    SOURCE_FIDELITY_LEGACY_UNKNOWN,
    SOURCE_FIDELITY_PRESERVED,
    AcceptedSnapshot,
    Calibration,
    SampleProject,
    SampleProjectError,
    add_image,
    checksum_ndarray,
    create_project,
    derive_working_image,
    labels_from_mask,
    normalize_mask,
)

MANIFEST_FILENAME = "manifest.json"
SOURCES_DIR = "assets/sources"
WORKING_DIR = "assets/working"
DRAFT_MASKS_DIR = "assets/draft_masks"
ACCEPTED_MASKS_DIR = "assets/accepted_masks"
LABELS_DIR = "assets/labels"


class SampleProjectIoError(SampleProjectError):
    """Raised for non-destructive project load/save failures."""



def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SampleProjectIoError(f"missing or unreadable image asset: {path}")
    if image.dtype != np.uint8:
        image = cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image



def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")
    try:
        ok = cv2.imwrite(str(temporary_path), image)
        if not ok:
            raise SampleProjectIoError(f"failed writing image asset: {path}")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _asset_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise SampleProjectIoError(f"project asset path escapes project folder: {relative_path}")
    return candidate


def _write_labels(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp.npy")
    try:
        with temporary_path.open("wb") as handle:
            np.save(handle, np.asarray(labels, dtype=np.int32), allow_pickle=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_source_array(path: Path, source: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp.npy")
    try:
        with temporary_path.open("wb") as handle:
            np.save(handle, source, allow_pickle=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_source_array(path: Path) -> np.ndarray:
    try:
        source = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SampleProjectIoError(f"missing or corrupt source asset: {path}") from exc
    try:
        derive_working_image(source)
    except SampleProjectError as exc:
        raise SampleProjectIoError(f"unsupported source asset {path}: {exc}") from exc
    return source


def _load_labels(root: Path, relative_path: Optional[str], mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if not relative_path:
        return labels_from_mask(mask) if mask is not None else None
    try:
        labels = np.load(_asset_path(root, relative_path), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SampleProjectIoError(f"missing or corrupt label asset: {relative_path}") from exc
    if labels.ndim != 2 or labels.dtype != np.int32:
        raise SampleProjectIoError(f"label asset must be a two-dimensional int32 array: {relative_path}")
    if mask is None or labels.shape != mask.shape or not np.array_equal(labels > 0, mask > 0):
        raise SampleProjectIoError(f"label asset is inconsistent with its foreground mask: {relative_path}")
    return labels.copy()



def _atomic_write_manifest(project_root: Path, payload: Dict[str, Any]) -> None:
    manifest_path = project_root / MANIFEST_FILENAME
    temporary_path = project_root / (MANIFEST_FILENAME + ".tmp")
    project_root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    temporary_path.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary_path, manifest_path)



def save_project(project: SampleProject, project_root: Path) -> Path:
    _validate_project_for_save(project)
    root = Path(project_root)
    root.mkdir(parents=True, exist_ok=True)

    # Write/refresh assets first, then atomically replace the manifest.
    for image in project.images:
        if image.source_image is None:
            raise SampleProjectIoError(f"image {image.image_id} has no source array in memory")

        source_relpath = f"{SOURCES_DIR}/{image.source_checksum_sha256}.npy"
        source_path = root / source_relpath
        _write_source_array(source_path, image.source_image)
        image.source_asset_path = source_relpath

        if image.working_image is None:
            raise SampleProjectIoError(f"image {image.image_id} has no working image in memory")
        working_checksum = checksum_ndarray(image.working_image)
        working_relpath = f"{WORKING_DIR}/{working_checksum}.png"
        _write_image(root / working_relpath, image.working_image)
        image.working_asset_path = working_relpath

        if image.draft_mask is not None:
            draft_relpath = f"{DRAFT_MASKS_DIR}/{checksum_ndarray(image.draft_mask)}.png"
            _write_image(root / draft_relpath, image.draft_mask)
            image.draft_mask_asset_path = draft_relpath
            if image.draft_labels is not None:
                labels_relpath = f"{LABELS_DIR}/{checksum_ndarray(image.draft_labels)}.npy"
                _write_labels(root / labels_relpath, image.draft_labels)
                image.draft_labels_asset_path = labels_relpath
            else:
                image.draft_labels_asset_path = None
        else:
            image.draft_mask_asset_path = None
            image.draft_labels_asset_path = None

        if image.accepted_mask is not None:
            accepted_snapshot_id = image.accepted_snapshot.snapshot_id if image.accepted_snapshot else image.image_id
            accepted_relpath = f"{ACCEPTED_MASKS_DIR}/{accepted_snapshot_id}.png"
            _write_image(root / accepted_relpath, image.accepted_mask)
            image.accepted_mask_asset_path = accepted_relpath
            if image.accepted_labels is not None:
                labels_relpath = f"{LABELS_DIR}/{checksum_ndarray(image.accepted_labels)}.npy"
                _write_labels(root / labels_relpath, image.accepted_labels)
                image.accepted_labels_asset_path = labels_relpath
            else:
                image.accepted_labels_asset_path = None
        else:
            image.accepted_mask_asset_path = None
            image.accepted_labels_asset_path = None

    _write_run_history(project, root)
    _atomic_write_manifest(project_root=root, payload=project.to_manifest_dict())
    return root / MANIFEST_FILENAME


def _validate_project_for_save(project: SampleProject) -> None:
    if project.schema_version != PROJECT_SCHEMA_VERSION or not project.project_id:
        raise SampleProjectIoError("project schema or stable project ID is invalid")
    if not isinstance(project.analysis_setup, dict):
        raise SampleProjectIoError("analysis_setup must be a dictionary")
    if project.unavailable_history:
        raise SampleProjectIoError("Historical runs are unavailable; refusing to overwrite their manifest references")
    seen_runs = set()
    for run in project.analysis_runs:
        try:
            validate_run(run)
        except (ValueError, KeyError, TypeError) as exc:
            raise SampleProjectIoError(f"invalid analysis run: {exc}") from exc
        if run.project_id != project.project_id or run.run_id in seen_runs:
            raise SampleProjectIoError("Historical run project identity or run ID is invalid")
        seen_runs.add(run.run_id)
    seen_ids = set()
    for image in project.images:
        if not image.image_id or image.image_id in seen_ids:
            raise SampleProjectIoError("project image IDs must be present and unique")
        seen_ids.add(image.image_id)
        source = image.source_image
        working = image.working_image
        if source is None or working is None:
            raise SampleProjectIoError(f"image {image.image_id} has no source or working image")
        if source.shape[:2] != (image.height, image.width):
            raise SampleProjectIoError(f"image {image.image_id} source dimensions are invalid")
        if checksum_ndarray(source) != image.source_checksum_sha256:
            raise SampleProjectIoError(f"image {image.image_id} source checksum does not match its pixels")
        derived_working, conversion_id = derive_working_image(source)
        if image.processing_roi is None:
            raise SampleProjectIoError(f"image {image.image_id} processing ROI is missing")
        x0, y0, x1, y1 = image.processing_roi
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise SampleProjectIoError(f"image {image.image_id} processing ROI is outside source dimensions")
        if working.dtype != np.uint8 or working.shape[:2] != (y1 - y0, x1 - x0):
            raise SampleProjectIoError(f"image {image.image_id} working image does not match its ROI")
        if image.processing_conversion_id != conversion_id:
            raise SampleProjectIoError(f"image {image.image_id} processing conversion ID does not match its source")
        if not np.array_equal(working, derived_working[y0:y1, x0:x1]):
            raise SampleProjectIoError(f"image {image.image_id} working pixels do not match its source conversion and ROI")
        if image.calibration.units_per_px is not None:
            value = image.calibration.units_per_px
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise SampleProjectIoError(f"image {image.image_id} calibration must be finite and positive")
        for kind, mask, labels in (
            ("draft", image.draft_mask, image.draft_labels),
            ("accepted", image.accepted_mask, image.accepted_labels),
        ):
            if mask is None:
                if labels is not None:
                    raise SampleProjectIoError(f"image {image.image_id} {kind} labels have no mask")
                continue
            if mask.dtype != np.uint8 or mask.shape != working.shape[:2] or not set(np.unique(mask)).issubset({0, 255}):
                raise SampleProjectIoError(f"image {image.image_id} {kind} mask is invalid")
            if labels is None or labels.dtype != np.int32 or labels.shape != mask.shape:
                raise SampleProjectIoError(f"image {image.image_id} {kind} labels are missing or invalid")
            if not np.array_equal(labels > 0, mask > 0):
                raise SampleProjectIoError(f"image {image.image_id} {kind} labels do not match its mask")
        if image.accepted_mask is not None:
            snapshot = image.accepted_snapshot
            if snapshot is None:
                raise SampleProjectIoError(f"image {image.image_id} accepted mask has no snapshot")
            if checksum_ndarray(image.accepted_mask) != snapshot.checksum_sha256:
                raise SampleProjectIoError(f"image {image.image_id} accepted mask checksum does not match")
            if snapshot.source_checksum_sha256 != image.source_checksum_sha256:
                raise SampleProjectIoError(f"image {image.image_id} accepted snapshot source provenance is stale")
            if snapshot.working_image_checksum_sha256 != checksum_ndarray(working):
                raise SampleProjectIoError(f"image {image.image_id} accepted snapshot working-image provenance is stale")
            if snapshot.processing_roi != image.processing_roi:
                raise SampleProjectIoError(f"image {image.image_id} accepted snapshot ROI is stale")
        elif image.accepted_snapshot is not None:
            raise SampleProjectIoError(f"image {image.image_id} accepted snapshot has no mask")
        if image.analysis_included and image.accepted_snapshot is None:
            raise SampleProjectIoError(f"image {image.image_id} is included without an accepted snapshot")



def _load_calibration(data: Dict[str, Any]) -> Calibration:
    units_per_px = data.get("units_per_px")
    unit_name = data.get("unit_name")
    if units_per_px is None or unit_name is None:
        return Calibration()
    if isinstance(units_per_px, bool) or not isinstance(unit_name, str):
        raise SampleProjectIoError("calibration value or unit has an invalid type")
    value = float(units_per_px)
    if not math.isfinite(value) or value <= 0:
        raise SampleProjectIoError("calibration value must be finite and positive")
    return Calibration(units_per_px=value, unit_name=unit_name)



def _load_accepted_snapshot(data: Optional[Dict[str, Any]]) -> Optional[AcceptedSnapshot]:
    if data is None:
        return None
    return AcceptedSnapshot(
        snapshot_id=str(data["snapshot_id"]),
        checksum_sha256=str(data["checksum_sha256"]),
        source_checksum_sha256=str(data["source_checksum_sha256"]),
        working_image_checksum_sha256=str(data["working_image_checksum_sha256"]),
        processing_roi=tuple(int(value) for value in data["processing_roi"]),
        source_revision=int(data["source_revision"]),
        draft_revision=int(data["draft_revision"]),
        calibration_revision=int(data["calibration_revision"]),
        source_kind=str(data["source_kind"]),
        source_details=dict(data.get("source_details", {})),
        created_at_utc=str(data["created_at_utc"]),
    )



def _load_mask_if_present(root: Path, relpath: Optional[str], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not relpath:
        return None
    path = _asset_path(root, relpath)
    mask = _read_image(path)
    return normalize_mask(mask, shape)



def open_project(project_root: Path) -> SampleProject:
    root = Path(project_root)
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise SampleProjectIoError(f"project manifest not found: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SampleProjectIoError(f"project manifest is malformed: {exc}") from exc

    if not isinstance(manifest, dict):
        raise SampleProjectIoError("project manifest root must be a JSON object")
    schema_version = manifest.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in (1, 2, PROJECT_SCHEMA_VERSION)
    ):
        raise SampleProjectIoError(
            f"unsupported project schema version: {schema_version}; expected {PROJECT_SCHEMA_VERSION}"
        )

    project_id = manifest.get("project_id")
    sample_name = manifest.get("sample_name")
    notes = manifest.get("notes", "")
    if not isinstance(project_id, str) or not project_id.strip():
        raise SampleProjectIoError("project manifest has no stable project ID")
    if not isinstance(sample_name, str) or not sample_name.strip():
        raise SampleProjectIoError("project manifest has no sample name")
    if not isinstance(notes, str):
        raise SampleProjectIoError("project notes must be text")
    project = create_project(sample_name, notes=notes)
    project.project_id = project_id
    project.analysis_revision = int(manifest.get("analysis_revision", 0))
    analysis_setup = manifest.get("analysis_setup", {})
    if not isinstance(analysis_setup, dict):
        raise SampleProjectIoError("analysis_setup must be a JSON object when present")
    project.analysis_setup = dict(analysis_setup)
    presentation_errors = manifest.get("analysis_presentation_errors", {})
    if not isinstance(presentation_errors, dict):
        raise SampleProjectIoError("analysis_presentation_errors must be an object")
    project.analysis_presentation_errors = presentation_errors

    images = manifest.get("images")
    if not isinstance(images, list):
        raise SampleProjectIoError("project manifest is missing image records")

    seen_image_ids = set()
    for image_data in images:
        if not isinstance(image_data, dict):
            raise SampleProjectIoError("project image record must be a JSON object")
        image_id = str(image_data.get("image_id"))
        if not image_id or image_id == "None" or image_id in seen_image_ids:
            raise SampleProjectIoError("project image IDs must be present and unique")
        seen_image_ids.add(image_id)
        included_value = image_data.get("analysis_included", False)
        if not isinstance(included_value, bool):
            raise SampleProjectIoError(f"image {image_id} analysis_included must be Boolean")
        source_asset_path = image_data.get("source_asset_path")
        if not isinstance(source_asset_path, str) or not source_asset_path:
            raise SampleProjectIoError(f"image {image_id} is missing source asset path")

        if schema_version == 1:
            source = _read_image(_asset_path(root, source_asset_path))
        else:
            source = _read_source_array(_asset_path(root, source_asset_path))
        loaded = add_image(
            project,
            source,
            source_filename=str(image_data.get("source_filename") or ""),
            display_name=str(image_data.get("display_name") or ""),
        )
        loaded.image_id = image_id
        loaded.source_asset_path = source_asset_path

        width = int(image_data.get("width", source.shape[1]))
        height = int(image_data.get("height", source.shape[0]))
        if (source.shape[1], source.shape[0]) != (width, height):
            raise SampleProjectIoError(
                f"image {image_id} source dimensions mismatch manifest ({source.shape[1]}x{source.shape[0]} != {width}x{height})"
            )

        checksum = str(image_data.get("source_checksum_sha256") or "")
        actual_checksum = checksum_ndarray(source)
        if checksum and checksum != actual_checksum:
            raise SampleProjectIoError(f"image {image_id} source checksum mismatch")
        loaded.source_checksum_sha256 = checksum or actual_checksum
        derived_working, conversion_id = derive_working_image(source)
        loaded.source_fidelity = (
            SOURCE_FIDELITY_LEGACY_UNKNOWN if schema_version == 1
            else str(image_data.get("source_fidelity", SOURCE_FIDELITY_PRESERVED))
        )
        persisted_conversion_id = image_data.get("processing_conversion_id", conversion_id)
        if schema_version >= 2 and persisted_conversion_id != conversion_id:
            raise SampleProjectIoError(f"image {image_id} processing conversion ID is unsupported")
        loaded.processing_conversion_id = conversion_id

        working_asset_path = image_data.get("working_asset_path")
        if working_asset_path:
            working_image = _read_image(_asset_path(root, working_asset_path))
        else:
            working_image = derived_working.copy()
        processing_roi_data = image_data.get("processing_roi")
        if processing_roi_data is None:
            processing_roi = (0, 0, width, height)
        else:
            if not isinstance(processing_roi_data, list) or len(processing_roi_data) != 4:
                raise SampleProjectIoError(f"image {image_id} processing ROI is malformed")
            processing_roi = tuple(int(value) for value in processing_roi_data)
        x0, y0, x1, y1 = processing_roi
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise SampleProjectIoError(f"image {image_id} processing ROI is outside source dimensions")
        if (x1 - x0, y1 - y0) != (working_image.shape[1], working_image.shape[0]):
            raise SampleProjectIoError(f"image {image_id} working dimensions do not match processing ROI")
        if not np.array_equal(working_image, derived_working[y0:y1, x0:x1]):
            raise SampleProjectIoError(f"image {image_id} working pixels do not match source conversion and ROI")
        if working_asset_path and Path(working_asset_path).stem != checksum_ndarray(working_image):
            raise SampleProjectIoError(f"image {image_id} working image checksum mismatch")
        loaded.working_image = working_image.copy()
        loaded.processing_roi = processing_roi
        loaded.working_asset_path = working_asset_path

        revisions = image_data.get("revisions") or {}
        loaded.source_revision = int(revisions.get("source", 0))
        loaded.calibration_revision = int(revisions.get("calibration", 0))
        loaded.draft_revision = int(revisions.get("draft", 0))
        loaded.accepted_revision = int(revisions.get("accepted", 0))
        loaded.analysis_included = included_value
        loaded.calibration = _load_calibration(dict(image_data.get("calibration") or {}))
        loaded.draft_processing_settings = dict(image_data.get("draft_processing_settings") or {})
        loaded.draft_source_kind = str(image_data.get("draft_source_kind") or "processing")
        loaded.draft_source_details = dict(image_data.get("draft_source_details") or {})

        shape = working_image.shape[:2]
        loaded.draft_mask_asset_path = image_data.get("draft_mask_asset_path")
        loaded.draft_mask = _load_mask_if_present(root, loaded.draft_mask_asset_path, shape)
        if loaded.draft_mask is not None and not image_data.get("draft_labels_asset_path"):
            raise SampleProjectIoError(f"image {image_id} draft labels asset path is missing")
        if loaded.draft_mask is not None and Path(loaded.draft_mask_asset_path).stem != checksum_ndarray(loaded.draft_mask):
            raise SampleProjectIoError(f"image {image_id} draft mask checksum mismatch")
        loaded.draft_labels_asset_path = image_data.get("draft_labels_asset_path")
        loaded.draft_labels = _load_labels(root, loaded.draft_labels_asset_path, loaded.draft_mask)
        if loaded.draft_labels_asset_path and Path(loaded.draft_labels_asset_path).stem != checksum_ndarray(loaded.draft_labels):
            raise SampleProjectIoError(f"image {image_id} draft labels checksum mismatch")

        loaded.accepted_mask_asset_path = image_data.get("accepted_mask_asset_path")
        loaded.accepted_mask = _load_mask_if_present(root, loaded.accepted_mask_asset_path, shape)
        if loaded.accepted_mask is not None and not image_data.get("accepted_labels_asset_path"):
            raise SampleProjectIoError(f"image {image_id} accepted labels asset path is missing")
        loaded.accepted_labels_asset_path = image_data.get("accepted_labels_asset_path")
        loaded.accepted_labels = _load_labels(root, loaded.accepted_labels_asset_path, loaded.accepted_mask)
        loaded.accepted_snapshot = _load_accepted_snapshot(image_data.get("accepted_snapshot"))

        if loaded.accepted_mask is not None:
            if loaded.accepted_snapshot is None:
                raise SampleProjectIoError(f"image {image_id} has an accepted mask without snapshot metadata")
            if loaded.accepted_snapshot.source_checksum_sha256 != loaded.source_checksum_sha256:
                raise SampleProjectIoError(f"image {image_id} accepted snapshot source checksum mismatch")
            if loaded.accepted_snapshot.working_image_checksum_sha256 != checksum_ndarray(working_image):
                raise SampleProjectIoError(f"image {image_id} accepted snapshot working-image checksum mismatch")
            if loaded.accepted_snapshot.processing_roi != processing_roi:
                raise SampleProjectIoError(f"image {image_id} accepted snapshot ROI mismatch")
            actual_mask_checksum = checksum_ndarray(loaded.accepted_mask)
            if actual_mask_checksum != loaded.accepted_snapshot.checksum_sha256:
                raise SampleProjectIoError(f"image {image_id} accepted mask checksum mismatch")
            if loaded.accepted_labels_asset_path and Path(loaded.accepted_labels_asset_path).stem != checksum_ndarray(loaded.accepted_labels):
                raise SampleProjectIoError(f"image {image_id} accepted labels checksum mismatch")
        elif loaded.accepted_snapshot is not None:
            raise SampleProjectIoError(f"image {image_id} has accepted snapshot metadata but its mask asset is missing")

        if loaded.analysis_included and loaded.accepted_snapshot is None:
            raise SampleProjectIoError(f"image {image_id} is included but has no accepted snapshot")

    if schema_version >= 3:
        _load_run_history(project, root, manifest.get("analysis_runs", []))
    return project


def _write_history_bytes(path: Path, payload: bytes) -> None:
    if path.is_file() and digest(path.read_bytes()) == digest(payload):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_run_history(project, root):
    for run in project.analysis_runs:
        data = validate_run(run)
        for asset in data["assets"]:
            payload = project.analysis_assets.get(asset["sha256"])
            if payload is None or digest(payload) != asset["sha256"]:
                raise SampleProjectIoError(f"Missing or damaged historical asset {asset['sha256']}")
            _write_history_bytes(_asset_path(root, asset["path"]), payload)
        _write_history_bytes(_asset_path(root, run.index_entry()["path"]), run.payload)


def _load_run_history(project, root, index):
    if not isinstance(index, list):
        project.unavailable_history.append({"diagnostic": "Run history index is malformed"})
        return
    seen = set()
    for entry in index:
        try:
            if not isinstance(entry, dict) or type(entry.get("schema")) is not int or entry.get("schema") not in (1, RUN_SCHEMA_VERSION):
                raise ValueError("Unsupported historical run schema")
            if entry.get("project_id") != project.project_id or entry.get("run_id") in seen:
                raise ValueError("Historical run identity mismatch or duplicate")
            payload = _asset_path(root, entry["path"]).read_bytes()
            if digest(payload) != entry["sha256"]:
                raise ValueError("Historical run checksum mismatch")
            run = SavedAnalysisRun(entry["run_id"], project.project_id, payload)
            data = validate_run(run)
            if entry["schema"] != run.schema:
                raise ValueError("Historical run index schema does not match its payload")
            if entry["path"] != run.index_entry()["path"]:
                raise ValueError("Historical run path does not match its immutable identity")
            loaded_assets = {}
            for asset in data["assets"]:
                content = project.analysis_assets.get(asset["sha256"])
                if content is None:
                    content = _asset_path(root, asset["path"]).read_bytes()
                if digest(content) != asset["sha256"]:
                    raise ValueError("Historical asset checksum mismatch")
                loaded_assets[asset["sha256"]] = content
            validate_historical_assets(data, loaded_assets)
            project.analysis_assets.update(loaded_assets)
            project.analysis_runs.append(run)
            seen.add(run.run_id)
        except (OSError, ValueError, TypeError, KeyError, SampleProjectIoError) as exc:
            unavailable = dict(entry) if isinstance(entry, dict) else {"entry": entry}
            unavailable["diagnostic"] = str(exc)
            project.unavailable_history.append(unavailable)



def clone_project_folder(source_folder: Path, destination_folder: Path) -> Path:
    src = Path(source_folder)
    dst = Path(destination_folder)
    if dst.exists():
        raise SampleProjectIoError(f"destination already exists: {dst}")
    shutil.copytree(src, dst)
    return dst
