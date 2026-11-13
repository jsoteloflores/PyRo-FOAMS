from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from .analysis_runs import digest, json_bytes, validate_run

EXPORT_SCHEMA_VERSION = 1

TABLE_KEYS = {
    "sample": ("project_id", "run_id"),
    "methods": ("project_id", "run_id", "method_id"),
    "images": ("project_id", "run_id", "image_id", "snapshot_id"),
    "groups": ("project_id", "run_id", "method_id", "group_id"),
    "group_images": ("project_id", "run_id", "method_id", "group_id", "image_id", "snapshot_id"),
    "objects": ("project_id", "run_id", "method_id", "image_id", "snapshot_id", "object_id"),
    "distributions": ("project_id", "run_id", "method_id", "row_ordinal"),
    "summary_metrics": ("project_id", "run_id", "method_id", "metric_id", "population"),
    "diagnostics": ("project_id", "run_id", "method_id", "diagnostic_ordinal"),
}


def run_tables(record):
    data = validate_run(record)
    if "tables" in data:
        return data["tables"]
    base = {"project_id": record.project_id, "run_id": record.run_id}
    tables = {name: [] for name in TABLE_KEYS}
    tables["sample"].append({**base, "sample_name": data["sample_name"], "completed_at_utc": data["completed_at_utc"],
        "preset": data["preset"], "application_version": data["application"]["version"], "application_revision": data["application"]["revision"]})
    images = {image["image_id"]: image for image in data["images"]}
    for image in images.values():
        row = {**base, **{name: value for name, value in image.items() if name != "asset_ids"}}
        row.update({f"roi_{axis}": value for axis, value in zip(("x0", "y0", "x1", "y1"), image["processing_roi"])})
        row["calibration_unit"] = image["unit_name"]
        row.update({kind + "_file": f"inputs/{checksum}.npy" for kind, checksum in image["asset_ids"].items()})
        tables["images"].append(row)
    for method in data["methods"]:
        method_id = method["method_id"]
        common = {**base, "method_id": method_id}
        native = method["native"]
        tables["methods"].append({**common, "outcome": method["outcome"], "scientific_method": native["method"] if native else None,
            "perimeter_profile": native.get("profile") if native else None, "settings": method["settings"],
            "application_version": data["application"]["version"], "application_revision": data["application"]["revision"]})
        diagnostics = list(method["diagnostics"])
        if native is not None:
            if isinstance(native["diagnostics"], list):
                diagnostics.extend(native["diagnostics"])
            else:
                for category, identities in native["diagnostics"].items():
                    if isinstance(identities, list) and identities:
                        diagnostics.append({"stage": "distribution", "severity": "info", "code": category,
                                            "message": category, "context": {"identities": identities}})
        for index, diagnostic in enumerate(diagnostics):
            tables["diagnostics"].append({**common, "diagnostic_ordinal": index, "stage": diagnostic.get("stage", "scientific"),
                "severity": diagnostic.get("severity", "warning"), "code": diagnostic.get("code"),
                "message": diagnostic.get("message"), "context": diagnostic.get("context", {})})
        if native is None:
            continue
        assignments = {item["image_id"]: item for item in native.get("assignments", [])}
        image_records = {item["image_index"]: item for item in native.get("image_records", [])}
        distribution_rows = tables["distributions"]
        ordinal = 0
        for group in native["groups"]:
            group_id = group["group_id"]
            area = group.get("area", {})
            tables["groups"].append({**common, "group_id": group_id, "source_slot": group.get("source_slot"),
                "scale_px_per_mm": group.get("scale_px_per_mm"), "phase_corrected_area_mm2": area.get("phase_corrected_area_mm2"),
                "border_corrected_area_mm2": area.get("border_corrected_area_mm2")})
            members = group.get("images", [item["image_id"] for item in image_records.values()
                                          if item["magnification_group_id"] == group_id])
            for image_id in members:
                tables["group_images"].append({**common, "group_id": group_id, "image_id": image_id,
                    "snapshot_id": images[image_id]["snapshot_id"], "source_key": assignments.get(image_id, {}).get("source_key"),
                    "source_slot": assignments.get(image_id, {}).get("source_slot"),
                    "input_order": assignments.get(image_id, {}).get("input_order")})
            values = group["histogram"] if method_id == "legacy" else group["distribution"]
            for index, count in enumerate(values["counts"]):
                if method_id == "legacy":
                    label = values["bin_labels_mm"][index]
                    normalization = group["normalization"]
                    lower = upper = None
                    na = normalization["na_per_mm2"][index] if normalization else None
                    denominator = area["phase_corrected_area_mm2"]
                else:
                    label = None
                    bin_ = native["bin_spec"]["bins"][index]
                    lower, upper = bin_["lower_mm"], bin_["upper_mm"]
                    na, denominator = values["number_densities_per_mm2"][index], values["eligible_areas_mm2"][index]
                distribution_rows.append({**common, "stage": "prepared", "group_id": group_id, "row_ordinal": ordinal,
                    "bin_index": index, "label_mm": label, "lower_mm": lower, "upper_mm": upper, "count": count,
                    "plot_diameter_mm": label if method_id == "legacy" else bin_["geometric_midpoint_mm"],
                    "area_mm2": denominator, "na_per_mm2": na, "signed_nv_per_mm3": None, "volume_fraction": None,
                    "diameter_representation": "label_only" if method_id == "legacy" else "interval"})
                ordinal += 1
        selected = native["selected"]
        if method_id == "legacy":
            for index, row in enumerate(selected["rows"]):
                group = next(group for group in native["groups"] if group["group_id"] == row["group_id"])
                source_index = row["global_bin_index"]
                count = group["histogram"]["counts"][source_index] if isinstance(source_index, int) and source_index < len(group["histogram"]["counts"]) else None
                distribution_rows.append({**common, "stage": "selected", "group_id": row["group_id"], "row_ordinal": ordinal,
                    "source_slot": row["source_slot"], "bin_index": source_index, "label_mm": row["label_mm"],
                    "plot_diameter_mm": row["label_mm"],
                    "lower_mm": None, "upper_mm": None, "count": count, "area_mm2": group["area"]["phase_corrected_area_mm2"],
                    "na_per_mm2": row["na_per_mm2"], "signed_nv_per_mm3": native["conversion"]["signed_nv_per_mm3"][index],
                    "volume_fraction": native["volume"]["adjusted_volume_fractions"][index], "diameter_representation": "label_only"})
                ordinal += 1
        else:
            for index, group_id in enumerate(selected["source_group_ids"]):
                reconstruction = native["reconstruction"]
                distribution_rows.append({**common, "stage": "selected", "group_id": group_id, "row_ordinal": ordinal,
                    "bin_index": selected["selected_bin_indices"][index], "label_mm": None,
                    "plot_diameter_mm": native["bin_spec"]["bins"][selected["selected_bin_indices"][index]]["geometric_midpoint_mm"],
                    "lower_mm": selected["lower_edges_mm"][index], "upper_mm": selected["upper_edges_mm"][index],
                    "count": selected["counts"][index], "area_mm2": selected["eligible_areas_mm2"][index],
                    "na_per_mm2": selected["number_densities_per_mm2"][index],
                    "signed_nv_per_mm3": reconstruction["signed_nv_per_mm3"][index] if reconstruction else None,
                    "volume_fraction": None, "diameter_representation": "interval"})
                ordinal += 1
        geometry = {(row["image_id"], row["component_id"]): row for row in native.get("raw_rows", [])}
        for object_ in native["objects"]:
            if method_id == "legacy":
                image_id, object_id = object_["image_id"], object_["component_id"]
                row = {**object_, **geometry.get((image_id, object_id), {})}
                row["shape_population"] = object_["shape_population"]
            else:
                image_id = image_records[object_["image_index"]]["image_id"]
                object_id = object_["label"]
                row = dict(object_)
                row["selected"] = {"value": None, "availability": "unavailable", "reason": "modern density nesting issues no raw-object selection"}
                row["shape_population"] = None
            row.update(common, image_id=image_id, snapshot_id=images[image_id]["snapshot_id"], object_id=object_id,
                       measured=True, length_unit="mm", area_unit="mm^2")
            tables["objects"].append(row)

        def metrics(value, prefix, population):
            if isinstance(value, dict) and "availability" not in value:
                for name, item in value.items():
                    if name not in ("diagnostics", "source_commit", "source_lines", "source_file", "source_repository", "method", "rows", "shape_result"):
                        metrics(item, prefix + "." + name, population)
            elif value is None or isinstance(value, (int, float)) or isinstance(value, dict) and "availability" in value:
                unit = "%" if "percent" in prefix else "mm^-3" if "nv_per_mm3" in prefix else "mm^2" if "mm2" in prefix else "mm" if prefix.endswith("_mm") else "deg" if "_deg" in prefix else "1"
                tables["summary_metrics"].append({**common, "metric_id": prefix, "value": value, "unit": unit, "population": population})
        for section in ("volume", "summary", "categories", "shape_statistics", "reconstruction"):
            if section in native:
                value = native[section]
                if section == "reconstruction" and value is None:
                    value = {"value": None, "availability": "unavailable", "reason": "reconstruction disabled"}
                metrics(value, section, "shape-summary" if section == "shape_statistics" else "selected-distribution")
    return tables


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite value must carry an explicit availability status")
        return format(value, ".17g")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return value


def export_selected_run(record, assets, destination, cancel=None, presentation_errors=()):
    data = validate_run(record)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Selected-run export requires a fresh directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{record.run_id}-", dir=destination.parent))
    files = {}

    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Selected-run export cancelled")

    def write(name, payload, count=None):
        check_cancel()
        path = temporary / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files[name] = {"sha256": digest(payload), "row_count": count, "schema": EXPORT_SCHEMA_VERSION}

    try:
        dictionary = {}
        tables = run_tables(record)
        for name, rows in tables.items():
            columns = list(TABLE_KEYS[name])
            for row in rows:
                for column in row:
                    if column not in columns:
                        columns.append(column)
            value_columns = tuple(columns)
            normalized = []
            for row in rows:
                check_cancel()
                converted = dict(row)
                for column in value_columns:
                    value = row.get(column)
                    if column not in TABLE_KEYS[name]:
                        availability = value.get("availability") if isinstance(value, dict) else "available" if value is not None else "unavailable"
                        if isinstance(value, dict) and "availability" not in value:
                            availability = "available"
                        reason = value.get("reason", "") if isinstance(value, dict) else "not supplied by this method" if value is None else ""
                        converted[column] = value.get("value") if isinstance(value, dict) and "availability" in value else value
                        converted[column + "_availability"] = availability
                        converted[column + "_reason"] = reason
                        for suffix in ("_availability", "_reason"):
                            if column + suffix not in columns:
                                columns.append(column + suffix)
                normalized.append(converted)
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            identities = set()
            for row in normalized:
                key = tuple(row.get(column) for column in TABLE_KEYS[name])
                if key in identities:
                    raise ValueError(f"Duplicate primary key in {name}: {key}")
                identities.add(key)
                writer.writerow({column: _cell(row.get(column)) for column in columns})
            write(name + ".csv", stream.getvalue().encode("utf-8"), len(rows))
            dictionary[name + ".csv"] = {"primary_key": TABLE_KEYS[name], "columns": columns,
                "grain": "one native " + name.rstrip("s") + " per primary key", "nullable": "Empty with *_availability/*_reason; zero is numeric zero",
                "foreign_keys": ["project_id/run_id -> sample.csv", "method_id -> methods.csv when present",
                                 "image_id/snapshot_id -> images.csv when present", "method_id/group_id -> groups.csv when present"],
                "units": {column: "mm^-2" if "na_per_mm2" in column else "mm^-3" if "nv_per_mm3" in column else
                          "mm^2" if "mm2" in column else "mm/px" if column == "mm_per_px" else "px/mm" if column == "scale_px_per_mm" else
                          "mm" if column.endswith("_mm") else "px^2" if column == "area_px" else "px" if column.endswith("_px") or column.startswith(("bbox_", "roi_", "centroid_")) else
                          "deg" if column.endswith("_deg") else "row unit_name^2" if column == "area_units2" else "row unit_name" if column.endswith("_units") else
                          "row unit" if column == "value" else "1 or text/identifier" for column in columns}}
        settings_payload = {"run_id": record.run_id, "draft_settings": data["draft_settings"],
                    "methods": [{"method_id": method["method_id"], "settings": method["settings"]} for method in data["methods"]]}
        write("settings.json", json_bytes(settings_payload))
        write("run.json", record.payload)
        write("data_dictionary.json", json_bytes(dictionary))
        for artifact in data["canonical_original"]:
            payload = assets.get(artifact["sha256"])
            if payload is None or digest(payload) != artifact["sha256"]:
                raise ValueError("Canonical Original artifact is missing or damaged")
            write("original_compatible/" + artifact["filename"], payload)
        for asset in data["assets"]:
            if asset["kind"] == "numpy-array":
                payload = assets.get(asset["sha256"])
                if payload is None or digest(payload) != asset["sha256"]:
                    raise ValueError("Captured input asset is missing or damaged")
                write("inputs/" + asset["sha256"] + ".npy", payload)
        write("README.md", EXPORT_README.encode("utf-8"))
        write("manifest.json", json_bytes({"schema": EXPORT_SCHEMA_VERSION, "complete": True,
            "project_id": record.project_id, "run_id": record.run_id, "record_sha256": digest(record.payload),
            "application": data["application"], "presentation_errors": list(presentation_errors), "files": files,
            "table_availability": {name: "available" if rows else "header-only; no supplied rows" for name, rows in tables.items()}}))
        check_cancel()
        if destination.exists():
            raise FileExistsError("Export directory appeared during staging")
        os.rename(temporary, destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


EXPORT_README = """# Selected Analysis Run

This bundle exports one captured physical sample and run, not current project drafts.
Original and Modern populations, denominators and grids are not interchangeable.
CSV is UTF-8, comma separated, with binary64 round-trip precision. Null values are
empty with availability/reason fields; real zero and signed inverse values remain numeric.
Original distribution labels have no invented edges. Duplicate labels retain distinct
row ordinals. Modern nesting issues density selections, not an Original raw-object selection.
Reconstruction disabled is unavailable, not zero. Canonical TSV/sidecar files, when
available, are preserved byte-for-byte under original_compatible, not tidy CSV tables.
Full keys, grains, foreign keys and units are in data_dictionary.json.

## Join example (standard library only)

```python
import csv
from pathlib import Path

def joined_objects(folder, method_id):
    folder = Path(folder)
    def rows(name):
        with (folder / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    image_keys = ("project_id", "run_id", "image_id", "snapshot_id")
    images = {tuple(row[key] for key in image_keys): row for row in rows("images.csv")}
    memberships = {}
    for row in rows("group_images.csv"):
        if row["method_id"] == method_id:
            key = tuple(row[field] for field in image_keys)
            if key in memberships:
                raise ValueError("Multiple groups for an image; do not multiply object rows")
            memberships[key] = row
    for object_row in rows("objects.csv"):
        if object_row["method_id"] == method_id:
            key = tuple(object_row[field] for field in image_keys)
            yield object_row, images[key], memberships[key]

external_rows = list(joined_objects("sample_a", "modern"))
external_rows.extend(joined_objects("sample_b", "modern"))
# Stable project/run IDs keep samples separate; no cross-method/bin aggregation.
```
"""
