from __future__ import annotations

import math

NA_PLOT = "2D number density - N_A"
NV_PLOT = "3D number density - N_V (inverse reconstruction)"


def reconstruction_series(data, tables):
    series = []
    messages = []
    for method in data["methods"]:
        method_id = method["method_id"]
        native = method["native"]
        if method["outcome"] != "success" or native is None:
            messages.extend(f"{method_id} failed: {item['message']}" for item in method["diagnostics"])
            continue
        rows = [row for row in tables["distributions"] if row["method_id"] == method_id and row["stage"] == "selected"]
        if method_id == "legacy":
            title = "Original FOAMS inverse Saltykov reconstruction"
            positions = [row["label_mm"] for row in native["selected"]["rows"]]
            values = native["conversion"]["signed_nv_per_mm3"]
            coordinates = "Original label-only diameter positions (mm)"
            warnings = native.get("diagnostics", [])
        else:
            reconstruction = native.get("reconstruction")
            if reconstruction is None:
                messages.append("Modern: 3D reconstruction was not calculated for this run")
                continue
            title = "Modern spherical 3D reconstruction"
            positions = reconstruction.get("operator", {}).get("representative_diameters_mm")
            if positions is None:
                messages.append("Modern: native reconstructed class geometry is unavailable in this run")
                continue
            values = reconstruction["signed_nv_per_mm3"]
            coordinates = "Modern representative sphere diameters: native class upper edges (mm)"
            warnings = {name: reconstruction.get(name) for name in (
                "status", "upper_tail_assumption", "upper_range_truncated", "upper_range_advisory",
                "negative_absolute_bin_indices", "materially_negative_absolute_bin_indices",
                "roundoff_negative_absolute_bin_indices", "negative_classification_tolerance_per_mm3")}
        if len(positions) != len(values) or len(rows) != len(values):
            raise ValueError(f"{method_id}: reconstruction values, native coordinates and selected row identities disagree")
        points = []
        for index, (diameter, value, row) in enumerate(zip(positions, values, rows)):
            points.append({"project_id": data["project_id"], "run_id": data["run_id"], "method_id": method_id,
                "row_ordinal": row["row_ordinal"], "selected_ordinal": index, "group_id": row["group_id"],
                "bin_index": row["bin_index"], "diameter_mm": diameter, "signed_nv_per_mm3": value})
        series.append({"method_id": method_id, "title": title, "coordinates": coordinates,
                       "warnings": warnings, "points": points})
    return series, messages


def visible_points(points, *, log_x=True, log_y=False):
    visible = []
    omitted = {"invalid_diameter": 0, "unavailable": 0, "zero": 0, "negative": 0}
    for point in points:
        diameter = point["diameter_mm"]
        value = point["signed_nv_per_mm3"]
        if not isinstance(diameter, (int, float)) or not math.isfinite(diameter) or log_x and diameter <= 0:
            omitted["invalid_diameter"] += 1
        elif not isinstance(value, (int, float)) or not math.isfinite(value):
            omitted["unavailable"] += 1
        elif log_y and value <= 0:
            omitted["zero" if value == 0 else "negative"] += 1
        else:
            visible.append(point)
    return visible, omitted
