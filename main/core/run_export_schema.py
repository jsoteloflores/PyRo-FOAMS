from __future__ import annotations

SCALAR_METRICS = (
    ("volume", "target_vesicularity_percent", "%", "normalization-target", "setting"),
    ("volume", "target_volume_fraction", "1", "normalization-target", "setting"),
    ("volume", "normalization_factor", "1", "selected-distribution", "result"),
    ("volume", "raw_fraction_sum", "1", "selected-distribution", "result"),
    ("volume", "adjusted_fraction_sum", "1", "selected-distribution", "result"),
    ("volume", "nv_derived_vesicularity_percent", "%", "selected-distribution", "result"),
    ("volume", "normalization_residual", "1", "selected-distribution", "result"),
    ("summary", "correction_percent", "%", "number-density-correction", "setting"),
    ("summary", "solid_fraction", "1", "number-density-correction", "result"),
    ("summary", "total_nv_per_mm3", "mm^-3", "selected-distribution", "result"),
    ("summary", "corrected_nv_per_mm3", "mm^-3", "selected-distribution", "result"),
    ("categories", "lower_count", "count", "complete-histogram", "result"),
    ("categories", "middle_count", "count", "complete-histogram", "result"),
    ("categories", "upper_count", "count", "complete-histogram", "result"),
    ("categories", "boundary_count", "count", "complete-histogram", "result"),
    ("categories", "total_count", "count", "complete-histogram", "result"),
    ("categories", "lower_percentage", "%", "complete-histogram", "result"),
    ("categories", "middle_percentage", "%", "complete-histogram", "result"),
    ("categories", "upper_percentage", "%", "complete-histogram", "result"),
    ("categories", "lower_cutoff_mm", "mm", "configuration", "setting"),
    ("categories", "upper_cutoff_mm", "mm", "configuration", "setting"),
    ("shape_statistics", "maximum_configured_scale_px_per_mm", "px/mm", "configuration", "setting"),
    ("shape_statistics", "minimum_diameter_px", "px", "configuration", "setting"),
    ("shape_statistics", "threshold_mm", "mm", "shape-threshold", "setting"),
    ("shape_statistics", "threshold_count", "count", "shape-threshold", "result"),
    ("shape_statistics", "summary_population_count", "count", "shape-summary", "result"),
    ("reconstruction", "maximum_absolute_residual_per_mm2", "mm^-2", "selected-distribution", "result"),
    ("reconstruction", "normalized_infinity_residual", "1", "selected-distribution", "result"),
    ("reconstruction", "forward_consistency_tolerance_per_mm2", "mm^-2", "selected-distribution", "setting"),
    ("reconstruction", "negative_classification_tolerance_per_mm3", "mm^-3", "selected-distribution", "setting"),
)
SHAPE_REDUCTIONS = ("maximum", "minimum", "mean", "standard_deviation", "median", "mode", "n", "population_count")
SHAPE_UNITS = {"aspect_ratio": "1", "elongation": "1", "regularity": "1", "eccentricity": "1"}


def _quantity(common, metric_id, source, unit, population, kind="result", status=None, reason=None, **extra):
    if isinstance(source, dict):
        value = source.get("value")
        reason = source.get("reason") or reason
        availability = source.get("availability") or ("available" if value is not None else "unavailable")
        classification = source.get("legacy_classification")
        upstream = source.get("upstream_reason")
        operands = source.get("operands", [])
    else:
        value = source
        availability = "available" if value is not None else "unavailable"
        classification = upstream = None
        operands = []
    return {**common, "metric_id": metric_id, "value": value, "unit": unit, "population": population,
            "kind": kind, "availability": availability, "reason": reason or ("not supplied by source" if value is None else ""),
            "status": status or availability, "classification": classification, "upstream_reason": upstream,
            "operands": operands, "included_identities": [], "excluded_identities": [], **extra}


def summary_rows(common, native):
    rows = []
    for section, name, unit, population, kind in SCALAR_METRICS:
        source = native.get(section)
        if isinstance(source, dict):
            rows.append(_quantity(common, section + "." + name, source.get(name), unit, population, kind))
    if "reconstruction" in native and native["reconstruction"] is None:
        rows.append(_quantity(common, "reconstruction", None, "1", "selected-distribution", reason="reconstruction disabled"))
    for statistic in native.get("shape_statistics", {}).get("statistic_rows", []):
        metric = statistic["metric"]
        if metric not in SHAPE_UNITS:
            raise ValueError(f"Unsupported shape metric projection: {metric}")
        for reduction in SHAPE_REDUCTIONS:
            rows.append(_quantity(common, f"shape_statistics.{metric}.{reduction}", statistic[reduction],
                "count" if reduction in ("n", "population_count") else SHAPE_UNITS[metric], "shape-summary",
                status=statistic["status"], reason=statistic["reason"],
                included_identities=statistic["included_identities"], excluded_identities=statistic["excluded_identities"]))
    return rows


IDENTITY_FIELDS = ("project_id", "run_id", "method_id", "image_id", "snapshot_id", "object_id", "group_id")
TABLE_COLUMNS = {
    "sample": ("project_id", "run_id", "sample_name", "completed_at_utc", "preset", "application_version", "application_revision"),
    "methods": ("project_id", "run_id", "method_id", "outcome", "scientific_method", "perimeter_profile", "settings", "application_version", "application_revision"),
    "images": ("project_id", "run_id", "image_id", "snapshot_id", "display_name", "accepted_checksum_sha256", "accepted_labels_checksum_sha256",
        "source_checksum_sha256", "working_image_checksum_sha256", "source_revision", "accepted_revision", "calibration_revision",
        "units_per_px", "unit_name", "mm_per_px", "calibration_unit", "processing_roi", "roi_x0", "roi_y0", "roi_x1", "roi_y1",
        "asset_metadata", "source_image_file", "working_image_file", "accepted_mask_file", "accepted_labels_file"),
    "groups": ("project_id", "run_id", "method_id", "group_id", "source_slot", "scale_px_per_mm", "phase_corrected_area_mm2", "border_corrected_area_mm2"),
    "group_images": ("project_id", "run_id", "method_id", "group_id", "image_id", "snapshot_id", "source_key", "source_slot", "input_order"),
    "objects": ("project_id", "run_id", "method_id", "image_id", "snapshot_id", "object_id", "group_id", "component_id", "label", "image_index",
        "source_slot", "source_object_index", "output_index", "measured", "preparation_retained", "selected", "shape_population",
        "shape_threshold_population", "shape_histogram_population", "shape_included_metrics", "shape_excluded_metrics",
        "equivalent_diameter_mm", "area_mm2", "perimeter_mm", "major_axis_mm", "minor_axis_mm", "orientation_project_deg",
        "area_px", "perimeter_px", "centroid_x", "centroid_y", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "touches_border",
        "eq_diam_px", "circularity", "major_axis_px", "minor_axis_px", "aspect_ratio", "orientation_deg", "feret_max_px", "feret_min_px",
        "units_per_px", "area_units2", "eq_diam_units", "major_axis_units", "minor_axis_units", "feret_max_units", "feret_min_units",
        "unit_name", "length_unit", "area_unit"),
    "distributions": ("project_id", "run_id", "method_id", "row_ordinal", "stage", "group_id", "source_slot", "bin_index", "label_mm",
        "lower_mm", "upper_mm", "plot_diameter_mm", "count", "area_mm2", "na_per_mm2", "signed_nv_per_mm3", "volume_fraction", "diameter_representation"),
    "summary_metrics": ("project_id", "run_id", "method_id", "metric_id", "population", "value", "unit", "kind", "availability", "reason", "status",
        "classification", "upstream_reason", "operands", "included_identities", "excluded_identities"),
    "diagnostics": ("project_id", "run_id", "method_id", "diagnostic_ordinal", "stage", "severity", "code", "message", "context"),
}
TABLE_GRAINS = {
    "sample": "One captured physical sample and explicit analysis request.",
    "methods": "One independent requested method outcome, including failed methods.",
    "images": "One captured accepted image/snapshot; independent of current project inclusion or name.",
    "groups": "One run-scoped, method-specific resolution group; never cross-method pooled.",
    "group_images": "One captured image membership in one native method resolution group.",
    "objects": "One native measured component/label in a captured image; identities are not cross-method matches.",
    "distributions": "One native prepared or selected row, preserving order and duplicate Original labels by ordinal.",
    "summary_metrics": "One explicitly mapped quantity/reduction and population; unrounded source value, not recalculated statistics.",
    "diagnostics": "One source diagnostic per requested method and ordinal, including failed method diagnostics.",
}

FIELD_UNITS = {}
for _field in set().union(*TABLE_COLUMNS.values()):
    FIELD_UNITS[_field] = "text"
for _field in ("source_revision", "accepted_revision", "calibration_revision", "source_slot", "input_order", "source_object_index", "output_index",
               "component_id", "label", "image_index", "object_id", "row_ordinal", "bin_index", "count", "diagnostic_ordinal"):
    FIELD_UNITS[_field] = "count" if _field == "count" else "ordinal or native identifier"
for _field in ("measured", "preparation_retained", "selected", "shape_population", "shape_threshold_population", "shape_histogram_population", "touches_border"):
    FIELD_UNITS[_field] = "boolean"
for _field in ("roi_x0", "roi_y0", "roi_x1", "roi_y1", "perimeter_px", "centroid_x", "centroid_y", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
               "eq_diam_px", "major_axis_px", "minor_axis_px", "feret_max_px", "feret_min_px"):
    FIELD_UNITS[_field] = "px"
for _field in ("equivalent_diameter_mm", "perimeter_mm", "major_axis_mm", "minor_axis_mm", "label_mm", "lower_mm", "upper_mm", "plot_diameter_mm"):
    FIELD_UNITS[_field] = "mm"
for _field in ("area_mm2", "phase_corrected_area_mm2", "border_corrected_area_mm2"):
    FIELD_UNITS[_field] = "mm^2"
for _field in ("circularity", "aspect_ratio", "volume_fraction"):
    FIELD_UNITS[_field] = "1"
for _field in ("orientation_project_deg", "orientation_deg"):
    FIELD_UNITS[_field] = "deg"
for _field in ("eq_diam_units", "major_axis_units", "minor_axis_units", "feret_max_units", "feret_min_units"):
    FIELD_UNITS[_field] = "row unit_name"
FIELD_UNITS.update(area_px="px^2", area_units2="row unit_name^2", units_per_px="row unit_name/px",
                   mm_per_px="mm/px", scale_px_per_mm="px/mm", na_per_mm2="mm^-2", signed_nv_per_mm3="mm^-3", value="row unit")

FIELD_DEFINITIONS = {name: name.replace("_", " ") + "; retained native field, nullable when the method does not supply it."
                     for name in FIELD_UNITS}
FIELD_DEFINITIONS.update({
    "sample_name": "Sample display name captured at execution; not a join key and may contain Unicode, quotes or newlines.",
    "completed_at_utc": "UTC completion timestamp captured by the run, not the current save time.",
    "preset": "Original FOAMS 1.0.5, Modern analysis, or Compare methods as explicitly requested.",
    "application_version": "Installed application version at execution; empty when unavailable.",
    "application_revision": "Git revision at execution; empty when unavailable. Dirty-state metadata remains in run.json.",
    "outcome": "Independent success or failed outcome for this method; failed methods have no fabricated native rows.",
    "scientific_method": "Public native method identifier, not the UI preset name.",
    "perimeter_profile": "Original public perimeter profile identifier; not applicable to Modern geometry.",
    "settings": "JSON captured scientific settings for this method; raw invalid edits remain in settings.json/run.json.",
    "display_name": "Captured image display name; duplicate names are permitted and never establish identity.",
    "accepted_checksum_sha256": "Checksum of the captured accepted binary mask array.",
    "accepted_labels_checksum_sha256": "Checksum of the captured native Modern labels array.",
    "source_checksum_sha256": "Checksum of original source pixels, including retained original dtype/channels.",
    "working_image_checksum_sha256": "Checksum of captured working-ROI pixels.",
    "units_per_px": "Captured physical calibration in unit_name per pixel.",
    "mm_per_px": "Captured physical calibration normalized to millimetres per pixel without display rounding.",
    "scale_px_per_mm": "Original configured group scale in pixels per millimetre.",
    "processing_roi": "JSON source-coordinate [x0,y0,x1,y1] working ROI with exclusive upper bounds.",
    "source_key": "Captured canonical Original source-image key, separate from project-image identity.",
    "source_slot": "Original resolution slot, coarse-to-fine; not a Modern size-bin ownership decision.",
    "input_order": "Captured Original source assignment order; not display/list sorting.",
    "preparation_retained": "Original component retained by the preparation minimum-diameter filter.",
    "phase_corrected_area_mm2": "Original group area1 from the public area result; not a Modern sampled-area denominator.",
    "border_corrected_area_mm2": "Original group area2 from the public area result; distinct from area1.",
    "equivalent_diameter_mm": "Original native equivalent component diameter in millimetres.",
    "orientation_project_deg": "Original public moment orientation under its project angle convention.",
    "orientation_deg": "Modern public ellipse-fit orientation in degrees, nullable if unavailable.",
    "count": "Native histogram or selected interval count; actual zero is not an unavailable value.",
    "stage": "Prepared or selected native distribution stage, or diagnostic processing stage in diagnostics.csv.",
    "na_per_mm2": "Public native N_A for this bin/label, using that method's captured normalization/eligible area.",
    "signed_nv_per_mm3": "Public signed inverse number density; negative values are retained without clipping.",
    "volume_fraction": "Original adjusted volume fraction for the selected row; not supplied by Modern density nesting.",
    "metric_id": "Explicit section.metric.reduction identifier; structural reason/classification metadata is never a separate metric.",
    "population": "Named source population, including configuration for deliberately exported setting values.",
    "unit": "Declared scientific unit for this metric value; availability/reason fields always have textual semantics.",
    "project_id": "Stable physical-sample project identity; all joins start here and with run_id.",
    "run_id": "Immutable captured request identity; never a row position or sample display name.",
    "method_id": "Native method namespace: legacy or modern. Different methods have incompatible populations/grids.",
    "image_id": "Stable captured project-image identity; duplicate display filenames are permitted.",
    "snapshot_id": "Accepted segmentation identity at execution, not the current accepted segmentation.",
    "object_id": "Native component/label identity scoped to project/run/method/image/snapshot.",
    "group_id": "Resolution-group identity scoped to run and method; used with group_images, not a pooled sample.",
    "row_ordinal": "Unique native distribution row order within method, including duplicate Original labels.",
    "shape_population": "Membership in summary_population_identities, not the threshold/histogram population.",
    "shape_threshold_population": "Membership in the source threshold_identities; may differ from shape summary membership.",
    "shape_histogram_population": "Membership in the source shape_histogram.population_identities.",
    "shape_included_metrics": "JSON list of shape metrics whose included_identities contain this native object.",
    "shape_excluded_metrics": "JSON list of shape metrics whose excluded_identities contain this native object.",
    "selected": "Original raw-selection membership; Modern density nesting supplies no analogous raw-object selection.",
    "value": "Unrounded public source value/reduction; unavailable is empty and qualified by availability/reason.",
    "availability": "Text status available/unavailable/not_projected; no numerical unit.",
    "reason": "Source-domain undefined/absence reason accompanying one scientific value; not a separate metric.",
    "status": "Original public statistic row status retained independently of per-value availability.",
    "kind": "result or setting; configured cutoffs/scales are explicitly distinguished from scientific reductions.",
    "included_identities": "JSON native [image_id,component_id] pairs for this metric, scoped by run/method; snapshot resolves via images.",
    "excluded_identities": "JSON native [image_id,component_id] pairs excluded by this metric's validity rules.",
    "area_mm2": "Native object area or distribution denominator in this table's row grain; not foreground-substituted eligible area.",
    "label_mm": "Original label-only diameter. No artificial interval edges are inferred.",
    "lower_mm": "Modern half-open interval lower edge, where supplied; absent for Original label-only rows.",
    "upper_mm": "Modern interval upper edge with the source final-edge convention; absent for Original label-only rows.",
    "plot_diameter_mm": "Public native geometric midpoint for Modern or Original label; retained for deterministic plotting.",
})


def column_definition(column):
    for suffix in ("_availability", "_reason"):
        if column.endswith(suffix):
            return {"unit": "text", "definition": "Text " + suffix[1:] + " for " + column[:-len(suffix)] + "; not a scientific value."}
    return {"unit": FIELD_UNITS[column], "definition": FIELD_DEFINITIONS[column]}


def foreign_keys(table):
    keys = []
    if table != "sample":
        keys.append({"columns": ["project_id", "run_id"], "table": "sample.csv", "references": ["project_id", "run_id"]})
    if table in ("groups", "group_images", "objects", "distributions", "summary_metrics", "diagnostics"):
        keys.append({"columns": ["project_id", "run_id", "method_id"], "table": "methods.csv", "references": ["project_id", "run_id", "method_id"]})
    if table in ("objects", "group_images"):
        keys.append({"columns": ["project_id", "run_id", "image_id", "snapshot_id"], "table": "images.csv", "references": ["project_id", "run_id", "image_id", "snapshot_id"]})
    if table in ("group_images", "distributions"):
        keys.append({"columns": ["project_id", "run_id", "method_id", "group_id"], "table": "groups.csv", "references": ["project_id", "run_id", "method_id", "group_id"]})
    return keys
