"""Deterministic FOAMS 1.0.5 table projections and atomic TSV bundles."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import numpy as np

from .legacy_pipeline import (
    FOAMS105_AREA_OUTPUT_SOURCE_KEYS,
    FOAMS105_PIPELINE_SOURCE_COMMIT,
    FOAMS105_PIPELINE_SOURCE_REPOSITORY,
    Foams105AnalysisResult,
)
from .legacy_summaries import Foams105RealValue

FOAMS105_EXPORT_METHOD = "foams_1_0_5_reproducible_table_bundle_v1"
FOAMS105_EXPORT_SCHEMA_VERSION = "1"

NA_MAG_HEADERS = (
    "bins",
    "freq_mag1",
    "NA_mag1",
    "freq_mag2",
    "NA_mag2",
    "freq_mag3",
    "NA_mag3",
    "freq_mag4",
    "NA_mag4",
)
RAW_HEADERS = (
    "EqDi",
    "Area",
    "Perimeter",
    "Long-axis",
    "Short-axis",
    "Orientation",
    "Mag type",
)
CALC_HEADERS = (
    "bins",
    "NA",
    "NV",
    "Volume fraction",
    "Cumulative volume",
    "log(NV>L)",
    "logL",
    "Ln(n)",
)
AREA_HEADERS = ("image", "area mm2")
VES_HEADERS = (
    "im_ves%",
    "mag1_ves%",
    "magtwos_ves%",
    "magthrees_ves%",
    "magfours_ves%",
    "NV-derived ves%",
    "Mult Factor",
    "Density derived ves",
    "2D_int_ves",
)
MISC_HEADERS = (
    "#Objects measured",
    "#Images",
    "NV",
    "NVcorr",
    "low_cutoff",
    "high_cutoff",
    "Perc>",
    "<Perc<",
    "Perc<",
)
SHAPE_HEADERS = ("eccentricity", "elongation", "aspect ratio", "regularity")
STATS_HEADERS = ("", "N", "max", "min", "mean", "std dev", "med", "mode")


class Foams105ExportError(RuntimeError):
    """Raised when an export projection or atomic bundle write fails."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        context: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = MappingProxyType(dict(context or {}))


@dataclass(frozen=True)
class Foams105ExportDiagnostic:
    code: str
    table: Optional[str]
    row_index: Optional[int]
    column: Optional[str]
    reason: Optional[str]
    legacy_classification: Optional[str] = None
    context: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Foams105ExportTableProjection:
    filename: str
    headers: Tuple[str, ...]
    rows: Tuple[Tuple[object, ...], ...]
    status: str
    unavailable_reason: Optional[str] = None


@dataclass(frozen=True)
class Foams105ExportTablesResult:
    analysis_result: Foams105AnalysisResult
    tables: Tuple[Foams105ExportTableProjection, ...]
    diagnostics: Tuple[Foams105ExportDiagnostic, ...]
    complete: bool
    method: str = FOAMS105_EXPORT_METHOD
    schema_version: str = FOAMS105_EXPORT_SCHEMA_VERSION


@dataclass(frozen=True)
class Foams105ExportBundleResult:
    output_directory: str
    table_filenames: Tuple[str, ...]
    sidecar_filenames: Tuple[str, ...]
    table_statuses: Tuple[Tuple[str, str], ...]
    checksums_sha256: Tuple[Tuple[str, str], ...]
    complete: bool
    manifest_filename: str = "manifest.json"
    diagnostics_filename: str = "diagnostics.json"


def _diagnostic_for_cell(
    table: str,
    row_index: int,
    column: str,
    value: object,
) -> Optional[Foams105ExportDiagnostic]:
    if isinstance(value, Foams105RealValue) and value.value is None:
        return Foams105ExportDiagnostic(
            code="undefined_numeric_cell",
            table=table,
            row_index=row_index,
            column=column,
            reason=value.reason or "undefined",
            legacy_classification=value.legacy_classification,
            context=(("operands", repr(value.operands)),),
        )
    if value is None:
        return Foams105ExportDiagnostic(
            code="unavailable_numeric_cell",
            table=table,
            row_index=row_index,
            column=column,
            reason="source_metadata_unavailable",
        )
    return None


def _table(
    filename: str,
    headers: Tuple[str, ...],
    rows: Tuple[Tuple[object, ...], ...],
    *,
    status: str = "available",
    unavailable_reason: Optional[str] = None,
) -> Foams105ExportTableProjection:
    if status not in ("available", "partial", "unavailable"):
        raise Foams105ExportError("table status is invalid", code="invalid_table_status", context={"filename": filename})
    if status == "unavailable" and not unavailable_reason:
        raise Foams105ExportError("unavailable table requires a reason", code="missing_unavailable_reason", context={"filename": filename})
    if status != "unavailable" and any(len(row) != len(headers) for row in rows):
        raise Foams105ExportError(
            "table row width does not match its header",
            code="table_width_mismatch",
            context={"filename": filename, "header_width": len(headers)},
        )
    return Foams105ExportTableProjection(filename, headers, rows, status, unavailable_reason)


def _raw_row(row) -> Tuple[object, ...]:
    return (
        row.equivalent_diameter_mm,
        row.area_mm2,
        row.perimeter_mm,
        row.major_axis_mm,
        row.minor_axis_mm,
        row.orientation_project_deg,
        row.source_slot,
    )


def _area_out_projection(result: Foams105AnalysisResult, diagnostics: list) -> Foams105ExportTableProjection:
    images = [
        image
        for group in result.prepared_result.groups
        for image in group.images
        if image.image_input.area_output_selected
    ]
    key_order = {key: index for index, key in enumerate(FOAMS105_AREA_OUTPUT_SOURCE_KEYS)}
    images.sort(key=lambda image: key_order[image.image_input.source_image_key])
    nonzero = [image for image in images if image.image_area_result.phase_corrected_area_mm2 != 0.0]
    if len(nonzero) != len(images):
        diagnostics.append(
            Foams105ExportDiagnostic(
                code="area_out_source_index_length_mismatch",
                table="Area_out.txt",
                row_index=None,
                column=None,
                reason="source removes exact-zero areas but retains image numbers 1..num_im",
                context=(
                    ("selected_image_count", str(len(images))),
                    ("nonzero_area_count", str(len(nonzero))),
                    ("zero_area_image_ids", repr(tuple(image.image_input.image_id for image in images if image.image_area_result.phase_corrected_area_mm2 == 0.0))),
                ),
            )
        )
        return _table(
            "Area_out.txt",
            AREA_HEADERS,
            (),
            status="unavailable",
            unavailable_reason="area_count_does_not_match_source_image_indices",
        )
    rows = tuple(
        (index, image.image_area_result.phase_corrected_area_mm2)
        for index, image in enumerate(images, start=1)
    )
    return _table("Area_out.txt", AREA_HEADERS, rows)


def build_foams105_export_tables(analysis_result: Foams105AnalysisResult) -> Foams105ExportTablesResult:
    """Project stored stage results into the nine source table schemas."""

    if not isinstance(analysis_result, Foams105AnalysisResult):
        raise Foams105ExportError("analysis_result has the wrong type", code="invalid_analysis_result")
    diagnostics = []
    prepared = analysis_result.prepared_result
    groups_by_slot = {group.source_slot: group for group in prepared.groups}

    na_rows = []
    labels = prepared.label_preparation_result.bin_labels_mm
    for index, label in enumerate(labels):
        cells = [label]
        for slot in range(1, 5):
            group = groups_by_slot.get(slot)
            if group is None:
                cells.extend((0, 0.0))
            else:
                cells.append(group.histogram_result.counts[index])
                normalization = group.normalization_result
                cells.append(None if normalization is None else normalization.na_per_mm2[index])
        na_rows.append(tuple(cells))
    na_table = _table("NA_mag.txt", NA_MAG_HEADERS, tuple(na_rows))

    raw_table = _table(
        "Raw_data.txt",
        RAW_HEADERS,
        tuple(_raw_row(row) for row in analysis_result.raw_data_rows),
    )
    rawcutoffs_table = _table(
        "Rawcutoffs_data.txt",
        RAW_HEADERS,
        tuple(_raw_row(row) for row in analysis_result.rawcutoffs_data_rows),
    )

    conversion = analysis_result.conversion_result
    binned = analysis_result.binned_selection_result
    volume = analysis_result.volume_result
    summaries = analysis_result.distribution_summary_result.rows
    if not (
        len(conversion.bin_labels_mm)
        == len(binned.na_per_mm2)
        == len(volume.rows)
        == len(summaries)
    ):
        raise Foams105ExportError(
            "Calc_out stage vectors have inconsistent lengths",
            code="calc_out_length_mismatch",
            context={
                "conversion_rows": len(conversion.bin_labels_mm),
                "na_rows": len(binned.na_per_mm2),
                "volume_rows": len(volume.rows),
                "summary_rows": len(summaries),
            },
        )
    calc_rows = tuple(
        (
            conversion.bin_labels_mm[index],
            binned.na_per_mm2[index],
            conversion.signed_nv_per_mm3[index],
            volume.adjusted_volume_fractions[index],
            volume.cumulative_adjusted_volume_fractions[index],
            summaries[index].log10_cumulative_nv,
            summaries[index].log10_label,
            summaries[index].ln_number_density_per_width,
        )
        for index in range(len(conversion.bin_labels_mm))
    )
    calc_table = _table("Calc_out.txt", CALC_HEADERS, calc_rows)

    area_table = _area_out_projection(analysis_result, diagnostics)

    ves = analysis_result.vesicularity_result
    percentages_by_slot = {
        group.source_slot: ves.group_vesicularity_percent[index]
        for index, group in enumerate(ves.groups)
    }
    density_derived = prepared.source_metadata.density_derived_vesicularity_percent
    ves_status = "partial" if density_derived is None else "available"
    if density_derived is None:
        diagnostics.append(
            Foams105ExportDiagnostic(
                code="density_derived_vesicularity_missing",
                table="Ves_out.txt",
                row_index=0,
                column="Density derived ves",
                reason="required original source setting was not supplied",
            )
        )
    ves_row = (
        ves.mean_group_vesicularity_percent,
        percentages_by_slot.get(1, 0.0),
        percentages_by_slot.get(2, 0.0),
        percentages_by_slot.get(3, 0.0),
        percentages_by_slot.get(4, 0.0),
        volume.nv_derived_vesicularity_percent,
        volume.normalization_factor,
        density_derived,
        ves.integrated_vesicularity_percent,
    )
    ves_table = _table("Ves_out.txt", VES_HEADERS, (ves_row,), status=ves_status)

    category = analysis_result.size_category_result
    misc_row = (
        analysis_result.raw_selection_result.selected_count,
        analysis_result.image_count,
        analysis_result.distribution_summary_result.total_nv_per_mm3,
        analysis_result.distribution_summary_result.corrected_nv_per_mm3,
        category.lower_cutoff_mm,
        category.upper_cutoff_mm,
        category.upper_percentage,
        category.middle_percentage,
        category.lower_percentage,
    )
    misc_table = _table("Misc_out.txt", MISC_HEADERS, (misc_row,))

    shape_hist = analysis_result.shape_export_histogram_result
    shape_counts = (
        shape_hist.eccentricity.counts,
        shape_hist.elongation.counts,
        shape_hist.aspect_ratio.counts,
        shape_hist.regularity.counts,
    )
    if len({len(values) for values in shape_counts}) != 1:
        raise Foams105ExportError("Shape_out histogram vectors differ in length", code="shape_out_length_mismatch")
    shape_table = _table(
        "Shape_out.txt",
        SHAPE_HEADERS,
        tuple(tuple(values[index] for values in shape_counts) for index in range(len(shape_counts[0]))),
    )

    stats_rows = []
    statistic_labels = {
        "aspect_ratio": "asp_rat",
        "elongation": "elong",
        "regularity": "regularity",
        "eccentricity": "eccent",
    }
    for statistic in analysis_result.shape_statistics_result.statistic_rows:
        stats_rows.append(
            (
                statistic_labels[statistic.metric],
                statistic.n,
                statistic.display_maximum,
                statistic.display_minimum,
                statistic.display_mean,
                statistic.display_standard_deviation,
                statistic.display_median,
                statistic.display_mode,
            )
        )
    stats_table = _table("Stats_out.txt", STATS_HEADERS, tuple(stats_rows))

    tables = (
        na_table,
        raw_table,
        rawcutoffs_table,
        calc_table,
        area_table,
        ves_table,
        misc_table,
        shape_table,
        stats_table,
    )
    return Foams105ExportTablesResult(
        analysis_result=analysis_result,
        tables=tables,
        diagnostics=tuple(diagnostics),
        complete=all(table.status == "available" for table in tables),
    )




def _format_cell(
    value: object,
    *,
    table: str,
    row_index: int,
    column: str,
    diagnostics: list,
) -> str:
    if isinstance(value, Foams105RealValue):
        if value.value is None:
            diagnostics.append(
                Foams105ExportDiagnostic(
                    code="undefined_numeric_cell",
                    table=table,
                    row_index=row_index,
                    column=column,
                    reason=value.reason or "undefined",
                    legacy_classification=value.legacy_classification,
                    context=(("operands", repr(value.operands)),),
                )
            )
            return "NA"
        value = value.value
    if value is None:
        diagnostics.append(
            Foams105ExportDiagnostic(
                code="unavailable_numeric_cell",
                table=table,
                row_index=row_index,
                column=column,
                reason="source_metadata_unavailable",
            )
        )
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise Foams105ExportError(
                "table contains a nonfinite number without an explicit status",
                code="unclassified_nonfinite_cell",
                context={"table": table, "row_index": row_index, "column": column},
            )
        if table == "Stats_out.txt":
            return format(number, ".4f")
        return format(number, ".17g")
    if isinstance(value, str):
        if any(char in value for char in ("\t", "\r", "\n")):
            raise Foams105ExportError(
                "table text cell contains a TSV delimiter",
                code="invalid_text_cell",
                context={"table": table, "row_index": row_index, "column": column},
            )
        return value
    raise Foams105ExportError(
        "table contains an unsupported cell type",
        code="unsupported_table_cell",
        context={"table": table, "row_index": row_index, "column": column, "type": type(value).__name__},
    )


def _render_table(table: Foams105ExportTableProjection, diagnostics: list) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(table.headers)
    for row_index, row in enumerate(table.rows):
        writer.writerow(
            tuple(
                _format_cell(value, table=table.filename, row_index=row_index, column=table.headers[column_index], diagnostics=diagnostics)
                for column_index, value in enumerate(row)
            )
        )
    return stream.getvalue().encode("utf-8")


def _identity_sidecars(analysis: Foams105AnalysisResult) -> Tuple[Tuple[str, object], ...]:
    prepared = analysis.prepared_result
    return (
        (
            "Raw_data.identity.json",
            tuple(
                {
                    "output_index": row.output_index,
                    "source_slot": row.source_slot,
                    "group_id": row.group_id,
                    "source_object_index": row.source_object_index,
                    "image_id": row.image_id,
                    "component_id": row.component_id,
                }
                for row in analysis.raw_data_rows
            ),
        ),
        (
            "Rawcutoffs_data.identity.json",
            tuple(
                {
                    "output_index": row.output_index,
                    "source_slot": row.source_slot,
                    "group_id": row.group_id,
                    "source_object_index": row.source_object_index,
                    "image_id": row.image_id,
                    "component_id": row.component_id,
                }
                for row in analysis.rawcutoffs_data_rows
            ),
        ),
        (
            "Shape_out.identity.json",
            {
                "population_identities": analysis.shape_export_histogram_result.population_identities,
                "eccentricity_centers": analysis.shape_export_histogram_result.eccentricity.centers_or_edges,
                "elongation_centers": analysis.shape_export_histogram_result.elongation.centers_or_edges,
                "aspect_ratio_centers": analysis.shape_export_histogram_result.aspect_ratio.centers_or_edges,
                "regularity_edges": analysis.shape_export_histogram_result.regularity.centers_or_edges,
                "orientation_centers": analysis.shape_export_histogram_result.orientation_project_deg.centers_or_edges,
            },
        ),
        (
            "NA_mag.source_indices.json",
            tuple(
                {
                    "group_id": group_trace.group_id,
                    "source_slot": group_trace.source_slot,
                    "range_origin": analysis.binned_selection_result.range_origin,
                    "requested_range": group_trace.requested_lower_label_mm,
                    "requested_upper": group_trace.requested_upper_label_mm,
                    "selected_label_indices": group_trace.selected_label_indices,
                    "density_source_indices": group_trace.density_source_indices,
                    "source_slice_policy": group_trace.source_slice_policy,
                }
                for group_trace in analysis.binned_selection_result.group_traces
            ),
        ),
        (
            "input_identity.json",
            {
                "source_image_mapping": prepared.source_image_mapping,
                "images": tuple(
                    {
                        "image_id": image.image_input.image_id,
                        "source_image_key": image.image_input.source_image_key,
                        "group_id": group.group_id,
                        "source_slot": group.source_slot,
                        "dtype": image.image_input.segmented_image.dtype.str,
                        "shape": image.image_input.segmented_image.shape,
                        "sha256": image.input_sha256,
                        "area_output_selected": image.image_input.area_output_selected,
                        "area1_phase_corrected_mm2": image.image_area_result.phase_corrected_area_mm2,
                        "area2_border_corrected_mm2": image.image_area_result.border_corrected_area_mm2,
                    }
                    for group in prepared.groups
                    for image in group.images
                ),
            },
        ),
    )


def _plot_diagnostics(analysis: Foams105AnalysisResult) -> Tuple[Foams105ExportDiagnostic, ...]:
    diagnostics = []
    for slot_index, slot in enumerate(analysis.bulk_na_plot_data_result.slots_descending):
        for code in slot.diagnostics:
            diagnostics.append(
                Foams105ExportDiagnostic(
                    code=code,
                    table="bulk_NA_plot_data",
                    row_index=slot_index,
                    column="NA",
                    reason=(
                        "source slot 4 uses slot 3 counts and phase-corrected area1"
                        if code == "source_slot4_uses_slot3_counts_and_area"
                        else code
                    ),
                    context=(
                        ("source_slot", str(slot.source_slot)),
                        ("count_source_slot", str(slot.count_source_slot)),
                        ("denominator_source_slot", str(slot.denominator_source_slot)),
                        ("density_source_indices", repr(slot.density_source_indices)),
                    ),
                )
            )
    histogram_series = (
        analysis.shape_histogram_result.eccentricity,
        analysis.shape_histogram_result.elongation,
        analysis.shape_histogram_result.aspect_ratio,
        analysis.shape_histogram_result.regularity,
        analysis.shape_histogram_result.orientation_project_deg,
    )
    for series in histogram_series:
        for code in series.diagnostics:
            diagnostics.append(
                Foams105ExportDiagnostic(
                    code=code,
                    table="Shape_out.txt",
                    row_index=None,
                    column=series.field,
                    reason=code,
                    context=(
                        ("under_range_count", str(series.under_range_count)),
                        ("over_range_count", str(series.over_range_count)),
                        ("omitted_count", str(series.omitted_count)),
                    ),
                )
            )
    return tuple(diagnostics)


def write_foams105_export_bundle(
    analysis_result: Foams105AnalysisResult,
    output_directory: str | os.PathLike[str],
) -> Foams105ExportBundleResult:
    """Write reproducible TSV tables and sidecars into a fresh directory."""

    tables = build_foams105_export_tables(analysis_result)
    destination = Path(output_directory).expanduser().resolve()
    if destination.exists():
        raise Foams105ExportError(
            "output directory already exists; refusing file collisions",
            code="output_directory_exists",
            context={"output_directory": str(destination)},
        )
    parent = destination.parent
    temporary = None
    try:
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(parent)))
        serialization_diagnostics = list(tables.diagnostics)
        checksums = {}
        table_filenames = []
        table_statuses = []
        for table in tables.tables:
            table_statuses.append((table.filename, table.status))
            if table.status == "unavailable":
                continue
            payload = _render_table(table, serialization_diagnostics)
            _write_bytes(temporary / table.filename, payload)
            checksums[table.filename] = hashlib.sha256(payload).hexdigest()
            table_filenames.append(table.filename)

        sidecar_filenames = []
        for filename, content in _identity_sidecars(analysis_result):
            payload = _json_bytes(content)
            _write_bytes(temporary / filename, payload)
            checksums[filename] = hashlib.sha256(payload).hexdigest()
            sidecar_filenames.append(filename)

        plot_diagnostics = _plot_diagnostics(analysis_result)
        serialization_diagnostics.extend(plot_diagnostics)
        diagnostics_payload = _json_bytes(
            {
                "diagnostics": tuple(serialization_diagnostics),
                "pipeline_diagnostics": analysis_result.diagnostics,
            "vesicularity_diagnostics": analysis_result.vesicularity_result.diagnostics,
            "volume_diagnostics": analysis_result.volume_result.diagnostics,
            "distribution_summary_diagnostics": analysis_result.distribution_summary_result.diagnostics,
            "size_category_diagnostics": (),
                "unavailable_tables": tuple(
                    {"filename": table.filename, "reason": table.unavailable_reason}
                    for table in tables.tables
                    if table.status == "unavailable"
                ),
            }
        )
        _write_bytes(temporary / "diagnostics.json", diagnostics_payload)
        checksums["diagnostics.json"] = hashlib.sha256(diagnostics_payload).hexdigest()

        manifest = _build_manifest(
            analysis_result,
            tables,
            checksums,
            output_filenames=tuple(table_filenames + sidecar_filenames + ["diagnostics.json", "manifest.json"]),
        )
        manifest_payload = _json_bytes(manifest)
        _write_bytes(temporary / "manifest.json", manifest_payload)
        checksums["manifest.json"] = hashlib.sha256(manifest_payload).hexdigest()

        if destination.exists():
            raise Foams105ExportError(
                "output directory appeared during bundle creation",
                code="output_directory_exists",
                context={"output_directory": str(destination)},
            )
        os.rename(temporary, destination)
        temporary = None
        return Foams105ExportBundleResult(
            output_directory=str(destination),
            table_filenames=tuple(table_filenames),
            sidecar_filenames=tuple(sidecar_filenames),
            table_statuses=tuple(table_statuses),
            checksums_sha256=tuple(sorted(checksums.items())),
            complete=tables.complete,
        )
    except Foams105ExportError:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        raise
    except Exception as exc:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        raise Foams105ExportError(
            "export bundle write failed; temporary output was removed",
            code="bundle_write_failed",
            context={"exception_type": type(exc).__name__, "output_directory": str(destination)},
        ) from exc


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)


def _json_bytes(value: object) -> bytes:
    try:
        converted = _jsonable(value)
        return (json.dumps(converted, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    except Foams105ExportError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise Foams105ExportError(
            "JSON serialization failed",
            code="json_serialization_failed",
            context={"exception_type": type(exc).__name__},
        ) from exc


def _build_manifest(
    analysis: Foams105AnalysisResult,
    tables: Foams105ExportTablesResult,
    checksums: Mapping[str, str],
    *,
    output_filenames: Tuple[str, ...],
) -> Mapping[str, object]:
    prepared = analysis.prepared_result
    settings = {
        "preparation": prepared.preparation_settings,
        "analysis": analysis.analysis_settings,
        "source_metadata": prepared.source_metadata,
        "configured_groups": tuple(
            {
                "group_id": group.group_id,
                "source_slot": group.source_slot,
                "scale_px_per_mm": group.scale_px_per_mm,
                "image_ids": tuple(image.image_input.image_id for image in group.images),
            }
            for group in prepared.groups
        ),
        "applied_ranges": analysis.applied_ranges,
        "suggested_ranges": None if analysis.cutoff_suggestion_result is None else analysis.cutoff_suggestion_result.suggested_ranges,
    }
    return {
        "schema_version": FOAMS105_EXPORT_SCHEMA_VERSION,
        "method": FOAMS105_EXPORT_METHOD,
        "source": {
            "repository": FOAMS105_PIPELINE_SOURCE_REPOSITORY,
            "commit": FOAMS105_PIPELINE_SOURCE_COMMIT,
            "actual_pyro_revision": prepared.source_metadata.pyro_revision or "unknown",
        },
        "settings": settings,
        "source_image_keys_order": FOAMS105_AREA_OUTPUT_SOURCE_KEYS,
        "source_image_mapping": prepared.source_image_mapping,
        "input_images": tuple(
            {
                "image_id": image.image_input.image_id,
                "source_image_key": image.image_input.source_image_key,
                "group_id": group.group_id,
                "source_slot": group.source_slot,
                "dtype": image.image_input.segmented_image.dtype.str,
                "shape": image.image_input.segmented_image.shape,
                "sha256": image.input_sha256,
                "scale_px_per_mm": image.image_input.scale_px_per_mm,
                "pore_phase_value": image.image_input.pore_phase_value,
                "excluded_phase_values": image.image_input.excluded_phase_values,
                "excluded_min_area_mm2": image.image_input.excluded_min_area_mm2,
            }
            for group in prepared.groups
            for image in group.images
        ),
        "perimeter": {
            "profile": prepared.perimeter_profile,
            "reference_runtime": analysis.shape_result.perimeter_reference_runtime,
            "reference_toolbox": analysis.shape_result.perimeter_reference_toolbox,
            "evidence_version": analysis.shape_result.perimeter_evidence_version,
            "evidence_hash": analysis.shape_result.perimeter_evidence_hash,
        },
        "moment_verification_boundary": analysis.shape_result.moment_verification_level,
        "bulk_na_plot_data": {
            "plot_ready": analysis.bulk_na_plot_data_result.plot_ready,
            "source_x_areas_nonzero_mm2": analysis.bulk_na_plot_data_result.source_x_areas_nonzero_mm2,
            "source_y_values": analysis.bulk_na_plot_data_result.source_y_values,
            "diagnostics": analysis.bulk_na_plot_data_result.diagnostics,
            "slots_descending": analysis.bulk_na_plot_data_result.slots_descending,
        },
        "shape_histograms": tuple(
            {
                "field": series.field,
                "method": series.method,
                "grid_type": series.grid_type,
                "centers_or_edges": series.centers_or_edges,
                "counts": series.counts,
                "population_identities": series.population_identities,
                "included_identities": series.included_identities,
                "omitted_identities": series.omitted_identities,
                "under_range_identities": series.under_range_identities,
                "over_range_identities": series.over_range_identities,
                "diagnostics": series.diagnostics,
            }
            for series in (
                analysis.shape_histogram_result.eccentricity,
                analysis.shape_histogram_result.elongation,
                analysis.shape_histogram_result.aspect_ratio,
                analysis.shape_histogram_result.regularity,
                analysis.shape_histogram_result.orientation_project_deg,
            )
        ),
        "table_statuses": tuple({"filename": table.filename, "status": table.status, "reason": table.unavailable_reason} for table in tables.tables),
        "bundle_status": "complete" if tables.complete else "partial",
        "output_filenames": output_filenames,
        "output_sha256": dict(sorted(checksums.items())),
        "manifest_sha256_note": "manifest self-checksum is returned by the writer and is not embedded recursively",
    }


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise Foams105ExportError("manifest contains nonfinite data", code="nonfinite_manifest_value")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise Foams105ExportError(
        "manifest contains an unsupported value",
        code="unsupported_manifest_value",
        context={"type": type(value).__name__},
    )
