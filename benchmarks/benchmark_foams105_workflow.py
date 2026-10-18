"""Benchmark the Brief 19 FOAMS workflow on deterministic synthetic masks."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2
import numpy as np

from main.core.legacy_exports import build_foams105_export_tables, write_foams105_export_bundle  # noqa: E402
from main.core.legacy_perimeter import FOAMS105_PERIMETER_PROFILE_CURRENT  # noqa: E402
from main.core.legacy_pipeline import (
    Foams105AnalysisSettings,
    Foams105PipelineGroupInput,
    Foams105PreparationSettings,
    Foams105SegmentedImageInput,
    Foams105SourceMetadata,
    analyze_foams105_prepared,
    prepare_foams105_analysis,
)  # noqa: E402
from main.core.legacy_selection import Foams105BinnedRange  # noqa: E402
from main.core.legacy_shape_statistics import summarize_foams105_shapes  # noqa: E402


@dataclass(frozen=True)
class OperationResult:
    samples_seconds: Tuple[float, ...]
    samples_peak_rss_delta_bytes: Tuple[Optional[int], ...]

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.samples_seconds)

    @property
    def range_seconds(self) -> Tuple[float, float]:
        return min(self.samples_seconds), max(self.samples_seconds)

    @property
    def median_peak_rss_delta_bytes(self) -> Optional[int]:
        values = [value for value in self.samples_peak_rss_delta_bytes if value is not None]
        return int(statistics.median(values)) if values else None


def _rss_bytes() -> Optional[int]:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("peak_paged_pool_usage", ctypes.c_size_t),
                ("paged_pool_usage", ctypes.c_size_t),
                ("peak_nonpaged_pool_usage", ctypes.c_size_t),
                ("nonpaged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        process_memory_info = psapi.GetProcessMemoryInfo
        process_memory_info.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        process_memory_info.restype = wintypes.BOOL
        success = process_memory_info(
            get_current_process(),
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.working_set_size) if success else None
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return None


def _sample_operation(function: Callable[[], object], repeats: int) -> Tuple[OperationResult, object]:
    function()
    durations = []
    rss_deltas = []
    last_result = None
    for _ in range(repeats):
        baseline_rss = _rss_bytes()
        peak_rss = [baseline_rss]
        stop = threading.Event()

        def sample_rss() -> None:
            while not stop.wait(0.005):
                current = _rss_bytes()
                if current is not None and (peak_rss[0] is None or current > peak_rss[0]):
                    peak_rss[0] = current

        monitor = threading.Thread(target=sample_rss, daemon=True)
        monitor.start()
        started = time.perf_counter()
        last_result = function()
        durations.append(time.perf_counter() - started)
        final_rss = _rss_bytes()
        stop.set()
        monitor.join()
        if final_rss is not None and (peak_rss[0] is None or final_rss > peak_rss[0]):
            peak_rss[0] = final_rss
        rss_deltas.append(
            None
            if baseline_rss is None or peak_rss[0] is None
            else max(0, peak_rss[0] - baseline_rss)
        )
    return OperationResult(tuple(durations), tuple(rss_deltas)), last_result


def _synthetic_input(
    image_size: int,
    component_count: int,
    *,
    shape_kind: str = "circles",
) -> Tuple[Foams105PipelineGroupInput, ...]:
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    if shape_kind == "sparse_l":
        start = image_size // 16
        end = image_size - start
        image[start, start:end] = 1
        image[start:end, start] = 1
    elif shape_kind == "circles":
        radius = max(4, image_size // 128)
        columns = math.ceil(math.sqrt(component_count))
        rows = math.ceil(component_count / columns)
        x_values = np.linspace(radius + 2, image_size - radius - 3, columns, dtype=np.int32)
        y_values = np.linspace(radius + 2, image_size - radius - 3, rows, dtype=np.int32)
        for component_index in range(component_count):
            x = int(x_values[component_index % columns])
            y = int(y_values[component_index // columns])
            cv2.circle(image, (x, y), radius, 1, thickness=-1)
    else:
        raise ValueError(f"unsupported benchmark shape kind: {shape_kind}")
    image_input = Foams105SegmentedImageInput(
        image_id=f"bench-{shape_kind}-{image_size}-{component_count}",
        source_image_key="mag1",
        segmented_image=image,
        scale_px_per_mm=1.0,
        pore_phase_value=1,
        excluded_phase_values=(None, None, None, None),
        excluded_min_area_mm2=(0.0, 0.0, 0.0, 0.0),
    )
    return (
        Foams105PipelineGroupInput(
            group_id="benchmark",
            source_slot=1,
            images=(image_input,),
        ),
    )


def _prepare(groups: Tuple[Foams105PipelineGroupInput, ...]):
    return prepare_foams105_analysis(
        groups,
        Foams105PreparationSettings(
            minimum_diameter_px=0.5,
            perimeter_profile=FOAMS105_PERIMETER_PROFILE_CURRENT,
        ),
        source_metadata=Foams105SourceMetadata(pyro_revision="brief19-benchmark"),
    )


def _settings(
    prepared,
    *,
    cutoff_indices: Tuple[int, int] = (0, 44),
    volume_mode: str = "unnormalized",
    shape_minimum_diameter_px: float = 0.0,
) -> Foams105AnalysisSettings:
    labels = prepared.label_preparation_result.bin_labels_mm
    return Foams105AnalysisSettings(
        selection_mode="manual",
        manual_ranges=(Foams105BinnedRange("benchmark", labels[cutoff_indices[0]], labels[cutoff_indices[1]]),),
        volume_normalization_mode=volume_mode,
        known_vesicularity_percent=None,
        size_lower_cutoff_mm=1e-6,
        size_upper_cutoff_mm=1e6,
        shape_minimum_diameter_px=shape_minimum_diameter_px,
        shape_statistics_mode="updated_minimum",
    )


def _candidate_identity() -> dict:
    repository_root = Path(__file__).resolve().parents[1]
    generated_paths = {
        "benchmarks/results_foams105_brief19.md",
        "benchmarks/results_foams105_brief19.json",
        "benchmarks/results_foams105_brief19_pre19a.md",
        "benchmarks/results_foams105_brief19_pre19a.json",
        "main/tests/fixtures/19a_foams105_reference.json.gz",
        "main/tests/fixtures/19a_foams105_reference.json.gz.manifest.json",
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
    )
    dirty_files = []
    excluded_files = []
    for line in status.stdout.splitlines():
        if len(line) < 4:
            continue
        state, name = line[:2], line[3:].strip()
        if name in generated_paths:
            excluded_files.append(name)
            continue
        path = repository_root / name
        dirty_files.append(
            {
                "status": state,
                "path": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            }
        )
    return {
        "commit": revision.stdout.strip() if revision.returncode == 0 else "unknown",
        "dirty": bool(status.stdout.strip()),
        "dirty_files": dirty_files,
        "generated_evidence_excluded": sorted(excluded_files),
    }


def _json_report(
    results: dict,
    workload_metadata: dict,
    repeats: int,
    workload_series: dict,
    operation_settings: dict,
) -> str:
    pre_change = _pre19a_comparison(results)
    payload = {
        "schema_version": 1,
        "benchmark": "brief19_foams105_workflow",
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "candidate": _candidate_identity(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "warmup_iterations": 1,
        "measured_iterations": repeats,
        "rss_method": "5ms process working-set sampling" if _rss_bytes() is not None else "unavailable",
        "workload_series": workload_series,
        "workloads": workload_metadata,
        "operation_settings": operation_settings,
        "pre_19a_comparison": pre_change,
        "operations": {
            f"{workload}/{operation}": {
                "samples_seconds": result.samples_seconds,
                "median_seconds": result.median_seconds,
                "range_seconds": result.range_seconds,
                "samples_peak_rss_delta_bytes": result.samples_peak_rss_delta_bytes,
                "median_peak_rss_delta_bytes": result.median_peak_rss_delta_bytes,
            }
            for workload, operations in results.items()
            for operation, result in operations.items()
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _pre19a_comparison(results: dict) -> dict:
    baseline_path = Path(__file__).with_name("results_foams105_brief19_pre19a.json")
    if not baseline_path.is_file():
        return {"status": "missing_preserved_pre19a_baseline"}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    comparison = {}
    operations = ("fresh_prepare", "complete_analysis_prepare_plus_reanalysis")
    for workload in ("512x512_100components", "1024x1024_500components", "2048x2048_1000components"):
        if workload not in results:
            continue
        for operation in operations:
            baseline_record = baseline["operations"].get(f"{workload}/{operation}")
            current_record = results[workload][operation]
            if baseline_record is None:
                continue
            before = float(baseline_record["median_seconds"])
            after = float(current_record.median_seconds)
            comparison[f"{workload}/{operation}"] = {
                "pre_19a_median_seconds": before,
                "bounded_median_seconds": after,
                "relative_time_change": (after - before) / before if before else None,
                "pre_19a_samples_seconds": baseline_record["samples_seconds"],
                "bounded_samples_seconds": current_record.samples_seconds,
            }
    return comparison


def _markdown_report(results: dict, workload_metadata: dict, repeats: int, workload_series: dict, operation_settings: dict) -> str:
    candidate = _candidate_identity()
    pre_change = _pre19a_comparison(results)
    lines = [
        "# Brief 19 FOAMS Workflow Benchmarks",
        "",
        f"- Python: `{sys.version.splitlines()[0]}`",
        f"- Platform: `{platform.platform()}`",
        f"- Processor: `{platform.processor() or 'not reported'}`",
        f"- NumPy/OpenCV: `{np.__version__}` / `{cv2.__version__}`",
        f"- Candidate Git commit: `{candidate['commit']}`; dirty: `{candidate['dirty']}`; dirty file hashes: {len(candidate['dirty_files'])}",
        f"- Iterations: 1 warm-up, {repeats} measured runs",
        "- Memory: sampled process working set at 5 ms intervals; short peaks may be missed and the sampling thread contributes overhead.",
        "- Circle workloads are synthetic and non-overlapping; the sparse L workload has a large bounding box relative to foreground.",
        "- These workloads are not the original MATLAB catalog or real images.",
        f"- Series: `{json.dumps(workload_series, sort_keys=True)}`",
        "",
        "## Preserved Pre-19a Comparison",
        "",
        "| Workload / operation | Pre-19a median s | Bounded median s | Time change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, item in pre_change.items():
        if not isinstance(item, dict):
            continue
        lines.append(
            f"| {key} | {item['pre_19a_median_seconds']:.6f} | {item['bounded_median_seconds']:.6f} "
            f"| {item['relative_time_change'] * 100:+.1f}% |"
        )
    lines.extend(
        (
        "",
        "| Workload | Operation | Changed settings | Median seconds | Range seconds | Median sampled RSS delta | Pixels | Foreground | Components | Selected rows | Export bytes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for workload, operations in results.items():
        metadata = workload_metadata[workload]
        for operation, result in operations.items():
            low, high = result.range_seconds
            memory = result.median_peak_rss_delta_bytes
            memory_text = "unavailable" if memory is None else f"{memory / (1024 * 1024):.2f} MiB"
            changed = operation_settings[workload].get(operation, {})
            lines.append(
                f"| {workload} | {operation} | `{json.dumps(changed, sort_keys=True)}` | {result.median_seconds:.6f} "
                f"| {low:.6f}-{high:.6f} | {memory_text} "
                f"| {metadata['pixels']} | {metadata['foreground_pixels']} | {metadata['components']} "
                f"| {metadata['selected_rows']} | {metadata['export_bytes']} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="Measured iterations per operation; at least five are required.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results_foams105_brief19.md"),
        help="Markdown output path; a sibling .json file records raw timings.",
    )
    args = parser.parse_args(argv)
    repeats = max(5, args.runs)
    workload_specs = (
        ("512x512_100components", 512, 100, "circles"),
        ("1024x1024_100components", 1024, 100, "circles"),
        ("2048x2048_100components", 2048, 100, "circles"),
        ("2048x2048_1000components", 2048, 1000, "circles"),
        ("1024x1024_500components", 1024, 500, "circles"),
        ("1024x1024_1000components", 1024, 1000, "circles"),
        ("1024x1024_sparse_l", 1024, 1, "sparse_l"),
    )
    workload_series = {
        "component_count_fixed_1024px": (
            "1024x1024_100components",
            "1024x1024_500components",
            "1024x1024_1000components",
        ),
        "image_size_fixed_100_components_approximately_fixed_density": (
            "512x512_100components",
            "1024x1024_100components",
            "2048x2048_100components",
        ),
        "pre_19a_comparison_2048px_1000_components": ("2048x2048_1000components",),
        "sparse_long_shape": ("1024x1024_sparse_l",),
    }
    results = {}
    workload_metadata = {}
    operation_settings = {}

    with tempfile.TemporaryDirectory(prefix="foams105-brief19-benchmark-") as temporary_root:
        root = Path(temporary_root)
        for workload, image_size, requested_components, shape_kind in workload_specs:
            groups = _synthetic_input(image_size, requested_components, shape_kind=shape_kind)
            source_image = groups[0].images[0].segmented_image
            foreground_pixels = int(np.count_nonzero(source_image == 1))
            pixels = image_size * image_size
            prepared_results, prepared = _sample_operation(lambda: _prepare(groups), repeats)
            baseline_settings = _settings(prepared)
            cutoff_indices = (20, 25)
            cutoff_settings = _settings(prepared, cutoff_indices=cutoff_indices)
            normalization_settings = _settings(prepared, volume_mode="integrated_2d")
            shape_minimum_px = max(1.0, image_size / 64.0)

            def complete_analysis():
                fresh_prepared = _prepare(groups)
                return analyze_foams105_prepared(fresh_prepared, _settings(fresh_prepared))

            full_results, analysis = _sample_operation(complete_analysis, repeats)
            unchanged_results, unchanged_analysis = _sample_operation(
                lambda: analyze_foams105_prepared(prepared, baseline_settings),
                repeats,
            )
            cutoff_results, cutoff_analysis = _sample_operation(
                lambda: analyze_foams105_prepared(prepared, cutoff_settings),
                repeats,
            )
            normalization_results, normalization_analysis = _sample_operation(
                lambda: analyze_foams105_prepared(prepared, normalization_settings),
                repeats,
            )
            configured_scales = tuple(group.scale_px_per_mm for group in prepared.groups)
            shape_results, shape_summary = _sample_operation(
                lambda: summarize_foams105_shapes(
                    unchanged_analysis.shape_result,
                    configured_scales_px_per_mm=configured_scales,
                    minimum_diameter_px=shape_minimum_px,
                    mode="updated_minimum",
                ),
                repeats,
            )
            export_index = [0]

            projection_results, projection = _sample_operation(
                lambda: build_foams105_export_tables(unchanged_analysis),
                repeats,
            )

            def write_prebuilt_projection():
                destination = root / f"{workload}-bundle-{export_index[0]}"
                export_index[0] += 1
                return write_foams105_export_bundle(
                    unchanged_analysis,
                    destination,
                    projection_result=projection,
                )

            write_results, bundle = _sample_operation(write_prebuilt_projection, repeats)
            export_bytes = sum(path.stat().st_size for path in Path(bundle.output_directory).iterdir() if path.is_file())
            rows, columns = np.nonzero(source_image == 1)
            bbox_area = 0 if rows.size == 0 else (int(rows.max()) - int(rows.min()) + 1) * (int(columns.max()) - int(columns.min()) + 1)
            workload_metadata[workload] = {
                "pixels": pixels,
                "foreground_pixels": foreground_pixels,
                "foreground_density": foreground_pixels / pixels,
                "bounding_box_pixels": bbox_area,
                "bbox_to_foreground_ratio": bbox_area / foreground_pixels if foreground_pixels else None,
                "components": len(prepared.groups[0].unfiltered_pore_objects),
                "selected_rows": unchanged_analysis.raw_selection_result.selected_count,
                "export_bytes": export_bytes,
                "export_table_count": len(projection.tables),
            }
            results[workload] = {
                "fresh_prepare": prepared_results,
                "complete_analysis_prepare_plus_reanalysis": full_results,
                "cached_unchanged_analysis": unchanged_results,
                "cached_cutoff_edit_reanalysis": cutoff_results,
                "cached_normalization_edit_reanalysis": normalization_results,
                "cached_shape_minimum_update": shape_results,
                "pure_table_projection": projection_results,
                "disk_write_of_prebuilt_projection": write_results,
            }
            labels = prepared.label_preparation_result.bin_labels_mm
            operation_settings[workload] = {
                "fresh_prepare": {"minimum_diameter_px": 0.5, "perimeter_profile": FOAMS105_PERIMETER_PROFILE_CURRENT},
                "complete_analysis_prepare_plus_reanalysis": {"selection_mode": "manual", "cutoff_indices": [0, 44], "volume_mode": "unnormalized", "shape_mode": "updated_minimum", "shape_minimum_diameter_px": 0.0},
                "cached_unchanged_analysis": {"selection_mode": "manual", "cutoff_indices": [0, 44], "volume_mode": "unnormalized", "shape_mode": "updated_minimum", "shape_minimum_diameter_px": 0.0},
                "cached_cutoff_edit_reanalysis": {"changed": "manual_ranges", "baseline_bin_indices": [0, 44], "edited_bin_indices": list(cutoff_indices), "edited_range_mm": [labels[cutoff_indices[0]], labels[cutoff_indices[1]]]},
                "cached_normalization_edit_reanalysis": {"changed": "volume_normalization_mode", "baseline": "unnormalized", "edited": "integrated_2d"},
                "cached_shape_minimum_update": {"changed": "shape_minimum_diameter_px", "baseline": 0.0, "edited": shape_minimum_px, "mode": shape_summary.mode, "operator": ">=", "api": "summarize_foams105_shapes", "pipeline_reanalysis": False},
                "pure_table_projection": {"operation": "build_foams105_export_tables", "disk_write": False},
                "disk_write_of_prebuilt_projection": {"operation": "write_foams105_export_bundle", "projection_prebuilt": True, "projection_recomputed": False},
            }

    markdown = _markdown_report(results, workload_metadata, repeats, workload_series, operation_settings)
    json_path = args.output.with_suffix(".json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8", newline="\n")
    json_path.write_text(
        _json_report(results, workload_metadata, repeats, workload_series, operation_settings),
        encoding="utf-8",
        newline="\n",
    )
    print(markdown, end="")
    print("JSON samples:", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
