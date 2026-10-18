"""Benchmark the Brief 19 FOAMS workflow on deterministic synthetic masks."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import statistics
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


def _synthetic_input(image_size: int, component_count: int) -> Tuple[Foams105PipelineGroupInput, ...]:
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    radius = max(4, image_size // 128)
    columns = math.ceil(math.sqrt(component_count))
    rows = math.ceil(component_count / columns)
    x_values = np.linspace(radius + 2, image_size - radius - 3, columns, dtype=np.int32)
    y_values = np.linspace(radius + 2, image_size - radius - 3, rows, dtype=np.int32)
    for component_index in range(component_count):
        x = int(x_values[component_index % columns])
        y = int(y_values[component_index // columns])
        cv2.circle(image, (x, y), radius, 1, thickness=-1)
    image_input = Foams105SegmentedImageInput(
        image_id=f"bench-{image_size}-{component_count}",
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


def _settings(prepared) -> Foams105AnalysisSettings:
    labels = prepared.label_preparation_result.bin_labels_mm
    return Foams105AnalysisSettings(
        selection_mode="manual",
        manual_ranges=(Foams105BinnedRange("benchmark", labels[0], labels[-1]),),
        volume_normalization_mode="unnormalized",
        known_vesicularity_percent=None,
        size_lower_cutoff_mm=1e-6,
        size_upper_cutoff_mm=1e6,
        shape_minimum_diameter_px=0.0,
        shape_statistics_mode="initial_analysis",
    )


def _json_report(
    results: dict,
    workload_metadata: dict,
    repeats: int,
) -> str:
    payload = {
        "schema_version": 1,
        "benchmark": "brief19_foams105_workflow",
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "warmup_iterations": 1,
        "measured_iterations": repeats,
        "rss_method": "5ms process working-set sampling" if _rss_bytes() is not None else "unavailable",
        "workloads": workload_metadata,
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


def _markdown_report(results: dict, workload_metadata: dict, repeats: int) -> str:
    lines = [
        "# Brief 19 FOAMS Workflow Benchmarks",
        "",
        f"- Python: `{sys.version.splitlines()[0]}`",
        f"- Platform: `{platform.platform()}`",
        f"- Processor: `{platform.processor() or 'not reported'}`",
        f"- NumPy/OpenCV: `{np.__version__}` / `{cv2.__version__}`",
        f"- Iterations: 1 warm-up, {repeats} measured runs",
        "- Memory: sampled process working set at 5 ms intervals; short peaks may be missed.",
        "- Workloads are synthetic, non-overlapping circular pore masks, not the original MATLAB catalog images.",
        "",
        "| Workload | Operation | Median seconds | Range seconds | Median sampled RSS delta | Pixels | Components | Selected rows | Export bytes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for workload, operations in results.items():
        metadata = workload_metadata[workload]
        for operation, result in operations.items():
            low, high = result.range_seconds
            memory = result.median_peak_rss_delta_bytes
            memory_text = "unavailable" if memory is None else f"{memory / (1024 * 1024):.2f} MiB"
            lines.append(
                f"| {workload} | {operation} | {result.median_seconds:.6f} "
                f"| {low:.6f}-{high:.6f} | {memory_text} "
                f"| {metadata['pixels']} | {metadata['components']} | {metadata['selected_rows']} | {metadata['export_bytes']} |"
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
    workload_specs = ((512, 100), (1024, 500), (2048, 1000))
    results = {}
    workload_metadata = {}

    with tempfile.TemporaryDirectory(prefix="foams105-brief19-benchmark-") as temporary_root:
        root = Path(temporary_root)
        for image_size, component_count in workload_specs:
            workload = f"{image_size}x{image_size}_{component_count}components"
            groups = _synthetic_input(image_size, component_count)
            pixels = image_size * image_size
            prepared_results, prepared = _sample_operation(lambda: _prepare(groups), repeats)
            settings = _settings(prepared)

            def complete_analysis():
                fresh_prepared = _prepare(groups)
                return analyze_foams105_prepared(fresh_prepared, settings)

            full_results, analysis = _sample_operation(complete_analysis, repeats)
            reanalysis_results, reanalysis = _sample_operation(
                lambda: analyze_foams105_prepared(prepared, settings),
                repeats,
            )
            export_index = [0]

            def project_and_write():
                destination = root / f"{workload}-bundle-{export_index[0]}"
                export_index[0] += 1
                return write_foams105_export_bundle(reanalysis, destination)

            export_results, bundle = _sample_operation(project_and_write, repeats)
            exports = build_foams105_export_tables(reanalysis)
            export_bytes = sum(path.stat().st_size for path in Path(bundle.output_directory).iterdir() if path.is_file())
            workload_metadata[workload] = {
                "pixels": pixels,
                "components": len(prepared.groups[0].unfiltered_pore_objects),
                "selected_rows": reanalysis.raw_selection_result.selected_count,
                "export_bytes": export_bytes,
                "export_table_count": len(exports.tables),
            }
            results[workload] = {
                "fresh_prepare": prepared_results,
                "complete_analysis_prepare_plus_reanalysis": full_results,
                "cached_cutoff_normalization_shape_reanalysis": reanalysis_results,
                "export_projection_and_write": export_results,
            }

    markdown = _markdown_report(results, workload_metadata, repeats)
    json_path = args.output.with_suffix(".json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8", newline="\n")
    json_path.write_text(_json_report(results, workload_metadata, repeats), encoding="utf-8", newline="\n")
    print(markdown, end="")
    print("JSON samples:", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
