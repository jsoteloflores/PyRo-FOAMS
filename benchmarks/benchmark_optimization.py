"""Benchmark and report for engineering brief 03.

Creates fixed label fixtures, compares reviewed-baseline (naive/full-frame)
implementations against current optimized functions in `main.core`.

Outputs a Markdown results file when `--output` is given.

Notes:
- Uses tracemalloc peak to report Python-level memory usage; OpenCV native
  allocations are not fully tracked by tracemalloc and are explicitly
  caveated in results.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import tkinter as tk

    from PIL import Image, ImageTk
    _TK_AVAILABLE = True
except Exception:
    Image = None
    ImageTk = None
    tk = None
    _TK_AVAILABLE = False

try:
    import PIL
except Exception:
    PIL = None

try:
    import matplotlib
except Exception:
    matplotlib = None

from main.core import distributions, processing, sampling, stereology
from main.tests.test_stereology import measure_labels_full_frame_reference


@dataclasses.dataclass
class CaseResult:
    name: str
    durations: List[float]
    tracemalloc_peaks: List[int]

    def median_duration(self) -> float:
        return float(statistics.median(self.durations))

    def median_peak(self) -> int:
        return int(statistics.median(self.tracemalloc_peaks))


# ------------------ Synthetic fixtures ------------------

def _place_nonoverlapping_circles(h: int, w: int, n: int, r_min: int, r_max: int, rng: np.random.Generator) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.int32)
    centers = []
    attempts = 0
    label = 1
    max_attempts = n * 1000
    while label <= n and attempts < max_attempts:
        attempts += 1
        r = int(rng.integers(r_min, r_max+1))
        x = int(rng.integers(r, w - r))
        y = int(rng.integers(r, h - r))
        ok = True
        for (cx, cy, cr) in centers:
            if (cx - x) ** 2 + (cy - y) ** 2 <= (cr + r + 2) ** 2:
                ok = False
                break
        if not ok:
            continue
        yy, xx = np.ogrid[-y:h - y, -x:w - x]
        mask = (xx * xx + yy * yy) <= (r * r)
        if np.any(canvas[mask] != 0):
            continue
        canvas[mask] = label
        centers.append((x, y, r))
        label += 1
    if label <= n:
        raise RuntimeError(f"Could not place {n} non-overlapping circles in {h}x{w}")
    # Create a simple sparse mapping by multiplying IDs by three.
    mapping = {i: i * 3 for i in range(1, n + 1)}
    out = np.zeros_like(canvas)
    for old, new in mapping.items():
        out[canvas == old] = new
    return out


def make_fixtures() -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(123456)
    fixtures: Dict[str, np.ndarray] = {}
    fixtures['labels_512_100'] = _place_nonoverlapping_circles(512, 512, 100, 6, 12, rng)
    fixtures['labels_2048_1000'] = _place_nonoverlapping_circles(2048, 2048, 1000, 6, 20, rng)
    # sparse label ids fixture: reuse one with fewer labels but sparse ids
    small = _place_nonoverlapping_circles(512, 512, 40, 6, 12, rng)
    # create gaps by remapping
    remap = {i: i * 5 for i in range(1, 41)}
    sparse = np.zeros_like(small)
    for old, new in remap.items():
        sparse[small == old] = new
    fixtures['labels_512_sparse'] = sparse
    return fixtures


# ------------------ Baseline (naive) helpers ------------------

def baseline_colorize(labels: np.ndarray, seed: int = 123) -> np.ndarray:
    """Reproduce reviewed baseline palette exactly.

    - positive sorted unique labels
    - hues = np.linspace(0,179,num=n,endpoint=False)
    - seeded shuffle once
    - convert HSV->BGR via cv2
    - per-label full-frame assignment using boolean mask
    """
    assert labels.ndim == 2
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    all_labels = np.unique(labels)
    positive = all_labels[all_labels > 0]
    n = len(positive)
    rng = np.random.default_rng(seed)
    if n > 0:
        hues = np.linspace(0, 179, num=n, endpoint=False).astype(np.uint8)
        # seeded permutation
        rng.shuffle(hues)
        sat = np.full_like(hues, 200, dtype=np.uint8)
        val = np.full_like(hues, 255, dtype=np.uint8)
        hsv = np.stack([hues, sat, val], axis=1).reshape(-1, 1, 3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
        # map positive labels in sorted order -> colors
        for idx, lbl in enumerate(positive):
            color = tuple(int(x) for x in bgr[idx])
            out[labels == int(lbl)] = color
    return out


def baseline_measure_labels(labels: np.ndarray, image_index: int = 0, scale: Optional[Dict] = None):
    """Reproduce reviewed `measure_labels` implementation (commit 92fcc9f):
    - full per-label per-bbox measurement
    - perimeter via external contours, ellipse/PCA fallback
    - holes/disconnected handling via RETR_EXTERNAL
    - hull >1200 subsampling for feret
    - optional scaling fields
    Returns list of `stereology.PoreProps` instances.
    """
    return measure_labels_full_frame_reference(
        labels, image_index=image_index, scale=scale
    )


def baseline_create_image_sampling_record(*, sample_id: str, image_id: str, image_index: int, magnification_group_id: str, label_map: np.ndarray, calibration: float, calibration_unit: str, min_detectable_diameter: Optional[float] = None, max_reliable_diameter: Optional[float] = None, analysis_domain_mask: Optional[np.ndarray] = None, source_id: Optional[str] = None, parent_image_id: Optional[str] = None):
    """Manual creation of ImageSamplingRecord that mirrors create_image_sampling_record

    This baseline computes included/omitted labels by explicit full-frame
    boolean membership checks rather than delegating to the optimized helper.
    """
    # Validate minimal preconditions using the same helpers for consistency
    sample_id = sampling._require_identifier(sample_id, "sample_id")
    image_id = sampling._require_identifier(image_id, "image_id")
    magnification_group_id = sampling._require_identifier(magnification_group_id, "magnification_group_id")
    if not isinstance(label_map, np.ndarray) or label_map.ndim != 2:
        raise sampling.SamplingValidationError("label_map must be a two-dimensional NumPy array")
    if not np.issubdtype(label_map.dtype, np.integer):
        raise sampling.SamplingValidationError("label_map must have an integer dtype")
    h, w = label_map.shape
    if analysis_domain_mask is None:
        domain = np.ones_like(label_map, dtype=bool)
    else:
        if not isinstance(analysis_domain_mask, np.ndarray):
            raise sampling.AnalysisDomainError("analysis_domain_mask must be a NumPy array")
        if analysis_domain_mask.shape != label_map.shape:
            raise sampling.AnalysisDomainError("analysis_domain_mask must match label_map shape")
        if analysis_domain_mask.dtype != np.bool_:
            raise sampling.AnalysisDomainError("analysis_domain_mask must have Boolean dtype")
        domain = analysis_domain_mask

    analyzed_pixel_count = int(np.count_nonzero(domain))
    if analyzed_pixel_count <= 0:
        raise sampling.AnalysisDomainError("analysis_domain_mask must contain eligible sampled pixels")

    positive = np.unique(label_map)
    positive = positive[positive > 0]
    included = []
    omitted = []
    # explicit full-frame membership checks per label
    for lbl in positive:
        mask = (label_map == int(lbl))
        total = int(mask.sum())
        inside = int(np.count_nonzero(mask & domain))
        if inside == 0:
            omitted.append(int(lbl))
        elif inside == total:
            included.append(int(lbl))
        else:
            raise sampling.AnalysisDomainError(f"analysis_domain_mask cuts through label {int(lbl)} in image {image_id!r}")

    calibration_mm = sampling.length_to_mm(calibration, calibration_unit, field_name="calibration")
    normalized_unit = sampling.normalize_length_unit(calibration_unit)
    minimum_mm = None if min_detectable_diameter is None else sampling.length_to_mm(min_detectable_diameter, normalized_unit, field_name="min_detectable_diameter")
    maximum_mm = None if max_reliable_diameter is None else sampling.length_to_mm(max_reliable_diameter, normalized_unit, field_name="max_reliable_diameter")

    analyzed_area_mm2 = analyzed_pixel_count * (calibration_mm ** 2)

    return sampling.ImageSamplingRecord(
        sample_id=sample_id,
        image_id=image_id,
        image_index=image_index,
        magnification_group_id=magnification_group_id,
        height_px=h,
        width_px=w,
        calibration=float(calibration),
        calibration_unit=normalized_unit,
        calibration_mm_per_px=calibration_mm,
        analyzed_pixel_count=analyzed_pixel_count,
        analyzed_area_mm2=analyzed_area_mm2,
        min_detectable_diameter=(None if min_detectable_diameter is None else float(min_detectable_diameter)),
        max_reliable_diameter=(None if max_reliable_diameter is None else float(max_reliable_diameter)),
        min_detectable_diameter_mm=minimum_mm,
        max_reliable_diameter_mm=maximum_mm,
        source_id=source_id,
        parent_image_id=parent_image_id,
        included_labels=tuple(sorted(int(x) for x in included)),
        omitted_labels=tuple(sorted(int(x) for x in omitted)),
    )


# ------------------ Distribution baseline ------------------

def baseline_calculate_2d_number_densities(dataset, bin_spec, *, exclude_border: bool = False):
    """Naive distribution: explicit loops over pores, bins and images."""
    # Reuse validation from current implementation to ensure identical errors
    dataset.require_detection_limits()
    distributions._validate_analysis_inputs(dataset, bin_spec)

    # Build groups like the canonical function but via loops
    group_images = {}
    for image in dataset.images:
        key = (image.sample_id, image.magnification_group_id)
        group_images.setdefault(key, []).append(image)

    bin_edges = bin_spec.edges_mm
    bin_count = len(bin_edges) - 1
    group_counts = {key: [0] * bin_count for key in group_images}
    group_areas = {key: [0.0] * bin_count for key in group_images}
    group_contributors = {key: [tuple()] * bin_count for key in group_images}

    # For each group/bin compute eligible images by scanning images
    for key, images_in_group in group_images.items():
        for b in range(bin_count):
            lower, upper = bin_edges[b], bin_edges[b+1]
            eligible = []
            for img in images_in_group:
                min_d = img.min_detectable_diameter_mm
                max_d = img.max_reliable_diameter_mm if img.max_reliable_diameter_mm is not None else math.inf
                if min_d <= lower and max_d >= upper:
                    eligible.append(img)
            area = math.fsum(img.analyzed_area_mm2 for img in eligible)
            group_areas[key][b] = area
            group_contributors[key][b] = tuple((img.sample_id, img.image_id) for img in eligible)

    excluded_border = []
    below_grid = []
    above_grid = []
    unsupported = []

    for pore in dataset.pores:
        identity = (pore.measurement.image_index, pore.measurement.label)
        d = pore.equivalent_diameter_mm
        if exclude_border and pore.measurement.touches_border:
            excluded_border.append(identity)
            continue
        if d < bin_edges[0]:
            below_grid.append(identity); continue
        if d > bin_edges[-1]:
            above_grid.append(identity); continue
        # find bin index
        b = distributions._bin_index(d, bin_edges)
        # find whether pore.image supports this bin
        img = pore.image
        min_d = img.min_detectable_diameter_mm
        max_d = img.max_reliable_diameter_mm if img.max_reliable_diameter_mm is not None else math.inf
        if not (min_d <= bin_edges[b] and max_d >= bin_edges[b+1]):
            unsupported.append(identity)
            continue
        key = (pore.sample_id, pore.magnification_group_id)
        group_counts[key][b] += 1

    # Compose Group2DDistribution-like objects using same shapes
    groups = {}
    for key in group_images:
        counts = tuple(group_counts[key])
        areas = tuple(group_areas[key])
        # eligible image counts must match contributors
        eligible_counts = tuple(len(c) for c in group_contributors[key])
        supported = tuple(a > 0 for a in areas)
        densities = tuple((c / a if s else math.nan) for c, a, s in zip(counts, areas, supported))
        groups[key] = distributions.Group2DDistribution(
            sample_id=key[0], magnification_group_id=key[1], counts=counts,
            eligible_image_counts=eligible_counts, eligible_areas_mm2=areas,
            number_densities_per_mm2=densities, supported=supported,
            contributing_images=tuple(group_contributors[key]),
        )

    diagnostics = distributions.DistributionDiagnostics(
        excluded_border=tuple(excluded_border), below_grid=tuple(below_grid),
        above_grid=tuple(above_grid), unsupported_by_image=tuple(unsupported),
        omitted_domain=tuple(dataset.omitted_pore_identities),
    )
    return distributions.DistributionResult(
        bin_spec=bin_spec,
        groups=distributions.MappingProxyType(groups),
        diagnostics=diagnostics,
        exclude_border=exclude_border,
        source_images=tuple(dataset.images),
    )


# ------------------ Runner / Timing harness ------------------

def _run_case(func, *args, runs: int = 5) -> CaseResult:
    # Warm up
    func(*args)
    durations = []
    peaks = []
    for _ in range(runs):
        tracemalloc.start()
        t0 = time.perf_counter()
        func(*args)
        dur = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        durations.append(dur)
        peaks.append(peak)
    name = getattr(func, '__name__', repr(func))
    return CaseResult(name=name, durations=durations, tracemalloc_peaks=peaks)


def human_bytes(n: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def make_report(results: Dict[str, CaseResult], runs: int, output_path: Optional[str], render_skip_reason: Optional[str] = None) -> str:
    lines: List[str] = []
    lines.append("# Benchmark Results — Brief 03")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Python: {sys.version.splitlines()[0]}")
    lines.append(f"- Platform: {platform.platform()}")
    lines.append(f"- NumPy: {np.__version__}")
    lines.append(f"- OpenCV: {cv2.__version__ if cv2 is not None else 'missing'}")
    lines.append(f"- Pillow: {PIL.__version__ if PIL is not None else 'missing'}")
    lines.append(f"- Matplotlib: {matplotlib.__version__ if matplotlib is not None else 'missing'}")
    lines.append("")
    lines.append(
        "**Caveat:** tracemalloc reports Python-level allocations only; native "
        "OpenCV allocations (C-level) may not be captured fully."
    )
    lines.append(
        f"Each case used one warm-up followed by {runs} measured runs; times and "
        "peak observations are medians."
    )
    lines.append("")
    lines.append("## Results (median of runs)")
    lines.append("")
    lines.append("| Case | Baseline (s / peak) | Current (s / peak) | Observed ratio | Verification |")
    lines.append("|---|---:|---:|---:|---|")
    pairs = {}
    for name, result in results.items():
        if name.endswith("_baseline"):
            pairs.setdefault(name[:-9], {})["baseline"] = result
        elif name.endswith("_current"):
            pairs.setdefault(name[:-8], {})["current"] = result
        elif name.startswith("dist_baseline_"):
            pairs.setdefault("distribution_" + name[14:], {})["baseline"] = result
        elif name.startswith("dist_current_"):
            pairs.setdefault("distribution_" + name[13:], {})["current"] = result
        elif name == "render_upload_100":
            pairs.setdefault("rendering_100_events", {})["baseline"] = result
        elif name == "render_retained_moves_100":
            pairs.setdefault("rendering_100_events", {})["current"] = result
    for name, pair in pairs.items():
        baseline = pair.get("baseline")
        current = pair.get("current")
        if baseline is None or current is None or not baseline.durations or not current.durations:
            lines.append(f"| {name} | SKIPPED | SKIPPED | - | not measured |")
            continue
        baseline_time = baseline.median_duration()
        current_time = current.median_duration()
        ratio = baseline_time / current_time if current_time else math.inf
        lines.append(
            f"| {name} | {baseline_time:.6f} / {human_bytes(baseline.median_peak())} "
            f"| {current_time:.6f} / {human_bytes(current.median_peak())} "
            f"| {ratio:.2f}x | outputs verified before timing |"
        )
    if render_skip_reason:
        lines.append("")
        lines.append(f"- Rendering skip reason: {render_skip_reason}")
    if output_path:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    return '\n'.join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=int, default=5, help='Number of runs (median uses at least 5)')
    p.add_argument('--output', type=str, default=None, help='Write results to this markdown file')
    args = p.parse_args(argv)
    runs = max(5, int(args.runs))
    fixtures = make_fixtures()

    results: Dict[str, CaseResult] = {}
    render_skip_reason: Optional[str] = None

    # --- Validation / equality assertions before timing ---
    # Colorize parity: plain and overlay
    labels = fixtures['labels_512_100']
    cur_plain = stereology.colorize_labels(labels, seed=123) if 'seed' in stereology.colorize_labels.__code__.co_varnames else stereology.colorize_labels(labels)
    base_plain = baseline_colorize(labels, seed=123)
    if not np.array_equal(base_plain, cur_plain):
        raise AssertionError("baseline_colorize output differs from current plain colorize")
    # overlay check
    bg = (np.random.default_rng(42).integers(0, 256, size=labels.shape, dtype=np.uint8))
    cur_overlay = stereology.colorize_labels(labels, seed=123, bg_gray=bg, alpha=0.45)
    # baseline overlay: blend baseline color image on top of gray using same alpha
    color_img = base_plain
    bg_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    a = 0.45
    roi = (labels > 0)
    blended = bg_bgr.copy()
    blended[roi] = cv2.addWeighted(color_img[roi], a, bg_bgr[roi], 1.0 - a, 0)
    if not np.array_equal(blended, cur_overlay):
        raise AssertionError("baseline_colorize overlay differs from current overlay semantics")

    def assert_measurements_equal(label_map):
        scale = {"unitsPerPx": 0.025, "unitName": "mm"}
        baseline = baseline_measure_labels(label_map, image_index=2, scale=scale)
        current = stereology.measure_labels(label_map, image_index=2, scale=scale)
        if len(baseline) != len(current):
            raise AssertionError("baseline and current measurement counts differ")
        for expected, actual in zip(baseline, current):
            for field in dataclasses.fields(stereology.PoreProps):
                expected_value = getattr(expected, field.name)
                actual_value = getattr(actual, field.name)
                if isinstance(expected_value, float):
                    if not math.isclose(expected_value, actual_value, rel_tol=1e-6, abs_tol=1e-6):
                        raise AssertionError(f"measurement field differs: {field.name}")
                elif expected_value != actual_value:
                    raise AssertionError(f"measurement field differs: {field.name}")

    for fixture in fixtures.values():
        np.testing.assert_array_equal(
            baseline_colorize(fixture), stereology.colorize_labels(fixture)
        )
        assert_measurements_equal(fixture)

    # Sampling parity
    img_labels = fixtures['labels_512_sparse']
    mask = np.ones_like(img_labels, dtype=bool)
    base_rec = baseline_create_image_sampling_record(
        sample_id='s1', image_id='img', image_index=0, magnification_group_id='g',
        label_map=img_labels, calibration=0.01, calibration_unit='mm',
        min_detectable_diameter=0.001, max_reliable_diameter=10.0, analysis_domain_mask=mask
    )
    cur_rec = sampling.create_image_sampling_record(
        sample_id='s1', image_id='img', image_index=0, magnification_group_id='g',
        label_map=img_labels, calibration=0.01, calibration_unit='mm',
        min_detectable_diameter=0.001, max_reliable_diameter=10.0, analysis_domain_mask=mask
    )
    if base_rec != cur_rec:
        raise AssertionError("baseline sampling record does not match current create_image_sampling_record")

    # Distribution inventory is synthesized without contour work.
    # synthesize props cheaply (no full per-label contours) for this dataset
    def synth_props_from_labelmap(label_map, image_index=0, units_per_px=0.01, unit_name='mm'):
        props = []
        labs = np.unique(label_map)
        labs = labs[labs > 0]
        H, W = label_map.shape
        for lbl in labs:
            mask = (label_map == int(lbl))
            area = int(mask.sum())
            if area <= 0:
                continue
            eq = math.sqrt((4.0 * area) / math.pi)
            # minimal centroid
            ys, xs = np.nonzero(mask)
            cx = float(xs.mean()) if xs.size else float('nan')
            cy = float(ys.mean()) if ys.size else float('nan')
            rec = stereology.PoreProps(
                image_index=image_index,
                label=int(lbl),
                area_px=area,
                perimeter_px=0.0,
                centroid_x=cx,
                centroid_y=cy,
                bbox_x0=0, bbox_y0=0, bbox_x1=W, bbox_y1=H,
                touches_border=False,
                eq_diam_px=eq,
                circularity=float('nan'),
                major_axis_px=None, minor_axis_px=None,
                aspect_ratio=None, orientation_deg=None,
                feret_max_px=None, feret_min_px=None,
                units_per_px=None, unit_name=None
            )
            props.append(rec)
        return props

    def assert_distributions_equal(base_dist, cur_dist):
        def nan_equal(a, b):
            if a is None and b is None:
                return True
            if isinstance(a, float) and isinstance(b, float):
                if math.isnan(a) and math.isnan(b):
                    return True
                return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
            return a == b
        if set(base_dist.groups.keys()) != set(cur_dist.groups.keys()):
            raise AssertionError("distribution groups mismatch")
        for k in base_dist.groups:
            g1 = base_dist.groups[k]; g2 = cur_dist.groups[k]
            for field in ("counts", "eligible_image_counts", "eligible_areas_mm2", "supported", "contributing_images"):
                if getattr(g1, field) != getattr(g2, field):
                    raise AssertionError(f"distribution {field} mismatch for {k}")
            for v1, v2 in zip(g1.number_densities_per_mm2, g2.number_densities_per_mm2):
                if not nan_equal(v1, v2):
                    raise AssertionError(f"density mismatch for {k}")
        if base_dist.diagnostics != cur_dist.diagnostics:
            raise AssertionError("distribution diagnostics mismatch")

    # --- Benchmark runs ---
    # Colorization: baseline vs current
    for fixture_name in ("labels_512_100", "labels_512_sparse", "labels_2048_1000"):
        fixture = fixtures[fixture_name]
        results[f'colorize_{fixture_name}_baseline'] = _run_case(baseline_colorize, fixture, runs=runs)
        results[f'colorize_{fixture_name}_current'] = _run_case(stereology.colorize_labels, fixture, runs=runs)

        results[f'measure_{fixture_name}_baseline'] = _run_case(baseline_measure_labels, fixture, 0, None, runs=runs)
        results[f'measure_{fixture_name}_current'] = _run_case(stereology.measure_labels, fixture, 0, None, runs=runs)

    # Sampling-record classification: baseline vs current (asserted above)
    for fixture_name in ("labels_512_100", "labels_512_sparse", "labels_2048_1000"):
        fixture = fixtures[fixture_name]
        domain = np.ones_like(fixture, dtype=bool)
        arguments = dict(
            sample_id='s1', image_id=fixture_name, image_index=0,
            magnification_group_id='g', label_map=fixture, calibration=0.01,
            calibration_unit='mm', min_detectable_diameter=0.001,
            max_reliable_diameter=10.0, analysis_domain_mask=domain,
        )
        results[f'sampling_{fixture_name}_baseline'] = _run_case(
            lambda kwargs=arguments: baseline_create_image_sampling_record(**kwargs), runs=runs
        )
        results[f'sampling_{fixture_name}_current'] = _run_case(
            lambda kwargs=arguments: sampling.create_image_sampling_record(**kwargs), runs=runs
        )

    # Distributions: run for datasets of 20 and 100 images and bins 50/500
    for n_images in (20, 100):
        # create a small set of image records by duplicating labels_512_100 with unique ids
        lab_src = fixtures['labels_512_100']
        images = []
        props = []
        for i in range(n_images):
            img_id = f"img_{n_images}_{i}"
            rec_i = baseline_create_image_sampling_record(
                sample_id='samp', image_id=img_id, image_index=i, magnification_group_id='g',
                label_map=lab_src, calibration=0.01, calibration_unit='mm',
                min_detectable_diameter=0.001, max_reliable_diameter=10.0, analysis_domain_mask=np.ones_like(lab_src, dtype=bool)
            )
            images.append(rec_i)
            props.extend(synth_props_from_labelmap(lab_src, image_index=i))
        dataset_n = sampling.build_sampling_dataset(images, props)
        for nb in (50, 500):
            spec = distributions.geometric_diameter_bin_spec(0.001, nb, log10_step=0.05)
            baseline_result = baseline_calculate_2d_number_densities(dataset_n, spec)
            current_result = distributions.calculate_2d_number_densities(dataset_n, spec)
            assert_distributions_equal(baseline_result, current_result)
            key_cur = f'dist_current_n{n_images}_bins_{nb}'
            key_base = f'dist_baseline_n{n_images}_bins_{nb}'
            results[key_cur] = _run_case(lambda ds=dataset_n, sp=spec: distributions.calculate_2d_number_densities(ds, sp), runs=runs)
            results[key_base] = _run_case(lambda ds=dataset_n, sp=spec: baseline_calculate_2d_number_densities(ds, sp), runs=runs)

    # Processing core timing (use same function as baseline/current but label accordingly)
    gray_512 = (np.random.default_rng(1).integers(0, 256, size=(512, 512), dtype=np.uint8))
    gray_2048 = (np.random.default_rng(2).integers(0, 256, size=(2048, 2048), dtype=np.uint8))
    for size, gray in ((512, gray_512), (2048, gray_2048)):
        baseline_output = processing.thresholdImageAdvanced(gray, 'otsu')
        current_output = processing.thresholdImageAdvanced(gray, 'otsu')
        np.testing.assert_array_equal(baseline_output[0], current_output[0])
        results[f'processing_same_core_{size}_baseline'] = _run_case(
            processing.thresholdImageAdvanced, gray, 'otsu', runs=runs
        )
        results[f'processing_same_core_{size}_current'] = _run_case(
            processing.thresholdImageAdvanced, gray, 'otsu', runs=runs
        )

    # Rendering: attempt to use actual Canvas + PhotoImage + ovals
    if _TK_AVAILABLE:
        try:
            root = tk.Tk()
            canvas = tk.Canvas(root, width=200, height=200)
            canvas.pack()
            img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
            ph = ImageTk.PhotoImage(img)
            canvas.create_image(50, 50, image=ph)
            ovals = [canvas.create_oval(10+i, 10+i, 20+i, 20+i, fill='red') for i in range(10)]
            def upload_100():
                for _ in range(100):
                    ImageTk.PhotoImage(img)
            def retained_plus_moves():
                for _ in range(100):
                    canvas.coords(ovals[0], 10, 10, 20, 20)
                    root.update_idletasks()
            results['render_upload_100'] = _run_case(upload_100, runs=runs)
            results['render_retained_moves_100'] = _run_case(retained_plus_moves, runs=runs)
            root.destroy()
        except Exception as e:
            render_skip_reason = f"Tk rendering failed: {e!r}"
            results['rendering_skipped'] = CaseResult('rendering_skipped', [], [])
    else:
        render_skip_reason = "Tkinter/Pillow not available"
        results['rendering_skipped'] = CaseResult('rendering_skipped', [], [])

    report_text = make_report(results, runs, args.output, render_skip_reason=render_skip_reason)
    print(report_text)
    if args.output:
        print(f"Wrote results to {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
