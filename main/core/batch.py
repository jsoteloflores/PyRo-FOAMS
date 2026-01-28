# main/core/batch.py
# Parallel batch processing utilities using ProcessPoolExecutor
# All functions are top-level and picklable for multiprocessing

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .processing import DEFAULTS, runSeparationPipeline
from .stereology import PoreProps, measure_labels

# Type aliases for clarity
ImageArray = np.ndarray  # np.uint8, shape (H,W) grayscale or (H,W,3) BGR
BinaryMask = np.ndarray  # np.uint8, shape (H,W), values in {0, 255}
LabelMap = np.ndarray  # np.int32, shape (H,W), 0=background, 1..N=objects
ScaleDict = Dict[str, Union[float, str]]  # e.g., {"unitsPerPx": 0.01, "unitName": "mm"}
ThreshParams = Dict[str, Any]  # parameter dict for thresholding
SepParams = Dict[str, Any]  # parameter dict for separation
MetaDict = Dict[str, Any]  # metadata dict returned from functions
ProgressCallback = Callable[[int, int], None]  # (completed, total) -> None


# ---------- Worker functions (must be top-level for pickle) ----------

def _process_single_image(
    img: ImageArray,
    thresh_params: ThreshParams,
    sep_params: SepParams,
    image_index: int = 0,
    scale: Optional[ScaleDict] = None,
    measure: bool = True
) -> Tuple[int, BinaryMask, Optional[LabelMap], MetaDict, Optional[List[PoreProps]]]:
    """
    Process a single image: threshold + separation + optional measurement.
    Returns (index, binary, labels, meta, props_or_None).

    This function is designed to be called in a worker process.
    """
    binary, labels, meta = runSeparationPipeline(img, thresh_params, sep_params)

    props = None
    if measure and labels is not None:
        props = measure_labels(labels, image_index=image_index, scale=scale)

    return (image_index, binary, labels, meta, props)


def _process_single_image_wrapper(args: Tuple) -> Tuple:
    """Unpacks args tuple for ProcessPoolExecutor.map()."""
    return _process_single_image(*args)


# ---------- Public API ----------

def process_batch_parallel(
    images: List[ImageArray],
    thresh_params: Optional[ThreshParams] = None,
    sep_params: Optional[SepParams] = None,
    scales: Optional[List[Optional[ScaleDict]]] = None,
    measure: bool = True,
    max_workers: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None
) -> Tuple[List[BinaryMask], List[Optional[LabelMap]], List[Optional[List[PoreProps]]]]:
    """
    Process multiple images in parallel using ProcessPoolExecutor.

    Parameters:
    -----------
    images : List[np.ndarray]
        List of grayscale or BGR images to process.
    thresh_params : Dict
        Thresholding parameters (see DEFAULTS["common"], DEFAULTS["otsu"], etc.)
    sep_params : Dict
        Separation parameters (see DEFAULTS["separation"])
    scales : List[Optional[Dict]]
        Per-image scale info, e.g. [{"unitsPerPx": 0.01, "unitName": "mm"}, ...]
    measure : bool
        If True, also compute per-pore measurements.
    max_workers : int, optional
        Max parallel workers. Defaults to min(cpu_count, len(images)).
    progress_callback : Callable[[int, int], None], optional
        Called with (completed_count, total_count) after each image finishes.

    Returns:
    --------
    binaries : List[np.ndarray]
        Binary masks (uint8, 0/255) for each image.
    labels_list : List[Optional[np.ndarray]]
        Label maps (int32) for each image, or None if separation disabled.
    props_list : List[Optional[List[PoreProps]]]
        Per-pore measurements for each image, or None if measure=False.
    """
    n = len(images)
    if n == 0:
        return [], [], []

    # Defaults
    if thresh_params is None:
        thresh_params = {"method": "otsu", "polarity": "auto"}
    if sep_params is None:
        sep_params = dict(DEFAULTS["separation"])
    if scales is None:
        scales = [None] * n

    # Determine worker count
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, n)
    max_workers = max(1, min(max_workers, n))

    # Build args for each image
    args_list = [
        (images[i], thresh_params, sep_params, i, scales[i] if i < len(scales) else None, measure)
        for i in range(n)
    ]

    # Results placeholders (maintain order)
    binaries: List[Optional[np.ndarray]] = [None] * n
    labels_list: List[Optional[np.ndarray]] = [None] * n
    props_list: List[Optional[List[PoreProps]]] = [None] * n

    completed = 0

    # For small batches or single image, skip multiprocessing overhead
    if n <= 2 or max_workers <= 1:
        for args in args_list:
            idx, binary, labels, meta, props = _process_single_image(*args)
            binaries[idx] = binary
            labels_list[idx] = labels
            props_list[idx] = props
            completed += 1
            if progress_callback:
                progress_callback(completed, n)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_single_image_wrapper, args): i
                for i, args in enumerate(args_list)
            }

            for future in as_completed(futures):
                try:
                    idx, binary, labels, meta, props = future.result()
                    binaries[idx] = binary
                    labels_list[idx] = labels
                    props_list[idx] = props
                except Exception as e:
                    # Log but continue with other images
                    print(f"Error processing image {futures[future]}: {e}")

                completed += 1
                if progress_callback:
                    progress_callback(completed, n)

    return binaries, labels_list, props_list  # type: ignore


def process_batch_sequential(
    images: List[ImageArray],
    thresh_params: Optional[ThreshParams] = None,
    sep_params: Optional[SepParams] = None,
    scales: Optional[List[Optional[ScaleDict]]] = None,
    measure: bool = True,
    progress_callback: Optional[ProgressCallback] = None
) -> Tuple[List[BinaryMask], List[Optional[LabelMap]], List[Optional[List[PoreProps]]]]:
    """
    Process multiple images sequentially (no multiprocessing).
    Same interface as process_batch_parallel for easy swapping.
    """
    return process_batch_parallel(
        images=images,
        thresh_params=thresh_params,
        sep_params=sep_params,
        scales=scales,
        measure=measure,
        max_workers=1,
        progress_callback=progress_callback
    )


# ---------- Convenience wrappers ----------

def threshold_batch(
    images: List[ImageArray],
    method: str = "otsu",
    polarity: str = "auto",
    **kwargs: Any
) -> List[BinaryMask]:
    """
    Apply thresholding to a batch of images (no separation).
    Returns list of binary masks.
    """
    thresh_params = {"method": method, "polarity": polarity, **kwargs}
    sep_params = {"method": "none"}
    binaries, _, _ = process_batch_parallel(
        images, thresh_params=thresh_params, sep_params=sep_params, measure=False
    )
    return binaries


def measure_batch(
    labels_list: List[Optional[LabelMap]],
    scales: Optional[List[Optional[ScaleDict]]] = None
) -> List[PoreProps]:
    """
    Measure all pores across a batch of label maps.
    Returns a flat list of PoreProps (all images combined).
    """
    from .stereology import measure_dataset
    return measure_dataset(labels_list, scales)
