# PyRo-FOAMS AI Agent Instructions

## Project Overview
PyRo-FOAMS is a Python-based image analysis toolkit for pore/foam analysis. It implements a three-stage image processing pipeline with GUI front-ends and stereological measurement capabilities, primarily using OpenCV, NumPy, and Tkinter.

**Three-Stage Pipeline:**
1. **Preprocessing** ([preprocessinggui.py](../main/preprocessinggui.py)): Image loading, cropping, batch operations
2. **Processing** ([processing_gui.py](../main/processing_gui.py)): Thresholding (Otsu, adaptive, percentile, eyedropper) + separation (watershed)
3. **Stereology/Postprocessing** ([stereology_gui.py](../main/stereology_gui.py), [postprocessing_gui.py](../main/postprocessing_gui.py)): Per-pore measurements, CSV export, visualization

## Architecture Patterns

### Logic Layer / Core Algorithms
**Separation of concerns**: GUI modules import non-GUI logic modules only:
- [preprocessing.py](../main/preprocessing.py) – Image I/O, crop utilities (rect/margin conversions)
- [processing.py](../main/processing.py) – Thresholding and watershed separation; contains `DEFAULTS` dict with all config params
- [stereology.py](../main/stereology.py) – Pure data model (`PoreProps` dataclass) and measurement functions; no image dependencies

### GUI Architecture
All GUIs use `tkinter` with a two-panel layout (canvas display + controls):
- GUI modules handle canvas rendering, event bindings, and state management
- Canvas images use `PIL.Image` + `ImageTk` for display
- Image color conversions: BGR (OpenCV) ↔ RGB (PIL) handled in utility functions (`_to_pil_disp()`, `_np_to_pil()`)

### Data Representations
- **Images**: `np.ndarray` (uint8); grayscale (H,W) or color (H,W,3) in BGR format
- **Rects**: `(x0, y0, x1, y1)` – inclusive-exclusive style (x1, y1 are exclusive bounds)
- **Masks**: Boolean arrays or uint8 (auto-converted by `_ensure_bool_mask()`)
- **Labels**: uint16/uint32 uint for separated objects; 0 = background

## Key Design Decisions

### Pillow Compatibility Shim
Handle both legacy and modern Pillow versions for resampling:
```python
if hasattr(Image, "Resampling"):
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", Image.BICUBIC)
```

### Image Loading Strategy
[preprocessing.py#loadImage()](../main/preprocessing.py#L17) handles conversion to uint8; grayscale-by-default for processing:
- Flag: `cv2.IMREAD_GRAYSCALE` (default) or `cv2.IMREAD_COLOR`
- Normalization applied if dtype ≠ uint8

### Default Configuration
[processing.py#DEFAULTS](../main/processing.py#L8) centralizes all tunable params (CLAHE, median blur, threshold methods, watershed settings). GUI modules read from this dict to populate controls.

### Rect ↔ Margin Conversions
[preprocessing.py](../main/preprocessing.py) provides bidirectional conversion:
- `clampRectToImage(rect, shape)` → clipped rect or None if degenerate
- `rectToMargins(rect, shape)` → (left, top, right, bottom)

## Testing Strategy
Tests in [unitTest/](../main/unitTest/) use standard `unittest` framework:
- `sys.path.append()` to import from parent `main/` directory
- Placeholder test structure in [testBinning.py](../main/unitTest/testBinning.py) shows pattern

## Dependencies & Imports
**Core**: cv2, numpy, tkinter, PIL, matplotlib (stereology_gui only)  
**Within-package imports**: GUIs import logic modules (preprocessing, processing, stereology); logic modules may import sibling logic modules only

## Conventions
- Use `from __future__ import annotations` for forward-ref type hints
- Type hints: `from typing import Dict, Tuple, List, Optional, Iterable`
- Private utility functions prefixed with `_` (e.g., `_ensure_bool_mask()`)
- GUI event handlers often named `on_<action>()` or `_on_<action>()`
- Constants (colors, limits) defined at module top: `OverlayColor = (255, 80, 255)`

## Common Workflows
- **Adding a threshold method**: Add to `DEFAULTS` dict, implement in [processing.py#thresholdImageAdvanced()](../main/processing.py#L61)
- **Adding a measurement**: Add field to [stereology.PoreProps](../main/stereology.py#L14) dataclass, compute in measurement function
- **Batch processing**: Use [preprocessing.py](../main/preprocessing.py) crop utilities with loop over image list
- **Visualization**: Use `labelsToColor()` from [processing.py](../main/processing.py) for colorized label display

## Notes for Agents
- **Maintain BGR convention:** OpenCV <> PIL conversions must be explicit. See `_npToPil()` / `_to_pil_disp()` in GUI modules.
- **Canvas updates:** Call `update_idletasks()` after image/state changes and debounce heavy redraws.
- **Labels vs masks:** Segmentation `labels` are int32 label maps (0=background). Masks for OpenCV ops must be uint8 with values {0,255} — never bool.
- **Test both grayscale and color flows:** Many helpers accept either; prefer calling `_prepGray`/`_prep_gray` where present.

## Canonical Data Contracts (important)
- `image_gray`: `np.uint8`, shape (H, W)
- `image_bgr`: `np.uint8`, shape (H, W, 3) — OpenCV BGR ordering
- `mask_binary`: `np.uint8`, shape (H, W), values in {0,255} (use `mask.astype(np.uint8)*255` to coerce)
- `labels`: `np.int32`, shape (H, W), background `0`, objects `1..N`. Watershed boundaries may be `-1` until normalized.

## Watershed & OpenCV contracts
- Input image for `cv2.watershed` must be 8-bit 3-channel (`uint8`) and the marker map must be 32-bit single-channel (`int32`).
- After `cv2.watershed`, treat any negative labels (e.g. `-1`) as boundaries; map to `0` for downstream consumers unless you explicitly want boundary pixels preserved.

## Performance & Optimization (practical)
- Prefer vectorized operations and OpenCV C APIs: use `connectedComponentsWithStats` and `np.bincount` instead of Python loops over `np.unique(labels)`.
- Colorize labels by LUT (palette[labels]) or a single vectorized gather rather than per-label Python loops.
- For per-label measurements, compute ROI (bounding box) and run contour/Feret calculations on the ROI only.
- Debounce UI-triggered recompute (parameter traces, `<Configure>`) — use `after`/`after_idle` and cancel pending callbacks to avoid recompute storms.

## Refactor tags / immediate checklist for agents
- ✅ R_dtype_contracts: normalize/coerce dtypes at module boundaries (GUI ↔ core) — DONE (`ensure_mask_uint8`, `ensure_labels_int32` in widgets.py).
- ✅ R_debounce_ui: centralize a `debounce` helper for GUI events — DONE (`widgets.py`, applied to all `<Configure>` bindings).
- ✅ R_vectorize_labels: replace per-label full-frame scans with CC stats / `np.bincount`/LUTs — DONE (`removeSmallAreas`, `clearBorderTouching`, `postSeparateCleanup` in processing.py).
- ✅ R_split_ui_core: move compute-heavy functions into pure callables under `core/` — DONE. New structure:
  - `main/core/__init__.py` — package init, re-exports all public APIs
  - `main/core/processing.py` — thresholding, cleanup, watershed, pipeline
  - `main/core/stereology.py` — PoreProps, measurements, colorization, CSV export
  - `main/core/preprocessing.py` — image I/O, crop utilities
  - Original `main/*.py` files are now thin re-export shims for backward compatibility.
- ✅ R_parallel_batch_ready: ensure per-image pipeline functions are top-level and picklable for `ProcessPoolExecutor` — DONE. New module:
  - `main/core/batch.py` — `process_batch_parallel()`, `process_batch_sequential()`, `threshold_batch()`, `measure_batch()`
  - Worker function `_process_single_image()` is top-level and picklable
  - Supports `max_workers` and optional `progress_callback` for UI integration

## Testing, CI & Packaging
- Tests live under `main/unitTest/` using `unittest`. New test files follow `test_*.py` naming.
- **Test coverage**: 118 tests across 5 files:
  - `test_processing.py` — thresholding, cleanup, watershed, separation (31 tests)
  - `test_stereology.py` — measurements, colorization, CSV export (19 tests)
  - `test_widgets.py` — dtype helpers, debounce (14 tests)
  - `test_batch.py` — parallel batch processing (28 tests)
  - `test_preprocessing.py` — image I/O, crop utilities (26 tests)
- CI: GitHub Actions workflow at `.github/workflows/ci.yml` runs lint (Ruff) + tests on Linux and Windows.
- Packaging: `pyproject.toml` at repo root defines the package. Install locally with `pip install -e .`
- Entry point: `python -m main` launches the preprocessing GUI. Also available as `pyro-foams` command after install.

## Quick how-to (run locally)
```bash
# Install in editable mode
pip install -e .

# Or run directly without install
cd main && python preprocessinggui.py

# Run as module
python -m main

# Run unit tests
python -m unittest discover -v main/unitTest -p "test_*.py"
```

## When to call an agent
- Ask the agent to implement one focused change at a time: "Normalize mask dtype in `postprocessing_gui.py` and add tests" or "Introduce debounce helper and apply to `<Configure>` handlers".

---
If anything here is unclear or you want a different emphasis (packaging, profiling harness, or a starter refactor PR), tell me which item to prioritize next.

## Debounce & UI-threading (practical example)
GUI responsiveness is critical; heavy compute must not run on Tk's main thread. Use a debounce helper to collapse frequent events (resize, param traces) into a single scheduled work item:

```python
def debounce(widget, delay_ms=200):
        """Return a decorator that schedules the wrapped function after `delay_ms`.
        Use the returned wrapper to replace direct event callbacks.
        """
        def _decorator(fn):
                timer_id_name = f"_debounce_{id(fn)}"
                def _wrapper(*args, **kwargs):
                        prev = getattr(widget, timer_id_name, None)
                        if prev is not None:
                                widget.after_cancel(prev)
                        setattr(widget, timer_id_name, widget.after(delay_ms, lambda: fn(*args, **kwargs)))
                return _wrapper
        return _decorator

# usage in a <Configure> handler:
# canvas.bind("<Configure>", debounce(self.canvas, 150)(_on_resize))
```

When doing batch or long-running work, spawn a worker (ProcessPoolExecutor or ThreadPoolExecutor) and show a non-blocking `BusyDialog` while keeping the UI responsive. Always catch and report exceptions back on the main thread (use `after` to marshal results).

## Watershed example (correct input/output contracts)
OpenCV `cv2.watershed` expects a 3-channel 8-bit image and a 32-bit marker map. Example pattern:

```python
# fg_mask: uint8 {0,255}
dist = cv2.distanceTransform((fg_mask>0).astype(np.uint8), cv2.DIST_L2, 5)
peaks = ... # find peaks as uint8 mask
num, markers = cv2.connectedComponents(peaks, connectivity=8)
markers = markers.astype(np.int32)
# ensure markers outside fg are 0
markers[fg_mask==0] = 0
ws_img = cv2.cvtColor((255 - ((dist/dist.max())*255).astype(np.uint8)), cv2.COLOR_GRAY2BGR)
cv2.watershed(ws_img, markers)
# markers may contain -1 for boundaries; convert to 0 and keep int32 labels
markers[markers < 0] = 0
```

## Minimal CI / test snippet
Add a GitHub Actions job to run linters and tests. The following is a minimal `ci.yml` job for reference (place under `.github/workflows/`):

```yaml
name: CI
on: [push, pull_request]
jobs:
    test:
        runs-on: ubuntu-latest
        steps:
            - uses: actions/checkout@v4
            - uses: actions/setup-python@v4
                with:
                    python-version: '3.11'
            - name: Install deps
                run: python -m pip install -r requirements.txt
            - name: Run tests
                run: python -m unittest discover -v main/unitTest
```

This repo currently uses `unittest` in `main/unitTest/`. Consider migrating to `pytest` and adding `ruff`/`black` steps once tests pass.
