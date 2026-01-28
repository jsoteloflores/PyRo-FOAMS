# PyRo-FOAMS AI Agent Instructions

## Project Overview
PyRo-FOAMS is a Python-based image analysis toolkit for pore/foam analysis. It implements a three-stage image processing pipeline with GUI front-ends and stereological measurement capabilities, primarily using OpenCV, NumPy, and Tkinter.

**Three-Stage Pipeline:**
1. **Preprocessing** ([gui/preprocessing.py](../main/gui/preprocessing.py)): Image loading, cropping, batch operations
2. **Processing** ([gui/processing.py](../main/gui/processing.py)): Thresholding (Otsu, adaptive, percentile, eyedropper) + separation (watershed)
3. **Stereology/Postprocessing** ([gui/stereology.py](../main/gui/stereology.py), [gui/postprocessing.py](../main/gui/postprocessing.py)): Per-pore measurements, CSV export, visualization

## Project Structure

```
main/
├── core/                     # Pure algorithms (no GUI dependencies)
│   ├── __init__.py           # Re-exports all public APIs
│   ├── batch.py              # Parallel batch processing
│   ├── preprocessing.py      # Image I/O, crop utilities
│   ├── processing.py         # Thresholding, watershed, cleanup
│   └── stereology.py         # Measurements, colorization, CSV
├── gui/                      # Tkinter-based user interfaces
│   ├── __init__.py           # GUI package init
│   ├── preprocessing.py      # Image loading, cropping GUI
│   ├── processing.py         # Thresholding + separation GUI
│   ├── stereology.py         # Measurements + histograms GUI
│   ├── postprocessing.py     # Mask editor GUI
│   └── widgets.py            # Shared utilities (debounce, BusyDialog)
├── tests/                    # Unit tests
│   ├── test_batch.py
│   ├── test_preprocessing.py
│   ├── test_processing.py
│   ├── test_stereology.py
│   └── test_widgets.py
├── __init__.py               # Package root with re-exports
└── __main__.py               # Entry point: python -m main
```

## Architecture Patterns

### Logic Layer / Core Algorithms
**Separation of concerns**: GUI modules import from `core/` only:
- `core/preprocessing.py` – Image I/O, crop utilities (rect/margin conversions)
- `core/processing.py` – Thresholding and watershed separation; contains `DEFAULTS` dict
- `core/stereology.py` – Pure data model (`PoreProps` dataclass) and measurement functions
- `core/batch.py` – Parallel batch processing with ProcessPoolExecutor

### GUI Architecture
All GUIs use `tkinter` with a two-panel layout (canvas display + controls):
- GUI modules handle canvas rendering, event bindings, and state management
- Canvas images use `PIL.Image` + `ImageTk` for display
- Image color conversions: BGR (OpenCV) ↔ RGB (PIL) handled in utility functions

### Data Representations
- **Images**: `np.ndarray` (uint8); grayscale (H,W) or color (H,W,3) in BGR format
- **Rects**: `(x0, y0, x1, y1)` – inclusive-exclusive style (x1, y1 are exclusive bounds)
- **Masks**: uint8 arrays with values {0, 255} (use `ensure_mask_uint8()`)
- **Labels**: int32 for separated objects; 0 = background (use `ensure_labels_int32()`)

## Key Design Decisions

### Import Pattern
All modules use relative imports within the package:
```python
from ..core.processing import thresholdImageAdvanced, DEFAULTS
from .widgets import debounce, ensure_mask_uint8
```

### Pillow Compatibility Shim
Handle both legacy and modern Pillow versions for resampling:
```python
if hasattr(Image, "Resampling"):
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", Image.BICUBIC)
```

### Default Configuration
`core/processing.py#DEFAULTS` centralizes all tunable params (CLAHE, median blur, threshold methods, watershed settings).

## Canonical Data Contracts
- `image_gray`: `np.uint8`, shape (H, W)
- `image_bgr`: `np.uint8`, shape (H, W, 3) — OpenCV BGR ordering
- `mask_binary`: `np.uint8`, shape (H, W), values in {0,255}
- `labels`: `np.int32`, shape (H, W), background `0`, objects `1..N`

## Testing, CI & Packaging
- Tests live under `main/tests/` using `unittest`. Test files follow `test_*.py` naming.
- **Test coverage**: 118 tests across 5 files
- CI: GitHub Actions workflow at `.github/workflows/ci.yml` runs lint (Ruff) + tests
- Packaging: `pyproject.toml` at repo root. Install with `pip install -e .`
- Entry point: `python -m main` launches the preprocessing GUI

## Quick how-to (run locally)
```bash
# Install in editable mode
pip install -e .

# Run as module
python -m main

# Run unit tests
cd main && python -m unittest discover -v tests -p "test_*.py"
```

## Common Workflows
- **Adding a threshold method**: Add to `DEFAULTS` dict, implement in `core/processing.py#thresholdImageAdvanced()`
- **Adding a measurement**: Add field to `core/stereology.PoreProps` dataclass, compute in measurement function
- **Batch processing**: Use `core/batch.py` functions for parallel processing
- **Visualization**: Use `colorize_labels()` from `core/stereology.py` for colorized label display

## Conventions
- Use `from __future__ import annotations` for forward-ref type hints
- Type hints: `from typing import Dict, Tuple, List, Optional, Iterable`
- Private utility functions prefixed with `_`
- GUI event handlers named `on_<action>()` or `_on_<action>()`

## Notes for Agents
- **Maintain BGR convention:** OpenCV ↔ PIL conversions must be explicit
- **Labels vs masks:** Segmentation `labels` are int32 label maps (0=background). Masks for OpenCV ops must be uint8 with values {0,255}
- **Use dtype helpers:** Always coerce dtypes at module boundaries with `ensure_mask_uint8()` and `ensure_labels_int32()`
- **Debounce UI events:** Use `debounce()` from `gui/widgets.py` for `<Configure>` and parameter trace callbacks
