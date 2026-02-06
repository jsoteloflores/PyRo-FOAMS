from __future__ import annotations

# preprocessing_gui.py
# Preprocessing GUI:
# - Load up to 20 images
# - Dynamic grid (5 columns x up to 4 rows), fills row-by-row
# - Thumbs auto-resize with window
# - Double-click a thumb -> Enlarged viewer with zoom/pan
# - Batch Crop: draw ROI in enlarged view -> confirm current/all (with relative margins)
# - Pixel Scale Calibration (Part 2):
#     * Calibrate mode: click first point, second point constrained to same vertical pixel
#     * Enter real-world length + units, apply to current or all
#     * Manual scale entry (units/px or px/unit)
#     * Per-image scale is displayed and stored
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

# Pillow resampling shim
if hasattr(Image, "Resampling"):  # Pillow >= 9.1 (incl. 10+)
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", Image.BICUBIC)

# Package imports
from ..core.preprocessing import applyCropBatch, clampRectToImage, cropWithRect
from . import postprocessing, processing  # sibling GUI modules
from .widgets import debounce, ensure_mask_uint8


def _tk_parent(owner) -> tk.Misc:
    """Return a valid Tk parent widget for dialogs."""
    # If owner is already a Tk widget
    if isinstance(owner, tk.Misc):
        try:
            return owner.winfo_toplevel()
        except Exception:
            pass
    # Try common attributes you may have stored
    for attr in ("root", "master", "parent"):
        p = getattr(owner, attr, None)
        if isinstance(p, tk.Misc):
            try:
                return p.winfo_toplevel()
            except Exception:
                return p
    # Fallback to the default root (created by main)
    root = tk._get_default_root()
    if root is None:
        raise RuntimeError("No Tk root available; create tk.Tk() before opening dialogs.")
    return root


MAX_IMAGES = 20
PREVIEW_MAX_PIXELS = 1_200_000  # ~1.2 MP for thumbnails/grids

class BusyDialog(tk.Toplevel):
    def __init__(self, parent, title="Working…", mode="indeterminate", maximum=100):
        if not isinstance(parent, tk.Misc):
            parent = tk._get_default_root()
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=title).pack(anchor="w", pady=(0,8))
        self.pb = ttk.Progressbar(frm, orient="horizontal", length=360,
                                  mode=mode, maximum=maximum)
        self.pb.pack(fill="x")
        if mode == "indeterminate":
            self.pb.start(10)
        self.update_idletasks()

    def set_progress(self, value):
        try:
            self.pb["value"] = value
            self.update_idletasks()
        except Exception:
            pass

    def close(self):
        try:
            if str(self.pb["mode"]) == "indeterminate":
                self.pb.stop()
        except Exception:
            pass
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

def safeReadImage(path: str) -> np.ndarray:
    """Robust image reader: handles Windows paths, 16-bit tiffs, alpha."""
    data = np.fromfile(path, dtype=np.uint8)         # avoids Unicode path issues
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.dtype == np.uint16:                       # scale 16-bit → 8-bit
        img = (img / 257.0).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:          # drop alpha
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] == 1:          # squeeze single-channel
        img = img[:, :, 0]
    return img

def makePreview(img: np.ndarray, max_pixels: int = PREVIEW_MAX_PIXELS) -> np.ndarray:
    """Downscale for preview so the grid stays snappy."""
    h, w = img.shape[:2]
    pix = h * w
    if pix <= max_pixels:
        return img
    s = (max_pixels / float(pix)) ** 0.5
    new_w = max(1, int(w * s))
    new_h = max(1, int(h * s))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
MAX_IMAGES = 20
GRID_COLS = 5
GRID_ROWS = 4  # capacity 20

Rect = Tuple[int, int, int, int]  # (x0,y0,x1,y1)


class PreprocessApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.root = master
        self.master.title("PyFOAMS – Preprocessing")
        self.images: list[np.ndarray] = []
        self.previews: list[np.ndarray] = []
        self.imagePaths: list[str] = []
        self.thumbPhotos: list[ImageTk.PhotoImage] = []  # keep refs so Tk doesn’t GC
        self.currentIndex = 0

        self.paths: List[str] = []
        self.selectedIndex: Optional[int] = None

        # Per-image scales: dict or None, e.g. {"unitsPerPx": 0.005, "unitName": "mm"}
        self.scales: List[Optional[Dict[str, float | str]]] = []

        self._thumbPhotos: Dict[int, ImageTk.PhotoImage] = {}
        self._thumbSizes: Dict[int, Tuple[int, int]] = {}  # cached sizes per cell
        self.masks: list[Optional[np.ndarray]] = []   # aligns 1:1 with self.images
        self.displayModeVar = tk.StringVar(value="original")  # "original" | "mask" | "overlay"


        self._buildUi()

    # --------------------------- UI scaffolding ---------------------------

    def _buildUi(self):
        # Toolbar
        self.toolbar = ttk.Frame(self.master)
        self.toolbar.pack(side="top", fill="x")

        ttk.Button(self.toolbar, text="Load Images", command=self.onLoad).pack(side="left", padx=6, pady=6)
        self.cropAllVar = tk.BooleanVar(value=True)  # default apply to all
        ttk.Button(self.toolbar, text="Batch Crop", command=self.onBatchCropClick).pack(side="left", padx=6)
        ttk.Checkbutton(self.toolbar, text="Use relative margins", variable=self.cropAllVar).pack(side="left", padx=6)
        ttk.Button(self.toolbar, text="Proceed with Processing", command=self.onProceedProcessing)\
   .pack(side="left", padx=12)
        ttk.Separator(self.toolbar, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Label(self.toolbar, text="Thumbs:").pack(side="left")
        for txt, val in [("Original", "original"), ("Mask", "mask"), ("Overlay", "overlay")]:
            ttk.Radiobutton(self.toolbar, text=txt, value=val, variable=self.displayModeVar, command=self._redrawAllThumbs).pack(side="left", padx=4)

        ttk.Button(self.toolbar, text="Edit Masks…", command=self.onOpenEditor).pack(side="left", padx=10)

        self.statusVar = tk.StringVar(value="No images loaded")
        ttk.Label(self.toolbar, textvariable=self.statusVar).pack(side="left", padx=12)

        # Grid frame
        self.gridFrame = ttk.Frame(self.master)
        self.gridFrame.pack(side="top", fill="both", expand=True)

        # Create a 5x4 grid of canvases (placeholders)
        self.cells: List[tk.Canvas] = []
        for r in range(GRID_ROWS):
            self.gridFrame.rowconfigure(r, weight=1)
        for c in range(GRID_COLS):
            self.gridFrame.columnconfigure(c, weight=1)

        for i in range(GRID_ROWS * GRID_COLS):
            canvas = tk.Canvas(self.gridFrame, bg="#222", highlightthickness=1, highlightbackground="#444")
            r = i // GRID_COLS
            c = i % GRID_COLS
            canvas.grid(row=r, column=c, sticky="nsew")
            canvas.bind("<Configure>", debounce(canvas, 100)(lambda e, idx=i: self._redrawThumb(idx)))
            canvas.bind("<Button-1>", lambda e, idx=i: self._onThumbClick(idx))
            canvas.bind("<Double-Button-1>", lambda e, idx=i: self._openEnlarged(idx))
            self.cells.append(canvas)

        # Resize handler (update thumbs) - debounced to avoid redraw storms
        self.master.bind("<Configure>", debounce(self.master, 150)(lambda e: self._redrawAllThumbs()))

    def onBatchCropClick(self):
        """Open enlarged viewer on the selected image and start crop mode."""
        if not self.images:
            messagebox.showwarning("Batch Crop", "Load images first.")
            return

        idx = self.selectedIndex if self.selectedIndex is not None else 0
        viewer = self._openEnlarged(idx, startCrop=True)  # start crop right away

        # If the main toolbar has a 'Use relative margins' checkbox, sync it into the viewer
        try:
            if hasattr(self, "cropAllVar") and hasattr(viewer, "useMarginsVar"):
                viewer.useMarginsVar.set(bool(self.cropAllVar.get()))
        except Exception:
            pass

    def onProceedProcessing(self):
        if not getattr(self, "images", None):
            messagebox.showwarning("Processing", "Load images first.")
            return

        def _receive_from_processing(binaries):
            # Normalize to uint8 {0,255} for OpenCV compatibility
            self.masks = [
                ensure_mask_uint8(b, self.images[i].shape[:2]) if b is not None else None
                for i, b in enumerate(binaries)
            ]
            self._redrawAllThumbs()
            self.statusVar.set("Received masks from Processing.")

        win = processing.ProcessingWindow(
            parent=self.master,
            images=self.images,
            paths=self.paths,
            scales=self.scales if hasattr(self, "scales") else [None] * len(self.images)
            # <-- no resultsCallback here
        )
        # Set it after construction (works no matter which version is imported)
        setattr(win, "resultsCallback", _receive_from_processing)

    # --------------------------- Image loading ---------------------------

    def onLoad(self):
        """Legacy hook from toolbar: call the robust multi-select loader."""
        return self.onOpenImages()

    def onOpenImages(self):
        """Open up to MAX_IMAGES images, create previews, render grid."""
        from tkinter import messagebox
        paths = filedialog.askopenfilenames(
            title=f"Select up to {MAX_IMAGES} images",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp")]
        )
        if not paths:
            return

        # Cap count
        paths = paths[:MAX_IMAGES]

        imgs, thumbs, ok_paths, errs = [], [], [], []
        dlg = None
        try:
            dlg = BusyDialog(self.root, title="Loading images…", mode="indeterminate")
            for p in paths:
                try:
                    im = safeReadImage(p)
                    imgs.append(im)
                    thumbs.append(makePreview(im))
                    ok_paths.append(p)
                except Exception as e:
                    errs.append(f"{os.path.basename(p)}: {e}")
        finally:
            if dlg: dlg.close()

        if not imgs:
            messagebox.showerror("Load images", "No images could be loaded." + ("\n" + "\n".join(errs[:6]) if errs else ""))
            return
        if errs:
            messagebox.showwarning("Some images skipped", "Problems:\n" + "\n".join(errs[:8]))

        # Store
        self.images = imgs
        self.previews = thumbs
        self.imagePaths = ok_paths
        self.paths = ok_paths
        self.currentIndex = 0
        self.scales = [None] * len(self.images) if not hasattr(self, "scales") or len(getattr(self, "scales", [])) != len(self.images) else self.scales
        self.masks = [None] * len(self.images)   # start with no masks

        # Render grid
        self._renderGrid()

    # --------------------------- Editing ---------------------------
    def onOpenEditor(self):
        if not self.images:
            messagebox.showwarning("Post-processing", "Load images first.")
            return

        def _receive_masks(new_masks: List[Optional[np.ndarray]]):
            # Normalize to uint8 {0,255} for OpenCV compatibility
            self.masks = [
                ensure_mask_uint8(m, self.images[i].shape[:2]) if m is not None else None
                for i, m in enumerate(new_masks)
            ]
            self._redrawAllThumbs()
            self.statusVar.set("Masks updated.")

        postprocessing.PostprocessWindow(
            parent=self.master,
            images=self.images,
            masks=self.masks,
            paths=self.paths,
            startIndex=self.selectedIndex or 0,
            onMasksUpdated=_receive_masks
        )

    # --------------------------- Grid rendering ---------------------------

    def _redrawAllThumbs(self):
        for i in range(GRID_ROWS * GRID_COLS):
            self._redrawThumb(i)

    def _onThumbClick(self, idx: int):
        """Select the clicked thumbnail and redraw the grid (adds blue outline)."""
        if 0 <= idx < len(self.images):
            self.selectedIndex = idx
            self._redrawAllThumbs()


    def _renderGrid(self):
        """Refresh all 5x4 canvases with the loaded images and update status."""
        # If you want to use previews instead of full res, swap self.images -> self.previews below.
        for i in range(GRID_ROWS * GRID_COLS):
            self._redrawThumb(i)
        n = len(self.images)
        if n == 0:
            self.statusVar.set("No images loaded")
        else:
            self.statusVar.set(f"Loaded {n} image(s)")


    def _redrawThumb(self, idx: int):
        canvas = self.cells[idx]
        canvas.delete("all")

        if idx >= len(self.images):
            return

        # Use downscaled preview for speed
        src = self.previews[idx] if idx < len(self.previews) else self.images[idx]
        mode = self.displayModeVar.get()
        mask = None
        if idx < len(self.masks):
            mask = self.masks[idx]

        # Build the thumbnail to display
        if mode == "original" or mask is None:
            show = src
        elif mode == "mask":
            if mask.ndim == 3:
                mm = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mm = mask
            # Resize mask to preview size for display
            mm_disp = cv2.resize(mm, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_NEAREST)
            show = mm_disp
        else:  # overlay
            if src.ndim == 2:
                base = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            else:
                base = src
            mm = mask if mask is not None else np.zeros(base.shape[:2], np.uint8)
            mm_disp = cv2.resize(mm, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
            show = self._thumbOverlay(base, mm_disp, alpha=0.45)

        pil = self._npToPil(show)
        cw = max(1, canvas.winfo_width()); ch = max(1, canvas.winfo_height())
        iw, ih = pil.size
        s = min(cw / float(iw), ch / float(ih)); s = min(s, 1.0)
        new_w = max(1, int(round(iw * s))); new_h = max(1, int(round(ih * s)))
        if (new_w, new_h) != (iw, ih):
            pil = pil.resize((new_w, new_h), resample=RESAMPLE_LANCZOS)

        photo = ImageTk.PhotoImage(pil)
        self._thumbPhotos[idx] = photo
        canvas.create_image((cw - new_w) // 2, (ch - new_h) // 2, anchor="nw", image=photo)

        if self.selectedIndex == idx:
            canvas.create_rectangle(2, 2, cw - 2, ch - 2, outline="#66b3ff", width=2)


    # --------------------------- Enlarged viewer ---------------------------

    def _openEnlarged(self, idx: int, startCrop: bool = False):
        if idx >= len(self.images):
            return None
        self.selectedIndex = idx
        self._redrawAllThumbs()

        viewer = EnlargedViewer(
            parent=self.master,
            images=self.images,
            paths=self.paths,
            scales=getattr(self, "scales", [None] * len(self.images)) if hasattr(self, "scales") else [None] * len(self.images),
            startIndex=idx,
            onImagesUpdated=self._onImagesUpdated,
            onScalesUpdated=getattr(self, "_onScalesUpdated", lambda s: None)  # no-op if not present
        )

        if startCrop:
            # wait for window to render, then enter crop mode
            self.master.after(100, viewer._startCrop)

        return viewer


    def _onImagesUpdated(self, newImages: List[np.ndarray]):
        """Callback when batch operations (like crop) modify images."""
        self.images = newImages
        self._redrawAllThumbs()

    def _onScalesUpdated(self, newScales: List[Optional[Dict[str, float | str]]]):
        self.scales = newScales
        # Optional: reflect scale somewhere in main window if desired

    # --------------------------- Utilities ---------------------------

    def _npToPil(self, arr: np.ndarray) -> Image.Image:
        a = arr
        if a.ndim == 2:
            return Image.fromarray(a, mode="L")
        if a.ndim == 3 and a.shape[2] == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            return Image.fromarray(a, mode="RGB")
        if a.ndim == 3 and a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(a, mode="RGBA")
        # Fallback to grayscale
        g = a.mean(axis=2).astype(np.uint8) if a.ndim == 3 else a.astype(np.uint8)
        return Image.fromarray(g, mode="L")

    def _thumbOverlay(self, base_bgr_or_gray: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        if mask is None:
            return base_bgr_or_gray
        if base_bgr_or_gray.ndim == 2:
            base = cv2.cvtColor(base_bgr_or_gray, cv2.COLOR_GRAY2BGR)
        else:
            base = base_bgr_or_gray
        m = (mask > 0).astype(np.uint8)
        overlay = base.copy()
        overlay[:, :, 1] = np.maximum(overlay[:, :, 1], m * 255)
        overlay[:, :, 2] = np.maximum(overlay[:, :, 2], m * 255)
        return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)


# ============================== Enlarged Viewer ==============================

class EnlargedViewer(tk.Toplevel):
    """Modal enlarged viewer with zoom/pan, crop overlay, and pixel-scale calibration.

    Gestures:
      - Mouse wheel: zoom (centered at cursor)
      - Middle mouse drag or Pan mode: pan
      - Left+Right held together: pan (alternative gesture)
      - Buttons on toolbar for zoom/pan/reset (mobile-friendly)
      - Crop: draw rectangle, confirm "current / all" and margins mode
      - Calibrate: click start point, then click second point → constrained to same vertical pixel (same y)
    """

    def __init__(
        self,
        parent: tk.Tk,
        images: List[np.ndarray],
        paths: List[str],
        scales: List[Optional[Dict[str, float | str]]],
        startIndex: int,
        onImagesUpdated,
        onScalesUpdated
    ):
        super().__init__(parent)
        self.title("Viewer")
        self.transient(parent)
        self.grab_set()

        self.images = images
        self.paths = paths
        self.scales = scales
        self.index = startIndex
        self.onImagesUpdated = onImagesUpdated
        self.onScalesUpdated = onScalesUpdated

        self._photo = None
        self._pil = None
        self._npImg = None  # cached numpy array for fast viewport cropping
        self._scale = 1.0
        self._offset = np.array([0.0, 0.0], dtype=float)  # pan offset
        self._panActive = False
        self._lastDrag = None
        
        # Pan motion coalescing (reduce render calls during drag)
        self._pendingPanPt: Optional[Tuple[int, int]] = None
        self._panMotionScheduled = False

        # Crop overlay state
        self._cropMode = False
        self._cropRect: Optional[Rect] = None

        # Calibration state
        self._calibMode = False
        self._calibStart: Optional[Tuple[int, int]] = None  # image coords (x,y)
        self._calibEnd: Optional[Tuple[int, int]] = None    # image coords (x,y), y constrained to start.y
        self._calibPxDist: Optional[float] = None

        # Drawn ids
        self._calibLineId = None
        self._calibTextId = None

        # Build UI
        self._buildUi()
        self._loadCurrent()

        self.geometry("1150x780")
        self.minsize(820, 560)

    # -------- UI --------

    def _buildUi(self):
        # Toolbar
        tb = ttk.Frame(self)
        tb.pack(side="top", fill="x")

        ttk.Button(tb, text="Prev", command=self._prev).pack(side="left", padx=4, pady=6)
        ttk.Button(tb, text="Next", command=self._next).pack(side="left", padx=4)

        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Button(tb, text="Zoom In", command=lambda: self._zoomAtCenter(1.25)).pack(side="left", padx=2)
        ttk.Button(tb, text="Zoom Out", command=lambda: self._zoomAtCenter(0.8)).pack(side="left", padx=2)
        ttk.Button(tb, text="Reset View", command=self._resetView).pack(side="left", padx=8)
        self.panVar = tk.BooleanVar(value=False)
        ttk.Checkbutton(tb, text="Pan Mode", variable=self.panVar, command=self._updatePanMode).pack(side="left", padx=6)

        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Button(tb, text="Crop", command=self._startCrop).pack(side="left", padx=4)
        self.useMarginsVar = tk.BooleanVar(value=True)
        ttk.Checkbutton(tb, text="Use relative margins", variable=self.useMarginsVar).pack(side="left", padx=6)
        ttk.Button(tb, text="Apply Crop…", command=self._applyCropDialog).pack(side="left", padx=4)

        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)

        # Calibration controls
        self.calibStatusVar = tk.StringVar(value="Scale: not set")
        ttk.Button(tb, text="Calibrate", command=self._startCalibration).pack(side="left", padx=4)
        ttk.Button(tb, text="Set Scale…", command=self._manualScaleDialog).pack(side="left", padx=4)
        ttk.Label(tb, textvariable=self.calibStatusVar).pack(side="left", padx=12)

        # Canvas
        self.canvas = tk.Canvas(self, bg="#111", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)

        # Bindings - debounce resize to avoid render storms
        self.canvas.bind("<Configure>", debounce(self.canvas, 100)(lambda e: self._render()))
        self.canvas.bind("<MouseWheel>", self._onWheel)  # Windows/macOS
        self.canvas.bind("<Button-4>", self._onWheel)    # some X11
        self.canvas.bind("<Button-5>", self._onWheel)
        self.canvas.bind("<ButtonPress-2>", self._onPanStart)
        self.canvas.bind("<B2-Motion>", self._onPanDrag)
        self.canvas.bind("<ButtonRelease-2>", self._onPanEnd)

        # Left+Right simult pan
        self.canvas.bind("<ButtonPress-1>", self._onButtonComboDown)
        self.canvas.bind("<ButtonPress-3>", self._onButtonComboDown)
        self.canvas.bind("<ButtonRelease-1>", self._onButtonComboUp)
        self.canvas.bind("<ButtonRelease-3>", self._onButtonComboUp)
        self.canvas.bind("<B1-Motion>", self._onComboDrag)
        self.canvas.bind("<B3-Motion>", self._onComboDrag)

        # Crop interactions (added with "+", so they stack with other bindings)
        self.canvas.bind("<ButtonPress-1>", self._onCropPress, add="+")
        self.canvas.bind("<B1-Motion>", self._onCropDrag, add="+")
        self.canvas.bind("<ButtonRelease-1>", self._onCropRelease, add="+")
        self.canvas.bind("<Double-Button-1>", self._onCropConfirm, add="+")

        # Calibration interactions (Ctrl-click required)
        self.canvas.bind("<Control-Button-1>", self._onCalibPress, add="+")
        self.canvas.bind("<Motion>", self._onCalibMotion, add="+")
        self.canvas.bind("<Control-ButtonRelease-1>", self._onCalibRelease, add="+")


    # -------- Image IO / render --------

    def _loadCurrent(self):
        img = self.images[self.index]
        self._npImg = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self._pil = self._npToPil(img)
        self._resetView()
        self._updateTitle()
        self._updateScaleStatus()
        self._render()

    def _updateTitle(self):
        name = os.path.basename(self.paths[self.index]) if self.paths and self.index < len(self.paths) else f"Image {self.index+1}"
        self.title(f"Viewer – {name} ({self.index+1}/{len(self.images)})")

    def _resetView(self):
        self._scale = 1.0
        self._offset[:] = 0.0
        self._render()

    def _render(self):
        if self._npImg is None:
            return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        img_h, img_w = self._npImg.shape[:2]

        # Fit to view initially (respect current scale)
        baseScale = min(cw / float(img_w), ch / float(img_h))
        baseScale = min(baseScale, 1.0)
        combined_scale = baseScale * self._scale

        disp_w = max(1, int(round(img_w * combined_scale)))
        disp_h = max(1, int(round(img_h * combined_scale)))
        ox = int((cw - disp_w) / 2 + self._offset[0])
        oy = int((ch - disp_h) / 2 + self._offset[1])

        # Store for coordinate transforms
        self._combined_scale = combined_scale
        self._ox, self._oy = ox, oy

        # Use viewport cropping for performance at high zoom
        if combined_scale <= 1.0:
            # Zoomed out: resize whole image (fast enough)
            resized = cv2.resize(self._npImg, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            pil = self._npToPil(resized)
            img_ox, img_oy = ox, oy
        else:
            # Zoomed in: crop visible region first, then resize (much faster for 4K)
            img_x0 = int(max(0, (0 - ox) / combined_scale))
            img_y0 = int(max(0, (0 - oy) / combined_scale))
            img_x1 = int(min(img_w, (cw - ox) / combined_scale + 1))
            img_y1 = int(min(img_h, (ch - oy) / combined_scale + 1))

            # Add margin
            margin = 2
            img_x0 = max(0, img_x0 - margin)
            img_y0 = max(0, img_y0 - margin)
            img_x1 = min(img_w, img_x1 + margin)
            img_y1 = min(img_h, img_y1 + margin)

            crop = self._npImg[img_y0:img_y1, img_x0:img_x1]
            if crop.size == 0:
                self.canvas.delete("all")
                return

            crop_dw = max(1, int(round((img_x1 - img_x0) * combined_scale)))
            crop_dh = max(1, int(round((img_y1 - img_y0) * combined_scale)))
            crop_resized = cv2.resize(crop, (crop_dw, crop_dh), interpolation=cv2.INTER_NEAREST)

            # Create output buffer and place crop at correct position
            output = np.zeros((ch, cw, 3), dtype=np.uint8)
            canvas_x0 = int(ox + img_x0 * combined_scale)
            canvas_y0 = int(oy + img_y0 * combined_scale)

            src_x0 = max(0, -canvas_x0)
            src_y0 = max(0, -canvas_y0)
            dst_x0 = max(0, canvas_x0)
            dst_y0 = max(0, canvas_y0)
            copy_w = min(crop_dw - src_x0, cw - dst_x0)
            copy_h = min(crop_dh - src_y0, ch - dst_y0)

            if copy_w > 0 and copy_h > 0:
                output[dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = \
                    crop_resized[src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]

            pil = self._npToPil(output)
            img_ox, img_oy = 0, 0  # buffer fills canvas

        self.canvas.delete("all")
        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.create_image(img_ox, img_oy, anchor="nw", image=self._photo, tags="img")

        # Draw overlays
        self._drawCropOverlay()
        self._drawCalibrationOverlay()

    def _npToPil(self, arr: np.ndarray) -> Image.Image:
        a = arr
        if a.ndim == 2:
            return Image.fromarray(a, mode="L")
        if a.ndim == 3 and a.shape[2] == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            return Image.fromarray(a, mode="RGB")
        if a.ndim == 3 and a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(a, mode="RGBA")
        g = a.mean(axis=2).astype(np.uint8) if a.ndim == 3 else a.astype(np.uint8)
        return Image.fromarray(g, mode="L")

    # -------- Zoom & Pan --------

    def _zoomAtCenter(self, factor: float):
        self._scale *= factor
        self._scale = max(0.1, min(self._scale, 20.0))
        self._render()

    def _zoomAtPoint(self, cx: int, cy: int, factor: float):
        """Zoom centered at canvas point (cx, cy)."""
        if self._npImg is None:
            return
        img_h, img_w = self._npImg.shape[:2]
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

        # Current scale and origin
        baseScale = min(cw / float(img_w), ch / float(img_h), 1.0)
        s_old = baseScale * self._scale
        disp_w_old = int(round(img_w * s_old))
        disp_h_old = int(round(img_h * s_old))
        ox_old = int((cw - disp_w_old) / 2 + self._offset[0])
        oy_old = int((ch - disp_h_old) / 2 + self._offset[1])

        # Position in image coords under cursor
        ix = (cx - ox_old) / max(s_old, 1e-6)
        iy = (cy - oy_old) / max(s_old, 1e-6)

        # Apply zoom
        self._scale *= factor
        self._scale = max(0.1, min(self._scale, 20.0))

        # New scale
        s_new = baseScale * self._scale
        disp_w_new = int(round(img_w * s_new))
        disp_h_new = int(round(img_h * s_new))

        # Adjust offset so that (ix, iy) still maps to (cx, cy)
        base_ox = (cw - disp_w_new) / 2
        base_oy = (ch - disp_h_new) / 2
        self._offset[0] = cx - ix * s_new - base_ox
        self._offset[1] = cy - iy * s_new - base_oy

        self._render()

    def _onWheel(self, event):
        # Normalize delta
        delta = 1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else -1
        factor = 1.1 if delta > 0 else 0.9
        self._zoomAtPoint(event.x, event.y, factor)

    def _updatePanMode(self):
        self._panActive = bool(self.panVar.get())

    def _onPanStart(self, event):
        self._panActive = True
        self._lastDrag = (event.x, event.y)

    def _schedulePanMotion(self):
        if self._panMotionScheduled:
            return
        self._panMotionScheduled = True
        self.after_idle(self._processPanMotion)

    def _processPanMotion(self):
        self._panMotionScheduled = False
        if not self._panActive or self._pendingPanPt is None or self._lastDrag is None:
            return
        dx = self._pendingPanPt[0] - self._lastDrag[0]
        dy = self._pendingPanPt[1] - self._lastDrag[1]
        self._offset += np.array([dx, dy], dtype=float)
        self._lastDrag = self._pendingPanPt
        self._render()

    def _onPanDrag(self, event):
        if not self._panActive or self._lastDrag is None:
            return
        self._pendingPanPt = (event.x, event.y)
        self._schedulePanMotion()

    def _onPanEnd(self, event):
        self._lastDrag = None
        if not self.panVar.get():
            self._panActive = False

    # Left+Right combined as pan
    def _onButtonComboDown(self, event):
        state = event.state
        left_down = (state & 0x100) != 0 or event.num == 1
        right_down = (state & 0x400) != 0 or event.num == 3
        if left_down and right_down:
            self._panActive = True
            self._lastDrag = (event.x, event.y)

    def _onComboDrag(self, event):
        if self._panActive and self._lastDrag is not None:
            self._pendingPanPt = (event.x, event.y)
            self._schedulePanMotion()

    def _onButtonComboUp(self, event):
        self._lastDrag = None
        if not self.panVar.get():
            self._panActive = False

    # -------- Crop overlay --------

    def _startCrop(self):
        self._cropMode = True
        self._calibMode = False
        self._cropRect = None
        self._calibStart = None
        self._calibEnd = None
        self._render()

    def _applyCropDialog(self):
        if not self._cropRect:
            messagebox.showwarning("Crop", "Draw a crop box first (Crop button, then drag).")
            return

        # Confirm dialog
        apply_all = messagebox.askyesno(
            "Apply Crop",
            "Apply this crop to ALL images?\n\nYes = all images\nNo = current image only"
        )

        useMargins = self.useMarginsVar.get()

        # Compute rect in image coordinates
        rect_img = self._canvasRectToImageRect(self._cropRect)
        if rect_img is None:
            messagebox.showwarning("Crop", "Crop is out of bounds or too small.")
            return

        # Apply
        if apply_all:
            newImages = applyCropBatch(self.images, rect=rect_img, margins=None, useMargins=useMargins)
            # Replace only successful crops
            for i, cropped in enumerate(newImages):
                if cropped is not None:
                    self.images[i] = cropped
        else:
            img = self.images[self.index]
            rect_img = clampRectToImage(rect_img, img.shape)
            if rect_img is None:
                messagebox.showwarning("Crop", "Crop is invalid for this image.")
                return
            self.images[self.index] = cropWithRect(img, rect_img)

        self.onImagesUpdated(self.images)
        self._cropMode = False
        self._cropRect = None
        self._render()
        messagebox.showinfo("Crop", "Crop applied.")

    def _drawCropOverlay(self):
        if not self._cropMode or self._cropRect is None:
            return
        x0, y0, x1, y1 = self._cropRect
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00ff88", width=2, dash=(4, 2))
        r = 5
        for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1),
                       ((x0+x1)//2, y0), ((x0+x1)//2, y1), (x0, (y0+y1)//2), (x1, (y0+y1)//2)):
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="#00ff88", fill="", width=2)

    def _onCropPress(self, event):
        if not self._cropMode:
            return
        self._cropRect = (event.x, event.y, event.x, event.y)
        self._render()

    def _onCropDrag(self, event):
        if not self._cropMode or self._cropRect is None:
            return
        x0, y0, _, _ = self._cropRect
        self._cropRect = (min(x0, event.x), min(y0, event.y), max(x0, event.x), max(y0, event.y))
        self._render()

    def _onCropRelease(self, event):
        if not self._cropMode or self._cropRect is None:
            return
        # clamp to canvas bounds
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        x0, y0, x1, y1 = self._cropRect
        x0 = max(0, min(x0, cw - 1))
        y0 = max(0, min(y0, ch - 1))
        x1 = max(1, min(x1, cw))
        y1 = max(1, min(y1, ch))
        if x1 - x0 < 3 or y1 - y0 < 3:
            self._cropRect = None
        else:
            self._cropRect = (x0, y0, x1, y1)
        self._render()

    def _onCropConfirm(self, event):
        if self._cropMode and self._cropRect is not None:
            # convenience: double-click confirms then opens apply dialog
            self._applyCropDialog()

    def _canvasRectToImageRect(self, rect: Rect) -> Optional[Rect]:
        """Convert a canvas rect to image pixel rect, accounting for scale/offset/fit."""
        if self._pil is None or rect is None:
            return None
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        iw, ih = self._pil.size

        baseScale = min(cw / float(iw), ch / float(ih))
        baseScale = min(baseScale, 1.0)
        s = baseScale * self._scale
        w = int(round(iw * s))
        h = int(round(ih * s))
        ox = int((cw - w) / 2 + self._offset[0])
        oy = int((ch - h) / 2 + self._offset[1])

        x0c, y0c, x1c, y1c = rect
        # Convert canvas coords to image coords
        x0i = int((x0c - ox) / max(s, 1e-6))
        y0i = int((y0c - oy) / max(s, 1e-6))
        x1i = int((x1c - ox) / max(s, 1e-6))
        y1i = int((y1c - oy) / max(s, 1e-6))

        # Clamp to image bounds
        x0i, x1i = min(x0i, x1i), max(x0i, x1i)
        y0i, y1i = min(y0i, y1i), max(y0i, y1i)
        r = clampRectToImage((x0i, y0i, x1i, y1i), (ih, iw))
        return r

    # -------- Calibration --------

    def _startCalibration(self):
        self._calibMode = True
        self._cropMode = False
        self._calibStart = None
        self._calibEnd = None
        self._calibPxDist = None
        self._render()
        messagebox.showinfo(
            "Calibration",
            "Ctrl-click the start point, then Ctrl-click the second point.\n"
            "The second point will be constrained to the same vertical pixel (same y).\n"
            "After the second Ctrl-click, you'll enter the real length and units."
        )


    def _onCalibPress(self, event):
        if not self._calibMode:
            return

        # Convert canvas->image coords
        pt = self._canvasToImagePoint(event.x, event.y)
        if pt is None:
            return
        x, y = pt

        if self._calibStart is None:
            self._calibStart = (x, y)
            self._calibEnd = None
        else:
            # finalize with constrained y
            x1, y1 = self._calibStart
            x2 = x
            y2 = y1  # same vertical pixel
            self._calibEnd = (x2, y2)
            self._calibPxDist = abs(x2 - x1)
            self._render()
            self._calibrationDialog()  # prompt for real length & units

    def _onCalibMotion(self, event):
        if not self._calibMode:
            return
        if self._calibStart is None:
            return
        pt = self._canvasToImagePoint(event.x, event.y)
        if pt is None:
            return
        x, y = pt
        x1, y1 = self._calibStart
        # live endpoint constrained to same y
        self._calibEnd = (x, y1)
        self._calibPxDist = abs(x - x1)
        self._render()

    def _onCalibRelease(self, event):
        # nothing extra; finalize handled on press of second point
        pass

    def _drawCalibrationOverlay(self):
        if not self._calibMode:
            return
        if self._calibStart is None:
            return

        # Convert image points back to canvas space for drawing
        a = self._imageToCanvasPoint(*self._calibStart)
        b = self._imageToCanvasPoint(*self._calibEnd) if self._calibEnd else None
        if a is None:
            return
        x0c, y0c = a

        # draw start point
        r = 4
        self.canvas.create_oval(x0c-r, y0c-r, x0c+r, y0c+r, outline="#ffaa00", width=2)

        if b is not None:
            x1c, y1c = b
            self.canvas.create_line(x0c, y0c, x1c, y1c, fill="#ffaa00", width=2)
            # label
            if self._calibPxDist is not None:
                txt = f"{int(round(self._calibPxDist))} px"
                self.canvas.create_text((x0c + x1c)//2, y0c - 12, text=txt, fill="#ffaa00", anchor="s")

    def _calibrationDialog(self):
        if self._calibPxDist is None or self._calibStart is None or self._calibEnd is None:
            return
        px = float(self._calibPxDist)

        # Simple dialog: real length + units + apply to all?
        dlg = tk.Toplevel(self)
        dlg.title("Calibration")
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text=f"Measured distance: {px:.3f} px").grid(row=0, column=0, columnspan=4, padx=8, pady=(10, 6), sticky="w")

        ttk.Label(dlg, text="Enter real length:").grid(row=1, column=0, padx=8, pady=4, sticky="e")
        lengthVar = tk.DoubleVar(value=1.0)
        ttk.Entry(dlg, textvariable=lengthVar, width=10).grid(row=1, column=1, padx=4, pady=4, sticky="w")

        ttk.Label(dlg, text="Units:").grid(row=1, column=2, padx=8, pady=4, sticky="e")
        unitVar = tk.StringVar(value="mm")
        ttk.Entry(dlg, textvariable=unitVar, width=8).grid(row=1, column=3, padx=4, pady=4, sticky="w")

        applyAllVar = tk.BooleanVar(value=False)
        ttk.Checkbutton(dlg, text="Apply to all images", variable=applyAllVar).grid(row=2, column=0, columnspan=4, padx=8, pady=6, sticky="w")

        # Buttons
        btns = ttk.Frame(dlg); btns.grid(row=3, column=0, columnspan=4, pady=(6, 10))
        def on_ok():
            try:
                realLen = float(lengthVar.get())
                unitName = unitVar.get().strip() or "mm"
                if realLen <= 0:
                    raise ValueError
            except Exception:
                messagebox.showerror("Calibration", "Please enter a positive real length.")
                return
            unitsPerPx = realLen / max(px, 1e-9)  # units per pixel
            self._setScale(unitsPerPx, unitName, applyAllVar.get())
            dlg.destroy()
        def on_cancel():
            dlg.destroy()

        ttk.Button(btns, text="OK", command=on_ok).pack(side="left", padx=8)
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="left", padx=8)

        dlg.wait_window()

        # Exit calibration mode after dialog
        self._calibMode = False
        self._calibStart = None
        self._calibEnd = None
        self._calibPxDist = None
        self._render()

    def _manualScaleDialog(self):
        # Dialog to set scale either as units/px or px/unit, with units; apply to current or all
        dlg = tk.Toplevel(self)
        dlg.title("Set Scale")
        dlg.transient(self)
        dlg.grab_set()

        modeVar = tk.StringVar(value="unitsPerPx")  # or "pxPerUnit"
        ttk.Radiobutton(dlg, text="Units per pixel", value="unitsPerPx", variable=modeVar).grid(row=0, column=0, padx=8, pady=(10,4), sticky="w")
        unitsPerPxVar = tk.DoubleVar(value=0.005)
        ttk.Entry(dlg, textvariable=unitsPerPxVar, width=10).grid(row=0, column=1, padx=4, pady=(10,4), sticky="w")

        ttk.Radiobutton(dlg, text="Pixels per unit", value="pxPerUnit", variable=modeVar).grid(row=1, column=0, padx=8, pady=4, sticky="w")
        pxPerUnitVar = tk.DoubleVar(value=200.0)
        ttk.Entry(dlg, textvariable=pxPerUnitVar, width=10).grid(row=1, column=1, padx=4, pady=4, sticky="w")

        ttk.Label(dlg, text="Units:").grid(row=2, column=0, padx=8, pady=4, sticky="e")
        unitVar = tk.StringVar(value="mm")
        ttk.Entry(dlg, textvariable=unitVar, width=8).grid(row=2, column=1, padx=4, pady=4, sticky="w")

        applyAllVar = tk.BooleanVar(value=False)
        ttk.Checkbutton(dlg, text="Apply to all images", variable=applyAllVar).grid(row=3, column=0, columnspan=2, padx=8, pady=6, sticky="w")

        btns = ttk.Frame(dlg); btns.grid(row=4, column=0, columnspan=2, pady=(6, 10))
        def on_ok():
            unitName = unitVar.get().strip() or "mm"
            mode = modeVar.get()
            try:
                if mode == "unitsPerPx":
                    val = float(unitsPerPxVar.get())
                    if val <= 0:
                        raise ValueError
                    unitsPerPx = val
                else:
                    val = float(pxPerUnitVar.get())
                    if val <= 0:
                        raise ValueError
                    unitsPerPx = 1.0 / val
            except Exception:
                messagebox.showerror("Set Scale", "Please enter a positive value.")
                return
            self._setScale(unitsPerPx, unitName, applyAllVar.get())
            dlg.destroy()
        def on_cancel():
            dlg.destroy()

        ttk.Button(btns, text="OK", command=on_ok).pack(side="left", padx=8)
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="left", padx=8)

        dlg.wait_window()

    def _setScale(self, unitsPerPx: float, unitName: str, applyAll: bool):
        # Save into scales list
        if applyAll:
            for i in range(len(self.scales)):
                self.scales[i] = {"unitsPerPx": float(unitsPerPx), "unitName": unitName}
        else:
            self.scales[self.index] = {"unitsPerPx": float(unitsPerPx), "unitName": unitName}

        # Callback to parent
        self.onScalesUpdated(self.scales)
        self._updateScaleStatus()
        self._render()
        messagebox.showinfo("Scale", f"Scale set to {unitsPerPx:.6g} {unitName}/px" + (" (applied to all)" if applyAll else ""))

    def _updateScaleStatus(self):
        s = self.scales[self.index] if (0 <= self.index < len(self.scales)) else None
        if s and "unitsPerPx" in s and "unitName" in s:
            self.calibStatusVar.set(f"Scale: {s['unitsPerPx']:.6g} {s['unitName']}/px")
        else:
            self.calibStatusVar.set("Scale: not set")

    # -------- Navigation --------

    def _prev(self):
        if self.index > 0:
            self.index -= 1
            self._cropMode = False
            self._calibMode = False
            self._calibStart = None
            self._calibEnd = None
            self._loadCurrent()

    def _next(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self._cropMode = False
            self._calibMode = False
            self._calibStart = None
            self._calibEnd = None
            self._loadCurrent()

    # -------- Helpers: coordinate transforms --------

    def _currentViewParams(self):
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        iw, ih = self._pil.size if self._pil is not None else (1, 1)
        baseScale = min(cw / float(iw), ch / float(ih))
        baseScale = min(baseScale, 1.0)
        s = baseScale * self._scale
        w = int(round(iw * s))
        h = int(round(ih * s))
        ox = int((cw - w) / 2 + self._offset[0])
        oy = int((ch - h) / 2 + self._offset[1])
        return s, ox, oy

    def _canvasToImagePoint(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        if self._pil is None:
            return None
        s, ox, oy = self._currentViewParams()
        iw, ih = self._pil.size
        ix = int((x - ox) / max(s, 1e-6))
        iy = int((y - oy) / max(s, 1e-6))
        if 0 <= ix < iw and 0 <= iy < ih:
            return ix, iy
        return None

    def _imageToCanvasPoint(self, ix: int, iy: int) -> Optional[Tuple[int, int]]:
        if self._pil is None:
            return None
        s, ox, oy = self._currentViewParams()
        x = int(ix * s + ox)
        y = int(iy * s + oy)
        return x, y


def main():
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        style.configure("TButton", padding=4)
        style.configure("TLabel", padding=2)
    except Exception:
        pass
    PreprocessApp(root)  # app kept alive by mainloop
    root.geometry("1400x900")
    root.minsize(900, 600)
    root.mainloop()


if __name__ == "__main__":
    main()
