# processing_gui.py
# Step 2 GUI: Thresholding + Separation with live preview
from __future__ import annotations
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import List, Dict, Optional

from PIL import Image, ImageTk

# Pillow shim
if hasattr(Image, "Resampling"):
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", Image.BICUBIC)

import stereology
import stereology_gui

from processing import (
    DEFAULTS,
    thresholdImageAdvanced,
    runSeparationPipeline,
    labelsToColor,
)
from core.batch import process_batch_parallel
from widgets import debounce


class BusyDialog(tk.Toplevel):
    def __init__(self, parent, title="Working…", mode="indeterminate", maximum=100):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()  # block interaction with parent
        self.protocol("WM_DELETE_WINDOW", lambda: None)  # disable close
        self._pickPointCanvas = None  # (x,y) on left canvas for marker
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=title).pack(anchor="w", pady=(0,8))
        self.pb = ttk.Progressbar(frm, orient="horizontal", length=360,
                                  mode=mode, maximum=maximum)
        self.pb.pack(fill="x")
        if mode == "indeterminate":
            self.pb.start(10)  # ms
        self.update_idletasks()

    def set_progress(self, value):
        # Only for 'determinate'
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


class ProcessingWindow(tk.Toplevel):
    def __init__(self, parent,
                 images,
                 paths=None,
                 scales=None,
                 resultsCallback=None):
        super().__init__(parent)
        self.title("PyFOAMS – Processing (Threshold + Separation)")
        self.transient(parent)
        self.grab_set()

        # Core data
        self.images = images
        self.paths = paths or [f"Image {i+1}" for i in range(len(images))]
        self.scales = scales or [None] * len(images)
        self.resultsCallback = resultsCallback

        # Must exist before any _showCurrent() calls/bindings
        self.currentIndex = 0

        # Outputs
        self.binaries = [None] * len(self.images)
        self.labels   = [None] * len(self.images)

        # Copy defaults/settings
        self.settings = {
            "common": dict(DEFAULTS["common"]),
            "otsu": dict(DEFAULTS["otsu"]),
            "adaptive": dict(DEFAULTS["adaptive"]),
            "percentile": dict(DEFAULTS["percentile"]),
            "pick": dict(DEFAULTS["pick"]),
            "separation": dict(DEFAULTS["separation"]),
        }

        # GUI state vars
        self.methodVar = tk.StringVar(value="otsu")
        self.useDefaultsVar = tk.BooleanVar(value=True)
        self.polarityVar = tk.StringVar(value=self.settings["common"]["polarity"])

        # Pick mode
        self.pickModeVar = tk.BooleanVar(value=False)
        self.pickValueVar = tk.IntVar(value=128)
        self.pickTolVar = tk.IntVar(value=int(self.settings["pick"]["pickTolerance"]))

        # Separation vars
        self.sepMethodVar = tk.StringVar(value=self.settings["separation"]["method"])
        self.fillHolesVar = tk.BooleanVar(value=bool(self.settings["separation"]["fillHoles"]))
        self.minAreaVar = tk.IntVar(value=int(self.settings["separation"]["minAreaPx"]))
        self.distanceBlurVar = tk.IntVar(value=int(self.settings["separation"]["distanceBlurK"]))
        self.peakMinDistVar = tk.IntVar(value=int(self.settings["separation"]["peakMinDistance"]))
        self.peakRelThrVar = tk.DoubleVar(value=float(self.settings["separation"]["peakRelThreshold"]))
        self.connectivityVar = tk.IntVar(value=int(self.settings["separation"]["connectivity"]))
        self.clearBorderVar = tk.BooleanVar(value=bool(self.settings["separation"]["clearBorder"]))
        self.overlayAlphaVar = tk.DoubleVar(value=float(self.settings["separation"]["overlayAlpha"]))

        # View mode
        self.viewModeVar = tk.StringVar(value="labels")  # "binary" | "labels"
        self.overlayOnOriginalVar = tk.BooleanVar(value=True)

        # Left/right canvas state
        self._leftPhoto = None
        self._rightPhoto = None
        self._leftScale = 1.0
        self._rightScale = 1.0
        self._leftOrigin = (0, 0)
        self._rightOrigin = (0, 0)
        self._leftDispSize = (0, 0)
        self._rightDispSize = (0, 0)
        self._pickPointCanvas = None  # used by eyedropper crosshair

        # Build UI, bind, and show
        self._buildUi()

        # Start in manual mode (no auto recompute)
        self.autoPreviewVar.set(False)

        self._bindEvents()
        self.after_idle(self._showCurrent)  # <-- add this


        self.geometry("1280x800")
        self.minsize(980, 640)


    # ----------------- UI -----------------

    def _buildUi(self):
        # Top controls
        top = ttk.Frame(self); top.pack(side="top", fill="x", padx=6, pady=6)

        ttk.Button(top, text="Prev", command=self.prevImage).pack(side="left", padx=2)
        ttk.Button(top, text="Next", command=self.nextImage).pack(side="left", padx=2)
        self.statusVar = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.statusVar).pack(side="left", padx=10)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)

        # Threshold method + defaults + advanced
        ttk.Label(top, text="Method:").pack(side="left")
        for text, val in [("Otsu", "otsu"), ("Adaptive", "adaptive"), ("Percentile", "percentile"), ("Pick", "pick")]:
            ttk.Radiobutton(top, text=text, variable=self.methodVar, value=val).pack(side="left", padx=4)

        ttk.Checkbutton(top, text="Use defaults", variable=self.useDefaultsVar).pack(side="left", padx=12)
        ttk.Button(top, text="Advanced…", command=self.openAdvancedDialog).pack(side="left")
        # after Advanced… button
        self.autoPreviewVar = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Auto preview", variable=self.autoPreviewVar).pack(side="left", padx=8)
        ttk.Button(top, text="Recompute", command=self.recomputeNow).pack(side="left", padx=4)

        # ADD THIS LINE to start in manual mode:
        self.autoPreviewVar.set(False)


        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(top, text="Polarity:").pack(side="left")
        for text, val in [("Auto", "auto"), ("Pores darker", "poresDarker"), ("Pores brighter", "poresBrighter")]:
            ttk.Radiobutton(top, text=text, variable=self.polarityVar, value=val).pack(side="left", padx=4)

        # Pick controls
        self.pickPanel = ttk.Frame(self); self.pickPanel.pack(side="top", fill="x", padx=6)
        ttk.Checkbutton(self.pickPanel, text="Pick from image (click left)", variable=self.pickModeVar).pack(side="left")
        ttk.Label(self.pickPanel, text="Gray:").pack(side="left", padx=(10, 2))
        ttk.Entry(self.pickPanel, textvariable=self.pickValueVar, width=5).pack(side="left")
        ttk.Label(self.pickPanel, text="± tol").pack(side="left", padx=(8, 2))
        ttk.Scale(self.pickPanel, from_=0, to=60, orient="horizontal", variable=self.pickTolVar, length=150).pack(side="left")
        self._updatePickPanelVisibility()

        # Separation panel
        sep = ttk.LabelFrame(self, text="Separation"); sep.pack(side="top", fill="x", padx=6, pady=6)
        ttk.Label(sep, text="Method:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Radiobutton(sep, text="None", value="none", variable=self.sepMethodVar).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(sep, text="Watershed", value="watershed", variable=self.sepMethodVar).grid(row=0, column=2, sticky="w")

        ttk.Checkbutton(sep, text="Fill holes", variable=self.fillHolesVar).grid(row=1, column=0, sticky="w", padx=6)
        ttk.Label(sep, text="Min area (px)").grid(row=1, column=1, sticky="e"); ttk.Entry(sep, textvariable=self.minAreaVar, width=7).grid(row=1, column=2, sticky="w")

        ttk.Label(sep, text="Distance blur k").grid(row=2, column=0, sticky="e"); ttk.Entry(sep, textvariable=self.distanceBlurVar, width=7).grid(row=2, column=1, sticky="w")
        ttk.Label(sep, text="Peak min distance").grid(row=2, column=2, sticky="e"); ttk.Entry(sep, textvariable=self.peakMinDistVar, width=7).grid(row=2, column=3, sticky="w")
        ttk.Label(sep, text="Peak rel threshold").grid(row=2, column=4, sticky="e"); ttk.Entry(sep, textvariable=self.peakRelThrVar, width=7).grid(row=2, column=5, sticky="w")

        ttk.Label(sep, text="Connectivity").grid(row=3, column=0, sticky="e")
        ttk.Radiobutton(sep, text="4", variable=self.connectivityVar, value=4).grid(row=3, column=1, sticky="w")
        ttk.Radiobutton(sep, text="8", variable=self.connectivityVar, value=8).grid(row=3, column=2, sticky="w")
        ttk.Checkbutton(sep, text="Clear border", variable=self.clearBorderVar).grid(row=3, column=3, sticky="w")

        ttk.Label(sep, text="Overlay alpha").grid(row=3, column=4, sticky="e"); ttk.Entry(sep, textvariable=self.overlayAlphaVar, width=7).grid(row=3, column=5, sticky="w")

        # View panel
        view = ttk.LabelFrame(self, text="View"); view.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Radiobutton(view, text="Binary", value="binary", variable=self.viewModeVar).pack(side="left", padx=6)
        ttk.Radiobutton(view, text="Labels", value="labels", variable=self.viewModeVar).pack(side="left", padx=6)
        ttk.Checkbutton(view, text="Overlay on original", variable=self.overlayOnOriginalVar).pack(side="left", padx=12)

        # Canvases
        center = ttk.Frame(self); center.pack(side="top", fill="both", expand=True, padx=6, pady=6)
        self.leftCanvas = tk.Canvas(center, bg="#202020", highlightthickness=1, highlightbackground="#3a3a3a")
        self.rightCanvas = tk.Canvas(center, bg="#202020", highlightthickness=1, highlightbackground="#3a3a3a")
        self.leftCanvas.pack(side="left", fill="both", expand=True)
        self.rightCanvas.pack(side="left", fill="both", expand=True)
        self._leftPhoto = None; self._rightPhoto = None
        self._leftScale = 1.0;   self._rightScale = 1.0

        # Bottom actions
        bottom = ttk.Frame(self); bottom.pack(side="bottom", fill="x", padx=6, pady=6)
        ttk.Button(bottom, text="Apply to Current", command=self.applyToCurrent).pack(side="left")
        ttk.Button(bottom, text="Apply to All", command=self.applyToAll).pack(side="left", padx=6)
        ttk.Button(bottom, text="Save Masks…", command=self.saveMasks).pack(side="left", padx=6)
        ttk.Button(bottom, text="Save Labels…", command=self.saveLabels).pack(side="left", padx=6)
        ttk.Button(bottom, text="Stereology…", command=self.openStereology).pack(side="left", padx=6)

        ttk.Button(bottom, text="Export Measurements…", command=self.exportMeasurements).pack(side="left", padx=6)


    def _bindEvents(self):
        def bind_var(var):
            var.trace_add("write", self._onParamChanged)

        for var in (
            self.methodVar, self.useDefaultsVar, self.polarityVar,
            self.sepMethodVar, self.fillHolesVar, self.minAreaVar, self.distanceBlurVar,
            self.peakMinDistVar, self.peakRelThrVar, self.connectivityVar,
            self.clearBorderVar, self.overlayAlphaVar, self.viewModeVar, self.overlayOnOriginalVar
        ):
            bind_var(var)

        self.pickValueVar.trace_add("write", self._onParamChanged)
        self.pickTolVar.trace_add("write", self._onParamChanged)

        self.leftCanvas.bind("<Button-1>", self._onLeftClickPick)

        # IMPORTANT: debounced to avoid recompute storms on resize
        self.leftCanvas.bind("<Configure>", debounce(self.leftCanvas, 150)(lambda e: self._showCurrent()))
        self.rightCanvas.bind("<Configure>", debounce(self.rightCanvas, 150)(lambda e: self._onParamChanged()))
        # After binding other vars...
        self.methodVar.trace_add("write", lambda *_: self._refreshCursor())
        self.pickModeVar.trace_add("write", lambda *_: self._refreshCursor())
        
    def _refreshCursor(self):
        # Crosshair when method is 'pick' or explicit pick mode is enabled
        use_cross = (self.methodVar.get() == "pick") or bool(self.pickModeVar.get())
        try:
            self.leftCanvas.config(cursor="crosshair" if use_cross else "")
        except Exception:
            pass


    def _onParamChanged(self, *args):
        # Only recompute automatically if Auto preview is ON
        if self.autoPreviewVar.get():
            self._recomputePreview()
        else:
            # Light hint in manual mode
            self.statusVar.set("Params changed (manual mode). Click Recompute.")

    def recomputeNow(self):
        """Recompute preview with a modal progress bar (blocks UI)."""
        dlg = BusyDialog(self, title="Recomputing preview…", mode="indeterminate")
        try:
            self._recomputePreview()
        finally:
            dlg.close()



    # ----------------- Navigation -----------------

    def prevImage(self):
        if self.currentIndex > 0:
            self.currentIndex -= 1
            self._showCurrent()

    def nextImage(self):
        if self.currentIndex < len(self.images) - 1:
            self.currentIndex += 1
            self._showCurrent()

    # ----------------- Display -----------------

    def _npToPil(self, arr):
        a = np.asarray(arr)
        if a.ndim == 3 and a.shape[2] == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            return Image.fromarray(a, mode="RGB")
        if a.ndim == 2:
            if a.dtype != np.uint8:
                a = cv2.normalize(a.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return Image.fromarray(a, mode="L")
        if a.ndim == 3 and a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(a, mode="RGBA")
        g = a.mean(axis=2).astype(np.uint8) if a.ndim == 3 else a.astype(np.uint8)
        return Image.fromarray(g, mode="L")

    def _displayOn(self, canvas, pilImg, keepScaleAttr, photoAttr):
        canvas.update_idletasks()
        cw, ch = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        iw, ih = pilImg.size
        s = min(cw / float(iw), ch / float(ih)); s = min(s, 1.0)
        new_w = max(1, int(round(iw * s))); new_h = max(1, int(round(ih * s)))
        if (new_w, new_h) != (iw, ih):
            pilImg = pilImg.resize((new_w, new_h), resample=RESAMPLE_LANCZOS)
        ox = (cw - new_w) // 2
        oy = (ch - new_h) // 2
        canvas.delete("all")
        photo = ImageTk.PhotoImage(pilImg)
        setattr(self, photoAttr, photo)
        setattr(self, keepScaleAttr, s)
        # remember placement so we can map clicks and draw markers
        if keepScaleAttr == "_leftScale":
            self._leftOrigin = (ox, oy)
            self._leftDispSize = (new_w, new_h)
        elif keepScaleAttr == "_rightScale":
            self._rightOrigin = (ox, oy)
            self._rightDispSize = (new_w, new_h)
        canvas.create_image(ox, oy, anchor="nw", image=photo)

    def _drawPickMarker(self):
        # Draw small crosshair on left canvas at last pick point (canvas coords)
        self.leftCanvas.delete("pick_marker")
        if not self._pickPointCanvas:
            return
        x, y = self._pickPointCanvas
        r = 6
        self.leftCanvas.create_line(x - r, y, x + r, y, fill="#00eaff", width=2, tags="pick_marker")
        self.leftCanvas.create_line(x, y - r, x, y + r, fill="#00eaff", width=2, tags="pick_marker")


    def _showCurrent(self):
        if not hasattr(self, "currentIndex"):
            self.currentIndex = 0
        img = self.images[self.currentIndex]
        self._displayOn(self.leftCanvas, self._npToPil(img), "_leftScale", "_leftPhoto")

        if self.autoPreviewVar.get():
            self._recomputePreview()
        else:
            self.statusVar.set("Manual mode: press Recompute to update preview.")

        self.statusVar.set(f"{os.path.basename(self.paths[self.currentIndex])}  ({self.currentIndex+1}/{len(self.images)})")

    # ----------------- Parameters -----------------

    def _currentThreshParams(self) -> Dict[str, float | int | bool | str]:
        m = self.methodVar.get()
        params = dict(self.settings["common"])
        params["polarity"] = self.polarityVar.get()
        if m != "pick":
            if not self.useDefaultsVar.get():
                params.update(self.settings[m])
            else:
                params.update(DEFAULTS[m])
        else:
            params.update(self.settings["pick"])
            params["pickTolerance"] = int(self.pickTolVar.get())
            params["pickValue"] = int(self.pickValueVar.get())
        return params

    def _currentSepParams(self) -> Dict[str, float | int | bool | str]:
        s = self.settings["separation"]
        params = {
            "method": self.sepMethodVar.get(),
            "fillHoles": bool(self.fillHolesVar.get()),
            "minAreaPx": int(self.minAreaVar.get()),
            "distanceBlurK": int(self.distanceBlurVar.get()),
            "peakMinDistance": int(self.peakMinDistVar.get()),
            "peakRelThreshold": float(self.peakRelThrVar.get()),
            "connectivity": int(self.connectivityVar.get()),
            "clearBorder": bool(self.clearBorderVar.get()),
            "overlayAlpha": float(self.overlayAlphaVar.get()),
        }
        return params

    def _onParamChanged(self, *args):
        # Only recompute automatically if Auto preview is ON
        if self.autoPreviewVar.get():
            self._recomputePreview()
        else:
            # Give a gentle hint in manual mode
            self.statusVar.set("Params changed (manual mode). Click Recompute.")

    # ----------------- Preview -----------------

    def _recomputePreview(self):
        if not self.images:
            return
        img = self.images[self.currentIndex]
        method = self.methodVar.get()
        tparams = self._currentThreshParams()
        sparams = self._currentSepParams()

        try:
            if method == "pick" and "pickValue" not in tparams:
                # wait until user picks something
                gray = self._prepGrayLocal(img)
                preview = np.zeros_like(gray)
                labels = None
            else:
                binary, labels, _ = runSeparationPipeline(img, tparams, sparams)
                preview = binary if (self.viewModeVar.get() == "binary" or labels is None) else \
                          self._labelsPreview(labels, img, sparams)
        except Exception as e:
            self.statusVar.set(f"Error: {e}")
            return

        self._lastBinary = binary if 'binary' in locals() else None
        self._lastLabels = labels if 'labels' in locals() else None

        # Left = original, Right = preview
        self._displayOn(self.leftCanvas, self._npToPil(img), "_leftScale", "_leftPhoto")
        self._displayOn(self.rightCanvas, self._npToPil(preview), "_rightScale", "_rightPhoto")

    def _labelsPreview(self, labels: np.ndarray, img, sparams) -> np.ndarray:
        alpha = float(np.clip(sparams.get("overlayAlpha", 0.45), 0.0, 1.0))
        gray = self._prepGrayLocal(img) if self.overlayOnOriginalVar.get() else None
        return labelsToColor(labels, bgGray=gray, alpha=alpha)

    def _prepGrayLocal(self, src) -> np.ndarray:
        a = src
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        if a.dtype != np.uint8:
            a = cv2.normalize(a.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return a


    # ----------------- Eyedropper -----------------

    def _onLeftClickPick(self, event):
        # Allow picking when in pick method OR when the explicit toggle is on
        if not (self.methodVar.get() == "pick" or self.pickModeVar.get()):
            return

        # Map canvas click -> image pixel
        img = self.images[self.currentIndex]
        gray = self._prepGrayLocal(img)

        # Get placement info of the left image on the canvas
        ox, oy = getattr(self, "_leftOrigin", (0,0))
        s = getattr(self, "_leftScale", 1.0)
        # Position within the displayed image
        dx = event.x - ox
        dy = event.y - oy
        # Reject clicks outside the image bounds
        if dx < 0 or dy < 0:
            return
        disp_w, disp_h = getattr(self, "_leftDispSize", (gray.shape[1], gray.shape[0]))
        if dx >= disp_w or dy >= disp_h:
            return
        # Convert to image pixel coords
        px = int(dx / max(s, 1e-6))
        py = int(dy / max(s, 1e-6))
        h, w = gray.shape[:2]
        if px < 0 or py < 0 or px >= w or py >= h:
            return

        # Update pick value and visual marker
        val = int(gray[py, px])
        self.pickValueVar.set(val)
        self._pickPointCanvas = (event.x, event.y)
        self._drawPickMarker()
        self.statusVar.set(f"Picked gray={val}")

        # Only recompute automatically if Auto preview is ON
        if self.autoPreviewVar.get():
            self._recomputePreview()
        else:
            self.statusVar.set(f"Picked gray={val} (manual mode). Click Recompute.")


    # ----------------- Advanced dialogs -----------------

    def openAdvancedDialog(self):
        method = self.methodVar.get()
        self._openAdvancedFor(method)

    def _openAdvancedFor(self, method: str):
        dlg = tk.Toplevel(self)
        dlg.title(f"Advanced Settings – {method.capitalize()}")
        dlg.transient(self); dlg.grab_set()

        # Common preprocessing
        row = 0
        cmn = self.settings["common"]
        useCLAHEVar = tk.BooleanVar(value=bool(cmn["useCLAHE"]))
        claheClipVar = tk.DoubleVar(value=float(cmn["claheClip"]))
        claheTileVar = tk.IntVar(value=int(cmn["claheTile"]))
        medianKVar = tk.IntVar(value=int(cmn["medianK"]))
        gaussianKVar = tk.IntVar(value=int(cmn["gaussianK"]))
        applyOpenCloseVar = tk.BooleanVar(value=bool(cmn["applyOpenClose"]))
        morphKVar = tk.IntVar(value=int(cmn["morphK"]))

        ttk.Label(dlg, text="Common preprocessing").grid(row=row, column=0, columnspan=4, pady=(8,4), sticky="w"); row += 1
        ttk.Checkbutton(dlg, text="CLAHE", variable=useCLAHEVar).grid(row=row, column=0, sticky="w", padx=8)
        ttk.Label(dlg, text="Clip").grid(row=row, column=1, sticky="e"); ttk.Entry(dlg, textvariable=claheClipVar, width=8).grid(row=row, column=2, sticky="w"); row += 1
        ttk.Label(dlg, text="Tile").grid(row=row, column=1, sticky="e"); ttk.Entry(dlg, textvariable=claheTileVar, width=8).grid(row=row, column=2, sticky="w"); row += 1
        ttk.Label(dlg, text="Median k (0=off)").grid(row=row, column=1, sticky="e"); ttk.Entry(dlg, textvariable=medianKVar, width=8).grid(row=row, column=2, sticky="w"); row += 1
        ttk.Label(dlg, text="Gaussian k (0=off)").grid(row=row, column=1, sticky="e"); ttk.Entry(dlg, textvariable=gaussianKVar, width=8).grid(row=row, column=2, sticky="w"); row += 1
        ttk.Checkbutton(dlg, text="Preview cleanup (open+close)", variable=applyOpenCloseVar).grid(row=row, column=0, sticky="w", padx=8); row += 1
        ttk.Label(dlg, text="Morph k").grid(row=row, column=1, sticky="e"); ttk.Entry(dlg, textvariable=morphKVar, width=8).grid(row=row, column=2, sticky="w"); row += 1

        # Method-specific
        if method == "adaptive":
            ad = self.settings["adaptive"]
            adaptiveBlockVar = tk.IntVar(value=int(ad["adaptiveBlock"]))
            adaptiveCVar = tk.IntVar(value=int(ad["adaptiveC"]))
            ttk.Label(dlg, text="Adaptive").grid(row=row, column=0, columnspan=4, pady=(8,4), sticky="w"); row += 1
            ttk.Label(dlg, text="Block (odd)").grid(row=row, column=1, sticky="e"); ttk.Entry(dlg, textvariable=adaptiveBlockVar, width=8).grid(row=row, column=2, sticky="w"); row += 1
            ttk.Label(dlg, text="C").grid(row=row, column=1, sticky="e"); ttk.Entry(dlg, textvariable=adaptiveCVar, width=8).grid(row=row, column=2, sticky="w"); row += 1
        elif method == "percentile":
            pe = self.settings["percentile"]
            percentileVar = tk.DoubleVar(value=float(pe["percentile"]))
            ttk.Label(dlg, text="Percentile").grid(row=row, column=0, columnspan=4, pady=(8,4), sticky="w"); row += 1
            ttk.Label(dlg, text="Percentile (0..100)").grid(row=row, column=1, sticky="e"); ttk.Entry(dlg, textvariable=percentileVar, width=8).grid(row=row, column=2, sticky="w"); row += 1
        elif method == "pick":
            pk = self.settings["pick"]
            pickTolVar = tk.IntVar(value=int(pk["pickTolerance"]))
            ttk.Label(dlg, text="Pick").grid(row=row, column=0, columnspan=4, pady=(8,4), sticky="w"); row += 1
            ttk.Label(dlg, text="Tolerance ±").grid(row=row, column=1, sticky="e"); ttk.Entry(dlg, textvariable=pickTolVar, width=8).grid(row=row, column=2, sticky="w"); row += 1

        # Buttons
        row += 1
        btns = ttk.Frame(dlg); btns.grid(row=row, column=0, columnspan=4, pady=(10,8))
        def on_ok():
            cmn = self.settings["common"]
            cmn["useCLAHE"] = bool(useCLAHEVar.get())
            cmn["claheClip"] = float(claheClipVar.get())
            cmn["claheTile"] = int(claheTileVar.get())
            cmn["medianK"] = int(medianKVar.get())
            cmn["gaussianK"] = int(gaussianKVar.get())
            cmn["applyOpenClose"] = bool(applyOpenCloseVar.get())
            cmn["morphK"] = int(morphKVar.get())

            if method == "adaptive":
                self.settings["adaptive"]["adaptiveBlock"] = int(adaptiveBlockVar.get())
                self.settings["adaptive"]["adaptiveC"] = int(adaptiveCVar.get())
            elif method == "percentile":
                self.settings["percentile"]["percentile"] = float(percentileVar.get())
            elif method == "pick":
                self.settings["pick"]["pickTolerance"] = int(pickTolVar.get())

            dlg.destroy()
            self.useDefaultsVar.set(False)
            self._recomputePreview()
        def on_cancel():
            dlg.destroy()

        ttk.Button(btns, text="OK", command=on_ok).pack(side="left", padx=8)
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="left", padx=8)

    def _updatePickPanelVisibility(self):
        if self.methodVar.get() == "pick":
            self.pickPanel.pack(side="top", fill="x", padx=6)
        else:
            self.pickPanel.forget()


    def exportMeasurements(self):
        """
        Run stereology on current labels array(s) and export a CSV.
        Also optionally save colorized label previews if user chooses a folder.
        """
        # sanity
        if not any(L is not None for L in getattr(self, "labels", [])):
            messagebox.showwarning("Stereology", "No labels available. Apply to current or all first.")
            return

        # choose CSV path
        out_csv = filedialog.asksaveasfilename(
            title="Save pore measurements CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not out_csv:
            return

        # run measurements across images
        try:
            props = stereology.measure_dataset(self.labels, getattr(self, "scales", None))
            stereology.save_props_csv(out_csv, props)
        except Exception as e:
            messagebox.showerror("Export", f"Failed to compute/export measurements:\n{e}")
            return

        # offer to save color previews (optional)
        if messagebox.askyesno("Export", "Measurements saved.\n\nAlso save colorized label previews?"):
            out_dir = filedialog.askdirectory(title="Choose folder for colorized labels")
            if out_dir:
                saved = 0
                for i, L in enumerate(self.labels):
                    if L is None:
                        continue
                    # background for overlay if user is in "overlay on original" mode:
                    bg = None
                    if self.overlayOnOriginalVar.get():
                        bg = self._prepGrayLocal(self.images[i])
                    color = stereology.colorize_labels(L, seed=123 + i, bg_gray=bg, alpha=float(self.overlayAlphaVar.get()))
                    name = os.path.splitext(os.path.basename(self.paths[i]))[0]
                    cv2.imwrite(os.path.join(out_dir, f"{name}_labels_color.png"), color)
                    saved += 1
                messagebox.showinfo("Export", f"Saved {saved} colorized label image(s).")
            else:
                messagebox.showinfo("Export", "Measurements saved (no images exported).")
        else:
            messagebox.showinfo("Export", "Measurements saved.")

    
    def openStereology(self):
        if not any(L is not None for L in getattr(self, "labels", [])):
            messagebox.showwarning("Stereology", "No labels available. Apply to current or all first.")
            return
        try:
            stereology_gui.StereologyWindow(
                parent=self,
                images=self.images,
                labels=self.labels,
                paths=self.paths,
                scales=getattr(self, "scales", None),
                start_index=self.currentIndex if hasattr(self, "currentIndex") else 0
            )
        except Exception as e:
            messagebox.showerror("Stereology", f"Failed to open stereology window:\n{e}")

    # ----------------- Apply / Save -----------------

    def applyToCurrent(self):
        # Must have a computed preview
        if not hasattr(self, "_lastBinary"):
            messagebox.showwarning("Apply", "Nothing to apply yet.")
            return

        idx = self.currentIndex
        n = len(self.images)
        if not hasattr(self, "binaries") or len(self.binaries) != n:
            self.binaries = [None] * n
        if not hasattr(self, "labels") or len(self.labels) != n:
            self.labels = [None] * n

        dlg = BusyDialog(self, title="Applying to current (full-res)…", mode="indeterminate")
        try:
            try: self.attributes("-disabled", True)
            except Exception: pass
            try: self.config(cursor="watch")
            except Exception: pass

            self.binaries[idx] = self._lastBinary.copy() if getattr(self, "_lastBinary", None) is not None else None
            self.labels[idx]   = self._lastLabels.copy()  if getattr(self, "_lastLabels", None)  is not None else None

            cb = getattr(self, "resultsCallback", None)
            if callable(cb):
                try:
                    cb(self.binaries)
                except Exception as e:
                    print("resultsCallback error:", e)

            try: self.update_idletasks()
            except Exception: pass

        finally:
            dlg.close()
            try: self.config(cursor="")
            except Exception: pass
            try: self.attributes("-disabled", False)
            except Exception: pass

        messagebox.showinfo("Apply", f"Applied to current image ({idx+1}/{n}).")



    def applyToAll(self):
        """Apply current settings to all images using parallel processing."""
        if not self.images:
            return

        n = len(self.images)
        if not hasattr(self, "binaries") or len(self.binaries) != n:
            self.binaries = [None] * n
        if not hasattr(self, "labels") or len(self.labels) != n:
            self.labels = [None] * n

        tparams = self._currentThreshParams()
        sparams = self._currentSepParams()

        # Determine worker count (use half of CPU cores for responsiveness)
        import os as _os
        max_workers = max(1, (_os.cpu_count() or 4) // 2)
        # For small batches, sequential is faster (no process spawn overhead)
        if n <= 3:
            max_workers = 1

        dlg = BusyDialog(self, title=f"Applying to all ({max_workers} workers)…", mode="determinate", maximum=n)
        
        def progress_cb(completed: int, total: int):
            try:
                dlg.set_progress(completed)
                self.update_idletasks()
            except Exception:
                pass

        try:
            try: self.attributes("-disabled", True)
            except Exception: pass
            try: self.config(cursor="watch")
            except Exception: pass

            # Use parallel batch processing
            binaries, labels_list, _ = process_batch_parallel(
                images=self.images,
                thresh_params=tparams,
                sep_params=sparams,
                scales=getattr(self, "scales", None),
                measure=False,  # measurement done separately in stereology
                max_workers=max_workers,
                progress_callback=progress_cb
            )

            # Copy results
            for i in range(n):
                self.binaries[i] = binaries[i]
                self.labels[i] = labels_list[i]

            saved = sum(1 for b in self.binaries if b is not None)

        except Exception as e:
            messagebox.showerror("Apply", f"Batch processing failed:\n{e}")
            saved = 0
        finally:
            dlg.close()
            try: self.config(cursor="")
            except Exception: pass
            try: self.attributes("-disabled", False)
            except Exception: pass

        cb = getattr(self, "resultsCallback", None)
        if callable(cb):
            try:
                cb(self.binaries)
            except Exception as e:
                print("resultsCallback error:", e)

        messagebox.showinfo("Apply", f"Applied to {saved}/{n} images (parallel, {max_workers} workers).")


    def saveMasks(self):
        if not any(self.binaries):
            messagebox.showwarning("Save", "No masks to save. Apply to current or all first.")
            return
        outDir = filedialog.askdirectory(title="Choose output folder for masks")
        if not outDir:
            return
        saved = 0
        for i, m in enumerate(self.binaries):
            if m is None: continue
            name = os.path.splitext(os.path.basename(self.paths[i]))[0]
            path = os.path.join(outDir, f"{name}_mask.png")
            cv2.imwrite(path, m)
            saved += 1
        messagebox.showinfo("Save", f"Saved {saved} mask(s) to {outDir}.")

    def saveLabels(self):
        if not any(self.labels):
            messagebox.showwarning("Save", "No labels to save. Apply first.")
            return
        outDir = filedialog.askdirectory(title="Choose output folder for labels")
        if not outDir:
            return
        saved = 0
        for i, L in enumerate(self.labels):
            if L is None: continue
            name = os.path.splitext(os.path.basename(self.paths[i]))[0]
            # Save color preview and raw .npy
            color = labelsToColor(L, bgGray=None, alpha=0.0)
            cv2.imwrite(os.path.join(outDir, f"{name}_labels_color.png"), color)
            np.save(os.path.join(outDir, f"{name}_labels.npy"), L.astype(np.int32))
            saved += 1
        messagebox.showinfo("Save", f"Saved {saved} labeled file(s) to {outDir}.")
    
    
