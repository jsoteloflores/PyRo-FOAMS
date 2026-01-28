# stereology_gui.py
# Standalone stereology window: view colorized labels, compute metrics, plot histograms, export CSV

from __future__ import annotations

import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageTk

matplotlib.use("TkAgg")  # embed in Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Package imports
from ..core.stereology import (
    PoreProps,
    colorize_labels,
    measure_dataset,
    measure_labels,
    save_props_csv,
)
from .widgets import debounce

# ------------ utils: PIL conversion ------------

def _np_to_pil(arr: np.ndarray) -> Image.Image:
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


def _prep_gray(src: np.ndarray) -> np.ndarray:
    a = src
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if a.dtype != np.uint8:
        a = cv2.normalize(a.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return a


# ------------ main window ------------

class StereologyWindow(tk.Toplevel):
    """
    Separate window to visualize colorized labels, compute per-pore measurements,
    show histograms, and export CSV. Works on current image or aggregate of all images.
    """
    def __init__(
        self,
        parent: tk.Misc,
        images: List[np.ndarray],
        labels: List[Optional[np.ndarray]],
        paths: Optional[List[str]] = None,
        scales: Optional[List[Optional[Dict[str, float | str]]]] = None,
        start_index: int = 0
    ):
        super().__init__(parent)
        self.title("PyFOAMS – Stereology")
        self.transient(parent)
        self.grab_set()

        # data
        self.images = images
        self.labels = labels
        self.paths = paths or [f"Image {i+1}" for i in range(len(images))]
        self.scales = scales or [None] * len(images)

        self.index = max(0, min(start_index, len(images) - 1))

        # cached color overlays (to avoid recolorizing every time)
        self._color_cache: Dict[int, np.ndarray] = {}

        # ------------- UI state -------------
        self.aggregateVar   = tk.StringVar(value="current")  # "current" | "all"
        self.metricVar      = tk.StringVar(value="eq_diam")  # default histogram metric
        self.binsVar        = tk.IntVar(value=30)
        self.logYVar        = tk.BooleanVar(value=False)
        self.excludeBorderVar = tk.BooleanVar(value=False)

        self.alphaVar       = tk.DoubleVar(value=0.45)       # overlay alpha
        self.overlayVar     = tk.BooleanVar(value=True)      # overlay on original
        self.seedVar        = tk.IntVar(value=123)

        # units: "px" or "units"
        self.unitsModeVar   = tk.StringVar(value="px")
        self._unitName      = self._infer_common_unit_name()

        # area filter (pixel-based; when units selected, we apply in units^2)
        self.minAreaVar     = tk.DoubleVar(value=0.0)
        self.maxAreaVar     = tk.DoubleVar(value=0.0)  # 0 => no upper cap

        # measured data cache for current settings
        self._last_props: List[PoreProps] = []

        # build UI and initial paint
        self._build_ui()
        self._refresh_units_controls()
        self._update_view()
        self._compute_and_plot()

        self.geometry("1280x820")
        self.minsize(1000, 680)

    # ------------- UI -------------

    def _build_ui(self):
        # top bar
        top = ttk.Frame(self); top.pack(side="top", fill="x", padx=6, pady=6)

        ttk.Button(top, text="Prev", command=self._prev).pack(side="left")
        ttk.Button(top, text="Next", command=self._next).pack(side="left", padx=(4, 12))

        ttk.Label(top, text="Aggregation:").pack(side="left")
        ttk.Radiobutton(top, text="Current image", value="current", variable=self.aggregateVar,
                        command=self._on_settings_changed).pack(side="left")
        ttk.Radiobutton(top, text="All images", value="all", variable=self.aggregateVar,
                        command=self._on_settings_changed).pack(side="left", padx=(0, 10))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(top, text="Metric:").pack(side="left")
        metric_cb = ttk.Combobox(top, width=16, textvariable=self.metricVar, state="readonly",
                                 values=[
                                     "eq_diam", "area", "circularity",
                                     "feret_max", "feret_min",
                                     "major_axis", "minor_axis",
                                 ])
        metric_cb.pack(side="left")
        metric_cb.bind("<<ComboboxSelected>>", lambda e: self._compute_and_plot())
        metric_cb.bind("<<ComboboxSelected>>", lambda e: self._compute_and_plot())

        ttk.Button(top, text="Help (?)", command=self._open_metrics_help)\
        .pack(side="left", padx=(6, 10))

        ttk.Label(top, text="Bins:").pack(side="left", padx=(10,2))
        ttk.Spinbox(top, from_=5, to=200, textvariable=self.binsVar, width=5,
                    command=self._compute_and_plot).pack(side="left")
        ttk.Checkbutton(top, text="Log Y", variable=self.logYVar,
                        command=self._compute_and_plot).pack(side="left", padx=(8, 10))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Checkbutton(top, text="Exclude border-touching", variable=self.excludeBorderVar,
                        command=self._compute_and_plot).pack(side="left", padx=(0, 10))

        ttk.Label(top, text="Area filter (min/max)").pack(side="left")
        ttk.Entry(top, textvariable=self.minAreaVar, width=8).pack(side="left")
        ttk.Entry(top, textvariable=self.maxAreaVar, width=8).pack(side="left", padx=(2, 10))
        ttk.Button(top, text="Apply Filters", command=self._compute_and_plot).pack(side="left")

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)

        # units control
        self.unitsLabel = ttk.Label(top, text="Units:")
        self.unitsLabel.pack(side="left")
        self.unitsCombo = ttk.Combobox(top, width=10, textvariable=self.unitsModeVar, state="readonly",
                                       values=["px", "units"])
        self.unitsCombo.pack(side="left")
        self.unitsCombo.bind("<<ComboboxSelected>>", lambda e: self._compute_and_plot())

        # right side actions
        actions = ttk.Frame(self); actions.pack(side="top", fill="x", padx=6, pady=(0,6))
        ttk.Button(actions, text="Export CSV…", command=self._export_csv).pack(side="left")
        ttk.Button(actions, text="Save Figure…", command=self._save_figure).pack(side="left", padx=(6,12))

        ttk.Label(actions, text="Overlay α").pack(side="left")
        ttk.Scale(actions, from_=0.0, to=1.0, orient="horizontal", length=120,
                  variable=self.alphaVar, command=lambda _=None: self._update_view()).pack(side="left", padx=(0,6))
        ttk.Checkbutton(actions, text="Overlay on original", variable=self.overlayVar,
                        command=self._update_view).pack(side="left", padx=(0,8))
        ttk.Label(actions, text="Color seed").pack(side="left")
        ttk.Spinbox(actions, from_=-999999, to=999999, width=8, textvariable=self.seedVar,
                    command=self._invalidate_color_cache).pack(side="left")

        # main split: left viewer, right notebook (plot + table)
        main = ttk.Frame(self); main.pack(side="top", fill="both", expand=True, padx=6, pady=6)

        # left viewer
        left = ttk.Frame(main); left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, bg="#151515", highlightthickness=1, highlightbackground="#333")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Configure>", debounce(self.canvas, 150)(lambda e: self._update_view()))

        # right notebook (plot + table)
        right = ttk.Notebook(main); right.pack(side="left", fill="both", expand=True, padx=(8,0))

        # plot tab
        plot_tab = ttk.Frame(right)
        right.add(plot_tab, text="Histogram")

        self.fig = Figure(figsize=(5,4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=plot_tab)
        self.canvas_mpl.draw()
        self.canvas_mpl.get_tk_widget().pack(side="top", fill="both", expand=True)
        try:
            toolbar = NavigationToolbar2Tk(self.canvas_mpl, plot_tab)
            toolbar.update()
        except Exception:
            pass

        # table tab
        table_tab = ttk.Frame(right)
        right.add(table_tab, text="Table")

        cols = [
            "image","label","area_px","eq_diam_px","circularity",
            "feret_max_px","feret_min_px","major_px","minor_px","aspect","orient_deg","border"
        ]
        self.tree = ttk.Treeview(table_tab, columns=cols, show="headings", height=12)
        for c, w in zip(cols,
                        [60,50,80,90,90,90,90,90,90,80,90,60]):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        vsb = ttk.Scrollbar(table_tab, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_tab, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.pack(side="top", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        # status line
        self.statusVar = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.statusVar).pack(side="bottom", fill="x", padx=8, pady=(0,6))

    # ------------- helpers -------------

    def _infer_common_unit_name(self) -> Optional[str]:
        """Return a common unit name if all scaled entries share one; else None."""
        names = []
        for s in (self.scales or []):
            n = None
            if isinstance(s, dict) and "unitsPerPx" in s:
                if s.get("unitName"):
                    n = str(s["unitName"])
                else:
                    n = ""
            names.append(n)
        # consider only images that actually have a scale
        have = [n for n in names if n is not None]
        if not have:
            return None
        # all same?
        if all(n == have[0] for n in have):
            return have[0] or ""  # empty string ok
        return None

    def _refresh_units_controls(self):
        """Enable/disable 'units' choice depending on available scales."""
        if self._unitName is None:
            # no consistent scaling → pixels only
            self.unitsModeVar.set("px")
            self.unitsCombo.configure(values=["px"])
            self.unitsLabel.configure(text="Units: px")
        else:
            # allow px or scaled
            self.unitsCombo.configure(values=["px", "units"])
            txt = f"Units: ({'px' if self.unitsModeVar.get()=='px' else (self._unitName or 'units')})"
            self.unitsLabel.configure(text=txt)

    def _invalidate_color_cache(self):
        self._color_cache.clear()
        self._update_view()

    # ------------- viewer -------------

    def _current_colorized(self) -> Optional[np.ndarray]:
        """Return BGR overlay for current image index (cached)."""
        i = self.index
        L = self.labels[i] if (0 <= i < len(self.labels)) else None
        if L is None:
            return None
        if i in self._color_cache:
            return self._color_cache[i]

        bg = _prep_gray(self.images[i]) if self.overlayVar.get() else None
        color = colorize_labels(
            L, seed=int(self.seedVar.get()), bg_gray=bg, alpha=float(self.alphaVar.get())
        )
        self._color_cache[i] = color
        return color

    def _update_view(self):
        self.statusVar.set(self._status_text())
        img = self._current_colorized()
        if img is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width()//2,
                self.canvas.winfo_height()//2,
                text="No labels for this image.\n(Apply segmentation in Processing first.)",
                fill="#cccccc"
            )
            return

        # if overlay toggle or alpha changes, rebuild cache for current index only
        i = self.index
        bg = _prep_gray(self.images[i]) if self.overlayVar.get() else None
        self._color_cache[i] = colorize_labels(
            self.labels[i], seed=int(self.seedVar.get()), bg_gray=bg, alpha=float(self.alphaVar.get())
        )
        img = self._color_cache[i]

        pil = _np_to_pil(img)
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        iw, ih = pil.size
        s = min(cw/float(iw), ch/float(ih)); s = min(s, 1.0)
        new_w = max(1, int(round(iw*s))); new_h = max(1, int(round(ih*s)))
        if (new_w, new_h) != (iw, ih):
            pil = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.BICUBIC)
        ox = (cw - new_w)//2; oy = (ch - new_h)//2

        self.canvas.delete("all")
        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo)

    def _status_text(self) -> str:
        name = os.path.basename(self.paths[self.index]) if self.paths else f"Image {self.index+1}"
        n_all = sum(1 for L in self.labels if L is not None)
        return f"{name}   ({self.index+1}/{len(self.images)})   |   labeled images: {n_all}/{len(self.images)}"

    # ------------- measurement + plotting -------------

    def _collect_props(self) -> List[PoreProps]:
        # gather per aggregation mode
        if self.aggregateVar.get() == "current":
            L = self.labels[self.index]
            if L is None:
                return []
            scale = self.scales[self.index] if self.scales and self.index < len(self.scales) else None
            props = measure_labels(L, image_index=self.index, scale=scale)
        else:
            props = measure_dataset(self.labels, self.scales)

        # filters
        if self.excludeBorderVar.get():
            props = [p for p in props if not p.touches_border]

        # area min/max filter
        amin = float(self.minAreaVar.get() or 0.0)
        amax = float(self.maxAreaVar.get() or 0.0)

        use_units = (self.unitsModeVar.get() == "units") and (self._unitName is not None)
        def area_val(p: PoreProps) -> float:
            return p.area_units2 if use_units and (p.area_units2 is not None) else float(p.area_px)

        if amin > 0.0:
            props = [p for p in props if area_val(p) >= amin]
        if amax > 0.0:
            props = [p for p in props if area_val(p) <= amax]

        return props

    def _values_for_metric(self, props: List[PoreProps]) -> Tuple[np.ndarray, str, str]:
        """
        Return array of metric values, xlabel, ylabel.
        """
        use_units = (self.unitsModeVar.get() == "units") and (self._unitName is not None)
        metric = self.metricVar.get()

        vals = []
        label = ""
        if metric == "eq_diam":
            for p in props:
                v = p.eq_diam_units if use_units and (p.eq_diam_units is not None) else p.eq_diam_px
                if v is not None and not math.isnan(v):
                    vals.append(float(v))
            label = f"Equivalent diameter ({self._unitName or 'units'})" if use_units else "Equivalent diameter (px)"
        elif metric == "area":
            for p in props:
                v = p.area_units2 if use_units and (p.area_units2 is not None) else float(p.area_px)
                vals.append(float(v))
            label = f"Area ({self._unitName or 'units'}²)" if use_units else "Area (px²)"
        elif metric == "circularity":
            vals = [float(p.circularity) for p in props if p.circularity is not None and not math.isnan(p.circularity)]
            label = "Circularity (4πA/P²)"
        elif metric == "feret_max":
            for p in props:
                v = p.feret_max_units if use_units and (p.feret_max_units is not None) else p.feret_max_px
                if v is not None:
                    vals.append(float(v))
            label = f"Max Feret ({self._unitName or 'units'})" if use_units else "Max Feret (px)"
        elif metric == "feret_min":
            for p in props:
                v = p.feret_min_units if use_units and (p.feret_min_units is not None) else p.feret_min_px
                if v is not None:
                    vals.append(float(v))
            label = f"Min Feret ({self._unitName or 'units'})" if use_units else "Min Feret (px)"
        elif metric == "major_axis":
            for p in props:
                v = p.major_axis_units if use_units and (p.major_axis_units is not None) else p.major_axis_px
                if v is not None:
                    vals.append(float(v))
            label = f"Ellipse major ({self._unitName or 'units'})" if use_units else "Ellipse major (px)"
        elif metric == "minor_axis":
            for p in props:
                v = p.minor_axis_units if use_units and (p.minor_axis_units is not None) else p.minor_axis_px
                if v is not None:
                    vals.append(float(v))
            label = f"Ellipse minor ({self._unitName or 'units'})" if use_units else "Ellipse minor (px)"
        else:
            vals = []
            label = metric

        return np.asarray(vals, dtype=np.float64), label, "Count"

    def _compute_and_plot(self):
        props = self._collect_props()
        self._last_props = props  # cache for table/export

        # table refresh (keep it simple)
        for it in self.tree.get_children():
            self.tree.delete(it)
        for p in props:
            self.tree.insert("", "end", values=(
                p.image_index, p.label, p.area_px, f"{p.eq_diam_px:.3f}" if p.eq_diam_px is not None else "",
                f"{p.circularity:.3f}" if p.circularity is not None else "",
                f"{p.feret_max_px:.3f}" if p.feret_max_px is not None else "",
                f"{p.feret_min_px:.3f}" if p.feret_min_px is not None else "",
                f"{p.major_axis_px:.3f}" if p.major_axis_px is not None else "",
                f"{p.minor_axis_px:.3f}" if p.minor_axis_px is not None else "",
                f"{p.aspect_ratio:.3f}" if p.aspect_ratio is not None else "",
                f"{p.orientation_deg:.1f}" if p.orientation_deg is not None else "",
                "yes" if p.touches_border else "no"
            ))

        # histogram
        vals, xlabel, ylabel = self._values_for_metric(props)
        self.ax.clear()
        if vals.size == 0:
            self.ax.text(0.5, 0.5, "No data for selected metric/filters", ha="center", va="center")
            self.ax.set_xticks([]); self.ax.set_yticks([])
        else:
            bins = max(5, min(400, int(self.binsVar.get())))
            self.ax.hist(vals, bins=bins, edgecolor="black")
            if self.logYVar.get():
                self.ax.set_yscale("log")
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
            self.ax.grid(True, alpha=0.25, linestyle="--")
            self.ax.set_title(f"N = {vals.size}")
        self.canvas_mpl.draw_idle()

    def _open_metrics_help(self):
        # Modal, scrollable help window
        dlg = tk.Toplevel(self)
        dlg.title("Stereology metrics – Help")
        dlg.transient(self)
        dlg.grab_set()
        dlg.geometry("620x540")

        container = ttk.Frame(dlg, padding=12)
        container.pack(fill="both", expand=True)

        txt = tk.Text(container, wrap="word")
        vsb = ttk.Scrollbar(container, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)

        HELP = (
            "METRICS (per labeled pore)\n"
            "\n"
            "• Area (area / area_px / area_units2)\n"
            "  Pixel count inside the pore. If a scale is provided, it’s converted to units².\n"
            "\n"
            "• Equivalent diameter (eq_diam / eq_diam_px / eq_diam_units)\n"
            "  Diameter of a circle with the same area:\n"
            "      eq_diam = sqrt(4 · A / π)\n"
            "  A is area in px² (or in units² when scaled). Useful as a size proxy.\n"
            "\n"
            "• Circularity (4πA / P²)\n"
            "  1.0 for a perfect circle; smaller values indicate elongation/irregularity.\n"
            "  P is perimeter along the pixel boundary (discrete estimate).\n"
            "\n"
            "• Max Feret / Min Feret (feret_max / feret_min)\n"
            "  Caliper diameters: the maximum and minimum distances between two parallel\n"
            "  tangents of the pore. Approximated via a minimum-area rotated rectangle or\n"
            "  rotating-calipers on the contour. Reported in px or scaled units.\n"
            "\n"
            "• Ellipse major / minor axis (major_axis / minor_axis)\n"
            "  Axes of the best-fit ellipse (from image moments / cv2.fitEllipse).\n"
            "  Often close to Feret lengths but not identical for irregular shapes.\n"
            "\n"
            "• Aspect ratio (aspect_ratio)\n"
            "  major_axis / minor_axis (≥ 1). Larger means more elongated.\n"
            "\n"
            "• Orientation (orientation_deg)\n"
            "  Angle (degrees) of the fitted ellipse’s major axis relative to +x (image\n"
            "  columns), increasing counter-clockwise. Note: for highly concave shapes,\n"
            "  ellipse fit may be less representative.\n"
            "\n"
            "• Border-touching (touches_border)\n"
            "  True if the pore contacts the image edge. Such pores are often excluded\n"
            "  from size statistics to avoid truncation bias.\n"
            "\n"
            "PLOTTING & FILTERS\n"
            "• Aggregation: ‘Current image’ plots only the active frame; ‘All images’ builds\n"
            "  an aggregate across the dataset.\n"
            "• Units: switch between pixels and calibrated units (enabled when a common\n"
            "  scale exists). Length metrics convert with your units/px; areas with (units/px)².\n"
            "• Area filter: keep pores within [min, max] (0 = no bound); applied in the\n"
            "  currently selected units mode.\n"
            "• Exclude border-touching: removes censored pores from the distributions.\n"
            "• Bins / Log Y: control histogram bin count and y-axis scaling.\n"
        )

        txt.insert("1.0", HELP)
        txt.configure(state="disabled")

        # Close button row
        btns = ttk.Frame(dlg, padding=(0,8,0,0))
        btns.pack(side="bottom", fill="x")
        ttk.Button(btns, text="Close", command=dlg.destroy).pack(side="right")



    # ------------- actions -------------

    def _export_csv(self):
        if not self._last_props:
            messagebox.showwarning("Export", "No measurements available.")
            return
        out = filedialog.asksaveasfilename(
            title="Save measurements CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not out:
            return
        try:
            save_props_csv(out, self._last_props)
            messagebox.showinfo("Export", f"Saved measurements to:\n{out}")
        except Exception as e:
            messagebox.showerror("Export", f"Failed to save CSV:\n{e}")

    def _save_figure(self):
        out = filedialog.asksaveasfilename(
            title="Save histogram figure",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        if not out:
            return
        try:
            self.fig.savefig(out, bbox_inches="tight", dpi=200)
            messagebox.showinfo("Save Figure", f"Saved figure to:\n{out}")
        except Exception as e:
            messagebox.showerror("Save Figure", f"Failed to save figure:\n{e}")

    # ------------- events -------------

    def _on_settings_changed(self):
        self._compute_and_plot()

    def _prev(self):
        if self.index > 0:
            self.index -= 1
            self._update_view()
            if self.aggregateVar.get() == "current":
                self._compute_and_plot()

    def _next(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self._update_view()
            if self.aggregateVar.get() == "current":
                self._compute_and_plot()
