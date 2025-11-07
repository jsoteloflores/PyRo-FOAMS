# postprocessing_gui.py
from __future__ import annotations
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Dict, Tuple
from PIL import Image, ImageTk

# Pillow resampling shim
if hasattr(Image, "Resampling"):
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", Image.BICUBIC)

OverlayColor = (255, 80, 255)  # magenta tint for mask overlay (BGR)
OverlayAlpha = 0.45            # display-only alpha
HISTORY_LIMIT = 40             # per-image undo depth

# ---------- utils ----------
def _ensure_bool_mask(mask: Optional[np.ndarray], shape_hw: Tuple[int, int]) -> np.ndarray:
    """Return a boolean mask with the given HxW shape."""
    h, w = shape_hw
    if mask is None:
        return np.zeros((h, w), dtype=bool)
    m = mask
    if m.dtype != np.bool_:
        if m.dtype == np.uint8:
            if m.max() > 1:
                m = (m >= 128)
            else:
                m = (m > 0)
        else:
            m = (m != 0)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m

def _to_pil_disp(bgr_or_gray: np.ndarray) -> Image.Image:
    a = bgr_or_gray
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

def _fit_scale_and_origin(img_wh: Tuple[int, int], canvas_wh: Tuple[int, int]) -> Tuple[float, int, int, int, int]:
    """Return (scale s, ox, oy, disp_w, disp_h) to fit image (w,h) into canvas (cw,ch)."""
    iw, ih = img_wh
    cw, ch = max(1, canvas_wh[0]), max(1, canvas_wh[1])
    s = min(cw / float(iw), ch / float(ih)); s = min(s, 1.0)
    disp_w = max(1, int(round(iw * s)))
    disp_h = max(1, int(round(ih * s)))
    ox = (cw - disp_w) // 2
    oy = (ch - disp_h) // 2
    return s, ox, oy, disp_w, disp_h

def _disk_kernel(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    d = 2 * r + 1
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    return (xx*xx + yy*yy) <= (r*r)

def _line_points(p0: Tuple[int, int], p1: Tuple[int, int], step: int) -> List[Tuple[int,int]]:
    x0, y0 = p0; x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    dist = max(1.0, float(np.hypot(dx, dy)))
    n = max(1, int(dist / max(1, step)))
    xs = np.linspace(x0, x1, n+1, dtype=int)
    ys = np.linspace(y0, y1, n+1, dtype=int)
    return list(zip(xs.tolist(), ys.tolist()))

class BusyDialog(tk.Toplevel):
    def __init__(self, parent, title="Working…", mode="indeterminate", maximum=100):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=title).pack(anchor="w", pady=(0,8))
        self.pb = ttk.Progressbar(frm, orient="horizontal", length=360, mode=mode, maximum=maximum)
        self.pb.pack(fill="x")
        if mode == "indeterminate":
            self.pb.start(10)
        self.update_idletasks()
    def set_progress(self, v):
        try:
            self.pb["value"] = v; self.update_idletasks()
        except Exception:
            pass
    def close(self):
        try:
            if str(self.pb["mode"]) == "indeterminate": self.pb.stop()
        except Exception: pass
        try: self.grab_release()
        except Exception: pass
        self.destroy()

# ---------- per-image history ----------
class _History:
    def __init__(self, limit=HISTORY_LIMIT):
        self.limit = int(limit)
        self.stack: List[np.ndarray] = []
        self.idx: int = -1

    def ensure_initial(self, mask: np.ndarray):
        if not self.stack:
            self.stack = [mask.copy()]
            self.idx = 0

    def commit(self, mask: np.ndarray):
        """Push a new version. Truncate any redo tail."""
        if self.idx < len(self.stack) - 1:
            self.stack = self.stack[:self.idx + 1]
        self.stack.append(mask.copy())
        if len(self.stack) > self.limit:
            overflow = len(self.stack) - self.limit
            self.stack = self.stack[overflow:]
        self.idx = len(self.stack) - 1

    def can_undo(self) -> bool:
        return self.idx > 0

    def can_redo(self) -> bool:
        return self.idx < len(self.stack) - 1

    def undo(self) -> Optional[np.ndarray]:
        if not self.can_undo(): return None
        self.idx -= 1
        return self.stack[self.idx].copy()

    def redo(self) -> Optional[np.ndarray]:
        if not self.can_redo(): return None
        self.idx += 1
        return self.stack[self.idx].copy()

# =================== Main Window ===================
class PostprocessWindow(tk.Toplevel):
    """
    Binary-only label editor with:
      - Hard-edged brush/eraser (0/1 only), preview circle, size slider+spinbox+hotkeys
      - Flood-Fill tool: Fill (add) / Unfill (remove), 4/8 connectivity
      - Undo/Redo per image (Ctrl+Z / Ctrl+Y), history limit configurable
      - Process Islands: remove small fg components, fill small holes
      - Live overlay display (magenta) on the original image
    """
    def __init__(self,
                 parent,
                 images: List[np.ndarray],
                 masks: List[Optional[np.ndarray]],
                 paths: List[str],
                 startIndex: int = 0,
                 onMasksUpdated=None):
        super().__init__(parent)
        self.title("Mask Editor")
        self.transient(parent)
        self.grab_set()

        # Data
        self.images = images
        self.paths  = paths or [f"Image {i+1}" for i in range(len(images))]
        self.masks  = [ _ensure_bool_mask(m, img.shape[:2]) for m, img in zip(masks, images) ]
        if len(self.masks) < len(self.images):
            for i in range(len(self.masks), len(self.images)):
                h, w = self.images[i].shape[:2]
                self.masks.append(np.zeros((h, w), dtype=bool))
        self.index  = max(0, min(startIndex, len(self.images)-1))
        self.onMasksUpdated = onMasksUpdated

        # Per-image history
        self._histories = [_History(HISTORY_LIMIT) for _ in range(len(self.images))]
        for i, m in enumerate(self.masks):
            self._histories[i].ensure_initial(m)

        # View state
        self._photo = None
        self._base_disp_bgr = None
        self._disp_bgr = None
        self._scale = 1.0
        self._ox = 0; self._oy = 0
        self._disp_wh = (1,1)

        # Tool state
        # paint / erase / fill_add / fill_remove
        self.modeVar = tk.StringVar(value="paint")
        self.radiusVar = tk.IntVar(value=20)
        self._kernel_cache: Dict[int, np.ndarray] = {}
        self._last_img_pt: Optional[Tuple[int,int]] = None
        self._stroke_active = False
        self._pending_canvas_pt: Optional[Tuple[int,int]] = None
        self._motion_scheduled = False

        # Island / connectivity settings (also used by Flood-Fill)
        self.connVar = tk.IntVar(value=8)  # 4 or 8
        self.minIslandVar = tk.IntVar(value=50)
        self.maxHoleVar = tk.IntVar(value=50)
        self.ignoreBorderVar = tk.BooleanVar(value=True)

        # Build UI
        self._build_ui()
        self._load_current()
        self.geometry("1200x820")
        self.minsize(900, 600)

        # Hotkeys
        self.bind("<KeyPress-bracketleft>", lambda e: self._nudge_size(-1))
        self.bind("<KeyPress-bracketright>", lambda e: self._nudge_size(+1))
        self.bind("<Shift-KeyPress-bracketleft>", lambda e: self._nudge_size(-5))
        self.bind("<Shift-KeyPress-bracketright>", lambda e: self._nudge_size(+5))
        self.bind("<Control-z>", lambda e: self._undo())
        self.bind("<Control-y>", lambda e: self._redo())
        self.bind("<Control-Z>", lambda e: self._redo())  # common alternative

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self); top.pack(side="top", fill="x", padx=8, pady=6)
        ttk.Button(top, text="Prev", command=self._prev).pack(side="left", padx=2)
        ttk.Button(top, text="Next", command=self._next).pack(side="left", padx=2)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(top, text="Undo (Ctrl+Z)", command=self._undo).pack(side="left", padx=2)
        ttk.Button(top, text="Redo (Ctrl+Y)", command=self._redo).pack(side="left", padx=2)
        self.statusVar = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.statusVar).pack(side="left", padx=12)

        body = ttk.Frame(self); body.pack(side="top", fill="both", expand=True, padx=8, pady=(0,8))

        # Canvas
        self.canvas = tk.Canvas(body, bg="#1a1a1a", highlightthickness=1, highlightbackground="#3a3a3a")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Motion>", self._on_motion_preview)
        self.canvas.bind("<Leave>", lambda e: self._clear_preview_circle())
        # drawing
        self.canvas.bind("<ButtonPress-1>", self._on_stroke_start)
        self.canvas.bind("<B1-Motion>", self._on_stroke_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_stroke_end)
        # ctrl+wheel changes size
        self.canvas.bind("<Control-MouseWheel>", self._on_wheel_size)
        self.canvas.bind("<Control-Button-4>", self._on_wheel_size)  # some X11
        self.canvas.bind("<Control-Button-5>", self._on_wheel_size)

        # Controls (right)
        side = ttk.Frame(body); side.pack(side="left", fill="y", padx=(10,0))

        tool = ttk.LabelFrame(side, text="Tool"); tool.pack(fill="x", pady=4)
        ttk.Radiobutton(tool, text="Brush (add)", value="paint", variable=self.modeVar).grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Radiobutton(tool, text="Eraser (remove)", value="erase", variable=self.modeVar).grid(row=1, column=0, sticky="w", padx=8, pady=0)
        ttk.Radiobutton(tool, text="Fill (add)", value="fill_add", variable=self.modeVar).grid(row=2, column=0, sticky="w", padx=8, pady=(8,0))
        ttk.Radiobutton(tool, text="Unfill (remove)", value="fill_remove", variable=self.modeVar).grid(row=3, column=0, sticky="w", padx=8, pady=0)

        sz = ttk.LabelFrame(side, text="Size"); sz.pack(fill="x", pady=6)
        row=0
        ttk.Label(sz, text="Radius (px):").grid(row=row, column=0, sticky="w", padx=8, pady=(6,2))
        sp = ttk.Spinbox(sz, from_=1, to=512, textvariable=self.radiusVar, width=6, justify="right")
        sp.grid(row=row, column=1, sticky="w", padx=(0,8), pady=(6,2))
        row+=1
        sld = ttk.Scale(sz, from_=1, to=256, orient="horizontal", variable=self.radiusVar, length=180)
        sld.grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=(0,8))
        sz.columnconfigure(0, weight=1)

        proc = ttk.LabelFrame(side, text="Process Islands / Connectivity"); proc.pack(fill="x", pady=10)
        ttk.Label(proc, text="Connectivity:").grid(row=0, column=0, sticky="e", padx=8, pady=4)
        ttk.Radiobutton(proc, text="4", value=4, variable=self.connVar).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(proc, text="8", value=8, variable=self.connVar).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(proc, text="Ignore border-touching (for remove)", variable=self.ignoreBorderVar).grid(row=1, column=0, columnspan=3, sticky="w", padx=8)

        ttk.Label(proc, text="Remove fg islands <=").grid(row=2, column=0, sticky="e", padx=8, pady=2)
        ttk.Entry(proc, textvariable=self.minIslandVar, width=7).grid(row=2, column=1, sticky="w")
        ttk.Label(proc, text="px").grid(row=2, column=2, sticky="w")

        ttk.Label(proc, text="Fill holes <=").grid(row=3, column=0, sticky="e", padx=8, pady=2)
        ttk.Entry(proc, textvariable=self.maxHoleVar, width=7).grid(row=3, column=1, sticky="w")
        ttk.Label(proc, text="px").grid(row=3, column=2, sticky="w")

        btns = ttk.Frame(proc); btns.grid(row=4, column=0, columnspan=3, pady=(8,6))
        ttk.Button(btns, text="Apply", command=self._apply_islands).pack(side="left", padx=6)

        bottom = ttk.Frame(self); bottom.pack(side="bottom", fill="x", padx=8, pady=8)
        ttk.Button(bottom, text="Reset Mask (current)", command=self._reset_current_mask).pack(side="left")
        ttk.Button(bottom, text="Done", command=self._on_close).pack(side="right")

    # ---------- Image load/render ----------
    def _load_current(self):
        img = self.images[self.index]
        name = os.path.basename(self.paths[self.index]) if (self.paths and self.index < len(self.paths)) else f"Image {self.index+1}"
        self.statusVar.set(f"{name}  ({self.index+1}/{len(self.images)})   —  Brush radius: {self.radiusVar.get()} px")
        # ensure initial state present
        self._histories[self.index].ensure_initial(self.masks[self.index])
        self._render_full()

    def _on_resize(self, event):
        self._render_full()

    def _render_full(self):
        if not self.images: return
        img = self.images[self.index]
        h, w = img.shape[:2]
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        s, ox, oy, dw, dh = _fit_scale_and_origin((w, h), (cw, ch))
        self._scale, self._ox, self._oy, self._disp_wh = s, ox, oy, (dw, dh)

        if img.ndim == 2:
            base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            base = img
        self._base_disp_bgr = cv2.resize(base, (dw, dh), interpolation=cv2.INTER_AREA)
        self._compose_overlay_full()

    def _compose_overlay_full(self):
        if self._base_disp_bgr is None: return
        dw, dh = self._disp_wh
        disp = self._base_disp_bgr.copy()

        m = self.masks[self.index].astype(np.uint8)
        m_disp = cv2.resize(m, (dw, dh), interpolation=cv2.INTER_NEAREST)
        if m_disp.any():
            tint = np.zeros_like(disp)
            tint[:, :, 0] = np.maximum(tint[:, :, 0], m_disp * OverlayColor[0])
            tint[:, :, 1] = np.maximum(tint[:, :, 1], m_disp * OverlayColor[1])
            tint[:, :, 2] = np.maximum(tint[:, :, 2], m_disp * OverlayColor[2])
            disp = cv2.addWeighted(tint, OverlayAlpha, disp, 1.0 - OverlayAlpha, 0.0)

        self._disp_bgr = disp
        self._update_canvas()

    def _update_canvas(self, preview_circle: Optional[Tuple[int,int,int]]=None):
        if self._disp_bgr is None: return
        self.canvas.delete("all")
        pil = _to_pil_disp(self._disp_bgr)
        photo = ImageTk.PhotoImage(pil)
        self._photo = photo
        self.canvas.create_image(self._ox, self._oy, anchor="nw", image=photo, tags="img")
        if preview_circle is not None:
            x, y, r = preview_circle
            mode = self.modeVar.get()
            color = "#00ff66" if mode in ("paint", "fill_add") else "#ff4d4d"
            self.canvas.create_oval(x-r, y-r, x+r, y+r, outline=color, width=2, tags="preview")

    def _clear_preview_circle(self):
        try: self.canvas.delete("preview")
        except Exception: pass

    # ---------- Coordinate transforms ----------
    def _canvas_to_image_pt(self, x: int, y: int) -> Optional[Tuple[int,int]]:
        img = self.images[self.index]
        h, w = img.shape[:2]
        s, ox, oy = self._scale, self._ox, self._oy
        xi = int((x - ox) / max(s, 1e-6))
        yi = int((y - oy) / max(s, 1e-6))
        if 0 <= xi < w and 0 <= yi < h:
            return (xi, yi)
        return None

    def _image_to_canvas_pt(self, xi: int, yi: int) -> Tuple[int,int]:
        s, ox, oy = self._scale, self._ox, self._oy
        x = int(xi * s + ox); y = int(yi * s + oy)
        return x, y

    # ---------- Brush mechanics ----------
    def _kernel(self, radius: int) -> np.ndarray:
        r = max(1, int(radius))
        k = self._kernel_cache.get(r)
        if k is None:
            k = _disk_kernel(r)
            self._kernel_cache[r] = k
        return k

    def _stamp(self, center_xy: Tuple[int,int], paint: bool):
        m = self.masks[self.index]
        h, w = m.shape[:2]
        r = max(1, int(self.radiusVar.get()))
        K = self._kernel(r)
        cx, cy = center_xy
        x0 = cx - r; y0 = cy - r
        x1 = cx + r; y1 = cy + r
        rx0, ry0 = max(0, x0), max(0, y0)
        rx1, ry1 = min(w, x1+1), min(h, y1+1)
        if rx1 <= rx0 or ry1 <= ry0: return
        kx0 = rx0 - x0; ky0 = ry0 - y0
        kx1 = kx0 + (rx1 - rx0); ky1 = ky0 + (ry1 - ry0)
        if paint:
            m[ry0:ry1, rx0:rx1] |= K[ky0:ky1, kx0:kx1]
        else:
            m[ry0:ry1, rx0:rx1] &= ~K[ky0:ky1, kx0:kx1]

    def _stroke_line(self, p0_img: Tuple[int,int], p1_img: Tuple[int,int], paint: bool):
        step = max(1, self.radiusVar.get() // 2)
        for p in _line_points(p0_img, p1_img, step):
            self._stamp(p, paint)

    # ---------- Flood-Fill ----------
    def _flood_fill(self, seed_xy: Tuple[int,int], add_mode: bool):
        """
        add_mode=True  -> fill background component at seed into foreground (set 1)
        add_mode=False -> remove foreground component at seed (set 0)
        """
        m = self.masks[self.index]
        y, x = seed_xy[1], seed_xy[0]
        conn = 8 if int(self.connVar.get()) == 8 else 4

        if add_mode:
            src = ~m
            if not src[y, x]:
                return
            num, labels = cv2.connectedComponents(src.astype(np.uint8), connectivity=conn)
            lab = labels[y, x]
            if lab == 0:  # background label of src (shouldn't happen if src[y,x] True)
                return
            m[labels == lab] = True
        else:
            src = m
            if not src[y, x]:
                return
            num, labels = cv2.connectedComponents(src.astype(np.uint8), connectivity=conn)
            lab = labels[y, x]
            if lab == 0:
                return
            m[labels == lab] = False

    # ---------- Stroke event flow w/ coalescing ----------
    def _on_motion_preview(self, event):
        r_px = int(round(self.radiusVar.get() * self._scale))
        self._update_canvas(preview_circle=(event.x, event.y, max(1, r_px)))

    def _on_stroke_start(self, event):
        ipt = self._canvas_to_image_pt(event.x, event.y)
        if ipt is None: return

        mode = self.modeVar.get()
        # For flood-fill modes: single-shot operation + commit history
        if mode in ("fill_add", "fill_remove"):
            self._commit_before()
            self._flood_fill(ipt, add_mode=(mode == "fill_add"))
            self._compose_overlay_full()
            self._commit_after()
            self._on_motion_preview(event)
            self._notify_parent()
            return

        # Brush/eraser stroke begins
        self._commit_before()
        self._stroke_active = True
        self._last_img_pt = ipt
        paint = (mode == "paint")
        self._stamp(ipt, paint)
        self._compose_overlay_full()
        self._on_motion_preview(event)

    def _schedule_motion(self):
        if self._motion_scheduled: return
        self._motion_scheduled = True
        self.after_idle(self._process_motion)

    def _on_stroke_motion(self, event):
        if not self._stroke_active: return
        self._pending_canvas_pt = (event.x, event.y)
        self._schedule_motion()

    def _process_motion(self):
        self._motion_scheduled = False
        if not self._stroke_active or self._pending_canvas_pt is None: return
        cx, cy = self._pending_canvas_pt
        ipt = self._canvas_to_image_pt(cx, cy)
        if ipt is None: return
        if self._last_img_pt is None:
            self._last_img_pt = ipt
        paint = (self.modeVar.get() == "paint")
        self._stroke_line(self._last_img_pt, ipt, paint)
        self._last_img_pt = ipt
        self._compose_overlay_full()
        r_px = int(round(self.radiusVar.get() * self._scale))
        self._update_canvas(preview_circle=(cx, cy, max(1, r_px)))

    def _on_stroke_end(self, event):
        if not self._stroke_active: return
        self._stroke_active = False
        self._last_img_pt = None
        self._pending_canvas_pt = None
        self._compose_overlay_full()
        self._on_motion_preview(event)
        self._commit_after()
        self._notify_parent()

    # ---------- Undo / Redo ----------
    def _commit_before(self):
        """Ensure history has the *current* state as the base of this edit if needed."""
        # History already holds current state as last snapshot; nothing to do here.
        # (We separate before/after methods for clarity and possible future diffs.)
        pass

    def _commit_after(self):
        self._histories[self.index].commit(self.masks[self.index])

    def _undo(self):
        h = self._histories[self.index]
        m = h.undo()
        if m is None: return
        self.masks[self.index] = m
        self._compose_overlay_full()
        self._notify_parent()

    def _redo(self):
        h = self._histories[self.index]
        m = h.redo()
        if m is None: return
        self.masks[self.index] = m
        self._compose_overlay_full()
        self._notify_parent()

    # ---------- Process Islands ----------
    def _apply_islands(self):
        self._commit_before()
        m = self.masks[self.index].copy()
        conn = 4 if int(self.connVar.get()) == 4 else 8
        min_island = max(0, int(self.minIslandVar.get()))
        max_hole = max(0, int(self.maxHoleVar.get()))
        ignore_border = bool(self.ignoreBorderVar.get())

        # Remove small foreground islands
        if min_island > 0:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=conn)
            remove = np.zeros(num, dtype=bool)
            H, W = m.shape
            for lab in range(1, num):
                area = int(stats[lab, cv2.CC_STAT_AREA])
                if area < min_island:
                    if ignore_border:
                        x, y, w, h, _ = stats[lab]
                        touches = (x==0) or (y==0) or (x+w>=W) or (y+h>=H)
                        if touches:
                            continue
                    remove[lab] = True
            m[remove[labels]] = False

        # Fill small holes
        if max_hole > 0:
            inv = ~m
            num, labels, stats, _ = cv2.connectedComponentsWithStats(inv.astype(np.uint8), connectivity=conn)
            keep_bg = np.zeros(num, dtype=bool)
            H, W = m.shape
            for lab in range(1, num):
                area = int(stats[lab, cv2.CC_STAT_AREA])
                x, y, w, h, _ = stats[lab]
                touches_border = (x==0) or (y==0) or (x+w>=W) or (y+h>=H)
                if touches_border or area >= max_hole:
                    keep_bg[lab] = True
            fill = ~keep_bg[labels]
            inv[fill] = False
            m = ~inv

        self.masks[self.index] = m
        self._compose_overlay_full()
        self._commit_after()
        self._notify_parent()

    # ---------- Helpers / nav ----------
    def _reset_current_mask(self):
        self._commit_before()
        h, w = self.images[self.index].shape[:2]
        self.masks[self.index] = np.zeros((h, w), dtype=bool)
        self._compose_overlay_full()
        self._commit_after()
        self._notify_parent()

    def _prev(self):
        if self.index > 0:
            self.index -= 1
            self._load_current()

    def _next(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self._load_current()

    def _on_close(self):
        self._notify_parent()
        self.destroy()

    def _notify_parent(self):
        if callable(self.onMasksUpdated):
            try:
                self.onMasksUpdated(self.masks)
            except Exception as e:
                print("onMasksUpdated error:", e)

    # ---------- size & preview ----------
    def _on_wheel_size(self, event):
        delta = 1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else -1
        self._nudge_size(+1 if delta > 0 else -1)

    def _nudge_size(self, d: int):
        r = self.radiusVar.get()
        r = int(np.clip(r + d, 1, 512))
        self.radiusVar.set(r)
        base = self.statusVar.get().split("—")[0]
        self.statusVar.set(base + f"—  Brush radius: {r} px")
