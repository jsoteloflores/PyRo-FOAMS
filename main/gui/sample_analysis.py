from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import asdict, replace
from tkinter import ttk
from typing import Callable, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, NullLocator

from ..core.legacy_cutoffs import Foams105CutoffGroupInput, suggest_foams105_cutoffs
from ..core.legacy_perimeter import (
    FOAMS105_PERIMETER_PROFILE_CURRENT,
    FOAMS105_PERIMETER_PROFILE_OLD,
)
from ..core.reconstruction import UPPER_TAIL_ASSUMPTION
from ..core.sample_analysis import (
    CompareRunResult,
    LegacyPreparedContext,
    LegacyRunConfig,
    ModernPreparedResult,
    ModernRunConfig,
    SampleAnalysisError,
    analyze_legacy_prepared,
    analyze_modern_prepared,
    build_legacy_group_inputs,
    prepare_legacy_analysis,
    prepare_modern_analysis,
    validate_analysis_setup,
)
from ..core.sample_project import (
    ProjectAnalysisSnapshot,
    SampleProject,
    calibration_mm_per_px,
    create_analysis_snapshot,
)

_PERIMETER_LABEL_TO_ID = {
    "R2025b": FOAMS105_PERIMETER_PROFILE_CURRENT,
    "PerimeterOld": FOAMS105_PERIMETER_PROFILE_OLD,
}
_PERIMETER_ID_TO_LABEL = {
    FOAMS105_PERIMETER_PROFILE_CURRENT: "R2025b",
    FOAMS105_PERIMETER_PROFILE_OLD: "PerimeterOld",
}


class SampleAnalysisWorkspace(tk.Toplevel):
    """Integrated sample-analysis workspace for Original, Modern, and Compare paths."""

    def __init__(
        self,
        parent: tk.Misc,
        project: SampleProject,
        snapshot: ProjectAnalysisSnapshot,
        mark_unsaved: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.title("PyRo-FOAMS - Sample Analysis")
        self.transient(parent)
        self.project = project
        self.snapshot = snapshot
        self._mark_unsaved = mark_unsaved
        self._project_owner = getattr(mark_unsaved, "__self__", None)
        self._project_session = getattr(self._project_owner, "projectSessionId", None)
        self._closed = False
        self._owned_after_ids = set()
        self._result_queue: "queue.Queue[tuple[int, object, Optional[BaseException], ProjectAnalysisSnapshot]]" = queue.Queue()
        self._job_generation = 0
        self._active_generation = 0
        self._worker: Optional[threading.Thread] = None
        self._last_completed_snapshot: Optional[ProjectAnalysisSnapshot] = None
        self._completed_signature = None
        self._active_request = None
        self._busy = False
        self._control_states = []
        self._source_keys = ()
        self._prepared_legacy = None
        self._prepared_modern = None
        self._setting_proposal = False

        self._legacy_prepare_cache: dict[tuple, object] = {}
        self._latest_legacy = None
        self._latest_modern = None
        self._latest_compare = None

        self.modeVar = tk.StringVar(value="Original FOAMS 1.0.5")
        self.minimumDiameterVar = tk.StringVar(value="2.0")
        self.perimeterProfileVar = tk.StringVar(value="R2025b")
        self.selectionModeVar = tk.StringVar(value="autosmart")
        self.manualRangesVar = tk.StringVar(value="")
        self.volumeModeVar = tk.StringVar(value="unnormalized")
        self.knownVesicularityVar = tk.StringVar(value="")
        self.sizeLowerCutoffVar = tk.StringVar(value="0.5")
        self.sizeUpperCutoffVar = tk.StringVar(value="1.0")
        self.shapeMinimumVar = tk.StringVar(value="")
        self.shapeModeVar = tk.StringVar(value="initial_analysis")

        self.modernMinDetectableVar = tk.StringVar(value="")
        self.modernMaxReliableVar = tk.StringVar(value="")
        self.modernTransitionBinsVar = tk.StringVar(value="")
        self.modernReconstructVar = tk.BooleanVar(value=False)
        self.modernUpperTailVar = tk.BooleanVar(value=False)
        self.modernEdgesVar = tk.StringVar(value="")
        self.modernSegmentsVar = tk.StringVar(value="")

        self.statusVar = tk.StringVar(value="Ready")

        self._build_ui()
        self._load_persisted_setup()
        self._variables = {
            name: value for name, value in vars(self).items() if isinstance(value, tk.Variable) and name != "statusVar"
        }
        for variable in self._variables.values():
            variable.trace_add("write", self._settings_changed)
        self.manualRangesVar.trace_add("write", self._manual_changed)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.geometry("1200x800")
        self.minsize(980, 640)
        self._refresh_summary_banner()
        self._schedule_after(200, self._watch_inputs)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        summary = ttk.LabelFrame(root, text="Accepted Analysis Snapshot", padding=8)
        summary.pack(fill="x")
        self.snapshotSummary = ttk.Label(summary, justify="left", wraplength=1120)
        self.snapshotSummary.pack(fill="x")

        setup = ttk.LabelFrame(root, text="Setup", padding=4)
        setup.pack(fill="x", pady=(8, 0))
        setup_canvas = tk.Canvas(setup, height=230, highlightthickness=0)
        setup_canvas.grid(row=0, column=0, sticky="nsew")
        setup.columnconfigure(0, weight=1)
        setup.rowconfigure(0, weight=1)
        ttk.Scrollbar(setup, orient="vertical", command=setup_canvas.yview).grid(row=0, column=1, sticky="ns")
        ttk.Scrollbar(setup, orient="horizontal", command=setup_canvas.xview).grid(row=1, column=0, sticky="ew")
        setup_canvas.configure(yscrollcommand=setup.children['!scrollbar'].set,
                       xscrollcommand=setup.children['!scrollbar2'].set)
        controls = ttk.Frame(setup_canvas, padding=8)
        self.setupControls = controls
        setup_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", lambda event: setup_canvas.configure(scrollregion=setup_canvas.bbox("all")))

        ttk.Label(controls, text="Method preset").grid(row=0, column=0, sticky="w")
        mode_combo = ttk.Combobox(
            controls,
            textvariable=self.modeVar,
            values=("Original FOAMS 1.0.5", "Modern analysis", "Compare methods"),
            state="readonly",
            width=28,
        )
        mode_combo.grid(row=0, column=1, sticky="w", padx=(8, 16))

        ttk.Label(controls, text="Min diameter (px)").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.minimumDiameterVar, width=10).grid(
            row=0, column=3, sticky="w", padx=(8, 16)
        )

        ttk.Label(controls, text="Perimeter profile").grid(row=0, column=4, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.perimeterProfileVar,
            values=("R2025b", "PerimeterOld"),
            state="readonly",
            width=14,
        ).grid(row=0, column=5, sticky="w", padx=(8, 0))

        ttk.Label(controls, text="Selection mode").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.selectionModeVar,
            values=("autosmart", "manual"),
            state="readonly",
            width=14,
        ).grid(row=1, column=1, sticky="w", padx=(8, 16), pady=(8, 0))

        ttk.Label(controls, text="Manual ranges").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.manualRangesVar, width=42).grid(
            row=1, column=3, columnspan=3, sticky="w", padx=(8, 0), pady=(8, 0)
        )

        ttk.Label(controls, text="Normalization").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.volumeModeVar,
            values=("unnormalized", "integrated_2d", "known_vesicularity"),
            state="readonly",
            width=18,
        ).grid(row=2, column=1, sticky="w", padx=(8, 16), pady=(8, 0))

        ttk.Label(controls, text="Known vesicularity (%)").grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.knownVesicularityVar, width=10).grid(
            row=2, column=3, sticky="w", padx=(8, 16), pady=(8, 0)
        )

        ttk.Label(controls, text="Category lower/upper (mm)").grid(row=2, column=4, sticky="w", pady=(8, 0))
        category_row = ttk.Frame(controls)
        category_row.grid(row=2, column=5, sticky="w", pady=(8, 0))
        ttk.Entry(category_row, textvariable=self.sizeLowerCutoffVar, width=8).pack(side="left")
        ttk.Label(category_row, text="/").pack(side="left", padx=4)
        ttk.Entry(category_row, textvariable=self.sizeUpperCutoffVar, width=8).pack(side="left")

        ttk.Label(controls, text="Shape minimum (px)").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.shapeMinimumVar, width=10).grid(
            row=3, column=1, sticky="w", padx=(8, 16), pady=(8, 0)
        )

        ttk.Label(controls, text="Shape mode").grid(row=3, column=2, sticky="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.shapeModeVar,
            values=("initial_analysis", "updated_minimum"),
            state="readonly",
            width=18,
        ).grid(row=3, column=3, sticky="w", padx=(8, 16), pady=(8, 0))

        ttk.Label(controls, text="Modern min/max reliable (mm)").grid(row=3, column=4, sticky="w", pady=(8, 0))
        modern_limits = ttk.Frame(controls)
        modern_limits.grid(row=3, column=5, sticky="w", pady=(8, 0))
        ttk.Entry(modern_limits, textvariable=self.modernMinDetectableVar, width=8).pack(side="left")
        ttk.Label(modern_limits, text="/").pack(side="left", padx=4)
        ttk.Entry(modern_limits, textvariable=self.modernMaxReliableVar, width=8).pack(side="left")

        ttk.Label(controls, text="Modern bin edges (mm)").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.modernEdgesVar, width=40).grid(row=4, column=1, columnspan=2, sticky="ew", padx=8)
        ttk.Checkbutton(
            controls,
            text="Enable spherical 3D reconstruction",
            variable=self.modernReconstructVar,
        ).grid(row=4, column=3, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(controls, text="Modern group:start:stop bins").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.modernSegmentsVar, width=45).grid(row=5, column=1, columnspan=3, sticky="ew", padx=8)
        ttk.Checkbutton(controls, text="No 3D size classes above the selected upper edge", variable=self.modernUpperTailVar).grid(row=6, column=0, columnspan=6, sticky="w", pady=(8, 0))

        action_frame = ttk.Frame(root)
        action_frame.pack(fill="x", pady=(8, 0))
        action_canvas = tk.Canvas(action_frame, height=44, highlightthickness=0)
        action_canvas.pack(fill="x")
        action_scroll = ttk.Scrollbar(action_frame, orient="horizontal", command=action_canvas.xview)
        action_scroll.pack(fill="x")
        action_canvas.configure(xscrollcommand=action_scroll.set)
        actions = ttk.Frame(action_canvas)
        action_canvas.create_window((0, 0), window=actions, anchor="nw")
        actions.bind("<Configure>", lambda event: (action_canvas.configure(scrollregion=action_canvas.bbox("all")),
                              action_canvas.configure(height=actions.winfo_reqheight())))
        self.runButton = ttk.Button(actions, text="Run analysis", command=self._run_analysis)
        self.runButton.pack(side="left")
        self.suggestButton = ttk.Button(actions, text="Suggest cutoffs", command=self._suggest_cutoffs)
        self.suggestButton.pack(side="left", padx=(8, 0))
        self.saveSetupButton = ttk.Button(actions, text="Save setup", command=self._save_current_setup)
        self.saveSetupButton.pack(side="left", padx=8)
        self.prepareButton = ttk.Button(actions, text="Prepare / inspect", command=lambda: self._start_job("prepare"))
        self.prepareButton.pack(side="left", padx=8)
        self.previewButton = ttk.Button(actions, text="Review selection", command=lambda: self._start_job("preview"))
        self.previewButton.pack(side="left")
        self.shapeUpdateButton = ttk.Button(actions, text="Update shape minimum", command=self._update_shape_minimum)
        self.shapeUpdateButton.pack(side="left", padx=8)
        self.cancelButton = ttk.Button(actions, text="Cancel", command=self._cancel_job, state="disabled")
        self.cancelButton.pack(side="left", padx=8)
        ttk.Button(actions, text="Back to preparation", command=self._on_close).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(actions, text="Measurements", command=self._open_measurements).pack(side="left", padx=8)
        ttk.Label(root, textvariable=self.statusVar, wraplength=900).pack(fill="x", pady=4)

        results = ttk.LabelFrame(root, text="Results", padding=8)
        results.pack(fill="both", expand=True, pady=(8, 0))
        views = ttk.Notebook(results)
        views.pack(fill="both", expand=True)
        table_frame = ttk.Frame(views)
        plot_frame = ttk.Frame(views)
        details_frame = ttk.Frame(views)
        views.add(table_frame, text="Distributions")
        views.add(plot_frame, text="Plot")
        views.add(details_frame, text="Details / raw identities")
        views.bind("<<NotebookTabChanged>>", lambda event: setup_canvas.configure(
            height=110 if views.index(views.select()) == 1 else 230))
        columns = ("method", "group", "bin", "lower", "upper", "count", "area", "na", "nv", "volume")
        self.distributionTable = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="extended")
        for column, label in zip(columns, ("Method", "Group", "Bin", "Lower mm", "Upper mm", "Count", "Area mm^2", "N_A mm^-2", "N_V mm^-3", "Volume fraction")):
            self.distributionTable.heading(column, text=label)
            self.distributionTable.column(column, width=110, stretch=False)
        self.distributionTable.grid(row=0, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        vertical = ttk.Scrollbar(table_frame, orient="vertical", command=self.distributionTable.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=self.distributionTable.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.distributionTable.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.distributionTable.bind("<<TreeviewSelect>>", self._select_bins)
        self.binGroupVar = tk.StringVar(value="")
        self.binGroupSelector = ttk.Combobox(table_frame, textvariable=self.binGroupVar, state="readonly")
        self.binGroupSelector.grid(row=2, column=0, sticky="w")
        self.applyBinsButton = ttk.Button(table_frame, text="Apply selected bins to group", command=self._apply_selected_bins)
        self.applyBinsButton.grid(row=3, column=0, sticky="w")
        self.figure = Figure(figsize=(6, 3), dpi=100, constrained_layout=True)
        self.plotAxes = self.figure.add_subplot(111)
        self.plotCanvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.plotCanvas.get_tk_widget().pack(fill="both", expand=True)
        self.resultsText = tk.Text(details_frame, wrap="word", height=14)
        self.resultsText.pack(fill="both", expand=True)
        for container, canvas in ((controls, setup_canvas), (actions, action_canvas)):
            pending = [container]
            while pending:
                widget = pending.pop()
                pending.extend(widget.winfo_children())
                widget.bind("<FocusIn>", lambda event, target=widget, viewport=canvas, content=container:
                            self._reveal_control(target, viewport, content), add="+")

    def _reveal_control(self, widget, canvas, content) -> None:
        left = widget.winfo_rootx() - content.winfo_rootx()
        top = widget.winfo_rooty() - content.winfo_rooty()
        if left < canvas.canvasx(0) or left + widget.winfo_width() > canvas.canvasx(canvas.winfo_width()):
            canvas.xview_moveto(max(0, left - 8) / max(1, content.winfo_width()))
        if top < canvas.canvasy(0) or top + widget.winfo_height() > canvas.canvasy(canvas.winfo_height()):
            canvas.yview_moveto(max(0, top - 8) / max(1, content.winfo_height()))

    def _save_current_setup(self) -> None:
        self._save_setup(self.modeVar.get(), None, None)
        self.statusVar.set("Setup saved; no calculation performed")

    def _load_persisted_setup(self) -> None:
        setup = self.project.analysis_setup if isinstance(self.project.analysis_setup, dict) else {}
        if not setup:
            return
        try:
            validate_analysis_setup(setup)
        except SampleAnalysisError as exc:
            self.statusVar.set(f"{exc}; defaults displayed, saved setup retained")
            return
        mode = setup.get("mode")
        if mode in ("Original FOAMS 1.0.5", "Modern analysis", "Compare methods"):
            self.modeVar.set(mode)
        legacy = setup.get("legacy", {}) if isinstance(setup.get("legacy"), dict) else {}
        modern = setup.get("modern", {}) if isinstance(setup.get("modern"), dict) else {}
        legacy = {key: value for key, value in legacy.items() if value is not None}
        if "minimum_diameter_px" in legacy:
            self.minimumDiameterVar.set(str(legacy["minimum_diameter_px"]))
        if "perimeter_profile" in legacy:
            self.perimeterProfileVar.set(_PERIMETER_ID_TO_LABEL.get(str(legacy["perimeter_profile"]), "R2025b"))
        if "selection_mode" in legacy:
            self.selectionModeVar.set(str(legacy["selection_mode"]))
        if "manual_ranges" in legacy and isinstance(legacy["manual_ranges"], (list, tuple)):
            pieces = []
            for item in legacy["manual_ranges"]:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    pieces.append(f"{item[0]}:{item[1]}:{item[2]}")
            self.manualRangesVar.set(";".join(pieces))
        if "volume_normalization_mode" in legacy:
            self.volumeModeVar.set(str(legacy["volume_normalization_mode"]))
        if "known_vesicularity_percent" in legacy and legacy["known_vesicularity_percent"] is not None:
            self.knownVesicularityVar.set(str(legacy["known_vesicularity_percent"]))
        if "shape_minimum_diameter_px" in legacy and legacy["shape_minimum_diameter_px"] is not None:
            self.shapeMinimumVar.set(str(legacy["shape_minimum_diameter_px"]))
        if "shape_statistics_mode" in legacy:
            self.shapeModeVar.set(str(legacy["shape_statistics_mode"]))
        if "size_lower_cutoff_mm" in legacy:
            self.sizeLowerCutoffVar.set(str(legacy["size_lower_cutoff_mm"]))
        if "size_upper_cutoff_mm" in legacy:
            self.sizeUpperCutoffVar.set(str(legacy["size_upper_cutoff_mm"]))

        if "min_detectable_mm" in modern and modern["min_detectable_mm"] is not None:
            self.modernMinDetectableVar.set(str(modern["min_detectable_mm"]))
        if "max_reliable_mm" in modern and modern["max_reliable_mm"] is not None:
            self.modernMaxReliableVar.set(str(modern["max_reliable_mm"]))
        if "transition_bins" in modern and isinstance(modern["transition_bins"], list):
            self.modernTransitionBinsVar.set(",".join(str(item) for item in modern["transition_bins"]))
        self.modernReconstructVar.set(bool(modern.get("enable_reconstruction", False)))
        self.modernUpperTailVar.set(modern.get("upper_tail_assumption") == UPPER_TAIL_ASSUMPTION)
        if modern.get("bin_edges_mm"):
            self.modernEdgesVar.set(",".join(str(value) for value in modern["bin_edges_mm"]))
        if modern.get("nesting_segments"):
            self.modernSegmentsVar.set(";".join(f"{group}:{start}:{stop}" for group, start, stop in modern["nesting_segments"]))
        self._source_keys = tuple(tuple(pair) for pair in setup.get("source_keys", legacy.get("source_keys", ())))
        drafts = setup.get("drafts", {})
        if isinstance(drafts, dict):
            for name, value in drafts.items():
                variable = getattr(self, name, None)
                if isinstance(variable, tk.Variable) and name != "statusVar" and isinstance(value, (str, bool)):
                    variable.set(value)
        self.statusVar.set("Saved setup restored; no completed calculation restored")
        missing = set(setup.get("referenced_images", ())) - {image.image_id for image in self.snapshot.images}
        if missing:
            self.statusVar.set("Saved input references changed; refresh/revalidate required")

    def _refresh_summary_banner(self) -> None:
        lines = [
            f"Sample: {self.project.sample_name}",
            f"Included accepted images: {len(self.snapshot.images)} | Binary pore masks | {self.modeVar.get()}",
        ]
        lines.append(", ".join(f"{image.display_name} ({image.mm_per_px:.6g} mm/px)" for image in self.snapshot.images[:4]))
        if len(self.snapshot.images) > 4:
            lines.append(f"and {len(self.snapshot.images) - 4} more; full identities in Details")
        self.snapshotSummary.configure(text="\n".join(lines))

    def _parse_manual_ranges(self) -> tuple[tuple[str, float, float], ...]:
        raw = self.manualRangesVar.get().strip()
        if not raw:
            return ()
        result = []
        for token in raw.split(";"):
            part = token.strip()
            if not part:
                continue
            pieces = [value.strip() for value in part.split(":")]
            if len(pieces) != 3:
                raise SampleAnalysisError("manual ranges must be group:lower:upper entries separated by ';'")
            group_id, lower_text, upper_text = pieces
            result.append((group_id, float(lower_text), float(upper_text)))
        return tuple(result)

    def _parse_legacy_config(self) -> LegacyRunConfig:
        minimum = float(self.minimumDiameterVar.get())
        known_text = self.knownVesicularityVar.get().strip()
        known = float(known_text) if known_text else None
        shape_text = self.shapeMinimumVar.get().strip()
        shape_minimum = float(shape_text) if shape_text and self.shapeModeVar.get() == "updated_minimum" else minimum
        perimeter_label = self.perimeterProfileVar.get()
        perimeter_profile = _PERIMETER_LABEL_TO_ID.get(perimeter_label)
        if perimeter_profile is None:
            raise SampleAnalysisError(f"unsupported perimeter profile label: {perimeter_label}")

        lower_cutoff = float(self.sizeLowerCutoffVar.get())
        upper_cutoff = float(self.sizeUpperCutoffVar.get())

        return LegacyRunConfig(
            minimum_diameter_px=minimum,
            perimeter_profile=perimeter_profile,
            selection_mode=self.selectionModeVar.get(),
            manual_ranges=self._parse_manual_ranges(),
            volume_normalization_mode=self.volumeModeVar.get(),
            known_vesicularity_percent=known,
            size_lower_cutoff_mm=lower_cutoff,
            size_upper_cutoff_mm=upper_cutoff,
            shape_minimum_diameter_px=shape_minimum,
            shape_statistics_mode=self.shapeModeVar.get(),
            source_keys=self._source_keys,
        )

    def _parse_modern_config(self) -> ModernRunConfig:
        min_text = self.modernMinDetectableVar.get().strip()
        max_text = self.modernMaxReliableVar.get().strip()
        transition_text = self.modernTransitionBinsVar.get().strip()
        if transition_text:
            transitions = tuple(int(part.strip()) for part in transition_text.split(",") if part.strip())
        else:
            transitions = ()
        return ModernRunConfig(
            bin_edges_mm=tuple(float(part) for part in self.modernEdgesVar.get().split(",")) if self.modernEdgesVar.get().strip() else None,
            min_detectable_mm=float(min_text) if min_text else None,
            max_reliable_mm=float(max_text) if max_text else None,
            transition_bins=transitions,
            enable_reconstruction=bool(self.modernReconstructVar.get()),
            nesting_segments=tuple(
                (parts[0].strip(), int(parts[1]), int(parts[2]))
                for parts in (token.split(":") for token in self.modernSegmentsVar.get().split(";"))
            ) if self.modernSegmentsVar.get().strip() else (),
            upper_tail_assumption=UPPER_TAIL_ASSUMPTION if self.modernUpperTailVar.get() else None,
        )

    def _legacy_prepare_key(self, snapshot: ProjectAnalysisSnapshot, config: LegacyRunConfig) -> tuple:
        return (
            snapshot.project_id,
            tuple(sorted((item.image_id, item.snapshot_id, item.mm_per_px, item.processing_roi) for item in snapshot.images)),
            config.source_keys,
            float(config.minimum_diameter_px),
            str(config.perimeter_profile),
        )

    def _parse_and_snapshot(self):
        mode = self.modeVar.get()
        if mode not in ("Original FOAMS 1.0.5", "Modern analysis", "Compare methods"):
            raise SampleAnalysisError("unsupported method preset")
        legacy_cfg = self._parse_legacy_config() if mode in ("Original FOAMS 1.0.5", "Compare methods") else None
        modern_cfg = self._parse_modern_config() if mode in ("Modern analysis", "Compare methods") else None
        snapshot = create_analysis_snapshot(self.project)
        self.snapshot = snapshot
        self._refresh_summary_banner()
        return mode, snapshot, legacy_cfg, modern_cfg

    def _save_setup(self, mode: str, legacy_cfg: Optional[LegacyRunConfig], modern_cfg: Optional[ModernRunConfig]) -> None:
        previous = self.project.analysis_setup or {}
        payload = {
            "schema": 2,
            "mode": mode,
            "legacy": asdict(legacy_cfg) if legacy_cfg is not None else previous.get("legacy", {}),
            "modern": asdict(modern_cfg) if modern_cfg is not None else previous.get("modern", {}),
            "source_keys": list(getattr(self, "_source_keys", ())),
            "drafts": {name: variable.get() for name, variable in getattr(self, "_variables", {}).items()},
            "referenced_images": [image.image_id for image in self.snapshot.images] if hasattr(self, "snapshot") else [],
        }
        validate_analysis_setup(payload)
        self.project.analysis_setup = payload
        if self._mark_unsaved is not None:
            self._mark_unsaved()

    def _schedule_after(self, delay_ms: int, callback) -> None:
        holder = {}

        def invoke() -> None:
            after_id = holder.get("id")
            if after_id is not None:
                self._owned_after_ids.discard(after_id)
            if not self._closed:
                callback()

        after_id = self.after(delay_ms, invoke)
        holder["id"] = after_id
        self._owned_after_ids.add(after_id)

    def _run_analysis(self) -> None:
        self._start_job("run")

    def _input_signature(self) -> tuple:
        owner_current = self._project_owner is None or (
            getattr(self._project_owner, "project", None) is self.project
            and getattr(self._project_owner, "projectSessionId", None) == self._project_session)
        return (self.project.project_id, self._project_session, owner_current, tuple(sorted(
            (image.image_id, image.accepted_snapshot.snapshot_id if image.accepted_snapshot else None,
             calibration_mm_per_px(image),
             image.source_revision, image.processing_roi)
            for image in self.project.images if image.analysis_included
        )))

    def _settings_signature(self) -> tuple:
        return tuple((name, variable.get()) for name, variable in self._variables.items() if name != "binGroupVar")

    def _settings_changed(self, *_args) -> None:
        if self._completed_signature is not None and self._settings_signature() != self._completed_signature:
            self.statusVar.set("Settings changed - rerun required; previous result retained")

    def _manual_changed(self, *_args) -> None:
        if not self._setting_proposal:
            self.selectionModeVar.set("manual")

    def _watch_inputs(self) -> None:
        if self._last_completed_snapshot is not None and self._completed_inputs != self._input_signature():
            self.statusVar.set("Inputs changed - rerun required; previous result retained")
        self._schedule_after(200, self._watch_inputs)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            self._control_states = []
            pending = [self.setupControls, self.binGroupSelector, self.applyBinsButton]
            while pending:
                widget = pending.pop()
                pending.extend(widget.winfo_children())
                if isinstance(widget, (ttk.Entry, ttk.Combobox, ttk.Checkbutton, ttk.Button)):
                    self._control_states.append((widget, str(widget.cget("state"))))
                    widget.configure(state="disabled")
        else:
            for widget, state in self._control_states:
                widget.configure(state=state)
            self._control_states = []
        for button in (self.runButton, self.suggestButton, self.prepareButton, self.previewButton, self.shapeUpdateButton, self.saveSetupButton):
            button.configure(state="disabled" if busy else "normal")
        self.cancelButton.configure(state="normal" if busy else "disabled")

    def _cancel_job(self) -> None:
        self._active_generation += 1
        self._active_request = None
        self._set_busy(False)
        self.statusVar.set("Cancelled; previous result retained")

    def _open_measurements(self) -> None:
        from .stereology import StereologyWindow
        snapshot = create_analysis_snapshot(self.project)
        StereologyWindow(self, images=snapshot.working_images, labels=snapshot.labels, scales=snapshot.scales)

    def _update_shape_minimum(self) -> None:
        if not self._busy:
            self.shapeModeVar.set("updated_minimum")
            self._start_job("run")

    def _start_job(self, action: str) -> None:
        if self._busy:
            return
        try:
            if not self._input_signature()[2]:
                raise SampleAnalysisError("Project session changed; reopen analysis from the current project")
            mode = self.modeVar.get()
            snapshot = create_analysis_snapshot(self.project)
            self.snapshot = snapshot
            legacy_cfg = modern_cfg = None
            legacy_error = modern_error = None
            if mode in ("Original FOAMS 1.0.5", "Compare methods"):
                try:
                    if action in ("prepare", "suggest"):
                        legacy_cfg = LegacyRunConfig(minimum_diameter_px=float(self.minimumDiameterVar.get()),
                            perimeter_profile=_PERIMETER_LABEL_TO_ID[self.perimeterProfileVar.get()], source_keys=self._source_keys)
                    else:
                        legacy_cfg = self._parse_legacy_config()
                    _, _, assignments = build_legacy_group_inputs(snapshot, self._source_keys)
                    mapping = dict(self._source_keys)
                    mapping.update((item.image_id, item.source_key) for item in assignments)
                    self._source_keys = tuple(sorted(mapping.items()))
                    legacy_cfg = replace(legacy_cfg, source_keys=self._source_keys)
                except Exception as exc:
                    legacy_error = exc
            if mode in ("Modern analysis", "Compare methods") and action != "suggest":
                try:
                    modern_cfg = self._parse_modern_config()
                except Exception as exc:
                    modern_error = exc
            if mode != "Compare methods" and (legacy_error or modern_error):
                raise legacy_error or modern_error
            if action == "suggest" and legacy_cfg is None:
                raise SampleAnalysisError("Select Original or Compare before suggesting legacy cutoffs")
            self._save_setup(mode, legacy_cfg if action not in ("prepare", "suggest") else None, modern_cfg)
        except Exception as exc:
            self._show_failure("Validation", exc)
            return

        self._job_generation = max(self._job_generation, self._active_generation) + 1
        generation = self._job_generation
        self._active_generation = generation
        request = dict(action=action, mode=mode, inputs=self._input_signature(), settings=self._settings_signature(),
                       snapshot=snapshot, legacy=legacy_cfg, modern=modern_cfg)
        self._active_request = request
        self._set_busy(True)
        self.statusVar.set(f"{action.capitalize()}: {mode}...")
        key = self._legacy_prepare_key(snapshot, legacy_cfg) if legacy_cfg is not None else None
        cached = self._legacy_prepare_cache.get(key)

        def worker() -> None:
            payload = dict(legacy=None, modern=None, legacy_error=legacy_error, modern_error=modern_error,
                           legacy_prepared=None, modern_prepared=None, suggestion=None, key=key)
            try:
                if legacy_cfg is not None:
                    try:
                        context = cached or prepare_legacy_analysis(snapshot, legacy_cfg)
                        payload["legacy_prepared"] = context
                        if action == "suggest":
                            groups = tuple(Foams105CutoffGroupInput(group.group_id, group.scale_px_per_mm,
                                group.normalization_result.na_per_mm2) for group in context.prepared.groups
                                if group.normalization_result is not None)
                            payload["suggestion"] = suggest_foams105_cutoffs(
                                context.prepared.label_preparation_result.bin_labels_mm, groups)
                        elif action != "prepare":
                            payload["legacy"] = analyze_legacy_prepared(context, legacy_cfg)
                    except Exception as exc:
                        payload["legacy_error"] = exc
                if modern_cfg is not None:
                    try:
                        prepared = prepare_modern_analysis(snapshot, modern_cfg)
                        payload["modern_prepared"] = prepared
                        if action != "prepare":
                            payload["modern"] = analyze_modern_prepared(snapshot, prepared, modern_cfg)
                    except Exception as exc:
                        payload["modern_error"] = exc
                self._result_queue.put((generation, payload, None, snapshot))
            except Exception as exc:
                self._result_queue.put((generation, None, exc, snapshot))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
        self._schedule_after(30, self._poll_results)

    def _poll_results(self) -> None:
        if self._closed or not self._busy:
            return
        try:
            generation, result, error, snapshot = self._result_queue.get_nowait()
        except queue.Empty:
            self._schedule_after(30, self._poll_results)
            return

        if generation != self._active_generation:
            if self._busy:
                self._schedule_after(30, self._poll_results)
            return

        request = self._active_request
        self._set_busy(False)
        if request is None or request["inputs"] != self._input_signature() or request["settings"] != self._settings_signature():
            self.statusVar.set("Discarded stale completion; refresh and rerun required")
            return
        if error is not None:
            self._show_failure(request["mode"], error)
            return

        previous_view = self._capture_view()
        try:
            context = result["legacy_prepared"]
            if context is not None:
                self._legacy_prepare_cache[result["key"]] = context
                self._prepared_legacy = context
            self._prepared_modern = result["modern_prepared"]
            if request["action"] in ("prepare", "suggest"):
                self._render_prepared(context, self._prepared_modern)
                suggestion = result["suggestion"]
                if suggestion is not None:
                    self._setting_proposal = True
                    self.manualRangesVar.set(";".join(f"{item.group_id}:{item.lower_label_mm!r}:{item.upper_label_mm!r}"
                        for item in suggestion.suggested_ranges))
                    self._setting_proposal = False
                    self.selectionModeVar.set("manual")
                self.statusVar.set("Prepared distributions available" if suggestion is None else "Suggested ranges ready for review (Manual)")
            elif request["mode"] == "Compare methods":
                comparison = CompareRunResult(result["legacy"], result["legacy_error"], result["modern"], result["modern_error"])
                self._render_compare(comparison)
                self._latest_compare = comparison
                self.statusVar.set("Compare partial failure" if result["legacy_error"] or result["modern_error"] else "Compare run complete")
            elif result["legacy"] is not None:
                self._render_legacy(result["legacy"])
                self._latest_legacy = result["legacy"]
                self.statusVar.set("Original run complete")
            elif result["modern"] is not None:
                self._render_modern(result["modern"])
                self._latest_modern = result["modern"]
                self.statusVar.set("Modern run complete")
            elif result["modern_prepared"] is not None or context is not None:
                if self._last_completed_snapshot is None:
                    self._render_prepared(context, result["modern_prepared"])
            if result["legacy"] is not None or result["modern"] is not None:
                self._last_completed_snapshot = snapshot
                self._completed_inputs = request["inputs"]
                self._completed_signature = request["settings"]
                self._completed_request = request
            self._append_details([f"Captured request: {request['mode']} / {request['action']}",
                f"Legacy settings: {request['legacy']!r}", f"Modern settings: {request['modern']!r}",
                f"Project: {snapshot.project_id}"] + [f"{item.image_id} snapshot={item.snapshot_id}" for item in snapshot.images])
            for method in ("legacy", "modern"):
                if result[f"{method}_error"] is not None:
                    self._show_failure(method, result[f"{method}_error"])
            self._refresh_summary_banner()
        except Exception as exc:
            self._restore_view(previous_view)
            self._show_failure("Rendering", exc)

    def _suggest_cutoffs(self) -> None:
        self._start_job("suggest")

    def _append_details(self, lines) -> None:
        self.resultsText.configure(state="normal")
        self.resultsText.insert("end", "\n" + "\n".join(lines))
        self.resultsText.configure(state="disabled")

    def _show_failure(self, method, error) -> None:
        self.statusVar.set(f"{method} failed; previous results retained")
        self._append_details([f"{method}: Analysis run failed", str(error),
            f"stage={getattr(error, 'stage', method)} context={getattr(error, 'context', {})}"])

    def _clear_distribution(self) -> None:
        self.distributionTable.delete(*self.distributionTable.get_children())
        self._table_rows = {}
        self.plotAxes.clear()
        self._nonpositive_plot_count = 0

    def _capture_view(self):
        return (self.resultsText.get("1.0", "end"),
                [(self.distributionTable.item(row_id, "values"), self._table_rows.get(row_id))
                 for row_id in self.distributionTable.get_children()],
                [(line.get_xdata(), line.get_ydata(), line.get_label()) for line in self.plotAxes.lines])

    def _restore_view(self, view) -> None:
        text, rows, curves = view
        self._clear_distribution()
        self._set_results([text])
        for values, metadata in rows:
            row_id = self.distributionTable.insert("", "end", values=values)
            self._table_rows[row_id] = metadata
        for diameters, values, name in curves:
            self._plot_values(name, diameters, values)
        self._finish_plot()

    def _add_row(self, method, group, index, lower, upper, count, area, na, nv="", volume="", kind="selected") -> None:
        row_id = self.distributionTable.insert("", "end", values=(method, group, index, lower, upper, count, area, na, nv, volume))
        self._table_rows[row_id] = (kind, group, index, lower, upper)

    def _plot_values(self, name, diameters, values) -> None:
        pairs = [(diameter, value) for diameter, value in zip(diameters, values)
                 if diameter > 0 and value > 0]
        self._nonpositive_plot_count += len(values) - len(pairs)
        if pairs:
            self.plotAxes.plot([pair[0] for pair in pairs], [pair[1] for pair in pairs], marker=".", label=name)

    def _finish_plot(self) -> None:
        self.plotAxes.set_xlabel("Diameter / label (mm)")
        self.plotAxes.set_ylabel("N_A (mm^-2)")
        if self.plotAxes.lines:
            self.plotAxes.set_xscale("log")
            self.plotAxes.set_yscale("log")
            self.plotAxes.xaxis.set_major_locator(LogLocator(numticks=3))
            self.plotAxes.yaxis.set_major_locator(LogLocator(numticks=3))
            self.plotAxes.xaxis.set_minor_locator(NullLocator())
            self.plotAxes.yaxis.set_minor_locator(NullLocator())
            self.plotAxes.legend(fontsize=8)
        self.plotAxes.set_title(f"Nonpositive/unavailable rows omitted from log plot: {self._nonpositive_plot_count}", fontsize=9)
        self.plotCanvas.draw_idle()

    def _render_prepared(self, legacy, modern, clear=True) -> None:
        if clear:
            self._clear_distribution()
            self._set_results(["Prepared distributions (not a completed analysis)"])
        group_ids = []
        if legacy is not None:
            for group in legacy.prepared.groups:
                group_ids.append(group.group_id)
                histogram = group.histogram_result
                normalization = group.normalization_result
                self._append_details([f"Original {group.group_id}: area1={group.group_area_result.phase_corrected_area_mm2!r} "
                    f"area2={group.group_area_result.border_corrected_area_mm2!r}; scale={group.scale_px_per_mm!r} px/mm; images={len(group.images)}",
                    f"Diagnostics: {group.diagnostics!r}"])
                for index, (label, count) in enumerate(zip(histogram.bin_labels_mm, histogram.counts)):
                    na = normalization.na_per_mm2[index] if normalization is not None else "unavailable"
                    self._add_row("Original prepared", group.group_id, index, label, label, count,
                                  group.group_area_result.phase_corrected_area_mm2, na, kind="legacy")
                if normalization is not None:
                    self._plot_values(f"Original {group.group_id}", histogram.bin_labels_mm, normalization.na_per_mm2)
        if modern is not None:
            group_ids.extend(modern.group_ids)
            distribution = modern.distribution_result
            self._append_details([f"Modern measured objects: {len(modern.measurements)}",
                                  f"Density availability: {modern.distribution_error or 'available'}"])
            if distribution is None:
                for pore in modern.measurements:
                    self._append_details([f"Modern image_index={pore.image_index} label={pore.label} diameter_mm={pore.eq_diam_units!r}"])
            else:
                for (_, group_id), group in distribution.groups.items():
                    for index, bin_ in enumerate(distribution.bin_spec.bins):
                        self._add_row("Modern prepared", group_id, index, bin_.lower_mm, bin_.upper_mm,
                            group.counts[index], group.eligible_areas_mm2[index], group.number_densities_per_mm2[index], kind="modern")
                    self._plot_values(f"Modern {group_id}",
                        tuple(bin_.geometric_midpoint_mm for bin_ in distribution.bin_spec.bins), group.number_densities_per_mm2)
                    self._append_details([f"Modern {group_id} supported bins: {group.supported!r}"])
                self._append_details([f"Modern distribution diagnostics: {distribution.diagnostics!r}"])
        self.binGroupSelector.configure(values=tuple(dict.fromkeys(group_ids)))
        self._finish_plot()

    def _select_bins(self, _event=None) -> None:
        selected = [self._table_rows[row_id] for row_id in self.distributionTable.selection()]
        if selected:
            self.binGroupVar.set(selected[0][1])

    def _apply_selected_bins(self) -> None:
        if self._busy:
            return
        selected = [self._table_rows[row_id] for row_id in self.distributionTable.selection()]
        try:
            if not selected or len({row[0] for row in selected}) != 1:
                raise SampleAnalysisError("Select prepared bins from one method")
            group_id = self.binGroupVar.get()
            if not group_id:
                raise SampleAnalysisError("Select a source group")
            indices = sorted({row[2] for row in selected})
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise SampleAnalysisError("Select contiguous bins")
            if selected[0][0] == "legacy":
                ranges = [item for item in self._parse_manual_ranges() if item[0] != group_id]
                ranges.append((group_id, min(row[3] for row in selected), max(row[4] for row in selected)))
                self.manualRangesVar.set(";".join(f"{group}:{lower!r}:{upper!r}" for group, lower, upper in ranges))
            elif selected[0][0] == "modern":
                config = self._parse_modern_config()
                segments = [segment for segment in config.nesting_segments if segment[0] != group_id]
                segments.append((group_id, indices[0], indices[-1] + 1))
                segments.sort(key=lambda segment: segment[1])
                self.modernSegmentsVar.set(";".join(f"{group}:{start}:{stop}" for group, start, stop in segments))
            else:
                raise SampleAnalysisError("Select prepared distribution rows, not composite rows")
        except Exception as exc:
            self._show_failure("Bin selection", exc)

    def _render_legacy(self, result) -> None:
        analysis = result.analysis
        lines = [
            "Original FOAMS 1.0.5",
            f"Method: {analysis.method}",
            f"Image count: {analysis.image_count}",
            f"Prepared groups: {len(analysis.prepared_result.groups)}",
            f"Selected rows: {analysis.binned_selection_result.output_row_count}",
            f"Raw selected objects: {analysis.raw_selection_result.selected_count}",
            f"Volume mode: {analysis.analysis_settings.volume_normalization_mode}",
            f"Diagnostics: {len(analysis.diagnostics)}",
            "",
            "Pre-nesting group distributions:",
        ]
        for group in analysis.prepared_result.groups:
            normalization = group.normalization_result
            status = "available" if normalization is not None else "unavailable"
            lines.append(
                f" - {group.group_id} slot={group.source_slot} scale_px_per_mm={group.scale_px_per_mm:.8g} "
                + f"images={len(group.images)} count_rows={len(group.histogram_result.counts)} "
                + f"area1={group.group_area_result.phase_corrected_area_mm2:.8g} "
                + f"area2={group.group_area_result.border_corrected_area_mm2:.8g} normalization={status}"
            )
        lines.extend([
            "",
            "Applied selected rows (label_mm -> N_A):",
        ])
        for row in analysis.binned_selection_result.rows:
            lines.append(f" - slot={row.source_slot} group={row.group_id} label={row.label_mm:.8g} na={row.na_per_mm2:.8g}")
        if not getattr(self, "_render_append", False):
            self._clear_distribution()
            self._set_results(lines)
        else:
            self._append_details(lines)
        self._render_prepared(LegacyPreparedContext(result.prepared, result.source_assignments, ()), None, clear=False)
        for index, row in enumerate(analysis.binned_selection_result.rows):
            self._add_row("Original selected", row.group_id, index, row.label_mm, row.label_mm, "",
                "", row.na_per_mm2, analysis.conversion_result.signed_nv_per_mm3[index],
                analysis.volume_result.adjusted_volume_fractions[index])
        self._append_details([f"Raw selected identities: {analysis.raw_selection_result.selected_identities!r}",
            f"Shape settings: {analysis.analysis_settings.shape_statistics_mode}; threshold={analysis.analysis_settings.shape_minimum_diameter_px!r}",
            f"Shape statistics: {analysis.shape_statistics_result!r}",
            f"Volume summary: {analysis.volume_result!r}", f"Diagnostics: {analysis.diagnostics!r}"])
        self._plot_values("Original selected", tuple(row.label_mm for row in analysis.binned_selection_result.rows),
                      tuple(row.na_per_mm2 for row in analysis.binned_selection_result.rows))
        self._finish_plot()

    def _render_modern(self, result) -> None:
        distribution = result.distribution_result
        nested = result.nesting_result
        lines = [
            "Modern analysis",
            f"Group count: {len(result.group_ids)}",
            f"Bin count: {len(result.bin_edges_mm) - 1}",
            f"Selected bins: {len(nested.selected_bin_indices)}",
            f"Nested rows: {len(nested.counts)}",
            f"Reconstruction: {'enabled' if result.reconstruction_result is not None else 'disabled'}",
            "",
            "Group N_A summary:",
        ]
        for key, group in distribution.groups.items():
            lines.append(
                f" - {key[1]}: bins={len(group.counts)}, total_count={sum(group.counts)}, "
                + f"supported_bins={sum(1 for flag in group.supported if flag)}"
            )
        lines.append("")
        lines.append("Nesting source rows:")
        for index, group_id in enumerate(nested.source_group_ids):
            lines.append(
                f" - bin_index={nested.selected_bin_indices[index]} group={group_id} count={nested.counts[index]} "
                + f"area={nested.eligible_areas_mm2[index]:.8g} na={nested.number_densities_per_mm2[index]:.8g}"
            )
        if not getattr(self, "_render_append", False):
            self._clear_distribution()
            self._set_results(lines)
        else:
            self._append_details(lines)
        self._render_prepared(None, ModernPreparedResult(distribution, None, (), distribution.source_images,
                      result.bin_edges_mm, result.group_ids), clear=False)
        for index, group_id in enumerate(nested.source_group_ids):
            self._add_row("Modern selected", group_id, nested.selected_bin_indices[index], nested.lower_edges_mm[index],
                nested.upper_edges_mm[index], nested.counts[index], nested.eligible_areas_mm2[index],
                nested.number_densities_per_mm2[index],
                result.reconstruction_result.signed_nv_per_mm3[index] if result.reconstruction_result is not None else "")
        self._append_details([f"Overlap diagnostics: {nested.overlap_diagnostics!r}",
            f"Reconstruction (signed): {result.reconstruction_result!r}"])
        self._plot_values("Modern selected", tuple((lower * upper) ** 0.5 for lower, upper in zip(nested.lower_edges_mm, nested.upper_edges_mm)),
                          nested.number_densities_per_mm2)
        self._finish_plot()

    def _render_compare(self, result: CompareRunResult) -> None:
        previous_text = self.resultsText.get("1.0", "end")
        self._clear_distribution()
        lines = ["Compare methods", ""]
        if result.legacy_result is not None:
            lines.append("Legacy path: success")
            lines.append(
                f" - selected_rows={result.legacy_result.analysis.binned_selection_result.output_row_count}, "
                + f"raw_selected={result.legacy_result.analysis.raw_selection_result.selected_count}"
            )
        else:
            lines.append("Legacy path: failed")
            lines.append(f" - error={result.legacy_error}")

        lines.append("")
        if result.modern_result is not None:
            lines.append("Modern path: success")
            lines.append(
                f" - selected_bins={len(result.modern_result.nesting_result.selected_bin_indices)}, "
                + f"nested_rows={len(result.modern_result.nesting_result.counts)}"
            )
        else:
            lines.append("Modern path: failed")
            lines.append(f" - error={result.modern_error}")

        lines.append("")
        lines.append("Shared accepted snapshot identities:")
        for item in self.snapshot.images:
            lines.append(f" - {item.image_id} snapshot={item.snapshot_id}")
        self._set_results(lines)
        self._render_append = True
        try:
            if result.legacy_result is not None:
                self._render_legacy(result.legacy_result)
            elif self._latest_legacy is not None:
                self._append_details(["Previous Original result (not this request)"])
                self._render_legacy(self._latest_legacy)
            if result.modern_result is not None:
                self._render_modern(result.modern_result)
            elif self._latest_modern is not None:
                self._append_details(["Previous Modern result (not this request)"])
                self._render_modern(self._latest_modern)
            if result.legacy_result is None or result.modern_result is None:
                self._append_details(["Earlier result metadata (not this request):", previous_text])
        finally:
            self._render_append = False
        if result.legacy_result is not None:
            self._latest_legacy = result.legacy_result
        if result.modern_result is not None:
            self._latest_modern = result.modern_result

    def _set_results(self, lines) -> None:
        self.resultsText.configure(state="normal")
        self.resultsText.delete("1.0", "end")
        self.resultsText.insert("1.0", "\n".join(lines))
        self.resultsText.configure(state="disabled")

    def _on_close(self) -> None:
        self._closed = True
        for after_id in tuple(self._owned_after_ids):
            try:
                self.after_cancel(after_id)
            except Exception:
                pass
            self._owned_after_ids.discard(after_id)
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()
