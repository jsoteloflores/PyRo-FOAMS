from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, NullLocator

from ..core.analysis_runs import append_run, create_run_projection, error_diagnostic
from ..core.legacy_cutoffs import Foams105CutoffGroupInput, suggest_foams105_cutoffs
from ..core.legacy_perimeter import (
    FOAMS105_PERIMETER_PROFILE_CURRENT,
    FOAMS105_PERIMETER_PROFILE_OLD,
)
from ..core.reconstruction import UPPER_TAIL_ASSUMPTION
from ..core.run_exports import export_selected_run, run_tables
from ..core.sample_analysis import (
    CapturedAnalysisRequest,
    CompareRunResult,
    LegacyPreparedContext,
    LegacyRunConfig,
    MethodResultEnvelope,
    ModernPreparedResult,
    ModernRunConfig,
    SampleAnalysisError,
    analyze_legacy_prepared,
    analyze_modern_prepared,
    build_legacy_group_inputs,
    normalize_analysis_setup,
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
        snapshot: Optional[ProjectAnalysisSnapshot],
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
        self._workspace_session = uuid.uuid4().hex
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
        self._successful_results = {}
        self._current_view = None
        self._displayed_request = None
        self._displayed_envelopes = ()
        self._displayed_outcomes = ()
        self._diagnostic_inputs = None
        self._selected_saved_run = None
        self._saved_view_data = None
        self._saved_labels = {}
        self._history_queue = queue.Queue()
        self._view_generation = 0
        self._history_loading = False
        self._object_page = 0
        self._distribution_page = 0
        self._distribution_rows = []
        self._object_rows = []
        self._object_row_by_id = {}
        self._export_cancel = threading.Event()
        self._export_queue = queue.Queue()
        self._exporting = False
        self._projection_views = {}
        self._table_run_id = None
        self._history_previous = None

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
        self._set_results([f"Sample: {self.project.sample_name}", "Included accepted binary-mask inputs:"]
            + [f"{image.display_name}: {image.image_id} snapshot={image.snapshot_id} calibration={image.mm_per_px!r} mm/px"
                    for image in self.snapshot.images] if self.snapshot is not None else
                [f"Sample: {self.project.sample_name}", "History view; no current analysis snapshot captured"])
        self._variables = {
            name: value for name, value in vars(self).items() if isinstance(value, tk.Variable)
            and name not in ("statusVar", "resultViewVar", "resultProvenanceVar", "tableMethodVar", "tableGroupVar", "tableStageVar")
        }
        for variable in self._variables.values():
            variable.trace_add("write", self._settings_changed)
        self.manualRangesVar.trace_add("write", self._manual_changed)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.geometry("1200x800")
        self.minsize(980, 640)
        self._refresh_summary_banner()
        self._schedule_after(200, self._watch_inputs)
        self._update_result_choices()
        if self._current_input_problem():
            self.statusVar.set("Current inputs unavailable; saved history remains accessible")
        if self.project.unavailable_history:
            self._append_details(["Unavailable historical runs:", json.dumps(self.project.unavailable_history, ensure_ascii=False, indent=2)])
            self.statusVar.set("Historical runs unavailable; see Details. Preparation remains usable")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        self._layout_root = root
        self._layout_pending = False
        self._setup_visible = True
        self._plot_entered = False
        root.columnconfigure(0, weight=1)
        root.rowconfigure(5, weight=1)

        summary = ttk.Frame(root)
        summary.grid(row=0, column=0, sticky="ew")
        summary.columnconfigure(0, weight=1)
        self.snapshotSummary = ttk.Label(summary, justify="left")
        self.snapshotSummary.grid(row=0, column=0, sticky="w")
        self.includedSummary = ttk.Label(summary)
        self.includedSummary.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self._summary_frame = summary

        setup = ttk.LabelFrame(root, text="Setup", padding=4)
        setup.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.setupBody = setup
        setup_canvas = tk.Canvas(setup, height=230, highlightthickness=0)
        self._setup_canvas = setup_canvas
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

        primary = ttk.Frame(root)
        primary.grid(row=2, column=0, sticky="ew", pady=4)
        self._primary_actions = primary
        self.runButton = ttk.Button(primary, text="Run analysis", command=self._run_analysis)
        self.runButton.grid(row=0, column=0, sticky="w")
        self.cancelButton = ttk.Button(primary, text="Cancel", command=self._cancel_job, state="disabled")
        self.cancelButton.grid(row=0, column=1, sticky="w", padx=8)
        self.setupToggleButton = ttk.Button(primary, text="Hide setup", command=self._toggle_setup)
        self.setupToggleButton.grid(row=0, column=2, sticky="w")
        self.backButton = ttk.Button(primary, text="Back to preparation", command=self._on_close)
        self.backButton.grid(row=0, column=3, sticky="w", padx=8)
        action_frame = ttk.Frame(root)
        action_frame.grid(row=3, column=0, sticky="ew")
        self._secondary_actions = action_frame
        action_canvas = tk.Canvas(action_frame, height=44, highlightthickness=0)
        action_canvas.pack(fill="x")
        action_scroll = ttk.Scrollbar(action_frame, orient="horizontal", command=action_canvas.xview)
        action_scroll.pack(fill="x")
        action_canvas.configure(xscrollcommand=action_scroll.set)
        actions = ttk.Frame(action_canvas)
        action_canvas.create_window((0, 0), window=actions, anchor="nw")
        actions.bind("<Configure>", lambda event: (action_canvas.configure(scrollregion=action_canvas.bbox("all")),
                              action_canvas.configure(height=actions.winfo_reqheight())))
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
        self.useRunSettingsButton = ttk.Button(actions, text="Use this run's settings", command=self._use_run_settings)
        self.useRunSettingsButton.pack(side="left", padx=8)
        self.exportRunButton = ttk.Button(actions, text="Export selected run", command=self._export_run)
        self.exportRunButton.pack(side="left", padx=8)
        ttk.Button(actions, text="Measurements", command=self._open_measurements).pack(side="left", padx=8)
        status_frame = ttk.Frame(root)
        status_frame.grid(row=4, column=0, sticky="ew", pady=4)
        status_frame.columnconfigure(0, weight=1)
        self._status_frame = status_frame
        self._status_canvas = tk.Canvas(status_frame, height=24, highlightthickness=0)
        self._status_canvas.grid(row=0, column=0, sticky="ew")
        self.statusLabel = ttk.Label(self._status_canvas, textvariable=self.statusVar, justify="left")
        self._status_window = self._status_canvas.create_window((0, 0), window=self.statusLabel, anchor="nw")
        self._status_scroll = ttk.Scrollbar(status_frame, orient="vertical", command=self._status_canvas.yview)
        self._status_canvas.configure(yscrollcommand=self._status_scroll.set)
        self.statusLabel.bind("<Configure>", self._queue_layout)

        results = ttk.Frame(root, padding=(0, 4))
        results.grid(row=5, column=0, sticky="nsew")
        results.columnconfigure(0, weight=1)
        results.rowconfigure(2, weight=1)
        self._results_frame = results
        self.resultViewVar = tk.StringVar(value="Current request")
        self.resultProvenanceVar = tk.StringVar(value="No completed result selected")
        self.resultViewSelector = ttk.Combobox(results, textvariable=self.resultViewVar,
            values=("Current request",), state="readonly", width=60)
        self.resultViewSelector.grid(row=0, column=0, sticky="ew")
        self.resultViewSelector.bind("<<ComboboxSelected>>", self._select_result_view)
        self.resultProvenanceLabel = ttk.Label(results, textvariable=self.resultProvenanceVar, justify="left")
        self.resultProvenanceLabel.grid(row=1, column=0, sticky="ew", pady=4)
        views = ttk.Notebook(results)
        views.grid(row=2, column=0, sticky="nsew")
        self.resultsNotebook = views
        table_frame = ttk.Frame(views)
        plot_frame = ttk.Frame(views)
        details_frame = ttk.Frame(views)
        views.add(table_frame, text="Distributions")
        views.add(plot_frame, text="Plot")
        views.add(details_frame, text="Details / raw identities")
        overview_frame = ttk.Frame(views)
        objects_frame = ttk.Frame(views)
        views.insert(0, overview_frame, text="Overview")
        views.tab(details_frame, text="Details")
        views.add(objects_frame, text="Objects / shape")
        self._plot_frame = plot_frame
        self._overview_frame = overview_frame
        self.overviewText = tk.Text(overview_frame, wrap="word", height=10)
        self.overviewText.pack(fill="both", expand=True)
        self.objectsTable = ttk.Treeview(objects_frame, columns=("method", "image", "object", "diameter", "area", "selected", "shape"), show="headings", height=10)
        for name, heading in zip(self.objectsTable.cget("columns"), ("Method", "Image ID", "Object ID", "Diameter mm", "Area mm^2", "Raw selected", "Shape population")):
            self.objectsTable.heading(name, text=heading)
            self.objectsTable.column(name, width=135, stretch=False)
        objects_frame.columnconfigure(0, weight=1)
        objects_frame.rowconfigure(0, weight=1)
        self.objectsTable.grid(row=0, column=0, sticky="nsew")
        object_scroll = ttk.Scrollbar(objects_frame, orient="vertical", command=self.objectsTable.yview)
        object_scroll.grid(row=0, column=1, sticky="ns")
        object_horizontal = ttk.Scrollbar(objects_frame, orient="horizontal", command=self.objectsTable.xview)
        object_horizontal.grid(row=1, column=0, sticky="ew")
        self.objectsTable.configure(yscrollcommand=object_scroll.set, xscrollcommand=object_horizontal.set)
        object_actions = ttk.Frame(objects_frame)
        object_actions.grid(row=2, column=0, sticky="ew")
        ttk.Button(object_actions, text="Previous page", command=lambda: self._page_objects(-1)).pack(side="left")
        ttk.Button(object_actions, text="Next page", command=lambda: self._page_objects(1)).pack(side="left", padx=8)
        self.objectPageLabel = ttk.Label(object_actions)
        self.objectPageLabel.pack(side="left")
        ttk.Button(object_actions, text="Shape statistics", command=self._inspect_shape_statistics).pack(side="left", padx=8)
        for column in self.objectsTable.cget("columns"):
            self.objectsTable.heading(column, command=lambda name=column: self._sort_object_page(name))
        self.objectsTable.bind("<<TreeviewSelect>>", self._inspect_object)
        views.bind("<<NotebookTabChanged>>", self._results_tab_changed)
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
        table_filters = ttk.Frame(table_frame)
        table_filters.grid(row=4, column=0, sticky="ew")
        self.tableMethodVar = tk.StringVar(value="All")
        self.tableGroupVar = tk.StringVar(value="All")
        self.tableStageVar = tk.StringVar(value="All")
        self.tableFilters = []
        for variable, options in ((self.tableMethodVar, ("All", "legacy", "modern")),
                                  (self.tableGroupVar, ("All",)), (self.tableStageVar, ("All", "prepared", "selected"))):
            control = ttk.Combobox(table_filters, textvariable=variable, values=options, state="readonly", width=12)
            control.pack(side="left", padx=4)
            control.bind("<<ComboboxSelected>>", lambda event: self._filter_saved_tables())
            self.tableFilters.append(control)
        ttk.Button(table_filters, text="Previous", command=lambda: self._page_distributions(-1)).pack(side="left", padx=4)
        ttk.Button(table_filters, text="Next", command=lambda: self._page_distributions(1)).pack(side="left")
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
        root.bind("<Configure>", self._queue_layout)
        self.resultProvenanceLabel.bind("<Configure>", self._queue_layout)

    def _queue_layout(self, _event=None) -> None:
        if not self._closed and not self._layout_pending:
            self._layout_pending = True
            self._schedule_after(0, self._reflow_analysis_layout)

    def _toggle_setup(self) -> None:
        self._set_setup_visible(not self._setup_visible)

    def _set_setup_visible(self, visible: bool) -> None:
        self._setup_visible = visible
        if visible:
            self.setupBody.grid()
            self._secondary_actions.grid()
        else:
            self.setupBody.grid_remove()
            self._secondary_actions.grid_remove()
        self.setupToggleButton.configure(text="Hide setup" if visible else "Show setup")
        self._queue_layout()

    def _results_tab_changed(self, _event=None) -> None:
        if self.resultsNotebook.select() == str(self._plot_frame) and not self._plot_entered:
            self._plot_entered = True
            self._set_setup_visible(False)
        self._queue_layout()

    def _reflow_analysis_layout(self) -> None:
        self._layout_pending = False
        width = max(1, self._layout_root.winfo_width() - 24)
        for label in (self.statusLabel, self.resultProvenanceLabel):
            if int(label.cget("wraplength") or 0) != width:
                label.configure(wraplength=width)
        self._status_canvas.itemconfigure(self._status_window, width=width)
        line_height = tkfont.nametofont("TkDefaultFont").metrics("linespace")
        status_height = min(self.statusLabel.winfo_reqheight(), line_height * 2 + 4)
        if int(self._status_canvas.cget("height")) != status_height:
            self._status_canvas.configure(height=status_height)
        self._status_canvas.configure(scrollregion=self._status_canvas.bbox("all"))
        if self.statusLabel.winfo_reqheight() > status_height:
            self._status_scroll.grid(row=0, column=1, sticky="ns")
        else:
            self._status_scroll.grid_remove()
        reserve = 240 + self.resultViewSelector.winfo_reqheight() + self.resultProvenanceLabel.winfo_reqheight() + line_height + 32
        self._layout_root.rowconfigure(5, minsize=reserve)
        if self._setup_visible:
            fixed = self._summary_frame.winfo_reqheight() + self._primary_actions.winfo_reqheight() + self._status_frame.winfo_reqheight() + self._secondary_actions.winfo_reqheight() + 40
            setup_overhead = max(0, self.setupBody.winfo_reqheight() - self._setup_canvas.winfo_reqheight())
            height = max(40, min(230, self._layout_root.winfo_height() - reserve - fixed - setup_overhead))
            if int(self._setup_canvas.cget("height")) != height:
                self._setup_canvas.configure(height=height)
        self._refresh_summary_banner()

    def _reveal_control(self, widget, canvas, content) -> None:
        left = widget.winfo_rootx() - content.winfo_rootx()
        top = widget.winfo_rooty() - content.winfo_rooty()
        if left < canvas.canvasx(0) or left + widget.winfo_width() > canvas.canvasx(canvas.winfo_width()):
            canvas.xview_moveto(max(0, left - 8) / max(1, content.winfo_width()))
        if top < canvas.canvasy(0) or top + widget.winfo_height() > canvas.canvasy(canvas.winfo_height()):
            canvas.yview_moveto(max(0, top - 8) / max(1, content.winfo_height()))

    def _save_current_setup(self) -> None:
        try:
            self._save_setup(self.modeVar.get(), None, None)
            self.statusVar.set("Setup saved; no calculation performed")
        except SampleAnalysisError as exc:
            self._show_failure("Save setup", exc)

    def _load_persisted_setup(self) -> None:
        try:
            setup = normalize_analysis_setup(self.project.analysis_setup)
        except SampleAnalysisError as exc:
            self.statusVar.set(f"{exc}; defaults displayed, saved setup retained")
            return
        if not setup:
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
        missing = set(setup.get("referenced_images", ())) - {image.image_id for image in self.project.images}
        if missing:
            self.statusVar.set("Saved input references changed; refresh/revalidate required")

    def _refresh_summary_banner(self) -> None:
        included_count = sum(image.analysis_included for image in self.project.images)
        self.includedSummary.configure(text=f"{included_count} included images | Binary masks")
        available = max(80, self._summary_frame.winfo_width() - self.includedSummary.winfo_reqwidth() - 16)
        font = tkfont.nametofont("TkDefaultFont")
        name = self.project.sample_name
        while len(name) > 1 and font.measure("Sample: " + name) > available:
            name = name[:-2] + "." if name.endswith(".") else name[:-1] + "."
        text = "Sample: " + name
        if self.snapshotSummary.cget("text") != text:
            self.snapshotSummary.configure(text=text)

    def _current_input_problem(self):
        included = [image for image in self.project.images if image.analysis_included]
        if not included:
            return "No accepted images are included for current analysis"
        for image in included:
            if image.accepted_snapshot is None:
                return f"{image.display_name}: accepted segmentation unavailable"
            if calibration_mm_per_px(image) is None:
                return f"{image.display_name}: current calibration unavailable"
        return None

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
        previous = normalize_analysis_setup(self.project.analysis_setup)
        payload = {
            "schema": 2,
            "mode": mode,
            "legacy": asdict(legacy_cfg) if legacy_cfg is not None else previous.get("legacy", {}),
            "modern": asdict(modern_cfg) if modern_cfg is not None else previous.get("modern", {}),
            "source_keys": list(getattr(self, "_source_keys", ())),
            "drafts": {name: variable.get() for name, variable in getattr(self, "_variables", {}).items()},
            "referenced_images": [image.image_id for image in self.snapshot.images] if getattr(self, "snapshot", None) is not None else previous.get("referenced_images", []),
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
        if self._displayed_request is not None and self._view_settings_stale():
            self.statusVar.set("Settings changed - rerun required; previous result retained")
        self._refresh_result_provenance()

    def _manual_changed(self, *_args) -> None:
        if not self._setting_proposal:
            self.selectionModeVar.set("manual")

    def _watch_inputs(self) -> None:
        current_inputs = self._input_signature()
        operation_diagnostic = self.statusVar.get().startswith(("Discarded stale", "Cancelled", "Rendering failed",
            "legacy failed", "modern failed", "Validation failed", "Save setup failed")) and self._diagnostic_inputs == current_inputs
        if (not self._busy and not operation_diagnostic and self._displayed_request is not None
                and self._displayed_request.input_signature != current_inputs):
            self.statusVar.set("Inputs changed - rerun required; previous result retained")
        self._refresh_result_provenance()
        self._schedule_after(200, self._watch_inputs)

    def _method_settings_signature(self, method: str) -> tuple:
        return tuple((name, variable.get()) for name, variable in self._variables.items()
            if name not in ("modeVar", "binGroupVar") and name.startswith("modern") == (method == "modern"))

    def _view_settings_stale(self) -> bool:
        saved = dict(self._displayed_request.method_settings)
        methods = tuple(envelope.method for envelope in self._displayed_envelopes)
        if not methods:
            methods = ("legacy", "modern") if self._displayed_request.mode == "Compare methods" else (
                "modern" if self._displayed_request.mode == "Modern analysis" else "legacy",)
        return any(saved[method] != self._method_settings_signature(method) for method in methods)

    def _refresh_result_provenance(self) -> None:
        if self._selected_saved_run is not None and self._saved_view_data is not None:
            data = self._saved_view_data
            captured = {(image["image_id"], image["snapshot_id"], image["mm_per_px"], tuple(image["processing_roi"])) for image in data["images"]}
            current = {(image.image_id, image.accepted_snapshot.snapshot_id if image.accepted_snapshot else None,
                        calibration_mm_per_px(image), image.processing_roi) for image in self.project.images if image.analysis_included}
            prefix = "Saved run" if self.project.analysis_runs and self._selected_saved_run.run_id == self.project.analysis_runs[-1].run_id else "Previous saved run"
            stale = " | Inputs changed - rerun required" if captured != current else ""
            if self._current_input_problem():
                stale += " | Current inputs unavailable"
            draft_changed = any(getattr(self, name).get() != value for name, value in data["draft_settings"].items()
                                if name in self._variables and name != "binGroupVar")
            if draft_changed:
                stale += " | Settings changed"
            self.resultProvenanceVar.set(prefix + stale + " | " + self._selected_saved_run.run_id[:8] + " | "
                + "; ".join(f"{method['method_id']}: {method['outcome']}" for method in data["methods"]))
            return
        request = self._displayed_request
        if request is None:
            return
        previous = self.resultViewVar.get() != "Current request"
        state = "Previous successful result" if previous else "Current request"
        if self._current_input_problem():
            state += " | Current inputs unavailable"
        if request.input_signature != self._input_signature():
            state += " | Inputs changed - rerun required"
        elif self._view_settings_stale():
            state += " | Settings changed - rerun required"
        self.resultProvenanceVar.set(f"{state} | {request.request_id[:8]} | "
            + "; ".join(f"{method}: {status}" for method, status in self._displayed_outcomes))

    def _append_request_details(self, request: CapturedAnalysisRequest) -> None:
        saved = self._projection_views.get(request.request_id)
        self._append_details([f"Request ID: {request.request_id}", f"Session ID: {request.session_id}",
            f"Sample: {saved['sample_name'] if saved else self.project.sample_name}",
            f"Captured request: {request.mode} / {request.action}",
            f"Legacy settings: {request.legacy_config!r}", f"Modern settings: {request.modern_config!r}",
            f"Captured editable settings: {request.draft_settings!r}", f"Project: {request.snapshot.project_id}"]
            + [f"{item.image_id} snapshot={item.snapshot_id} calibration={item.mm_per_px!r} mm/px "
               f"accepted_checksum={item.accepted_checksum_sha256}" for item in request.snapshot.images])

    def _update_result_choices(self) -> None:
        current_id = self._current_view[2].request_id if self._current_view is not None else None
        options = ["Current request"] + [label for label, envelope in self._successful_results.items()
                                         if envelope.request.request_id != current_id]
        self._saved_labels = {f"Saved #{index + 1} {run.completed_at_utc[:19]} | {run.preset} | "
            f"{','.join(run.outcomes)} | {run.run_id[:8]}": run for index, run in enumerate(self.project.analysis_runs)}
        options.extend(self._saved_labels)
        self.resultViewSelector.configure(values=options)

    def _present_result(self, kind, result, request, envelopes, previous=False) -> None:
        self._selected_saved_run = None
        self._saved_view_data = None
        if not previous:
            self.resultViewVar.set("Current request")
        if kind == "compare":
            self._render_compare(result)
        elif kind == "legacy":
            self._render_legacy(result)
        elif kind == "modern":
            self._render_modern(result)
        elif kind == "failed":
            self._clear_distribution()
            self._set_results([request.mode, "Analysis failed; no numerical result for this request"])
            self._finish_plot()
        else:
            self._render_prepared(*result)
        if previous:
            self._append_details(["Previous successful result; not part of the current comparison"])
        self._append_request_details(request)
        self._displayed_request = request
        self._displayed_envelopes = envelopes
        self._displayed_outcomes = tuple((method, "success" if getattr(result, f"{method}_result") is not None else "failed")
                        for method in ("legacy", "modern")) if kind == "compare" else (
                        (("modern" if request.mode == "Modern analysis" else "legacy"), "failed") if kind == "failed" else
                        (kind, "prepared" if kind == "prepared" else "success"),)
        self._selected_view_label = self.resultViewVar.get()
        self._refresh_result_provenance()
        data = self._projection_views.get(request.request_id)
        if data is not None:
            self._populate_saved_supplement(data, data["tables"])

    def _select_result_view(self, _event=None) -> None:
        view = self._capture_view()
        selected_label = self.resultViewVar.get()
        try:
            if selected_label in self._saved_labels:
                self._load_saved_run(self._saved_labels[selected_label])
            elif selected_label == "Current request":
                if self._current_view is None:
                    return
                self._present_result(*self._current_view)
            else:
                envelope = self._successful_results[selected_label]
                self._present_result(envelope.method, envelope.result, envelope.request, (envelope,), previous=True)
        except Exception as exc:
            self._restore_view(view)
            self._show_failure("Rendering", exc)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            self._control_states = []
            pending = [self.setupControls, self.binGroupSelector, self.applyBinsButton, self.resultViewSelector]
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
        for button in (self.runButton, self.suggestButton, self.prepareButton, self.previewButton, self.shapeUpdateButton,
                   self.saveSetupButton, self.useRunSettingsButton, self.exportRunButton):
            button.configure(state="disabled" if busy else "normal")
        self.cancelButton.configure(state="normal" if busy else "disabled")

    def _cancel_job(self) -> None:
        if self._history_loading and self._history_previous is not None:
            self._restore_view(self._history_previous)
        self._view_generation += 1
        self._history_loading = False
        self._export_cancel.set()
        self._active_generation += 1
        self._active_request = None
        self._set_busy(False)
        self.statusVar.set("Cancelled; previous result retained")
        self._diagnostic_inputs = self._input_signature()

    def _open_measurements(self) -> None:
        from .stereology import StereologyWindow
        try:
            snapshot = create_analysis_snapshot(self.project)
        except ValueError as exc:
            self._show_failure("Measurements", exc)
            return
        StereologyWindow(self, images=snapshot.working_images, labels=snapshot.labels, scales=snapshot.scales)

    def _update_shape_minimum(self) -> None:
        if not self._busy:
            self.shapeModeVar.set("updated_minimum")
            self._start_job("run")

    def _start_job(self, action: str) -> None:
        if self._busy:
            return
        if self._history_loading:
            self._view_generation += 1
            self._history_loading = False
            if self._history_previous is not None:
                self._restore_view(self._history_previous)
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
            if action != "run" and mode != "Compare methods" and (legacy_error or modern_error):
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
        request["captured"] = CapturedAnalysisRequest(uuid.uuid4().hex, self._project_session or self._workspace_session,
            mode, action, snapshot, request["inputs"], request["settings"],
            tuple((method, self._method_settings_signature(method)) for method in ("legacy", "modern")),
            legacy_cfg, modern_cfg)
        self._active_request = request
        self._set_busy(True)
        self.statusVar.set(f"{action.capitalize()}: {mode}...")
        key = self._legacy_prepare_key(snapshot, legacy_cfg) if legacy_cfg is not None else None
        cached = self._legacy_prepare_cache.get(key)
        sample_name = self.project.sample_name
        source_arrays = tuple((image.image_id, image.source_image) for image in self.project.images)

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
                if action == "run":
                    try:
                        payload["run_projection"] = create_run_projection(request["captured"], payload, sample_name, source_arrays)
                        payload["run_data"] = payload["run_projection"].record.data()
                    except Exception as exc:
                        payload["projection_error"] = exc
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
            self._diagnostic_inputs = self._input_signature()
            return
        if error is not None:
            self._show_failure(request["mode"], error)
            return

        projection = result.get("run_projection")
        if projection is not None:
            append_run(self.project, projection)
            self._projection_views[projection.record.run_id] = result["run_data"]
            self._update_result_choices()
            if self._mark_unsaved is not None:
                self._mark_unsaved()
        if result.get("projection_error") is not None:
            self._show_failure("Run projection", result["projection_error"])

        previous_view = self._capture_view()
        try:
            captured = request["captured"]
            envelopes = tuple(MethodResultEnvelope(method, captured, result[method])
                              for method in ("legacy", "modern") if result[method] is not None)
            new_view = None
            context = result["legacy_prepared"]
            if context is not None:
                self._legacy_prepare_cache[result["key"]] = context
                self._prepared_legacy = context
            self._prepared_modern = result["modern_prepared"]
            if request["action"] in ("prepare", "suggest"):
                new_view = ("prepared", (context, self._prepared_modern), captured, ())
                self._present_result(*new_view)
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
                new_view = ("compare", comparison, captured, envelopes)
                self._present_result(*new_view)
                self.statusVar.set("Compare partial failure" if result["legacy_error"] or result["modern_error"] else "Compare run complete")
            elif result["legacy"] is not None:
                new_view = ("legacy", result["legacy"], captured, envelopes)
                self._present_result(*new_view)
                self.statusVar.set("Original run complete")
            elif result["modern"] is not None:
                new_view = ("modern", result["modern"], captured, envelopes)
                self._present_result(*new_view)
                self.statusVar.set("Modern run complete")
            elif result["modern_prepared"] is not None or context is not None:
                if self._last_completed_snapshot is None:
                    new_view = ("prepared", (context, result["modern_prepared"]), captured, ())
                    self._present_result(*new_view)
            if new_view is None and projection is not None:
                new_view = ("failed", None, captured, ())
                self._present_result(*new_view)
            if new_view is not None:
                self._current_view = new_view
                self.resultViewVar.set("Current request")
                for envelope in envelopes:
                    label = f"Previous successful result: {envelope.method} / {captured.request_id[:12]}"
                    self._successful_results[label] = envelope
                self._update_result_choices()
                self._refresh_result_provenance()
                if projection is not None:
                    data = result["run_data"]
                    self._populate_saved_supplement(data, data["tables"])
                    self.resultsNotebook.select(self._overview_frame)
                if request["mode"] == "Compare methods" and request["action"] not in ("prepare", "suggest"):
                    self._latest_compare = comparison
                if result["legacy"] is not None:
                    self._latest_legacy = result["legacy"]
                if result["modern"] is not None:
                    self._latest_modern = result["modern"]
            if result["legacy"] is not None or result["modern"] is not None:
                self._last_completed_snapshot = snapshot
                self._completed_inputs = request["inputs"]
                self._completed_signature = request["settings"]
                self._completed_request = request
            for method in ("legacy", "modern"):
                if result[f"{method}_error"] is not None:
                    self._show_failure(method, result[f"{method}_error"])
            if result.get("projection_error") is not None:
                self._show_failure("Run projection", result["projection_error"])
            self._refresh_summary_banner()
        except Exception as exc:
            self._restore_view(previous_view)
            if projection is not None:
                self.project.analysis_presentation_errors.setdefault(projection.record.run_id, []).append(
                    error_diagnostic("presentation", exc, "rendering"))
            self._show_failure("Rendering", exc)

    def _suggest_cutoffs(self) -> None:
        self._start_job("suggest")

    def _load_saved_run(self, record):
        self._history_previous = self._capture_view()
        self._view_generation += 1
        generation = self._view_generation
        self._history_loading = True
        self.cancelButton.configure(state="normal")
        self.statusVar.set(f"Loading saved run {record.run_id[:8]}")

        def worker():
            try:
                self._history_queue.put((generation, record, record.data(), run_tables(record), None))
            except Exception as exc:
                self._history_queue.put((generation, record, None, None, exc))
        threading.Thread(target=worker, daemon=True).start()
        self._schedule_after(20, self._poll_history)

    def _poll_history(self):
        if not self._history_loading:
            return
        try:
            generation, record, data, tables, error = self._history_queue.get_nowait()
        except queue.Empty:
            self._schedule_after(20, self._poll_history)
            return
        if generation != self._view_generation:
            self._schedule_after(20, self._poll_history)
            return
        self._history_loading = False
        self.cancelButton.configure(state="disabled")
        if error is not None:
            self._show_failure("History load", error)
            return
        previous = self._capture_view()
        try:
            self._clear_distribution()
            details = {key: value for key, value in data.items() if key not in ("tables", "methods")}
            details["methods"] = [{key: value for key, value in method.items() if key != "native"} for method in data["methods"]]
            self._set_results([json.dumps(details, ensure_ascii=False, indent=2)])
            self._populate_saved_supplement(data, tables)
            self._selected_saved_run = record
            self._saved_view_data = data
            self._displayed_request = None
            self._displayed_envelopes = ()
            self._selected_view_label = self.resultViewVar.get()
            self._filter_saved_tables()
            self._plot_saved_rows()
            self._refresh_result_provenance()
            self.statusVar.set(f"Saved run {record.run_id[:8]} loaded; no science rerun")
        except Exception as exc:
            self._restore_view(previous)
            self._show_failure("History rendering", exc)

    def _populate_saved_supplement(self, data, tables):
        self._table_run_id = data["run_id"]
        self.overviewText.configure(state="normal")
        self.overviewText.delete("1.0", "end")
        lines = [data["sample_name"], data["completed_at_utc"], f"{len(data['images'])} captured images", ""]
        for method in data["methods"]:
            lines.append(f"{method['method_id']}: {method['outcome']}")
            if method["native"] is not None:
                native = method["native"]
                lines.append(f"Groups: {len(native['groups'])}; measured objects: {len(native['objects'])}")
                if method["method_id"] == "legacy":
                    lines.append(f"Raw selected objects: {native['raw_selection']['selected_count']}")
                else:
                    lines.append(f"Selected bin counts: {native['selected']['counts']}")
            lines.extend(f"{diagnostic['stage']}: {diagnostic['message']}" for diagnostic in method["diagnostics"])
        lines.extend(["", "Native summaries:"])
        for diagnostic in self.project.analysis_presentation_errors.get(data["run_id"], ()):
            lines.append(f"Presentation error: {diagnostic['message']}")
        lines.extend(f"{row['method_id']} {row['metric_id']}: {row['value']} {row['unit']}"
                     for row in tables["summary_metrics"][:100])
        self.overviewText.insert("1.0", "\n".join(lines))
        self.overviewText.configure(state="disabled")
        self._distribution_rows = tables["distributions"]
        self._object_rows = tables["objects"]
        self._object_page = self._distribution_page = 0
        self.tableFilters[1].configure(values=("All", *sorted({row["group_id"] for row in tables["groups"]})))
        self._page_objects(0)

    @staticmethod
    def _display_cell(value):
        if value is None:
            return ""
        if isinstance(value, dict):
            return "unavailable: " + value.get("reason", "")
        return value

    def _filter_saved_tables(self):
        if self._table_run_id is None:
            return
        self._clear_distribution()
        filtered = [row for row in self._distribution_rows if
                    (self.tableMethodVar.get() == "All" or row["method_id"] == self.tableMethodVar.get()) and
                    (self.tableGroupVar.get() == "All" or row["group_id"] == self.tableGroupVar.get()) and
                    (self.tableStageVar.get() == "All" or row["stage"] == self.tableStageVar.get())]
        start = self._distribution_page * 250
        for row in filtered[start:start + 250]:
            method = "Original" if row["method_id"] == "legacy" else "Modern"
            lower = row["label_mm"] if row["label_mm"] is not None else row["lower_mm"]
            upper = row["label_mm"] if row["label_mm"] is not None else row["upper_mm"]
            self._add_row(method + " " + row["stage"], row["group_id"], row["bin_index"], lower, upper,
                self._display_cell(row["count"]), self._display_cell(row["area_mm2"]), self._display_cell(row["na_per_mm2"]),
                self._display_cell(row["signed_nv_per_mm3"]), self._display_cell(row["volume_fraction"]),
                kind=row["method_id"] if row["stage"] == "prepared" else "selected")
        self._plot_saved_rows()

    def _plot_saved_rows(self):
        self.plotAxes.clear()
        self._nonpositive_plot_count = 0
        series = {}
        for row in self._distribution_rows:
            key = (row["method_id"], row["group_id"], row["stage"])
            diameter = row["plot_diameter_mm"]
            value = row["na_per_mm2"]
            if isinstance(value, (int, float)):
                series.setdefault(key, []).append((diameter, value))
            else:
                self._nonpositive_plot_count += 1
        for key, pairs in series.items():
            self._plot_values(" ".join(key), tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs))
        self._finish_plot()

    def _page_distributions(self, offset):
        self._distribution_page = max(0, min(max(0, (len(self._distribution_rows) - 1) // 250), self._distribution_page + offset))
        self._filter_saved_tables()

    def _page_objects(self, offset):
        self._object_page = max(0, min(max(0, (len(self._object_rows) - 1) // 250), self._object_page + offset))
        self.objectsTable.delete(*self.objectsTable.get_children())
        self._object_row_by_id = {}
        for row in self._object_rows[self._object_page * 250:(self._object_page + 1) * 250]:
            row_id = self.objectsTable.insert("", "end", values=tuple(self._display_cell(value) for value in (
                row["method_id"], row["image_id"], row["object_id"], row.get("equivalent_diameter_mm", row.get("eq_diam_units")),
                row.get("area_mm2", row.get("area_units2")), row.get("selected"), row.get("shape_population"))))
            self._object_row_by_id[row_id] = row
        self.objectPageLabel.configure(text=f"Page {self._object_page + 1} | {len(self._object_rows)} native objects")

    def _inspect_object(self, _event=None):
        selected = self.objectsTable.selection()
        if selected:
            row = self._object_row_by_id.get(selected[0])
            if row is not None:
                self._append_details([json.dumps(row, ensure_ascii=False, indent=2)])

    def _inspect_shape_statistics(self):
        data = self._saved_view_data or self._projection_views.get(self._table_run_id)
        if data is not None:
            values = {method["method_id"]: method["native"].get("shape_statistics", {"availability": "unavailable", "reason": "not supplied by this method"})
                      for method in data["methods"] if method["native"] is not None}
            self._append_details([json.dumps(values, ensure_ascii=False, indent=2)])
            self.resultsNotebook.select(self.resultsText.master)

    def _sort_object_page(self, column):
        rows = list(self.objectsTable.get_children())
        def key(row_id):
            value = self.objectsTable.set(row_id, column)
            try:
                return (0, float(value))
            except ValueError:
                return (1, value)
        for index, row_id in enumerate(sorted(rows, key=key)):
            self.objectsTable.move(row_id, "", index)

    def _use_run_settings(self):
        data = self._saved_view_data
        if data is None and self._displayed_request is not None:
            data = self._projection_views.get(self._displayed_request.request_id)
        if data is None:
            return
        for name, value in data["draft_settings"].items():
            if name in self._variables:
                self._variables[name].set(value)
        legacy = next((method for method in data["methods"] if method["method_id"] == "legacy"), None)
        if legacy is not None and legacy["settings"] is not None:
            self._source_keys = tuple(tuple(item) for item in legacy["settings"].get("source_keys", ()))
        self.statusVar.set("Saved run settings copied to editable setup; accepted inputs unchanged")

    def _export_run(self):
        if self._exporting:
            return
        record = self._selected_saved_run
        if record is None and self._displayed_request is not None:
            record = next((run for run in self.project.analysis_runs if run.run_id == self._displayed_request.request_id), None)
        if record is None:
            self._show_failure("Export", ValueError("Select a completed saved run"))
            return
        parent = filedialog.askdirectory(parent=self, title=f"Export run {record.run_id[:8]} into a fresh directory")
        if not parent:
            return
        self._export_cancel = threading.Event()
        cancellation = self._export_cancel
        self._exporting = True
        self.exportRunButton.configure(state="disabled")
        self.cancelButton.configure(state="normal")
        self.statusVar.set(f"Exporting selected run {record.run_id[:8]}")
        errors = tuple(self.project.analysis_presentation_errors.get(record.run_id, ()))

        def worker():
            try:
                destination = export_selected_run(record, self.project.analysis_assets,
                    Path(parent) / ("run-" + record.run_id), cancellation, errors)
                self._export_queue.put((record, destination, None))
            except Exception as exc:
                self._export_queue.put((record, None, exc))
        threading.Thread(target=worker, daemon=True).start()
        self._schedule_after(20, self._poll_export)

    def _poll_export(self):
        try:
            record, destination, error = self._export_queue.get_nowait()
        except queue.Empty:
            self._schedule_after(20, self._poll_export)
            return
        self._exporting = False
        self.exportRunButton.configure(state="disabled" if self._busy else "normal")
        self.cancelButton.configure(state="disabled")
        if error is not None:
            self._show_failure("Export", error)
        else:
            self.statusVar.set(f"Export complete: selected run {record.run_id[:8]} | {destination}")

    def _append_details(self, lines) -> None:
        self.resultsText.configure(state="normal")
        self.resultsText.insert("end", "\n" + "\n".join(lines))
        self.resultsText.configure(state="disabled")

    def _show_failure(self, method, error) -> None:
        self.statusVar.set(f"{method} failed; previous results retained")
        self._diagnostic_inputs = self._input_signature()
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
                [(line.get_xdata(), line.get_ydata(), line.get_label()) for line in self.plotAxes.lines],
                self._displayed_request, self._displayed_envelopes,
                self.resultProvenanceVar.get(), getattr(self, "_selected_view_label", "Current request"), self._displayed_outcomes,
                (self._selected_saved_run, self._saved_view_data, self._distribution_rows, self._object_rows, self._table_run_id,
                 self.overviewText.get("1.0", "end"), self._object_page, self._distribution_page))

    def _restore_view(self, view) -> None:
        text, rows, curves, request, envelopes, provenance, label, outcomes, saved = view
        self._clear_distribution()
        self._set_results([text])
        for values, metadata in rows:
            row_id = self.distributionTable.insert("", "end", values=values)
            self._table_rows[row_id] = metadata
        for diameters, values, name in curves:
            self._plot_values(name, diameters, values)
        self._finish_plot()
        self._displayed_request = request
        self._displayed_envelopes = envelopes
        self.resultProvenanceVar.set(provenance)
        self.resultViewVar.set(label)
        self._selected_view_label = label
        self._displayed_outcomes = outcomes
        (self._selected_saved_run, self._saved_view_data, self._distribution_rows, self._object_rows,
         self._table_run_id, overview, self._object_page, self._distribution_page) = saved
        self.overviewText.configure(state="normal")
        self.overviewText.delete("1.0", "end")
        self.overviewText.insert("1.0", overview)
        self.overviewText.configure(state="disabled")
        self._page_objects(0)

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
        self._plot_values("Modern selected", tuple(distribution.bin_spec.bins[index].geometric_midpoint_mm for index in nested.selected_bin_indices),
                          nested.number_densities_per_mm2)
        self._finish_plot()

    def _render_compare(self, result: CompareRunResult) -> None:
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

        self._set_results(lines)
        self._render_append = True
        try:
            if result.legacy_result is not None:
                self._render_legacy(result.legacy_result)
            if result.modern_result is not None:
                self._render_modern(result.modern_result)
        finally:
            self._render_append = False
        self._finish_plot()

    def _set_results(self, lines) -> None:
        self.resultsText.configure(state="normal")
        self.resultsText.delete("1.0", "end")
        self.resultsText.insert("1.0", "\n".join(lines))
        self.resultsText.configure(state="disabled")

    def _on_close(self) -> None:
        self._view_generation += 1
        self._export_cancel.set()
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
