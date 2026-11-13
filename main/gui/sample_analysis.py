from __future__ import annotations

import json
import math
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable, Optional

from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, NullLocator

from ..core.analysis_plots import NA_PLOT, NV_PLOT, reconstruction_series, visible_points
from ..core.analysis_runs import append_run, create_run_projection, error_diagnostic, json_bytes
from ..core.legacy_cutoffs import Foams105CutoffGroupInput, suggest_foams105_cutoffs
from ..core.legacy_perimeter import (
    FOAMS105_PERIMETER_PROFILE_CURRENT,
    FOAMS105_PERIMETER_PROFILE_OLD,
)
from ..core.reconstruction import UPPER_TAIL_ASSUMPTION
from ..core.run_exports import TABLE_PROJECTION_SCHEMA_VERSION, export_selected_run, run_tables
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
    FOAMS105_PERIMETER_PROFILE_CURRENT: "MATLAB Perimeter \u2014 verified against R2025b",
    FOAMS105_PERIMETER_PROFILE_OLD: "MATLAB PerimeterOld \u2014 verified against R2025b",
}
_PERIMETER_LABEL_TO_ID.update({label: profile for profile, label in _PERIMETER_ID_TO_LABEL.items()})


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
        self._history_request = None
        self._plotted_metadata = {}
        self._plot_data = None
        self._graph_refreshing = False

        self.modeVar = tk.StringVar(value="Original FOAMS 1.0.5")
        self.minimumDiameterVar = tk.StringVar(value="2.0")
        self.perimeterProfileVar = tk.StringVar(value=_PERIMETER_ID_TO_LABEL[FOAMS105_PERIMETER_PROFILE_CURRENT])
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
        self.plotTypeVar = tk.StringVar(value=NA_PLOT)
        self.plotMethodVar = tk.StringVar(value="All methods")
        self.plotStageVar = tk.StringVar(value="All series")
        self.plotScaleVar = tk.StringVar(value="Log Y")
        self.plotXAxisVar = tk.StringVar(value="Log diameter")

        self._build_ui()
        self._load_persisted_setup()
        self._set_results([f"Sample: {self.project.sample_name}", "Included accepted binary-mask inputs:"]
            + [f"{image.display_name}: {image.image_id} snapshot={image.snapshot_id} calibration={image.mm_per_px!r} mm/px"
                    for image in self.snapshot.images] if self.snapshot is not None else
                [f"Sample: {self.project.sample_name}", "History view; no current analysis snapshot captured"])
        self._variables = {
            name: value for name, value in vars(self).items() if isinstance(value, tk.Variable)
            and name not in ("statusVar", "resultViewVar", "resultProvenanceVar", "tableMethodVar", "tableGroupVar", "tableStageVar")
            and not name.startswith("plot")
        }
        for variable in self._variables.values():
            variable.trace_add("write", self._settings_changed)
        self.manualRangesVar.trace_add("write", self._manual_changed)
        for variable in (self.modeVar, self.selectionModeVar, self.volumeModeVar, self.shapeModeVar, self.modernReconstructVar):
            variable.trace_add("write", self._settings_visibility)
        self._settings_visibility()
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
        root = ttk.Frame(self, padding=(10, 4))
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

        self._build_method_settings(controls)

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
        self._build_plot_controls(plot_frame)
        self.plotAxes = self.figure.add_subplot(111)
        self.plotCanvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.plotCanvas.get_tk_widget().pack(fill="both", expand=True)
        self.plotNavigation = NavigationToolbar2(self.plotCanvas)
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

    def _build_method_settings(self, controls):
        ttk.Label(controls, text="Method preset").pack(anchor="w")
        self.methodPresetSelector = ttk.Combobox(controls, textvariable=self.modeVar,
            values=("Original FOAMS 1.0.5", "Modern analysis", "Compare methods"), state="readonly", width=30)
        self.methodPresetSelector.pack(anchor="w", pady=(0, 8))
        self.originalSettingsToggle = ttk.Button(controls, text="Original settings", command=lambda: self._toggle_method_section("legacy"))
        self.originalSettingsToggle.pack(anchor="w")
        self.originalSettings = ttk.Frame(controls, padding=(0, 4))
        self.originalSettings.pack(fill="x")
        self.modernSettingsToggle = ttk.Button(controls, text="Modern settings", command=lambda: self._toggle_method_section("modern"))
        self.modernSettingsToggle.pack(anchor="w")
        self.modernSettings = ttk.Frame(controls, padding=(0, 4))
        self.modernSettings.pack(fill="x")
        self._method_expanded = {"legacy": True, "modern": True}
        self._advanced_original_open = False
        self.settingRows = {}
        self.settingInputs = {}
        self._modern_feedback = {}
        self._method_rows = {"legacy": [], "modern": []}

        def field(parent, key, label, variable, values=None, method="legacy"):
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label).pack(anchor="w")
            widget = ttk.Combobox(row, textvariable=variable, values=values, state="readonly", width=46) if values is not None else ttk.Entry(row, textvariable=variable, width=46)
            widget.pack(anchor="w", fill="x")
            if method == "modern":
                self._modern_feedback[key] = ttk.Label(row, foreground="#a32929", wraplength=660, justify="left")
            self.settingRows[key] = row
            self.settingInputs[key] = widget
            self._method_rows[method].append((key, row))
            return row

        field(self.originalSettings, "minimum", "Minimum object diameter (pixels)", self.minimumDiameterVar)
        field(self.originalSettings, "selection", "Selection", self.selectionModeVar, ("autosmart", "manual"))
        field(self.originalSettings, "manual", "Manual group:lower:upper labels (mm)", self.manualRangesVar)
        field(self.originalSettings, "normalization", "Normalization mode", self.volumeModeVar, ("unnormalized", "integrated_2d", "known_vesicularity"))
        field(self.originalSettings, "vesicularity", "Known vesicularity (%)", self.knownVesicularityVar)
        field(self.originalSettings, "category_lower", "Lower size-category boundary (mm)", self.sizeLowerCutoffVar)
        field(self.originalSettings, "category_upper", "Upper size-category boundary (mm)", self.sizeUpperCutoffVar)
        field(self.originalSettings, "shape_mode", "Shape mode", self.shapeModeVar, ("initial_analysis", "updated_minimum"))
        field(self.originalSettings, "shape_minimum", "Updated shape minimum (pixels)", self.shapeMinimumVar)
        self.advancedOriginalButton = ttk.Button(self.originalSettings, text="Advanced Original settings", command=self._toggle_advanced_original)
        self.advancedOriginalButton.pack(anchor="w", pady=4)
        advanced = ttk.Frame(self.originalSettings)
        advanced.pack(fill="x")
        self.advancedOriginalSettings = advanced
        field(advanced, "perimeter", "Perimeter calculation", self.perimeterProfileVar, tuple(_PERIMETER_ID_TO_LABEL.values()))
        self.perimeterHelp = ttk.Label(advanced, justify="left", wraplength=660,
            text="R2025b is the MATLAB release used to verify both perimeter estimators, not image resolution. No MATLAB installation is required. Estimates affect perimeter-dependent shape measures; PerimeterOld is not a separate old-runtime capture.")
        self.perimeterHelp.pack(fill="x", pady=4)
        field(self.modernSettings, "detection", "Minimum detectable diameter (mm)", self.modernMinDetectableVar, method="modern")
        self.detectionHelp = ttk.Label(self.modernSettings, text="Minimum pore diameter reliably detectable at this image resolution, in mm.", wraplength=660, justify="left")
        self.detectionHelp.pack(fill="x")
        self.modernValidationLabel = self._modern_feedback["detection"]
        field(self.modernSettings, "reliability", "Maximum reliable diameter (mm, optional)", self.modernMaxReliableVar, method="modern")
        field(self.modernSettings, "edges", "Bin edges (mm)", self.modernEdgesVar, method="modern")
        field(self.modernSettings, "segments", "Source-group plan: group:start:stop bins", self.modernSegmentsVar, method="modern")
        self.singleGroupHelp = ttk.Label(self.modernSettings, text="Single resolution group: all bins use that group's automatic ownership.", wraplength=660)
        self.singleGroupHelp.pack(fill="x")
        self.modernReconstructionCheck = ttk.Checkbutton(self.modernSettings, text="Enable spherical 3D reconstruction", variable=self.modernReconstructVar)
        self.modernReconstructionCheck.pack(anchor="w", pady=4)
        self.modernUpperTailCheck = ttk.Checkbutton(self.modernSettings, text="Assume no larger 3D pores above the selected upper edge", variable=self.modernUpperTailVar)
        self.modernUpperTailCheck.pack(anchor="w")
        self._modern_feedback["reconstruction"] = ttk.Label(self.modernSettings, foreground="#a32929", wraplength=660, justify="left")

    def _toggle_method_section(self, method):
        self._method_expanded[method] = not self._method_expanded[method]
        self._settings_visibility()

    def _toggle_advanced_original(self):
        self._advanced_original_open = not self._advanced_original_open
        self._settings_visibility()

    def _settings_visibility(self, *_args):
        mode = self.modeVar.get()
        original = mode in ("Original FOAMS 1.0.5", "Compare methods")
        modern = mode in ("Modern analysis", "Compare methods")
        for widget in (self.originalSettings, self.modernSettings, self.originalSettingsToggle, self.modernSettingsToggle):
            widget.pack_forget()
        if original:
            if mode == "Compare methods":
                self.originalSettingsToggle.pack(fill="x")
            if mode != "Compare methods" or self._method_expanded["legacy"]:
                self.originalSettings.pack(fill="x")
        if modern:
            if mode == "Compare methods":
                self.modernSettingsToggle.pack(fill="x")
            if mode != "Compare methods" or self._method_expanded["modern"]:
                self.modernSettings.pack(fill="x")
        for method, rows in self._method_rows.items():
            for _key, row in rows:
                row.pack_forget()
            for key, row in rows:
                show = key != "manual" or self.selectionModeVar.get() == "manual"
                show = show and (key != "vesicularity" or self.volumeModeVar.get() == "known_vesicularity")
                show = show and (key != "shape_minimum" or self.shapeModeVar.get() == "updated_minimum")
                scales = {calibration_mm_per_px(image) for image in self.project.images if image.analysis_included and calibration_mm_per_px(image) is not None}
                show = show and (key != "segments" or len(scales) > 1)
                if show:
                    row.pack(fill="x", pady=3)
        self.advancedOriginalSettings.pack_forget()
        if self._advanced_original_open:
            self.advancedOriginalSettings.pack(fill="x")
        self.advancedOriginalButton.pack_forget()
        self.advancedOriginalButton.pack(anchor="w", pady=4, before=self.advancedOriginalSettings if self._advanced_original_open else None)
        for widget in (self.detectionHelp, self.modernReconstructionCheck):
            widget.pack_forget()
        self.detectionHelp.pack(fill="x", before=self.settingRows["reliability"])
        self.modernReconstructionCheck.pack(anchor="w", pady=4)
        self.modernUpperTailCheck.pack_forget()
        if self.modernReconstructVar.get():
            self.modernUpperTailCheck.pack(anchor="w")
        feedback = self._modern_feedback["reconstruction"]
        feedback.pack_forget()
        if self.modernReconstructVar.get() and feedback.cget("text"):
            feedback.pack(fill="x")
        self.singleGroupHelp.pack_forget()
        if len({calibration_mm_per_px(image) for image in self.project.images if image.analysis_included}) == 1:
            self.singleGroupHelp.pack(fill="x")
        self.suggestButton.pack_forget()
        self.shapeUpdateButton.pack_forget()
        if original:
            self.suggestButton.pack(side="left", padx=8)
            self.shapeUpdateButton.pack(side="left", padx=8)
        self._queue_layout()

    def _build_plot_controls(self, parent):
        viewport = ttk.Frame(parent)
        viewport.pack(fill="x")
        self.plotToolbar = viewport
        canvas = tk.Canvas(viewport, height=60, highlightthickness=0)
        canvas.pack(fill="x")
        horizontal = ttk.Scrollbar(viewport, orient="horizontal", command=canvas.xview)
        horizontal.pack(fill="x")
        canvas.configure(xscrollcommand=horizontal.set)
        toolbar = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=toolbar, anchor="nw")
        self._plot_toolbar_canvas = canvas
        self._plot_toolbar_content = toolbar
        self._plot_controls = []
        for label, variable, values, width in (
            ("Plot type", self.plotTypeVar, (NA_PLOT, NV_PLOT), 52),
            ("Method", self.plotMethodVar, ("All methods", "Original", "Modern"), 14),
            ("Series", self.plotStageVar, ("All series", "Prepared", "Selected"), 12),
            ("Y axis", self.plotScaleVar, ("Linear Y", "Log Y"), 10),
            ("Diameter", self.plotXAxisVar, ("Log diameter", "Linear diameter"), 16)):
            group = ttk.Frame(toolbar)
            ttk.Label(group, text=label).pack(anchor="w")
            selector = ttk.Combobox(group, textvariable=variable, values=values, state="readonly", width=width)
            selector.pack(fill="x")
            selector.bind("<<ComboboxSelected>>", lambda event, field=label: self._graph_option_changed(field))
            self._plot_controls.append(group)
            if label == "Plot type":
                self.plotTypeSelector = selector
            elif label == "Series":
                self.plotSeriesSelector = selector
        actions = ttk.Frame(toolbar)
        self._plot_controls.append(actions)
        self.panGraphButton = ttk.Button(actions, text="Pan", command=lambda: self.plotNavigation.pan())
        self.panGraphButton.pack(side="left")
        self.zoomGraphButton = ttk.Button(actions, text="Zoom", command=lambda: self.plotNavigation.zoom())
        self.zoomGraphButton.pack(side="left")
        self.resetGraphButton = ttk.Button(actions, text="Reset view", command=self._refresh_graph)
        self.resetGraphButton.pack(side="left")
        self.saveGraphButton = ttk.Button(actions, text="Save graph", command=self._save_graph)
        self.saveGraphButton.pack(side="left")
        self.signedGraphButton = ttk.Button(actions, text="Show signed values", command=self._show_signed_graph)
        self.signedGraphButton.pack(side="left")
        self.graphSettingsButton = ttk.Button(actions, text="Draft reconstruction settings", command=self._show_graph_settings)
        self.graphSettingsButton.pack(side="left")
        self.graphSettingsButton.pack_forget()
        self.signedGraphButton.pack_forget()
        self.graphInfoLabel = ttk.Label(toolbar, justify="left")
        self.graphInfoLabel.grid(row=1, column=0, columnspan=6, sticky="w")
        for column, group in enumerate(self._plot_controls):
            group.grid(row=0, column=column, sticky="sw", padx=3)
            pending = [group]
            while pending:
                widget = pending.pop()
                pending.extend(widget.winfo_children())
                widget.bind("<FocusIn>", lambda event, target=widget: self._reveal_control(target, canvas, toolbar), add="+")
        toolbar.bind("<Configure>", lambda event: (canvas.configure(scrollregion=canvas.bbox("all"), height=toolbar.winfo_reqheight()), self._queue_layout()))
        self.inverseGraphButton = ttk.Button(self._secondary_actions.children['!canvas'].children['!frame'],
            text="View inverse reconstruction", command=self._open_inverse_graph)
        self.inverseGraphButton.pack(side="left", padx=8)

    def _graph_option_changed(self, field):
        self._invalidate_pending_graph_history()
        if field == "Plot type":
            self.plotScaleVar.set("Linear Y" if self.plotTypeVar.get() == NV_PLOT else "Log Y")
        self._refresh_graph()

    def _show_signed_graph(self):
        self.plotScaleVar.set("Linear Y")
        self._invalidate_pending_graph_history()
        self._refresh_graph()

    def _open_inverse_graph(self):
        self.plotTypeVar.set(NV_PLOT)
        self.plotScaleVar.set("Linear Y")
        self._invalidate_pending_graph_history()
        self.resultsNotebook.select(self._plot_frame)
        self._refresh_graph()

    def _invalidate_pending_graph_history(self):
        if self._history_loading:
            variables = (self.plotTypeVar, self.plotMethodVar, self.plotStageVar, self.plotScaleVar, self.plotXAxisVar)
            values = tuple(variable.get() for variable in variables)
            self._invalidate_history()
            if self._history_previous is not None:
                self._restore_view(self._history_previous)
            for variable, value in zip(variables, values):
                variable.set(value)

    def _show_graph_settings(self):
        self._set_setup_visible(True)
        self.modeVar.set("Modern analysis" if self.modeVar.get() != "Compare methods" else "Compare methods")
        self._method_expanded["modern"] = True
        self._settings_visibility()
        self._reveal_control(self.modernReconstructionCheck, self._setup_canvas, self.setupControls)

    def _graph_source(self):
        if self._selected_saved_run is not None:
            return self._saved_view_data
        if self._displayed_request is not None:
            return self._projection_views.get(self._displayed_request.request_id)
        return None

    def _refresh_graph(self):
        if self._graph_refreshing:
            return
        self._graph_refreshing = True
        try:
            data = self._graph_source()
            self._plot_data = data
            if self.plotTypeVar.get() == NV_PLOT:
                self._render_inverse_graph(data)
            elif data is not None:
                self._plot_saved_rows(data=data)
            else:
                self._plotted_metadata = {}
                self._sync_graph_controls()
                self.graphInfoLabel.configure(text="Prepare or select a completed run to inspect its distribution")
        finally:
            self._graph_refreshing = False

    def _sync_graph_controls(self, messages=()):
        inverse = self.plotTypeVar.get() == NV_PLOT
        self.plotSeriesSelector.configure(state="disabled" if inverse else "readonly")
        self.graphSettingsButton.pack_forget()
        self.signedGraphButton.pack_forget()
        if inverse and any("was not calculated" in message for message in messages):
            self.graphSettingsButton.pack(side="left")
        if inverse and self.plotScaleVar.get() == "Log Y":
            self.signedGraphButton.pack(side="left")

    def _render_inverse_graph(self, data):
        self.plotAxes.clear()
        self._nonpositive_plot_count = 0
        if data is None:
            self._sync_graph_controls()
            self.graphInfoLabel.configure(text="No computed reconstruction is selected")
            self._plotted_metadata = {}
            self.plotCanvas.draw_idle()
            return
        tables = data.get("tables")
        if self._selected_saved_run is not None:
            tables = {"distributions": self._distribution_rows}
        series, messages = reconstruction_series(data, tables)
        log_y = self.plotScaleVar.get() == "Log Y"
        log_x = self.plotXAxisVar.get() == "Log diameter"
        self._sync_graph_controls(messages)
        plotted = []
        all_omitted = {"invalid_diameter": 0, "unavailable": 0, "zero": 0, "negative": 0}
        for item in series:
            if self.plotMethodVar.get() != "All methods" and item["method_id"] != ("legacy" if self.plotMethodVar.get() == "Original" else "modern"):
                continue
            points, omitted = visible_points(item["points"], log_x=log_x, log_y=log_y)
            for key, count in omitted.items():
                all_omitted[key] += count
            self.plotAxes.plot([point["diameter_mm"] for point in points], [point["signed_nv_per_mm3"] for point in points],
                               linestyle="none", marker="o" if item["method_id"] == "legacy" else "s", markersize=4,
                               label=item["title"])
            negatives = [point for point in points if point["signed_nv_per_mm3"] < 0]
            if negatives:
                self.plotAxes.plot([point["diameter_mm"] for point in negatives], [point["signed_nv_per_mm3"] for point in negatives],
                                   color="#bb2638", marker="x", linestyle="none", label=("Original" if item["method_id"] == "legacy" else "Modern") + " signed negative values")
            plotted.append({**item, "omitted": omitted})
        if not log_y:
            self.plotAxes.axhline(0, color="#555555", linewidth=0.8, label="Zero")
        self.plotAxes.set_xscale("log" if log_x else "linear")
        if log_y:
            self.plotAxes.set_yscale("log")
            if not any(len(line.get_ydata()) for line in self.plotAxes.lines):
                self.plotAxes.set_ylim(1, 10)
        else:
            self.plotAxes.set_yscale("linear")
        self.plotAxes.set_xlabel(plotted[0]["coordinates"] if len(plotted) == 1 else "Diameter (mm): Original labels / Modern native upper-edge class representatives")
        self.plotAxes.set_ylabel("N_V (mm^-3 per native class)")
        title = plotted[0]["title"] if len(plotted) == 1 else "Separate native inverse reconstructions"
        self.plotAxes.set_title(title, fontsize=10)
        if self.plotAxes.lines:
            self.plotAxes.legend(fontsize=8)
        info = messages + [f"Omitted: zero={all_omitted['zero']}, negative={all_omitted['negative']}, unavailable={all_omitted['unavailable']}, invalid diameter={all_omitted['invalid_diameter']}"]
        self.graphInfoLabel.configure(text=" | ".join(info))
        self._plotted_metadata = {"run_id": data["run_id"], "project_id": data["project_id"], "sample_name": data["sample_name"],
            "plot_type": NV_PLOT, "x_scale": self.plotAxes.get_xscale(), "y_scale": self.plotAxes.get_yscale(),
            "omitted": all_omitted, "series": plotted, "messages": messages,
            "caption": f"{data['sample_name'][:70]} | run {data['run_id'][:12]} | {title}"}
        self.plotCanvas.draw_idle()

    def _save_graph(self):
        if self._busy or self._history_loading:
            self.statusVar.set("Wait for the selected run to finish loading before saving its graph")
            return
        if not self._plotted_metadata.get("run_id"):
            self.statusVar.set("Select a completed run before saving a graph")
            return
        path = filedialog.asksaveasfilename(parent=self, title=f"Save graph for run {self._plotted_metadata['run_id'][:8]}",
            defaultextension=".png", filetypes=(("PNG", "*.png"), ("SVG", "*.svg")))
        if not path:
            return
        path = Path(path)
        try:
            if path.suffix.lower() not in (".png", ".svg"):
                raise ValueError("Graph output must be PNG or SVG")
            self._plotted_metadata["x_limits"] = list(self.plotAxes.get_xlim())
            self._plotted_metadata["y_limits"] = list(self.plotAxes.get_ylim())
            caption = self.figure.text(0.01, -0.025, self._plotted_metadata["caption"], fontsize=7)
            try:
                self.figure.savefig(path, dpi=180, bbox_inches="tight")
            finally:
                caption.remove()
            path.with_suffix(path.suffix + ".json").write_bytes(json_bytes(self._plotted_metadata))
            self.statusVar.set(f"Graph saved for run {self._plotted_metadata['run_id'][:8]}: {path.name}")
        except Exception as exc:
            self._show_failure("Save graph", exc)

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
        reserve = 240 + self.resultViewSelector.winfo_reqheight() + self.resultProvenanceLabel.winfo_reqheight() + line_height + 32 + self.plotToolbar.winfo_reqheight()
        self._layout_root.rowconfigure(5, minsize=reserve)
        if self._setup_visible:
            fixed = self._summary_frame.winfo_reqheight() + self._primary_actions.winfo_reqheight() + self._status_frame.winfo_reqheight() + self._secondary_actions.winfo_reqheight() + 40
            setup_overhead = max(0, self.setupBody.winfo_reqheight() - self._setup_canvas.winfo_reqheight())
            height = max(40, min(230, self._layout_root.winfo_height() - reserve - fixed - setup_overhead))
            if int(self._setup_canvas.cget("height")) != height:
                self._setup_canvas.configure(height=height)
        self._refresh_summary_banner()
        for label in (self.perimeterHelp, self.detectionHelp, self.singleGroupHelp, *self._modern_feedback.values()):
            if int(label.cget("wraplength") or 0) != width:
                label.configure(wraplength=width)

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
                profile = _PERIMETER_LABEL_TO_ID.get(self.perimeterProfileVar.get())
                if profile is not None:
                    self.perimeterProfileVar.set(_PERIMETER_ID_TO_LABEL[profile])
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
        known = float(known_text) if known_text and self.volumeModeVar.get() == "known_vesicularity" else None
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
            manual_ranges=self._parse_manual_ranges() if self.selectionModeVar.get() == "manual" else (),
            volume_normalization_mode=self.volumeModeVar.get(),
            known_vesicularity_percent=known,
            size_lower_cutoff_mm=lower_cutoff,
            size_upper_cutoff_mm=upper_cutoff,
            shape_minimum_diameter_px=shape_minimum,
            shape_statistics_mode=self.shapeModeVar.get(),
            source_keys=self._source_keys,
        )

    def _parse_modern_config(self) -> ModernRunConfig:
        def parse(text, converter, field):
            try:
                return converter(text)
            except (ValueError, IndexError) as exc:
                raise SampleAnalysisError(f"{field}: {exc}") from exc

        min_text = self.modernMinDetectableVar.get().strip()
        max_text = self.modernMaxReliableVar.get().strip()
        transition_text = self.modernTransitionBinsVar.get().strip()
        if transition_text:
            transitions = tuple(int(part.strip()) for part in transition_text.split(",") if part.strip())
        else:
            transitions = ()
        groups = {calibration_mm_per_px(image) for image in self.project.images if image.analysis_included}
        return ModernRunConfig(
            bin_edges_mm=parse(self.modernEdgesVar.get(), lambda text: tuple(float(part) for part in text.split(",")), "Bin edges") if self.modernEdgesVar.get().strip() else None,
            min_detectable_mm=parse(min_text, float, "Minimum detectable diameter") if min_text else None,
            max_reliable_mm=parse(max_text, float, "Maximum reliable diameter") if max_text else None,
            transition_bins=transitions,
            enable_reconstruction=bool(self.modernReconstructVar.get()),
            nesting_segments=parse(self.modernSegmentsVar.get(), lambda text: tuple(
                (parts[0].strip(), int(parts[1]), int(parts[2]))
                for parts in (token.split(":") for token in text.split(";"))
            ), "Source-group segment plan") if len(groups) > 1 and self.modernSegmentsVar.get().strip() else (),
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
        self._clear_supplement()
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
        self._refresh_graph()

    def _select_result_view(self, _event=None) -> None:
        view = self._capture_view()
        selected_label = self.resultViewVar.get()
        self._invalidate_history()
        if self._busy or self._exporting:
            self.resultViewVar.set(getattr(self, "_selected_view_label", "Current request"))
            return
        try:
            if selected_label in self._saved_labels:
                self._load_saved_run(self._saved_labels[selected_label])
            elif selected_label == "Current request":
                if self._current_view is None:
                    self._clear_supplement()
                    self._clear_distribution()
                    self._set_results(["No completed current request"])
                    self._finish_plot()
                    self._selected_saved_run = self._saved_view_data = self._displayed_request = None
                    self._displayed_envelopes = ()
                    self._selected_view_label = selected_label
                    self.resultProvenanceVar.set("No completed current request")
                    return
                self._present_result(*self._current_view)
                self.statusVar.set("Showing current request " + self._current_view[2].request_id[:8])
            else:
                envelope = self._successful_results[selected_label]
                self._present_result(envelope.method, envelope.result, envelope.request, (envelope,), previous=True)
                self.statusVar.set("Showing previous result " + envelope.request.request_id[:8])
        except Exception as exc:
            self._restore_view(view)
            self._show_failure("Rendering", exc)
        finally:
            self._sync_operation_controls()

    def _invalidate_history(self):
        self._view_generation += 1
        self._history_loading = False
        self._history_request = None
        self._sync_operation_controls()

    def _sync_operation_controls(self):
        self.cancelButton.configure(state="normal" if self._busy or self._history_loading or self._exporting else "disabled")
        self.resultViewSelector.configure(state="disabled" if self._busy or self._exporting else "readonly")

    def _clear_supplement(self):
        self._table_run_id = None
        self._distribution_rows = []
        self._object_rows = []
        self._object_row_by_id = {}
        self._object_page = self._distribution_page = 0
        self.overviewText.configure(state="normal")
        self.overviewText.delete("1.0", "end")
        self.overviewText.configure(state="disabled")
        self.objectsTable.delete(*self.objectsTable.get_children())
        self.objectPageLabel.configure(text="No completed run population selected")
        self.tableGroupVar.set("All")
        self.tableFilters[1].configure(values=("All",))

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
        self._sync_operation_controls()

    def _cancel_job(self) -> None:
        if self._history_loading and self._history_previous is not None:
            self._restore_view(self._history_previous)
        self._invalidate_history()
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
            if self.modeVar.get() not in ("Original FOAMS 1.0.5", "Compare methods"):
                self._show_failure("Shape update", ValueError("Shape minimum updates require the Original method"))
                return
            self.shapeModeVar.set("updated_minimum")
            self._start_job("run")

    def _start_job(self, action: str) -> None:
        if self._busy or self._exporting:
            return
        if self._history_loading:
            if self._history_previous is not None:
                self._restore_view(self._history_previous)
        self._invalidate_history()
        try:
            if not self._input_signature()[2]:
                raise SampleAnalysisError("Project session changed; reopen analysis from the current project")
            mode = self.modeVar.get()
            if action == "suggest" and mode not in ("Original FOAMS 1.0.5", "Compare methods"):
                raise ValueError("Suggest cutoffs requires the Original method")
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
                self._modern_validation()
                try:
                    modern_cfg = self._parse_modern_config()
                except Exception as exc:
                    modern_error = exc
                    self._modern_validation(exc)
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
        generation = self._view_generation
        if not self._input_signature()[2]:
            raise ValueError("Project session changed; reopen Results / history from the current project")
        request = (generation, self.resultViewVar.get(), record.run_id, self.project, self.project.project_id, self._project_session)
        self._history_request = request
        self._history_loading = True
        self._sync_operation_controls()
        self.statusVar.set(f"Loading saved run {record.run_id[:8]}")

        def worker():
            try:
                self._history_queue.put((request, record, record.data(), run_tables(record), None))
            except Exception as exc:
                self._history_queue.put((request, record, None, None, exc))
        threading.Thread(target=worker, daemon=True).start()
        self._schedule_after(20, self._poll_history)

    def _poll_history(self):
        if not self._history_loading:
            return
        try:
            request, record, data, tables, error = self._history_queue.get_nowait()
        except queue.Empty:
            self._schedule_after(20, self._poll_history)
            return
        generation, label, run_id, project, project_id, session = request
        if (generation != self._view_generation or request is not self._history_request
                or label != self.resultViewVar.get() or run_id != record.run_id
                or project is not self.project or project_id != self.project.project_id
                or session != self._project_session or not self._input_signature()[2]):
            if request is self._history_request:
                self._invalidate_history()
                if self._history_previous is not None:
                    self._restore_view(self._history_previous)
                self.statusVar.set("Discarded incompatible history completion")
                return
            self._schedule_after(20, self._poll_history)
            return
        self._history_loading = False
        self._history_request = None
        self._sync_operation_controls()
        if error is not None:
            self._restore_view(self._history_previous)
            self._show_failure("History load", error)
            return
        previous = self._history_previous
        try:
            self._clear_distribution()
            details = {key: value for key, value in data.items() if key not in ("tables", "methods")}
            details["display_table_projection"] = {"schema": TABLE_PROJECTION_SCHEMA_VERSION, "source_run_schema": record.schema,
                "basis": "retained native projections; no scientific recomputation"}
            details["methods"] = [{key: value for key, value in method.items() if key != "native"} for method in data["methods"]]
            self._set_results([json.dumps(details, ensure_ascii=False, indent=2)])
            self._populate_saved_supplement(data, tables)
            self._selected_saved_run = record
            self._saved_view_data = data
            self._displayed_request = None
            self._displayed_envelopes = ()
            self.resultViewVar.set(label)
            self._selected_view_label = label
            self._filter_saved_tables()
            self._refresh_graph()
            self._refresh_result_provenance()
            self.statusVar.set(f"Saved run {record.run_id[:8]} loaded; no science rerun")
        except Exception as exc:
            self._restore_view(previous)
            self._show_failure("History rendering", exc)

    def _populate_saved_supplement(self, data, tables):
        self._table_run_id = data["run_id"]
        self.overviewText.configure(state="normal")
        self.overviewText.delete("1.0", "end")
        lines = [data["sample_name"], data["completed_at_utc"], f"{len(data['images'])} captured images",
             f"Table projection {TABLE_PROJECTION_SCHEMA_VERSION}; captured record schema {data['schema']}", ""]
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
        for row in tables["summary_metrics"][:100]:
            value = row["value"] if row["availability"] == "available" else "unavailable: " + row["reason"]
            lines.append(f"{row['method_id']} {row['metric_id']}: {value} {row['unit']}")
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

    def _plot_saved_rows(self, data=None):
        data = data or self._graph_source()
        if self.plotTypeVar.get() == NV_PLOT:
            self._render_inverse_graph(data)
            return
        self.plotSeriesSelector.configure(state="readonly")
        self.graphSettingsButton.pack_forget()
        self.signedGraphButton.pack_forget()
        self.plotAxes.clear()
        self._nonpositive_plot_count = 0
        series = {}
        source_rows = self._distribution_rows if self._selected_saved_run is not None or data is None else data["tables"]["distributions"]
        plotted_rows = []
        for row in source_rows:
            if self.plotMethodVar.get() != "All methods" and row["method_id"] != ("legacy" if self.plotMethodVar.get() == "Original" else "modern"):
                continue
            if self.plotStageVar.get() != "All series" and row["stage"] != self.plotStageVar.get().lower():
                continue
            plotted_rows.append(row)
            key = (row["method_id"], row["group_id"], row["stage"])
            diameter = row["plot_diameter_mm"]
            value = row["na_per_mm2"]
            if isinstance(value, (int, float)):
                series.setdefault(key, []).append((diameter, value))
            else:
                self._nonpositive_plot_count += 1
        for key, pairs in series.items():
            name = ("Original" if key[0] == "legacy" else "Modern") + " " + " ".join(key[1:])
            self._plot_values(name, tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs))
        self._finish_plot()
        messages = []
        if data is not None:
            messages = [f"{method['method_id']} failed: {item['message']}" for method in data["methods"] if method["outcome"] == "failed" for item in method["diagnostics"]]
            methods = "/".join("Original" if method == "legacy" else "Modern" for method in sorted({row["method_id"] for row in plotted_rows}))
            self._plotted_metadata = {"run_id": data["run_id"], "project_id": data["project_id"], "sample_name": data["sample_name"],
                "plot_type": NA_PLOT, "x_scale": self.plotAxes.get_xscale(), "y_scale": self.plotAxes.get_yscale(),
                "omitted_nonpositive_or_unavailable": self._nonpositive_plot_count, "rows": plotted_rows,
                "messages": messages, "caption": f"{data['sample_name'][:70]} | run {data['run_id'][:12]} | {methods} native 2D number density"}
        self.graphInfoLabel.configure(text=" | ".join(messages + [f"Selected-scale omissions: {self._nonpositive_plot_count}; N_A in mm^-2"]))

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
        if self._exporting or self._busy or self._history_loading:
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
        self._sync_operation_controls()
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
        self._sync_operation_controls()
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
        if method == "modern":
            self._modern_validation(error)

    def _modern_validation(self, error=None):
        for label in self._modern_feedback.values():
            label.configure(text="")
            label.pack_forget()
        if error is None:
            return
        message = str(error)
        key = "detection"
        for field, tokens in (("reconstruction", ("upper_tail", "upper tail", "3d", "reconstruct")),
                              ("reliability", ("max_reliable", "maximum reliable")),
                              ("segments", ("segment", "plan", "group")), ("edges", ("bin", "edge"))):
            if any(token in message.lower() for token in tokens):
                key = field
                break
        included = [image for image in self.project.images if image.analysis_included]
        for image in included:
            message = message.replace(image.image_id, image.display_name)
        label = self._modern_feedback[key]
        label.configure(text=", ".join(image.display_name for image in included) + ": " + message)
        label.pack(fill="x")

    def _clear_distribution(self) -> None:
        self.distributionTable.delete(*self.distributionTable.get_children())
        self._table_rows = {}
        self.plotAxes.clear()
        self._nonpositive_plot_count = 0
        self._plotted_metadata = {}
        self._plot_data = None

    def _capture_view(self):
        return (self.resultsText.get("1.0", "end-1c"),
                [(self.distributionTable.item(row_id, "values"), self._table_rows.get(row_id))
                 for row_id in self.distributionTable.get_children()],
                                [(line.get_xdata(), line.get_ydata(), line.get_label(),
                                    {"marker": line.get_marker(), "linestyle": line.get_linestyle(), "color": line.get_color()}) for line in self.plotAxes.lines],
                self._displayed_request, self._displayed_envelopes,
                self.resultProvenanceVar.get(), getattr(self, "_selected_view_label", "Current request"), self._displayed_outcomes,
                (self._selected_saved_run, self._saved_view_data, self._distribution_rows, self._object_rows, self._table_run_id,
                 self.overviewText.get("1.0", "end-1c"), self._object_page, self._distribution_page),
                (self._plotted_metadata, self._plot_data, self.graphInfoLabel.cget("text"),
                 tuple(variable.get() for variable in (self.plotTypeVar, self.plotMethodVar, self.plotStageVar, self.plotScaleVar, self.plotXAxisVar)),
                 self.plotAxes.get_xscale(), self.plotAxes.get_yscale(), self.plotAxes.get_xlim(), self.plotAxes.get_ylim(),
                 self.plotAxes.get_xlabel(), self.plotAxes.get_ylabel(), self.plotAxes.get_title()))

    def _restore_view(self, view) -> None:
        text, rows, curves, request, envelopes, provenance, label, outcomes, saved, graph = view
        self._clear_distribution()
        self._set_results([text])
        for values, metadata in rows:
            row_id = self.distributionTable.insert("", "end", values=values)
            self._table_rows[row_id] = metadata
        for diameters, values, name, style in curves:
            if name == "Zero":
                self.plotAxes.axhline(0, color=style["color"], linewidth=0.8, label="Zero")
            else:
                self.plotAxes.plot(diameters, values, label=name, **style)
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
        (self._plotted_metadata, self._plot_data, notice, options, xscale, yscale, xlim, ylim, xlabel, ylabel, title) = graph
        for variable, value in zip((self.plotTypeVar, self.plotMethodVar, self.plotStageVar, self.plotScaleVar, self.plotXAxisVar), options):
            variable.set(value)
        self.graphInfoLabel.configure(text=notice)
        self._sync_graph_controls(self._plotted_metadata.get("messages", ()))
        self.plotAxes.set_xscale(xscale)
        self.plotAxes.set_yscale(yscale)
        self.plotAxes.set_xlim(xlim)
        self.plotAxes.set_ylim(ylim)
        self.plotAxes.set_xlabel(xlabel)
        self.plotAxes.set_ylabel(ylabel)
        self.plotAxes.set_title(title, fontsize=10)
        if curves:
            self.plotAxes.legend(fontsize=8)
        self.plotCanvas.draw_idle()

    def _add_row(self, method, group, index, lower, upper, count, area, na, nv="", volume="", kind="selected") -> None:
        row_id = self.distributionTable.insert("", "end", values=(method, group, index, lower, upper, count, area, na, nv, volume))
        self._table_rows[row_id] = (kind, group, index, lower, upper)

    def _plot_values(self, name, diameters, values) -> None:
        pairs = [(diameter, value) for diameter, value in zip(diameters, values)
                 if math.isfinite(diameter) and math.isfinite(value)
                 and (self.plotXAxisVar.get() != "Log diameter" or diameter > 0)
                 and (self.plotScaleVar.get() != "Log Y" or value > 0)]
        self._nonpositive_plot_count += len(values) - len(pairs)
        if pairs:
            self.plotAxes.plot([pair[0] for pair in pairs], [pair[1] for pair in pairs], marker=".", label=name)

    def _finish_plot(self) -> None:
        self.plotAxes.set_xlabel("Diameter / label (mm)")
        self.plotAxes.set_ylabel("N_A (mm^-2)")
        if self.plotAxes.lines:
            self.plotAxes.set_xscale("log" if self.plotXAxisVar.get() == "Log diameter" else "linear")
            self.plotAxes.set_yscale("log" if self.plotScaleVar.get() == "Log Y" else "linear")
            for axis, scale in ((self.plotAxes.xaxis, self.plotAxes.get_xscale()), (self.plotAxes.yaxis, self.plotAxes.get_yscale())):
                if scale == "log":
                    axis.set_major_locator(LogLocator(numticks=3))
                    axis.set_minor_locator(NullLocator())
            self.plotAxes.legend(fontsize=8)
        self.plotAxes.set_title(f"Rows omitted for selected scales: {self._nonpositive_plot_count}", fontsize=9)
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
