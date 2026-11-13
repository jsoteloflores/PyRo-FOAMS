from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Optional

from ..core.legacy_cutoffs import Foams105CutoffGroupInput, suggest_foams105_cutoffs
from ..core.legacy_perimeter import (
    FOAMS105_PERIMETER_PROFILE_CURRENT,
    FOAMS105_PERIMETER_PROFILE_OLD,
)
from ..core.sample_analysis import (
    CompareRunResult,
    LegacyRunConfig,
    ModernRunConfig,
    SampleAnalysisError,
    analyze_legacy_prepared,
    prepare_legacy_analysis,
    run_compare_analysis,
    run_modern_analysis,
)
from ..core.sample_project import ProjectAnalysisSnapshot, SampleProject, create_analysis_snapshot

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
        self.grab_set()
        self.project = project
        self.snapshot = snapshot
        self._mark_unsaved = mark_unsaved
        self._closed = False
        self._owned_after_ids = set()
        self._result_queue: "queue.Queue[tuple[int, object, Optional[BaseException], ProjectAnalysisSnapshot]]" = queue.Queue()
        self._job_generation = 0
        self._active_generation = 0
        self._worker: Optional[threading.Thread] = None
        self._last_completed_snapshot: Optional[ProjectAnalysisSnapshot] = None

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

        self.statusVar = tk.StringVar(value="Ready")

        self._build_ui()
        self._load_persisted_setup()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.geometry("1200x800")
        self.minsize(980, 640)
        self._refresh_summary_banner()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        summary = ttk.LabelFrame(root, text="Accepted Analysis Snapshot", padding=8)
        summary.pack(fill="x")
        self.snapshotSummary = ttk.Label(summary, justify="left", wraplength=1120)
        self.snapshotSummary.pack(fill="x")

        controls = ttk.LabelFrame(root, text="Setup", padding=8)
        controls.pack(fill="x", pady=(8, 0))

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

        ttk.Label(controls, text="Modern transition bins").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.modernTransitionBinsVar, width=28).grid(
            row=4, column=1, sticky="w", padx=(8, 16), pady=(8, 0)
        )
        ttk.Checkbutton(
            controls,
            text="Enable spherical 3D reconstruction",
            variable=self.modernReconstructVar,
        ).grid(row=4, column=2, columnspan=3, sticky="w", pady=(8, 0))

        actions = ttk.Frame(root)
        actions.pack(fill="x", pady=(8, 0))
        self.runButton = ttk.Button(actions, text="Run analysis", command=self._run_analysis)
        self.runButton.pack(side="left")
        self.suggestButton = ttk.Button(actions, text="Suggest cutoffs", command=self._suggest_cutoffs)
        self.suggestButton.pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Back to preparation", command=self._on_close).pack(
            side="left", padx=(8, 0)
        )
        ttk.Label(actions, textvariable=self.statusVar).pack(side="left", padx=(12, 0))

        results = ttk.LabelFrame(root, text="Results", padding=8)
        results.pack(fill="both", expand=True, pady=(8, 0))
        self.resultsText = tk.Text(results, wrap="word", height=24)
        self.resultsText.pack(fill="both", expand=True)

    def _load_persisted_setup(self) -> None:
        setup = self.project.analysis_setup if isinstance(self.project.analysis_setup, dict) else {}
        if not setup:
            return
        if setup.get("schema") != 1:
            self.statusVar.set("Saved setup schema is unsupported; using defaults")
            return
        mode = setup.get("mode")
        if mode in ("Original FOAMS 1.0.5", "Modern analysis", "Compare methods"):
            self.modeVar.set(mode)
        legacy = setup.get("legacy", {}) if isinstance(setup.get("legacy"), dict) else {}
        modern = setup.get("modern", {}) if isinstance(setup.get("modern"), dict) else {}
        if "minimum_diameter_px" in legacy:
            self.minimumDiameterVar.set(str(legacy["minimum_diameter_px"]))
        if "perimeter_profile" in legacy:
            self.perimeterProfileVar.set(_PERIMETER_ID_TO_LABEL.get(str(legacy["perimeter_profile"]), "R2025b"))
        if "selection_mode" in legacy:
            self.selectionModeVar.set(str(legacy["selection_mode"]))
        if "manual_ranges" in legacy and isinstance(legacy["manual_ranges"], list):
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

    def _refresh_summary_banner(self) -> None:
        lines = [
            f"Sample: {self.project.sample_name}",
            f"Project ID: {self.project.project_id}",
            f"Included accepted images: {len(self.snapshot.images)}",
        ]
        for image in self.snapshot.images:
            lines.append(
                " - "
                + f"{image.display_name} [{image.image_id}] snapshot={image.snapshot_id}, "
                + f"calibration={image.mm_per_px:.8g} mm/px"
            )
        if self._last_completed_snapshot is not None:
            old_ids = tuple(item.snapshot_id for item in self._last_completed_snapshot.images)
            new_ids = tuple(item.snapshot_id for item in self.snapshot.images)
            if old_ids != new_ids or self._last_completed_snapshot.analysis_revision != self.snapshot.analysis_revision:
                lines.append("Inputs changed - rerun required")
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
        shape_minimum = float(shape_text) if shape_text else None
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
            min_detectable_mm=float(min_text) if min_text else None,
            max_reliable_mm=float(max_text) if max_text else None,
            transition_bins=transitions,
            enable_reconstruction=bool(self.modernReconstructVar.get()),
        )

    def _legacy_prepare_key(self, snapshot: ProjectAnalysisSnapshot, config: LegacyRunConfig) -> tuple:
        return (
            snapshot.project_id,
            snapshot.analysis_revision,
            tuple((item.image_id, item.snapshot_id, item.mm_per_px, item.processing_roi) for item in snapshot.images),
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
        payload = {
            "schema": 1,
            "mode": mode,
            "legacy": {
                "minimum_diameter_px": getattr(legacy_cfg, "minimum_diameter_px", None),
                "perimeter_profile": getattr(legacy_cfg, "perimeter_profile", None),
                "selection_mode": getattr(legacy_cfg, "selection_mode", None),
                "manual_ranges": list(getattr(legacy_cfg, "manual_ranges", ()) or ()),
                "volume_normalization_mode": getattr(legacy_cfg, "volume_normalization_mode", None),
                "known_vesicularity_percent": getattr(legacy_cfg, "known_vesicularity_percent", None),
                "size_lower_cutoff_mm": getattr(legacy_cfg, "size_lower_cutoff_mm", None),
                "size_upper_cutoff_mm": getattr(legacy_cfg, "size_upper_cutoff_mm", None),
                "shape_minimum_diameter_px": getattr(legacy_cfg, "shape_minimum_diameter_px", None),
                "shape_statistics_mode": getattr(legacy_cfg, "shape_statistics_mode", None),
            },
            "modern": {
                "min_detectable_mm": getattr(modern_cfg, "min_detectable_mm", None),
                "max_reliable_mm": getattr(modern_cfg, "max_reliable_mm", None),
                "transition_bins": list(getattr(modern_cfg, "transition_bins", ()) or ()),
                "enable_reconstruction": bool(getattr(modern_cfg, "enable_reconstruction", False)),
            },
        }
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
        try:
            mode, snapshot, legacy_cfg, modern_cfg = self._parse_and_snapshot()
            self._save_setup(mode, legacy_cfg, modern_cfg)
        except Exception as exc:
            messagebox.showerror("Sample Analysis", str(exc))
            return

        self._job_generation += 1
        generation = self._job_generation
        self._active_generation = generation
        self.runButton.configure(state="disabled")
        self.suggestButton.configure(state="disabled")
        self.statusVar.set(f"Running {mode}...")

        def worker() -> None:
            try:
                if mode == "Original FOAMS 1.0.5":
                    assert legacy_cfg is not None
                    key = self._legacy_prepare_key(snapshot, legacy_cfg)
                    context = self._legacy_prepare_cache.get(key)
                    if context is None:
                        context = prepare_legacy_analysis(snapshot, legacy_cfg)
                        self._legacy_prepare_cache[key] = context
                    result = analyze_legacy_prepared(context, legacy_cfg)
                elif mode == "Modern analysis":
                    assert modern_cfg is not None
                    result = run_modern_analysis(snapshot, modern_cfg)
                else:
                    result = run_compare_analysis(snapshot, legacy_cfg, modern_cfg)
                self._result_queue.put((generation, result, None, snapshot))
            except BaseException as exc:  # pylint: disable=broad-except
                self._result_queue.put((generation, None, exc, snapshot))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
        self._schedule_after(30, self._poll_results)

    def _poll_results(self) -> None:
        if self._closed:
            return
        try:
            generation, result, error, snapshot = self._result_queue.get_nowait()
        except queue.Empty:
            self._schedule_after(30, self._poll_results)
            return

        if generation != self._active_generation:
            return

        self.runButton.configure(state="normal")
        self.suggestButton.configure(state="normal")
        if error is not None:
            self.statusVar.set("Run failed")
            lines = ["Analysis run failed", "", str(error)]
            stage = getattr(error, "stage", None)
            if stage:
                lines.append(f"stage={stage}")
            cause_type = getattr(error, "cause_type", None)
            if cause_type:
                lines.append(f"cause_type={cause_type}")
            cause_code = getattr(error, "cause_code", None)
            if cause_code:
                lines.append(f"cause_code={cause_code}")
            context = getattr(error, "context", None)
            if isinstance(context, dict) and context:
                lines.append("context:")
                for key, value in sorted(context.items()):
                    lines.append(f" - {key}={value}")
            self._set_results(lines)
            return

        try:
            if isinstance(result, CompareRunResult):
                self._latest_compare = result
                self._render_compare(result)
                self.statusVar.set("Compare run complete")
            elif hasattr(result, "analysis"):
                self._latest_legacy = result
                self._render_legacy(result)
                self.statusVar.set("Original run complete")
            else:
                self._latest_modern = result
                self._render_modern(result)
                self.statusVar.set("Modern run complete")
            self._last_completed_snapshot = snapshot
            self.snapshot = snapshot
            self._refresh_summary_banner()
        except Exception as exc:
            self.statusVar.set("Render failed")
            messagebox.showerror("Sample Analysis", f"Result rendering failed: {exc}")

    def _suggest_cutoffs(self) -> None:
        try:
            mode, snapshot, legacy_cfg, _modern_cfg = self._parse_and_snapshot()
            if mode not in ("Original FOAMS 1.0.5", "Compare methods"):
                raise SampleAnalysisError("cutoff suggestion is available for Original FOAMS setup only")
            assert legacy_cfg is not None
            key = self._legacy_prepare_key(snapshot, legacy_cfg)
            context = self._legacy_prepare_cache.get(key)
            if context is None:
                context = prepare_legacy_analysis(snapshot, legacy_cfg)
                self._legacy_prepare_cache[key] = context

            groups = []
            for group in context.prepared.groups:
                if group.normalization_result is None:
                    continue
                groups.append(
                    Foams105CutoffGroupInput(
                        group.group_id,
                        group.scale_px_per_mm,
                        group.normalization_result.na_per_mm2,
                    )
                )
            if not groups:
                raise SampleAnalysisError("no normalized groups are available for cutoff suggestion")
            suggestion = suggest_foams105_cutoffs(
                context.prepared.label_preparation_result.bin_labels_mm,
                tuple(groups),
            )
            manual = [
                f"{item.group_id}:{item.lower_label_mm}:{item.upper_label_mm}"
                for item in suggestion.suggested_ranges
            ]
            self.manualRangesVar.set(";".join(manual))
            self.selectionModeVar.set("manual")
            self.statusVar.set("Suggested cutoffs applied to manual ranges")
        except Exception as exc:
            messagebox.showerror("Suggest cutoffs", str(exc))

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
        self._set_results(lines)

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
        self._set_results(lines)

    def _render_compare(self, result: CompareRunResult) -> None:
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
