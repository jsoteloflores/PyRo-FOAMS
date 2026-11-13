from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Optional

from ..core.sample_analysis import (
    CompareRunResult,
    LegacyRunConfig,
    ModernRunConfig,
    run_compare_analysis,
    run_legacy_analysis,
    run_modern_analysis,
)
from ..core.sample_project import ProjectAnalysisSnapshot, SampleProject


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
        self._result_queue: "queue.Queue[tuple[int, object, Optional[BaseException]]]" = queue.Queue()
        self._job_generation = 0
        self._active_generation = 0
        self._worker: Optional[threading.Thread] = None
        self._latest_legacy = None
        self._latest_modern = None
        self._latest_compare = None

        self.modeVar = tk.StringVar(value="Original FOAMS 1.0.5")
        self.minimumDiameterVar = tk.StringVar(value="2.0")
        self.perimeterProfileVar = tk.StringVar(value="R2025b")
        self.selectionModeVar = tk.StringVar(value="autosmart")
        self.volumeModeVar = tk.StringVar(value="unnormalized")
        self.knownVesicularityVar = tk.StringVar(value="")
        self.shapeMinimumVar = tk.StringVar(value="")
        self.modernReconstructVar = tk.BooleanVar(value=False)
        self.statusVar = tk.StringVar(value="Ready")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.geometry("1160x760")
        self.minsize(900, 620)
        self._refresh_summary_banner()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        summary = ttk.LabelFrame(root, text="Accepted Analysis Snapshot", padding=8)
        summary.pack(fill="x")
        self.snapshotSummary = ttk.Label(summary, justify="left")
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
        ttk.Entry(controls, textvariable=self.minimumDiameterVar, width=10).grid(row=0, column=3, sticky="w", padx=(8, 16))

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

        ttk.Label(controls, text="Normalization").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.volumeModeVar,
            values=("unnormalized", "integrated_2d", "known_vesicularity"),
            state="readonly",
            width=18,
        ).grid(row=1, column=3, sticky="w", padx=(8, 16), pady=(8, 0))

        ttk.Label(controls, text="Known vesicularity (%)").grid(row=1, column=4, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.knownVesicularityVar, width=10).grid(row=1, column=5, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(controls, text="Shape minimum (px)").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.shapeMinimumVar, width=10).grid(row=2, column=1, sticky="w", padx=(8, 16), pady=(8, 0))

        ttk.Checkbutton(
            controls,
            text="Modern: reconstruct spherical 3D",
            variable=self.modernReconstructVar,
        ).grid(row=2, column=2, columnspan=3, sticky="w", pady=(8, 0))

        actions = ttk.Frame(root)
        actions.pack(fill="x", pady=(8, 0))
        self.runButton = ttk.Button(actions, text="Run analysis", command=self._run_analysis)
        self.runButton.pack(side="left")
        ttk.Button(actions, text="Back to preparation", command=self._on_close).pack(side="left", padx=(8, 0))
        ttk.Label(actions, textvariable=self.statusVar).pack(side="left", padx=(12, 0))

        results = ttk.LabelFrame(root, text="Results", padding=8)
        results.pack(fill="both", expand=True, pady=(8, 0))
        self.resultsText = tk.Text(results, wrap="word", height=22)
        self.resultsText.pack(fill="both", expand=True)

    def _refresh_summary_banner(self) -> None:
        lines = [f"Sample: {self.project.sample_name}", f"Project ID: {self.project.project_id}", f"Included accepted images: {len(self.snapshot.images)}"]
        for image in self.snapshot.images:
            lines.append(
                " - "
                + f"{image.display_name} [{image.image_id}] snapshot={image.snapshot_id}, "
                + f"calibration={image.mm_per_px:.8g} mm/px"
            )
        if self.project.analysis_revision != self.snapshot.analysis_revision:
            lines.append("Inputs changed - rerun required")
        self.snapshotSummary.configure(text="\n".join(lines))

    def _parse_legacy_config(self) -> LegacyRunConfig:
        minimum = float(self.minimumDiameterVar.get())
        known_text = self.knownVesicularityVar.get().strip()
        known = float(known_text) if known_text else None
        perimeter = "PerimeterOld" if self.perimeterProfileVar.get() == "PerimeterOld" else "Perimeter"
        shape_text = self.shapeMinimumVar.get().strip()
        shape_minimum = float(shape_text) if shape_text else None
        return LegacyRunConfig(
            minimum_diameter_px=minimum,
            perimeter_profile=perimeter,
            selection_mode=self.selectionModeVar.get(),
            manual_ranges=(),
            volume_normalization_mode=self.volumeModeVar.get(),
            known_vesicularity_percent=known,
            size_lower_cutoff_mm=0.5,
            size_upper_cutoff_mm=1.0,
            shape_minimum_diameter_px=shape_minimum,
            shape_statistics_mode="initial_analysis",
        )

    def _parse_modern_config(self) -> ModernRunConfig:
        return ModernRunConfig(enable_reconstruction=bool(self.modernReconstructVar.get()))

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
        self._refresh_summary_banner()
        try:
            legacy_cfg = self._parse_legacy_config()
            modern_cfg = self._parse_modern_config()
        except ValueError:
            messagebox.showerror("Sample Analysis", "Enter valid numeric settings.")
            return

        mode = self.modeVar.get()
        self._job_generation += 1
        generation = self._job_generation
        self._active_generation = generation
        self.runButton.configure(state="disabled")
        self.statusVar.set(f"Running {mode}...")

        if self._mark_unsaved is not None:
            self._mark_unsaved()

        self.project.analysis_setup = {
            "schema": 1,
            "mode": mode,
            "legacy": {
                "minimum_diameter_px": legacy_cfg.minimum_diameter_px,
                "perimeter_profile": legacy_cfg.perimeter_profile,
                "selection_mode": legacy_cfg.selection_mode,
                "volume_normalization_mode": legacy_cfg.volume_normalization_mode,
                "known_vesicularity_percent": legacy_cfg.known_vesicularity_percent,
                "shape_minimum_diameter_px": legacy_cfg.shape_minimum_diameter_px,
            },
            "modern": {
                "enable_reconstruction": modern_cfg.enable_reconstruction,
            },
        }

        def worker() -> None:
            try:
                if mode == "Original FOAMS 1.0.5":
                    result = run_legacy_analysis(self.snapshot, legacy_cfg)
                elif mode == "Modern analysis":
                    result = run_modern_analysis(self.snapshot, modern_cfg)
                else:
                    result = run_compare_analysis(self.snapshot, legacy_cfg, modern_cfg)
                self._result_queue.put((generation, result, None))
            except BaseException as exc:  # pylint: disable=broad-except
                self._result_queue.put((generation, None, exc))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
        self._schedule_after(30, self._poll_results)

    def _poll_results(self) -> None:
        if self._closed:
            return
        try:
            generation, result, error = self._result_queue.get_nowait()
        except queue.Empty:
            self._schedule_after(30, self._poll_results)
            return

        if generation != self._active_generation:
            return

        self.runButton.configure(state="normal")
        if error is not None:
            self.statusVar.set("Run failed")
            messagebox.showerror("Sample Analysis", str(error))
            return

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

    def _render_legacy(self, result) -> None:
        analysis = result.analysis
        lines = [
            "Original FOAMS 1.0.5",
            f"Method: {analysis.method}",
            f"Image count: {analysis.image_count}",
            f"Prepared groups: {len(analysis.prepared_result.groups)}",
            f"Selected rows: {analysis.binned_selection_result.selected_row_count}",
            f"Raw selected objects: {analysis.raw_selection_result.selected_count}",
            f"Volume mode: {analysis.analysis_settings.volume_normalization_mode}",
            f"Diagnostics: {len(analysis.diagnostics)}",
            "",
            "Group contributions:",
        ]
        for group in analysis.prepared_result.groups:
            normalization = group.normalization_result
            status = "available" if normalization is not None else "unavailable"
            lines.append(
                f" - {group.group_id} slot={group.source_slot} scale_px_per_mm={group.scale_px_per_mm:.8g} "
                + f"images={len(group.images)} area1={group.group_area_result.phase_corrected_area_mm2:.8g} "
                + f"area2={group.group_area_result.border_corrected_area_mm2:.8g} normalization={status}"
            )
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
        self._set_results(lines)

    def _render_compare(self, result: CompareRunResult) -> None:
        lines = [
            "Compare methods",
            "",
            f"Legacy selected rows: {result.legacy.analysis.binned_selection_result.selected_row_count}",
            f"Legacy raw selected: {result.legacy.analysis.raw_selection_result.selected_count}",
            f"Modern selected bins: {len(result.modern.nesting_result.selected_bin_indices)}",
            f"Modern nested rows: {len(result.modern.nesting_result.counts)}",
            "",
            "Both paths used the same accepted snapshot identities:",
        ]
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
