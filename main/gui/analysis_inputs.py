from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk

from ..core.sample_project import SUPPORTED_CALIBRATION_UNITS_TO_MM

MICROMETRES = "\u00b5m"
LENGTH_UNITS = ("m", "cm", "mm", MICROMETRES, "nm")


def normalized_unit(unit):
    unit = str(unit or "mm").lower()
    if SUPPORTED_CALIBRATION_UNITS_TO_MM.get(unit) == 0.001:
        return MICROMETRES
    if SUPPORTED_CALIBRATION_UNITS_TO_MM.get(unit) == 0.000001:
        return "nm"
    return unit if unit in LENGTH_UNITS else "mm"


def inherited_unit(project):
    units = {normalized_unit(image.calibration.unit_name) for image in project.images
             if image.analysis_included and image.calibration.is_calibrated}
    return next(iter(units)) if len(units) == 1 else "mm"


def unit_scale(unit):
    return SUPPORTED_CALIBRATION_UNITS_TO_MM[normalized_unit(unit)]


class CommittedNumber(ttk.Frame):
    def __init__(self, parent, canonical, units, *, physical=True, suffix="", changed=None):
        super().__init__(parent)
        self.canonical = canonical
        self.units = units
        self.physical = physical
        self.changed = changed
        self.text = tk.StringVar(self)
        self.error = tk.StringVar(self)
        self.entry = ttk.Entry(self, textvariable=self.text, width=16)
        self.entry.grid(row=0, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)
        self.unitLabel = ttk.Label(self, text=suffix)
        self.unitLabel.grid(row=0, column=1, padx=4)
        self.errorLabel = ttk.Label(self, textvariable=self.error, foreground="#a32929", wraplength=480)
        self.entry.bind("<Return>", self.commit)
        self.entry.bind("<FocusOut>", self.commit)
        self._trace = canonical.trace_add("write", self.render)
        self.bind("<Destroy>", self._destroyed, add="+")
        self.render()

    def _destroyed(self, event):
        if event.widget is self:
            self.canonical.trace_remove("write", self._trace)
            self.canonical = None
            self.units = None
            self.changed = None
            self.text = None
            self.error = None

    def render(self, *_args):
        raw = self.canonical.get()
        try:
            number = float(raw)
            raw = format(number / unit_scale(self.units.get()) if self.physical else number, ".12g")
        except ValueError:
            pass
        self._rendered = raw
        self.text.set(raw)
        if self.physical:
            self.unitLabel.configure(text=self.units.get())
        self.error.set("")
        self.errorLabel.grid_remove()

    def commit(self, _event=None):
        raw = self.text.get().strip()
        if raw == self._rendered:
            return True
        try:
            if raw:
                number = float(raw)
                if not math.isfinite(number):
                    raise ValueError("Enter a finite number")
                value = repr(number * unit_scale(self.units.get()) if self.physical else number)
            else:
                value = ""
        except ValueError:
            self.error.set("Enter a finite number")
            self.errorLabel.grid(row=1, column=0, columnspan=2, sticky="w")
            return False
        if value != self.canonical.get():
            self.canonical.set(value)
            if self.changed is not None:
                self.changed()
        return True


class IntervalEditor(ttk.Frame):
    def __init__(self, parent, canonical, units):
        super().__init__(parent)
        self.canonical = canonical
        self.units = units
        self.boundaries = []
        self.numbers = []
        self._writing = False
        self.rows = ttk.Frame(self)
        self.rows.pack(fill="x")
        self.addButton = ttk.Button(self, text="Add interval", command=self.add_interval)
        self.addButton.pack(anchor="w")
        self.message = ttk.Label(self, foreground="#a32929", wraplength=540)
        self.message.pack(fill="x")
        ttk.Label(self, text="Classes include the lower edge and exclude the upper edge; the final upper edge is included.",
                  wraplength=600).pack(fill="x")
        self._trace = canonical.trace_add("write", self.restore)
        self.bind("<Destroy>", self._destroyed, add="+")
        self.restore()

    def _destroyed(self, event):
        if event.widget is self:
            self.canonical.trace_remove("write", self._trace)
            self.boundaries.clear()
            self.numbers.clear()

    def restore(self, *_args):
        if self._writing:
            return
        raw = self.canonical.get()
        self.boundaries = [tk.StringVar(self, value=value.strip()) for value in raw.split(",")] if raw else []
        self._draw()

    def _draw(self):
        for widget in self.rows.winfo_children():
            widget.destroy()
        self.numbers = []
        for index in range(max(0, len(self.boundaries) - 1)):
            row = ttk.Frame(self.rows)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"{index + 1}.").pack(side="left", padx=3)
            for boundary in (index, index + 1):
                number = CommittedNumber(row, self.boundaries[boundary], self.units, changed=self.publish)
                number.pack(side="left", padx=3)
                self.numbers.append(number)
            ttk.Button(row, text="Remove interval", command=lambda selected=index: self.remove_interval(selected)).pack(side="left", padx=3)
        self.validate()

    def add_interval(self):
        if not self.boundaries:
            self.boundaries.append(tk.StringVar(self, value=""))
        self.boundaries.append(tk.StringVar(self, value=""))
        self._draw()
        self.numbers[-1].entry.focus_set()

    def remove_interval(self, index):
        if len(self.boundaries) == 2:
            self.boundaries = []
        else:
            del self.boundaries[index + 1]
        self._draw()
        self.publish()

    def validate(self):
        try:
            if len(self.boundaries) < 2:
                raise ValueError("Add a size interval and enter both boundaries")
            edges = [float(variable.get()) for variable in self.boundaries]
            if any(not math.isfinite(edge) or edge <= 0 for edge in edges):
                raise ValueError("Class boundaries must be finite and positive")
            if any(upper <= lower for lower, upper in zip(edges, edges[1:])):
                raise ValueError("Each upper boundary must exceed its lower boundary; shared boundaries prevent gaps and overlaps")
            self.message.configure(text="")
            return True
        except ValueError as exc:
            self.message.configure(text=str(exc))
            return False

    def publish(self):
        if not self.validate():
            return False
        self._writing = True
        try:
            self.canonical.set(",".join(variable.get() for variable in self.boundaries))
        finally:
            self._writing = False
        return True

    def commit(self):
        return all([number.commit() for number in self.numbers]) and self.publish()

    def render(self):
        for number in self.numbers:
            number.render()


class GroupRangeEditor(ttk.Frame):
    def __init__(self, parent, canonical, units, groups, intervals=None):
        super().__init__(parent)
        self.canonical = canonical
        self.units = units
        self.groups = groups
        self.intervals = intervals
        self.items = []
        self._writing = False
        self._restore_error = False
        self.rows = ttk.Frame(self)
        self.rows.pack(fill="x")
        self.addButton = ttk.Button(self, text="Add source range", command=self.add_range)
        self.addButton.pack(anchor="w")
        self.message = ttk.Label(self, foreground="#a32929", wraplength=560)
        self.message.pack(fill="x")
        self._trace = canonical.trace_add("write", self.restore)
        self.bind("<Destroy>", self._destroyed, add="+")
        self.restore()

    def _destroyed(self, event):
        if event.widget is self:
            self.canonical.trace_remove("write", self._trace)
            self.items.clear()
            self.numbers.clear()

    def restore(self, *_args):
        if self._writing:
            return
        self.items = []
        self._restore_error = False
        try:
            for token in self.canonical.get().split(";"):
                if token.strip():
                    group, lower, upper = token.split(":")
                    self.items.append((tk.StringVar(self, value=group), tk.StringVar(self, value=lower), tk.StringVar(self, value=upper)))
            self._draw()
        except ValueError:
            self._restore_error = True
            self._draw()
            self.message.configure(text="Saved source range needs correction")

    def add_range(self):
        self._restore_error = False
        choices = self.groups()
        self.items.append((tk.StringVar(self, value=next(iter(choices), "")), tk.StringVar(self), tk.StringVar(self)))
        self._draw()

    def remove_range(self, index):
        del self.items[index]
        self._draw()
        self.publish()

    def _draw(self):
        for widget in self.rows.winfo_children():
            widget.destroy()
        self.numbers = []
        self.groupSelectors = []
        self.intervalSelectors = []
        choices = self.groups()
        for index, (group, lower, upper) in enumerate(self.items):
            row = ttk.Frame(self.rows)
            row.pack(fill="x", pady=3)
            label = tk.StringVar(row, value=choices.get(group.get(), group.get()))
            selector = ttk.Combobox(row, textvariable=label, values=tuple(choices.values()), state="readonly", width=24)
            selector.pack(anchor="w")
            def choose(_event, variable=group, selected=label):
                variable.set(next(key for key, value in self.groups().items() if value == selected.get()))
                self.publish()
            selector.bind("<<ComboboxSelected>>", choose)
            self.groupSelectors.append(selector)
            bounds = ttk.Frame(row)
            bounds.pack(anchor="w")
            if self.intervals is None:
                for text, variable in (("Lower", lower), ("Upper", upper)):
                    ttk.Label(bounds, text=text).pack(side="left", padx=3)
                    number = CommittedNumber(bounds, variable, self.units, changed=self.publish)
                    number.pack(side="left", padx=3)
                    self.numbers.append(number)
            else:
                intervals = self.intervals()
                for text, variable, offset in (("First class", lower, 0), ("Last class", upper, 1)):
                    ttk.Label(bounds, text=text).pack(side="left", padx=3)
                    try:
                        position = int(variable.get()) - offset
                        current = intervals[position] if 0 <= position < len(intervals) else ""
                    except ValueError:
                        current = ""
                    displayed = tk.StringVar(row, value=current)
                    selector = ttk.Combobox(bounds, textvariable=displayed, values=intervals, state="readonly", width=26)
                    selector.pack(side="left", padx=3)
                    def select_interval(_event, target=variable, selected=displayed, shift=offset):
                        target.set(str(self.intervals().index(selected.get()) + shift))
                        self.publish()
                    selector.bind("<<ComboboxSelected>>", select_interval)
                    self.intervalSelectors.append(selector)
            ttk.Button(row, text="Remove source range", command=lambda selected=index: self.remove_range(selected)).pack(anchor="w")

    def publish(self):
        try:
            for group, lower, upper in self.items:
                low = float(lower.get())
                high = float(upper.get())
                if (not group.get() or not math.isfinite(low) or not math.isfinite(high) or high < low
                    or self.intervals is not None and high == low or low < 0):
                    raise ValueError("Select a source group and an increasing range")
            self._writing = True
            self.canonical.set(";".join(":".join(variable.get() for variable in item) for item in self.items))
            self.message.configure(text="")
            return True
        except ValueError:
            self.message.configure(text="Select a source group and both increasing range endpoints")
            return False
        finally:
            self._writing = False

    def commit(self):
        return not self._restore_error and all([number.commit() for number in self.numbers]) and self.publish()

    def render(self):
        self._draw()
