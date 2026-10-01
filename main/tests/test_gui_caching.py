import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main.core.stereology import PoreProps
from main.gui.postprocessing import PostprocessWindow, _History
from main.gui.preprocessing import PreprocessApp
from main.gui.stereology import StereologyWindow


class FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class FakeTree:
    def __init__(self):
        self.rows = []
        self.insert_count = 0
        self.delete_count = 0

    def get_children(self):
        return tuple(range(len(self.rows)))

    def delete(self, _item):
        self.delete_count += 1
        self.rows.pop()

    def insert(self, _parent, _position, values):
        self.insert_count += 1
        self.rows.append(values)


class FakeAxes:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class TestStereologyCaches(unittest.TestCase):
    def test_measurements_reused_until_revision_changes(self):
        window = object.__new__(StereologyWindow)
        window.labels = [np.array([[0, 1]], dtype=np.int32)]
        window.scales = [None]
        window.label_revisions = [2]
        window.calibration_revisions = [3]
        window._measurement_cache = {}

        with patch("main.gui.stereology.measure_labels", return_value=[]) as measure:
            window._measure_image(0)
            window._measure_image(0)
            self.assertEqual(measure.call_count, 1)
            window.label_revisions[0] += 1
            window._measure_image(0)
            self.assertEqual(measure.call_count, 2)
            self.assertEqual(len(window._measurement_cache), 1)

    def test_colorization_reused_for_same_display_key(self):
        window = object.__new__(StereologyWindow)
        window.index = 0
        window.labels = [np.array([[0, 1]], dtype=np.int32)]
        window.images = [np.zeros((1, 2), dtype=np.uint8)]
        window.image_revisions = [1]
        window.label_revisions = [4]
        window.seedVar = FakeVar(123)
        window.overlayVar = FakeVar(True)
        window.alphaVar = FakeVar(0.45)
        window._color_cache = {}

        colored = np.zeros((1, 2, 3), dtype=np.uint8)
        with patch("main.gui.stereology.colorize_labels", return_value=colored) as colorize:
            self.assertIs(window._current_colorized(), colored)
            self.assertIs(window._current_colorized(), colored)
            self.assertEqual(colorize.call_count, 1)
            window.alphaVar.value = 0.5
            window._current_colorized()
            self.assertEqual(colorize.call_count, 2)

    def test_plot_only_changes_do_not_rebuild_table(self):
        prop = PoreProps(
            0, 1, 10, 12.0, 2.0, 3.0, 0, 0, 4, 5, False,
            3.5, 0.8, 4.0, 2.0, 2.0, 10.0, 4.5, 2.1,
        )
        window = object.__new__(StereologyWindow)
        window.tree = FakeTree()
        window.ax = FakeAxes()
        window.canvas_mpl = type("Canvas", (), {"draw_idle": lambda self: None})()
        window.binsVar = FakeVar(20)
        window.logYVar = FakeVar(False)
        window._table_signature = None
        window._collect_props = lambda: [prop]
        window._values_for_metric = lambda props: (np.array([1.0]), "x", "Count")

        window._compute_and_plot()
        window.binsVar.value = 40
        window.logYVar.value = True
        window._compute_and_plot()

        self.assertEqual(window.tree.insert_count, 1)
        self.assertEqual(window.tree.delete_count, 0)


class TestPostprocessingEvents(unittest.TestCase):
    def test_hover_only_moves_preview_circle(self):
        calls = []
        window = object.__new__(PostprocessWindow)
        window.radiusVar = FakeVar(10)
        window._scale = 0.5
        window._set_preview_circle = lambda *args: calls.append(args)
        event = type("Event", (), {"x": 7, "y": 9})()

        window._on_motion_preview(event)

        self.assertEqual(calls, [(7, 9, 5)])

    def test_one_hundred_hover_events_allocate_no_photos(self):
        window = object.__new__(PostprocessWindow)
        window.radiusVar = FakeVar(10)
        window._scale = 0.5
        window._set_preview_circle = lambda *_args: None
        event = type("Event", (), {"x": 7, "y": 9})()

        with patch("main.gui.postprocessing.ImageTk.PhotoImage") as photo:
            for _ in range(100):
                window._on_motion_preview(event)
        photo.assert_not_called()

    def test_stroke_release_flushes_endpoint_before_commit(self):
        calls = []
        window = object.__new__(PostprocessWindow)
        window._stroke_active = True
        window._last_img_pt = (1, 2)
        window._pending_canvas_pt = (8, 9)
        window.modeVar = FakeVar("paint")
        window.radiusVar = FakeVar(3)
        window._scale = 1.0
        window._canvas_to_image_pt = lambda x, y: (x, y)
        window._stroke_line = lambda start, end, paint: calls.append(
            ("stroke", start, end, paint)
        )
        window._compose_overlay_full = lambda: calls.append(("compose",))
        window._set_preview_circle = lambda *args: calls.append(("preview", *args))
        window._commit_after = lambda: calls.append(("commit",))
        window._notify_parent = lambda: calls.append(("notify",))
        event = type("Event", (), {"x": 10, "y": 11})()

        window._on_stroke_end(event)

        self.assertEqual(calls[0], ("stroke", (1, 2), (10, 11), True))
        self.assertLess(calls.index(("compose",)), calls.index(("commit",)))

    def test_release_paints_endpoint_and_undo_restores_mask(self):
        window = object.__new__(PostprocessWindow)
        window.index = 0
        window.masks = [np.zeros((16, 16), dtype=bool)]
        history = _History(4)
        history.ensure_initial(window.masks[0])
        window._histories = [history]
        window._kernel_cache = {}
        window.radiusVar = FakeVar(1)
        window.modeVar = FakeVar("paint")
        window._scale = 1.0
        window._stroke_active = True
        window._last_img_pt = (2, 2)
        window._pending_canvas_pt = (8, 8)
        window._canvas_to_image_pt = lambda x, y: (x, y)
        window._compose_overlay_full = lambda: None
        window._on_motion_preview = lambda _event: None
        window._notify_parent = lambda: None
        event = type("Event", (), {"x": 10, "y": 11})()

        window._on_stroke_end(event)
        self.assertTrue(window.masks[0][11, 10])
        window._undo()
        self.assertFalse(window.masks[0].any())


class TestRevisionOwnership(unittest.TestCase):
    def test_mask_change_invalidates_only_affected_label(self):
        app = object.__new__(PreprocessApp)
        app.images = [np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8)]
        app.masks = [np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8)]
        app.labels = [np.ones((2, 2), np.int32), np.ones((2, 2), np.int32)]
        app.maskRevisions = [2, 4]
        app.labelRevisions = [3, 5]
        edited = app.masks[0].copy()
        edited[0, 0] = 255

        changed = app._acceptEditedMasks([edited, app.masks[1].copy()])

        self.assertEqual(changed, [0])
        self.assertEqual(app.maskRevisions, [3, 4])
        self.assertEqual(app.labelRevisions, [4, 5])
        self.assertIsNone(app.labels[0])
        self.assertIsNotNone(app.labels[1])


if __name__ == "__main__":
    unittest.main()
