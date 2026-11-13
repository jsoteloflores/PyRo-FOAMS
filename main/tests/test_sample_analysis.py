from __future__ import annotations

import unittest

import numpy as np

from main.core.sample_analysis import (
    SampleAnalysisError,
    build_legacy_group_inputs,
    run_modern_analysis,
)
from main.core.sample_project import (
    accept_draft,
    add_image,
    create_analysis_snapshot,
    create_project,
    set_analysis_included,
    set_calibration,
    update_draft,
)


class TestSampleAnalysisAdapters(unittest.TestCase):
    def _snapshot_with_three_images(self):
        project = create_project("Sample")
        image_a = add_image(project, np.zeros((40, 40), dtype=np.uint8), "a.png", "dup.png")
        image_b = add_image(project, np.zeros((40, 40), dtype=np.uint8), "b.png", "dup.png")
        image_c = add_image(project, np.zeros((40, 40), dtype=np.uint8), "c.png", "third.png")

        mask_a = np.zeros((40, 40), dtype=np.uint8)
        mask_b = np.zeros((40, 40), dtype=np.uint8)
        mask_c = np.zeros((40, 40), dtype=np.uint8)
        mask_a[4:16, 4:16] = 255
        mask_b[18:30, 18:30] = 255
        mask_c[6:12, 24:32] = 255

        for image, mask in ((image_a, mask_a), (image_b, mask_b), (image_c, mask_c)):
            update_draft(image, mask=mask, labels=None, settings={})
            accept_draft(image)
            set_analysis_included(image, True)

        set_calibration(image_a, 0.003, "mm")
        set_calibration(image_b, 3.0, "um")
        set_calibration(image_c, 0.005, "mm")
        return project, create_analysis_snapshot(project)

    def test_build_legacy_group_inputs_groups_by_exact_resolution(self):
        _project, snapshot = self._snapshot_with_three_images()
        group_inputs, groups, assignments = build_legacy_group_inputs(snapshot)

        self.assertEqual(len(groups), 2)
        self.assertEqual([group.source_slot for group in groups], [1, 2])
        # 0.005 mm/px is coarser than 0.003 mm/px -> smaller px_per_mm first.
        self.assertAlmostEqual(groups[0].mm_per_px, 0.005)
        self.assertAlmostEqual(groups[1].mm_per_px, 0.003)
        self.assertEqual(tuple(group.group_id for group in groups), ("group-1", "group-2"))
        self.assertEqual(len(group_inputs[0].images), 1)
        self.assertEqual(len(group_inputs[1].images), 2)
        self.assertEqual([item.source_key for item in assignments], ["mag1", "first", "second"])
        self.assertEqual([item.input_order for item in assignments], [0, 1, 2])

    def test_build_legacy_group_inputs_rejects_more_than_four_groups(self):
        project = create_project("TooMany")
        for index in range(5):
            image = add_image(project, np.zeros((20, 20), dtype=np.uint8), f"{index}.png")
            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[2:6, 2:6] = 255
            update_draft(image, mask=mask, labels=None, settings={})
            accept_draft(image)
            set_calibration(image, 0.001 + index * 0.001, "mm")
            set_analysis_included(image, True)
        snapshot = create_analysis_snapshot(project)
        with self.assertRaisesRegex(SampleAnalysisError, "up to 4 resolution groups"):
            build_legacy_group_inputs(snapshot)

    def test_legacy_adapter_uses_accepted_snapshot_not_newer_draft(self):
        project, snapshot = self._snapshot_with_three_images()
        accepted = snapshot.images[0].accepted_mask.copy()
        first_image = project.images[0]
        newer_draft = np.zeros_like(first_image.accepted_mask)
        newer_draft[25:35, 25:35] = 255
        update_draft(first_image, mask=newer_draft, labels=None, settings={"newer": True})

        group_inputs, _groups, _assignments = build_legacy_group_inputs(snapshot)
        prepared_first = None
        for group in group_inputs:
            for image_input in group.images:
                if image_input.image_id == first_image.image_id:
                    prepared_first = image_input.segmented_image
                    break
            if prepared_first is not None:
                break
        self.assertIsNotNone(prepared_first)
        self.assertTrue(np.array_equal(prepared_first, (accepted > 0).astype(np.uint8)))
        self.assertFalse(np.array_equal(prepared_first, (newer_draft > 0).astype(np.uint8)))

    def test_run_modern_analysis_returns_nested_distribution(self):
        _project, snapshot = self._snapshot_with_three_images()
        result = run_modern_analysis(snapshot)

        self.assertGreaterEqual(len(result.bin_edges_mm), 2)
        self.assertGreaterEqual(len(result.nesting_result.selected_bin_indices), 1)
        self.assertEqual(result.nesting_result.sample_id, snapshot.project_id)


if __name__ == "__main__":
    unittest.main()
