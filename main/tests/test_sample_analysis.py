from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from main.core.sample_analysis import (
    CompareRunResult,
    LegacyRunConfig,
    ModernRunConfig,
    SampleAnalysisError,
    build_legacy_group_inputs,
    prepare_modern_analysis,
    run_compare_analysis,
    run_modern_analysis,
    validate_analysis_setup,
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
from main.gui.sample_analysis import SampleAnalysisWorkspace


class TestSampleAnalysisAdapters(unittest.TestCase):
    def test_setup_schema_rejects_future_versions_and_duplicate_sources(self):
        with self.assertRaisesRegex(SampleAnalysisError, "unsupported"):
            validate_analysis_setup({"schema": 99})
        with self.assertRaisesRegex(SampleAnalysisError, "duplicates"):
            validate_analysis_setup({"schema": 2, "mode": "Modern analysis",
                "source_keys": [("image-a", "first"), ("image-b", "first")]})

    def test_modern_plan_retains_explicit_small_from_fine_assignment(self):
        _, snapshot = self._snapshot_with_three_images()
        fine_first = ModernRunConfig(min_detectable_mm=0.001,
            nesting_segments=(("group-2", 0, 4), ("group-1", 4, 8)))
        coarse_first = ModernRunConfig(min_detectable_mm=0.001,
            nesting_segments=(("group-1", 0, 4), ("group-2", 4, 8)))
        fine = run_modern_analysis(snapshot, fine_first)
        coarse = run_modern_analysis(snapshot, coarse_first)
        self.assertEqual(fine.nesting_result.source_group_ids, ("group-2",) * 4 + ("group-1",) * 4)
        self.assertNotEqual(fine.nesting_result.counts, coarse.nesting_result.counts)
        for index, group_id in enumerate(fine.nesting_result.source_group_ids):
            native = fine.distribution_result.groups[(snapshot.project_id, group_id)]
            self.assertEqual(fine.nesting_result.counts[index], native.counts[index])
            self.assertEqual(fine.nesting_result.eligible_areas_mm2[index], native.eligible_areas_mm2[index])

    def test_unspecified_detection_preserves_measurements_and_none(self):
        _, snapshot = self._snapshot_with_three_images()
        prepared = prepare_modern_analysis(snapshot)
        self.assertGreater(len(prepared.measurements), 0)
        self.assertIsNotNone(prepared.distribution_error)
        self.assertIsNone(prepared.distribution_result)
        self.assertTrue(all(image.min_detectable_diameter_mm is None for image in prepared.image_records))

    def test_modern_rejects_invalid_plan_instead_of_repairing(self):
        _, snapshot = self._snapshot_with_three_images()
        plans = (
            (("group-2", 0, 4.5), ("group-1", 4.5, 8)),
            (("group-2", -1, 4), ("group-1", 4, 8)),
            (("group-2", 0, 4), ("group-2", 4, 8)),
            (("group-2", 0, 4), ("group-1", 5, 8)),
            (("unknown", 0, 8),),
        )
        for segments in plans:
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                run_modern_analysis(snapshot, ModernRunConfig(min_detectable_mm=0.001, nesting_segments=segments))
        with self.assertRaisesRegex(SampleAnalysisError, "transition numbers alone"):
            run_modern_analysis(snapshot, ModernRunConfig(min_detectable_mm=0.001, transition_bins=(4,)))

    def test_reconstruction_requires_separate_assumption_selection(self):
        _, snapshot = self._snapshot_with_three_images()
        config = dict(min_detectable_mm=0.001, nesting_segments=(("group-2", 0, 4), ("group-1", 4, 8)),
                      enable_reconstruction=True)
        with self.assertRaisesRegex(SampleAnalysisError, "explicit selection"):
            run_modern_analysis(snapshot, ModernRunConfig(**config))
        result = run_modern_analysis(snapshot, ModernRunConfig(**config, upper_tail_assumption="zero_beyond_selected_upper_edge"))
        self.assertIsNotNone(result.reconstruction_result)

    def test_stable_legacy_source_keys_survive_reorder_and_rename(self):
        project, snapshot = self._snapshot_with_three_images()
        _, _, before = build_legacy_group_inputs(snapshot)
        mapping = tuple((item.image_id, item.source_key) for item in before)
        project.images.reverse()
        project.images[0].display_name = "renamed"
        _, _, after = build_legacy_group_inputs(create_analysis_snapshot(project), mapping)
        self.assertEqual([(item.image_id, item.source_key) for item in before],
                         [(item.image_id, item.source_key) for item in after])

    def test_modern_three_and_five_resolution_groups(self):
        for count in (3, 5):
            project = create_project("Resolutions")
            for index in range(count):
                image = add_image(project, np.zeros((40, 40), dtype=np.uint8), f"{index}.png")
                mask = np.zeros((40, 40), dtype=np.uint8)
                mask[4:10 + index, 4:10 + index] = 255
                update_draft(image, mask=mask, labels=None, settings={})
                accept_draft(image)
                set_analysis_included(image, True)
                set_calibration(image, 0.002 * (index + 1), "mm")
            snapshot = create_analysis_snapshot(project)
            segments = tuple((f"group-{count - index}", index, index + 1) for index in range(count))
            result = run_modern_analysis(snapshot, ModernRunConfig(min_detectable_mm=0.0001,
                bin_edges_mm=tuple(0.001 * 2 ** index for index in range(count + 1)), nesting_segments=segments))
            self.assertEqual(result.nesting_result.source_group_ids, tuple(segment[0] for segment in segments))

    def test_saving_modern_preserves_legacy_setup(self):
        project = create_project("Settings")
        workspace = SimpleNamespace(project=project, _mark_unsaved=None)
        legacy = LegacyRunConfig(minimum_diameter_px=3.0, manual_ranges=(("group-1", 0.1, 0.2),))
        SampleAnalysisWorkspace._save_setup(workspace, "Original FOAMS 1.0.5", legacy, None)
        saved_legacy = dict(project.analysis_setup["legacy"])
        SampleAnalysisWorkspace._save_setup(workspace, "Modern analysis", None, ModernRunConfig())
        self.assertEqual(project.analysis_setup["legacy"], saved_legacy)

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
        result = run_modern_analysis(
            snapshot,
            ModernRunConfig(min_detectable_mm=0.001, nesting_segments=(("group-2", 0, 4), ("group-1", 4, 8))),
        )

        self.assertGreaterEqual(len(result.bin_edges_mm), 2)
        self.assertGreaterEqual(len(result.nesting_result.selected_bin_indices), 1)
        self.assertEqual(result.nesting_result.sample_id, snapshot.project_id)

    def test_run_compare_analysis_preserves_partial_success(self):
        _project, snapshot = self._snapshot_with_three_images()
        result = run_compare_analysis(
            snapshot,
            legacy_config=LegacyRunConfig(minimum_diameter_px=-1.0),
            modern_config=ModernRunConfig(min_detectable_mm=0.001, nesting_segments=(("group-2", 0, 4), ("group-1", 4, 8))),
        )

        self.assertIsInstance(result, CompareRunResult)
        self.assertIsNone(result.legacy_result)
        self.assertIsNotNone(result.legacy_error)
        self.assertIsNotNone(result.modern_result)
        self.assertIsNone(result.modern_error)


if __name__ == "__main__":
    unittest.main()
