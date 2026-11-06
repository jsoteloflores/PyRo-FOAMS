import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import sample_project_io
from core.sample_project import (
    SampleProjectError,
    accept_draft,
    add_image,
    calibration_mm_per_px,
    checksum_ndarray,
    create_project,
    included_images_missing_calibration,
    replace_working_image,
    set_analysis_included,
    set_calibration,
    update_draft,
)
from core.sample_project_io import (
    MANIFEST_FILENAME,
    SampleProjectIoError,
    clone_project_folder,
    open_project,
    save_project,
)


class TestSampleProjectModel(unittest.TestCase):
    def test_duplicate_basenames_get_stable_distinct_ids(self):
        project = create_project("Sample")
        img = np.zeros((12, 16), dtype=np.uint8)
        first = add_image(project, img, source_filename="a/path/dup.png", display_name="dup.png")
        second = add_image(project, img, source_filename="b/path/dup.png", display_name="dup.png")

        self.assertNotEqual(first.image_id, second.image_id)
        self.assertEqual(first.display_name, second.display_name)

    def test_acceptance_and_inclusion_are_independent(self):
        project = create_project("Sample")
        image = add_image(project, np.zeros((20, 20), dtype=np.uint8), source_filename="img.png")

        with self.assertRaises(SampleProjectError):
            set_analysis_included(image, True)

        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:12, 5:12] = 255
        update_draft(image, mask=mask, labels=None, settings={"method": "otsu"})
        accept_draft(image, source_kind="draft", source_details={"test": True})
        set_analysis_included(image, True)

        self.assertTrue(image.analysis_included)

    def test_reaccept_preserves_inclusion_and_replaces_snapshot(self):
        project = create_project("Sample")
        image = add_image(project, np.zeros((30, 30), dtype=np.uint8), source_filename="img.png")

        first_mask = np.zeros((30, 30), dtype=np.uint8)
        first_mask[2:6, 2:6] = 255
        update_draft(image, mask=first_mask, labels=None, settings={})
        first = accept_draft(image, source_kind="draft", source_details={})
        set_analysis_included(image, True)

        second_mask = np.zeros((30, 30), dtype=np.uint8)
        second_mask[15:25, 15:25] = 255
        update_draft(image, mask=second_mask, labels=np.ones((30, 30), dtype=np.int32), settings={})
        second = accept_draft(image, source_kind="draft", source_details={})

        self.assertTrue(image.analysis_included)
        self.assertNotEqual(first.snapshot_id, second.snapshot_id)
        self.assertTrue(np.array_equal(image.accepted_mask, second_mask))
        # Accepted labels are derived from mask, not stale draft labels.
        self.assertEqual(int(image.accepted_labels[0, 0]), 0)

    def test_draft_changes_do_not_mutate_accepted_snapshot(self):
        project = create_project("Sample")
        image = add_image(project, np.zeros((20, 20), dtype=np.uint8), source_filename="img.png")

        accepted_mask = np.zeros((20, 20), dtype=np.uint8)
        accepted_mask[1:5, 1:5] = 255
        update_draft(image, mask=accepted_mask, labels=None, settings={})
        accept_draft(image, source_kind="draft", source_details={})

        draft_mask = np.zeros((20, 20), dtype=np.uint8)
        draft_mask[10:15, 10:15] = 255
        update_draft(image, mask=draft_mask, labels=None, settings={})

        self.assertTrue(np.array_equal(image.accepted_mask, accepted_mask))
        self.assertFalse(np.array_equal(image.accepted_mask, image.draft_mask))

    def test_calibration_conversion_and_invalid_units(self):
        project = create_project("Sample")
        image = add_image(project, np.zeros((10, 10), dtype=np.uint8), source_filename="img.png")

        set_calibration(image, 2.0, "um")
        self.assertAlmostEqual(calibration_mm_per_px(image), 0.002)

        set_calibration(image, 1.0, "unknown_unit")
        self.assertIsNone(calibration_mm_per_px(image))
        with self.assertRaises(SampleProjectError):
            set_calibration(image, True, "mm")

    def test_included_images_report_missing_calibration(self):
        project = create_project("Sample")
        image = add_image(project, np.zeros((8, 8), dtype=np.uint8), source_filename="img.png")
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        update_draft(image, mask=mask, labels=None, settings={})
        accept_draft(image, source_kind="draft", source_details={})
        set_analysis_included(image, True)

        missing = included_images_missing_calibration(project)
        self.assertEqual([item.image_id for item in missing], [image.image_id])

    def test_crop_preserves_original_source_and_records_roi(self):
        project = create_project("Sample")
        source = np.arange(100, dtype=np.uint8).reshape(10, 10)
        image = add_image(project, source, source_filename="img.png")
        cropped = source[2:8, 3:9].copy()

        replace_working_image(image, cropped, (3, 2, 9, 8))

        np.testing.assert_array_equal(image.source_image, source)
        np.testing.assert_array_equal(image.working_image, cropped)
        self.assertEqual(image.processing_roi, (3, 2, 9, 8))


class TestSampleProjectPersistence(unittest.TestCase):
    def test_save_and_reopen_preserves_ids_and_accepted_masks(self):
        project = create_project("Rock Sample")
        image1 = np.zeros((24, 24), dtype=np.uint8)
        image2 = np.zeros((24, 24), dtype=np.uint8)
        image2[4:12, 4:12] = 80
        rec1 = add_image(project, image1, source_filename="c:/images/dup.png", display_name="dup.png")
        rec2 = add_image(project, image2, source_filename="d:/images/dup.png", display_name="dup.png")

        mask1 = np.zeros((24, 24), dtype=np.uint8)
        mask1[1:8, 1:8] = 255
        mask2 = np.zeros((24, 24), dtype=np.uint8)
        mask2[10:20, 10:20] = 255
        labels2 = np.zeros((24, 24), dtype=np.int32)
        labels2[10:15, 10:20] = 7
        labels2[15:20, 10:20] = 12
        update_draft(rec1, mask=mask1, labels=None, settings={"method": "adaptive"})
        update_draft(rec2, mask=mask2, labels=labels2, settings={"method": "otsu"})
        accept_draft(rec1, source_kind="draft", source_details={"stage": "test"})
        accept_draft(rec2, source_kind="draft", source_details={"stage": "test"})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            save_project(project, root)
            loaded = open_project(root)

            self.assertEqual([img.image_id for img in loaded.images], [rec1.image_id, rec2.image_id])
            self.assertTrue(np.array_equal(loaded.images[0].accepted_mask, mask1))
            self.assertTrue(np.array_equal(loaded.images[1].accepted_mask, mask2))
            np.testing.assert_array_equal(loaded.images[1].accepted_labels, labels2)

    def test_reopen_after_moving_project_folder(self):
        project = create_project("Portable")
        image = add_image(project, np.zeros((16, 16), dtype=np.uint8), source_filename="missing/original/path.png")
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[3:9, 3:9] = 255
        update_draft(image, mask=mask, labels=None, settings={})
        accept_draft(image, source_kind="draft", source_details={})

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source_project"
            moved = Path(directory) / "moved_project"
            save_project(project, source)
            clone_project_folder(source, moved)
            loaded = open_project(moved)
            self.assertEqual(loaded.sample_name, "Portable")
            self.assertTrue(np.array_equal(loaded.images[0].accepted_mask, mask))

    def test_cropped_project_reopens_with_original_and_working_pixels(self):
        project = create_project("Cropped")
        source = np.arange(400, dtype=np.uint8).reshape(20, 20)
        image = add_image(project, source, source_filename="no-longer-available.png")
        cropped = source[4:16, 3:15].copy()
        replace_working_image(image, cropped, (3, 4, 15, 16))
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[2:8, 2:8] = 255
        update_draft(image, mask=mask, labels=None, settings={})
        accept_draft(image)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cropped_project"
            save_project(project, root)
            loaded = open_project(root)
            np.testing.assert_array_equal(loaded.images[0].source_image, source)
            np.testing.assert_array_equal(loaded.images[0].working_image, cropped)
            np.testing.assert_array_equal(loaded.images[0].accepted_mask, mask)
            self.assertEqual(loaded.images[0].processing_roi, (3, 4, 15, 16))
            self.assertEqual(
                loaded.images[0].accepted_snapshot.working_image_checksum_sha256,
                checksum_ndarray(cropped),
            )
            self.assertEqual(loaded.images[0].accepted_snapshot.processing_roi, (3, 4, 15, 16))

    def test_interrupted_save_leaves_previous_manifest_and_assets_usable(self):
        project = create_project("Before")
        source = np.arange(400, dtype=np.uint8).reshape(20, 20)
        image = add_image(project, source, source_filename="original.png")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            save_project(project, root)
            manifest_before = (root / MANIFEST_FILENAME).read_bytes()
            replace_working_image(image, source[2:18, 1:17], (1, 2, 17, 18))
            project.sample_name = "After"

            real_write = sample_project_io._write_image
            writes = 0

            def fail_second_write(path, array):
                nonlocal writes
                writes += 1
                if writes == 2:
                    raise SampleProjectIoError("simulated interrupted asset write")
                return real_write(path, array)

            with patch("core.sample_project_io._write_image", side_effect=fail_second_write):
                with self.assertRaises(SampleProjectIoError):
                    save_project(project, root)

            self.assertEqual((root / MANIFEST_FILENAME).read_bytes(), manifest_before)
            loaded = open_project(root)
            self.assertEqual(loaded.sample_name, "Before")
            np.testing.assert_array_equal(loaded.images[0].working_image, source)

    def test_asset_path_cannot_escape_project_folder(self):
        project = create_project("Path Check")
        add_image(project, np.zeros((8, 8), dtype=np.uint8), source_filename="img.png")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            save_project(project, root)
            manifest_path = root / MANIFEST_FILENAME
            payload = json_load(manifest_path)
            payload["images"][0]["working_asset_path"] = "../../outside.png"
            manifest_path.write_text(to_json(payload), encoding="utf-8")
            with self.assertRaisesRegex(SampleProjectIoError, "escapes project folder"):
                open_project(root)

    def test_manifest_rejects_non_boolean_inclusion(self):
        project = create_project("Boolean Check")
        add_image(project, np.zeros((8, 8), dtype=np.uint8), source_filename="img.png")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            save_project(project, root)
            manifest_path = root / MANIFEST_FILENAME
            payload = json_load(manifest_path)
            payload["images"][0]["analysis_included"] = "false"
            manifest_path.write_text(to_json(payload), encoding="utf-8")
            with self.assertRaisesRegex(SampleProjectIoError, "must be Boolean"):
                open_project(root)

    def test_missing_asset_and_unsupported_schema_raise_clear_errors(self):
        project = create_project("Broken")
        image = add_image(project, np.zeros((8, 8), dtype=np.uint8), source_filename="x.png")
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:4, 1:4] = 255
        update_draft(image, mask=mask, labels=None, settings={})
        accept_draft(image, source_kind="draft", source_details={})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            save_project(project, root)

            manifest_path = root / MANIFEST_FILENAME
            payload = json_load(manifest_path)
            payload["schema_version"] = 999
            manifest_path.write_text(to_json(payload), encoding="utf-8")
            with self.assertRaises(SampleProjectIoError):
                open_project(root)

            payload["schema_version"] = 1
            payload["images"][0]["source_asset_path"] = "assets/sources/missing.png"
            manifest_path.write_text(to_json(payload), encoding="utf-8")
            with self.assertRaises(SampleProjectIoError):
                open_project(root)


def json_load(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def to_json(payload) -> str:
    import json

    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


if __name__ == "__main__":
    unittest.main()
