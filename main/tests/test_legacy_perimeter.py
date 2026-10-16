from __future__ import annotations

import csv
import json
import math
import unittest
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np

from main.core.legacy_measurements import measure_foams105_phase
from main.core.legacy_perimeter import (
    FOAMS105_PERIMETER_PROFILE_CURRENT,
    FOAMS105_PERIMETER_PROFILE_OLD,
    FOAMS105_PERIMETER_SELECTION_METHOD,
    Foams105PerimeterNumericalError,
    Foams105PerimeterValidationError,
    measure_foams105_perimeter,
    select_foams105_perimeter,
)
from main.core.legacy_raw_selection import (
    Foams105RawGroupInput,
    Foams105RawObject,
    select_foams105_raw_objects,
)
from main.core.legacy_selection import Foams105BinnedRange

CATALOG_PATH = Path(__file__).parent / "fixtures" / "16a_foams105_perimeter_cases.json"
EVIDENCE_ROOT = Path(__file__).parent / "fixtures" / "16b_perimeter_evidence"
REFERENCE_COMPONENTS_PATH = EVIDENCE_ROOT / "reference_capture" / "components.tsv"
SUPPLEMENTARY_CASES_PATH = EVIDENCE_ROOT / "review_supplementary_cases.json"
SUPPLEMENTARY_TSV_PATH = EVIDENCE_ROOT / "review_supplementary.tsv"

REFERENCE_COMPONENTS_LF_SHA256 = "3447fd9b019d438d0649dc41956efc12f96a09be9bf41ae719b9e385ef0acdcf"
SUPPLEMENTARY_CASES_LF_SHA256 = "d652f8966c599586a58e1d752362122439717758eefc2d4ccbd44e0b7e068967"
SUPPLEMENTARY_TSV_LF_SHA256 = "0794d43c2f38c05ad3dfb4c4605f6fc2e11fce894ff15dd9b112e17b6df47a89"


def _image_from_pixels(shape, pixels):
    image = np.zeros(shape, dtype=np.uint8)
    for row, col in pixels:
        image[row, col] = 1
    return image


def _raw_object(image_id, component):
    return Foams105RawObject(
        image_id=image_id,
        component_id=component.component_id,
        equivalent_diameter_mm=component.equivalent_diameter_mm,
        area_mm2=component.area_mm2,
    )


class TestFoams105PerimeterReferenceCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        cls.cases_by_id = {case["id"]: case for case in cls.catalog["cases"]}
        for case in cls.catalog["cases"]:
            if case["stage"] == "raw_phase_mask":
                image = _image_from_pixels(tuple(case["shape"]), case["foreground_pixels"])
                labels = np.zeros(tuple(case["shape"]), dtype=np.int32)
                measured = measure_foams105_phase(image, int(case["phase_value"]), 1.0)
                labels[:, :] = np.asarray(measured.label_image, dtype=np.int32)
            else:
                labels = np.zeros(tuple(case["shape"]), dtype=np.int32)
                for row, col, label in case["label_entries"]:
                    labels[int(row), int(col)] = int(label)
            case["_labels"] = labels

        cls.reference_rows = []
        with REFERENCE_COMPONENTS_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row["perimeter_status"] != "success" or row["perimeter_old_status"] != "success":
                    continue
                cls.reference_rows.append(
                    {
                        "case_id": row["case_id"],
                        "component_id": int(row["native_component_id"]),
                        "perimeter_px": float(row["perimeter_px"]),
                        "perimeter_old_px": float(row["perimeter_old_px"]),
                    }
                )

    def _canonical_measurement_for_case(self, case_id):
        case = self.cases_by_id[case_id]
        if case["stage"] == "raw_phase_mask":
            image = _image_from_pixels(tuple(case["shape"]), case["foreground_pixels"])
            return measure_foams105_phase(image, int(case["phase_value"]), 1.0)

        labels = np.asarray(case["_labels"], dtype=np.int32)
        image = (labels > 0).astype(np.uint8)
        measured = measure_foams105_phase(image, 1, 1.0)
        if not np.array_equal(np.asarray(measured.label_image, dtype=np.int32), labels):
            return None
        return measured

    def test_reference_capture_matches_both_profiles(self):
        matched = 0
        skipped_cases = set()
        measurements_by_case = {}
        for row in self.reference_rows:
            case_id = row["case_id"]
            if case_id not in measurements_by_case:
                measurements_by_case[case_id] = self._canonical_measurement_for_case(case_id)

            measurement = measurements_by_case[case_id]
            if measurement is None:
                skipped_cases.add(case_id)
                continue

            current = measure_foams105_perimeter(
                case_id,
                measurement,
                profile=FOAMS105_PERIMETER_PROFILE_CURRENT,
            )
            old = measure_foams105_perimeter(
                case_id,
                measurement,
                profile=FOAMS105_PERIMETER_PROFILE_OLD,
            )

            current_component = next(item for item in current.components if item.component_id == row["component_id"])
            old_component = next(item for item in old.components if item.component_id == row["component_id"])
            self.assertTrue(
                math.isclose(current_component.perimeter_px, row["perimeter_px"], rel_tol=1e-12, abs_tol=1e-12)
            )
            self.assertTrue(
                math.isclose(old_component.perimeter_px, row["perimeter_old_px"], rel_tol=1e-12, abs_tol=1e-12)
            )
            matched += 1

        self.assertEqual(skipped_cases, {"narrow_neck_final"})
        self.assertEqual(matched, 30)

    def test_reference_capture_hashes_are_pinned(self):
        components_digest = sha256(REFERENCE_COMPONENTS_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        supplementary_cases_digest = sha256(
            SUPPLEMENTARY_CASES_PATH.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        supplementary_tsv_digest = sha256(SUPPLEMENTARY_TSV_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        self.assertEqual(components_digest, REFERENCE_COMPONENTS_LF_SHA256)
        self.assertEqual(supplementary_cases_digest, SUPPLEMENTARY_CASES_LF_SHA256)
        self.assertEqual(supplementary_tsv_digest, SUPPLEMENTARY_TSV_LF_SHA256)


class TestFoams105PerimeterSupplementaryEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.supp_cases = json.loads(SUPPLEMENTARY_CASES_PATH.read_text(encoding="utf-8"))
        cls.supp_rows = {}
        with SUPPLEMENTARY_TSV_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                tokens = line.rstrip("\n").split("\t")
                if not tokens or tokens[0] != "P":
                    continue
                case_id = int(tokens[1])
                component_id = int(tokens[2])
                current = float(tokens[3])
                old = float(tokens[4])
                cls.supp_rows[(case_id, component_id)] = (current, old)

    def test_supplementary_labels_and_profiles_match(self):
        component_rows = 0
        for case in self.supp_cases:
            image = np.asarray(case["raw"], dtype=np.uint8)
            expected_labels = np.asarray(case["expected_labels"], dtype=np.int32)
            case_id = int(case["id"])
            measured = measure_foams105_phase(image, 1, 1.0)
            self.assertTrue(np.array_equal(np.asarray(measured.label_image, dtype=np.int32), expected_labels))

            current = measure_foams105_perimeter(
                f"supp-{case_id}",
                measured,
                profile=FOAMS105_PERIMETER_PROFILE_CURRENT,
            )
            old = measure_foams105_perimeter(
                f"supp-{case_id}",
                measured,
                profile=FOAMS105_PERIMETER_PROFILE_OLD,
            )
            old_by_id = {item.component_id: item for item in old.components}
            for component in current.components:
                expected_current, expected_old = self.supp_rows[(case_id, component.component_id)]
                self.assertTrue(
                    math.isclose(component.perimeter_px, expected_current, rel_tol=1e-12, abs_tol=1e-12),
                    (case_id, component.component_id, component.perimeter_px, expected_current),
                )
                self.assertTrue(
                    math.isclose(
                        old_by_id[component.component_id].perimeter_px,
                        expected_old,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ),
                    (case_id, component.component_id, old_by_id[component.component_id].perimeter_px, expected_old),
                )
                component_rows += 1

        self.assertEqual(component_rows, 526)


class TestFoams105PerimeterProfilesAndValidation(unittest.TestCase):
    def test_profile_is_explicit_and_supported(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        image[2, 2] = 1
        measurement = measure_foams105_phase(image, 1, 1.0)
        with self.assertRaises(TypeError):
            measure_foams105_perimeter("missing-profile", measurement)  # type: ignore[call-arg]
        with self.assertRaises(Foams105PerimeterValidationError):
            measure_foams105_perimeter("bad-profile", measurement, profile="legacy")

    def test_known_shapes_match_expected_values(self):
        cases = (
            ("single", ((3, 3),), 0.0, 0.0),
            ("h2", ((3, 3), (3, 4)), 2.0, 1.96),
            ("h3", ((3, 2), (3, 3), (3, 4)), 4.0, 3.92),
            ("square2", ((2, 2), (2, 3), (3, 2), (3, 3)), 4.0, 3.556),
            ("plus", ((3, 4), (4, 3), (4, 4), (4, 5), (5, 4)), 4.0 * math.sqrt(2.0), 5.624),
        )
        for case_id, pixels, expected_old, expected_current in cases:
            image = _image_from_pixels((9, 9), pixels)
            measurement = measure_foams105_phase(image, 1, 1.0)
            old = measure_foams105_perimeter(case_id, measurement, profile=FOAMS105_PERIMETER_PROFILE_OLD)
            current = measure_foams105_perimeter(case_id, measurement, profile=FOAMS105_PERIMETER_PROFILE_CURRENT)
            self.assertEqual(len(old.components), 1)
            self.assertEqual(len(current.components), 1)
            self.assertTrue(math.isclose(old.components[0].perimeter_px, expected_old, rel_tol=1e-12, abs_tol=1e-12))
            self.assertTrue(
                math.isclose(current.components[0].perimeter_px, expected_current, rel_tol=1e-12, abs_tol=1e-12)
            )

    def test_valid_narrow_neck_and_raw_mask_topology(self):
        image = np.zeros((13, 13), dtype=np.uint8)
        image[4:6, 3:5] = 1
        image[4, 5] = 1
        image[4:6, 6:8] = 1
        narrow_neck = measure_foams105_phase(image, 1, 1.0)
        narrow_result = measure_foams105_perimeter(
            "narrow-neck",
            narrow_neck,
            profile=FOAMS105_PERIMETER_PROFILE_CURRENT,
        )
        self.assertEqual(narrow_result.component_count, 1)

        raw = np.zeros((13, 13), dtype=np.uint8)
        raw[0, 0] = 1
        raw[4:7, 4] = 1
        raw[4:7, 6] = 1
        raw[4, 4:7] = 1
        raw[6, 4:7] = 1
        raw[9, 9] = 1
        raw[10, 10] = 1
        measured = measure_foams105_phase(raw, 1, 1.0)
        perimeter = measure_foams105_perimeter(
            "raw-topology",
            measured,
            profile=FOAMS105_PERIMETER_PROFILE_CURRENT,
        )
        self.assertTrue(measured.filled_mask[5, 5])
        self.assertEqual(measured.removed_border_pixel_count, 1)
        self.assertEqual(perimeter.component_count, 3)
        self.assertEqual(sum(component.perimeter_px == 0.0 for component in perimeter.components), 2)

    def test_rejects_malformed_measurement_scalars_and_arrays(self):
        image = _image_from_pixels((7, 7), ((3, 3), (3, 4)))
        measurement = measure_foams105_phase(image, 1, 1.0)

        malformed_scale = replace(measurement, scale_px_per_mm=[[1.0]])
        with self.assertRaises(Foams105PerimeterValidationError) as caught:
            measure_foams105_perimeter(
                "nested-scale",
                malformed_scale,
                profile=FOAMS105_PERIMETER_PROFILE_CURRENT,
            )
        self.assertEqual(caught.exception.code, "inconsistent_measurement_result")

        oversized_scale = replace(measurement, scale_px_per_mm=10**400)
        with self.assertRaises(Foams105PerimeterValidationError) as caught:
            measure_foams105_perimeter(
                "oversized-scale",
                oversized_scale,
                profile=FOAMS105_PERIMETER_PROFILE_CURRENT,
            )
        self.assertEqual(caught.exception.code, "inconsistent_measurement_result")

        malformed_labels = replace(measurement, label_image=measurement.label_image.astype(np.float64))
        with self.assertRaises(Foams105PerimeterValidationError) as caught:
            measure_foams105_perimeter(
                "float-labels",
                malformed_labels,
                profile=FOAMS105_PERIMETER_PROFILE_CURRENT,
            )
        self.assertEqual(caught.exception.code, "inconsistent_measurement_result")

    def test_translation_rotation_reflection_and_calibration(self):
        base = _image_from_pixels((12, 12), ((2, 2), (2, 3), (3, 2), (3, 3), (4, 3)))
        moved = _image_from_pixels((12, 12), ((6, 7), (6, 8), (7, 7), (7, 8), (8, 8)))
        rot_h = _image_from_pixels((12, 12), ((4, 3), (4, 4), (4, 5), (4, 6)))
        rot_v = _image_from_pixels((12, 12), ((3, 4), (4, 4), (5, 4), (6, 4)))

        base_current = measure_foams105_perimeter("base", measure_foams105_phase(base, 1, 1.0), profile=FOAMS105_PERIMETER_PROFILE_CURRENT).components[0]
        moved_current = measure_foams105_perimeter("moved", measure_foams105_phase(moved, 1, 1.0), profile=FOAMS105_PERIMETER_PROFILE_CURRENT).components[0]
        self.assertTrue(math.isclose(base_current.perimeter_px, moved_current.perimeter_px, rel_tol=1e-12, abs_tol=1e-12))

        rot_h_current = measure_foams105_perimeter("rot-h", measure_foams105_phase(rot_h, 1, 1.0), profile=FOAMS105_PERIMETER_PROFILE_CURRENT).components[0]
        rot_v_current = measure_foams105_perimeter("rot-v", measure_foams105_phase(rot_v, 1, 1.0), profile=FOAMS105_PERIMETER_PROFILE_CURRENT).components[0]
        self.assertTrue(math.isclose(rot_h_current.perimeter_px, rot_v_current.perimeter_px, rel_tol=1e-12, abs_tol=1e-12))

        scaled = measure_foams105_perimeter(
            "scaled",
            measure_foams105_phase(base, 1, 4.0),
            profile=FOAMS105_PERIMETER_PROFILE_CURRENT,
        ).components[0]
        self.assertTrue(math.isclose(base_current.perimeter_px, scaled.perimeter_px, rel_tol=1e-12, abs_tol=1e-12))
        self.assertTrue(math.isclose(base_current.perimeter_mm / 4.0, scaled.perimeter_mm, rel_tol=1e-12, abs_tol=1e-12))

    def test_rejects_disconnected_or_holey_same_id_component(self):
        base = _image_from_pixels((9, 9), ((3, 3), (3, 4)))
        measurement = measure_foams105_phase(base, 1, 1.0)
        component = measurement.components[0]

        disconnected_labels = np.zeros((9, 9), dtype=np.int32)
        disconnected_labels[3, 3] = 1
        disconnected_labels[5, 5] = 1
        disconnected = replace(
            measurement,
            label_image=disconnected_labels,
            filled_mask=(disconnected_labels > 0),
            final_foreground_pixel_count=2,
            components=(
                replace(
                    component,
                    area_px=2,
                    first_pixel_rc=(3, 3),
                    area_mm2=2.0,
                    equivalent_diameter_mm=2.0 * math.sqrt(2.0 / math.pi),
                ),
            ),
        )
        with self.assertRaises(Foams105PerimeterValidationError) as caught:
            measure_foams105_perimeter("disconnected", disconnected, profile=FOAMS105_PERIMETER_PROFILE_CURRENT)
        self.assertEqual(caught.exception.code, "invalid_component_topology")

        ring_image = np.zeros((11, 11), dtype=np.uint8)
        ring_image[4:7, 4] = 1
        ring_image[4:7, 6] = 1
        ring_image[4, 4:7] = 1
        ring_image[6, 4:7] = 1
        ring_measurement = measure_foams105_phase(ring_image, 1, 1.0)
        ring_component = ring_measurement.components[0]
        ring_labels = np.zeros_like(ring_measurement.label_image, dtype=np.int32)
        ring_labels[4:7, 4] = 1
        ring_labels[4:7, 6] = 1
        ring_labels[4, 4:7] = 1
        ring_labels[6, 4:7] = 1
        tampered_ring = replace(
            ring_measurement,
            label_image=ring_labels,
            filled_mask=(ring_labels > 0),
            final_foreground_pixel_count=8,
            selected_pixel_count=8,
            retained_before_fill_pixel_count=8,
            filled_added_pixel_count=0,
            components=(
                replace(
                    ring_component,
                    area_px=8,
                    first_pixel_rc=(4, 4),
                    area_mm2=8.0,
                    equivalent_diameter_mm=2.0 * math.sqrt(8.0 / math.pi),
                ),
            ),
        )
        with self.assertRaises(Foams105PerimeterValidationError) as caught:
            measure_foams105_perimeter("holey", tampered_ring, profile=FOAMS105_PERIMETER_PROFILE_CURRENT)
        self.assertEqual(caught.exception.code, "invalid_component_topology")

    def test_nonclosure_error_is_typed(self):
        image = _image_from_pixels((8, 8), ((3, 3), (3, 4), (4, 3), (4, 4)))
        measurement = measure_foams105_phase(image, 1, 1.0)
        with patch(
            "main.core.legacy_perimeter._trace_component_boundary",
            side_effect=Foams105PerimeterNumericalError(
                "did not close",
                code="nonclosing_boundary_trace",
                context={"image_id": "x", "component_id": 1},
            ),
        ):
            with self.assertRaises(Foams105PerimeterNumericalError) as caught:
                measure_foams105_perimeter("x", measurement, profile=FOAMS105_PERIMETER_PROFILE_CURRENT)
            self.assertEqual(caught.exception.code, "nonclosing_boundary_trace")

    def test_many_components_grouped_traversal(self):
        image = np.zeros((64, 64), dtype=np.uint8)
        for index in range(0, 200):
            row = (index * 3) % 64
            col = (index * 5) % 64
            image[row, col] = 1
        measurement = measure_foams105_phase(image, 1, 1.0)
        result = measure_foams105_perimeter("many", measurement, profile=FOAMS105_PERIMETER_PROFILE_CURRENT)
        self.assertEqual(result.component_count, len(result.components))
        self.assertTrue(all(component.perimeter_px == 0.0 for component in result.components))


class TestFoams105PerimeterSelection(unittest.TestCase):
    def _build_raw(self):
        image_a = np.zeros((10, 10), dtype=np.uint8)
        image_a[2, 2] = 1
        image_a[2:4, 6:8] = 1
        image_a[6:9, 2:5] = 1
        measurement_a = measure_foams105_phase(image_a, 1, 1.0)
        perimeter_a = measure_foams105_perimeter(
            "image-a",
            measurement_a,
            profile=FOAMS105_PERIMETER_PROFILE_CURRENT,
        )

        image_b = np.zeros((10, 10), dtype=np.uint8)
        image_b[2:5, 2] = 1
        image_b[6:8, 6:8] = 1
        measurement_b = measure_foams105_phase(image_b, 1, 1.0)
        perimeter_b = measure_foams105_perimeter(
            "image-b",
            measurement_b,
            profile=FOAMS105_PERIMETER_PROFILE_CURRENT,
        )

        retained = (
            _raw_object("image-a", measurement_a.components[1]),
            _raw_object("image-a", measurement_a.components[2]),
            _raw_object("image-b", measurement_b.components[0]),
            _raw_object("image-b", measurement_b.components[1]),
        )
        raw = select_foams105_raw_objects(
            (1.0, 2.0, 3.0, 4.0),
            (Foams105RawGroupInput("g", 1, retained),),
            (Foams105BinnedRange("g", 1.0, 4.0),),
        )
        return raw, perimeter_a, perimeter_b

    def test_selection_join_and_immutability(self):
        raw, perimeter_a, perimeter_b = self._build_raw()
        result = select_foams105_perimeter(raw, (perimeter_a, perimeter_b))
        self.assertEqual(result.method, FOAMS105_PERIMETER_SELECTION_METHOD)
        self.assertEqual(result.profile, FOAMS105_PERIMETER_PROFILE_CURRENT)
        self.assertEqual(result.selected_count, len(result.rows))
        self.assertEqual(
            tuple((row.image_id, row.component_id) for row in result.rows),
            tuple((obj.image_id, obj.component_id) for obj in raw.selected_objects),
        )
        with self.assertRaises(FrozenInstanceError):
            result.rows[0].perimeter_px = 1.0  # type: ignore[misc]

    def test_selection_rejects_mixed_profiles_and_metadata_tampering(self):
        raw, perimeter_a, perimeter_b = self._build_raw()
        old_b = measure_foams105_perimeter(
            "image-b",
            perimeter_b.phase_measurement_result,
            profile=FOAMS105_PERIMETER_PROFILE_OLD,
        )
        with self.assertRaises(Foams105PerimeterValidationError) as caught:
            select_foams105_perimeter(raw, (perimeter_a, old_b))
        self.assertEqual(caught.exception.code, "mixed_perimeter_profiles")

        bad = replace(
            perimeter_a,
            components=(replace(perimeter_a.components[0], area_mm2=perimeter_a.components[0].area_mm2 + 0.5),)
            + perimeter_a.components[1:],
        )
        with self.assertRaises(Foams105PerimeterValidationError) as caught:
            select_foams105_perimeter(raw, (bad, perimeter_b))
        self.assertEqual(caught.exception.code, "inconsistent_perimeter_result")

    def test_selection_replays_once_per_supplied_image(self):
        raw, perimeter_a, perimeter_b = self._build_raw()
        extra = measure_foams105_perimeter(
            "extra",
            measure_foams105_phase(np.pad(np.eye(3, dtype=np.uint8), 2), 1, 1.0),
            profile=FOAMS105_PERIMETER_PROFILE_CURRENT,
        )
        with patch(
            "main.core.legacy_perimeter.measure_foams105_perimeter",
            wraps=measure_foams105_perimeter,
        ) as wrapped:
            select_foams105_perimeter(raw, (perimeter_a, perimeter_b, extra))
            self.assertEqual(wrapped.call_count, 3)

    def test_rejected_raw_objects_still_require_perimeter_identities(self):
        raw, perimeter_a, _ = self._build_raw()
        selected_from_a = select_foams105_raw_objects(
            raw.bin_labels_mm,
            raw.groups,
            (Foams105BinnedRange("g", 1.0, 2.0),),
        )
        with self.assertRaises(Foams105PerimeterValidationError) as caught:
            select_foams105_perimeter(selected_from_a, (perimeter_a,))
        self.assertEqual(caught.exception.code, "missing_perimeter_identity")


if __name__ == "__main__":
    unittest.main()
