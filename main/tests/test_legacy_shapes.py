from __future__ import annotations

import json
import math
import unittest
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from main.core.legacy_measurements import measure_foams105_phase
from main.core.legacy_raw_selection import (
    Foams105RawGroupInput,
    Foams105RawObject,
    select_foams105_raw_objects,
)
from main.core.legacy_selection import Foams105BinnedRange
from main.core.legacy_shapes import (
    FOAMS105_MOMENT_METHOD,
    FOAMS105_MOMENT_SELECTION_METHOD,
    FOAMS105_MOMENT_VERIFICATION_LEVEL,
    FOAMS105_ORIENTATION_TIE_REASON,
    FOAMS105_PERIMETER_STATUS_PENDING,
    Foams105MomentValidationError,
    measure_foams105_moment_geometry,
    select_foams105_moment_geometry,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "16_foams105_moment_cases.json"
FIXTURE_LF_SHA256 = "d5960a11cb8450d5e4bb58c00a320a888dba9f30d74aca4e31d0422caabe44d8"


def _build_image(shape, pixels):
    image = np.zeros(shape, dtype=np.uint8)
    for row, column in pixels:
        image[row, column] = 1
    return image


def _object_from_component(image_id, component):
    return Foams105RawObject(
        image_id,
        component.component_id,
        component.equivalent_diameter_mm,
        component.area_mm2,
    )


def _assert_close(test_case, actual, expected):
    test_case.assertTrue(
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"{actual!r} != {expected!r}",
    )


def _assert_unoriented_angle_equal(test_case, actual, expected):
    actual_norm = 90.0 if actual == -90.0 else actual
    expected_norm = 90.0 if expected == -90.0 else expected
    delta = ((actual_norm - expected_norm + 90.0) % 180.0) - 90.0
    test_case.assertTrue(
        abs(delta) <= 1e-12,
        f"unoriented angles differ: {actual_norm!r} vs {expected_norm!r}",
    )


def _independent_moment_profile(rows, cols):
    x = [float(column) for column in cols]
    y = [-float(row) for row in rows]
    count = float(len(x))
    xbar = sum(x) / count
    ybar = sum(y) / count
    cxx = (sum((value - xbar) ** 2 for value in x) / count) + (1.0 / 12.0)
    cyy = (sum((value - ybar) ** 2 for value in y) / count) + (1.0 / 12.0)
    cxy = sum((value_x - xbar) * (value_y - ybar) for value_x, value_y in zip(x, y)) / count
    delta = math.hypot(cxx - cyy, 2.0 * cxy)
    lambda_major = (cxx + cyy + delta) / 2.0
    lambda_minor = (cxx + cyy - delta) / 2.0
    major = 4.0 * math.sqrt(lambda_major)
    minor = 4.0 * math.sqrt(lambda_minor)
    ecc = math.sqrt(1.0 - (lambda_minor / lambda_major))
    if cxx == cyy and cxy == 0.0:
        theta = 0.0
        defined = False
    else:
        theta = math.degrees(0.5 * math.atan2(2.0 * cxy, cxx - cyy))
        if cxy == 0.0 and cxx < cyy:
            theta = 90.0
        if theta == -90.0:
            theta = 90.0
        defined = True
    return {
        "centroid_x": xbar,
        "centroid_y": ybar,
        "cxx": cxx,
        "cxy": cxy,
        "cyy": cyy,
        "major": major,
        "minor": minor,
        "eccentricity": ecc,
        "theta": theta,
        "defined": defined,
    }


class TestFoams105MomentFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_fixture_cases_match_analytical_profile(self):
        self.assertEqual(len(self.fixture["cases"]), 8)
        for case in self.fixture["cases"]:
            image = _build_image(tuple(case["shape"]), case["pixels"])
            measurement = measure_foams105_phase(image, 1, case["scale"])
            result = measure_foams105_moment_geometry(case["id"], measurement)
            with self.subTest(case=case["id"]):
                self.assertEqual(result.method, FOAMS105_MOMENT_METHOD)
                self.assertEqual(result.verification_level, FOAMS105_MOMENT_VERIFICATION_LEVEL)
                self.assertEqual(result.perimeter_status, FOAMS105_PERIMETER_STATUS_PENDING)
                self.assertEqual(len(result.components), len(case["components"]))
                for component, expected in zip(result.components, case["components"]):
                    self.assertEqual(component.component_id, expected["id"])
                    self.assertEqual(component.area_px, expected["area"])
                    _assert_close(self, component.central_cxx, expected["cxx"])
                    _assert_close(self, component.central_cxy, expected["cxy"])
                    _assert_close(self, component.central_cyy, expected["cyy"])
                    _assert_close(self, component.major_axis_px, expected["major_px"])
                    _assert_close(self, component.minor_axis_px, expected["minor_px"])
                    _assert_close(self, component.eccentricity, expected["eccentricity"])
                    _assert_close(
                        self,
                        component.orientation_regionprops_profile_deg,
                        expected["theta"],
                    )
                    _assert_close(
                        self,
                        component.orientation_project_deg,
                        expected["theta"] + 90.0,
                    )
                    self.assertEqual(component.orientation_defined, expected["defined"])
                    if expected["defined"]:
                        self.assertIsNone(component.orientation_tie_reason)
                    else:
                        self.assertEqual(
                            component.orientation_tie_reason,
                            FOAMS105_ORIENTATION_TIE_REASON,
                        )
                    _assert_close(
                        self,
                        component.major_axis_mm,
                        expected["major_px"] / case["scale"],
                    )
                    _assert_close(
                        self,
                        component.minor_axis_mm,
                        expected["minor_px"] / case["scale"],
                    )

    def test_fixture_hash_is_pinned(self):
        normalized = FIXTURE_PATH.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(sha256(normalized).hexdigest(), FIXTURE_LF_SHA256)


class TestFoams105MomentInvarianceAndTopology(unittest.TestCase):
    def test_translation_preserves_tensor_axes_and_eccentricity(self):
        shape = (12, 12)
        pixels = ((3, 3), (3, 4), (4, 3))
        shifted = tuple((row + 3, column + 2) for row, column in pixels)
        first = measure_foams105_moment_geometry(
            "base", measure_foams105_phase(_build_image(shape, pixels), 1, 1.0)
        ).components[0]
        second = measure_foams105_moment_geometry(
            "shifted", measure_foams105_phase(_build_image(shape, shifted), 1, 1.0)
        ).components[0]
        for field in (
            "central_cxx",
            "central_cxy",
            "central_cyy",
            "major_axis_px",
            "minor_axis_px",
            "eccentricity",
            "orientation_regionprops_profile_deg",
        ):
            _assert_close(self, getattr(first, field), getattr(second, field))

    def test_reflection_and_quarter_rotation_orientations(self):
        shape = (12, 12)
        positive = ((3, 3), (3, 4), (4, 3))
        reflected = ((3, 4), (3, 5), (4, 5))
        horizontal = ((5, 4), (5, 5), (5, 6))
        vertical = ((4, 5), (5, 5), (6, 5))

        positive_component = measure_foams105_moment_geometry(
            "positive", measure_foams105_phase(_build_image(shape, positive), 1, 1.0)
        ).components[0]
        reflected_component = measure_foams105_moment_geometry(
            "reflected", measure_foams105_phase(_build_image(shape, reflected), 1, 1.0)
        ).components[0]
        horizontal_component = measure_foams105_moment_geometry(
            "horizontal", measure_foams105_phase(_build_image(shape, horizontal), 1, 1.0)
        ).components[0]
        vertical_component = measure_foams105_moment_geometry(
            "vertical", measure_foams105_phase(_build_image(shape, vertical), 1, 1.0)
        ).components[0]

        _assert_close(
            self,
            reflected_component.orientation_regionprops_profile_deg,
            -positive_component.orientation_regionprops_profile_deg,
        )
        _assert_unoriented_angle_equal(
            self,
            horizontal_component.orientation_regionprops_profile_deg + 90.0,
            vertical_component.orientation_regionprops_profile_deg,
        )

    def test_calibration_scales_lengths_only(self):
        pixels = ((3, 3), (3, 4), (4, 3), (4, 4), (5, 4))
        image = _build_image((12, 12), pixels)
        first = measure_foams105_moment_geometry(
            "scale-1", measure_foams105_phase(image, 1, 1.0)
        ).components[0]
        second = measure_foams105_moment_geometry(
            "scale-4", measure_foams105_phase(image, 1, 4.0)
        ).components[0]
        _assert_close(self, second.major_axis_px, first.major_axis_px)
        _assert_close(self, second.minor_axis_px, first.minor_axis_px)
        _assert_close(self, second.major_axis_mm, first.major_axis_mm / 4.0)
        _assert_close(self, second.minor_axis_mm, first.minor_axis_mm / 4.0)
        _assert_close(self, second.area_mm2, first.area_mm2 / 16.0)
        _assert_close(self, second.eccentricity, first.eccentricity)
        _assert_close(
            self,
            second.orientation_regionprops_profile_deg,
            first.orientation_regionprops_profile_deg,
        )

    def test_hole_filling_border_removal_and_diagonal_connectivity(self):
        image = np.zeros((11, 11), dtype=np.uint8)
        image[0, 0] = 1
        image[4:7, 4] = 1
        image[4:7, 6] = 1
        image[4, 4:7] = 1
        image[6, 4:7] = 1
        ring_result = measure_foams105_phase(image, 1, 1.0)
        geometry = measure_foams105_moment_geometry("ring", ring_result)
        self.assertEqual(len(geometry.components), 1)
        self.assertEqual(geometry.components[0].area_px, 9)

        diagonal = np.zeros((7, 7), dtype=np.uint8)
        diagonal[2, 2] = 1
        diagonal[3, 3] = 1
        diagonal_geometry = measure_foams105_moment_geometry(
            "diagonal", measure_foams105_phase(diagonal, 1, 1.0)
        )
        self.assertEqual(tuple(component.component_id for component in diagonal_geometry.components), (1, 2))
        self.assertTrue(all(component.orientation_defined is False for component in diagonal_geometry.components))


class TestFoams105MomentValidationAndImmutability(unittest.TestCase):
    def test_empty_foreground_is_valid(self):
        measurement = measure_foams105_phase(np.zeros((5, 5), dtype=np.uint8), 1, 1.0)
        result = measure_foams105_moment_geometry("empty", measurement)
        self.assertEqual(result.components, ())
        self.assertEqual(result.foreground_pixel_count, 0)

    def test_invalid_image_id_and_measurement_type_are_rejected(self):
        measurement = measure_foams105_phase(np.zeros((5, 5), dtype=np.uint8), 1, 1.0)
        with self.assertRaises(Foams105MomentValidationError):
            measure_foams105_moment_geometry("", measurement)
        with self.assertRaises(Foams105MomentValidationError):
            measure_foams105_moment_geometry("x", object())

    def test_forged_measurement_mismatches_are_rejected(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[3, 3] = 1
        measurement = measure_foams105_phase(image, 1, 1.0)

        forged_labels = replace(
            measurement,
            label_image=np.asarray(measurement.label_image, dtype=np.float64),
        )
        with self.assertRaises(Foams105MomentValidationError):
            measure_foams105_moment_geometry("forged-label-dtype", forged_labels)

        bad_component = replace(
            measurement.components[0],
            first_pixel_rc=(0, 0),
        )
        with self.assertRaises(Foams105MomentValidationError):
            measure_foams105_moment_geometry(
                "forged-first-pixel",
                replace(measurement, components=(bad_component,)),
            )

        changed_area = replace(measurement.components[0], area_mm2=measurement.components[0].area_mm2 + 1.0)
        with self.assertRaises(Foams105MomentValidationError):
            measure_foams105_moment_geometry(
                "forged-area",
                replace(measurement, components=(changed_area,)),
            )

    def test_results_are_frozen_and_own_values(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[3, 3] = 1
        result = measure_foams105_moment_geometry(
            "frozen", measure_foams105_phase(image, 1, 1.0)
        )
        self.assertEqual(result.perimeter_status, FOAMS105_PERIMETER_STATUS_PENDING)
        with self.assertRaises(FrozenInstanceError):
            result.method = "changed"
        with self.assertRaises(FrozenInstanceError):
            result.components[0].major_axis_px = 0.0


class TestFoams105MomentSelectionAdapter(unittest.TestCase):
    def _geometry_inputs(self):
        image_a = np.zeros((12, 12), dtype=np.uint8)
        image_a[2, 2] = 1
        image_a[2:4, 6:8] = 1
        image_a[6:9, 2:5] = 1

        image_b = np.zeros((12, 12), dtype=np.uint8)
        image_b[2:4, 2:4] = 1
        image_b[6:9, 6:9] = 1

        measurement_a = measure_foams105_phase(image_a, 1, 1.0)
        measurement_b = measure_foams105_phase(image_b, 1, 1.0)
        geometry_a = measure_foams105_moment_geometry("image-a", measurement_a)
        geometry_b = measure_foams105_moment_geometry("image-b", measurement_b)

        group_a = Foams105RawGroupInput(
            "group-a",
            1,
            tuple(_object_from_component("image-a", component) for component in measurement_a.components),
        )
        group_b = Foams105RawGroupInput(
            "group-b",
            2,
            tuple(_object_from_component("image-b", component) for component in measurement_b.components),
        )
        raw = select_foams105_raw_objects(
            (1.0, 2.0, 4.0),
            (group_a, group_b),
            (
                Foams105BinnedRange("group-a", 4.0, 4.0),
                Foams105BinnedRange("group-b", 4.0, 4.0),
            ),
        )
        return raw, geometry_a, geometry_b

    def test_selection_preserves_order_and_identity_with_reused_component_numbers(self):
        raw, geometry_a, geometry_b = self._geometry_inputs()
        result = select_foams105_moment_geometry(raw, (geometry_a, geometry_b))
        self.assertEqual(result.method, FOAMS105_MOMENT_SELECTION_METHOD)
        self.assertEqual(result.selected_count, len(raw.rows))
        self.assertEqual(
            result.selected_identities,
            tuple((row.image_id, row.component_id) for row in raw.rows),
        )
        self.assertEqual(
            tuple(row.raw_output_index for row in result.rows),
            tuple(row.output_index for row in raw.rows),
        )
        self.assertEqual(
            tuple((row.source_slot, row.source_object_index) for row in result.rows),
            tuple((row.source_slot, row.source_object_index) for row in raw.rows),
        )

    def test_missing_duplicate_and_changed_identities_are_rejected(self):
        raw, geometry_a, geometry_b = self._geometry_inputs()

        missing = replace(geometry_a, components=geometry_a.components[1:])
        with self.assertRaises(Foams105MomentValidationError) as caught:
            select_foams105_moment_geometry(raw, (missing, geometry_b))
        self.assertEqual(caught.exception.code, "missing_geometry_identity")

        duplicate_image = replace(geometry_b, image_id="image-a")
        with self.assertRaises(Foams105MomentValidationError) as caught:
            select_foams105_moment_geometry(raw, (geometry_a, duplicate_image))
        self.assertEqual(caught.exception.code, "duplicate_image_id")

        changed_component = replace(
            geometry_a.components[0],
            area_mm2=geometry_a.components[0].area_mm2 + 1.0,
        )
        changed = replace(
            geometry_a,
            components=(changed_component,) + geometry_a.components[1:],
        )
        with self.assertRaises(Foams105MomentValidationError) as caught:
            select_foams105_moment_geometry(raw, (changed, geometry_b))
        self.assertEqual(caught.exception.code, "changed_geometry_identity")

    def test_forged_raw_selection_result_is_rejected(self):
        raw, geometry_a, geometry_b = self._geometry_inputs()
        forged = replace(raw, selected_count=raw.selected_count + 1)
        with self.assertRaises(Foams105MomentValidationError) as caught:
            select_foams105_moment_geometry(forged, (geometry_a, geometry_b))
        self.assertEqual(caught.exception.code, "inconsistent_raw_selection_result")


class TestFoams105MomentIndependentReference(unittest.TestCase):
    def test_independent_pixel_sum_profile_matches_implementation(self):
        image = np.zeros((16, 16), dtype=np.uint8)
        image[2:4, 2:5] = 1
        image[8, 3] = 1
        image[10:13, 10] = 1
        image[12, 11] = 1

        measurement = measure_foams105_phase(image, 1, 2.0)
        result = measure_foams105_moment_geometry("reference", measurement)
        labels = np.asarray(measurement.label_image)

        for component in result.components:
            rows, cols = np.nonzero(labels == component.component_id)
            independent = _independent_moment_profile(rows, cols)
            _assert_close(self, component.centroid_x_px, independent["centroid_x"])
            _assert_close(self, component.centroid_y_px, independent["centroid_y"])
            _assert_close(self, component.central_cxx, independent["cxx"])
            _assert_close(self, component.central_cxy, independent["cxy"])
            _assert_close(self, component.central_cyy, independent["cyy"])
            _assert_close(self, component.major_axis_px, independent["major"])
            _assert_close(self, component.minor_axis_px, independent["minor"])
            _assert_close(self, component.eccentricity, independent["eccentricity"])
            _assert_close(
                self,
                component.orientation_regionprops_profile_deg,
                independent["theta"],
            )
            self.assertEqual(component.orientation_defined, independent["defined"])


if __name__ == "__main__":
    unittest.main()
