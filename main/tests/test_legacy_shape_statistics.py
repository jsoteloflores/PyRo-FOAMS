from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import numpy as np

from main.core.legacy_measurements import measure_foams105_phase
from main.core.legacy_perimeter import (
    FOAMS105_PERIMETER_PROFILE_CURRENT,
    FOAMS105_PERIMETER_PROFILE_OLD,
    measure_foams105_perimeter,
)
from main.core.legacy_raw_selection import (
    Foams105RawGroupInput,
    Foams105RawObject,
    Foams105RawSelectionValidationError,
    select_foams105_raw_objects,
)
from main.core.legacy_selection import Foams105BinnedRange
from main.core.legacy_shape_statistics import (
    Foams105ShapeStatisticsNumericalError,
    Foams105ShapeStatisticsValidationError,
    Foams105ShapeValue,
    _derive_measures,
    _round_half_away_4,
    _statistics_row,
    derive_foams105_shapes,
    summarize_foams105_shapes,
)
from main.core.legacy_shapes import measure_foams105_moment_geometry


def _image(shape, pixels):
    result = np.zeros(shape, dtype=np.uint8)
    for row, column in pixels:
        result[row, column] = 1
    return result


def _build_pipeline(specifications, profile=FOAMS105_PERIMETER_PROFILE_CURRENT):
    measurements = {}
    geometry_results = []
    perimeter_results = []
    groups = []
    ranges = []
    for slot, (image_id, shape, pixels, scale) in enumerate(specifications, start=1):
        measured = measure_foams105_phase(_image(shape, pixels), 1, scale)
        measurements[image_id] = measured
        geometry_results.append(measure_foams105_moment_geometry(image_id, measured))
        perimeter_results.append(measure_foams105_perimeter(image_id, measured, profile=profile))
        objects = tuple(
            Foams105RawObject(image_id, item.component_id, item.equivalent_diameter_mm, item.area_mm2)
            for item in measured.components
        )
        group_id = f"group-{image_id}"
        groups.append(Foams105RawGroupInput(group_id, slot, objects))
        ranges.append(Foams105BinnedRange(group_id, 0.1, 10.0))

    raw = select_foams105_raw_objects((0.01, 0.1, 10.0), tuple(groups), tuple(ranges))
    result = derive_foams105_shapes(raw, tuple(geometry_results), tuple(perimeter_results))
    return result, measurements, tuple(geometry_results), tuple(perimeter_results)


def _row_with_regularity(base, index, value):
    return replace(
        base,
        image_id=f"stats-{index}",
        component_id=index + 1,
        regularity=Foams105ShapeValue(value, "defined"),
    )


class TestFoams105ShapeFormulae(unittest.TestCase):
    def _records(self, area, major, minor, perimeter):
        moment = type(
            "MomentInput",
            (),
            {
                "image_id": "formula-image",
                "component_id": 7,
                "area_mm2": area,
                "major_axis_mm": major,
                "minor_axis_mm": minor,
            },
        )()
        perimeter_record = type("PerimeterInput", (), {"perimeter_mm": perimeter})()
        return moment, perimeter_record

    def test_formula_fixture_unit_case(self):
        moment, perimeter = self._records(math.pi, 2.0, 2.0, 2.0 * math.pi)
        ellipse, regularity, form_factor, elongation, aspect, fitted = _derive_measures(moment, perimeter)
        self.assertTrue(math.isclose(ellipse, math.pi, rel_tol=1e-14))
        self.assertTrue(math.isclose(regularity.value, 1.0, rel_tol=1e-14))
        self.assertTrue(math.isclose(form_factor.value, 1.0, rel_tol=1e-14))
        self.assertEqual(elongation.value, 0.0)
        self.assertEqual(aspect.value, 1.0)
        self.assertTrue(math.isclose(fitted.value, 4.0 * math.pi, rel_tol=1e-14))

    def test_formula_fixture_asymmetric_case(self):
        moment, perimeter = self._records(2.0 * math.pi, 4.0, 2.0, 10.0)
        ellipse, regularity, form_factor, elongation, aspect, fitted = _derive_measures(moment, perimeter)
        self.assertTrue(math.isclose(ellipse, 2.0 * math.pi, rel_tol=1e-14))
        self.assertTrue(math.isclose(regularity.value, 1.0, rel_tol=1e-14))
        self.assertTrue(math.isclose(form_factor.value, 0.08 * math.pi**2, rel_tol=1e-14))
        self.assertTrue(math.isclose(elongation.value, 1.0 / 3.0, rel_tol=1e-14))
        self.assertEqual(aspect.value, 0.5)
        expected_fitted = 8.0 * (2.0 + (math.pi - 2.0) * (0.5**1.456))
        self.assertTrue(math.isclose(fitted.value, expected_fitted, rel_tol=1e-14))

    def test_zero_perimeter_is_explicit_positive_infinity(self):
        moment, perimeter = self._records(math.pi, 2.0, 1.0, 0.0)
        _, regularity, form_factor, elongation, aspect, fitted = _derive_measures(moment, perimeter)
        self.assertIsNone(form_factor.value)
        self.assertEqual(form_factor.status, "undefined")
        self.assertEqual(form_factor.reason, "zero_perimeter")
        self.assertEqual(form_factor.legacy_class, "positive_infinity")
        self.assertTrue(all(item.value is not None for item in (regularity, elongation, aspect, fitted)))

    def test_nonzero_underflow_and_overflow_are_contextual(self):
        moment, perimeter = self._records(1.0, 5e-324, 5e-324, 1.0)
        with self.assertRaises(Foams105ShapeStatisticsNumericalError) as caught:
            _derive_measures(moment, perimeter)
        self.assertEqual(caught.exception.code, "nonzero_underflow")
        self.assertEqual(caught.exception.context["expression"], "ellipse_area_mm2.semi_axes")

        moment, perimeter = self._records(1.0, 1e308, 1e308, 1.0)
        with self.assertRaises(Foams105ShapeStatisticsNumericalError) as caught:
            _derive_measures(moment, perimeter)
        self.assertEqual(caught.exception.code, "nonfinite_shape_arithmetic")
        self.assertIn("ellipse_area_mm2.axes_product", caught.exception.context["expression"])


class TestFoams105ShapeStatistics(unittest.TestCase):
    def test_literal_statistics_sample_mode_and_rounding(self):
        base, _, _, _ = _build_pipeline((
            ("base", (8, 8), ((2, 2),), 1.0),
        ))
        population = tuple(_row_with_regularity(base.rows[0], index, value) for index, value in enumerate((1.0, 1.0, 2.0, 2.0)))
        row = _statistics_row("regularity", population)
        self.assertEqual(row.n, 4)
        self.assertEqual(row.maximum, 2.0)
        self.assertEqual(row.minimum, 1.0)
        self.assertEqual(row.mean, 1.5)
        self.assertTrue(math.isclose(row.standard_deviation, math.sqrt(1.0 / 3.0), rel_tol=1e-15))
        self.assertEqual(row.median, 1.5)
        self.assertEqual(row.mode, 1.0)
        self.assertEqual(row.display_standard_deviation, 0.5774)
        self.assertEqual(_round_half_away_4(0.03125, metric="test"), 0.0313)
        self.assertEqual(_round_half_away_4(-0.03125, metric="test"), -0.0313)

    def test_matlab_roundn_binary_scale_boundary_values(self):
        boundary = 0.00035
        below = math.nextafter(boundary, -math.inf)
        above = math.nextafter(boundary, math.inf)
        expected = (
            (below, 0.0003),
            (boundary, 0.0004),
            (above, 0.0004),
            (-below, -0.0003),
            (-boundary, -0.0004),
            (-above, -0.0004),
            (0.00015, 0.0001),
            (-0.00015, -0.0001),
        )
        for value, rounded in expected:
            with self.subTest(value=value):
                self.assertEqual(_round_half_away_4(value, metric="roundn_boundary"), rounded)

        with self.assertRaises(Foams105ShapeStatisticsNumericalError) as caught:
            _round_half_away_4(1e308, metric="roundn_overflow")
        self.assertEqual(caught.exception.code, "nonfinite_rounding")
        self.assertEqual(caught.exception.context["stage"], "scale")

    def test_statistics_preserve_00035_raw_and_round_display_values(self):
        base, _, _, _ = _build_pipeline((
            ("base", (8, 8), ((2, 2),), 1.0),
        ))
        row = _statistics_row("regularity", (_row_with_regularity(base.rows[0], 0, 0.00035),))
        self.assertEqual(row.n, 1)
        self.assertEqual(row.minimum, 0.00035)
        self.assertEqual(row.maximum, 0.00035)
        self.assertEqual(row.mean, 0.00035)
        self.assertEqual(row.median, 0.00035)
        self.assertEqual(row.mode, 0.00035)
        self.assertEqual(row.standard_deviation, 0.0)
        self.assertEqual(row.display_minimum, 0.0004)
        self.assertEqual(row.display_maximum, 0.0004)
        self.assertEqual(row.display_mean, 0.0004)
        self.assertEqual(row.display_median, 0.0004)
        self.assertEqual(row.display_mode, 0.0004)
        self.assertEqual(row.display_standard_deviation, 0.0)

    def test_statistics_empty_singleton_distinct_ties_and_unrounded_mode(self):
        empty = _statistics_row("regularity", ())
        self.assertEqual(empty.status, "unavailable")
        self.assertEqual(empty.reason, "empty_population")
        self.assertEqual((empty.n, empty.maximum, empty.mean, empty.mode), (0, None, None, None))

        base, _, _, _ = _build_pipeline((("base", (8, 8), ((2, 2),), 1.0),))
        singleton = _statistics_row("regularity", (_row_with_regularity(base.rows[0], 0, 3.0),))
        self.assertEqual(singleton.standard_deviation, 0.0)
        self.assertEqual((singleton.minimum, singleton.maximum, singleton.mean, singleton.median, singleton.mode), (3.0,) * 5)

        all_distinct = tuple(_row_with_regularity(base.rows[0], i, value) for i, value in enumerate((3.0, 1.0, 2.0)))
        self.assertEqual(_statistics_row("regularity", all_distinct).mode, 1.0)
        equal_pair = tuple(_row_with_regularity(base.rows[0], i, 2.0) for i in range(2))
        self.assertEqual(_statistics_row("regularity", equal_pair).standard_deviation, 0.0)

        close_values = tuple(
            _row_with_regularity(base.rows[0], i, value)
            for i, value in enumerate((1.00001, 1.00002))
        )
        close_row = _statistics_row("regularity", close_values)
        self.assertEqual(close_row.mode, 1.00001)
        self.assertEqual(close_row.display_mode, 1.0)

        invalid_eccentricity = replace(close_values[0], eccentricity=float("inf"))
        with self.assertRaises(Foams105ShapeStatisticsValidationError):
            _statistics_row("eccentricity", (invalid_eccentricity,))


class TestFoams105ShapePipeline(unittest.TestCase):
    def setUp(self):
        self.specs = (
            ("image-a", (12, 12), ((2, 2), (5, 5), (5, 6), (6, 5), (6, 6)), 1.0),
            ("image-b", (12, 12), ((3, 3), (3, 4), (3, 5)), 2.0),
        )
        self.shape_result, self.measurements, self.geometry, self.perimeters = _build_pipeline(self.specs)

    def test_real_measurements_preserve_identity_order_and_metadata(self):
        rows = self.shape_result.rows
        self.assertEqual(
            tuple((row.image_id, row.component_id) for row in rows),
            self.shape_result.raw_selection_result.selected_identities,
        )
        self.assertEqual(tuple(row.output_index for row in rows), tuple(range(len(rows))))
        self.assertEqual(tuple(row.source_slot for row in rows), (1, 1, 2))
        self.assertEqual(len({row.component_id for row in rows if row.image_id == "image-a"}), 2)
        self.assertEqual(len({row.component_id for row in rows if row.image_id == "image-b"}), 1)
        self.assertEqual(self.shape_result.perimeter_profile, FOAMS105_PERIMETER_PROFILE_CURRENT)
        self.assertEqual(self.shape_result.perimeter_reference_runtime, "MATLAB 25.2.0.2998904 (R2025b)")
        self.assertEqual(self.shape_result.perimeter_evidence_version, self.perimeters[0].evidence_version)
        self.assertEqual(self.shape_result.perimeter_evidence_hash, self.perimeters[0].evidence_hash)
        self.assertEqual(self.shape_result.moment_verification_level, "analytical_pixel_moment_profile")
        self.assertTrue(all(row.orientation_project_deg == row.moment_component.orientation_project_deg for row in rows))
        self.assertTrue(all(row.orientation_tie_reason == row.moment_component.orientation_tie_reason for row in rows))
        singleton = next(row for row in rows if row.area_px == 1)
        self.assertEqual(singleton.perimeter_mm, 0.0)
        self.assertEqual(singleton.form_factor.reason, "zero_perimeter")
        self.assertIsNotNone(singleton.regularity.value)
        with self.assertRaises(FrozenInstanceError):
            rows[0].regularity = Foams105ShapeValue(9.0, "defined")  # type: ignore[misc]

    def test_equal_area_different_labels_are_rejected(self):
        horizontal = measure_foams105_phase(_image((9, 9), ((4, 3), (4, 4))), 1, 1.0)
        vertical = measure_foams105_phase(_image((9, 9), ((3, 4), (4, 4))), 1, 1.0)
        raw = select_foams105_raw_objects(
            (0.01, 0.1, 10.0),
            (Foams105RawGroupInput(
                "g",
                1,
                (Foams105RawObject("same", 1, horizontal.components[0].equivalent_diameter_mm, 2.0),),
            ),),
            (Foams105BinnedRange("g", 0.1, 10.0),),
        )
        moment = measure_foams105_moment_geometry("same", horizontal)
        perimeter = measure_foams105_perimeter("same", vertical, profile=FOAMS105_PERIMETER_PROFILE_CURRENT)
        with self.assertRaises(Foams105ShapeStatisticsValidationError) as caught:
            derive_foams105_shapes(raw, (moment,), (perimeter,))
        self.assertEqual(caught.exception.code, "phase_measurement_mismatch")
        self.assertEqual(caught.exception.context["image_id"], "same")

    def test_profile_changes_only_form_factor(self):
        spec = self.specs[0]
        image_id, shape, pixels, scale = spec
        measured = measure_foams105_phase(_image(shape, pixels), 1, scale)
        geometry = measure_foams105_moment_geometry(image_id, measured)
        perimeters_current = measure_foams105_perimeter(image_id, measured, profile=FOAMS105_PERIMETER_PROFILE_CURRENT)
        perimeters_old = measure_foams105_perimeter(image_id, measured, profile=FOAMS105_PERIMETER_PROFILE_OLD)
        group = Foams105RawGroupInput(
            "g",
            1,
            tuple(Foams105RawObject(image_id, item.component_id, item.equivalent_diameter_mm, item.area_mm2) for item in measured.components),
        )
        raw = select_foams105_raw_objects((0.01, 0.1, 10.0), (group,), (Foams105BinnedRange("g", 0.1, 10.0),))
        current = derive_foams105_shapes(raw, (geometry,), (perimeters_current,))
        old = derive_foams105_shapes(raw, (geometry,), (perimeters_old,))
        for new_row, old_row in zip(current.rows, old.rows):
            self.assertEqual(new_row.aspect_ratio, old_row.aspect_ratio)
            self.assertEqual(new_row.elongation, old_row.elongation)
            self.assertEqual(new_row.regularity, old_row.regularity)
            if new_row.perimeter_mm != old_row.perimeter_mm:
                self.assertNotEqual(new_row.form_factor, old_row.form_factor)

    def test_calibration_changes_lengths_but_not_dimensionless_measures(self):
        image_id, shape, pixels, _ = self.specs[0]
        outputs = []
        for scale in (1.0, 2.0):
            result, _, _, _ = _build_pipeline(((image_id, shape, pixels, scale),))
            outputs.append(result.rows[1])
        first, second = outputs
        for name in ("regularity", "form_factor", "elongation", "aspect_ratio"):
            self.assertEqual(getattr(first, name), getattr(second, name))
        self.assertTrue(math.isclose(first.major_axis_mm / 2.0, second.major_axis_mm, rel_tol=1e-14))
        self.assertTrue(math.isclose(first.ellipse_area_mm2 / 4.0, second.ellipse_area_mm2, rel_tol=1e-14))

    def test_threshold_modes_preserve_source_population_discrepancy(self):
        singleton = self.shape_result.rows[0]
        equality = summarize_foams105_shapes(
            self.shape_result,
            configured_scales_px_per_mm=(1.0, 2.0),
            minimum_diameter_px=singleton.equivalent_diameter_mm * 2.0,
            mode="initial_analysis",
        )
        updated_equality = summarize_foams105_shapes(
            self.shape_result,
            configured_scales_px_per_mm=(1.0, 2.0),
            minimum_diameter_px=singleton.equivalent_diameter_mm * 2.0,
            mode="updated_minimum",
        )
        self.assertNotIn((singleton.image_id, singleton.component_id), equality.threshold_identities)
        self.assertIn((singleton.image_id, singleton.component_id), updated_equality.threshold_identities)
        self.assertEqual(equality.summary_population_count, len(self.shape_result.rows))
        self.assertEqual(updated_equality.summary_population_count, updated_equality.threshold_count)

        below = summarize_foams105_shapes(
            self.shape_result,
            configured_scales_px_per_mm=(1.0,),
            minimum_diameter_px=float(np.nextafter(singleton.equivalent_diameter_mm, -np.inf)),
            mode="initial_analysis",
        )
        above = summarize_foams105_shapes(
            self.shape_result,
            configured_scales_px_per_mm=(1.0,),
            minimum_diameter_px=float(np.nextafter(singleton.equivalent_diameter_mm, np.inf)),
            mode="updated_minimum",
        )
        self.assertIn((singleton.image_id, singleton.component_id), below.threshold_identities)
        self.assertNotIn((singleton.image_id, singleton.component_id), above.threshold_identities)

    def test_summary_does_not_replay_images_and_uses_external_max_scale(self):
        with patch("main.core.legacy_shape_statistics.select_foams105_moment_geometry") as moment_replay, patch(
            "main.core.legacy_shape_statistics.select_foams105_perimeter"
        ) as perimeter_replay:
            result = summarize_foams105_shapes(
                self.shape_result,
                configured_scales_px_per_mm=(1.0, 2.0, 9.0),
                minimum_diameter_px=9.0,
                mode="updated_minimum",
            )
        moment_replay.assert_not_called()
        perimeter_replay.assert_not_called()
        self.assertEqual(result.maximum_configured_scale_px_per_mm, 9.0)
        self.assertEqual(result.threshold_mm, 1.0)

    def test_adapters_replay_once_per_image(self):
        with patch(
            "main.core.legacy_shapes.measure_foams105_moment_geometry",
            wraps=measure_foams105_moment_geometry,
        ) as moment_measure, patch(
            "main.core.legacy_perimeter.measure_foams105_perimeter",
            wraps=measure_foams105_perimeter,
        ) as perimeter_measure:
            derive_foams105_shapes(self.shape_result.raw_selection_result, self.geometry, self.perimeters)
        self.assertEqual(moment_measure.call_count, len(self.geometry))
        self.assertEqual(perimeter_measure.call_count, len(self.perimeters))

    def test_extra_components_and_nonconsecutive_selected_ids_are_allowed(self):
        image_id, shape, pixels, scale = self.specs[0]
        measurement = self.measurements[image_id]
        geometry = self.geometry[0]
        perimeter = self.perimeters[0]
        retained = measurement.components[1]
        raw = select_foams105_raw_objects(
            (0.01, 0.1, 10.0),
            (Foams105RawGroupInput(
                "one-object",
                1,
                (Foams105RawObject(image_id, retained.component_id, retained.equivalent_diameter_mm, retained.area_mm2),),
            ),),
            (Foams105BinnedRange("one-object", 0.1, 10.0),),
        )
        derived = derive_foams105_shapes(raw, (geometry,), (perimeter,))
        self.assertEqual(tuple(row.component_id for row in derived.rows), (2,))
        self.assertEqual(derived.rows[0].output_index, 0)
        self.assertEqual(derived.rows[0].source_object_index, 0)

    def test_duplicate_raw_identity_is_rejected(self):
        first = self.shape_result.raw_selection_result.groups[0].objects[0]
        duplicate_groups = (
            Foams105RawGroupInput("first", 1, (first,)),
            Foams105RawGroupInput("second", 2, (first,)),
        )
        with self.assertRaises(Foams105RawSelectionValidationError):
            select_foams105_raw_objects(
                (0.01, 0.1, 10.0),
                duplicate_groups,
                (Foams105BinnedRange("first", 0.1, 10.0), Foams105BinnedRange("second", 0.1, 10.0)),
            )

    def test_rejected_and_disabled_group_objects_still_require_geometry(self):
        measurement = measure_foams105_phase(_image((8, 8), ((3, 3),)), 1, 1.0)
        object_ = Foams105RawObject(
            "rejected-image",
            1,
            measurement.components[0].equivalent_diameter_mm,
            measurement.components[0].area_mm2,
        )
        raw = select_foams105_raw_objects(
            (1.0, 10.0),
            (Foams105RawGroupInput("rejected", 1, (object_,)),),
            (Foams105BinnedRange("rejected", 10.0, 10.0),),
        )
        self.assertEqual(raw.selected_count, 0)
        with self.assertRaises(Foams105ShapeStatisticsValidationError) as caught:
            derive_foams105_shapes(raw, (), ())
        self.assertEqual(caught.exception.code, "invalid_moment_input")
        self.assertEqual(caught.exception.context["cause_code"], "missing_geometry_identity")

        disabled_raw = select_foams105_raw_objects(
            (0.01, 0.1, 10.0),
            (
                Foams105RawGroupInput("active", 1, self.shape_result.raw_selection_result.groups[1].objects),
                Foams105RawGroupInput("disabled", 2, (object_,)),
            ),
            (Foams105BinnedRange("active", 0.1, 10.0), None),
        )
        with self.assertRaises(Foams105ShapeStatisticsValidationError) as caught:
            derive_foams105_shapes(disabled_raw, self.geometry[1:], self.perimeters[1:])
        self.assertEqual(caught.exception.context["cause_code"], "missing_geometry_identity")

    def test_mixed_profiles_wrong_units_forged_input_and_stale_selection_reject(self):
        old_b = measure_foams105_perimeter(
            "image-b",
            self.perimeters[1].phase_measurement_result,
            profile=FOAMS105_PERIMETER_PROFILE_OLD,
        )
        with self.assertRaises(Foams105ShapeStatisticsValidationError):
            derive_foams105_shapes(self.shape_result.raw_selection_result, self.geometry, (self.perimeters[0], old_b))

        wrong_units = replace(self.perimeters[0], length_unit="inch")
        with self.assertRaises(Foams105ShapeStatisticsValidationError) as caught:
            derive_foams105_shapes(self.shape_result.raw_selection_result, self.geometry, (wrong_units, self.perimeters[1]))
        self.assertEqual(caught.exception.code, "invalid_perimeter_input")

        forged_geometry = replace(self.geometry[0], scale_px_per_mm=2.0)
        with self.assertRaises(Foams105ShapeStatisticsValidationError):
            derive_foams105_shapes(self.shape_result.raw_selection_result, (forged_geometry, self.geometry[1]), self.perimeters)

        stale_raw = replace(self.shape_result.raw_selection_result, selected_count=99)
        with self.assertRaises(Foams105ShapeStatisticsValidationError):
            derive_foams105_shapes(stale_raw, self.geometry, self.perimeters)

        forged_shape = replace(
            self.shape_result,
            rows=(replace(self.shape_result.rows[0], regularity=Foams105ShapeValue(999.0, "defined")),)
            + self.shape_result.rows[1:],
        )
        with self.assertRaises(Foams105ShapeStatisticsValidationError) as caught:
            summarize_foams105_shapes(
                forged_shape,
                configured_scales_px_per_mm=(1.0,),
                minimum_diameter_px=0.0,
                mode="initial_analysis",
            )
        self.assertEqual(caught.exception.code, "forged_shape_result")

    def test_owned_arrays_empty_inputs_and_summary_argument_validation(self):
        label_image = self.shape_result.image_geometry_results[0].phase_measurement_result.label_image
        with self.assertRaises(ValueError):
            label_image[0, 0] = 1

        empty_raw = select_foams105_raw_objects(
            (0.1, 1.0, 2.0),
            (Foams105RawGroupInput("empty", 1, ()),),
            (Foams105BinnedRange("empty", 1.0, 2.0),),
        )
        empty_shapes = derive_foams105_shapes(empty_raw, (), ())
        empty_summary = summarize_foams105_shapes(
            empty_shapes,
            configured_scales_px_per_mm=(3.0,),
            minimum_diameter_px=0.0,
            mode="initial_analysis",
        )
        self.assertEqual(empty_summary.summary_population_count, 0)
        self.assertTrue(all(row.reason == "empty_population" for row in empty_summary.statistic_rows))

        for scales, minimum, mode in (
            ((), 1.0, "initial_analysis"),
            ((True,), 1.0, "initial_analysis"),
            ((1.0, "2"), 1.0, "initial_analysis"),
            (((1.0,),), 1.0, "initial_analysis"),
            ({"scale": 1.0}, 1.0, "initial_analysis"),
            ({1.0}, 1.0, "initial_analysis"),
            ((1.0,), True, "initial_analysis"),
            ((1.0,), -1.0, "initial_analysis"),
            ((1.0,), "1.0", "initial_analysis"),
            ((1.0,), 1.0, "unknown"),
        ):
            with self.subTest(scales=scales, minimum=minimum, mode=mode):
                with self.assertRaises(Foams105ShapeStatisticsValidationError):
                    summarize_foams105_shapes(
                        empty_shapes,
                        configured_scales_px_per_mm=scales,
                        minimum_diameter_px=minimum,
                        mode=mode,
                    )
        with self.assertRaises(TypeError):
            summarize_foams105_shapes(
                empty_shapes,
                configured_scales_px_per_mm=(1.0,),
                minimum_diameter_px=0.0,
            )  # type: ignore[call-arg]

    def test_summary_rejects_each_incompatible_cached_unit(self):
        empty_raw = select_foams105_raw_objects(
            (0.1, 1.0, 2.0),
            (Foams105RawGroupInput("empty", 1, ()),),
            (Foams105BinnedRange("empty", 1.0, 2.0),),
        )
        empty_shapes = derive_foams105_shapes(empty_raw, (), ())
        invalid_units = (
            ("length_unit", "inch", "mm"),
            ("area_unit", "inch^2", "mm^2"),
            ("angle_unit", "radian", "degree"),
        )
        for valid_result in (self.shape_result, empty_shapes):
            for mode in ("initial_analysis", "updated_minimum"):
                with self.subTest(result_count=valid_result.selected_count, mode=mode):
                    accepted = summarize_foams105_shapes(
                        valid_result,
                        configured_scales_px_per_mm=(1.0,),
                        minimum_diameter_px=0.0,
                        mode=mode,
                    )
                    self.assertEqual(accepted.shape_result, valid_result)
                    for field, actual, expected in invalid_units:
                        altered = replace(valid_result, **{field: actual})
                        with self.subTest(field=field):
                            with self.assertRaises(Foams105ShapeStatisticsValidationError) as caught:
                                summarize_foams105_shapes(
                                    altered,
                                    configured_scales_px_per_mm=(1.0,),
                                    minimum_diameter_px=0.0,
                                    mode=mode,
                                )
                            self.assertEqual(caught.exception.code, "incompatible_shape_result_unit")
                            self.assertEqual(caught.exception.context["field"], field)
                            self.assertEqual(caught.exception.context["expected"], expected)
                            self.assertEqual(caught.exception.context["actual"], actual)

    def test_positive_threshold_underflow_is_contextual(self):
        with self.assertRaises(Foams105ShapeStatisticsNumericalError) as caught:
            summarize_foams105_shapes(
                self.shape_result,
                configured_scales_px_per_mm=(float(np.finfo(np.float64).max),),
                minimum_diameter_px=float(np.nextafter(0.0, 1.0)),
                mode="initial_analysis",
            )
        self.assertEqual(caught.exception.code, "threshold_underflow")


if __name__ == "__main__":
    unittest.main()
