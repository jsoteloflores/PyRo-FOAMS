# FOAMS 1.0.5 component moment geometry and shape identity

This stage replays component-level moment geometry on accepted filled,
four-connected labels from the measurement stage. It implements major/minor
ellipse-axis lengths, eccentricity, source orientation, and the project's
`+90°` orientation representation while preserving component identity through
raw selection.

## Public API

- `measure_foams105_moment_geometry(image_id, phase_measurement_result)`
- `select_foams105_moment_geometry(raw_selection_result, image_geometry_results)`

The measurement API consumes an accepted `Foams105PhaseMeasurementResult`
without rerunning topology, labeling, hole fill, or contour fitting. The
selection API joins canonical raw-selection identities to measured geometry in
raw row order.

## Moment profile

For each component pixel at zero-based row/column `(r, c)`, use Cartesian
coordinates `x = c`, `y = -r` and population normalization by `N` pixels.

$$
C_{xx} = \operatorname{mean}((x-\bar{x})^2) + \frac{1}{12}
$$

$$
C_{yy} = \operatorname{mean}((y-\bar{y})^2) + \frac{1}{12}
$$

$$
C_{xy} = \operatorname{mean}((x-\bar{x})(y-\bar{y}))
$$

The `1/12` terms are the within-pixel unit-square variance contribution.

For the symmetric central-moment matrix:

$$
\Delta = \operatorname{hypot}(C_{xx}-C_{yy}, 2C_{xy})
$$

$$
\lambda_{\text{major}} = \frac{C_{xx}+C_{yy}+\Delta}{2}, \quad
\lambda_{\text{minor}} = \frac{C_{xx}+C_{yy}-\Delta}{2}
$$

$$
\text{MajorAxis}_{px} = 4\sqrt{\lambda_{\text{major}}}, \quad
\text{MinorAxis}_{px} = 4\sqrt{\lambda_{\text{minor}}}
$$

$$
\text{Eccentricity} = \sqrt{1-\frac{\lambda_{\text{minor}}}{\lambda_{\text{major}}}}
$$

Calibration is length-only:

$$
\text{MajorAxis}_{mm} = \frac{\text{MajorAxis}_{px}}{\text{scale}_{px/mm}}, \quad
\text{MinorAxis}_{mm} = \frac{\text{MinorAxis}_{px}}{\text{scale}_{px/mm}}
$$

A two-pass centered accumulation is used. If direct subtraction loses
positivity, a stable symmetric `2x2` eigensolver fallback is used before
raising a numerical error.

## Orientation

For anisotropic tensors:

$$
\theta = \deg\left(\frac{1}{2}\operatorname{atan2}(2C_{xy}, C_{xx}-C_{yy})\right)
$$

- `orientation_regionprops_profile_deg = θ` in `[-90, 90]`
- exact vertical axes normalize to `+90`
- `orientation_project_deg = θ + 90`

For exactly isotropic second moments (`Cxx == Cyy` and `Cxy == 0`), this
profile stores `theta=0`, `orientation_defined=False`, and
`orientation_tie_reason="isotropic_second_moments"`.

## Identity, perimeter, and verification boundaries

- Selection requires a canonical replay of raw selection.
- Every raw object identity `(image_id, component_id)` must resolve to geometry
  with unchanged calibrated area and equivalent diameter.
- Extra geometry rows are allowed; missing or changed identities are errors.
- Perimeter is intentionally not part of the verified moment API.
  `perimeter_status` is always `"pending_historical_profile"`.

Verification level is explicitly
`"analytical_pixel_moment_profile"`: this stage does not claim bitwise
`regionprops` parity, historical orientation tie parity, or historical
perimeter compatibility.

## Evidence and scope

The bundled fixture `16_foams105_moment_cases.json` contains analytical
unit-square cases (singleton, thin, isotropic, anisotropic, diagonal split,
empty foreground). It is not MATLAB output.

This stage does not implement perimeter compatibility (`Brief 16a`) or
shape-derived aggregate statistics (`Brief 17`).

Run the optimized-safe report with:

```bash
python -O -m main.examples.foams105_moment_geometry
```
