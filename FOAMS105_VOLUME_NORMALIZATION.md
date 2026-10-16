# FOAMS 1.0.5 volume fractions and vesicularity normalization

This stage reproduces the bounded equivalent-sphere volume-fraction and
normalization arithmetic associated with the pinned FOAMS 1.0.5 `analysis.m`.
It consumes accepted converter output and, for integrated normalization,
accepted two-dimensional vesicularity output.

## Public API

`calculate_foams105_volume_distribution(conversion_result, mode, ...)` accepts
an exact canonical `Foams105ConversionResult`. The supported modes are:

- `unnormalized`: retain the raw equivalent-sphere volume fractions;
- `integrated_2d`: normalize to the canonical integrated 2D vesicularity;
- `known_vesicularity`: normalize to a supplied percentage from 0 through 100.

The result retains the canonical upstream records, converter row order,
duplicate-label metadata, original signed N_V values, per-row sphere volumes,
raw and adjusted fractions, ordered cumulative fractions, sums, normalization
factor and residual, and contextual diagnostics.

## Arithmetic

For converter row $i$, with accepted mean projected height $H_i$ and signed
number density $N_{V,i}$:

$$
V_i = \frac{\pi H_i^3}{6}
$$

$$
f_i = V_i N_{V,i}
$$

The raw sum $F$ and every cumulative output use unchanged converter order:

$$
F = \sum_i f_i
$$

Unnormalized mode uses the factor $k=1$. A normalized mode with target
percentage $P$ uses:

$$
k = \frac{P/100}{F}
$$

The adjusted and cumulative fractions are:

$$
a_i = kf_i
$$

$$
C_i = \sum_{j=1}^{i} a_j
$$

Integrated mode uses `integrated_vesicularity_percent`, never the arithmetic
mean of per-group image percentages. Known-vesicularity input is a percentage,
not a unit fraction. Normalization changes volume fractions only; it does not
change N_V.

## Signed and zero behavior

Finite signed converter values, negative raw or adjusted fractions, negative
normalization factors, decreasing cumulative values, and integrated targets
outside 0 to 100 are preserved with diagnostics. Values are not clipped,
reordered, or merged, and the final sum is not forced to equal the target.

An exactly zero raw sum is valid in unnormalized mode. Both normalized modes
raise `Foams105VolumeDomainError` with code `zero_raw_volume_sum`, including a
zero target or cancellation between signed rows. Nonfinite operations and
nonzero underflow raise `Foams105VolumeNumericalError` with operation stage and
row context.

## Validation boundary

The converter is replayed once from its canonical inputs and complete result
equality is required. Integrated mode similarly replays the complete
vesicularity result. This validates each supplied record independently; the
caller remains responsible for supplying converter and vesicularity results
from the same analysis.

The bundled 10-case fixture contains calculated, hand-checkable numerical
expectations: eight successful cases and two zero-sum rejections. It is not
MATLAB output or an original workbook reference. Public integration tests use
canonical converter and synthetic-mask vesicularity records. This stage does
not establish plotting, cumulative-count, NV/NVcorr summary, shape, GUI,
export, original-image, or complete workflow parity.

Run the optimized-safe report with:

```bash
python -O -m main.examples.foams105_volume_normalization
```
