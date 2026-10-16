# FOAMS 1.0.5 density summaries and plotting quantities

This stage reproduces the bounded density reductions, paired volume bins, and
complete-histogram size categories associated with the pinned FOAMS 1.0.5
`analysis.m`. It consumes canonical Brief 14 volume output and does not modify
converter N_V values or normalized volume fractions.

## Public API

`summarize_foams105_distribution(volume_result, global_bin_labels_mm, ...)`
returns immutable rows containing lower edges, widths, midsizes, inclusive
reverse-cumulative N_V, plotting logarithms, per-width density, corrected N_V,
and consecutive paired volume fractions.

`summarize_foams105_size_categories(global_bin_labels_mm, count_groups,
lower_cutoff_mm, upper_cutoff_mm)` pools one through four complete histogram
vectors and returns strict-boundary category counts and percentages.

Both APIs accept Python and NumPy numeric scalars. Count values and source
slots must be integers; Booleans and integral-valued floats are rejected.
Results own immutable tuples, retain provenance metadata, and use typed
validation and numerical exceptions with stable codes and immutable context.

## Distribution arithmetic

The global grid contains 1 through 45 unique, finite, positive, strictly
increasing labels. Every selected label must match a grid value exactly.
Selected duplicates remain separate rows.

For selected labels $L_i$, the first lower edge is the preceding global label
when one exists. If the selection begins at the first grid label, it is
$L_0 / 10^{0.1}$. Lower edges are then:

$$
E = \operatorname{sort}([E_0] + L[:-1])
$$

Each width and midsize is:

$$
W_i = L_i - E_i, \qquad M_i = \frac{L_i + E_i}{2}
$$

Reverse cumulative N_V is inclusive and preserves selected row order:

$$
C_i = \sum_{j=i}^{n-1} N_{V,j}
$$

Forward total N_V is accumulated separately. Plotting quantities are
$\log_{10}(L_i)$, $\log_{10}(C_i)$, $N_{V,i}/W_i$, and
$\ln(N_{V,i}/W_i)$. Signed values are retained and diagnosed rather than
clipped or reordered.

Consecutive adjusted-volume rows are paired as $(0,1)$, $(2,3)$, and so on.
Each pair uses the second row's label and ordered fraction sum. An odd trailing
row remains a singleton, and every result records its source row indices.

## N_V correction

The solid fraction and corrected N_V are:

$$
S = 1 - 0.01P, \qquad N_{V,\mathrm{corr}} = \frac{N_V}{S}
$$

Unnormalized volume output requires a separate canonical 2D vesicularity
result. Integrated mode uses its embedded canonical 2D result and rejects a
redundant external result. Known-vesicularity mode uses its stored target and
also rejects an external result. Matching conversion and vesicularity
acquisition identity remains the caller's responsibility.

## Explicit undefined values

Expected real-domain gaps do not use NaN, infinity, or complex payloads.
`Foams105RealValue` instead records `value=None`, a stable reason, operands,
and the corresponding legacy classification. Reasons cover zero width,
logarithms of zero or negative values, zero solid fraction, and zero total
category count. A negative width is unreachable for valid grid input and is
rejected by validation. A negative solid fraction and finite corrected N_V
are retained with a diagnostic. Unexpected nonfinite arithmetic and nonzero
underflow raise `Foams105SummaryNumericalError` with operation-stage context.

## Size categories

Counts are pooled by global label with an exact total limit of $2^{53}-1$.
Category comparisons are strict:

- lower: label less than the lower cutoff;
- middle: label greater than the lower and less than the upper cutoff;
- upper: label greater than the upper cutoff;
- boundary: label equal to either cutoff.

Equal cutoffs contribute their shared boundary once. Boundary counts are
excluded from the three percentages. A zero total produces three explicit
`zero_total_count` records rather than zero percentages.

## Evidence boundary

The bundled fixture has five calculated distribution cases and four calculated
category cases. They are hand-checkable expectations, not MATLAB output or
original workbook references. The unchanged 29-row `Calc_out.xls` extraction
is used only to retain converter parity and demonstrate that its duplicate
label becomes an explicit zero-width summary row. All downstream summary
values in that report are calculated values.

This stage does not establish GUI plotting, shape, export, original-image, or
complete workflow parity. Run its optimized-safe report with:

```bash
python -O -m main.examples.foams105_density_summaries
```