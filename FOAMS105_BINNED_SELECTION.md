# FOAMS 1.0.5 checked binned selection

`select_foams105_binned_ranges()` applies explicit inclusive numeric label ranges to one to four original source slots. `select_foams105_suggested_ranges()` validates a complete `Foams105CutoffSuggestionResult` and delegates to the same manual selector.

## Contract

- Labels are a finite, positive, strictly increasing grid of 1 to 45 values.
- Groups retain contiguous original slots `1..G`; disabled middle slots are not renumbered.
- Each density vector is an immutable full-grid tuple of finite nonnegative values.
- Ranges are positional by source slot and keyed by group ID. `None` disables a slot; slot 1 must remain enabled.
- Bounds are literal, finite, positive, inclusive values. They are not snapped to labels.
- Active slots are concatenated in descending source-slot order, while rows inside each slot remain in ascending global-index order.
- Shared boundary rows, gaps, and overlaps are retained and reported. Rows are never sorted, shifted, filled, truncated, or deduplicated.
- Source label indices must exactly equal source density indices. Any mismatch raises `Foams105SelectionDomainError` with code `source_slice_mismatch`.
- More than 45 selected rows are retained, with `converter_length_supported=False`.

The selector does not call `convert_foams105_nv()`. Callers must make conversion explicit after checking converter eligibility.

## Fixture integrity

The analytical selection fixture is pinned to SHA-256
`acaa9f08ac3aaa3326f6d661b6df02c724d09eedb3b41020e1985da1a06f1a4b`.
The test normalizes only CRLF pairs to LF before hashing so text checkouts have
the same digest on Windows and POSIX systems. All other bytes remain covered,
including the final newline, spacing, key order, BOM presence and reference
values. Regression checks cover the real fixture, a synthesized CRLF checkout
and an in-memory numerical-value change.

## Scope

This is a checked replay of the binned range-selection stage from pinned `analysis.m`. It does not implement raw-object selection, shape behavior, vesicularity, or the complete original workflow.

Run the reconciliation report with:

```bash
python -m main.examples.foams105_binned_selection
```
