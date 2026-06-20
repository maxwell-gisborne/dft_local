# Typst diagnostics library

Reusable Typst helpers for static diagnostic bundle exports.

Generated diagnostic bundles import this directory through `lib/mod.typ`.
The exporter defaults to symlinking this directory into each bundle when it is
available, and falls back to an internal minimal library only when this project
library cannot be found.

Current public helpers:

- `diagnostic-figure(...)`
- `diagnostic-note(...)`
- `diagnostic-table(data)`
- `line-graph(data)`
- `dos-idos-view(data)`
- `region-surface-summary(data)`
- `unsupported-view(data)`

The input data is written by `dft_local.diagnostics.typst_bundle` as JSON under
the bundle `data/` directory.


## Role in the diagnostics system

This library is the static presentation layer for Typst diagnostic bundles. It
should stay small and data-driven. Python writes JSON files and generated wrapper
functions; this library defines reusable Typst rendering primitives.

The interactive HTML renderer may use custom elements and Datastar. The Typst
renderer should instead show the best honest static equivalent:

- ordinary graphs as line plots
- tables and matrices as static tables
- DOS/IDOS viewers as two line plots
- region-surface viewers as summary tables
- unsupported interactive views as explicit placeholders

If a helper is required by generated bundles, update both this library and the
fallback library writer in `src/dft_local/diagnostics/typst_bundle.py`.

