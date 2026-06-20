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
- `unsupported-view(data)`

The input data is written by `dft_local.diagnostics.typst_bundle` as JSON under
the bundle `data/` directory.
