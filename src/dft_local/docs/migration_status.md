# Migration status

This document records the current package checkpoint.

## Current checkpoint

The discovered test suite passes:

    168 passed

Run it with:

    dft-local test --timeout 120

## Moved into `dft_local`

### Core

Shared core modules now live locally:

- `core.numerics`
  - array aliases
  - unit definitions
  - read-only array helper
  - Hermitian projection
  - dense matrix diagnostics

- `core.sparse`
  - sparse BSR block-row extraction

- `core.geometry`
  - `GdElement`
  - nearest-neighbour graph
  - edge directions
  - edge/group labels

- `core.kernels`
  - `gd_inverse_label`
  - relative edge labels
  - `GdKernelArrays`
  - generic and fixed-irrep symbols
  - star symmetrisation and star-defect diagnostics

- `core.local_problem`
  - `SymbolPair`
  - `LocalProblem`

### Transport bands

Band/path/region continuation now lives in:

    src/dft_local/transport/bands/core.py

It owns:

- Hungarian matching helpers
- energy-prediction matching
- state-overlap matching
- gauge fixing
- degenerate-subspace alignment
- band-event detection
- `LocalPath`
- `LocalRegion`

### Transport Boltzmann

Boltzmann conductivity now lives in:

    src/dft_local/transport/boltzmann/core.py

It owns:

- Fermi-window weights
- symbol derivatives
- generalized Hellmann-Feynman velocities
- AC Boltzmann weights
- sample accumulation
- integrated conductivity matrix

## Tests moved/copied

dft_local owns copied business/physical tests for:

- Boltzmann conductivity
- band continuation
- region continuation
- core geometry/kernel behaviour

The discovered suite includes these local tests.

## Old package dependency status

Migrated transport/core implementation code has no direct old `dft_local` imports.

The remaining references to old `dft_local` are intentional:

- legacy diagnostic context loading
- documentation explaining separation from the old package
- repository-level compatibility context

## Diagnostics status

The diagnostics use explicit discovery through `diagnostics()` functions.

Current diagnostics:

- `dft_local.testsuite`
- `transport.boltzmann.calculation.overview`
- `transport.boltzmann.calculation.conductivity`
- `transport.bands.overview`

The server is ASGI-compatible and can be run with uvicorn reload.

## Run server

From repository root:

    dft-local serve --reload

or explicitly:

    DFT_LOCAL_DATA_ROOT=test_run/run_dir/data \
    uvicorn dft_local.diagnostics.server:app \
      --host 127.0.0.1 \
      --port 8765 \
      --reload \
      --reload-dir src/dft_local \
      --reload-dir src/dft_local

## Next good steps

1. Add more focused tests for `core.geometry`
2. Add more focused tests for `core.kernels`
3. Remove duplicate repository-level tests when confident
4. Add packaging entry points so `PYTHONPATH=src` is no longer needed
5. Consider splitting very large transport modules if they become hard to read
