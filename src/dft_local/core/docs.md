# Refactor core modules

The core package holds shared mathematical and numerical objects used by
domain modules.

## Current modules

### `core.numerics`

Owns small numerical helpers and unit definitions:

- `FloatArray`
- `IntArray`
- `BlockArray`
- `Units`
- `AU`
- `eVag`
- `freeze_array`
- `hermitian_part`
- `DenseMatrixDiagnostics`

### `core.kernels`

Owns the group-labelled kernel object:

- `gd_inverse_label`
- `relative_labels_for_row`
- `GdKernelArrays`

`GdKernelArrays` is the bridge from sparse block data to dense Fourier symbols.
It is used by:

- `core.local_problem`
- `transport.bands.core`
- `transport.boltzmann.core`

It now uses local data/geometry infrastructure:

- `core.geometry.EdgeGroupLabels`
- `core.sparse.block_row_raw`

### `core.local_problem`

Owns the dense generalized eigenproblem layer:

- `SymbolPair`
- `LocalProblem`

This is shared by band continuation and Boltzmann conductivity.

## Migration boundary

The physics/business logic has moved into `dft_local`.

Current checkpoint: discovered package suite passes.

Old transport/business dependencies have been removed. There are no direct old
`dft_local` implementation imports in the migrated transport/core code. The
remaining old-package references are diagnostic legacy context loading and
historical docs/tests.

Data/geometry infrastructure has been moved into core modules:

- `core.sparse` owns sparse block-row extraction
- `core.geometry` owns group-labelled geometry and edge labels
- `core.kernels` owns kernel arrays and Fourier symbols

Keep these layers out of `transport.bands` and `transport.boltzmann`; those
modules should stay focused on transport/band behaviour.
