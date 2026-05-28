# dft_local

`dft_local` is a local analysis package for sparse DFT Hamiltonian and overlap matrices.

It can:

- load sparse DFT matrix output
- build atom-block BSR data structures
- build group-labelled geometry
- construct Fourier symbols
- solve local generalized eigenproblems
- continue bands along paths and over regions
- rectify energy surfaces
- compute Boltzmann conductivity
- serve local diagnostics

## Install

From the repository root:

```bash
uv pip install -e .
```

or:

```bash
python -m pip install -e .
```

This installs the command:

```bash
dft-local
```

## Test

Run the full repository test suite:

```bash
pytest -q
```

Run the package-discovered suite:

```bash
dft-local test --timeout 120
```

Some tests require sparse DFT fixture data at:

```text
test_run/run_dir/data
```

This directory is ignored by git because it is large. If the data is somewhere else, set:

```bash
export DFT_LOCAL_DATA_ROOT=/path/to/data
```

Expected fixture files include:

```text
sparsematrix_metadata.dat
hamiltonian_sparse.mtx
overlap_sparse.mtx
```

## Command line

Run discovered package tests:

```bash
dft-local test --timeout 120
```

Run the diagnostic server:

```bash
dft-local serve --reload
```

The diagnostic server loads data from `DFT_LOCAL_DATA_ROOT` if set, otherwise from:

```text
test_run/run_dir/data
```

## Project layout

```text
src/dft_local/
├── __init__.py
├── __main__.py
├── core/
├── diagnostics/
├── diagnostics_pannel/
├── testsuite/
└── transport/
    ├── bands/
    └── boltzmann/
```

## `core`

Shared numerical, sparse-matrix, dataset, geometry, kernel, and local eigenproblem code.

### `core.numerics`

Small numerical helpers and unit definitions:

- `Units`
- `AU`
- `eVag`
- `freeze_array`
- `hermitian_part`
- `DenseMatrixDiagnostics`
- array type aliases

### `core.dataset`

Sparse DFT output loading:

- `SparseMetadata`
- `BasisMap`
- `SparseDataset`
- `AtomBlock`
- `atom_ordered_bsr`

`SparseDataset` owns the sparse Hamiltonian `H`, overlap matrix `S`, metadata, basis map, units, and atom-block inspection helpers.

### `core.sparse`

Sparse BSR block helpers:

- `block_row_raw`
- `block_view_bsr`
- `coupled_atoms_table`
- `coupled_atoms_table_by_distance`

These functions assume each BSR block corresponds to an atom-atom coupling.

### `core.geometry`

Geometry and edge-generator group labelling:

- `GdElement`
- `NearestNeighbourGraph`
- `EdgeDirections`
- `EdgeGroupLabels`
- `estimate_plane_basis`

This layer turns atomic positions into neighbour graphs and group labels.

### `core.kernels`

Group-labelled kernel arrays and Fourier symbols:

- `GdKernelArrays`
- `gd_inverse_label`
- `relative_labels_for_row`

`GdKernelArrays` converts sparse atom-block data into dense Fourier symbols for generic and fixed irreps.

### `core.local_problem`

Dense generalized eigenproblem layer:

- `SymbolPair`
- `LocalProblem`

A `SymbolPair` forms `H(k)` and `S(k)`. A `LocalProblem` solves the generalized eigenproblem.

## `transport`

Physical workflows.

### `transport.bands`

Band/path/region continuation:

- Hungarian matching helpers
- energy-prediction matching
- state-overlap matching
- gauge fixing
- degenerate-subspace alignment
- crossing/event detection
- `LocalPath`
- `LocalRegion`

This module supports continuation along paths and over two-dimensional regions.

### `transport.bands.energy_surface_rectification`

Second-pass energy-surface labelling over solved regions:

- `rectify_energy_arrays_across_u`
- `rectify_local_region_energy_surfaces`
- `energy_surface_roughness`
- `transverse_energy_prediction`
- `transverse_path_cost`

Use this when each path is locally continued, but neighbouring paths have inconsistent band labels.

### `transport.boltzmann`

Boltzmann conductivity:

- `fermi_window`
- symbol derivatives
- generalized Hellmann-Feynman velocities
- sample accumulation
- AC Boltzmann weights
- `BoltzmannConductivity`

This module computes conductivity from solved local symbols and velocities.

## `diagnostics`

Lightweight diagnostic infrastructure:

- structured diagnostic models
- explicit diagnostic discovery
- HTML rendering
- local diagnostic context
- ASGI diagnostic server

Diagnostics are exposed through domain modules using `diagnostics()` functions.

## `diagnostics_pannel`

Compatibility package for old imports using the historic misspelling `pannel`.

New code should prefer:

```python
import dft_local.diagnostics
```

Old code that imports this still works:

```python
import dft_local.diagnostics_pannel
```

## `testsuite`

Package-owned test discovery and runner.

Domain modules expose test targets through `tests()` functions. The command:

```bash
dft-local test --timeout 120
```

collects those targets and runs them through pytest.

## Public API

The top-level package exports the main API:

```python
from dft_local import SparseDataset
from dft_local import GdKernelArrays
from dft_local import SymbolPair, LocalProblem
from dft_local import LocalPath, LocalRegion
from dft_local import BoltzmannConductivity
```

Compatibility module paths are also kept:

```python
from dft_local.boltzmann_conductivity import BoltzmannConductivity
from dft_local.energy_surface_rectification import rectify_energy_arrays_across_u
```

## Typical workflow

Load data:

```python
from dft_local import SparseDataset

data = SparseDataset.load("test_run/run_dir/data")
```

Build geometry and labels:

```python
from dft_local import NearestNeighbourGraph, EdgeDirections, EdgeGroupLabels

geom = NearestNeighbourGraph.from_positions(data.metadata.positions)
edges = EdgeDirections.from_geometry(geom)
labels = EdgeGroupLabels.from_geometry(geom, edges)
```

Build kernels:

```python
from dft_local import GdKernelArrays

KH = GdKernelArrays.from_average(data.H, labels, matrix_name="H average")
KS = GdKernelArrays.from_average(data.S, labels, matrix_name="S average")

KH = KH.star_symmetrised(matrix_name="H average star")
KS = KS.star_symmetrised(matrix_name="S average star")
```

Solve a local symbol problem:

```python
from dft_local import SymbolPair

pair = SymbolPair(KH=KH, KS=KS, k1=0.0, k2=0.0)
problem = pair.form()
energies, vectors = problem.eigensystem()
```

Follow a band path:

```python
import numpy as np
from dft_local import LocalPath, eVag

points = [
    ("Γ", 0.0, 0.0),
    ("K", 2 * np.pi / 3, -2 * np.pi / 3),
    ("M", np.pi, 0.0),
    ("Γ", 0.0, 0.0),
]

path = LocalPath.from_points(
    KH,
    KS,
    points,
    points_per_segment=32,
    units=eVag,
).solve_continuation()
```

## Notes

- Large DFT run data is not stored in git.
- `test_run/` is ignored.
- `__pycache__/` and Python bytecode files are ignored.
- The active package is `dft_local`.
