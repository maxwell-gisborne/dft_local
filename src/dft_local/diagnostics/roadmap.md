# Diagnostics roadmap

The diagnostic server is the user-facing exploration layer for `dft_local`.

## Policy

Diagnostics are first-class. They should expose the real physics/data workflows,
not only prove that infrastructure works.

A diagnostic should:

- call domain code rather than duplicate it
- have URL-stable inputs
- render useful tables or graphs
- give readable errors
- be covered by a page-rendering test

## Priority diagnostics

### Dataset overview

Status: missing

Show:

- data root
- atom count
- basis size
- matrix shape
- H/S nonzero blocks
- units
- basic sparse sanity checks

### Geometry overview

Status: missing

Show:

- nearest-neighbour graph summary
- chosen anchor
- edge directions
- visited label count
- sublattice balance
- reconstruction error

### Kernel overview

Status: missing

Inputs:

- kernel choice

Show:

- support size
- matrix block shape
- star-defect diagnostics
- Hermiticity diagnostics

### Symbol point

Status: missing

Inputs:

- kernel choice
- `k1`
- `k2`
- irrep degree
- sigma for fixed irreps

Show:

- H(k) diagnostics
- S(k) diagnostics
- overlap eigenvalues
- band energies

### Band path

Status: high priority

Inputs:

- kernel choice
- path preset
- points per segment
- matching strategy

Show:

- band energy plot along path
- high-symmetry tick labels
- crossing/degeneracy events
- gauge/matching summary

### Region

Status: high priority

Inputs:

- kernel choice
- u/v grid
- matching strategy

Show:

- band surfaces or slices
- region continuation events
- min/max energy by band
- optional selected-band heatmap

### Energy rectification

Status: high priority

Inputs:

- solved region source
- prediction order
- accept ratio

Show:

- roughness before/after
- accepted path permutations
- before/after path or band plots

### Boltzmann conductivity

Status: exists but needs UI polish

Improve:

- split overview from expensive calculation
- show assumptions clearly
- show units
- add downloadable table later

## UI work

- left navigation grouped by domain
- readable card layout
- consistent forms
- readable tables
- graph components with axis labels
- useful error pages
