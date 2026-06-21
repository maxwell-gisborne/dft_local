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

Status: implemented

Diagnostic id: `dataset.overview`

Shows:

- data root
- atom count
- basis size
- symbol counts
- H/S matrix shape and BSR block structure summary
- units and disk-to-working conversion provenance

Follow-up:

- add links from dataset rows to matrix and geometry diagnostics

### Geometry overview

Status: implemented

Diagnostic id: `geometry.overview`

Shows:

- nearest-neighbour graph summary
- chosen anchor
- edge directions and classification alignment
- visited label count
- sublattice balance via `eps` counts
- G_d label ranges
- reconstruction error

### Matrix overview

Status: implemented

Diagnostic id: `matrix.overview`

Shows:

- H/S BSR shape and block rows
- scalar and atom-block nonzero counts
- row-block distribution
- global Hermiticity defects
- H/S atom-block support overlap

Follow-up:

- add sampled block-level Hermiticity mismatch table
- add optional worst-row / worst-block drilldown

### Kernel overview

Status: implemented

Diagnostic id: `kernel.overview`

Inputs:

- kernel choice: `anchored`, `anchored_star`, `average`, `average_star`

Shows:

- support size
- matrix block shape
- even/odd support split
- label ranges
- block norm summary
- star-defect diagnostics

Follow-up:

- add sampled symbol Hermiticity diagnostics at selected k-points
- add worst star-defect rows from `star_defect_table_filtered`

### Symbol point

Status: implemented

Diagnostic id: `symbol.point`

Inputs:

- kernel choice: `anchored`, `anchored_star`, `average`, `average_star`
- logical irrep coordinates `k1`, `k2`
- irrep degree: `1` or `2`
- `sigma` for degree-1 fixed irreps

Shows:

- dense H(k), S(k) diagnostics
- Hermiticity defects before and after taking Hermitian parts
- overlap eigenvalues
- generalized eigenvalues

Follow-up:

- add physical k-coordinate annotation from the embedding dual map
- add compact energy-window cards for band-edge inspection

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
## Current validation-domain state

The finite-field validation probe layer is now typed.

- All `finite_field_*_probe` functions return frozen dataclass probes rather than plain dictionaries.
- Scalar diagnostic fields are annotated with `Annotated[float, qscalar(...)]` and carry a `unit_context`.
- Finite-field diagnostic table rows use `diagnostic_scalar_quantity(...)` for typed scalar rendering.
- `test_finite_field_validation_probes_are_not_plain_dict_returns` guards against regressions to dict-returning finite-field probes.
- The latest full validation/render check passed with the finite-field typed probe layer enabled.

This does not yet mean the whole validation package is typed. Non-finite-field validation helpers such as production symbol/operator probes are postponed below.

## Postponed tasks

These are deliberately not part of the current finite-field typing slice, but should stay visible for later planning.

- Type the remaining non-finite-field validation probe domains, including production symbol/operator validation probes and their diagnostics rows.
- Revisit the postponed downloadable table task.
- Audit whether all diagnostic table rows use typed quantities rather than raw float formatting.
- Consider a broader guard test for the whole validation domain once all non-finite-field probes are typed.

## UI work

- left navigation grouped by domain
- readable card layout
- consistent forms
- readable tables
- graph components with axis labels
- useful error pages
