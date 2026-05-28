# Band/path continuation module

This module owns band continuation along paths and, later, over regions.

## Current migration status

Local in this package:

- `hungarian_order_from_costs`
- `hungarian_order_from_scores`
- `predicted_energies_from_history`
- `match_via_energies`
- `eigenvector_overlap_scores`
- `energy_degenerate_groups`
- `align_degenerate_group_with_reorder`
- `align_groups_and_fix_gauge`
- `match_via_overlap`
- `detect_energy_order_crossings_between_steps`
- `LocalPath`
- `LocalRegion`

No band/path/region implementation is imported from old `dft_local`.

The remaining old-package references are legacy diagnostic context loading,
historical docs, and repository-level compatibility tests.

## Behaviour owned by tests

The local tests cover:

- Hungarian assignment from costs
- energy prediction from band history
- energy-prediction band matching through crossings
- gauge fixing against previous eigenvectors
- degenerate-subspace Procrustes alignment
- connected-component degeneracy grouping
- overlap-based matching
- `LocalPath.solve_continuation` with both energy and overlap strategies
- propagation of continuation matching through simple synthetic examples

## Migration plan

1. Keep copied business tests green
2. Move pure helper functions locally
3. Move `LocalPath` locally
4. Move `LocalRegion` locally
5. Move remaining region helper functions locally
6. Move `SymbolPair` and `LocalProblem` into a symbol/local-problem domain
7. Move kernel arrays into `core.kernels`
8. Move graph/data extraction infrastructure when ready
