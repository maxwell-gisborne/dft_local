"""DFT local domain package."""

from __future__ import annotations

from dft_local.core.dataset import (
    AtomBlock,
    BasisMap,
    SparseDataset,
    SparseMetadata,
    atom_ordered_bsr,
    freeze_bsr,
    require_dir,
    require_file,
)
from dft_local.core.geometry import (
    EdgeDirections,
    EdgeGroupLabels,
    GdElement,
    NearestNeighbourGraph,
    estimate_plane_basis,
)
from dft_local.core.kernels import (
    GdKernelArrays,
    gd_inverse_label,
    relative_labels_for_row,
)
from dft_local.core.local_problem import LocalProblem, SymbolPair
from dft_local.core.numerics import (
    AU,
    BlockArray,
    DenseMatrixDiagnostics,
    FloatArray,
    IntArray,
    Units,
    eVag,
    freeze_array,
    hermitian_part,
)
from dft_local.core.sparse import (
    block_row_raw,
    block_view_bsr,
    coupled_atoms_table,
    coupled_atoms_table_by_distance,
)
from dft_local.transport.bands.core import (
    BandEvent,
    DegenerateGroupEvent,
    LocalPath,
    LocalRegion,
    align_degenerate_group_with_reorder,
    align_groups_and_fix_gauge,
    bz_hexagon_vertices,
    detect_energy_order_crossings_between_steps,
    eigenvector_overlap_scores,
    energy_degenerate_groups,
    grouped_hungarian_order_from_costs,
    hungarian_order_from_costs,
    hungarian_order_from_scores,
    match_via_energies,
    match_via_overlap,
    metric_between,
    predicted_energies_from_history,
    transverse_path_cost,
)
from dft_local.transport.bands.energy_surface_rectification import (
    EnergyRectificationReport,
    energy_surface_roughness,
    rectify_energy_arrays_across_u,
    rectify_local_region_energy_surfaces,
    transverse_energy_prediction,
)
from dft_local.transport.boltzmann.calculation.core import (
    BoltzmannConductivity,
    BoltzmannSampleResult,
    fermi_window,
    gd_symbol_derivative_fixed,
    gd_symbol_derivative_generic,
    gd_symbol_derivatives,
)


def fix_gauge_against_previous_arrays(U_prev, S_prev, U_curr):
    """Fix current eigenvector phases against previous eigenvectors."""

    import numpy as np

    U = np.array(U_curr, copy=True)
    overlaps = np.diag(U_prev.conj().T @ S_prev @ U)

    for i, z in enumerate(overlaps):
        mag = abs(z)
        if mag > 0:
            U[:, i] *= np.conj(z) / mag

    return U
