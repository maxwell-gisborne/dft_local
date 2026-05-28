# Business-logic tests for band/path continuation.
#
# Copied from repository-level continuation tests so the bands domain owns
# its matching, gauge, degeneracy, and continuation behaviour locally.

"""
Tests for LocalPath / LocalRegion continuation matching strategies.

These tests focus on the newer risk areas:

- energy-prediction matching versus state-overlap matching;
- Procrustes alignment inside degenerate subspaces;
- correct per-step storage of final overlap diagnostics;
- propagation of matching_strategy through LocalRegion;
- controlled toy Hamiltonians with known smooth energy sheets.

They are intentionally synthetic and do not need the BigDFT fixture data.
"""

from __future__ import annotations

import numpy as np
import pytest

from dft_local.core.numerics import eVag
from dft_local.transport.bands import core as bands


class IdentityOverlapKernel:
    def symbol_generic(self, k1: float, k2: float) -> np.ndarray:
        return np.eye(2, dtype=np.complex128)

    def symbol_fixed(self, k1: float, k2: float, sigma: int) -> np.ndarray:
        return np.eye(2, dtype=np.complex128)


class CrossingHamiltonianK1:
    """H(k) = diag(k1, -k1). Energy sheets cross at k1=0."""

    def symbol_generic(self, k1: float, k2: float) -> np.ndarray:
        return np.array([[k1, 0.0], [0.0, -k1]], dtype=np.complex128)

    def symbol_fixed(self, k1: float, k2: float, sigma: int) -> np.ndarray:
        return self.symbol_generic(k1, k2)


class CrossingHamiltonianK2:
    """H(k) = diag(k2, -k2). Useful for v-path region tests."""

    def symbol_generic(self, k1: float, k2: float) -> np.ndarray:
        return np.array([[k2, 0.0], [0.0, -k2]], dtype=np.complex128)

    def symbol_fixed(self, k1: float, k2: float, sigma: int) -> np.ndarray:
        return self.symbol_generic(k1, k2)


class RotatingTwoLevelHamiltonian:
    """
    H(theta) = R(theta) diag(-1, +1) R(theta)^T.

    The eigenvalues are constant and nondegenerate, while the eigenvectors rotate
    smoothly. State-overlap continuation should keep high overlap for small
    steps.
    """

    def symbol_generic(self, k1: float, k2: float) -> np.ndarray:
        theta = float(k1)
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        D = np.diag([-1.0, 1.0])
        return (R @ D @ R.T).astype(np.complex128)

    def symbol_fixed(self, k1: float, k2: float, sigma: int) -> np.ndarray:
        return self.symbol_generic(k1, k2)


@pytest.fixture
def fake_S() -> IdentityOverlapKernel:
    return IdentityOverlapKernel()


def make_path(H, k1, k2=None, *, strategy="energy_predict") -> bands.LocalPath:
    k1 = np.asarray(k1, dtype=np.float64)
    if k2 is None:
        k2 = np.zeros_like(k1)
    else:
        k2 = np.asarray(k2, dtype=np.float64)

    return bands.LocalPath.from_arrays(
        H,
        IdentityOverlapKernel(),
        k1,
        k2,
        units=eVag,
        matching_strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Low-level assignment and prediction helpers
# ---------------------------------------------------------------------------


def test_hungarian_order_from_costs_minimises_cost() -> None:
    costs = np.array([
        [10.0, 0.0, 10.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 10.0],
    ])

    order = bands.hungarian_order_from_costs(costs)

    assert order.tolist() == [1, 2, 0]
    assert np.allclose(costs[np.arange(3), order], 0.0)


def test_predicted_energies_from_history_constant_linear_quadratic() -> None:
    energies = np.array([
        [0.0, 10.0],
        [1.0, 8.0],
        [2.0, 6.0],
    ])

    assert np.allclose(
        bands.predicted_energies_from_history(energies, 1, order=0),
        [0.0, 10.0],
    )
    assert np.allclose(
        bands.predicted_energies_from_history(energies, 2, order=1),
        [2.0, 6.0],
    )
    assert np.allclose(
        bands.predicted_energies_from_history(energies, 3, order=2),
        [3.0, 4.0],
    )


def test_match_via_energies_assigns_to_prediction_not_sorted_order() -> None:
    energies = np.empty((4, 2), dtype=np.float64)
    energies[0] = [-1.0, 1.0]
    energies[1] = [-0.5, 0.5]
    energies[2] = [0.0, 0.0]

    # Raw eigensolver order at the next point is energy-sorted, but the smooth
    # sheets predicted from history are [+0.5, -0.5].
    E_raw = np.array([-0.5, 0.5])

    order, costs = bands.match_via_energies(E_raw, energies, 3, prediction_order=2)

    assert order.tolist() == [1, 0]
    assert np.allclose(E_raw[order], [0.5, -0.5])
    assert np.allclose(costs, 0.0)


def test_match_via_energies_rejects_shape_mismatch() -> None:
    energies = np.zeros((2, 3), dtype=np.float64)
    E_raw = np.zeros(2, dtype=np.float64)

    with pytest.raises(ValueError, match="shape mismatch"):
        bands.match_via_energies(E_raw, energies, 1)


# ---------------------------------------------------------------------------
# Overlap, gauge, and degenerate-subspace alignment
# ---------------------------------------------------------------------------


def test_fix_gauge_makes_diagonal_overlap_real_positive() -> None:
    U_prev = np.eye(2, dtype=np.complex128)
    phases = np.array([np.exp(0.7j), np.exp(-1.2j)])
    U_curr = U_prev * phases[None, :]
    S = np.eye(2, dtype=np.complex128)

    U_fixed = bands.fix_gauge_against_previous_arrays(U_prev, S, U_curr)
    overlaps = U_prev.conj().T @ S @ U_fixed

    assert np.allclose(np.imag(np.diag(overlaps)), 0.0, atol=1e-12)
    assert np.all(np.real(np.diag(overlaps)) > 0.0)
    assert np.allclose(np.abs(np.diag(overlaps)), 1.0)


def test_align_degenerate_group_with_reorder_aligns_rotated_subspace() -> None:
    theta = 0.37
    c = np.cos(theta)
    s = np.sin(theta)

    U_prev = np.eye(2, dtype=np.complex128)
    R = np.array([[c, -s], [s, c]], dtype=np.complex128)
    S = np.eye(2, dtype=np.complex128)

    # Current solver basis is an arbitrary rotation inside the same degenerate
    # subspace.
    U_curr = U_prev @ R

    U_aligned, singular_values, _order_after = bands.align_degenerate_group_with_reorder(
        U_prev,
        S,
        U_curr,
    )

    score = bands.eigenvector_overlap_scores(U_prev, S, U_aligned)

    assert np.allclose(singular_values, [1.0, 1.0], atol=1e-12)
    assert np.allclose(np.diag(score), [1.0, 1.0], atol=1e-12)
    assert np.allclose(score - np.diag(np.diag(score)), 0.0, atol=1e-12)


def test_energy_degenerate_groups_uses_connected_components() -> None:
    E = np.array([0.0, 0.05, 0.11, 2.0, 2.01])

    groups = bands.energy_degenerate_groups(E, gap_tol=0.06)

    assert [g.tolist() for g in groups] == [[0, 1, 2], [3, 4]]


def test_align_groups_and_fix_gauge_aligns_group_and_phase_fixes_isolated_band() -> None:
    theta = 0.4
    c = np.cos(theta)
    s = np.sin(theta)

    U_prev = np.eye(3, dtype=np.complex128)
    U_curr = np.eye(3, dtype=np.complex128)
    U_curr[:2, :2] = np.array([[c, -s], [s, c]], dtype=np.complex128)
    U_curr[:, 2] *= np.exp(1.3j)

    E = np.array([0.0, 0.0, 5.0])
    S = np.eye(3, dtype=np.complex128)

    U_fixed, events = bands.align_groups_and_fix_gauge(
        U_prev,
        S,
        U_curr,
        E,
        step=4,
        gap_tol=1e-12,
    )

    score = bands.eigenvector_overlap_scores(U_prev, S, U_fixed)

    assert len(events) == 1
    assert events[0].step == 4
    assert events[0].bands == (0, 1)
    assert events[0].subspace_score == pytest.approx(2.0, abs=1e-12)
    assert np.allclose(np.diag(score), [1.0, 1.0, 1.0], atol=1e-12)


def test_match_via_overlap_reports_degenerate_group_events() -> None:
    theta = 0.4
    c = np.cos(theta)
    s = np.sin(theta)

    prev_U = np.eye(2, dtype=np.complex128)
    U_raw = np.array([[c, -s], [s, c]], dtype=np.complex128)
    E_raw = np.array([0.0, 0.0])
    S = np.eye(2, dtype=np.complex128)

    E, U, matched_scores, events = bands.match_via_overlap(
        prev_U,
        S,
        E_raw,
        U_raw,
        fix_gauge=True,
        align_degenerate=True,
        degeneracy_tol=1e-12,
        step=3,
    )

    assert np.allclose(E, [0.0, 0.0])
    assert np.allclose(matched_scores, [1.0, 1.0], atol=1e-12)
    assert len(events) == 1
    assert events[0].step == 3
    assert events[0].bands == (0, 1)
    assert bands.eigenvector_overlap_scores(prev_U, S, U).diagonal().min() > 1.0 - 1e-12


# ---------------------------------------------------------------------------
# LocalPath continuation on controlled toy Hamiltonians
# ---------------------------------------------------------------------------


def test_local_path_energy_predict_follows_smooth_crossing_energy_sheets() -> None:
    t = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    path = make_path(CrossingHamiltonianK1(), t, strategy="energy_predict")

    solved = path.solve_continuation(fix_gauge=False, energy_prediction_order=2)

    assert solved.matching_strategy == "energy_predict"
    assert solved.energies.shape == (len(t), 2)
    assert np.allclose(solved.energies[:, 0], t, atol=1e-12)
    assert np.allclose(solved.energies[:, 1], -t, atol=1e-12)


def test_local_path_state_overlap_keeps_high_overlap_for_smooth_rotating_eigenvectors() -> None:
    theta = np.linspace(0.0, 0.5, 41)
    path = make_path(RotatingTwoLevelHamiltonian(), theta, strategy="state_overlap")

    solved = path.solve_continuation(fix_gauge=True, align_degenerate=True)

    assert solved.matching_strategy == "state_overlap"
    assert solved.energies.shape == (len(theta), 2)
    assert np.allclose(solved.energies[:, 0], -1.0, atol=1e-12)
    assert np.allclose(solved.energies[:, 1], 1.0, atol=1e-12)
    assert np.min(solved.overlaps) > 0.999


def test_local_path_invalid_initial_EU_shape_raises() -> None:
    path = make_path(CrossingHamiltonianK1(), [-1.0, 0.0], strategy="energy_predict")

    with pytest.raises(ValueError, match="initial U must be 2D"):
        path.solve_continuation(initial_EU=(np.array([0.0]), np.array([1.0])))


def test_local_path_invalid_matching_strategy_raises() -> None:
    path = make_path(CrossingHamiltonianK1(), [-1.0, 0.0], strategy="bad_strategy")

    with pytest.raises(ValueError, match="Unknown matching strategy"):
        path.solve_continuation()


def test_local_path_outputs_are_readonly_after_solve() -> None:
    path = make_path(CrossingHamiltonianK1(), [-1.0, 0.0, 1.0], strategy="energy_predict")

    solved = path.solve_continuation(fix_gauge=False)

    assert not solved.energies.flags.writeable
    assert not solved.vectors.flags.writeable
    assert not solved.overlaps.flags.writeable


# ---------------------------------------------------------------------------
# LocalRegion strategy propagation and simple energy-sheet behaviour
# ---------------------------------------------------------------------------


def test_local_region_from_parallelogram_shapes_and_strategy(fake_S: IdentityOverlapKernel) -> None:
    region = bands.LocalRegion.from_parallelogram(
        CrossingHamiltonianK2(),
        fake_S,
        origin=(0.0, -1.0),
        edge_u=(1.0, 0.0),
        edge_v=(0.0, 2.0),
        nu=4,
        nv=5,
        units=eVag,
        matching_strategy="energy_predict",
    )

    assert region.shape == (4, 5)
    assert region.k1.shape == (4, 5)
    assert region.k2.shape == (4, 5)
    assert region.matching_strategy == "energy_predict"


def test_local_region_energy_predict_passes_strategy_to_seed_and_v_paths(fake_S: IdentityOverlapKernel) -> None:
    region = bands.LocalRegion.from_parallelogram(
        CrossingHamiltonianK2(),
        fake_S,
        origin=(0.0, -1.0),
        edge_u=(1.0, 0.0),
        edge_v=(0.0, 2.0),
        nu=4,
        nv=5,
        units=eVag,
        matching_strategy="energy_predict",
    )

    assert region.seed_edge_path().matching_strategy == "energy_predict"
    assert region.v_path(0).matching_strategy == "energy_predict"

    override = region.v_path(0, matching_strategy="state_overlap")
    assert override.matching_strategy == "state_overlap"


def test_local_region_energy_predict_tracks_crossing_sheets_along_v(fake_S: IdentityOverlapKernel) -> None:
    region = bands.LocalRegion.from_parallelogram(
        CrossingHamiltonianK2(),
        fake_S,
        origin=(0.0, -1.0),
        edge_u=(1.0, 0.0),
        edge_v=(0.0, 2.0),
        nu=3,
        nv=5,
        units=eVag,
        matching_strategy="energy_predict",
    )

    solved = region.solve(fix_gauge=False, energy_prediction_order=2)

    assert solved.matching_strategy == "energy_predict"
    assert solved.energies.shape == (3, 5, 2)
    assert solved.vectors.shape == (3, 5, 2, 2)
    assert solved.overlaps_u.shape == (2, 2)
    assert solved.overlaps_v.shape == (3, 4, 2)
    assert solved.transverse_overlaps_u.shape == (2, 5, 2)
    assert solved.transverse_orders_u.shape == (2, 5, 2)

    expected = solved.k2[0, :]
    for i in range(solved.nu):
        assert np.allclose(solved.energies[i, :, 0], expected, atol=1e-12)
        assert np.allclose(solved.energies[i, :, 1], -expected, atol=1e-12)


def test_local_region_v_path_bounds_check(fake_S: IdentityOverlapKernel) -> None:
    region = bands.LocalRegion.from_parallelogram(
        CrossingHamiltonianK2(),
        fake_S,
        origin=(0.0, -1.0),
        edge_u=(1.0, 0.0),
        edge_v=(0.0, 2.0),
        nu=3,
        nv=5,
        units=eVag,
    )

    with pytest.raises(IndexError):
        region.v_path(-1)

    with pytest.raises(IndexError):
        region.v_path(3)
