from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from dft_local.energy_surface_rectification import (
    energy_surface_roughness,
    hungarian_order_from_costs,
    rectify_energy_arrays_across_u,
    rectify_local_region_energy_surfaces,
    transverse_energy_prediction,
    transverse_path_cost,
)


def make_smooth_region(nu=7, nv=9, nbands=3):
    """Create smooth separated synthetic energy sheets."""

    u = np.linspace(-1.0, 1.0, nu)
    v = np.linspace(-1.0, 1.0, nv)

    E = np.empty((nu, nv, nbands), dtype=np.float64)

    formulas = (
        lambda uu, vv: -2.0 + 0.2 * uu + 0.05 * vv + 0.02 * uu * vv,
        lambda uu, vv:  0.5 + 0.25 * uu - 0.03 * vv + 0.01 * uu**2,
        lambda uu, vv:  2.0 - 0.1 * uu + 0.08 * vv - 0.01 * vv**2,
    )

    for i, uu in enumerate(u):
        for j, vv in enumerate(v):
            for b in range(nbands):
                if b < len(formulas):
                    E[i, j, b] = formulas[b](uu, vv)
                else:
                    # Extra bands, if requested, are smooth and separated.
                    E[i, j, b] = (
                        2.0 * b
                        + 0.07 * (b + 1) * uu
                        - 0.04 * (b + 1) * vv
                        + 0.005 * uu * vv
                    )

    return E

def test_hungarian_order_from_costs_minimises_assignment():
    costs = np.array([[10.0, 0.0, 8.0], [2.0, 9.0, 1.0], [0.5, 3.0, 7.0]])
    order = hungarian_order_from_costs(costs)
    assert order.tolist() == [1, 2, 0]


def test_transverse_energy_prediction_constant_for_first_path():
    E = make_smooth_region(nu=3, nv=4, nbands=2)
    pred = transverse_energy_prediction(E, 1, prediction_order=1)
    np.testing.assert_allclose(pred, E[0])


def test_transverse_energy_prediction_linear_for_later_path():
    E = make_smooth_region(nu=4, nv=5, nbands=2)
    pred = transverse_energy_prediction(E, 2, prediction_order=1)
    np.testing.assert_allclose(pred, 2.0 * E[1] - E[0])


def test_transverse_path_cost_detects_permuted_path():
    E = make_smooth_region(nu=3, nv=8, nbands=3)
    cost = transverse_path_cost(E[0], E[1][:, [0, 2, 1]])
    order = hungarian_order_from_costs(cost)
    assert order.tolist() == [0, 2, 1]


def test_rectify_energy_arrays_fixes_whole_path_permutation():
    E = make_smooth_region(nu=8, nv=10, nbands=3)
    bad = np.array(E, copy=True)
    bad[4:, :, :] = bad[4:, :, [0, 2, 1]]

    before = energy_surface_roughness(bad)
    fixed, vectors, report = rectify_energy_arrays_across_u(
        bad,
        prediction_order=1,
        accept_ratio=0.999,
    )
    after = energy_surface_roughness(fixed)

    assert vectors is None
    assert np.any(report.accepted)
    assert after["rms_du2"] < 0.25 * before["rms_du2"]
    np.testing.assert_allclose(fixed, E, atol=1e-10)


def test_rectify_energy_arrays_preserves_vectors_with_same_order():
    E = make_smooth_region(nu=6, nv=7, nbands=3)
    bad = np.array(E, copy=True)
    bad[3:, :, :] = bad[3:, :, [2, 1, 0]]

    nu, nv, nbands = bad.shape
    dim = 4
    U = np.zeros((nu, nv, dim, nbands), dtype=np.complex128)
    for i in range(nu):
        for j in range(nv):
            for b in range(nbands):
                U[i, j, b % dim, b] = 1.0 + 0.0j

    U_bad = np.array(U, copy=True)
    U_bad[3:, :, :, :] = U_bad[3:, :, :, [2, 1, 0]]

    fixed, U_fixed, _report = rectify_energy_arrays_across_u(
        bad,
        U_bad,
        prediction_order=1,
        accept_ratio=0.999,
    )

    np.testing.assert_allclose(fixed, E, atol=1e-10)
    np.testing.assert_allclose(U_fixed, U, atol=1e-10)


def test_rectify_energy_arrays_rejects_insignificant_permutation():
    E = make_smooth_region(nu=6, nv=8, nbands=3)
    fixed, _U, report = rectify_energy_arrays_across_u(
        E,
        prediction_order=1,
        accept_ratio=0.5,
    )
    assert not np.any(report.accepted)
    np.testing.assert_allclose(fixed, E)


def test_rectify_energy_arrays_validates_shape():
    with pytest.raises(ValueError):
        rectify_energy_arrays_across_u(np.zeros((3, 4)))


def test_roughness_zero_for_linear_surfaces():
    nu, nv, nbands = 5, 6, 2
    u = np.arange(nu)[:, None, None]
    v = np.arange(nv)[None, :, None]
    b = np.arange(nbands)[None, None, :]
    E = 1.0 + 2.0 * u + 3.0 * v + 10.0 * b

    rough = energy_surface_roughness(E.astype(np.float64))
    assert rough["max_abs_du2"] == pytest.approx(0.0)
    assert rough["max_abs_dv2"] == pytest.approx(0.0)
    assert rough["max_abs_duv"] == pytest.approx(0.0)


@dataclass(frozen=True, slots=True)
class FakeRegion:
    energies: np.ndarray
    vectors: np.ndarray | None = None
    energy_rectification_orders: np.ndarray | None = None
    energy_rectification_costs: np.ndarray | None = None
    energy_rectification_accepted: np.ndarray | None = None


def test_rectify_local_region_energy_surfaces_updates_metadata():
    E = make_smooth_region(nu=6, nv=7, nbands=3)
    bad = np.array(E, copy=True)
    bad[3:, :, :] = bad[3:, :, [0, 2, 1]]
    region = FakeRegion(energies=bad)

    out = rectify_local_region_energy_surfaces(
        region,
        prediction_order=1,
        accept_ratio=0.999,
    )

    assert out.energy_rectification_orders is not None
    assert out.energy_rectification_costs is not None
    assert out.energy_rectification_accepted is not None
    assert np.any(out.energy_rectification_accepted)
    np.testing.assert_allclose(out.energies, E, atol=1e-10)
