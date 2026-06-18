from __future__ import annotations

import numpy as np

from dft_local.core.units import CONDUCTIVITY, VELOCITY, quantity_array_specs
from dft_local.transport.boltzmann.strong_dc.core import (
    BOHR_TO_M,
    BandIndexedStrongDcResult,
    band_indexed_strong_dc_from_velocity_grid,
    fermi_factor,
    lattice_mode_indices,
    lattice_mode_vectors_m,
)


def test_lattice_mode_indices_have_expected_centered_order() -> None:
    modes = lattice_mode_indices((4, 3))

    assert modes.shape == (4, 3, 2)
    np.testing.assert_array_equal(modes[:, 0, 0], np.array([0, 1, -2, -1]))
    np.testing.assert_array_equal(modes[0, :, 1], np.array([0, 1, -1]))


def test_lattice_mode_vectors_convert_bohr_lattice_basis_rows_to_metres() -> None:
    primitive_lattice_vectors_bohr = np.array([[2.0, 0.0], [0.0, 3.0]])

    r = lattice_mode_vectors_m(primitive_lattice_vectors_bohr, (3, 3))

    assert r.shape == (3, 3, 2)
    np.testing.assert_allclose(r[1, 0], BOHR_TO_M * np.array([2.0, 0.0]))
    np.testing.assert_allclose(r[0, 1], BOHR_TO_M * np.array([0.0, 3.0]))
    np.testing.assert_allclose(r[2, 0], BOHR_TO_M * np.array([-2.0, 0.0]))
    np.testing.assert_allclose(r[0, 2], BOHR_TO_M * np.array([0.0, -3.0]))


def test_fermi_factor_midpoint_and_limits_are_stable() -> None:
    values = fermi_factor(np.array([-1.0e-18, 0.0, 1.0e-18]), 0.0, 300.0)

    assert values[0] > 0.5
    assert values[1] == 0.5
    assert values[2] < 0.5
    assert np.all(np.isfinite(values))


def test_band_indexed_strong_dc_returns_finite_tensor() -> None:
    epsilon_Ha = np.array([
        [-0.2, -0.1, -0.2],
        [-0.1, 0.0, -0.1],
        [-0.2, -0.1, -0.2],
    ])
    velocity_m_per_s = np.zeros(epsilon_Ha.shape + (2,), dtype=float)
    velocity_m_per_s[:, :, 0] = 1.0e5
    velocity_m_per_s[:, :, 1] = -2.0e5

    result = band_indexed_strong_dc_from_velocity_grid(
        epsilon_Ha,
        velocity_m_per_s,
        np.eye(2),
        chemical_potential_J=0.0,
        temperature_K=300.0,
        relaxation_time_s=1.0e-14,
    )

    assert result.conductivity_tensor_S.shape == (2, 2)
    assert result.conductivity_mode_tensor_S.shape == epsilon_Ha.shape + (2, 2)
    assert np.all(np.isfinite(result.conductivity_tensor_S))
    assert np.isfinite(result.imaginary_leakage_S)


def test_band_indexed_strong_dc_result_has_quantity_schema() -> None:
    specs = quantity_array_specs(BandIndexedStrongDcResult)

    assert specs["velocity_m_per_s"].dimension == VELOCITY
    assert specs["velocity_m_per_s"].axes == ("k1", "k2", "cartesian")
    assert specs["conductivity_tensor_S"].dimension == CONDUCTIVITY
    assert specs["conductivity_tensor_S"].axes == ("cartesian", "cartesian")
    assert specs["conductivity_mode_tensor_S"].axes == ("k1", "k2", "cartesian", "cartesian")
