from __future__ import annotations

import numpy as np

from dft_local.transport.boltzmann.group_resolved.core import (
    BandResolvedConductivity,
    band_resolved_compact_conductivity,
)


def test_group_resolved_domain_exists() -> None:
    import dft_local.transport.boltzmann.group_resolved.core as core

    assert core.__name__.endswith("group_resolved.core")


def test_band_resolved_compact_conductivity_sums_to_scalar_tensor() -> None:
    velocities = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ],
        ],
        dtype=np.float64,
    )
    weights = np.array(
        [
            [0.5, 0.25],
            [0.125, 0.0625],
        ],
        dtype=np.complex128,
    )
    k_weights = np.array([2.0, 3.0], dtype=np.float64)

    result = band_resolved_compact_conductivity(
        velocities=velocities,
        weights=weights,
        physical_k_weights=k_weights,
    )

    assert isinstance(result, BandResolvedConductivity)
    assert result.sigma_band_k.shape == (2, 2, 2, 2)
    assert result.sigma_band.shape == (2, 2, 2)
    assert result.sigma.shape == (2, 2)

    direct = np.zeros((2, 2), dtype=np.complex128)
    for s in range(2):
        for n in range(2):
            direct += k_weights[s] * weights[s, n] * np.outer(
                velocities[s, :, n],
                velocities[s, :, n],
            )

    assert np.allclose(result.sigma, direct)
    assert np.allclose(result.sigma, np.sum(result.sigma_band, axis=0))


def test_band_resolved_compact_conductivity_rejects_shape_mismatch() -> None:
    velocities = np.zeros((2, 2, 3), dtype=np.float64)
    weights = np.zeros((2, 2), dtype=np.complex128)
    k_weights = np.ones(2, dtype=np.float64)

    try:
        band_resolved_compact_conductivity(
            velocities=velocities,
            weights=weights,
            physical_k_weights=k_weights,
        )
    except ValueError as exc:
        assert "weights shape mismatch" in str(exc)
    else:
        raise AssertionError("expected shape mismatch")



def test_band_resolved_compact_matches_existing_boltzmann_sigma() -> None:
    from dft_local.transport.boltzmann.calculation.test_conductivity_business_logic import (
        cosine_k1_kernel,
        identity_overlap_kernel,
        make_calc,
    )

    calc = make_calc(
        cosine_k1_kernel(),
        identity_overlap_kernel(),
        [np.pi / 2, -np.pi / 2],
        [0.0, 0.0],
        tau=2.0,
    ).run()

    assert calc.velocities is not None
    assert calc.ac_weights is not None
    assert calc.sigma is not None

    result = band_resolved_compact_conductivity(
        velocities=calc.velocities,
        weights=calc.ac_weights,
        physical_k_weights=calc.physical_k_weights,
    )

    assert np.allclose(result.sigma, calc.sigma)
    assert np.allclose(np.sum(result.sigma_band, axis=0), calc.sigma)
