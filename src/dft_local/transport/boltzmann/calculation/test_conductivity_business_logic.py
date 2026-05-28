"""Business-logic and physical-regression tests for Boltzmann conductivity.

Copied from the repository-level Boltzmann conductivity tests so the
transport domain owns its important physical checks locally.
"""

import numpy as np
import pytest

from dft_local.core.kernels import GdKernelArrays
from dft_local.core.local_problem import SymbolPair
from dft_local.core.numerics import AU, Units, eVag
from dft_local.transport.boltzmann.calculation.core import (
    K_B_HARTREE_PER_K,
    BoltzmannConductivity,
    fermi_window,
    gd_symbol_derivative_fixed,
    gd_symbol_derivative_generic,
    gd_symbol_derivatives,
)


def kernel(
    h_m,
    h_n,
    h_eps,
    blocks,
    *,
    name: str = "kernel",
) -> GdKernelArrays:
    return GdKernelArrays(
        h_m=np.asarray(h_m, dtype=np.int64),
        h_n=np.asarray(h_n, dtype=np.int64),
        h_eps=np.asarray(h_eps, dtype=np.int64),
        blocks=np.asarray(blocks, dtype=np.complex128),
        matrix_name=name,
    )


def identity_overlap_kernel(q: int = 1) -> GdKernelArrays:
    return kernel(
        [0],
        [0],
        [0],
        [np.eye(q)],
        name="identity S",
    )


def cosine_k1_kernel(scale: float = 1.0, q: int = 1) -> GdKernelArrays:
    return kernel(
        [1, -1],
        [0, 0],
        [0, 0],
        [0.5 * scale * np.eye(q), 0.5 * scale * np.eye(q)],
        name="cos k1",
    )


def cosine_k2_kernel(scale: float = 1.0, q: int = 1) -> GdKernelArrays:
    return kernel(
        [0, 0],
        [1, -1],
        [0, 0],
        [0.5 * scale * np.eye(q), 0.5 * scale * np.eye(q)],
        name="cos k2",
    )


def diagonal_cosine_k1_kernel(scales: list[float]) -> GdKernelArrays:
    D = np.diag(scales).astype(np.complex128)

    return kernel(
        [1, -1],
        [0, 0],
        [0, 0],
        [0.5 * D, 0.5 * D],
        name="diagonal cos k1",
    )


def shifted_overlap_k1_kernel(a: float) -> GdKernelArrays:
    return kernel(
        [0, 1, -1],
        [0, 0, 0],
        [0, 0, 0],
        [[[1.0]], [[0.5 * a]], [[0.5 * a]]],
        name="1 + a cos k1",
    )


def mixed_even_odd_kernel() -> GdKernelArrays:
    return kernel(
        [0, 1, -2, 1],
        [0, -1, 2, 3],
        [0, 0, 1, 1],
        [
            [[1.0 + 0.0j, 0.2 - 0.1j], [0.3 + 0.4j, -0.7 + 0.2j]],
            [[0.5 - 0.2j, 0.0 + 0.3j], [-0.1 + 0.2j, 0.8 + 0.0j]],
            [[-0.4 + 0.6j, 0.7 - 0.1j], [0.2 + 0.5j, 0.1 - 0.2j]],
            [[0.6 + 0.1j, -0.3 + 0.8j], [0.9 - 0.4j, -0.2 + 0.7j]],
        ],
        name="mixed even odd",
    )


def make_calc(
    KH: GdKernelArrays,
    KS: GdKernelArrays,
    k1,
    k2,
    *,
    weights=None,
    irrep_to_physical_k=None,
    units=AU,
    mu=0.0,
    temperature=None,
    omega=0.0,
    tau=1.0,
    charge=None,
) -> BoltzmannConductivity:
    k1 = np.asarray(k1, dtype=np.float64)
    k2 = np.asarray(k2, dtype=np.float64)

    if weights is None:
        weights = np.full(k1.size, 1.0 / k1.size, dtype=np.float64)

    if temperature is None:
        # Gives k_B T = 1 in AU
        temperature = 1.0 / K_B_HARTREE_PER_K

    if irrep_to_physical_k is None:
        irrep_to_physical_k = np.eye(2)

    return BoltzmannConductivity.from_arrays(
        KH,
        KS,
        k1,
        k2,
        irrep_weights=np.asarray(weights, dtype=np.float64),
        irrep_to_physical_k=np.asarray(irrep_to_physical_k, dtype=np.float64),
        units=units,
        mu=mu,
        temperature=temperature,
        omega=omega,
        tau=tau,
        charge=charge,
    )


def test_fermi_window_at_mu_in_au() -> None:
    E = np.array([0.0], dtype=np.float64)
    temperature = 1.0 / K_B_HARTREE_PER_K

    window = fermi_window(E, mu=0.0, temperature=temperature, units=AU)

    assert np.allclose(window, [0.25])


def test_fermi_window_at_mu_in_evag() -> None:
    E = np.array([1.5], dtype=np.float64)
    mu = 1.5
    temperature = 300.0

    window = fermi_window(E, mu=mu, temperature=temperature, units=eVag)

    kBT = K_B_HARTREE_PER_K * eVag.E * temperature
    assert np.allclose(window, [1.0 / (4.0 * kBT)])


def test_fermi_window_is_symmetric_about_mu() -> None:
    temperature = 1.0 / K_B_HARTREE_PER_K
    E = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

    window = fermi_window(E, mu=0.0, temperature=temperature, units=AU)

    assert np.allclose(window[0], window[-1])
    assert np.allclose(window[1], window[-2])
    assert window[2] > window[1] > window[0]


def test_fermi_window_large_arguments_remain_finite() -> None:
    temperature = 1.0 / K_B_HARTREE_PER_K
    E = np.array([-1000.0, 0.0, 1000.0])

    window = fermi_window(E, mu=0.0, temperature=temperature, units=AU)

    assert np.all(np.isfinite(window))
    assert np.all(window >= 0.0)
    assert window[0] == 0.0
    assert window[2] == 0.0


def test_fermi_window_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        fermi_window(
            np.array([0.0]),
            mu=0.0,
            temperature=0.0,
            units=AU,
        )


@pytest.mark.parametrize("axis", [0, 1])
def test_generic_symbol_derivative_matches_finite_difference(axis: int) -> None:
    K = mixed_even_odd_kernel()

    k1 = 0.37
    k2 = -0.12
    eps = 1e-6

    analytic = gd_symbol_derivative_generic(K, k1, k2, axis=axis)

    if axis == 0:
        plus = K.symbol_generic(k1 + eps, k2)
        minus = K.symbol_generic(k1 - eps, k2)
    else:
        plus = K.symbol_generic(k1, k2 + eps)
        minus = K.symbol_generic(k1, k2 - eps)

    numeric = (plus - minus) / (2.0 * eps)

    assert np.allclose(analytic, numeric, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("sigma", [-1, 1])
def test_fixed_symbol_derivative_matches_finite_difference(axis: int, sigma: int) -> None:
    K = mixed_even_odd_kernel()

    k1 = 0.29
    k2 = -0.41
    eps = 1e-6

    analytic = gd_symbol_derivative_fixed(K, k1, k2, sigma=sigma, axis=axis)

    if axis == 0:
        plus = K.symbol_fixed(k1 + eps, k2, sigma=sigma)
        minus = K.symbol_fixed(k1 - eps, k2, sigma=sigma)
    else:
        plus = K.symbol_fixed(k1, k2 + eps, sigma=sigma)
        minus = K.symbol_fixed(k1, k2 - eps, sigma=sigma)

    numeric = (plus - minus) / (2.0 * eps)

    assert np.allclose(analytic, numeric, atol=1e-8, rtol=1e-8)


def test_gd_symbol_derivatives_reject_bad_axis_indirectly() -> None:
    K = mixed_even_odd_kernel()

    with pytest.raises(ValueError, match="axis"):
        gd_symbol_derivative_generic(K, 0.0, 0.0, axis=2)

    with pytest.raises(ValueError, match="axis"):
        gd_symbol_derivative_fixed(K, 0.0, 0.0, sigma=1, axis=2)


def test_gd_symbol_derivatives_for_generic_pair_have_two_entries() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    pair = SymbolPair(KH=KH, KS=KS, k1=0.1, k2=0.2)
    dH = gd_symbol_derivatives(pair, KH)

    assert len(dH) == 2
    assert dH[0].shape == pair.form().Hk.shape
    assert dH[1].shape == pair.form().Hk.shape


def test_gd_symbol_derivatives_for_fixed_pair_have_two_entries() -> None:
    KH = cosine_k1_kernel()
    pair = SymbolPair(KH=KH, KS=identity_overlap_kernel(), k1=0.1, k2=0.2, degree=1, sigma=1)

    dH = gd_symbol_derivatives(pair, KH)

    assert len(dH) == 2
    assert dH[0].shape == (1, 1)
    assert dH[1].shape == (1, 1)


def test_gd_symbol_derivatives_rejects_missing_sigma_for_fixed_pair() -> None:
    KH = cosine_k1_kernel()
    pair = SymbolPair(KH=KH, KS=identity_overlap_kernel(), k1=0.1, k2=0.2, degree=1, sigma=None)

    with pytest.raises(ValueError, match="sigma"):
        gd_symbol_derivatives(pair, KH)


def test_from_arrays_rejects_k_shape_mismatch() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    with pytest.raises(ValueError, match="shape mismatch"):
        BoltzmannConductivity.from_arrays(
            KH,
            KS,
            np.array([0.0, 1.0]),
            np.array([0.0]),
            units=AU,
        )


def test_from_arrays_rejects_weight_shape_mismatch() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    with pytest.raises(ValueError, match="irrep_weights"):
        BoltzmannConductivity.from_arrays(
            KH,
            KS,
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            irrep_weights=np.array([1.0]),
            units=AU,
        )


def test_constructor_rejects_non_square_irrep_map() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()
    problem = SymbolPair(KH=KH, KS=KS, k1=0.0, k2=0.0).form()

    with pytest.raises(ValueError, match="square"):
        BoltzmannConductivity(
            problems=np.array([problem], dtype=object),
            irrep_points=np.array([[0.0, 0.0]]),
            irrep_weights=np.array([1.0]),
            irrep_to_physical_k=np.ones((2, 3)),
            units=AU,
        )


def test_constructor_rejects_singular_irrep_map() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()
    problem = SymbolPair(KH=KH, KS=KS, k1=0.0, k2=0.0).form()

    with pytest.raises(ValueError, match="invertible"):
        BoltzmannConductivity(
            problems=np.array([problem], dtype=object),
            irrep_points=np.array([[0.0, 0.0]]),
            irrep_weights=np.array([1.0]),
            irrep_to_physical_k=np.array([[1.0, 0.0], [0.0, 0.0]]),
            units=AU,
        )


def test_constructor_rejects_problem_count_mismatch() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()
    problem = SymbolPair(KH=KH, KS=KS, k1=0.0, k2=0.0).form()

    with pytest.raises(ValueError, match="number of problems"):
        BoltzmannConductivity(
            problems=np.array([problem], dtype=object),
            irrep_points=np.array([[0.0, 0.0], [1.0, 1.0]]),
            irrep_weights=np.array([0.5, 0.5]),
            irrep_to_physical_k=np.eye(2),
            units=AU,
        )


def test_constructor_rejects_negative_temperature() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    with pytest.raises(ValueError, match="temperature"):
        make_calc(
            KH,
            KS,
            [0.0],
            [0.0],
            temperature=-1.0,
        )


def test_run_rejects_non_local_problem() -> None:
    calc = BoltzmannConductivity(
        problems=np.array([object()], dtype=object),
        irrep_points=np.array([[0.0, 0.0]]),
        irrep_weights=np.array([1.0]),
        irrep_to_physical_k=np.eye(2),
        units=AU,
    )

    with pytest.raises(TypeError, match="LocalProblem"):
        calc.run()


def test_run_rejects_non_positive_overlap() -> None:
    KH = cosine_k1_kernel()
    KS = kernel([0], [0], [0], [[[-1.0]]], name="bad S")

    calc = make_calc(KH, KS, [0.0], [0.0])

    with pytest.raises(ValueError, match="positive definite"):
        calc.run()


def test_cosine_k1_single_sample_energies_velocities_and_sigma() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    calc = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        tau=2.0,
    ).run()

    assert calc.energies is not None
    assert calc.velocities is not None
    assert calc.ac_weights is not None
    assert calc.sigma is not None

    # Generic degree-2 irrep gives two identical components for this scalar kernel.
    assert calc.energies.shape == (1, 2)
    assert np.allclose(calc.energies[0], [0.0, 0.0], atol=1e-12)

    assert calc.velocities.shape == (1, 2, 2)
    assert np.allclose(calc.velocities[0, 0], [-1.0, -1.0], atol=1e-12)
    assert np.allclose(calc.velocities[0, 1], [0.0, 0.0], atol=1e-12)

    # k_B T = 1, E = mu gives window = 1 / 4.
    # tau = 2 gives weight = 1 / 2 per band.
    assert np.allclose(calc.ac_weights[0], [0.5, 0.5])

    expected = 1.0 / (2.0 * np.pi) ** 2

    assert np.allclose(calc.sigma[0, 0], expected)
    assert np.allclose(calc.sigma[0, 1], 0.0)
    assert np.allclose(calc.sigma[1, 0], 0.0)
    assert np.allclose(calc.sigma[1, 1], 0.0)


def test_cosine_k2_gives_y_velocity_only() -> None:
    KH = cosine_k2_kernel()
    KS = identity_overlap_kernel()

    calc = make_calc(
        KH,
        KS,
        [0.0],
        [np.pi / 2],
        tau=2.0,
    ).run()

    assert calc.velocities is not None
    assert calc.sigma is not None

    assert np.allclose(calc.velocities[0, 0], [0.0, 0.0], atol=1e-12)
    assert np.allclose(calc.velocities[0, 1], [-1.0, -1.0], atol=1e-12)

    expected = 1.0 / (2.0 * np.pi) ** 2

    assert np.allclose(calc.sigma[0, 0], 0.0)
    assert np.allclose(calc.sigma[0, 1], 0.0)
    assert np.allclose(calc.sigma[1, 0], 0.0)
    assert np.allclose(calc.sigma[1, 1], expected)


def test_multiple_q_channels_have_expected_velocities_up_to_degenerate_order() -> None:
    KH = diagonal_cosine_k1_kernel([1.0, 2.0])
    KS = identity_overlap_kernel(q=2)

    calc = make_calc(
        KH,
        KS,
        [np.pi / 3],
        [0.0],
        tau=1.0,
    ).run()

    assert calc.energies is not None
    assert calc.velocities is not None

    expected_energies = [0.5, 0.5, 1.0, 1.0]
    expected_vx = [-np.sqrt(3) / 2, -np.sqrt(3) / 2, -np.sqrt(3), -np.sqrt(3)]

    assert np.allclose(np.sort(calc.energies[0]), expected_energies)
    assert np.allclose(np.sort(calc.velocities[0, 0]), np.sort(expected_vx))
    assert np.allclose(calc.velocities[0, 1], 0.0)


def test_k_dependent_overlap_velocity_matches_derivative_of_ratio() -> None:
    a = 0.2

    KH = cosine_k1_kernel()
    KS = shifted_overlap_k1_kernel(a)

    k = np.pi / 3

    calc = make_calc(
        KH,
        KS,
        [k],
        [0.0],
        tau=1.0,
    ).run()

    assert calc.energies is not None
    assert calc.velocities is not None

    H = np.cos(k)
    dH = -np.sin(k)
    S = 1.0 + a * np.cos(k)
    dS = -a * np.sin(k)

    expected_E = H / S
    expected_v = (dH * S - H * dS) / S**2

    assert np.allclose(calc.energies[0], [expected_E, expected_E])
    assert np.allclose(calc.velocities[0, 0], [expected_v, expected_v])
    assert np.allclose(calc.velocities[0, 1], [0.0, 0.0])


def test_velocity_matrix_diagonal_matches_band_velocities() -> None:
    KH = diagonal_cosine_k1_kernel([1.0, 2.0])
    KS = identity_overlap_kernel(q=2)

    calc = make_calc(KH, KS, [np.pi / 3], [0.0])
    problem = calc.prepared_problem(0)
    E, U = calc.solve_problem(problem)

    velocities = calc.band_velocities(problem, E, U)
    Vx = calc.velocity_matrix_eig(problem, E, U, direction=0)
    Vy = calc.velocity_matrix_eig(problem, E, U, direction=1)

    assert np.allclose(np.diag(Vx).real, velocities[0])
    assert np.allclose(np.diag(Vy).real, velocities[1])


def test_velocity_matrix_for_scalar_cosine_is_hermitian() -> None:
    KH = cosine_k1_kernel()
    KS = shifted_overlap_k1_kernel(0.2)

    calc = make_calc(KH, KS, [0.7], [0.0])
    problem = calc.prepared_problem(0)
    E, U = calc.solve_problem(problem)

    Vx = calc.velocity_matrix_eig(problem, E, U, direction=0)

    assert np.allclose(Vx, Vx.conj().T)


def test_irrep_to_physical_k_scales_velocity_by_inverse_jacobian_direction() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    base = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        irrep_to_physical_k=np.eye(2),
    ).run()

    scaled = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        irrep_to_physical_k=np.array([[2.0, 0.0], [0.0, 1.0]]),
    ).run()

    assert base.velocities is not None
    assert scaled.velocities is not None

    assert np.allclose(scaled.velocities[0, 0], 0.5 * base.velocities[0, 0])
    assert np.allclose(scaled.velocities[0, 1], base.velocities[0, 1])


def test_irrep_to_physical_k_jacobian_and_velocity_conversion_cancel_for_sigma_xx_in_2d_scaling() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    base = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        irrep_to_physical_k=np.eye(2),
        tau=2.0,
    ).run()

    scaled = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        irrep_to_physical_k=2.0 * np.eye(2),
        tau=2.0,
    ).run()

    assert base.sigma is not None
    assert scaled.sigma is not None

    assert np.allclose(scaled.sigma[0, 0], base.sigma[0, 0])
    assert np.allclose(scaled.sigma[0, 1], base.sigma[0, 1])
    assert np.allclose(scaled.sigma[1, 0], base.sigma[1, 0])


def test_hbar_units_scale_velocity() -> None:
    slow_units = Units(
        E=1.0,
        L=1.0,
        e=1.0,
        hbar=2.0,
        name="hbar2",
    )

    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    base = make_calc(KH, KS, [np.pi / 2], [0.0], units=AU).run()
    slow = make_calc(KH, KS, [np.pi / 2], [0.0], units=slow_units).run()

    assert base.velocities is not None
    assert slow.velocities is not None

    assert np.allclose(slow.velocities, 0.5 * base.velocities)


def test_charge_sign_does_not_affect_longitudinal_weight() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    negative = make_calc(KH, KS, [np.pi / 2], [0.0], charge=-3.0).run()
    positive = make_calc(KH, KS, [np.pi / 2], [0.0], charge=3.0).run()

    assert negative.ac_weights is not None
    assert positive.ac_weights is not None
    assert np.allclose(negative.ac_weights, positive.ac_weights)


def test_ac_weight_matches_manual_complex_value() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    omega = 4.0
    tau = 0.25

    calc = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        omega=omega,
        tau=tau,
        charge=2.0,
    ).run()

    assert calc.ac_weights is not None

    expected = 4.0 * tau / (1.0 - 1j * omega * tau) * 0.25

    assert np.allclose(calc.ac_weights[0], [expected, expected])


def test_tau_scalar_band_and_sample_forms_agree_for_constant_values() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    k1 = np.array([np.pi / 2, -np.pi / 2])
    k2 = np.array([0.0, 0.0])

    scalar = make_calc(KH, KS, k1, k2, tau=2.0).run()
    band = make_calc(KH, KS, k1, k2, tau=np.array([2.0, 2.0])).run()
    sample = make_calc(KH, KS, k1, k2, tau=np.array([[2.0, 2.0], [2.0, 2.0]])).run()

    assert scalar.sigma is not None
    assert band.sigma is not None
    assert sample.sigma is not None

    assert np.allclose(band.sigma, scalar.sigma)
    assert np.allclose(sample.sigma, scalar.sigma)


def test_tau_wrong_shape_raises_on_run() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    calc = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        tau=np.array([1.0, 2.0, 3.0]),
    )

    with pytest.raises(ValueError, match="tau"):
        calc.run()


def test_negative_tau_raises_on_run() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    calc = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        tau=-1.0,
    )

    with pytest.raises(ValueError, match="tau"):
        calc.run()


def test_irregular_sample_weights_match_manual_weighted_sum() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    k1 = np.array([np.pi / 2, -np.pi / 2])
    k2 = np.array([0.0, 0.0])
    weights = np.array([0.25, 0.75])

    calc = make_calc(
        KH,
        KS,
        k1,
        k2,
        weights=weights,
        tau=2.0,
    ).run()

    assert calc.sigma is not None

    # At both samples E=0, window=1/4, tau=2, each band weight=1/2.
    # Two generic bands with v_x^2=1 give sample integrand 1 before k measure.
    expected = np.sum(weights) / (2.0 * np.pi) ** 2

    assert np.allclose(calc.sigma[0, 0], expected)
    assert np.allclose(calc.sigma[0, 1], 0.0)
    assert np.allclose(calc.sigma[1, 0], 0.0)
    assert np.allclose(calc.sigma[1, 1], 0.0)


def test_sample_order_does_not_change_total_sigma() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    k1 = np.array([np.pi / 2, np.pi / 3, -np.pi / 2])
    k2 = np.array([0.0, 0.0, 0.0])
    weights = np.array([0.2, 0.3, 0.5])

    calc_a = make_calc(KH, KS, k1, k2, weights=weights, tau=2.0).run()

    order = np.array([2, 0, 1])
    calc_b = make_calc(
        KH,
        KS,
        k1[order],
        k2[order],
        weights=weights[order],
        tau=2.0,
    ).run()

    assert calc_a.sigma is not None
    assert calc_b.sigma is not None

    assert np.allclose(calc_a.sigma, calc_b.sigma)


def test_sigma_k_sums_to_sigma() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    calc = make_calc(
        KH,
        KS,
        [np.pi / 2, np.pi / 3, -np.pi / 2],
        [0.0, 0.0, 0.0],
        weights=[0.2, 0.3, 0.5],
        tau=2.0,
    ).run()

    assert calc.sigma is not None
    assert calc.sigma_k is not None

    assert np.allclose(calc.sigma, np.sum(calc.sigma_k, axis=0))


def test_run_stores_arrays_with_expected_shapes() -> None:
    KH = diagonal_cosine_k1_kernel([1.0, 2.0])
    KS = identity_overlap_kernel(q=2)

    calc = make_calc(
        KH,
        KS,
        [np.pi / 2, np.pi / 3],
        [0.0, 0.0],
        tau=1.0,
    ).run()

    assert calc.energies is not None
    assert calc.vectors is not None
    assert calc.velocities is not None
    assert calc.ac_weights is not None
    assert calc.sigma_k is not None
    assert calc.sigma is not None

    assert calc.energies.shape == (2, 4)
    assert calc.vectors.shape == (2, 4, 4)
    assert calc.velocities.shape == (2, 2, 4)
    assert calc.ac_weights.shape == (2, 4)
    assert calc.sigma_k.shape == (2, 2, 2)
    assert calc.sigma.shape == (2, 2)


def test_require_solved_raises_before_run() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    calc = make_calc(KH, KS, [0.0], [0.0])

    with pytest.raises(ValueError, match="not been run"):
        calc.require_solved()


def test_require_solved_passes_after_run() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    calc = make_calc(KH, KS, [np.pi / 2], [0.0]).run()

    calc.require_solved()


def test_diagnostics_before_run() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    calc = make_calc(KH, KS, [0.0], [0.0], tau=1.0)
    d = calc.diagnostics()

    assert d["solved"] is False
    assert d["nk"] == 1
    assert d["dimension"] == 2
    assert "finite" not in d


def test_diagnostics_after_run() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    calc = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        tau=1.0,
    ).run()

    d = calc.diagnostics()

    assert d["solved"] is True
    assert d["finite"] is True
    assert d["nk"] == 1
    assert d["nbands"] == 2
    assert d["velocity_abs_max"] == 1.0
    assert d["sigma_norm"] > 0.0


def test_diagnostics_rows_are_key_value_pairs() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    calc = make_calc(KH, KS, [np.pi / 2], [0.0]).run()
    rows = calc.diagnostics_rows()

    assert rows
    assert all(isinstance(row, list) for row in rows)
    assert all(len(row) == 2 for row in rows)


def test_compute_sample_matches_stored_first_sample_after_run() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    calc = make_calc(
        KH,
        KS,
        [np.pi / 2, np.pi / 3],
        [0.0, 0.0],
        weights=[0.4, 0.6],
        tau=2.0,
    )

    direct = calc.compute_sample(0)
    calc.run()

    assert calc.energies is not None
    assert calc.velocities is not None
    assert calc.ac_weights is not None
    assert calc.sigma_k is not None

    assert np.allclose(calc.energies[0], direct.energies)
    assert np.allclose(calc.velocities[0], direct.velocities)
    assert np.allclose(calc.ac_weights[0], direct.ac_weights)
    assert np.allclose(calc.sigma_k[0], direct.sigma)


def test_physical_k_points_property() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    J = np.array([[2.0, 1.0], [0.5, -1.0]])

    calc = make_calc(
        KH,
        KS,
        [0.1, 0.2],
        [0.3, 0.4],
        irrep_to_physical_k=J,
    )

    expected = calc.irrep_points @ J.T

    assert np.allclose(calc.physical_k_points, expected)


def test_physical_k_weights_property() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    J = np.array([[2.0, 0.0], [0.0, 3.0]])
    raw_weights = np.array([0.25, 0.75])

    calc = make_calc(
        KH,
        KS,
        [0.1, 0.2],
        [0.3, 0.4],
        weights=raw_weights,
        irrep_to_physical_k=J,
    )

    expected = raw_weights * 6.0 / (2.0 * np.pi) ** 2

    assert np.allclose(calc.physical_k_weights, expected)


def test_ac_conductivity_becomes_complex_when_omega_nonzero() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    calc = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        tau=2.0,
        omega=3.0,
    ).run()

    assert calc.sigma is not None

    assert abs(calc.sigma[0, 0].imag) > 0.0


def test_dc_conductivity_is_real_for_real_kernel() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    calc = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        tau=2.0,
        omega=0.0,
    ).run()

    assert calc.sigma is not None

    assert np.allclose(calc.sigma.imag, 0.0)


def test_y_independent_kernel_has_zero_y_conductivity_for_many_samples() -> None:
    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    calc = make_calc(
        KH,
        KS,
        [0.2, 0.7, 1.3],
        [-0.4, 0.9, 2.0],
        weights=[0.2, 0.3, 0.5],
        tau=1.0,
    ).run()

    assert calc.velocities is not None
    assert calc.sigma is not None

    assert np.allclose(calc.velocities[:, 1, :], 0.0)
    assert np.allclose(calc.sigma[0, 1], 0.0)
    assert np.allclose(calc.sigma[1, 0], 0.0)
    assert np.allclose(calc.sigma[1, 1], 0.0)

def scaled_kernel_energy(K: GdKernelArrays, energy_scale: float) -> GdKernelArrays:
    """
    Scale kernel blocks by an energy conversion factor.

    Used to represent the same Hamiltonian in a different energy unit.
    For AU -> eVag, multiply H blocks by eVag.E.
    """

    return GdKernelArrays(
        h_m=K.h_m.copy(),
        h_n=K.h_n.copy(),
        h_eps=K.h_eps.copy(),
        blocks=energy_scale * K.blocks.copy(),
        matrix_name=f"{K.matrix_name} scaled by {energy_scale}",
    )


def same_physical_cosine_calc(
    *,
    units: Units,
    energy_scale: float,
    mu: float,
    tau: float,
    omega: float,
    k1: float = np.pi / 2,
) -> BoltzmannConductivity:
    """
    Build same physical cosine band in a chosen unit system.

    The disk/AU band is

        E(alpha) = cos(alpha)

    in Hartree, with alpha dimensionless. In another unit system, the
    Hamiltonian is multiplied by `units.E`, and physical k is alpha / units.L.
    """

    KH = scaled_kernel_energy(cosine_k1_kernel(), energy_scale)
    KS = identity_overlap_kernel()

    return make_calc(
        KH,
        KS,
        [k1],
        [0.0],
        weights=[1.0],
        irrep_to_physical_k=np.eye(2) / units.L,
        units=units,
        mu=mu,
        temperature=300.0,
        omega=omega,
        tau=tau,
    ).run()


def test_same_physical_velocity_au_vs_evag_converts_by_length_over_time() -> None:
    """
    Same physical band should give velocities related by the unit conversion.

    AU result has velocity unit

        Bohr / atomic-time

    eVag result has velocity unit

        Angstrom / second

    The expected conversion is

        v_eVag = v_AU * eVag.L / (eVag.hbar / eVag.E)

    because hbar / E is the time unit in seconds.
    """

    au = same_physical_cosine_calc(
        units=AU,
        energy_scale=1.0,
        mu=0.0,
        tau=1.0,
        omega=0.0,
    )

    evag = same_physical_cosine_calc(
        units=eVag,
        energy_scale=eVag.E,
        mu=0.0,
        tau=1.0,
        omega=0.0,
    )

    assert au.velocities is not None
    assert evag.velocities is not None

    time_unit_seconds = eVag.hbar / eVag.E
    expected_factor = eVag.L / time_unit_seconds

    assert np.allclose(
        evag.velocities[0, 0],
        expected_factor * au.velocities[0, 0],
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.allclose(evag.velocities[0, 1], 0.0)


def test_same_physical_fermi_window_au_vs_evag_converts_as_inverse_energy() -> None:
    """
    Same physical energy window should transform as inverse energy.

        [-df/dE]_eV = [-df/dE]_Hartree / eVag.E
    """

    temperature = 300.0

    E_au = np.array([0.0, 0.01, -0.02])
    E_ev = eVag.E * E_au

    win_au = fermi_window(
        E_au,
        mu=0.0,
        temperature=temperature,
        units=AU,
    )

    win_ev = fermi_window(
        E_ev,
        mu=0.0,
        temperature=temperature,
        units=eVag,
    )

    assert np.allclose(win_ev, win_au / eVag.E, rtol=1e-12, atol=1e-12)


def test_same_physical_mu_must_be_converted_with_energy_unit() -> None:
    """
    If the Hamiltonian is converted from Hartree to eV, mu must also be
    converted from Hartree to eV.

    This catches silent mistakes where energies are in eV but mu is still in
    Hartree.
    """

    temperature = 300.0
    mu_au = 0.01
    mu_ev = mu_au * eVag.E

    E_au = np.array([mu_au])
    E_ev = np.array([mu_ev])

    win_au = fermi_window(
        E_au,
        mu=mu_au,
        temperature=temperature,
        units=AU,
    )

    win_ev = fermi_window(
        E_ev,
        mu=mu_ev,
        temperature=temperature,
        units=eVag,
    )

    assert np.allclose(win_ev, win_au / eVag.E, rtol=1e-12, atol=1e-12)

    wrong_win_ev = fermi_window(
        E_ev,
        mu=mu_au,
        temperature=temperature,
        units=eVag,
    )

    assert not np.allclose(wrong_win_ev, win_ev)


def test_same_physical_ac_denominator_requires_omega_tau_dimensionless() -> None:
    """
    AC denominator should depend only on omega * tau.

    This catches accidental hbar or energy scaling inside the Drude denominator.
    """

    KH = cosine_k1_kernel()
    KS = identity_overlap_kernel()

    omega_tau = 3.0
    tau_a = 0.25
    tau_b = 2.0

    calc_a = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        tau=tau_a,
        omega=omega_tau / tau_a,
    ).run()

    calc_b = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        tau=tau_b,
        omega=omega_tau / tau_b,
    ).run()

    assert calc_a.ac_weights is not None
    assert calc_b.ac_weights is not None

    # Weight is proportional to tau / (1 - i omega tau).
    assert np.allclose(calc_b.ac_weights / calc_a.ac_weights, tau_b / tau_a)


def test_same_physical_sigma_au_vs_evag_has_expected_unit_conversion() -> None:
    """
    Conductivity integrand has units

        q^2 * tau * window * velocity^2 * k_measure

    For the same physical calculation:

        q^2 gives e^2
        window gives 1 / E
        velocity^2 gives (L / T)^2
        k_measure in 2D gives 1 / L^2

    so the conversion from AU-like output to eVag output is

        e^2 * tau_ev / tau_au * (1 / E) * (L / T)^2 * (1 / L^2)

    Here tau is passed in the same seconds in both calculations, so tau ratio is 1
    if AU is treated as a symbolic unit. Since AU has E=L=e=hbar=1, the factor is

        eVag.e^2 * (1 / eVag.E) * (eVag.E / eVag.hbar)^2

    The length factors cancel in 2D after using physical k-measure.
    """

    tau_seconds = 2.0e-14
    omega = 0.0

    au = same_physical_cosine_calc(
        units=AU,
        energy_scale=1.0,
        mu=0.0,
        tau=tau_seconds,
        omega=omega,
    )

    evag = same_physical_cosine_calc(
        units=eVag,
        energy_scale=eVag.E,
        mu=0.0,
        tau=tau_seconds,
        omega=omega,
    )

    assert au.sigma is not None
    assert evag.sigma is not None

    expected_factor = (
        eVag.e**2
        * (1.0 / eVag.E)
        * (eVag.E / eVag.hbar) ** 2
    )

    assert np.allclose(
        evag.sigma[0, 0],
        expected_factor * au.sigma[0, 0],
        rtol=1e-12,
        atol=1e-30,
    )


def test_wrong_irrep_to_physical_k_changes_velocity_detectably() -> None:
    """
    If eVag uses identity k-map instead of 1 / Angstrom map, physical velocity
    is wrong by the length factor.

    The 2D sigma_xx may remain unchanged for isotropic scaling, because the
    velocity-squared factor and the k-measure Jacobian cancel. Therefore this
    test checks velocity directly.
    """

    tau_seconds = 2.0e-14

    correct = same_physical_cosine_calc(
        units=eVag,
        energy_scale=eVag.E,
        mu=0.0,
        tau=tau_seconds,
        omega=0.0,
    )

    KH = scaled_kernel_energy(cosine_k1_kernel(), eVag.E)
    KS = identity_overlap_kernel()

    wrong = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        weights=[1.0],
        irrep_to_physical_k=np.eye(2),
        units=eVag,
        mu=0.0,
        temperature=300.0,
        omega=0.0,
        tau=tau_seconds,
    ).run()

    assert correct.velocities is not None
    assert wrong.velocities is not None

    assert not np.allclose(wrong.velocities, correct.velocities)

    expected_ratio = 1.0 / eVag.L

    assert np.allclose(
        wrong.velocities[0, 0] / correct.velocities[0, 0],
        expected_ratio,
        rtol=1e-12,
        atol=1e-12,
    )


def test_wrong_anisotropic_irrep_to_physical_k_changes_sigma_detectably() -> None:
    """
    Anisotropic k-map errors do change sigma components, because the Jacobian
    no longer cancels each velocity component in the same way.
    """

    KH = scaled_kernel_energy(cosine_k1_kernel(), eVag.E)
    KS = identity_overlap_kernel()

    tau_seconds = 2.0e-14

    correct = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        weights=[1.0],
        irrep_to_physical_k=np.array(
            [
                [1.0 / eVag.L, 0.0],
                [0.0, 2.0 / eVag.L],
            ]
        ),
        units=eVag,
        mu=0.0,
        temperature=300.0,
        omega=0.0,
        tau=tau_seconds,
    ).run()

    wrong = make_calc(
        KH,
        KS,
        [np.pi / 2],
        [0.0],
        weights=[1.0],
        irrep_to_physical_k=np.eye(2),
        units=eVag,
        mu=0.0,
        temperature=300.0,
        omega=0.0,
        tau=tau_seconds,
    ).run()

    assert correct.sigma is not None
    assert wrong.sigma is not None

    assert not np.allclose(
        wrong.sigma[0, 0],
        correct.sigma[0, 0],
        rtol=1e-12,
        atol=0.0,
    )

def test_energy_scaling_without_mu_scaling_changes_conductivity() -> None:
    """
    Same Hamiltonian converted to eV must use mu converted to eV.

    Choose mu at the test energy so the correct Fermi window is large, while
    the unconverted mu gives a very different result.
    """

    k1 = np.pi / 3
    tau_seconds = 2.0e-14

    # E_au = cos(pi / 3) = 0.5 Hartree
    mu_au = 0.5
    mu_ev = mu_au * eVag.E

    correct = same_physical_cosine_calc(
        units=eVag,
        energy_scale=eVag.E,
        mu=mu_ev,
        tau=tau_seconds,
        omega=0.0,
        k1=k1,
    )

    wrong = same_physical_cosine_calc(
        units=eVag,
        energy_scale=eVag.E,
        mu=mu_au,
        tau=tau_seconds,
        omega=0.0,
        k1=k1,
    )

    assert correct.ac_weights is not None
    assert wrong.ac_weights is not None
    assert correct.sigma is not None
    assert wrong.sigma is not None

    assert np.max(np.abs(correct.ac_weights)) > 0.0

    assert np.max(np.abs(wrong.ac_weights)) < (
        1e-100 * np.max(np.abs(correct.ac_weights))
    )

    assert np.max(np.abs(wrong.sigma)) < (
        1e-100 * np.max(np.abs(correct.sigma))
    )
