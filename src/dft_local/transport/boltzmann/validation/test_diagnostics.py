
from __future__ import annotations

import numpy as np

from dft_local.transport.boltzmann.validation.core import (
    antisymmetric_relative_norm,
    is_positive_semidefinite,
    tensor_invariant_report,
    validation_summary,
    weighted_outer_product_tensor,
)
from dft_local.transport.boltzmann.validation.diagnostics import diagnostics


def test_validation_summary_has_planned_operator_checks() -> None:
    summary = validation_summary()

    assert "operator approach" in summary.purpose
    assert "known-function end-to-end conductivity checks" in summary.planned_checks
    assert "basis-change covariance checks" in summary.planned_checks


def test_weighted_outer_product_tensor_is_symmetric_and_psd() -> None:
    velocity = np.array(
        [
            [[1.0, 2.0], [-1.0, 0.5]],
            [[0.0, -2.0], [3.0, 1.0]],
        ],
        dtype=np.float64,
    )
    weight = np.array(
        [
            [1.0, 0.25],
            [0.5, 2.0],
        ],
        dtype=np.float64,
    )

    tensor = weighted_outer_product_tensor(velocity, weight)

    np.testing.assert_allclose(tensor, tensor.T)
    assert is_positive_semidefinite(tensor)
    assert antisymmetric_relative_norm(tensor) == 0.0


def test_tensor_invariant_report_contains_expected_metrics() -> None:
    tensor = np.array(
        [
            [3.0, 0.25],
            [0.25, 2.0],
        ],
        dtype=np.float64,
    )

    report = tensor_invariant_report(tensor)

    assert report["trace"] == 5.0
    assert report["minimum_symmetric_eigenvalue"] > 0.0
    assert report["antisymmetric_relative_norm"] == 0.0


def test_validation_diagnostic_renders_scope_and_smoke_test() -> None:
    specs = {spec.id: spec for spec in diagnostics()}

    assert "transport.boltzmann.validation.overview" in specs

    result = specs["transport.boltzmann.validation.overview"].compute(None, {})

    assert result.title == "Boltzmann operator validation"

    section_ids = {section.id for section in result.sections}
    assert "boltzmann_validation_scope" in section_ids
    assert "boltzmann_validation_outer_product_smoke_test" in section_ids

    table_ids = {
        table.id
        for section in result.sections
        for table in section.tables
    }
    assert "boltzmann_validation_current_scope" in table_ids
    assert "boltzmann_validation_planned_checks" in table_ids
    assert "boltzmann_validation_outer_product_tensor" in table_ids
    assert "boltzmann_validation_outer_product_invariants" in table_ids


def test_validation_diagnostic_is_visible_to_default_loader() -> None:
    from dft_local.diagnostics.server import load_diagnostics

    specs = {spec.id: spec for spec in load_diagnostics()}

    assert "transport.boltzmann.validation.overview" in specs


def test_symbol_roundtrip_and_known_derivative_operator() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        apply_operator_from_symbol,
        central_difference_kernel,
        central_difference_symbol,
        finite_group_mode,
        reconstruct_kernel_from_symbol,
        symbol_from_kernel,
    )

    shape = (17, 19)
    dx = 0.25

    kernel = central_difference_kernel(shape, axis=0, spacing=dx)
    symbol = symbol_from_kernel(kernel)
    expected_symbol = central_difference_symbol(shape, axis=0, spacing=dx)
    recovered_kernel = reconstruct_kernel_from_symbol(symbol)

    np.testing.assert_allclose(symbol, expected_symbol, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(recovered_kernel, kernel, rtol=1.0e-12, atol=1.0e-12)

    mode_number = (3, 5)
    mode = finite_group_mode(shape, mode_number)
    derivative = apply_operator_from_symbol(symbol, mode)

    theta = 2.0 * np.pi * mode_number[0] / shape[0]
    expected = 1j * np.sin(theta) / dx * mode

    np.testing.assert_allclose(derivative, expected, rtol=1.0e-12, atol=1.0e-12)


def test_symbol_derivative_matches_known_energy_surface_derivative() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        analytic_central_difference_of_cosine_energy,
        apply_operator_from_kernel,
        central_difference_kernel,
        periodic_cosine_energy_surface,
    )

    shape = (17, 19)
    dx = 0.25
    dy = 0.40
    ax = 0.03
    ay = 0.02

    energy = periodic_cosine_energy_surface(
        shape,
        mu=-0.2,
        amplitude_x=ax,
        amplitude_y=ay,
    )

    dx_energy = apply_operator_from_kernel(
        central_difference_kernel(shape, axis=0, spacing=dx),
        energy,
    ).real
    dy_energy = apply_operator_from_kernel(
        central_difference_kernel(shape, axis=1, spacing=dy),
        energy,
    ).real

    expected_dx = analytic_central_difference_of_cosine_energy(
        shape,
        axis=0,
        spacing=dx,
        amplitude_x=ax,
        amplitude_y=ay,
    )
    expected_dy = analytic_central_difference_of_cosine_energy(
        shape,
        axis=1,
        spacing=dy,
        amplitude_x=ax,
        amplitude_y=ay,
    )

    np.testing.assert_allclose(dx_energy, expected_dx, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(dy_energy, expected_dy, rtol=1.0e-12, atol=1.0e-12)


def test_symbol_validation_probe_is_visible_in_diagnostic() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        operator_symbol_validation_probe,
    )

    probe = operator_symbol_validation_probe()

    assert probe["identity_mode_relative_error"] < 1.0e-12
    assert probe["kernel_symbol_roundtrip_error"] < 1.0e-12
    assert probe["dx_symbol_relative_error"] < 1.0e-12
    assert probe["dy_symbol_relative_error"] < 1.0e-12
    assert probe["dx_energy_surface_relative_error"] < 1.0e-12
    assert probe["dy_energy_surface_relative_error"] < 1.0e-12

    specs = {spec.id: spec for spec in diagnostics()}
    result = specs["transport.boltzmann.validation.overview"].compute(None, {})

    section_ids = {section.id for section in result.sections}
    assert "boltzmann_validation_symbol_checks" in section_ids

    symbol_section = next(
        section for section in result.sections
        if section.id == "boltzmann_validation_symbol_checks"
    )
    table_ids = {table.id for table in symbol_section.tables}

    assert "boltzmann_validation_symbol_errors" in table_ids


def test_production_gd_symbol_matches_known_analytic_symbol() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        expected_separable_cosine_derivative,
        expected_separable_cosine_symbol,
        gd_separable_cosine_kernel,
        generic_symbol_scalar_channels,
    )
    from dft_local.transport.boltzmann.calculation.core import (
        gd_symbol_derivative_fixed,
        gd_symbol_derivative_generic,
    )

    c0 = 1.25
    c1 = 0.70
    c2 = -0.30
    k1 = 0.37
    k2 = -0.44

    K = gd_separable_cosine_kernel(c0=c0, c1=c1, c2=c2)

    expected = expected_separable_cosine_symbol(k1, k2, c0=c0, c1=c1, c2=c2)
    expected_dk1 = expected_separable_cosine_derivative(k1, k2, axis=0, c1=c1, c2=c2)
    expected_dk2 = expected_separable_cosine_derivative(k1, k2, axis=1, c1=c1, c2=c2)

    fixed = K.symbol_fixed(k1, k2, sigma=1)
    generic = K.symbol_generic(k1, k2)

    np.testing.assert_allclose(fixed, [[expected]], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(generic_symbol_scalar_channels(generic), [expected, expected], rtol=1.0e-12, atol=1.0e-12)

    fixed_dk1 = gd_symbol_derivative_fixed(K, k1, k2, sigma=1, axis=0)
    fixed_dk2 = gd_symbol_derivative_fixed(K, k1, k2, sigma=1, axis=1)
    generic_dk1 = gd_symbol_derivative_generic(K, k1, k2, axis=0)
    generic_dk2 = gd_symbol_derivative_generic(K, k1, k2, axis=1)

    np.testing.assert_allclose(fixed_dk1, [[expected_dk1]], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(fixed_dk2, [[expected_dk2]], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(generic_symbol_scalar_channels(generic_dk1), [expected_dk1, expected_dk1], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(generic_symbol_scalar_channels(generic_dk2), [expected_dk2, expected_dk2], rtol=1.0e-12, atol=1.0e-12)


def test_production_gd_symbol_reproduces_known_energy_surface_and_derivative() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        expected_separable_cosine_derivative,
        expected_separable_cosine_symbol,
        gd_separable_cosine_kernel,
    )
    from dft_local.transport.boltzmann.calculation.core import gd_symbol_derivative_fixed

    c0 = 1.25
    c1 = 0.70
    c2 = -0.30
    K = gd_separable_cosine_kernel(c0=c0, c1=c1, c2=c2)

    for k1 in np.linspace(-np.pi, np.pi, 9, endpoint=False):
        for k2 in np.linspace(-np.pi, np.pi, 11, endpoint=False):
            expected = expected_separable_cosine_symbol(k1, k2, c0=c0, c1=c1, c2=c2)
            actual = float(K.symbol_fixed(k1, k2, sigma=1)[0, 0].real)

            expected_dk1 = expected_separable_cosine_derivative(k1, k2, axis=0, c1=c1, c2=c2)
            expected_dk2 = expected_separable_cosine_derivative(k1, k2, axis=1, c1=c1, c2=c2)

            actual_dk1 = float(gd_symbol_derivative_fixed(K, k1, k2, sigma=1, axis=0)[0, 0].real)
            actual_dk2 = float(gd_symbol_derivative_fixed(K, k1, k2, sigma=1, axis=1)[0, 0].real)

            np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
            np.testing.assert_allclose(actual_dk1, expected_dk1, rtol=1.0e-12, atol=1.0e-12)
            np.testing.assert_allclose(actual_dk2, expected_dk2, rtol=1.0e-12, atol=1.0e-12)


def test_production_gd_symbol_probe_is_visible_in_diagnostic() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        gd_symbol_production_validation_probe,
    )

    probe = gd_symbol_production_validation_probe()

    assert all(value < 1.0e-12 for value in probe.values())

    specs = {spec.id: spec for spec in diagnostics()}
    result = specs["transport.boltzmann.validation.overview"].compute(None, {})

    section_ids = {section.id for section in result.sections}
    assert "boltzmann_validation_production_symbol_checks" in section_ids

    section = next(
        section for section in result.sections
        if section.id == "boltzmann_validation_production_symbol_checks"
    )
    table_ids = {table.id for table in section.tables}

    assert "boltzmann_validation_production_symbol_errors" in table_ids
