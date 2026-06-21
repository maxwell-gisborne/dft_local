
from __future__ import annotations

import numpy as np

from dft_local.core.units import DisplayQuantity


def assert_display_quantity(value, expected: float | None = None) -> DisplayQuantity:
    assert isinstance(value, DisplayQuantity)
    if expected is not None:
        assert abs(value.value - expected) <= max(1e-12, abs(expected) * 1e-12)
    return value

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

    from dft_local.diagnostics.models import DiagnosticSection

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    section_ids = {section.id for section in sections}
    assert "boltzmann_validation_scope" in section_ids
    assert "boltzmann_validation_outer_product_smoke_test" in section_ids

    from dft_local.diagnostics.models import Table

    table_ids = {
        table.id
        for section in sections
        for table in section.tables
    } | {
        block.id
        for section in sections
        for block in section.body
        if isinstance(block, Table)
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

    from dft_local.diagnostics.models import DiagnosticSection

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    section_ids = {section.id for section in sections}
    assert "boltzmann_validation_symbol_checks" in section_ids

    symbol_section = next(
        section for section in sections
        if section.id == "boltzmann_validation_symbol_checks"
    )
    from dft_local.diagnostics.models import Table

    table_ids = {table.id for table in symbol_section.tables} | {
        block.id for block in symbol_section.body if isinstance(block, Table)
    }

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

    from dft_local.diagnostics.models import DiagnosticSection

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    section_ids = {section.id for section in sections}
    assert "boltzmann_validation_production_symbol_checks" in section_ids

    section = next(
        section for section in sections
        if section.id == "boltzmann_validation_production_symbol_checks"
    )
    from dft_local.diagnostics.models import Table

    table_ids = {table.id for table in section.tables} | {
        block.id for block in section.body if isinstance(block, Table)
    }

    assert "boltzmann_validation_production_symbol_errors" in table_ids

def test_finite_field_dc_validation_scaffold_sections_are_visible() -> None:
    specs = {spec.id: spec for spec in diagnostics()}

    assert "transport.boltzmann.validation.finite_field_dc" in specs

    result = specs["transport.boltzmann.validation.finite_field_dc"].compute(None, {})

    assert result.title == "Finite-field DC validation"

    from dft_local.diagnostics.models import DiagnosticSection, Table

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    section_ids = {section.id for section in sections}

    expected_sections = {
        "finite_field_dc_validation_overview",
        "finite_field_dc_validation_inputs",
        "finite_field_dc_validation_input_health",
        "finite_field_dc_validation_band_crossing_hazards",
        "finite_field_dc_validation_velocity_validation",
        "finite_field_dc_validation_vincent_reconstruction",
        "finite_field_dc_validation_strong_dc_validation",
        "finite_field_dc_validation_weak_dc_limit",
        "finite_field_dc_validation_mode_decomposition",
        "finite_field_dc_validation_analytic_toys",
        "finite_field_dc_validation_unit_scaling",
        "finite_field_dc_validation_k_convergence",
        "finite_field_dc_validation_symmetry",
    }

    assert expected_sections <= section_ids

    table_ids = {
        block.id
        for section in sections
        for block in section.body
        if isinstance(block, Table)
    }

    assert "finite_field_dc_validation_dashboard" in table_ids
    assert "finite_field_dc_validation_inputs_table" in table_ids
    assert "finite_field_dc_validation_vincent_reconstruction_table" in table_ids
    assert "finite_field_dc_validation_mode_decomposition_table" in table_ids

def test_finite_field_dc_validation_has_form_inputs() -> None:
    specs = {spec.id: spec for spec in diagnostics()}
    spec = specs["transport.boltzmann.validation.finite_field_dc"]

    input_names = {input_spec.name for input_spec in spec.inputs}

    assert {
        "dataset",
        "temperature",
        "mu",
        "tau",
        "units",
        "n_u",
        "n_v",
        "electric_field",
        "theta",
        "band_index",
        "symmetrization",
    } <= input_names

    parsed = {input_spec.name: input_spec.parse(None) for input_spec in spec.inputs}
    result = spec.compute(None, parsed)

    from dft_local.diagnostics.models import DiagnosticSection, Table

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    inputs_section = next(
        section for section in sections
        if section.id == "finite_field_dc_validation_inputs"
    )

    tables = {
        block.id: block
        for block in inputs_section.body
        if isinstance(block, Table)
    }

    assert "finite_field_dc_validation_inputs_table" in tables

    rows = {
        row.cells[0]: row.cells[1]
        for row in tables["finite_field_dc_validation_inputs_table"].rows
    }

    assert rows["dataset"] == "default"
    assert rows["N_u"] == 11
    assert rows["N_v"] == 11
    assert rows["symmetrization scheme"] == "star"

def test_finite_field_input_health_probe_checks_symbol_health() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        finite_field_input_health_probe,
    )

    probe = finite_field_input_health_probe(n_u=5, n_v=7, symmetrization="star")

    assert probe["sample_count"] == 35
    assert probe["h_star_defect_max"] < 1.0e-12
    assert probe["s_star_defect_max"] < 1.0e-12
    assert probe["h_hermitian_defect_rel_max"] < 1.0e-12
    assert probe["s_hermitian_defect_rel_max"] < 1.0e-12
    assert probe["s_eig_min"] > 1.0e-10
    assert probe["s_condition_number_abs_max"] >= 1.0
    assert probe["s_positive"] is True


def test_finite_field_dc_input_health_section_contains_real_metrics() -> None:
    specs = {spec.id: spec for spec in diagnostics()}
    spec = specs["transport.boltzmann.validation.finite_field_dc"]

    parsed = {input_spec.name: input_spec.parse(None) for input_spec in spec.inputs}
    parsed["n_u"] = 5
    parsed["n_v"] = 7

    result = spec.compute(None, parsed)

    from dft_local.diagnostics.models import DiagnosticSection, Table

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    input_health = next(
        section for section in sections
        if section.id == "finite_field_dc_validation_input_health"
    )

    tables = {
        block.id: block
        for block in input_health.body
        if isinstance(block, Table)
    }

    assert "finite_field_dc_validation_input_health_table" in tables

    rows = {
        row.cells[0]: row.cells[1]
        for row in tables["finite_field_dc_validation_input_health_table"].rows
    }

    assert rows["sample count"] == 35
    assert rows["S positive"] is True
    assert "dummy" not in set(rows.values())

def test_finite_field_band_crossing_hazard_probe_finds_toy_gap() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        finite_field_band_crossing_hazard_probe,
    )

    probe = finite_field_band_crossing_hazard_probe(
        n_u=10,
        n_v=10,
        gap_threshold=0.50,
        mass=0.20,
    )

    assert probe["sample_count"] == 100
    assert probe["min_gap"] >= 0.0
    assert probe["hazard_count"] >= 1
    assert probe["hazard_fraction"] > 0.0
    assert probe["has_hazard"] is True
    assert np.isfinite(probe["min_gap_k1"])
    assert np.isfinite(probe["min_gap_k2"])
    assert probe["max_gap_neighbour_jump"] >= 0.0


def test_finite_field_dc_band_crossing_section_contains_real_metrics() -> None:
    specs = {spec.id: spec for spec in diagnostics()}
    spec = specs["transport.boltzmann.validation.finite_field_dc"]

    parsed = {input_spec.name: input_spec.parse(None) for input_spec in spec.inputs}
    parsed["n_u"] = 10
    parsed["n_v"] = 10

    result = spec.compute(None, parsed)

    from dft_local.diagnostics.models import DiagnosticSection, Table

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    section = next(
        section for section in sections
        if section.id == "finite_field_dc_validation_band_crossing_hazards"
    )

    tables = {
        block.id: block
        for block in section.body
        if isinstance(block, Table)
    }

    assert "finite_field_dc_validation_band_crossing_hazards_table" in tables

    rows = {
        row.cells[0]: row.cells[1]
        for row in tables["finite_field_dc_validation_band_crossing_hazards_table"].rows
    }

    assert rows["sample count"] == 100
    assert rows["has hazard"] is True
    assert "dummy" not in set(rows.values())


def test_finite_field_velocity_validation_probe_checks_derivatives() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        finite_field_velocity_validation_probe,
    )

    probe = finite_field_velocity_validation_probe()

    assert probe["production_dk1_abs_error"] < 1.0e-12
    assert probe["production_dk2_abs_error"] < 1.0e-12
    assert probe["finite_difference_dk1_abs_error"] < 1.0e-9
    assert probe["finite_difference_dk2_abs_error"] < 1.0e-9
    assert probe["hellmann_feynman_dk1_abs_error"] < 1.0e-12
    assert probe["hellmann_feynman_dk2_abs_error"] < 1.0e-12
    assert probe["generic_fixed_symbol_abs_error"] < 1.0e-12
    assert probe["generic_fixed_dk1_abs_error"] < 1.0e-12
    assert probe["generic_fixed_dk2_abs_error"] < 1.0e-12


def test_finite_field_dc_velocity_section_contains_real_metrics() -> None:
    specs = {spec.id: spec for spec in diagnostics()}
    spec = specs["transport.boltzmann.validation.finite_field_dc"]

    parsed = {input_spec.name: input_spec.parse(None) for input_spec in spec.inputs}
    result = spec.compute(None, parsed)

    from dft_local.diagnostics.models import DiagnosticSection, Table

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    section = next(
        section for section in sections
        if section.id == "finite_field_dc_validation_velocity_validation"
    )

    tables = {
        block.id: block
        for block in section.body
        if isinstance(block, Table)
    }

    assert "finite_field_dc_validation_velocity_validation_table" in tables

    rows = {
        row.cells[0]: row.cells[1]
        for row in tables["finite_field_dc_validation_velocity_validation_table"].rows
    }

    assert_display_quantity(rows["production derivative dk1 error"], 0.0)
    assert_display_quantity(rows["Hellmann-Feynman dk1 error"], 0.0)
    assert "dummy" not in set(rows.values())


def test_finite_field_unit_scaling_probe_checks_core_factors() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        finite_field_unit_scaling_probe,
    )

    probe = finite_field_unit_scaling_probe()

    assert probe["atomic_energy_to_ev"] == 27.21138386
    assert probe["atomic_length_to_angstrom"] == 0.52917721092
    assert probe["velocity_factor_abs_error"] < 1.0e-12
    assert probe["fermi_window_ev_from_au_factor"] == 1.0 / 27.21138386
    assert probe["mu_conversion_required"] is True


def test_finite_field_dc_unit_scaling_section_contains_real_metrics() -> None:
    specs = {spec.id: spec for spec in diagnostics()}
    spec = specs["transport.boltzmann.validation.finite_field_dc"]

    parsed = {input_spec.name: input_spec.parse(None) for input_spec in spec.inputs}
    result = spec.compute(None, parsed)

    from dft_local.diagnostics.models import DiagnosticSection, Table

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    section = next(
        section for section in sections
        if section.id == "finite_field_dc_validation_unit_scaling"
    )

    tables = {
        block.id: block
        for block in section.body
        if isinstance(block, Table)
    }

    assert "finite_field_dc_validation_unit_scaling_table" in tables

    rows = {
        row.cells[0]: row.cells[1]
        for row in tables["finite_field_dc_validation_unit_scaling_table"].rows
    }

    assert_display_quantity(rows["atomic energy to eV"], 27.21138386)
    assert_display_quantity(rows["atomic length to Å"], 0.52917721092)
    assert rows["mu conversion required"] is True
    assert "dummy" not in set(rows.values())


def test_finite_field_analytic_toy_coverage_probe_summarises_real_toys() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        finite_field_analytic_toy_coverage_probe,
    )

    probe = finite_field_analytic_toy_coverage_probe()

    assert probe["toy_count"] == 4
    assert probe["separable_cosine_symbol_max_error"] < 1.0e-12
    assert probe["separable_cosine_derivative_max_error"] < 1.0e-9
    assert probe["identity_overlap_min_eig"] > 1.0e-10
    assert probe["periodic_dirac_hazard_count"] >= 1
    assert probe["velocity_hf_max_error"] < 1.0e-12
    assert probe["unit_velocity_factor_error"] < 1.0e-12
    assert probe["all_current_toys_pass"] is True


def test_finite_field_dc_analytic_toys_section_contains_real_metrics() -> None:
    specs = {spec.id: spec for spec in diagnostics()}
    spec = specs["transport.boltzmann.validation.finite_field_dc"]

    parsed = {input_spec.name: input_spec.parse(None) for input_spec in spec.inputs}
    result = spec.compute(None, parsed)

    from dft_local.diagnostics.models import DiagnosticSection, Table

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    section = next(
        section for section in sections
        if section.id == "finite_field_dc_validation_analytic_toys"
    )

    tables = {
        block.id: block
        for block in section.body
        if isinstance(block, Table)
    }

    assert "finite_field_dc_validation_analytic_toys_table" in tables

    rows = {
        row.cells[0]: row.cells[1]
        for row in tables["finite_field_dc_validation_analytic_toys_table"].rows
    }

    assert rows["toy count"] == 4
    assert rows["all current toys pass"] is True
    assert "dummy" not in set(rows.values())


def test_finite_field_k_convergence_probe_checks_periodic_measure() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        finite_field_k_convergence_probe,
    )

    probe = finite_field_k_convergence_probe()

    assert probe["grid_count"] == 5
    assert probe["coarsest_n"] == 5
    assert probe["finest_n"] == 23
    assert abs(probe["reference_average_grad_e_sq"] - 0.29) < 1.0e-15
    assert probe["finest_abs_error"] < 1.0e-12
    assert probe["max_abs_error"] < 1.0e-12
    assert probe["all_grid_errors_small"] is True


def test_finite_field_dc_k_convergence_section_contains_real_metrics() -> None:
    specs = {spec.id: spec for spec in diagnostics()}
    spec = specs["transport.boltzmann.validation.finite_field_dc"]

    parsed = {input_spec.name: input_spec.parse(None) for input_spec in spec.inputs}
    result = spec.compute(None, parsed)

    from dft_local.diagnostics.models import DiagnosticSection, Table

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    section = next(
        section for section in sections
        if section.id == "finite_field_dc_validation_k_convergence"
    )

    tables = {
        block.id: block
        for block in section.body
        if isinstance(block, Table)
    }

    assert "finite_field_dc_validation_k_convergence_table" in tables

    rows = {
        row.cells[0]: row.cells[1]
        for row in tables["finite_field_dc_validation_k_convergence_table"].rows
    }

    assert rows["grid count"] == 5
    assert rows["all grid errors small"] is True
    assert "dummy" not in set(rows.values())


def test_finite_field_symmetry_sanity_probe_checks_toy_symmetries() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        finite_field_symmetry_sanity_probe,
    )

    probe = finite_field_symmetry_sanity_probe()

    assert probe["sample_count"] == 289
    assert probe["energy_inversion_max_error"] < 1.0e-12
    assert probe["dk1_odd_max_error"] < 1.0e-12
    assert probe["dk2_odd_max_error"] < 1.0e-12
    assert probe["tensor_xx"] > 0.0
    assert probe["tensor_yy"] > 0.0
    assert abs(probe["tensor_xy"]) < 1.0e-12
    assert abs(probe["tensor_yx"]) < 1.0e-12
    assert probe["tensor_antisym_abs"] < 1.0e-12
    assert probe["all_symmetry_checks_pass"] is True


def test_finite_field_dc_symmetry_section_contains_real_metrics() -> None:
    specs = {spec.id: spec for spec in diagnostics()}
    spec = specs["transport.boltzmann.validation.finite_field_dc"]

    parsed = {input_spec.name: input_spec.parse(None) for input_spec in spec.inputs}
    result = spec.compute(None, parsed)

    from dft_local.diagnostics.models import DiagnosticSection, Table

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    section = next(
        section for section in sections
        if section.id == "finite_field_dc_validation_symmetry"
    )

    tables = {
        block.id: block
        for block in section.body
        if isinstance(block, Table)
    }

    assert "finite_field_dc_validation_symmetry_table" in tables

    rows = {
        row.cells[0]: row.cells[1]
        for row in tables["finite_field_dc_validation_symmetry_table"].rows
    }

    assert rows["sample count"] == 289
    assert rows["all symmetry checks pass"] is True
    assert "dummy" not in set(rows.values())


def test_finite_field_vincent_reconstruction_probe_reuses_ashcroft_domain() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        finite_field_vincent_reconstruction_probe,
    )

    probe = finite_field_vincent_reconstruction_probe()

    assert probe.target_trace > 0.0
    assert probe.weak_chain_trace > 0.0
    assert abs(probe.weak_chain_trace_percent_error) < 10.0
    assert probe.find_simplex_max_velocity_error > 1.0
    assert probe.best_adjacent_max_velocity_error < 1.0e-3
    assert probe.velocity_error_reduction > 1.0e8
    assert probe.best_adjacent_matches_vincent is True


def test_finite_field_dc_vincent_section_contains_real_metrics() -> None:
    specs = {spec.id: spec for spec in diagnostics()}
    spec = specs["transport.boltzmann.validation.finite_field_dc"]

    parsed = {input_spec.name: input_spec.parse(None) for input_spec in spec.inputs}
    result = spec.compute(None, parsed)

    from dft_local.diagnostics.models import DiagnosticSection, Table

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    section = next(
        section for section in sections
        if section.id == "finite_field_dc_validation_vincent_reconstruction"
    )

    tables = {
        block.id: block
        for block in section.body
        if isinstance(block, Table)
    }

    assert "finite_field_dc_validation_vincent_reconstruction_table" in tables

    rows = {
        row.cells[0]: row.cells[1]
        for row in tables["finite_field_dc_validation_vincent_reconstruction_table"].rows
    }

    assert rows["best adjacent matches Vincent"] is True
    assert "dummy" not in set(rows.values())


def test_finite_field_strong_dc_validation_probe_checks_mode_closure() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        finite_field_strong_dc_validation_probe,
    )

    probe = finite_field_strong_dc_validation_probe()

    assert probe.mode_count > 0
    assert probe.nonzero_mode_count > 0
    assert probe.strong_grid_trace > 0.0
    assert probe.weak_chain_grid_trace > 0.0
    assert abs(probe.strong_vs_weak_rel_trace_gap) < 0.2
    assert probe.mode_reconstruction_abs_error < 1.0e-18
    assert probe.imaginary_leakage_ratio < 1.0e-12
    assert probe.response_factor_finite is True
    assert probe.velocity_coefficients_finite is True
    assert probe.strong_dc_internal_pass is True


def test_finite_field_dc_strong_dc_section_contains_real_metrics() -> None:
    specs = {spec.id: spec for spec in diagnostics()}
    spec = specs["transport.boltzmann.validation.finite_field_dc"]

    parsed = {input_spec.name: input_spec.parse(None) for input_spec in spec.inputs}
    result = spec.compute(None, parsed)

    from dft_local.diagnostics.models import DiagnosticSection, Table

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    section = next(
        section for section in sections
        if section.id == "finite_field_dc_validation_strong_dc_validation"
    )

    tables = {
        block.id: block
        for block in section.body
        if isinstance(block, Table)
    }

    assert "finite_field_dc_validation_strong_dc_validation_table" in tables

    rows = {
        row.cells[0]: row.cells[1]
        for row in tables["finite_field_dc_validation_strong_dc_validation_table"].rows
    }

    assert rows["strong DC internal pass"] is True
    assert "dummy" not in set(rows.values())


def test_finite_field_weak_dc_limit_probe_checks_zero_field_limit() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        finite_field_weak_dc_limit_probe,
    )

    probe = finite_field_weak_dc_limit_probe()

    assert probe.field_row_count >= 3
    assert probe.zero_eta == 0.0
    assert probe.zero_relative_tensor_discrepancy < 1.0e-12
    assert abs(probe.zero_relative_trace_discrepancy) < 1.0e-12
    assert probe.relative_weak_limit_error < 1.0e-12
    assert probe.small_eta > 0.0
    assert probe.largest_eta > probe.small_eta
    assert probe.largest_relative_tensor_discrepancy >= probe.small_relative_tensor_discrepancy
    assert probe.weak_limit_pass is True


def test_finite_field_dc_weak_dc_limit_section_contains_real_metrics() -> None:
    specs = {spec.id: spec for spec in diagnostics()}
    spec = specs["transport.boltzmann.validation.finite_field_dc"]

    parsed = {input_spec.name: input_spec.parse(None) for input_spec in spec.inputs}
    result = spec.compute(None, parsed)

    from dft_local.diagnostics.models import DiagnosticSection, Table

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    section = next(
        section for section in sections
        if section.id == "finite_field_dc_validation_weak_dc_limit"
    )

    tables = {
        block.id: block
        for block in section.body
        if isinstance(block, Table)
    }

    assert "finite_field_dc_validation_weak_dc_limit_table" in tables

    rows = {
        row.cells[0]: row.cells[1]
        for row in tables["finite_field_dc_validation_weak_dc_limit_table"].rows
    }

    assert rows["weak-limit pass"] is True
    assert "dummy" not in set(rows.values())


def test_finite_field_mode_decomposition_probe_checks_closure() -> None:
    from dft_local.transport.boltzmann.validation.core import (
        finite_field_mode_decomposition_probe,
    )

    probe = finite_field_mode_decomposition_probe()

    assert probe.mode_count > 0
    assert probe.gamma_reconstruction_abs_error < 1.0e-8
    assert probe.rho_reconstruction_abs_error < 1.0e-12
    assert probe.mode_tensor_reconstruction_abs_error < 1.0e-18
    assert probe.conductivity_trace > 0.0
    assert 0.0 < probe.top_1_mode_fraction <= 1.0
    assert probe.top_1_mode_fraction <= probe.top_10_mode_fraction <= probe.top_100_mode_fraction <= 1.0
    assert probe.gamma_finite is True
    assert probe.rho_finite is True
    assert probe.response_finite is True
    assert probe.mode_tensor_finite is True
    assert probe.mode_closure_pass is True


def test_finite_field_dc_mode_decomposition_section_contains_real_metrics() -> None:
    specs = {spec.id: spec for spec in diagnostics()}
    spec = specs["transport.boltzmann.validation.finite_field_dc"]

    parsed = {input_spec.name: input_spec.parse(None) for input_spec in spec.inputs}
    result = spec.compute(None, parsed)

    from dft_local.diagnostics.models import DiagnosticSection, Table

    sections = tuple(result.sections) + tuple(
        block for block in result.body if isinstance(block, DiagnosticSection)
    )
    section = next(
        section for section in sections
        if section.id == "finite_field_dc_validation_mode_decomposition"
    )

    tables = {
        block.id: block
        for block in section.body
        if isinstance(block, Table)
    }

    assert "finite_field_dc_validation_mode_decomposition_table" in tables

    rows = {
        row.cells[0]: row.cells[1]
        for row in tables["finite_field_dc_validation_mode_decomposition_table"].rows
    }

    assert rows["mode closure pass"] is True
    assert "dummy" not in set(rows.values())
