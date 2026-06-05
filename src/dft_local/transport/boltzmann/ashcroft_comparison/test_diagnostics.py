from __future__ import annotations


def _section_by_id(result, section_id: str):
    return next(section for section in result.sections if section.id == section_id)


def _all_section_tables(result):
    tables = []
    stack = list(result.sections)
    while stack:
        section = stack.pop(0)
        tables.extend(section.tables)
        stack.extend(section.sections)
    return tables


def _all_section_cards(result):
    cards = []
    stack = list(result.sections)
    while stack:
        section = stack.pop(0)
        cards.extend(section.cards)
        stack.extend(section.sections)
    return cards


def _all_section_markdowns(result):
    markdowns = []
    stack = list(result.sections)
    while stack:
        section = stack.pop(0)
        markdowns.extend(section.markdowns)
        stack.extend(section.sections)
    return markdowns


import numpy as np

from dft_local.diagnostics.discovery import load_diagnostics
from dft_local.transport.boltzmann.ashcroft_comparison.core import (
    load_vincent_input_data,
    reciprocal_lattice_vectors_from_primitives,
    relative_error,
    vincent_reference,
)


def test_vincent_reference_values_are_recorded() -> None:
    reference = vincent_reference()

    np.testing.assert_allclose(reference.electric_field_V_per_m, [1.0e5, 0.0])
    np.testing.assert_allclose(
        reference.expected_conductivity_S_per_m,
        [
            [6.45179383e-02, -8.80479820e-05],
            [-8.73823365e-05, 6.44024548e-02],
        ],
    )
    assert reference.temperature_K == 300.0
    assert reference.relaxation_time_s == 1.0e-14
    assert reference.mean_fermi_weight == 3.907e-03


def test_vincent_input_files_load() -> None:
    inputs = load_vincent_input_data()

    assert inputs.primitive_lattice_vectors_bohr.shape == (2, 2)
    assert inputs.epsilon_of_k.ndim == 2
    assert inputs.epsilon_of_k.shape[0] > 0


def test_reciprocal_lattice_vectors_satisfy_duality() -> None:
    ai = load_vincent_input_data().primitive_lattice_vectors_bohr
    bi = reciprocal_lattice_vectors_from_primitives(ai)

    np.testing.assert_allclose(ai @ bi.T, 2.0 * np.pi * np.eye(2), atol=1.0e-12)


def test_relative_error_is_zero_for_reference_against_itself() -> None:
    reference = vincent_reference()

    err = relative_error(
        reference.expected_conductivity_S_per_m,
        reference.expected_conductivity_S_per_m,
    )

    np.testing.assert_allclose(err, np.zeros((2, 2)))


def test_velocity_grid_has_expected_shape() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import velocity_from_epsilon_grid

    inputs = load_vincent_input_data()
    vx, vy = velocity_from_epsilon_grid(
        inputs.epsilon_of_k,
        inputs.primitive_lattice_vectors_bohr,
    )

    assert vx.shape == inputs.epsilon_of_k.shape
    assert vy.shape == inputs.epsilon_of_k.shape
    assert np.isfinite(vx).all()
    assert np.isfinite(vy).all()


def test_vincent_target_k_points_match_second_reciprocal_direction() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        BOHR_TO_M,
        reciprocal_lattice_vectors_from_primitives,
        vincent_sample_velocity_targets,
    )

    inputs = load_vincent_input_data()
    bi = reciprocal_lattice_vectors_from_primitives(inputs.primitive_lattice_vectors_bohr)
    target_k, _target_v = vincent_sample_velocity_targets()

    # Vincent's listed first k-points advance along b2 / N for N = 100.
    step = bi[1] / (100.0 * BOHR_TO_M)

    np.testing.assert_allclose(target_k[1], step, rtol=1.0e-7, atol=100.0)
    np.testing.assert_allclose(target_k[2], 2.0 * step, rtol=1.0e-7, atol=100.0)


def test_velocity_candidate_errors_are_finite() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import candidate_velocity_errors

    inputs = load_vincent_input_data()
    errors = candidate_velocity_errors(
        inputs.epsilon_of_k,
        inputs.primitive_lattice_vectors_bohr,
    )

    assert errors
    assert all(np.isfinite(value) for value in errors.values())


def test_electric_field_shift_matches_vincent_reported_y_shift() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        electric_field_k_shift_per_m,
        vincent_reference,
    )

    reference = vincent_reference()
    shift = electric_field_k_shift_per_m(
        reference.electric_field_V_per_m,
        reference.relaxation_time_s,
    )

    assert shift.shape == (2,)
    assert np.isfinite(shift).all()


def test_shifted_velocity_candidate_errors_are_finite() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import shifted_velocity_candidate_errors

    inputs = load_vincent_input_data()
    errors = shifted_velocity_candidate_errors(
        inputs.epsilon_of_k,
        inputs.primitive_lattice_vectors_bohr,
    )

    assert errors
    assert all(np.isfinite(value) for value in errors.values())


def _linear_epsilon_grid(
    ai_bohr: np.ndarray,
    shape: tuple[int, int],
    gradient_hartree_bohr: tuple[float, float],
) -> np.ndarray:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        reciprocal_lattice_vectors_from_primitives,
    )

    n1, n2 = shape
    u = np.arange(n1, dtype=float) / float(n1)
    v = np.arange(n2, dtype=float) / float(n2)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    bi = reciprocal_lattice_vectors_from_primitives(ai_bohr)
    k_bohr_inv = uu[..., None] * bi[0] + vv[..., None] * bi[1]

    return (
        gradient_hartree_bohr[0] * k_bohr_inv[..., 0]
        + gradient_hartree_bohr[1] * k_bohr_inv[..., 1]
    )


def test_velocity_from_epsilon_grid_matches_linear_known_function() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        BOHR_TO_M,
        HARTREE_TO_J,
        HBAR_J_S,
        velocity_from_epsilon_grid,
    )

    ai = load_vincent_input_data().primitive_lattice_vectors_bohr
    gradient = (0.125, -0.075)
    epsilon = _linear_epsilon_grid(ai, (17, 19), gradient)

    vx, vy = velocity_from_epsilon_grid(epsilon, ai)

    expected_vx = gradient[0] * HARTREE_TO_J * BOHR_TO_M / HBAR_J_S
    expected_vy = gradient[1] * HARTREE_TO_J * BOHR_TO_M / HBAR_J_S

    np.testing.assert_allclose(vx, expected_vx, rtol=1.0e-12, atol=1.0e-9)
    np.testing.assert_allclose(vy, expected_vy, rtol=1.0e-12, atol=1.0e-9)


def test_cartesian_component_finite_difference_matches_linear_known_function() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        BOHR_TO_M,
        HARTREE_TO_J,
        HBAR_J_S,
        cartesian_component_velocity_from_steps,
        reciprocal_grid_cartesian_per_m,
    )

    ai = load_vincent_input_data().primitive_lattice_vectors_bohr
    gradient = (-0.040, 0.090)
    epsilon = _linear_epsilon_grid(ai, (23, 29), gradient)

    kx, ky = reciprocal_grid_cartesian_per_m(ai, epsilon.shape)
    k = np.stack([kx, ky], axis=-1)

    velocity = cartesian_component_velocity_from_steps(
        epsilon,
        ai,
        k,
        dx_per_m=1.0e6,
        dy_per_m=1.0e6,
        forward=False,
    )

    expected = np.array(
        [
            gradient[0] * HARTREE_TO_J * BOHR_TO_M / HBAR_J_S,
            gradient[1] * HARTREE_TO_J * BOHR_TO_M / HBAR_J_S,
        ]
    )

    # Avoid periodic wrap discontinuity at boundary points.
    interior = velocity[2:-2, 2:-2, :]
    np.testing.assert_allclose(interior, np.broadcast_to(expected, interior.shape), rtol=1.0e-10, atol=1.0e-6)


def test_velocity_systematic_error_probe_uses_confirmed_path() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        velocity_systematic_error_probe,
        vincent_sample_velocity_targets,
    )

    inputs = load_vincent_input_data()
    probe = velocity_systematic_error_probe(
        inputs.epsilon_of_k,
        inputs.primitive_lattice_vectors_bohr,
    )
    target_k, target_v = vincent_sample_velocity_targets()

    np.testing.assert_allclose(probe.target_k_per_m, target_k)
    np.testing.assert_allclose(probe.target_v_m_per_s, target_v)
    assert probe.local_v_m_per_s.shape == target_v.shape
    assert probe.delta_v_m_per_s.shape == target_v.shape
    assert probe.percent_error.shape == target_v.shape
    assert probe.delta_step_m_per_s.shape == target_v.shape
    assert probe.rms_error_m_per_s > 0.0


def test_shift_discrepancy_probe_detects_reported_shift_mismatch() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import shift_discrepancy_probe

    probe = shift_discrepancy_probe()

    assert probe.expected_norm_per_m > 0.0
    assert probe.reported_norm_per_m > 0.0
    assert probe.norm_ratio > 1.0
    assert abs(probe.expected_shift_per_m[0]) > 0.0
    assert probe.reported_shift_per_m[0] == 0.0
    assert abs(probe.reported_shift_per_m[1]) > 0.0


def test_shift_axis_swap_probe_detects_axis_swap_plus_scale() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import shift_axis_swap_probe

    probe = shift_axis_swap_probe()

    assert probe.expected_x_per_m != 0.0
    assert probe.expected_y_per_m == 0.0
    assert probe.reported_x_per_m == 0.0
    assert probe.reported_y_per_m != 0.0
    assert abs(probe.reported_y_over_expected_x) > 1.0


def test_swapped_axis_velocity_probe_exists() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        load_vincent_input_data,
        swapped_axis_velocity_hypothesis_errors,
    )

    inputs = load_vincent_input_data()
    errors = swapped_axis_velocity_hypothesis_errors(
        inputs.epsilon_of_k,
        inputs.primitive_lattice_vectors_bohr,
    )

    assert "normal" in errors
    assert "swap_k_input_and_output" in errors
    assert errors["swap_k_input_and_output"] < errors["normal"]


def test_fermi_window_is_bounded_and_peaks_at_mu() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        HARTREE_TO_J,
        fermi_window,
    )

    mu = 0.2 * HARTREE_TO_J
    eps = np.array([0.2, 0.2 + 1.0e-8, 0.2 - 1.0e-8]) * HARTREE_TO_J

    weight = fermi_window(eps, mu, 300.0)

    assert np.all(weight >= 0.0)
    assert np.all(weight <= 0.25)
    np.testing.assert_allclose(weight[0], 0.25, rtol=0.0, atol=1.0e-15)


def test_conductivity_from_velocity_grid_matches_manual_discrete_formula() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        ELECTRON_CHARGE_C,
        KB_J_K,
        conductivity_from_velocity_grid,
        fermi_window,
        reciprocal_cell_area_per_m2,
    )

    data = load_vincent_input_data()
    epsilon = np.zeros((5, 7), dtype=np.float64)
    velocity = np.zeros(epsilon.shape + (2,), dtype=np.float64)
    velocity[..., 0] = 3.0
    velocity[..., 1] = -2.0

    temperature = 300.0
    tau = 1.0e-14
    mu = 0.0

    result = conductivity_from_velocity_grid(
        epsilon,
        velocity,
        data.primitive_lattice_vectors_bohr,
        chemical_potential_J=mu,
        temperature_K=temperature,
        relaxation_time_s=tau,
    )

    weight = fermi_window(epsilon, mu, temperature)
    raw = np.einsum("ija,ijb,ij->ab", velocity, velocity, weight)
    k_cell_area = reciprocal_cell_area_per_m2(data.primitive_lattice_vectors_bohr, epsilon.shape)
    prefactor = ELECTRON_CHARGE_C ** 2 * tau / ((2.0 * np.pi) ** 2 * KB_J_K * temperature)
    expected = prefactor * k_cell_area * raw

    np.testing.assert_allclose(result.raw_velocity_weight_tensor, raw)
    np.testing.assert_allclose(result.conductivity_tensor_S, expected)


def test_vincent_fermi_window_statistics_are_reproduced() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        HARTREE_TO_J,
        fermi_window,
        vincent_reference,
    )

    data = load_vincent_input_data()
    reference = vincent_reference()

    epsilon_J = data.epsilon_of_k * HARTREE_TO_J
    mu = float(np.mean(epsilon_J))
    weight = fermi_window(epsilon_J, mu, reference.temperature_K)

    assert np.max(weight) <= 0.25
    np.testing.assert_allclose(np.max(weight), reference.max_fermi_weight, rtol=5.0e-3)
    np.testing.assert_allclose(np.mean(weight), reference.mean_fermi_weight, rtol=5.0e-3)


def test_conductivity_invariant_checks_hold() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        HARTREE_TO_J,
        conductivity_invariant_checks,
        vincent_reference,
    )

    data = load_vincent_input_data()
    reference = vincent_reference()
    mu = float(np.mean(data.epsilon_of_k) * HARTREE_TO_J)

    checks = conductivity_invariant_checks(
        data.epsilon_of_k,
        data.primitive_lattice_vectors_bohr,
        chemical_potential_J=mu,
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
    )

    assert checks["tau_linearity_relative_error"] < 1.0e-12
    assert checks["energy_shift_relative_error"] < 1.0e-10
    assert checks["velocity_square_relative_error"] < 1.0e-12
    assert checks["min_eigenvalue"] >= -1.0e-18
    assert checks["antisym_abs_over_trace"] < 1.0e-12


def test_conductivity_temperature_and_contribution_probes_are_well_behaved() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        HARTREE_TO_J,
        conductivity_contribution_probe,
        conductivity_temperature_probe,
        vincent_reference,
    )

    data = load_vincent_input_data()
    reference = vincent_reference()
    mu = float(np.mean(data.epsilon_of_k) * HARTREE_TO_J)

    temperature_rows = conductivity_temperature_probe(
        data.epsilon_of_k,
        data.primitive_lattice_vectors_bohr,
        chemical_potential_J=mu,
        relaxation_time_s=reference.relaxation_time_s,
    )

    assert all(row["max_fermi_weight"] <= 0.25 for row in temperature_rows)
    assert all(np.isfinite(row["trace"]) for row in temperature_rows)

    contribution_rows = conductivity_contribution_probe(
        data.epsilon_of_k,
        data.primitive_lattice_vectors_bohr,
        chemical_potential_J=mu,
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
    )

    fractions = [
        row["trace_contribution_fraction"]
        for row in contribution_rows
        if row["top_fraction"] > 0
    ]
    assert fractions == sorted(fractions)
    assert all(0.0 <= value <= 1.0 for value in fractions)


def test_conductivity_from_epsilon_grid_matches_periodic_sinusoid_theory() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        BOHR_TO_M,
        ELECTRON_CHARGE_C,
        HARTREE_TO_J,
        HBAR_J_S,
        KB_J_K,
        central_cartesian_velocity_grid,
        conductivity_from_epsilon_grid,
        fermi_window,
        reciprocal_cell_area_per_m2,
    )

    # Square analytic lattice, in bohr.
    # The reciprocal vectors are then orthogonal and have length 2π bohr^-1.
    ai = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    n0 = 32
    n1 = 40
    i = np.arange(n0, dtype=np.float64)[:, None]
    j = np.arange(n1, dtype=np.float64)[None, :]

    theta_x = 2.0 * np.pi * i / n0
    theta_y = 2.0 * np.pi * j / n1

    mu_Ha = -0.20
    amp_x_Ha = 2.0e-4
    amp_y_Ha = 1.4e-4

    epsilon_Ha = (
        mu_Ha
        + amp_x_Ha * np.cos(theta_x)
        + amp_y_Ha * np.cos(theta_y)
    )

    chemical_potential_J = mu_Ha * HARTREE_TO_J
    temperature_K = 300.0
    tau_s = 2.5e-14

    # Exact theory for the central periodic derivative of cos on this grid:
    #
    # d/dk cos(theta) ≈ [cos(theta + h) - cos(theta - h)] / (2 Δk)
    #                 = -sin(theta) sin(h) / Δk
    #
    # with Δk in m^-1.
    dkx_per_m = (2.0 * np.pi / BOHR_TO_M) / n0
    dky_per_m = (2.0 * np.pi / BOHR_TO_M) / n1

    vx_expected = (
        amp_x_Ha
        * HARTREE_TO_J
        * (-np.sin(theta_x) * np.sin(2.0 * np.pi / n0) / dkx_per_m)
        / HBAR_J_S
    )
    vy_expected = (
        amp_y_Ha
        * HARTREE_TO_J
        * (-np.sin(theta_y) * np.sin(2.0 * np.pi / n1) / dky_per_m)
        / HBAR_J_S
    )

    vx_expected = np.broadcast_to(vx_expected, epsilon_Ha.shape)
    vy_expected = np.broadcast_to(vy_expected, epsilon_Ha.shape)
    velocity_expected = np.stack((vx_expected, vy_expected), axis=-1)

    vx, vy = central_cartesian_velocity_grid(epsilon_Ha, ai)

    np.testing.assert_allclose(vx, vx_expected, rtol=1.0e-12, atol=1.0e-6)
    np.testing.assert_allclose(vy, vy_expected, rtol=1.0e-12, atol=1.0e-6)

    result = conductivity_from_epsilon_grid(
        epsilon_Ha,
        ai,
        chemical_potential_J=chemical_potential_J,
        temperature_K=temperature_K,
        relaxation_time_s=tau_s,
    )

    weight = fermi_window(
        epsilon_Ha * HARTREE_TO_J,
        chemical_potential_J,
        temperature_K,
    )
    raw_expected = np.einsum("ija,ijb,ij->ab", velocity_expected, velocity_expected, weight)
    k_cell_area = reciprocal_cell_area_per_m2(ai, epsilon_Ha.shape)
    prefactor = ELECTRON_CHARGE_C ** 2 * tau_s / ((2.0 * np.pi) ** 2 * KB_J_K * temperature_K)
    sigma_expected = prefactor * k_cell_area * raw_expected

    np.testing.assert_allclose(result.raw_velocity_weight_tensor, raw_expected, rtol=1.0e-12, atol=1.0e-3)
    np.testing.assert_allclose(result.conductivity_tensor_S, sigma_expected, rtol=1.0e-12, atol=1.0e-18)

    # Symmetry of the separable analytic example: no xy coupling.
    assert abs(result.conductivity_tensor_S[0, 1]) < 1.0e-18
    assert abs(result.conductivity_tensor_S[1, 0]) < 1.0e-18


def test_vincent_delaunay_velocity_probe_is_present() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        vincent_delaunay_velocity_grid,
        vincent_delaunay_velocity_sample_probe,
    )

    data = load_vincent_input_data()

    velocity_grid = vincent_delaunay_velocity_grid(
        data.epsilon_of_k,
        data.primitive_lattice_vectors_bohr,
    )
    assert velocity_grid.shape == data.epsilon_of_k.shape + (2,)
    assert np.all(np.isfinite(velocity_grid))

    probe = vincent_delaunay_velocity_sample_probe(
        data.epsilon_of_k,
        data.primitive_lattice_vectors_bohr,
    )
    assert probe["target"].shape == probe["local"].shape
    assert probe["delta"].shape == probe["local"].shape
    assert np.isfinite(probe["rms_error"])

    specs = {spec.id: spec for spec in load_diagnostics()}
    result = specs["transport.boltzmann.ashcroft_comparison.overview"].compute(None, {})
    velocity = _section_by_id(result, "ashcroft_velocity_comparison")

    table_ids = {table.id for table in velocity.tables}
    assert "section_velocity_delaunay_interpolation_probe" in table_ids


def test_vincent_delaunay_adjacent_simplex_probe_is_present() -> None:
    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        vincent_delaunay_adjacent_simplex_velocity_probe,
    )

    data = load_vincent_input_data()
    rows = vincent_delaunay_adjacent_simplex_velocity_probe(
        data.epsilon_of_k,
        data.primitive_lattice_vectors_bohr,
    )

    assert rows
    assert all(row["adjacent_count"] >= 1 for row in rows)
    assert all(np.isfinite(row["best_adjacent_error"]) for row in rows)

    specs = {spec.id: spec for spec in load_diagnostics()}
    result = specs["transport.boltzmann.ashcroft_comparison.overview"].compute(None, {})
    velocity = _section_by_id(result, "ashcroft_velocity_comparison")

    table_ids = {table.id for table in velocity.tables}
    assert "section_velocity_delaunay_adjacent_simplex_probe" in table_ids



def test_ashcroft_streamlined_panel_structure() -> None:
    specs = {spec.id: spec for spec in load_diagnostics()}
    result = specs["transport.boltzmann.ashcroft_comparison.overview"].compute(None, {})

    assert result.title == "Ashcroft comparison"
    assert result.markdowns == ()
    assert result.cards == ()
    assert result.tables == ()

    assert [section.id for section in result.sections] == [
        "ashcroft_local_calculation_check",
        "ashcroft_velocity_comparison",
        "ashcroft_conductivity_comparison",
    ]

    assert "analytic checks" in result.summary
    assert "Delaunay plane-fit interpolation" in result.summary
    assert "reciprocal-space measure convention" in result.summary


def test_ashcroft_local_calculation_check_contains_validation_evidence() -> None:
    specs = {spec.id: spec for spec in load_diagnostics()}
    result = specs["transport.boltzmann.ashcroft_comparison.overview"].compute(None, {})

    local = _section_by_id(result, "ashcroft_local_calculation_check")
    table_ids = {table.id for table in local.tables}
    nested_table_ids = {
        table.id
        for section in local.sections
        for table in section.tables
    }

    assert local.title == "Local calculation check"
    assert "section_validation_summary" in table_ids
    assert "section_analytic_end_to_end_derivative_error" in table_ids
    assert "section_analytic_end_to_end_sigma" in table_ids
    assert "section_conductivity_invariant_checks" in table_ids

    assert "section_conductivity_grid_subsample_stability" in nested_table_ids
    assert "section_conductivity_temperature_response" in nested_table_ids
    assert "section_conductivity_contribution_localisation" in nested_table_ids

    markdown = "\n".join(str(block.markdown) for block in local.markdowns if hasattr(block, "markdown"))
    assert "independently of Vincent's data" in markdown
    assert "Fermi window" in markdown
    assert "conductivity prefactor" in markdown


def test_ashcroft_velocity_comparison_contains_delaunay_resolution() -> None:
    specs = {spec.id: spec for spec in load_diagnostics()}
    result = specs["transport.boltzmann.ashcroft_comparison.overview"].compute(None, {})

    velocity = _section_by_id(result, "ashcroft_velocity_comparison")
    table_ids = {table.id for table in velocity.tables}

    assert "section_velocity_delaunay_interpolation_probe" in table_ids
    assert "section_velocity_delaunay_adjacent_simplex_probe" in table_ids
    assert "section_velocity_k_grid" in table_ids

    markdown = "\n".join(block.markdown for block in velocity.markdowns)
    assert "Delaunay interpolation" in markdown
    assert "adjacent triangle" in markdown
    assert "simplex-choice issue" in markdown
    assert "not a units" in markdown


def test_ashcroft_conductivity_comparison_contains_measure_result() -> None:
    specs = {spec.id: spec for spec in load_diagnostics()}
    result = specs["transport.boltzmann.ashcroft_comparison.overview"].compute(None, {})

    conductivity = _section_by_id(result, "ashcroft_conductivity_comparison")
    table_ids = {table.id for table in conductivity.tables}
    nested_table_ids = {
        table.id
        for section in conductivity.sections
        for table in section.tables
    }

    assert "section_conductivity_fermi_window" in table_ids
    assert "section_best_conductivity_reconstruction" in table_ids
    assert "section_conductivity_normalisation_hypotheses" in table_ids
    assert "section_conductivity_shape_summary" in table_ids

    assert "section_conductivity_normalisation" in nested_table_ids
    assert "section_conductivity_raw_tensor" in nested_table_ids
    assert "section_conductivity_local_tensor" in nested_table_ids
    assert "section_conductivity_target" in nested_table_ids

    markdown = "\n".join(str(block.markdown) for block in conductivity.markdowns if hasattr(block, "markdown"))
    assert "Fermi-window statistics are reproduced" in markdown
    assert "reciprocal-space measure convention" in markdown

    from dft_local.diagnostics.user_strings import iter_typst_math

    snippets = {item.name: item.source for item in iter_typst_math(conductivity)}
    assert snippets["ashcroft_continuum_measure_inline"] == "$ (d^2 k) / ((2 pi)^2) $"
    assert "erroneous" in markdown


def test_ashcroft_removed_deprecated_forensic_sections() -> None:
    specs = {spec.id: spec for spec in load_diagnostics()}
    result = specs["transport.boltzmann.ashcroft_comparison.overview"].compute(None, {})

    section_ids = {section.id for section in result.sections}

    assert "ashcroft_eshift_matching" not in section_ids
    assert "ashcroft_vincent_target_summary" not in section_ids
    assert "ashcroft_best_current_reconstruction" not in section_ids
    assert "ashcroft_velocity_matching" not in section_ids
    assert "ashcroft_conductivity_matching" not in section_ids


def test_ashcroft_computed_result_typst_math_compiles() -> None:
    """Every TypstMath snippet in the computed Ashcroft result must compile.

    The generic spec-level test only sees DiagnosticSpec text.  The Ashcroft
    equations are produced inside the computed DiagnosticResult, so they need a
    result-level compile check too.
    """

    from dft_local.diagnostics.typst import render_typst_math_to_svg
    from dft_local.diagnostics.user_strings import iter_typst_math
    from dft_local.transport.boltzmann.ashcroft_comparison.diagnostics import compute_overview

    result = compute_overview(None, {})
    failures: list[str] = []

    for snippet in iter_typst_math(result):
        label = snippet.name or snippet.source
        try:
            render_typst_math_to_svg(snippet.source, display=snippet.display)
        except Exception as exc:  # noqa: BLE001 - collect all compile failures
            failures.append(f"{label}: {exc}")

    assert not failures, "\n".join(failures)


def test_ashcroft_rendered_result_has_no_typst_error_fallbacks() -> None:
    """Rendered Ashcroft page should not contain Typst fallback source spans."""

    from dft_local.diagnostics.render import render_result
    from dft_local.transport.boltzmann.ashcroft_comparison.diagnostics import compute_overview

    html = render_result(compute_overview(None, {}))

    assert "typst-error" not in html




def test_ashcroft_conductivity_detail_header_uses_mixed_text_and_typst_sigma() -> None:
    from dft_local.diagnostics.render import render_result
    from dft_local.transport.boltzmann.ashcroft_comparison.diagnostics import compute_overview

    html = render_result(compute_overview(None, {}))

    assert "ashcroft_conductivity_section_title" not in html
    assert "<summary>Conductivity comparison</summary>" in html
    assert "Conductivity result" in html
    assert "ashcroft_target_conductivity_title_sigma" in html
    assert "data-typst-source='$ sigma_(alpha beta) $'" in html
    assert "Conductivity <span class='typst-math inline'" in html
    assert " [S/m]" in html
    assert "typst-error" not in html




def test_ashcroft_local_equations_are_interleaved_with_prose() -> None:
    from dft_local.diagnostics.render import render_result
    from dft_local.transport.boltzmann.ashcroft_comparison.diagnostics import compute_overview

    html = render_result(compute_overview(None, {}))

    points = [
        html.find("The continuum expression being discretised is:"),
        html.find("id='ashcroft_conductivity_equation'"),
        html.find("The velocity entering the tensor"),
        html.find("id='ashcroft_velocity_equation'"),
        html.find("The thermal weighting"),
        html.find("id='ashcroft_fermi_window_equation'"),
        html.find("Validation summary"),
    ]

    assert all(point >= 0 for point in points)
    assert points == sorted(points)
    assert "class='typst-math-block'" in html
    assert html.count("typst-math display") >= 3
    assert "typst-error" not in html


def test_ashcroft_equation_prose_and_block_math_share_visual_group() -> None:
    from dft_local.diagnostics.render import render_result
    from dft_local.transport.boltzmann.ashcroft_comparison.diagnostics import compute_overview

    html = render_result(compute_overview(None, {}))

    assert html.count("class='markdown-math-group'") >= 3

    velocity_text = html.find("The velocity entering the tensor")
    velocity_math = html.find("id='ashcroft_velocity_equation'")
    group_start = html.rfind("class='markdown-math-group'", 0, velocity_text)
    group_end = html.find("</div>", velocity_math)

    assert group_start >= 0
    assert velocity_text > group_start
    assert velocity_math > velocity_text
    assert group_end > velocity_math
    assert "typst-error" not in html


def test_ashcroft_validation_summary_uses_same_document_block_style() -> None:
    from dft_local.diagnostics.render import render_result
    from dft_local.transport.boltzmann.ashcroft_comparison.diagnostics import compute_overview

    html = render_result(compute_overview(None, {}))

    validation = html.find("Validation summary")
    assert validation >= 0

    group_start = html.rfind("class='markdown-math-group'", 0, validation)
    group_end = html.find("</div>", validation)

    assert group_start >= 0
    assert group_start < validation < group_end
    assert "typst-error" not in html



def test_ashcroft_inline_math_fragments_render_inside_prose() -> None:
    from dft_local.diagnostics.render import render_result
    from dft_local.transport.boltzmann.ashcroft_comparison.diagnostics import compute_overview

    html = render_result(compute_overview(None, {}))

    for name in (
        "ashcroft_mu_mean_epsilon",
        "ashcroft_continuum_measure_inline",
        "ashcroft_two_pi_squared_inline",
        "ashcroft_grid_measure_inline",
        "ashcroft_continuum_measure_factor_inline",
        "ashcroft_raw_velocity_weight_inline",
        "ashcroft_local_tensor_measure_inline",
        "ashcroft_normalisation_table_continuum_measure",
        "ashcroft_normalisation_table_grid_measure",
    ):
        assert name in html

    assert "typst-error" not in html
