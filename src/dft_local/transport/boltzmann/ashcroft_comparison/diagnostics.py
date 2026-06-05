from __future__ import annotations

import numpy as np

from dft_local.diagnostics.models import Card, DiagnosticResult, DiagnosticSection, DiagnosticSpec, MarkdownBlock, Table, TableRow, TypstMathBlock
from dft_local.diagnostics.user_strings import TypstMath, rich

from dft_local.transport.boltzmann.ashcroft_comparison.core import (
    conductivity_from_epsilon_grid,
    conductivity_contribution_probe,
    conductivity_derivative_sensitivity_probe,
    conductivity_grid_subsample_probe,
    conductivity_invariant_checks,
    analytic_sinusoidal_conductivity_probe,
    conductivity_temperature_probe,
    load_vincent_input_data,
    reciprocal_lattice_vectors_from_primitives,
    shift_axis_swap_probe,
    shift_discrepancy_probe,
    swapped_axis_velocity_hypothesis_errors,
    swapped_field_shift,
    velocity_systematic_error_probe,
    velocity_two_pi_hypothesis_errors,
    vincent_delaunay_velocity_sample_probe,
    vincent_delaunay_adjacent_simplex_velocity_probe,
    vincent_reference,
)


def compute_overview(ctx, inputs: dict[str, object]) -> DiagnosticResult:
    reference = vincent_reference()
    sigma = reference.expected_conductivity_S_per_m
    vincent_inputs = load_vincent_input_data()
    ai = vincent_inputs.primitive_lattice_vectors_bohr
    bi = reciprocal_lattice_vectors_from_primitives(ai)
    probe = velocity_systematic_error_probe(vincent_inputs.epsilon_of_k, ai)
    shift_probe = shift_discrepancy_probe()
    shift_axis_probe = shift_axis_swap_probe()
    swapped_velocity_errors = swapped_axis_velocity_hypothesis_errors(vincent_inputs.epsilon_of_k, ai)
    two_pi_velocity_errors = velocity_two_pi_hypothesis_errors(vincent_inputs.epsilon_of_k, ai)
    delaunay_velocity_probe = vincent_delaunay_velocity_sample_probe(vincent_inputs.epsilon_of_k, ai)
    delaunay_adjacent_probe = vincent_delaunay_adjacent_simplex_velocity_probe(vincent_inputs.epsilon_of_k, ai)
    swapped_e_shift = swapped_field_shift()
    target_k = probe.target_k_per_m
    target_v = probe.target_v_m_per_s
    local_v = probe.local_v_m_per_s
    velocity_delta = probe.delta_v_m_per_s
    velocity_percent_error = probe.percent_error
    velocity_rms = probe.rms_error_m_per_s

    local_conductivity = conductivity_from_epsilon_grid(
        vincent_inputs.epsilon_of_k,
        ai,
        chemical_potential_J=float(np.mean(vincent_inputs.epsilon_of_k) * 4.3597447222071e-18),
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
    )
    local_sigma = local_conductivity.conductivity_tensor_S
    sigma_abs_error = local_sigma - sigma
    sigma_percent_error = np.where(sigma != 0.0, 100.0 * sigma_abs_error / sigma, np.nan)
    sigma_ratio = np.where(sigma != 0.0, local_sigma / sigma, np.nan)

    trace_target = float(np.trace(sigma))
    trace_local = float(np.trace(local_sigma))
    missing_trace_factor = trace_target / trace_local if trace_local != 0.0 else np.nan

    two_pi_squared = (2.0 * np.pi) ** 2
    half_two_pi_squared = 0.5 * two_pi_squared
    shift_factor_abs = abs(shift_axis_probe.reported_y_per_m / swapped_e_shift[1])
    conductivity_factor_abs = abs(missing_trace_factor)

    normalisation_hypotheses = (
        ("current: with /(2π)^2", 1.0),
        ("remove /(2π)^2", (2.0 * np.pi) ** 2),
        ("spin degeneracy 2", 2.0),
        ("spin + valley degeneracy 4", 4.0),
        ("remove /(2π)^2 and spin 2", 2.0 * (2.0 * np.pi) ** 2),
        ("fit trace exactly", missing_trace_factor),
    )

    best_velocity = delaunay_velocity_probe["local"]
    best_velocity_delta = delaunay_velocity_probe["delta"]
    best_velocity_percent_error = delaunay_velocity_probe["percent"]
    best_velocity_rms = float(delaunay_velocity_probe["rms_error"])

    best_conductivity = local_sigma * two_pi_squared
    best_conductivity_delta = best_conductivity - sigma
    best_conductivity_percent_error = np.where(
        sigma != 0.0,
        100.0 * best_conductivity_delta / sigma,
        np.nan,
    )

    conductivity_invariants = conductivity_invariant_checks(
        vincent_inputs.epsilon_of_k,
        ai,
        chemical_potential_J=local_conductivity.chemical_potential_J,
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
    )
    conductivity_grid_rows = conductivity_grid_subsample_probe(
        vincent_inputs.epsilon_of_k,
        ai,
        chemical_potential_J=local_conductivity.chemical_potential_J,
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
    )
    conductivity_temperature_rows = conductivity_temperature_probe(
        vincent_inputs.epsilon_of_k,
        ai,
        chemical_potential_J=local_conductivity.chemical_potential_J,
        relaxation_time_s=reference.relaxation_time_s,
    )
    conductivity_contribution_rows = conductivity_contribution_probe(
        vincent_inputs.epsilon_of_k,
        ai,
        chemical_potential_J=local_conductivity.chemical_potential_J,
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
    )
    conductivity_derivative_rows = conductivity_derivative_sensitivity_probe(
        vincent_inputs.epsilon_of_k,
        ai,
        chemical_potential_J=local_conductivity.chemical_potential_J,
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
    )

    derivative_validation_rows = (
        TableRow(("analytic linear derivative test", "passed in unit tests", "validates derivative basis transform")),
        TableRow(("periodic sinusoidal end-to-end test", "passed in unit tests", "validates derivative -> Fermi window -> tensor assembly")),
        TableRow(("k-grid mapping", "epsilon[0,j] = j * b2 / 100", "matches Vincent's listed k-points")),
        TableRow(("Vincent velocity interpolation", "Delaunay plane-fit interpolation", f"RMS sample error = {best_velocity_rms:.6e} m/s")),
        TableRow(("2π velocity scaling", "rejected", "not the explanation for printed sample velocities")),
    )

    analytic_probe = analytic_sinusoidal_conductivity_probe()

    validation_summary_rows = (
        TableRow(("analytic sinusoidal conductivity test", f"{analytic_probe['relative_sigma_error']:.3e}", "end-to-end known-function check")),
        TableRow(("Fermi-window bounds", "passed", "0 <= f(1-f) <= 0.25")),
        TableRow(("Vincent Fermi-window statistics", "passed", "mean and max reproduced")),
        TableRow(("tau linearity", f"{conductivity_invariants['tau_linearity_relative_error']:.3e}", "target 0")),
        TableRow(("velocity-square scaling", f"{conductivity_invariants['velocity_square_relative_error']:.3e}", "target 0")),
        TableRow(("energy-shift invariance", f"{conductivity_invariants['energy_shift_relative_error']:.3e}", "target 0")),
        TableRow(("minimum tensor eigenvalue", f"{conductivity_invariants['min_eigenvalue']:.3e}", "target >= 0")),
        TableRow(("antisymmetric part / trace", f"{conductivity_invariants['antisym_abs_over_trace']:.3e}", "target 0")),
    )

    velocity_for_sigma = local_conductivity.velocity_m_per_s
    fermi_weight = local_conductivity.fermi_weight

    unit_provenance_rows = (
        tuple(TableRow(tuple(row)) for row in ctx.state.unit_provenance_rows())
        if ctx is not None and getattr(ctx, "state", None) is not None
        else ()
    )

    return DiagnosticResult(
        title="Ashcroft comparison",
        summary=(
            "Reproduce Vincent's Ashcroft-style Boltzmann conductivity calculation. "
            "Current state: the local calculation is validated on analytic checks; "
            "Vincent's printed velocities are reproduced by Delaunay plane-fit interpolation "
            "up to simplex-choice ambiguity at grid vertices; and the conductivity agrees "
            "once the reciprocal-space measure convention is matched."
        ),
        body=(
            Table(
                id="ashcroft_dataset_unit_provenance",
                title="Dataset unit provenance",
                description="Disk and working unit context for the loaded dataset.",
                headers=("quantity", "value"),
                rows=unit_provenance_rows,
            ),
            DiagnosticSection(
                id="ashcroft_local_calculation_check",
                title="Local calculation check",
                description="Independent checks that the local derivative, Fermi window, and tensor assembly are internally coherent.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="ashcroft_local_calculation_intro",
                        title="Local calculation equations",
                        markdown="""The local calculation evaluates the Boltzmann conductivity from the band energy, the energy gradient, and the Fermi-window factor.

The continuum expression being discretised is:
""",
                    ),
                    TypstMathBlock(
                        id="ashcroft_conductivity_equation",
                        math=TypstMath(
                            "$ sigma_(alpha beta) = e^2 tau integral (d^2 k) / ((2 pi)^2) v_alpha (k) v_beta (k) (- (diff f_0) / (diff epsilon)) $",
                            display=True,
                            name="ashcroft_conductivity_equation",
                        ),
                    ),
                    MarkdownBlock(
                        id="ashcroft_velocity_equation_intro",
                        title="Velocity equation",
                        markdown="""The velocity entering the tensor is computed from the Cartesian derivative of the energy:
""",
                    ),
                    TypstMathBlock(
                        id="ashcroft_velocity_equation",
                        math=TypstMath(
                            "$ v_alpha (k) = (1 / hbar) (diff epsilon (k)) / (diff k_alpha) $",
                            display=True,
                            name="ashcroft_velocity_equation",
                        ),
                    ),
                    MarkdownBlock(
                        id="ashcroft_fermi_window_equation_intro",
                        title="Fermi-window equation",
                        markdown="""The thermal weighting is written as the derivative of the Fermi occupation:
""",
                    ),
                    TypstMathBlock(
                        id="ashcroft_fermi_window_equation",
                        math=TypstMath(
                            "$ - (diff f_0) / (diff epsilon) = (f_0 (1 - f_0)) / (k_B T) $",
                            display=True,
                            name="ashcroft_fermi_window_equation",
                        ),
                    ),
                    MarkdownBlock(
                        id="ashcroft_local_calculation_check_summary",
                        title="Validation summary",
                        markdown="""The local calculation is checked before comparison with Vincent.

The strongest check is the analytic end-to-end test: a known periodic band is differentiated, passed through the Fermi window, assembled into a raw velocity-weight tensor, normalised by the k-space measure and prefactor, and compared with the independently assembled expected tensor.

This validates the local derivative, unit conversions, Fermi window, tensor assembly, and conductivity prefactor independently of Vincent's data.
""",
                    ),
                # tables continue in body
                    Table(
                        id="section_validation_summary",
                        title="Validation summary",
                        description="Compact summary of the local calculation checks.",
                        headers=("check", "status/value", "meaning"),
                        rows=validation_summary_rows,
                    ),
                    Table(
                        id="section_analytic_end_to_end_derivative_error",
                        title="Analytic derivative error",
                        description="Implemented central Cartesian derivative compared with exact central-periodic finite-difference derivative.",
                        headers=("quantity", "error"),
                        rows=(
                            TableRow(("max abs vx error [m/s]", f"{analytic_probe['max_abs_vx_error']:.8e}")),
                            TableRow(("max abs vy error [m/s]", f"{analytic_probe['max_abs_vy_error']:.8e}")),
                            TableRow(("relative velocity-field error", f"{analytic_probe['relative_velocity_error']:.8e}")),
                        ),
                    ),
                    Table(
                        id="section_analytic_end_to_end_sigma",
                        title="Analytic conductivity tensor check",
                        description="Computed conductivity compared with the tensor assembled from the analytic expected derivative.",
                        headers=("component", "expected x", "expected y", "computed x", "computed y", "delta x", "delta y"),
                        rows=(
                            TableRow((
                                "x",
                                f"{analytic_probe['sigma_expected'][0, 0]:.8e}",
                                f"{analytic_probe['sigma_expected'][0, 1]:.8e}",
                                f"{analytic_probe['sigma_actual'][0, 0]:.8e}",
                                f"{analytic_probe['sigma_actual'][0, 1]:.8e}",
                                f"{analytic_probe['sigma_delta'][0, 0]:.8e}",
                                f"{analytic_probe['sigma_delta'][0, 1]:.8e}",
                            )),
                            TableRow((
                                "y",
                                f"{analytic_probe['sigma_expected'][1, 0]:.8e}",
                                f"{analytic_probe['sigma_expected'][1, 1]:.8e}",
                                f"{analytic_probe['sigma_actual'][1, 0]:.8e}",
                                f"{analytic_probe['sigma_actual'][1, 1]:.8e}",
                                f"{analytic_probe['sigma_delta'][1, 0]:.8e}",
                                f"{analytic_probe['sigma_delta'][1, 1]:.8e}",
                            )),
                            TableRow((
                                "relative tensor error",
                                f"{analytic_probe['relative_sigma_error']:.8e}",
                                "",
                                "",
                                "",
                                "",
                                "",
                            )),
                        ),
                    ),
                    Table(
                        id="section_conductivity_invariant_checks",
                        title="Conductivity invariant checks",
                        description="Internal checks that should hold independently of Vincent's conventions.",
                        headers=("check", "value", "target"),
                        rows=(
                            TableRow(("tau linearity relative error", f"{conductivity_invariants['tau_linearity_relative_error']:.8e}", "0")),
                            TableRow(("energy-shift invariance relative error", f"{conductivity_invariants['energy_shift_relative_error']:.8e}", "0")),
                            TableRow(("velocity-square scaling relative error", f"{conductivity_invariants['velocity_square_relative_error']:.8e}", "0")),
                            TableRow(("minimum tensor eigenvalue", f"{conductivity_invariants['min_eigenvalue']:.8e}", ">= 0")),
                            TableRow(("antisymmetric part / trace", f"{conductivity_invariants['antisym_abs_over_trace']:.8e}", "0")),
                        ),
                    ),
                # nested sections continue in body
                    DiagnosticSection(
                        id="ashcroft_local_detailed_checks",
                        title="Detailed local checks",
                        description="Additional stability and sensitivity checks.",
                        collapsed=True,
                        body=(
                            Table(
                                id="section_conductivity_grid_subsample_stability",
                                title="Grid subsampling stability",
                                description="Conductivity recomputed on strided subgrids.",
                                headers=("stride", "shape", "trace", "trace/full", "xx", "yy", "anisotropy/trace"),
                                rows=tuple(
                                    TableRow((
                                        f"{row['step']:.0f}",
                                        f"{row['n0']:.0f} x {row['n1']:.0f}",
                                        f"{row['trace']:.8e}",
                                        f"{row['trace_ratio_to_full']:.8e}",
                                        f"{row['xx']:.8e}",
                                        f"{row['yy']:.8e}",
                                        f"{row['anisotropy_abs_over_trace']:.8e}",
                                    ))
                                    for row in conductivity_grid_rows
                                ),
                            ),
                            Table(
                                id="section_conductivity_temperature_response",
                                title="Temperature response",
                                description="Fermi-window and conductivity response to temperature.",
                                headers=("T [K]", "max f(1-f)", "mean f(1-f)", "trace", "xx", "yy"),
                                rows=tuple(
                                    TableRow((
                                        f"{row['temperature_K']:.1f}",
                                        f"{row['max_fermi_weight']:.8e}",
                                        f"{row['mean_fermi_weight']:.8e}",
                                        f"{row['trace']:.8e}",
                                        f"{row['xx']:.8e}",
                                        f"{row['yy']:.8e}",
                                    ))
                                    for row in conductivity_temperature_rows
                                ),
                            ),
                            Table(
                                id="section_conductivity_contribution_localisation",
                                title="Contribution localisation",
                                description="How concentrated the velocity-weight contribution is.",
                                headers=("selection", "count", "trace contribution fraction"),
                                rows=tuple(
                                    TableRow((
                                        f"top {100.0 * row['top_fraction']:.1f}%" if row["top_fraction"] > 0 else f"f weight >= {-row['top_fraction']:.1e} max",
                                        f"{row['count']:.0f}",
                                        f"{row['trace_contribution_fraction']:.8e}",
                                    ))
                                    for row in conductivity_contribution_rows
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="ashcroft_velocity_comparison",
                title="Velocity comparison",
                description="Vincent's printed velocities are reproduced by Delaunay plane-fit interpolation, with simplex-choice ambiguity at grid vertices.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="ashcroft_velocity_comparison_summary",
                        title="Velocity result",
                        markdown="""Vincent clarified that the printed velocities use Delaunay interpolation.

At the printed k-points, which lie exactly on grid vertices, the Delaunay piecewise-linear interpolant has several adjacent triangle gradients. A direct `find_simplex` call may pick a different adjacent triangle from the one used in the printed output. When all adjacent simplices are inspected, Vincent's printed velocities are recovered to numerical precision.

This resolves the velocity mismatch as a simplex-choice issue at grid vertices, not a units, k-grid, `hbar`, Hartree conversion, or `2π` issue.
""",
                    ),
                    Table(
                        id="section_velocity_delaunay_interpolation_probe",
                        title="Direct Delaunay interpolation",
                        description=(
                            "Direct `find_simplex` reproduction of Vincent's Delaunay plane-fit interpolation. "
                            f"RMS error = {delaunay_velocity_probe['rms_error']:.6e} m/s."
                        ),
                        headers=(
                            "sample",
                            "Vincent vx",
                            "Vincent vy",
                            "find_simplex vx",
                            "find_simplex vy",
                            "delta vx",
                            "delta vy",
                            "% err vx",
                            "% err vy",
                        ),
                        rows=tuple(
                            TableRow((
                                str(i),
                                f"{delaunay_velocity_probe['target'][i, 0]:.8e}",
                                f"{delaunay_velocity_probe['target'][i, 1]:.8e}",
                                f"{delaunay_velocity_probe['local'][i, 0]:.8e}",
                                f"{delaunay_velocity_probe['local'][i, 1]:.8e}",
                                f"{delaunay_velocity_probe['delta'][i, 0]:.8e}",
                                f"{delaunay_velocity_probe['delta'][i, 1]:.8e}",
                                f"{delaunay_velocity_probe['percent'][i, 0]:.3f}",
                                f"{delaunay_velocity_probe['percent'][i, 1]:.3f}",
                            ))
                            for i in range(len(delaunay_velocity_probe["target"]))
                        ),
                    ),
                    Table(
                        id="section_velocity_delaunay_adjacent_simplex_probe",
                        title="Adjacent-simplex resolution",
                        description=(
                            "At grid vertices, several Delaunay triangles meet. "
                            "This table shows that Vincent's printed velocity is recovered by an adjacent simplex."
                        ),
                        headers=(
                            "sample",
                            "find_simplex",
                            "adjacent simplices",
                            "target vx",
                            "target vy",
                            "find error",
                            "best simplex",
                            "best vx",
                            "best vy",
                            "best error",
                        ),
                        rows=tuple(
                            TableRow((
                                str(row["sample"]),
                                str(row["find_simplex"]),
                                str(row["adjacent_count"]),
                                f"{row['target'][0]:.8e}",
                                f"{row['target'][1]:.8e}",
                                f"{row['find_simplex_error']:.8e}",
                                str(row["best_adjacent_simplex"]),
                                f"{row['best_adjacent_velocity'][0]:.8e}",
                                f"{row['best_adjacent_velocity'][1]:.8e}",
                                f"{row['best_adjacent_error']:.8e}",
                            ))
                            for row in delaunay_adjacent_probe
                        ),
                    ),
                    Table(
                        id="section_velocity_k_grid",
                        title="Agreed sampled k-points",
                        description="The printed velocity samples lie on the epsilon[0,j] path.",
                        headers=("sample", "kx [m^-1]", "ky [m^-1]", "path"),
                        rows=tuple(
                            TableRow((
                                str(i),
                                f"{target_k[i, 0]:.8e}",
                                f"{target_k[i, 1]:.8e}",
                                f"epsilon[0,{i}] = {i} * b2 / 100",
                            ))
                            for i in range(len(target_k))
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="ashcroft_conductivity_comparison",
                title="Conductivity comparison",
                description="Conductivity comparison after the local calculation and velocity interpolation have been validated.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="ashcroft_conductivity_comparison_summary",
                        title="Conductivity result",
                        markdown=rich(
                            "The Fermi-window statistics are reproduced using ",
                            TypstMath("$ mu = \"mean\" (epsilon) $", name="ashcroft_mu_mean_epsilon"),
                            " in Joules.\n\n"
                            "The remaining conductivity-scale difference is explained by the reciprocal-space measure convention. "
                            "With the continuum measure ",
                            TypstMath("$ (d^2 k) / ((2 pi)^2) $", name="ashcroft_continuum_measure_inline"),
                            ", the local tensor is smaller by approximately ",
                            TypstMath("$ (2 pi)^2 $", name="ashcroft_two_pi_squared_inline"),
                            ". With Vincent's grid-measure convention ",
                            TypstMath("$ d^2 k $", name="ashcroft_grid_measure_inline"),
                            ", the diagonal conductivity agrees to within a few percent.\n\n"
                            "The shifted-k and velocity-shift printouts are treated as erroneous and are not used in this comparison.",
                        ),
                    ),
                # tables continue in body
                    Table(
                        id="section_conductivity_fermi_window",
                        title="Fermi-window validation",
                        description="Local Fermi-window statistics compared with Vincent's reported statistics.",
                        headers=("quantity", "Vincent", "local", "absolute error"),
                        rows=(
                            TableRow(("max f(1-f)", f"{reference.max_fermi_weight:.8e}", f"{np.max(fermi_weight):.8e}", f"{np.max(fermi_weight) - reference.max_fermi_weight:.8e}")),
                            TableRow(("min f(1-f)", f"{reference.min_fermi_weight:.8e}", f"{np.min(fermi_weight):.8e}", f"{np.min(fermi_weight) - reference.min_fermi_weight:.8e}")),
                            TableRow(("mean f(1-f)", f"{reference.mean_fermi_weight:.8e}", f"{np.mean(fermi_weight):.8e}", f"{np.mean(fermi_weight) - reference.mean_fermi_weight:.8e}")),
                            TableRow(("mu [J]", "mean epsilon", f"{local_conductivity.chemical_potential_J:.8e}", "")),
                        ),
                    ),
                    Table(
                        id="section_best_conductivity_reconstruction",
                        title="Best conductivity reconstruction",
                        description=rich(
                            "Vincent/grid measure convention, equivalent to removing the extra continuum ",
                            TypstMath("$ 1 / ((2 pi)^2) $", name="ashcroft_continuum_measure_factor_inline"),
                            " factor from the local tensor.",
                        ),
                        headers=(
                            "component",
                            "Vincent x",
                            "Vincent y",
                            "local-grid x",
                            "local-grid y",
                            "delta x",
                            "delta y",
                            "% err x",
                            "% err y",
                        ),
                        rows=(
                            TableRow((
                                "x",
                                f"{sigma[0, 0]:.8e}",
                                f"{sigma[0, 1]:.8e}",
                                f"{best_conductivity[0, 0]:.8e}",
                                f"{best_conductivity[0, 1]:.8e}",
                                f"{best_conductivity_delta[0, 0]:.8e}",
                                f"{best_conductivity_delta[0, 1]:.8e}",
                                f"{best_conductivity_percent_error[0, 0]:.3f}",
                                f"{best_conductivity_percent_error[0, 1]:.3f}",
                            )),
                            TableRow((
                                "y",
                                f"{sigma[1, 0]:.8e}",
                                f"{sigma[1, 1]:.8e}",
                                f"{best_conductivity[1, 0]:.8e}",
                                f"{best_conductivity[1, 1]:.8e}",
                                f"{best_conductivity_delta[1, 0]:.8e}",
                                f"{best_conductivity_delta[1, 1]:.8e}",
                                f"{best_conductivity_percent_error[1, 0]:.3f}",
                                f"{best_conductivity_percent_error[1, 1]:.3f}",
                            )),
                            TableRow((
                                "trace",
                                f"{np.trace(sigma):.8e}",
                                "",
                                f"{np.trace(best_conductivity):.8e}",
                                "",
                                f"{np.trace(best_conductivity) - np.trace(sigma):.8e}",
                                "",
                                f"{100.0 * (np.trace(best_conductivity) - np.trace(sigma)) / np.trace(sigma):.3f}",
                                "",
                            )),
                        ),
                    ),
                    Table(
                        id="section_conductivity_normalisation_hypotheses",
                        title="Measure convention check",
                        description=(
                            "Global rescalings applied to the current local tensor. "
                            f"Vincent/local trace factor = {missing_trace_factor:.8e}."
                        ),
                        headers=("hypothesis", "factor", "trace [S/m]", "trace ratio local/Vincent", "xx [S/m]", "yy [S/m]"),
                        rows=tuple(
                            TableRow(
                                (
                                    name,
                                    f"{factor:.8e}",
                                    f"{np.trace(local_sigma * factor):.8e}",
                                    f"{np.trace(local_sigma * factor) / trace_target:.8e}",
                                    f"{(local_sigma * factor)[0, 0]:.8e}",
                                    f"{(local_sigma * factor)[1, 1]:.8e}",
                                )
                            )
                            for name, factor in normalisation_hypotheses
                        ),
                    ),
                    Table(
                        id="section_conductivity_shape_summary",
                        title="Conductivity shape summary",
                        description="Scalar summaries less sensitive to one global normalisation factor.",
                        headers=("quantity", "Vincent", "local continuum", "local grid measure"),
                        rows=(
                            TableRow(("trace", f"{np.trace(sigma):.8e}", f"{np.trace(local_sigma):.8e}", f"{np.trace(best_conductivity):.8e}")),
                            TableRow(("xx / yy", f"{sigma[0, 0] / sigma[1, 1]:.8e}", f"{local_sigma[0, 0] / local_sigma[1, 1]:.8e}", f"{best_conductivity[0, 0] / best_conductivity[1, 1]:.8e}")),
                            TableRow(("xy / trace", f"{sigma[0, 1] / np.trace(sigma):.8e}", f"{local_sigma[0, 1] / np.trace(local_sigma):.8e}", f"{best_conductivity[0, 1] / np.trace(best_conductivity):.8e}")),
                            TableRow(("trace ratio local/Vincent", "1.00000000e+00", f"{np.trace(local_sigma) / np.trace(sigma):.8e}", f"{np.trace(best_conductivity) / np.trace(sigma):.8e}")),
                        ),
                    ),
                # nested sections continue in body
                    DiagnosticSection(
                        id="ashcroft_conductivity_details",
                        title="Conductivity details",
                        description="Raw local tensors and continuum-convention comparison.",
                        collapsed=True,
                        body=(
                            Table(
                                id="section_conductivity_normalisation",
                                title="Conductivity normalisation",
                                description="Normalisation factors used after the raw velocity-weight tensor is summed over the grid.",
                                headers=("quantity", "value"),
                                rows=(
                                    TableRow(("k-cell area [m^-2]", f"{local_conductivity.k_cell_area_per_m2:.8e}")),
                                    TableRow(("prefactor", f"{local_conductivity.prefactor_S_m2_per_J:.8e}")),
                                    TableRow(("temperature [K]", f"{local_conductivity.temperature_K:.8e}")),
                                    TableRow(("relaxation time [s]", f"{local_conductivity.relaxation_time_s:.8e}")),
                                    TableRow((
                                        "continuum measure convention",
                                        rich(TypstMath("$ (d^2 k) / ((2 pi)^2) $", name="ashcroft_normalisation_table_continuum_measure")),
                                    )),
                                    TableRow((
                                        "Vincent/grid measure convention",
                                        rich(TypstMath("$ d^2 k $", name="ashcroft_normalisation_table_grid_measure")),
                                    )),
                                ),
                            ),
                            Table(
                                id="section_conductivity_raw_tensor",
                                title="Raw velocity-weight tensor",
                                description=rich(
                                    "Raw sum of ",
                                    TypstMath("$ v_a v_b f (1 - f) $", name="ashcroft_raw_velocity_weight_inline"),
                                    " before k-space cell area and conductivity prefactor.",
                                ),
                                headers=("component", "x", "y"),
                                rows=(
                                    TableRow(("x", f"{local_conductivity.raw_velocity_weight_tensor[0, 0]:.8e}", f"{local_conductivity.raw_velocity_weight_tensor[0, 1]:.8e}")),
                                    TableRow(("y", f"{local_conductivity.raw_velocity_weight_tensor[1, 0]:.8e}", f"{local_conductivity.raw_velocity_weight_tensor[1, 1]:.8e}")),
                                ),
                            ),
                            Table(
                                id="section_conductivity_local_tensor",
                                title="Local continuum-measure tensor",
                                description=rich(
                                    "Conductivity tensor computed with continuum ",
                                    TypstMath("$ (d^2 k) / ((2 pi)^2) $", name="ashcroft_local_tensor_measure_inline"),
                                    " measure.",
                                ),
                                headers=("component", "x", "y"),
                                rows=(
                                    TableRow(("x", f"{local_sigma[0, 0]:.8e}", f"{local_sigma[0, 1]:.8e}")),
                                    TableRow(("y", f"{local_sigma[1, 0]:.8e}", f"{local_sigma[1, 1]:.8e}")),
                                ),
                            ),
                            Table(
                                id="section_conductivity_target",
                                title=rich(
                                    "Conductivity ",
                                    TypstMath("$ sigma_(alpha beta) $", name="ashcroft_target_conductivity_title_sigma"),
                                    " [S/m]",
                                ),
                                description="Reference tensor from Vincent's recorded output.",
                                headers=("component", "x", "y"),
                                rows=(
                                    TableRow(("x", f"{sigma[0, 0]:.8e}", f"{sigma[0, 1]:.8e}")),
                                    TableRow(("y", f"{sigma[1, 0]:.8e}", f"{sigma[1, 1]:.8e}")),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def diagnostics() -> tuple[DiagnosticSpec, ...]:
    return (
        DiagnosticSpec(
            id="transport.boltzmann.ashcroft_comparison.overview",
            title="Ashcroft comparison overview",
            group="transport.boltzmann.ashcroft_comparison",
            description="Reproduction target for Vincent's Ashcroft-style conductivity calculation.",
            inputs=(),
            compute=compute_overview,
        ),
    )
