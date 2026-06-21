"""Diagnostics for validating the Boltzmann operator approach."""

from __future__ import annotations

import numpy as np

from dft_local.core.units import (
    CHARGE,
    CONDUCTIVITY,
    DIMENSIONLESS,
    ELECTRON_VOLT,
    ENERGY,
    FEMTOSECOND,
    KELVIN,
    LENGTH,
    TEMPERATURE,
    TIME,
    VELOCITY,
    DisplayQuantity,
    Unit,
    diagnostic_scalar_quantity,
)
from dft_local.diagnostics.models import (
    DiagnosticResult,
    DiagnosticSection,
    DiagnosticSpec,
    InputSpec,
    MarkdownBlock,
    Table,
    TableRow,
)
from dft_local.transport.boltzmann.validation.core import (
    is_positive_semidefinite,
    tensor_invariant_report,
    validation_summary,
    weighted_outer_product_tensor,
    gd_symbol_production_validation_probe,
    finite_field_input_health_probe,
    FiniteFieldInputHealthProbe,
    finite_field_band_crossing_hazard_probe,
    FiniteFieldBandCrossingHazardProbe,
    finite_field_velocity_validation_probe,
    FiniteFieldVelocityValidationProbe,
    finite_field_unit_scaling_probe,
    FiniteFieldUnitScalingProbe,
    finite_field_analytic_toy_coverage_probe,
    FiniteFieldAnalyticToyCoverageProbe,
    finite_field_k_convergence_probe,
    FiniteFieldKConvergenceProbe,
    finite_field_symmetry_sanity_probe,
    FiniteFieldSymmetrySanityProbe,
    finite_field_vincent_reconstruction_probe,
    FiniteFieldVincentReconstructionProbe,
    finite_field_strong_dc_validation_probe,
    FiniteFieldStrongDcValidationProbe,
    finite_field_weak_dc_limit_probe,
    FiniteFieldWeakDcLimitProbe,
    FiniteFieldModeDecompositionProbe,
    finite_field_mode_decomposition_probe,
    operator_symbol_validation_probe,
)


def compute_overview(ctx, inputs) -> DiagnosticResult:
    summary = validation_summary()
    production_symbol_probe = gd_symbol_production_validation_probe()
    symbol_probe = operator_symbol_validation_probe()

    velocity = np.array(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[-1.0, 0.0], [0.0, -2.0]],
        ],
        dtype=np.float64,
    )
    weight = np.ones(velocity.shape[:-1], dtype=np.float64)

    tensor = weighted_outer_product_tensor(velocity, weight)
    invariants = tensor_invariant_report(tensor)

    return DiagnosticResult(
        title="Boltzmann operator validation",
        summary=summary.purpose,
        body=(
            DiagnosticSection(
                id="boltzmann_validation_scope",
                title="Validation scope",
                description="Purpose and planned checks for the operator-validation domain.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="boltzmann_validation_scope_commentary",
                        title="Why this domain exists",
                        markdown="""This domain is for validating the Boltzmann operator approach before comparing it with any external reference output.

The purpose is to build a chain of checks that are independent of Vincent, Ashcroft-specific data, or any particular printed intermediate value. Once these checks pass, external comparisons can be interpreted as convention matching problems rather than basic implementation validation.
""",
                    ),
                    Table(
                        id="boltzmann_validation_current_scope",
                        title="Current scope",
                        description="Initial scope of the validation domain.",
                        headers=("item", "description"),
                        rows=tuple(
                            TableRow((str(i + 1), item))
                            for i, item in enumerate(summary.current_scope)
                        ),
                    ),
                    Table(
                        id="boltzmann_validation_planned_checks",
                        title="Planned checks",
                        description="Checks to add as the operator approach is made explicit.",
                        headers=("check", "purpose"),
                        rows=tuple(
                            TableRow((item, "planned"))
                            for item in summary.planned_checks
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="boltzmann_validation_production_symbol_checks",
                title="Production symbol and derivative checks",
                description="Checks the same GdKernelArrays symbol and derivative path used for H(k) and S(k).",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="boltzmann_validation_production_symbol_checks_commentary",
                        title="What this checks",
                        markdown="""These checks use the production symbol mechanism, not a parallel FFT toy model.

The test kernel has a known analytic symbol,

    K(k1, k2) = c0 + c1 cos(k1) + c2 cos(k2)

and is represented as a `GdKernelArrays` object. The diagnostic then checks:

- `symbol_fixed` gives the expected analytic symbol
- `symbol_generic` gives the expected channels
- `gd_symbol_derivative_fixed` gives the expected analytic derivative
- `gd_symbol_derivative_generic` gives the expected derivative channels
- `SymbolPair(...).form()` produces the same local problem energy
- `gd_symbol_derivatives(...)` returns the same derivative symbols used by the Boltzmann calculation
- the production symbol reproduces the analytic energy surface over a grid
- the derivative of the operator agrees with the derivative of that energy surface

This is the validation layer closest to the actual `H(k)` and `S(k)` path.
""",
                    ),
                    Table(
                        id="boltzmann_validation_production_symbol_errors",
                        title="Production symbol error summary",
                        description="All errors should be near machine precision.",
                        headers=("check", "absolute error"),
                        rows=(
                            TableRow(("fixed symbol vs analytic K(k)", f"{production_symbol_probe['fixed_symbol_abs_error']:.8e}")),
                            TableRow(("generic symbol channels vs analytic K(k)", f"{production_symbol_probe['generic_symbol_channel_abs_error']:.8e}")),
                            TableRow(("fixed dK/dk1 vs analytic derivative", f"{production_symbol_probe['fixed_dk1_abs_error']:.8e}")),
                            TableRow(("fixed dK/dk2 vs analytic derivative", f"{production_symbol_probe['fixed_dk2_abs_error']:.8e}")),
                            TableRow(("generic dK/dk1 channels", f"{production_symbol_probe['generic_dk1_channel_abs_error']:.8e}")),
                            TableRow(("generic dK/dk2 channels", f"{production_symbol_probe['generic_dk2_channel_abs_error']:.8e}")),
                            TableRow(("SymbolPair energy", f"{production_symbol_probe['symbol_pair_energy_abs_error']:.8e}")),
                            TableRow(("gd_symbol_derivatives dk1", f"{production_symbol_probe['symbol_pair_dk1_abs_error']:.8e}")),
                            TableRow(("gd_symbol_derivatives dk2", f"{production_symbol_probe['symbol_pair_dk2_abs_error']:.8e}")),
                            TableRow(("energy surface max error", f"{production_symbol_probe['energy_surface_max_abs_error']:.8e}")),
                            TableRow(("energy-surface dk1 max error", f"{production_symbol_probe['energy_surface_dk1_max_abs_error']:.8e}")),
                            TableRow(("energy-surface dk2 max error", f"{production_symbol_probe['energy_surface_dk2_max_abs_error']:.8e}")),
                            TableRow(("Hellmann-Feynman dk1", f"{production_symbol_probe['hellmann_feynman_dk1_abs_error']:.8e}")),
                            TableRow(("Hellmann-Feynman dk2", f"{production_symbol_probe['hellmann_feynman_dk2_abs_error']:.8e}")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="boltzmann_validation_symbol_checks",
                title="Symbol and derivative checks",
                description="Checks that the finite-group symbol reproduces known operators and known energy-surface derivatives.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="boltzmann_validation_symbol_checks_commentary",
                        title="What this checks",
                        markdown="""These checks validate the finite-group symbol machinery.

They test:

- recovering an operator symbol from a known convolution kernel
- reconstructing the kernel from its symbol
- applying an operator through its symbol and reproducing the direct group function
- matching the analytic symbol of a periodic central-difference derivative
- applying the derivative operator to a known periodic energy surface and matching the expected derivative

This is the start of validating the operator approach independently of any external comparison data.
""",
                    ),
                    Table(
                        id="boltzmann_validation_symbol_errors",
                        title="Symbol and derivative error summary",
                        description="All values should be near machine precision.",
                        headers=("check", "relative error"),
                        rows=(
                            TableRow(("identity symbol reproduces Fourier mode", f"{symbol_probe['identity_mode_relative_error']:.8e}")),
                            TableRow(("kernel -> symbol -> kernel roundtrip", f"{symbol_probe['kernel_symbol_roundtrip_error']:.8e}")),
                            TableRow(("Dx kernel symbol matches analytic symbol", f"{symbol_probe['dx_symbol_relative_error']:.8e}")),
                            TableRow(("Dy kernel symbol matches analytic symbol", f"{symbol_probe['dy_symbol_relative_error']:.8e}")),
                            TableRow(("Dx energy-surface derivative", f"{symbol_probe['dx_energy_surface_relative_error']:.8e}")),
                            TableRow(("Dy energy-surface derivative", f"{symbol_probe['dy_energy_surface_relative_error']:.8e}")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="boltzmann_validation_outer_product_smoke_test",
                title="Weighted outer-product smoke test",
                description="Minimal algebraic check of the conductivity tensor structure.",
                collapsed=True,
                body=(
                    MarkdownBlock(
                        id="boltzmann_validation_outer_product_commentary",
                        title="What this checks",
                        markdown="""Relaxation-time conductivity is built from weighted velocity outer products.

A tensor of the form `sum_k w_k v(k) v(k)^T` with non-negative weights must be symmetric and positive semidefinite. This smoke test gives the new validation domain a concrete first invariant while more complete analytic operator tests are added.
""",
                    ),
                    Table(
                        id="boltzmann_validation_outer_product_tensor",
                        title="Smoke-test tensor",
                        description="Tensor assembled from a small artificial velocity grid with unit weights.",
                        headers=("component", "x", "y"),
                        rows=(
                            TableRow(("x", f"{tensor[0, 0]:.8e}", f"{tensor[0, 1]:.8e}")),
                            TableRow(("y", f"{tensor[1, 0]:.8e}", f"{tensor[1, 1]:.8e}")),
                        ),
                    ),
                    Table(
                        id="boltzmann_validation_outer_product_invariants",
                        title="Smoke-test invariants",
                        description="Basic invariants expected from a weighted outer-product tensor.",
                        headers=("quantity", "value", "target"),
                        rows=(
                            TableRow(("positive semidefinite", str(is_positive_semidefinite(tensor)), "True")),
                            TableRow(("trace", f"{invariants['trace']:.8e}", "> 0")),
                            TableRow(("minimum symmetric eigenvalue", f"{invariants['minimum_symmetric_eigenvalue']:.8e}", ">= 0")),
                            TableRow(("antisymmetric relative norm", f"{invariants['antisymmetric_relative_norm']:.8e}", "0")),
                            TableRow(("diagonal anisotropy / trace", f"{invariants['diagonal_anisotropy_over_trace']:.8e}", "diagnostic")),
                            TableRow(("offdiagonal / trace", f"{invariants['offdiagonal_over_trace']:.8e}", "diagnostic")),
                        ),
                    ),
                ),
            ),
        ),
    )


UNITLESS = Unit("", DIMENSIONLESS, 1.0)
PERCENT = Unit("%", DIMENSIONLESS, 0.01)
SIEMENS = Unit("S", (CHARGE ** 2) / (ENERGY * TIME), 1.0)
SIEMENS_PER_METER = Unit("S/m", CONDUCTIVITY, 1.0)
METER_PER_SECOND = Unit("m/s", VELOCITY, 1.0)
VOLT_PER_METER = Unit("V/m", ENERGY / (CHARGE * LENGTH), 1.0)


def _display_quantity(value: object, *, name: str = "finite-field value") -> DisplayQuantity:
    return DisplayQuantity(float(value), DIMENSIONLESS, UNITLESS, name=name)


def _fmt_probe_value(value):
    """Return typed diagnostic values, not preformatted number strings."""

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, np.floating)):
        return _display_quantity(value)
    if value is None:
        return None
    return str(value)


def _finite_field_input_health_rows(probe: FiniteFieldInputHealthProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "selected kernel source")),
        TableRow(("sample count", probe.sample_count, "N_u × N_v")),
        TableRow(("symmetrization", probe.symmetrization, "selected input")),
        TableRow(("H kernel star defect max", q("h_star_defect_max"), "near 0 after star; diagnostic value for raw/direct")),
        TableRow(("S kernel star defect max", q("s_star_defect_max"), "near 0 after star; diagnostic value for raw/direct")),
        TableRow(("H(k) Hermiticity defect rel max", q("h_hermitian_defect_rel_max"), "near 0")),
        TableRow(("S(k) Hermiticity defect rel max", q("s_hermitian_defect_rel_max"), "near 0")),
        TableRow(("min eig S(k)", q("s_eig_min"), "> 1e-10")),
        TableRow(("max cond S(k)", q("s_condition_number_abs_max"), "finite")),
        TableRow(("S positive", probe.s_positive, "True")),
        TableRow(("max neighbour energy jump", q("energy_neighbour_jump_max"), "coarse smoothness proxy; not a convergence proof")),
        TableRow(("symbol-grid convergence", "pending", "compare symbol/energy/velocity grids under N_u,N_v refinement")),
    )


def _finite_field_band_hazard_rows(probe: FiniteFieldBandCrossingHazardProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "controlled toy source")),
        TableRow(("sample count", probe.sample_count, "N_u × N_v")),
        TableRow(("toy mass", q("mass"), "small mass gives near-crossing stress test")),
        TableRow(("gap threshold", q("gap_threshold"), "hazard if gap below this")),
        TableRow(("minimum gap", q("min_gap"), "larger is safer for energy-ordered labels")),
        TableRow(("minimum-gap k1", q("min_gap_k1"), "location")),
        TableRow(("minimum-gap k2", q("min_gap_k2"), "location")),
        TableRow(("hazard count", probe.hazard_count, "number of sampled k-points below threshold")),
        TableRow(("hazard fraction", q("hazard_fraction"), "hazard count / sample count")),
        TableRow(("has hazard", probe.has_hazard, "diagnostic flag")),
        TableRow(("max band-0 neighbour jump", q("max_band0_neighbour_jump"), "energy-label smoothness proxy")),
        TableRow(("max band-1 neighbour jump", q("max_band1_neighbour_jump"), "energy-label smoothness proxy")),
        TableRow(("max gap neighbour jump", q("max_gap_neighbour_jump"), "gap smoothness proxy")),
    )


def _finite_field_velocity_rows(probe: FiniteFieldVelocityValidationProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "controlled production-symbol toy")),
        TableRow(("k1", q("k1"), "sample point")),
        TableRow(("k2", q("k2"), "sample point")),
        TableRow(("finite-difference epsilon", q("finite_difference_eps"), "step")),
        TableRow(("analytic dE/dk1", q("analytic_dk1"), "reference")),
        TableRow(("analytic dE/dk2", q("analytic_dk2"), "reference")),
        TableRow(("production derivative dk1 error", q("production_dk1_abs_error"), "near 0")),
        TableRow(("production derivative dk2 error", q("production_dk2_abs_error"), "near 0")),
        TableRow(("finite-difference dk1 error", q("finite_difference_dk1_abs_error"), "near finite-difference precision")),
        TableRow(("finite-difference dk2 error", q("finite_difference_dk2_abs_error"), "near finite-difference precision")),
        TableRow(("Hellmann-Feynman dk1 error", q("hellmann_feynman_dk1_abs_error"), "near 0")),
        TableRow(("Hellmann-Feynman dk2 error", q("hellmann_feynman_dk2_abs_error"), "near 0")),
        TableRow(("generic/fixed symbol error", q("generic_fixed_symbol_abs_error"), "near 0")),
        TableRow(("generic/fixed dk1 error", q("generic_fixed_dk1_abs_error"), "near 0")),
        TableRow(("generic/fixed dk2 error", q("generic_fixed_dk2_abs_error"), "near 0")),
        TableRow(("unit scaling status", probe.unit_scaling_status, "pending")),
        TableRow(("Vincent velocity status", probe.vincent_velocity_status, "pending")),
    )


def _finite_field_unit_scaling_rows(probe: FiniteFieldUnitScalingProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "unit constants and scaling conventions")),
        TableRow(("atomic energy to eV", q("atomic_energy_to_ev"), "27.21138386")),
        TableRow(("atomic length to Å", q("atomic_length_to_angstrom"), "0.52917721092")),
        TableRow(("hbar atomic", q("hbar_atomic"), "1 in atomic-unit context")),
        TableRow(("hbar eV Å context", q("hbar_ev_angstrom"), "seconds in eV working energy")),
        TableRow(("velocity AU to eVÅ factor", q("velocity_au_to_evag_factor"), "same physical velocity conversion")),
        TableRow(("expected velocity factor", q("expected_velocity_au_to_evag_factor"), "reference")),
        TableRow(("velocity factor abs error", q("velocity_factor_abs_error"), "near 0")),
        TableRow(("Fermi window eV from AU factor", q("fermi_window_ev_from_au_factor"), "inverse-energy conversion")),
        TableRow(("mu conversion required", probe.mu_conversion_required, "True")),
        TableRow(("conductivity SI status", probe.conductivity_si_status, "pending")),
    )


def _finite_field_analytic_toy_rows(probe: FiniteFieldAnalyticToyCoverageProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "summary of current real probes")),
        TableRow(("toy count", probe.toy_count, "controlled analytic cases")),
        TableRow(("cosine symbol max error", q("separable_cosine_symbol_max_error"), "near 0")),
        TableRow(("cosine derivative max error", q("separable_cosine_derivative_max_error"), "near 0")),
        TableRow(("identity overlap min eig", q("identity_overlap_min_eig"), "> 1e-10")),
        TableRow(("identity overlap condition", q("identity_overlap_condition"), "finite")),
        TableRow(("periodic Dirac min gap", q("periodic_dirac_min_gap"), "controlled near-crossing")),
        TableRow(("periodic Dirac hazard count", probe.periodic_dirac_hazard_count, ">= 1")),
        TableRow(("HF velocity max error", q("velocity_hf_max_error"), "near 0")),
        TableRow(("unit velocity factor error", q("unit_velocity_factor_error"), "near 0")),
        TableRow(("all current toys pass", probe.all_current_toys_pass, "True")),
        TableRow(("missing toy", probe.missing_toy, "next analytic target")),
    )


def _finite_field_k_convergence_rows(probe: FiniteFieldKConvergenceProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "controlled quadrature toy")),
        TableRow(("grid count", probe.grid_count, "number of refinements")),
        TableRow(("coarsest N", probe.coarsest_n, "first grid")),
        TableRow(("finest N", probe.finest_n, "last grid")),
        TableRow(("reference <|grad E|^2>", q("reference_average_grad_e_sq"), "analytic")),
        TableRow(("finest <|grad E|^2>", q("finest_average_grad_e_sq"), "numeric")),
        TableRow(("finest abs error", q("finest_abs_error"), "near 0")),
        TableRow(("max abs error", q("max_abs_error"), "near 0")),
        TableRow(("improved/equal steps", probe.improved_or_equal_steps, "non-regression count")),
        TableRow(("all grid errors small", probe.all_grid_errors_small, "True")),
        TableRow(("measure status", probe.measure_status, "normalisation check")),
        TableRow(("conductivity convergence status", probe.conductivity_convergence_status, "pending")),
    )


def _finite_field_symmetry_rows(probe: FiniteFieldSymmetrySanityProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "controlled symmetry toy")),
        TableRow(("sample count", probe.sample_count, "N × N")),
        TableRow(("E(k)-E(-k) max error", q("energy_inversion_max_error"), "near 0")),
        TableRow(("dk1 oddness max error", q("dk1_odd_max_error"), "near 0")),
        TableRow(("dk2 oddness max error", q("dk2_odd_max_error"), "near 0")),
        TableRow(("tensor xx", q("tensor_xx"), "positive")),
        TableRow(("tensor yy", q("tensor_yy"), "positive")),
        TableRow(("tensor xy", q("tensor_xy"), "near 0")),
        TableRow(("tensor yx", q("tensor_yx"), "near 0")),
        TableRow(("tensor antisym abs", q("tensor_antisym_abs"), "near 0")),
        TableRow(("all symmetry checks pass", probe.all_symmetry_checks_pass, "True")),
        TableRow(("dataset automorphism status", probe.dataset_automorphism_status, "pending")),
    )


def _finite_field_vincent_rows(probe: FiniteFieldVincentReconstructionProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "reused comparison domain")),
        TableRow(("Vincent target trace", q("target_trace"), "S/m")),
        TableRow(("weak-chain trace", q("weak_chain_trace"), "S/m")),
        TableRow(("weak-chain trace error %", q("weak_chain_trace_percent_error"), "residual")),
        TableRow(("strong-grid trace", q("strong_grid_trace"), "S/m")),
        TableRow(("strong-grid trace error %", q("strong_grid_trace_percent_error"), "separate spectral derivative residual")),
        TableRow(("shifted Eq. 8.30 trace", q("shifted_830_trace"), "S/m")),
        TableRow(("shifted Eq. 8.30 trace error %", q("shifted_830_trace_percent_error"), "hypothesis check")),
        TableRow(("find-simplex max velocity error", q("find_simplex_max_velocity_error"), "m/s")),
        TableRow(("best-adjacent max velocity error", q("best_adjacent_max_velocity_error"), "m/s")),
        TableRow(("velocity error reduction", q("velocity_error_reduction"), "large")),
        TableRow(("best adjacent matches Vincent", probe.best_adjacent_matches_vincent, "True")),
        TableRow(("residual status", probe.residual_status, "audit note")),
    )


def _finite_field_strong_dc_rows(probe: FiniteFieldStrongDcValidationProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "reused strong DC implementation")),
        TableRow(("mode count", probe.mode_count, "FFT modes")),
        TableRow(("nonzero mode count", probe.nonzero_mode_count, "active modes")),
        TableRow(("strong-grid trace", q("strong_grid_trace"), "S/m")),
        TableRow(("weak-chain grid trace", q("weak_chain_grid_trace"), "S/m")),
        TableRow(("Vincent target trace", q("vincent_target_trace"), "S/m")),
        TableRow(("strong/weak trace gap", q("strong_vs_weak_rel_trace_gap"), "known derivative residual")),
        TableRow(("strong/Vincent trace error %", q("strong_vs_vincent_percent_error"), "audit residual")),
        TableRow(("mode reconstruction abs error", q("mode_reconstruction_abs_error"), "near 0")),
        TableRow(("imaginary leakage", q("imaginary_leakage"), "near 0")),
        TableRow(("imaginary leakage ratio", q("imaginary_leakage_ratio"), "near 0")),
        TableRow(("strongest mode fraction", q("strongest_mode_fraction"), "finite")),
        TableRow(("occupation coeff shape", f"{probe.occupation_coeff_shape[0]} × {probe.occupation_coeff_shape[1]}", "grid shape")),
        TableRow(("response factor finite", probe.response_factor_finite, "True")),
        TableRow(("velocity coefficients finite", probe.velocity_coefficients_finite, "True")),
        TableRow(("strong DC internal pass", probe.strong_dc_internal_pass, "True")),
        TableRow(("residual status", probe.residual_status, "audit note")),
    )


def _finite_field_weak_dc_rows(probe: FiniteFieldWeakDcLimitProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "matched spectral-basis analytic toy")),
        TableRow(("field row count", probe.field_row_count, "eta sweep rows")),
        TableRow(("zero eta", q("zero_eta"), "0")),
        TableRow(("zero field", q("zero_field"), "V/m")),
        TableRow(("zero relative tensor discrepancy", q("zero_relative_tensor_discrepancy"), "near 0")),
        TableRow(("zero relative trace discrepancy", q("zero_relative_trace_discrepancy"), "near 0")),
        TableRow(("small eta", q("small_eta"), "first nonzero field")),
        TableRow(("small field", q("small_field"), "V/m")),
        TableRow(("small relative tensor discrepancy", q("small_relative_tensor_discrepancy"), "small")),
        TableRow(("small relative trace discrepancy", q("small_relative_trace_discrepancy"), "small")),
        TableRow(("largest eta", q("largest_eta"), "largest sweep field")),
        TableRow(("largest relative tensor discrepancy", q("largest_relative_tensor_discrepancy"), "nonlinear departure")),
        TableRow(("largest relative trace discrepancy", q("largest_relative_trace_discrepancy"), "nonlinear departure")),
        TableRow(("min nonzero eta", q("min_nonzero_eta"), "first finite field")),
        TableRow(("max eta", q("max_eta"), "largest finite field")),
        TableRow(("max field tensor discrepancy", q("max_field_tensor_discrepancy"), "finite")),
        TableRow(("max abs field trace discrepancy", q("max_abs_field_trace_discrepancy"), "finite")),
        TableRow(("relative weak-limit error", q("relative_weak_limit_error"), "near 0")),
        TableRow(("strong zero-field imaginary leakage", q("strong_zero_field_imaginary_leakage"), "near 0")),
        TableRow(("max imaginary leakage", q("max_imaginary_leakage"), "finite")),
        TableRow(("weak-limit pass", probe.weak_limit_pass, "True")),
        TableRow(("roundoff floor status", probe.roundoff_floor_status, "audit note")),
    )


def _finite_field_mode_decomposition_rows(probe: FiniteFieldModeDecompositionProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "strong DC mode result")),
        TableRow(("mode count", probe.mode_count, "FFT modes")),
        TableRow(("Gamma reconstruction abs error", q("gamma_reconstruction_abs_error"), "near 0")),
        TableRow(("rho reconstruction abs error", q("rho_reconstruction_abs_error"), "near 0")),
        TableRow(("mode tensor reconstruction abs error", q("mode_tensor_reconstruction_abs_error"), "near 0")),
        TableRow(("conductivity trace", q("conductivity_trace"), "S/m")),
        TableRow(("mode norm sum", q("conductivity_mode_norm_sum"), "finite")),
        TableRow(("top 1 mode fraction", q("top_1_mode_fraction"), "concentration")),
        TableRow(("top 10 mode fraction", q("top_10_mode_fraction"), "concentration")),
        TableRow(("top 100 mode fraction", q("top_100_mode_fraction"), "concentration")),
        TableRow(("Gamma abs max", q("gamma_abs_max"), "finite")),
        TableRow(("rho abs max", q("rho_abs_max"), "finite")),
        TableRow(("response abs max", q("response_abs_max"), "finite")),
        TableRow(("Gamma finite", probe.gamma_finite, "True")),
        TableRow(("rho finite", probe.rho_finite, "True")),
        TableRow(("response finite", probe.response_finite, "True")),
        TableRow(("mode tensor finite", probe.mode_tensor_finite, "True")),
        TableRow(("mode closure pass", probe.mode_closure_pass, "True")),
        TableRow(("residual status", probe.residual_status, "audit note")),
    )


def _finite_field_dc_inputs() -> tuple[InputSpec, ...]:
    return (
        InputSpec(
            "dataset",
            "Dataset",
            "str",
            "default",
            help="Dataset or fixture name used for the selected validation run.",
        ),
        InputSpec(
            "temperature",
            "Temperature T",
            "float",
            300.0,
            min_value=0.0,
            help="Temperature used in the Fermi window.",
        ),
        InputSpec(
            "mu",
            "Chemical potential μ",
            "float",
            1.23644,
            help="Chemical potential for the selected run, in the active energy units.",
        ),
        InputSpec(
            "tau",
            "Relaxation time τ",
            "float",
            1.0,
            min_value=0.0,
            help="Relaxation time used by the DC conductivity formula.",
        ),
        InputSpec(
            "units",
            "Units",
            "select",
            "eVAng",
            options=(("eVAng", "eV Å"), ("au", "atomic units"), ("si", "SI")),
            help="Internal unit convention for the validation calculation.",
        ),
        InputSpec(
            "n_u",
            "N_u",
            "int",
            11,
            min_value=1,
            help="Number of k-points in the first reciprocal coordinate.",
        ),
        InputSpec(
            "n_v",
            "N_v",
            "int",
            11,
            min_value=1,
            help="Number of k-points in the second reciprocal coordinate.",
        ),
        InputSpec(
            "electric_field",
            "Electric field E",
            "float",
            1.0,
            min_value=0.0,
            help="Finite electric-field strength for the selected run.",
        ),
        InputSpec(
            "theta",
            "Field angle θ",
            "float",
            0.0,
            help="Field direction angle in radians.",
        ),
        InputSpec(
            "band_index",
            "Band index n",
            "int",
            0,
            min_value=0,
            help="Energy-ordered band index to inspect.",
        ),
        InputSpec(
            "kernel_choice",
            "Kernel choice",
            "select",
            "average",
            options=(("anchored", "anchored"), ("average", "average")),
            help="Dataset-backed kernel family to inspect before applying the selected symmetrization scheme.",
        ),
        InputSpec(
            "symmetrization",
            "Symmetrization",
            "select",
            "star",
            options=(("star", "star"), ("direct", "direct"), ("raw", "raw")),
            help="Symmetrization scheme used to build or post-process the symbols.",
        ),
    )


def _finite_field_dc_input_rows(inputs) -> tuple[TableRow, ...]:
    return (
        TableRow(("dataset", str(inputs.get("dataset", "default")))),
        TableRow(("temperature T", DisplayQuantity(float(inputs.get("temperature", 300.0)), TEMPERATURE, KELVIN, name="temperature"))),
        TableRow(("chemical potential mu", DisplayQuantity(float(inputs.get("mu", 1.23644)), ENERGY, ELECTRON_VOLT, name="mu"))),
        TableRow(("relaxation time tau", DisplayQuantity(float(inputs.get("tau", 1.0)), TIME, FEMTOSECOND, name="tau"))),
        TableRow(("units", str(inputs.get("units", "eVAng")))),
        TableRow(("N_u", int(inputs.get("n_u", 11)))),
        TableRow(("N_v", int(inputs.get("n_v", 11)))),
        TableRow(("electric field E", DisplayQuantity(float(inputs.get("electric_field", 1.0)), ENERGY / (CHARGE * LENGTH), VOLT_PER_METER, name="electric field"))),
        TableRow(("field angle theta", DisplayQuantity(float(inputs.get("theta", 0.0)), DIMENSIONLESS, UNITLESS, name="theta"))),
        TableRow(("band index n", int(inputs.get("band_index", 0)))),
        TableRow(("kernel choice", str(inputs.get("kernel_choice", "average")))),
        TableRow(("symmetrization scheme", str(inputs.get("symmetrization", "star")))),
        TableRow(("band label convention", "energy ordering at each sampled k")),
        TableRow(("reciprocal 2π normalization", "physical audit required; not cosmetic")),
        TableRow(("k-domain / reciprocal measure", "sampled reciprocal cell; audited against Vincent and analytic checks")),
        TableRow(("conductivity normalization", "SI conductivity in S/m after reciprocal-measure and prefactor audit")),
        TableRow(("report status", "first-pass validation bundle with explicit pending rows")),
    )




def compute_finite_field_dc_validation(ctx, inputs) -> DiagnosticResult:
    """First-pass report for validating finite-field, band-labelled DC conductivity."""

    dashboard_rows = (
        TableRow(("input health", "complete", "selected H/S symbols define stable generalized eigenproblems")),
        TableRow(("band-crossing hazards", "toy-only", "near crossings and label hazards are mapped on a controlled k-space toy")),
        TableRow(("velocity validation", "partial", "analytic and finite-difference checks are live; Gamma/Vincent checks remain explicit pending rows")),
        TableRow(("Vincent reconstruction", "open-audit", "velocity samples are resolved; 2π normalization and residual few-percent conductivity gap remain visible audit items")),
        TableRow(("strong DC contact", "partial", "strong band-labelled result is checked on Vincent-grid inputs; shared-regime comparison still being tightened")),
        TableRow(("weak DC limit", "toy-only", "finite-field result approaches weak-field result as E -> 0 in matched spectral-basis toy")),
        TableRow(("mode closure", "partial", "Gamma, F, and tilde(rho) reconstruct the strong-grid conductivity object; dataset-backed closure remains pending")),
        TableRow(("analytic toys", "partial", "periodic known-input tests are live; finite-field Gamma/F/rho toy remains pending")),
        TableRow(("unit consistency", "partial", "core unit factors are checked; full end-to-end SI conductivity agreement remains pending")),
        TableRow(("k convergence", "toy-only", "periodic quadrature convergence is checked; dataset-backed conductivity convergence remains pending")),
        TableRow(("symmetry sanity", "toy-only", "toy tensor symmetries are checked; dataset-backed direction sweep remains pending")),
    )

    input_rows = _finite_field_dc_input_rows(inputs)
    kernel_choice = str(inputs.get("kernel_choice", "average"))
    symmetrization = str(inputs.get("symmetrization", "star"))

    if ctx is None:
        KH_input = None
        KS_input = None
        input_health_source = "controlled production GdKernelArrays toy; diagnostic context missing"
    else:
        if kernel_choice not in {"anchored", "average"}:
            raise ValueError(f"unknown kernel choice for finite-field input health: {kernel_choice!r}")
        KH_input, KS_input = ctx.kernels(kernel_choice)
        input_health_source = f"dataset-backed {kernel_choice} GdKernelArrays"

    input_health_probe = finite_field_input_health_probe(
        KH_input,
        KS_input,
        n_u=int(inputs.get("n_u", 11)),
        n_v=int(inputs.get("n_v", 11)),
        symmetrization=symmetrization,
        source=input_health_source,
    )
    band_hazard_probe = finite_field_band_crossing_hazard_probe(
        n_u=int(inputs.get("n_u", 11)),
        n_v=int(inputs.get("n_v", 11)),
    )
    velocity_probe = finite_field_velocity_validation_probe()
    unit_scaling_probe = finite_field_unit_scaling_probe()
    analytic_toy_probe = finite_field_analytic_toy_coverage_probe()
    k_convergence_probe = finite_field_k_convergence_probe()
    symmetry_probe = finite_field_symmetry_sanity_probe()
    vincent_probe = finite_field_vincent_reconstruction_probe()
    strong_dc_probe = finite_field_strong_dc_validation_probe()
    weak_dc_probe = finite_field_weak_dc_limit_probe()
    mode_decomposition_probe = finite_field_mode_decomposition_probe()

    return DiagnosticResult(
        title="Finite-field DC validation",
        summary="First-pass validation report for finite-field, band-labelled DC conductivity and its lattice-mode decomposition.",
        body=(
            DiagnosticSection(
                id="finite_field_dc_validation_overview",
                title="Overview",
                description="Top-level validation dashboard and report spine.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_overview_prose",
                        title="Validation target",
                        markdown="""This diagnostic validates the finite-field, band-labelled DC conductivity.

The target object is not the band-free formula and not only the Hellmann-Feynman derivative. The target is the finite-field band-labelled conductivity tensor, together with its lattice-mode decomposition into `Gamma`, `F`, and `tilde(rho)`.

The diagnostic is arranged as a validation ladder: first the inputs, then the velocity ingredients, then Vincent reconstruction, then strong and weak DC consistency, then mode closure, analytic tests, unit scaling, k-point convergence, and symmetry sanity.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_dashboard_status_key",
                        title="Status key",
                        description="Meaning of dashboard status labels. These labels describe evidence maturity, not final scientific truth.",
                        headers=("status", "meaning"),
                        rows=(
                            TableRow(("complete", "current evidence directly supports the section claim for the selected target")),
                            TableRow(("partial", "some real evidence exists, but important checks remain")),
                            TableRow(("toy-only", "real evidence exists, but only on analytic or synthetic toy inputs")),
                            TableRow(("open-audit", "real evidence exists, but a known discrepancy remains under investigation")),
                            TableRow(("pending", "planned evidence is visible but not implemented yet")),
                        ),
                    ),
                    Table(
                        id="finite_field_dc_validation_dashboard",
                        title="Validation dashboard",
                        description="First-pass status summary. Later sections provide the evidence and explicit pending rows.",
                        headers=("category", "status", "claim"),
                        rows=dashboard_rows,
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_inputs",
                title="Inputs",
                description="Calculation inputs, provenance, and conventions for the selected validation run.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_inputs_prose",
                        title="Why this block matters",
                        markdown="""The manifest freezes the calculation being validated.

The `2π` normalization is listed explicitly because it changes the physical conductivity number. It is not treated as a harmless notation choice once velocities, reciprocal-space measure, and SI conductivity normalization are assembled.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_inputs_table",
                        title="Selected run",
                        description="Selected inputs, conventions, and normalization choices for this validation run.",
                        headers=("input", "value"),
                        rows=input_rows,
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_input_health",
                title="Input health",
                description="Algebraic and sampling checks for the selected H and S symbol construction.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_input_health_prose",
                        title="Validation claim",
                        markdown="""**Claim.** For the selected symmetrization scheme, the Hamiltonian and overlap symbols define stable Hermitian generalized eigenproblems over the sampled k-domain.

This section uses the selected dataset-backed `GdKernelArrays` kernel family when diagnostic context is available, then applies the selected symmetrization scheme. `raw` reports the extracted kernels as-is, `star` applies kernel-level star symmetrization, and `direct` applies Hermitian projection after forming each local symbol. The overlap symbol is checked as Hermitian positive definite, not unitary.

The input-health table reports Hermiticity at two different stages. **H kernel star defect max** checks the local kernel before forming any k-space symbol, by comparing each inverse-paired block K_H(g^{-1}) with the adjoint of K_H(g). A nonzero value means the extracted local kernel is not itself star/Hermitian compatible. **H(k) Hermiticity defect rel max** checks the dense symbol after evaluating the kernel over the sampled k-grid. In `star` mode both should fall to roundoff; in `direct` mode only the formed symbol is repaired, so the kernel defect may remain.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_input_health_table",
                        title="Input-health metrics",
                        description="First real health table for the finite-field validation diagnostic.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_input_health_rows(input_health_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_input_health_placeholders",
                        title="Remaining input-health visual audits",
                        description="Dataset-backed scalar checks are now live. The remaining checks are visual audits needed to inspect smoothness and conditioning over the sampled k-domain.",
                        headers=("check", "status"),
                        rows=(
                            TableRow(("dataset-backed H/S health table", "complete")),
                            TableRow(("symbol smoothness plot", "pending")),
                            TableRow(("condition-number k-map", "pending")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_band_crossing_hazards",
                title="Band-crossing hazards",
                description="Validates the k-space hazard logic for fragile energy-ordered band labels.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_band_crossing_hazards_prose",
                        title="Validation claim",
                        markdown="""**Claim.** The diagnostic can identify k-space regions where near crossings make energy-ordered band labels fragile.

This first implementation uses a periodic two-level Dirac-like toy model. It is not a selected-band graphene map yet. It exists to make the hazard logic concrete: compute adjacent-band gaps, locate the minimum gap, count points below a threshold, and report neighbour-jump smoothness proxies. The production version should use the selected `band_index`, report the adjacent-band gap map, and overlay velocity anomalies near the flagged regions.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_band_crossing_hazards_table",
                        title="Band-crossing hazard metrics",
                        description="Toy-backed hazard table for energy-ordered band labels.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_band_hazard_rows(band_hazard_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_band_crossing_hazards_placeholders",
                        title="Remaining selected-band checks",
                        description="Dataset-backed checks needed before this section explains anomalies in the chosen conductivity band.",
                        headers=("check", "status"),
                        rows=(
                            TableRow(("selected-band adjacent-gap k-map", "pending")),
                            TableRow(("selected-band minimum-gap location table", "pending")),
                            TableRow(("velocity anomaly overlay near gap hazards", "pending")),
                            TableRow(("eigenvector-overlap / label-jump k-map", "pending")),
                            TableRow(("degenerate-subspace fallback check", "pending")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_velocity_validation",
                title="Velocity validation",
                description="Checks the derivative machinery used to build band velocities before conductivity is tested.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_velocity_validation_prose",
                        title="Validation claim",
                        markdown="""**Claim.** On a controlled periodic production-symbol toy, the velocity ingredients agree across analytic derivatives, finite differences, generalized Hellmann-Feynman derivatives, and fixed/generic symbol conventions.

This first implementation uses the separable cosine production-symbol toy. It validates the derivative machinery that finite-field conductivity will reuse. Physical `hbar`/unit-context scaling is handled in the unit-scaling section. Modal Gamma reconstruction and Vincent velocity comparison remain explicit pending rows rather than hidden assumptions.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_velocity_validation_table",
                        title="Velocity validation metrics",
                        description="First real velocity table for the finite-field validation diagnostic.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_velocity_rows(velocity_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_velocity_validation_placeholders",
                        title="Remaining production velocity checks",
                        description="Dataset-backed and Vincent-backed checks still needed before this section fully validates production band velocities.",
                        headers=("check", "status"),
                        rows=(
                            TableRow(("Gamma modal reconstruction", "pending")),
                            TableRow(("Vincent velocity comparison table", "pending")),
                            TableRow(("physical hbar/unit-context scaling", "pending")),
                            TableRow(("velocity k-map", "pending")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_vincent_reconstruction",
                title="Vincent reconstruction",
                description="Audits reconstruction of Vincent's reference calculation and isolates remaining convention differences.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_vincent_reconstruction_prose",
                        title="Validation claim",
                        markdown="""**Claim.** The implementation exposes which parts of Vincent's reference calculation are reconstructed and which parts remain convention or formula audit items.

This first finite-field validation section reuses the existing Ashcroft/Vincent comparison domain. It shows that the velocity samples are resolved by the adjacent-simplex Delaunay ambiguity. It also reports weak-chain, shifted Eq. 8.30, and strong-grid conductivity traces against Vincent's target trace. The conductivity residual is kept visible as an open audit item rather than treated as a solved validation proof.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_vincent_reconstruction_table",
                        title="Vincent reconstruction metrics",
                        description="Summary of existing Ashcroft/Vincent reconstruction checks.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_vincent_rows(vincent_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_vincent_reconstruction_placeholders",
                        title="Remaining Vincent audit checks",
                        description="Reconstruction checks still needed before this section can claim full agreement with Vincent's reference calculation.",
                        headers=("check", "status"),
                        rows=(
                            TableRow(("component-level Vincent tensor table", "pending")),
                            TableRow(("explicit 2π/grid-measure ablation table", "pending")),
                            TableRow(("chemical-potential convention sweep", "pending")),
                            TableRow(("direct finite-field reproduction against Vincent inputs", "pending")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_strong_dc_validation",
                title="Strong DC validation",
                description="Checks the band-indexed strong spectral DC tensor on Vincent-grid inputs before dataset-backed finite-field runs.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_strong_dc_validation_prose",
                        title="Validation claim",
                        markdown="""**Claim.** The band-labelled strong DC conductivity is internally closed as a lattice-mode spectral tensor and its residual against the weak-chain calculation is exposed rather than hidden.

This first implementation reuses the existing `BandIndexedStrongDcResult` on Vincent's epsilon grid. It checks that the mode tensor re-sums to the reported strong tensor, that Fourier coefficients and response factors are finite, and that imaginary leakage is negligible.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_strong_dc_validation_table",
                        title="Strong DC validation metrics",
                        description="Band-indexed strong spectral tensor checks on Vincent inputs.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_strong_dc_rows(strong_dc_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_strong_dc_validation_placeholders",
                        title="Remaining strong DC checks",
                        description="Checks still needed before this section validates the full dataset-backed finite-field conductivity target.",
                        headers=("check", "status"),
                        rows=(
                            TableRow(("component-level strong tensor table", "pending")),
                            TableRow(("temperature / smoothness regime table", "pending")),
                            TableRow(("nonzero-field response sweep", "pending")),
                            TableRow(("dataset-backed band-labelled strong DC run", "pending")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_weak_dc_limit",
                title="Weak DC limit",
                description="Small-field limit check on a matched spectral-basis analytic finite-field toy.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_weak_dc_limit_prose",
                        title="Validation claim",
                        markdown="""**Claim.** On a matched spectral-basis analytic toy, the finite-field DC conductivity approaches the weak-field DC result as E -> 0.

This first implementation reuses the analytic sinusoidal Ashcroft probe. It verifies the zero-field limit and reports finite-field departures across an eta sweep. It separates the clean matched-basis weak limit from the Vincent-grid derivative-definition residual exposed in the strong DC section. Dataset-backed electric-field sweeps remain pending.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_weak_dc_limit_table",
                        title="Weak DC limit metrics",
                        description="Matched spectral-basis strong/weak finite-field sweep.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_weak_dc_rows(weak_dc_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_weak_dc_limit_placeholders",
                        title="Remaining weak-limit checks",
                        description="Checks still needed before this section validates the dataset-backed finite-field weak-limit target.",
                        headers=("check", "status"),
                        rows=(
                            TableRow(("dataset-backed E sweep", "pending")),
                            TableRow(("finite-minus-weak error plot", "pending")),
                            TableRow(("asymptotic-window table", "pending")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_mode_decomposition",
                title="Mode decomposition",
                description="Closure checks for Gamma, F, and tilde(rho) on the strong-grid mode object.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_mode_decomposition_prose",
                        title="Validation claim",
                        markdown="""**Claim.** On the current strong-grid mode object, the lattice-mode decomposition into Gamma, F, and tilde(rho) reconstructs the sampled fields and total strong spectral DC tensor.

This first implementation checks the actual `BandIndexedStrongDcResult` mode objects: Gamma reconstructs the sampled velocity field, tilde(rho) reconstructs the sampled occupation, and summing the conductivity mode tensor reconstructs the total strong DC tensor. This validates the mode algebra used by the finite-field target, but dataset-backed finite-field mode closure remains pending.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_mode_decomposition_table",
                        title="Mode decomposition metrics",
                        description="Gamma/F/rho closure checks for the strong spectral DC tensor.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_mode_decomposition_rows(mode_decomposition_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_mode_decomposition_placeholders",
                        title="Remaining mode-decomposition checks",
                        description="Mode visualisation and dataset-backed finite-field closure checks still required.",
                        headers=("check", "status"),
                        rows=(
                            TableRow(("component-level mode contribution table", "pending")),
                            TableRow(("cumulative mode contribution curve", "pending")),
                            TableRow(("spatial mode maps", "pending")),
                            TableRow(("dataset-backed Gamma/F/rho closure", "pending")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_analytic_toys",
                title="Analytic toys",
                description="Closed-form systems used to test the implementation independent of BigDFT data.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_analytic_toys_prose",
                        title="Validation claim",
                        markdown="""**Claim.** Controlled analytic toys isolate algebra, derivatives, crossings, overlap health, and unit scaling before any BigDFT dataset-specific effects are introduced.

This section now summarises the real toy-backed probes used by the validation ladder. The missing analytic target is the finite-field lattice-mode closure toy for Gamma/F/rho decomposition.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_analytic_toys_table",
                        title="Analytic toy coverage",
                        description="Summary of currently implemented controlled analytic probes.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_analytic_toy_rows(analytic_toy_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_analytic_toys_placeholders",
                        title="Remaining analytic toy checks",
                        description="Analytic toy coverage still needed before the full finite-field validation target is independently covered.",
                        headers=("check", "status"),
                        rows=(
                            TableRow(("finite-field Gamma/F/rho closure toy", "pending")),
                            TableRow(("periodic two-band conductivity toy", "pending")),
                            TableRow(("active-subspace near-crossing toy", "pending")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_unit_scaling",
                title="Unit consistency",
                description="Core physical scaling factors needed for SI conversion checks.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_unit_scaling_prose",
                        title="Validation claim",
                        markdown="""**Claim.** The core unit factors needed by finite-field velocity and conductivity comparisons are explicit and numerically checked.

This first implementation checks the conversion factors that later calculation-level comparisons depend on: Hartree to eV, Bohr to Å, hbar in each working context, velocity scaling, inverse-energy Fermi-window scaling, and the requirement that mu be converted with the Hamiltonian. It does not yet prove that a full conductivity calculation is invariant under changing internal unit systems; that remains an explicit pending check.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_unit_scaling_table",
                        title="Unit consistency metrics",
                        description="First real unit table for the finite-field validation diagnostic.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_unit_scaling_rows(unit_scaling_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_unit_scaling_placeholders",
                        title="Remaining unit-consistency checks",
                        description="Calculation-level unit checks still needed before this section validates full SI conductivity invariance.",
                        headers=("check", "status"),
                        rows=(
                            TableRow(("same physical velocity AU/eVÅ calculation", "covered in Boltzmann tests; pending local summary")),
                            TableRow(("same physical conductivity AU/eVÅ/SI calculation", "pending")),
                            TableRow(("tau scaling plot", "pending")),
                            TableRow(("finite-field E scaling law table", "pending")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_k_convergence",
                title="k-point convergence",
                description="Grid-measure refinement checks before dataset-backed conductivity convergence.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_k_convergence_prose",
                        title="Validation claim",
                        markdown="""**Claim.** On a periodic analytic velocity-square toy, the sampled k-grid measure matches the exact full-period average under N_u/N_v refinement.

This first implementation checks the grid-measure part of conductivity convergence on a controlled periodic integrand. It is not yet a dataset-backed conductivity convergence table, but it does exercise the normalisation convention before the stronger conductivity comparison is wired in.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_k_convergence_table",
                        title="k-point convergence metrics",
                        description="First real k-grid convergence table for the finite-field validation diagnostic.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_k_convergence_rows(k_convergence_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_k_convergence_placeholders",
                        title="Remaining k-convergence checks",
                        description="Dataset-backed conductivity convergence still required for the full finite-field validation.",
                        headers=("check", "status"),
                        rows=(
                            TableRow(("sigma component convergence", "pending")),
                            TableRow(("trace / norm convergence", "pending")),
                            TableRow(("weak-limit error convergence", "pending")),
                            TableRow(("Vincent residual convergence", "pending")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_symmetry",
                title="Symmetry sanity",
                description="Toy tensor and k-inversion checks before dataset-backed lattice symmetry validation.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_symmetry_prose",
                        title="Validation claim",
                        markdown="""**Claim.** On a controlled separable-cosine toy, the expected k-inversion and velocity-square tensor symmetries are satisfied up to numerical roundoff.

This first implementation checks even energy under k inversion, odd derivatives under k inversion, and a symmetric diagonal velocity-square tensor with vanishing cross component. It does not yet validate dataset-backed H/S/H_star/S_star automorphisms or graphene direction-sweep symmetries; those remain explicit pending checks.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_symmetry_table",
                        title="Symmetry sanity metrics",
                        description="First real symmetry table for the finite-field validation diagnostic.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_symmetry_rows(symmetry_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_symmetry_placeholders",
                        title="Remaining symmetry checks",
                        description="Dataset-backed symmetry checks still required for the full finite-field validation.",
                        headers=("check", "status"),
                        rows=(
                            TableRow(("H/S/H_star/S_star automorphism checks", "pending")),
                            TableRow(("direction-sweep lattice periodicity", "pending")),
                            TableRow(("real graphene k inversion checks", "pending")),
                            TableRow(("finite-sample symmetry defect", "pending")),
                        ),
                    ),
                ),
            ),
        ),
    )


def diagnostics() -> tuple[DiagnosticSpec, ...]:
    return (
        DiagnosticSpec(
            id="transport.boltzmann.validation.overview",
            group="transport.boltzmann.validation",
            title="Boltzmann operator validation",
            description="Validate the Boltzmann operator approach independently of reference-output matching.",
            inputs=(),
            compute=compute_overview,
        ),
        DiagnosticSpec(
            id="transport.boltzmann.validation.finite_field_dc",
            group="transport.boltzmann.validation",
            title="Finite-field DC validation",
            description="Validate finite-field, band-labelled DC conductivity and its Gamma/F/tilde(rho) lattice-mode decomposition.",
            inputs=_finite_field_dc_inputs(),
            compute=compute_finite_field_dc_validation,
        ),
    )
