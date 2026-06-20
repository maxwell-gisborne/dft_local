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
    finite_field_band_crossing_hazard_probe,
    finite_field_velocity_validation_probe,
    finite_field_unit_scaling_probe,
    finite_field_analytic_toy_coverage_probe,
    finite_field_k_convergence_probe,
    finite_field_symmetry_sanity_probe,
    finite_field_vincent_reconstruction_probe,
    finite_field_strong_dc_validation_probe,
    finite_field_weak_dc_limit_probe,
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


def _finite_field_input_health_rows(probe: dict[str, object]) -> tuple[TableRow, ...]:
    return (
        TableRow(("source", _fmt_probe_value(probe["source"]), "controlled first implementation")),
        TableRow(("sample count", _fmt_probe_value(probe["sample_count"]), "N_u × N_v")),
        TableRow(("symmetrization", _fmt_probe_value(probe["symmetrization"]), "selected input")),
        TableRow(("H kernel star defect max", _fmt_probe_value(probe["h_star_defect_max"]), "near 0")),
        TableRow(("S kernel star defect max", _fmt_probe_value(probe["s_star_defect_max"]), "near 0")),
        TableRow(("H(k) Hermiticity defect rel max", _fmt_probe_value(probe["h_hermitian_defect_rel_max"]), "near 0")),
        TableRow(("S(k) Hermiticity defect rel max", _fmt_probe_value(probe["s_hermitian_defect_rel_max"]), "near 0")),
        TableRow(("min eig S(k)", _fmt_probe_value(probe["s_eig_min"]), "> 1e-10")),
        TableRow(("max cond S(k)", _fmt_probe_value(probe["s_condition_number_abs_max"]), "finite")),
        TableRow(("S positive", _fmt_probe_value(probe["s_positive"]), "True")),
        TableRow(("max neighbour energy jump", _fmt_probe_value(probe["energy_neighbour_jump_max"]), "smoothness proxy")),
    )


def _finite_field_band_hazard_rows(probe: dict[str, object]) -> tuple[TableRow, ...]:
    return (
        TableRow(("source", _fmt_probe_value(probe["source"]), "controlled first implementation")),
        TableRow(("sample count", _fmt_probe_value(probe["sample_count"]), "N_u × N_v")),
        TableRow(("toy mass", _fmt_probe_value(probe["mass"]), "small mass gives near-crossing stress test")),
        TableRow(("gap threshold", _fmt_probe_value(probe["gap_threshold"]), "hazard if gap below this")),
        TableRow(("minimum gap", _fmt_probe_value(probe["min_gap"]), "larger is safer for energy-ordered labels")),
        TableRow(("minimum-gap k1", _fmt_probe_value(probe["min_gap_k1"]), "location")),
        TableRow(("minimum-gap k2", _fmt_probe_value(probe["min_gap_k2"]), "location")),
        TableRow(("hazard count", _fmt_probe_value(probe["hazard_count"]), "number of sampled k-points below threshold")),
        TableRow(("hazard fraction", _fmt_probe_value(probe["hazard_fraction"]), "hazard count / sample count")),
        TableRow(("has hazard", _fmt_probe_value(probe["has_hazard"]), "diagnostic flag")),
        TableRow(("max band-0 neighbour jump", _fmt_probe_value(probe["max_band0_neighbour_jump"]), "energy-label smoothness proxy")),
        TableRow(("max band-1 neighbour jump", _fmt_probe_value(probe["max_band1_neighbour_jump"]), "energy-label smoothness proxy")),
        TableRow(("max gap neighbour jump", _fmt_probe_value(probe["max_gap_neighbour_jump"]), "gap-map smoothness proxy")),
    )


def _finite_field_velocity_rows(probe: dict[str, object]) -> tuple[TableRow, ...]:
    return (
        TableRow(("source", _fmt_probe_value(probe["source"]), "controlled first implementation")),
        TableRow(("k1", _fmt_probe_value(probe["k1"]), "sample point")),
        TableRow(("k2", _fmt_probe_value(probe["k2"]), "sample point")),
        TableRow(("analytic dE/dk1", _fmt_probe_value(probe["analytic_dk1"]), "reference")),
        TableRow(("analytic dE/dk2", _fmt_probe_value(probe["analytic_dk2"]), "reference")),
        TableRow(("production derivative dk1 error", _fmt_probe_value(probe["production_dk1_abs_error"]), "near 0")),
        TableRow(("production derivative dk2 error", _fmt_probe_value(probe["production_dk2_abs_error"]), "near 0")),
        TableRow(("finite-difference dk1 error", _fmt_probe_value(probe["finite_difference_dk1_abs_error"]), "near finite-difference precision")),
        TableRow(("finite-difference dk2 error", _fmt_probe_value(probe["finite_difference_dk2_abs_error"]), "near finite-difference precision")),
        TableRow(("Hellmann-Feynman dk1 error", _fmt_probe_value(probe["hellmann_feynman_dk1_abs_error"]), "near 0")),
        TableRow(("Hellmann-Feynman dk2 error", _fmt_probe_value(probe["hellmann_feynman_dk2_abs_error"]), "near 0")),
        TableRow(("generic/fixed symbol error", _fmt_probe_value(probe["generic_fixed_symbol_abs_error"]), "near 0")),
        TableRow(("generic/fixed dk1 error", _fmt_probe_value(probe["generic_fixed_dk1_abs_error"]), "near 0")),
        TableRow(("generic/fixed dk2 error", _fmt_probe_value(probe["generic_fixed_dk2_abs_error"]), "near 0")),
        TableRow(("unit scaling status", _fmt_probe_value(probe["unit_scaling_status"]), "pending")),
        TableRow(("Vincent velocity status", _fmt_probe_value(probe["vincent_velocity_status"]), "pending")),
    )


def _finite_field_unit_scaling_rows(probe: dict[str, object]) -> tuple[TableRow, ...]:
    return (
        TableRow(("source", _fmt_probe_value(probe["source"]), "controlled first implementation")),
        TableRow(("atomic energy to eV", _fmt_probe_value(probe["atomic_energy_to_ev"]), "27.21138386")),
        TableRow(("atomic length to Å", _fmt_probe_value(probe["atomic_length_to_angstrom"]), "0.52917721092")),
        TableRow(("hbar atomic", _fmt_probe_value(probe["hbar_atomic"]), "1 in atomic-unit context")),
        TableRow(("hbar eV Å context", _fmt_probe_value(probe["hbar_ev_angstrom"]), "seconds in eV working energy")),
        TableRow(("velocity AU to eVÅ factor", _fmt_probe_value(probe["velocity_au_to_evag_factor"]), "same physical velocity conversion")),
        TableRow(("velocity factor abs error", _fmt_probe_value(probe["velocity_factor_abs_error"]), "near 0")),
        TableRow(("Fermi window eV from AU factor", _fmt_probe_value(probe["fermi_window_ev_from_au_factor"]), "inverse-energy conversion")),
        TableRow(("mu conversion required", _fmt_probe_value(probe["mu_conversion_required"]), "True")),
        TableRow(("conductivity SI status", _fmt_probe_value(probe["conductivity_si_status"]), "pending")),
    )


def _finite_field_analytic_toy_rows(probe: dict[str, object]) -> tuple[TableRow, ...]:
    return (
        TableRow(("source", _fmt_probe_value(probe["source"]), "summary of current real probes")),
        TableRow(("toy count", _fmt_probe_value(probe["toy_count"]), "controlled analytic cases")),
        TableRow(("cosine symbol max error", _fmt_probe_value(probe["separable_cosine_symbol_max_error"]), "near 0")),
        TableRow(("cosine derivative max error", _fmt_probe_value(probe["separable_cosine_derivative_max_error"]), "near 0")),
        TableRow(("identity overlap min eig", _fmt_probe_value(probe["identity_overlap_min_eig"]), "> 1e-10")),
        TableRow(("identity overlap condition", _fmt_probe_value(probe["identity_overlap_condition"]), "finite")),
        TableRow(("periodic Dirac min gap", _fmt_probe_value(probe["periodic_dirac_min_gap"]), "controlled near-crossing")),
        TableRow(("periodic Dirac hazard count", _fmt_probe_value(probe["periodic_dirac_hazard_count"]), ">= 1")),
        TableRow(("HF velocity max error", _fmt_probe_value(probe["velocity_hf_max_error"]), "near 0")),
        TableRow(("unit velocity factor error", _fmt_probe_value(probe["unit_velocity_factor_error"]), "near 0")),
        TableRow(("all current toys pass", _fmt_probe_value(probe["all_current_toys_pass"]), "True")),
        TableRow(("missing toy", _fmt_probe_value(probe["missing_toy"]), "next analytic target")),
    )


def _finite_field_k_convergence_rows(probe: dict[str, object]) -> tuple[TableRow, ...]:
    return (
        TableRow(("source", _fmt_probe_value(probe["source"]), "controlled first implementation")),
        TableRow(("grid count", _fmt_probe_value(probe["grid_count"]), "number of refinements")),
        TableRow(("coarsest N", _fmt_probe_value(probe["coarsest_n"]), "first grid")),
        TableRow(("finest N", _fmt_probe_value(probe["finest_n"]), "last grid")),
        TableRow(("reference <|grad E|^2>", _fmt_probe_value(probe["reference_average_grad_e_sq"]), "analytic")),
        TableRow(("finest <|grad E|^2>", _fmt_probe_value(probe["finest_average_grad_e_sq"]), "numeric")),
        TableRow(("finest abs error", _fmt_probe_value(probe["finest_abs_error"]), "near 0")),
        TableRow(("max abs error", _fmt_probe_value(probe["max_abs_error"]), "near 0")),
        TableRow(("improved/equal steps", _fmt_probe_value(probe["improved_or_equal_steps"]), "non-regression count")),
        TableRow(("all grid errors small", _fmt_probe_value(probe["all_grid_errors_small"]), "True")),
        TableRow(("measure status", _fmt_probe_value(probe["measure_status"]), "normalisation check")),
        TableRow(("conductivity convergence status", _fmt_probe_value(probe["conductivity_convergence_status"]), "pending")),
    )


def _finite_field_symmetry_rows(probe: dict[str, object]) -> tuple[TableRow, ...]:
    return (
        TableRow(("source", _fmt_probe_value(probe["source"]), "controlled first implementation")),
        TableRow(("sample count", _fmt_probe_value(probe["sample_count"]), "N × N")),
        TableRow(("E(k)-E(-k) max error", _fmt_probe_value(probe["energy_inversion_max_error"]), "near 0")),
        TableRow(("dk1 oddness max error", _fmt_probe_value(probe["dk1_odd_max_error"]), "near 0")),
        TableRow(("dk2 oddness max error", _fmt_probe_value(probe["dk2_odd_max_error"]), "near 0")),
        TableRow(("tensor xx", _fmt_probe_value(probe["tensor_xx"]), "positive")),
        TableRow(("tensor yy", _fmt_probe_value(probe["tensor_yy"]), "positive")),
        TableRow(("tensor xy", _fmt_probe_value(probe["tensor_xy"]), "near 0")),
        TableRow(("tensor yx", _fmt_probe_value(probe["tensor_yx"]), "near 0")),
        TableRow(("tensor antisym abs", _fmt_probe_value(probe["tensor_antisym_abs"]), "near 0")),
        TableRow(("all symmetry checks pass", _fmt_probe_value(probe["all_symmetry_checks_pass"]), "True")),
        TableRow(("dataset automorphism status", _fmt_probe_value(probe["dataset_automorphism_status"]), "pending")),
    )


def _finite_field_vincent_rows(probe: dict[str, object]) -> tuple[TableRow, ...]:
    return (
        TableRow(("source", _fmt_probe_value(probe["source"]), "reused comparison domain")),
        TableRow(("Vincent target trace", _fmt_probe_value(probe["target_trace_S_per_m"]), "S/m")),
        TableRow(("weak-chain trace", _fmt_probe_value(probe["weak_chain_trace_S_per_m"]), "S/m")),
        TableRow(("weak-chain trace error %", _fmt_probe_value(probe["weak_chain_trace_percent_error"]), "residual")),
        TableRow(("strong-grid trace", _fmt_probe_value(probe["strong_grid_trace_S_per_m"]), "S/m")),
        TableRow(("strong-grid trace error %", _fmt_probe_value(probe["strong_grid_trace_percent_error"]), "separate spectral derivative residual")),
        TableRow(("shifted Eq. 8.30 trace", _fmt_probe_value(probe["shifted_830_trace_S_per_m"]), "S/m")),
        TableRow(("shifted Eq. 8.30 trace error %", _fmt_probe_value(probe["shifted_830_trace_percent_error"]), "hypothesis check")),
        TableRow(("find-simplex max velocity error", _fmt_probe_value(probe["find_simplex_max_velocity_error_m_per_s"]), "m/s")),
        TableRow(("best-adjacent max velocity error", _fmt_probe_value(probe["best_adjacent_max_velocity_error_m_per_s"]), "m/s")),
        TableRow(("velocity error reduction", _fmt_probe_value(probe["velocity_error_reduction"]), "large")),
        TableRow(("best adjacent matches Vincent", _fmt_probe_value(probe["best_adjacent_matches_vincent"]), "True")),
        TableRow(("residual status", _fmt_probe_value(probe["residual_status"]), "audit note")),
    )


def _finite_field_strong_dc_rows(probe: dict[str, object]) -> tuple[TableRow, ...]:
    return (
        TableRow(("source", _fmt_probe_value(probe["source"]), "reused strong DC implementation")),
        TableRow(("mode count", _fmt_probe_value(probe["mode_count"]), "FFT modes")),
        TableRow(("nonzero mode count", _fmt_probe_value(probe["nonzero_mode_count"]), "active modes")),
        TableRow(("strong-grid trace", _fmt_probe_value(probe["strong_grid_trace_S_per_m"]), "S/m")),
        TableRow(("weak-chain grid trace", _fmt_probe_value(probe["weak_chain_grid_trace_S_per_m"]), "S/m")),
        TableRow(("Vincent target trace", _fmt_probe_value(probe["vincent_target_trace_S_per_m"]), "S/m")),
        TableRow(("strong/weak trace gap", _fmt_probe_value(probe["strong_vs_weak_rel_trace_gap"]), "known derivative residual")),
        TableRow(("strong/Vincent trace error %", _fmt_probe_value(probe["strong_vs_vincent_percent_error"]), "audit residual")),
        TableRow(("mode reconstruction abs error", _fmt_probe_value(probe["mode_reconstruction_abs_error"]), "near 0")),
        TableRow(("imaginary leakage", _fmt_probe_value(probe["imaginary_leakage_S"]), "near 0")),
        TableRow(("imaginary leakage ratio", _fmt_probe_value(probe["imaginary_leakage_ratio"]), "near 0")),
        TableRow(("strongest mode fraction", _fmt_probe_value(probe["strongest_mode_fraction"]), "finite")),
        TableRow(("occupation coeff shape", f"{probe['occupation_coeff_shape_0']} × {probe['occupation_coeff_shape_1']}", "grid shape")),
        TableRow(("response factor finite", _fmt_probe_value(probe["response_factor_finite"]), "True")),
        TableRow(("velocity coefficients finite", _fmt_probe_value(probe["velocity_coefficients_finite"]), "True")),
        TableRow(("strong DC internal pass", _fmt_probe_value(probe["strong_dc_internal_pass"]), "True")),
        TableRow(("residual status", _fmt_probe_value(probe["residual_status"]), "audit note")),
    )


def _finite_field_weak_dc_rows(probe: dict[str, object]) -> tuple[TableRow, ...]:
    return (
        TableRow(("source", _fmt_probe_value(probe["source"]), "matched spectral-basis analytic toy")),
        TableRow(("field row count", _fmt_probe_value(probe["field_row_count"]), "eta sweep rows")),
        TableRow(("zero eta", _fmt_probe_value(probe["zero_eta"]), "0")),
        TableRow(("zero field", _fmt_probe_value(probe["zero_field_V_per_m"]), "V/m")),
        TableRow(("zero relative tensor discrepancy", _fmt_probe_value(probe["zero_relative_tensor_discrepancy"]), "near 0")),
        TableRow(("zero relative trace discrepancy", _fmt_probe_value(probe["zero_relative_trace_discrepancy"]), "near 0")),
        TableRow(("small eta", _fmt_probe_value(probe["small_eta"]), "first nonzero field")),
        TableRow(("small field", _fmt_probe_value(probe["small_field_V_per_m"]), "V/m")),
        TableRow(("small relative tensor discrepancy", _fmt_probe_value(probe["small_relative_tensor_discrepancy"]), "small")),
        TableRow(("small relative trace discrepancy", _fmt_probe_value(probe["small_relative_trace_discrepancy"]), "small")),
        TableRow(("largest eta", _fmt_probe_value(probe["largest_eta"]), "largest sweep field")),
        TableRow(("largest relative tensor discrepancy", _fmt_probe_value(probe["largest_relative_tensor_discrepancy"]), "nonlinear departure")),
        TableRow(("largest relative trace discrepancy", _fmt_probe_value(probe["largest_relative_trace_discrepancy"]), "nonlinear departure")),
        TableRow(("relative weak-limit error", _fmt_probe_value(probe["relative_weak_limit_error"]), "near 0")),
        TableRow(("strong zero-field imaginary leakage", _fmt_probe_value(probe["strong_zero_field_imaginary_leakage"]), "near 0")),
        TableRow(("max imaginary leakage", _fmt_probe_value(probe["max_imaginary_leakage"]), "finite")),
        TableRow(("weak-limit pass", _fmt_probe_value(probe["weak_limit_pass"]), "True")),
        TableRow(("roundoff floor status", _fmt_probe_value(probe["roundoff_floor_status"]), "audit note")),
    )


def _finite_field_mode_decomposition_rows(probe: dict[str, object]) -> tuple[TableRow, ...]:
    return (
        TableRow(("source", _fmt_probe_value(probe["source"]), "strong DC mode result")),
        TableRow(("mode count", _fmt_probe_value(probe["mode_count"]), "FFT modes")),
        TableRow(("Gamma reconstruction abs error", _fmt_probe_value(probe["gamma_reconstruction_abs_error"]), "near 0")),
        TableRow(("rho reconstruction abs error", _fmt_probe_value(probe["rho_reconstruction_abs_error"]), "near 0")),
        TableRow(("mode tensor reconstruction abs error", _fmt_probe_value(probe["mode_tensor_reconstruction_abs_error"]), "near 0")),
        TableRow(("conductivity trace", _fmt_probe_value(probe["conductivity_trace_S_per_m"]), "S/m")),
        TableRow(("mode norm sum", _fmt_probe_value(probe["conductivity_mode_norm_sum"]), "finite")),
        TableRow(("top 1 mode fraction", _fmt_probe_value(probe["top_1_mode_fraction"]), "concentration")),
        TableRow(("top 10 mode fraction", _fmt_probe_value(probe["top_10_mode_fraction"]), "concentration")),
        TableRow(("top 100 mode fraction", _fmt_probe_value(probe["top_100_mode_fraction"]), "concentration")),
        TableRow(("Gamma abs max", _fmt_probe_value(probe["gamma_abs_max"]), "finite")),
        TableRow(("rho abs max", _fmt_probe_value(probe["rho_abs_max"]), "finite")),
        TableRow(("response abs max", _fmt_probe_value(probe["response_abs_max"]), "finite")),
        TableRow(("Gamma finite", _fmt_probe_value(probe["gamma_finite"]), "True")),
        TableRow(("rho finite", _fmt_probe_value(probe["rho_finite"]), "True")),
        TableRow(("response finite", _fmt_probe_value(probe["response_finite"]), "True")),
        TableRow(("mode tensor finite", _fmt_probe_value(probe["mode_tensor_finite"]), "True")),
        TableRow(("mode closure pass", _fmt_probe_value(probe["mode_closure_pass"]), "True")),
        TableRow(("residual status", _fmt_probe_value(probe["residual_status"]), "audit note")),
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
            0.0,
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
            "symmetrization",
            "Symmetrization",
            "select",
            "star",
            options=(("star", "star"), ("direct", "direct"), ("raw", "raw")),
            help="Symmetrization scheme used to build the symbols.",
        ),
    )


def _finite_field_dc_input_rows(inputs) -> tuple[TableRow, ...]:
    return (
        TableRow(("dataset", str(inputs.get("dataset", "default")))),
        TableRow(("temperature T", DisplayQuantity(float(inputs.get("temperature", 300.0)), TEMPERATURE, KELVIN, name="temperature"))),
        TableRow(("chemical potential mu", DisplayQuantity(float(inputs.get("mu", 0.0)), ENERGY, ELECTRON_VOLT, name="mu"))),
        TableRow(("relaxation time tau", DisplayQuantity(float(inputs.get("tau", 1.0)), TIME, FEMTOSECOND, name="tau"))),
        TableRow(("units", str(inputs.get("units", "eVAng")))),
        TableRow(("N_u", int(inputs.get("n_u", 11)))),
        TableRow(("N_v", int(inputs.get("n_v", 11)))),
        TableRow(("electric field E", DisplayQuantity(float(inputs.get("electric_field", 1.0)), ENERGY / (CHARGE * LENGTH), VOLT_PER_METER, name="electric field"))),
        TableRow(("field angle theta", DisplayQuantity(float(inputs.get("theta", 0.0)), DIMENSIONLESS, UNITLESS, name="theta"))),
        TableRow(("band index n", int(inputs.get("band_index", 0)))),
        TableRow(("symmetrization scheme", str(inputs.get("symmetrization", "star")))),
        TableRow(("reciprocal 2π normalization", "dummy; must be physically audited, not treated as cosmetic convention")),
        TableRow(("conductivity normalization", "dummy")),
    )


def _finite_field_dc_section(
    *,
    section_id: str,
    title: str,
    description: str,
    claim: str,
    evidence: tuple[tuple[str, str, str], ...],
    placeholders: tuple[str, ...],
    collapsed: bool = False,
) -> DiagnosticSection:
    return DiagnosticSection(
        id=section_id,
        title=title,
        description=description,
        collapsed=collapsed,
        body=(
            MarkdownBlock(
                id=f"{section_id}_prose",
                title="Validation claim",
                markdown=f"""**Claim.** {claim}

This section is currently a scaffold. The tables below name the evidence that must be filled in by later diagnostics.
""",
            ),
            Table(
                id=f"{section_id}_evidence_plan",
                title="Evidence plan",
                description="Checks, plots, or tables that will turn this section from prose into validation evidence.",
                headers=("evidence", "status", "purpose"),
                rows=tuple(TableRow(row) for row in evidence),
            ),
            Table(
                id=f"{section_id}_placeholders",
                title="Diagnostic placeholders",
                description="Concrete diagnostic blocks to replace with computed outputs.",
                headers=("placeholder", "status"),
                rows=tuple(TableRow((item, "dummy")) for item in placeholders),
            ),
        ),
    )


def compute_finite_field_dc_validation(ctx, inputs) -> DiagnosticResult:
    """Scaffold for validating finite-field, band-labelled DC conductivity."""

    dashboard_rows = (
        TableRow(("input health", "dummy", "starred H/S symbols define stable generalized eigenproblems")),
        TableRow(("band-crossing hazards", "dummy", "near crossings and label jumps are mapped in k-space")),
        TableRow(("velocity validation", "dummy", "velocity agrees with analytic, finite-difference, unit, Gamma, and Vincent checks")),
        TableRow(("Vincent reconstruction", "dummy", "2π normalization and residual few-percent gap are isolated")),
        TableRow(("strong DC contact", "dummy", "strong band-labelled result agrees with Vincent/Ashcroft form in shared regime")),
        TableRow(("weak DC limit", "dummy", "finite-field result approaches weak-field result as E -> 0")),
        TableRow(("mode closure", "dummy", "Gamma, F, and tilde(rho) reconstruct total conductivity")),
        TableRow(("analytic toys", "dummy", "periodic known-input tests give known outputs")),
        TableRow(("unit consistency", "dummy", "same physical calculation agrees after SI conversion")),
        TableRow(("k convergence", "dummy", "conductivity stabilizes under N_u, N_v refinement")),
        TableRow(("symmetry sanity", "dummy", "tensor and direction sweeps obey expected symmetries")),
    )

    input_rows = _finite_field_dc_input_rows(inputs)
    input_health_probe = finite_field_input_health_probe(
        n_u=int(inputs.get("n_u", 11)),
        n_v=int(inputs.get("n_v", 11)),
        symmetrization=str(inputs.get("symmetrization", "star")),
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
        summary="Scaffold for validating finite-field, band-labelled DC conductivity and its lattice-mode decomposition.",
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
                        id="finite_field_dc_validation_dashboard",
                        title="Validation dashboard",
                        description="Dummy pass/warn/fail summary. Later sections should feed this table.",
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
                        description="Dummy values until the real diagnostic inputs are wired in.",
                        headers=("input", "value"),
                        rows=input_rows,
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_input_health",
                title="Input health",
                description="Algebraic and sampling checks for starred H and S symbols.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_input_health_prose",
                        title="Validation claim",
                        markdown="""**Claim.** The starred Hamiltonian and overlap symbols define stable Hermitian generalized eigenproblems over the sampled k-domain.

This first implementation uses controlled production `GdKernelArrays` toy kernels. It validates the same symbol path as the real calculation without loading the full dataset. The raw overlap symbol is checked as Hermitian positive definite, not unitary.
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
                        title="Remaining placeholders",
                        description="Dataset-backed diagnostics still to replace the toy-backed first implementation.",
                        headers=("placeholder", "status"),
                        rows=(
                            TableRow(("dataset-backed H/S health table", "pending")),
                            TableRow(("symbol smoothness plot", "pending")),
                            TableRow(("condition-number k-map", "pending")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_band_crossing_hazards",
                title="Band-crossing hazards",
                description="Maps k-space regions where energy-ordered band labels become fragile.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_band_crossing_hazards_prose",
                        title="Validation claim",
                        markdown="""**Claim.** For the selected energy-ordered band, the diagnostic identifies k-space regions where near crossings or eigenvector jumps can contaminate velocity and conductivity.

This first implementation uses a periodic two-level Dirac-like toy model. It is not a real graphene band map yet. It exists to make the hazard logic concrete: compute adjacent-band gaps, locate the minimum gap, count points below a threshold, and report neighbour-jump smoothness proxies.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_band_crossing_hazards_table",
                        title="Band-crossing hazard metrics",
                        description="First real hazard table for energy-ordered band labels.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_band_hazard_rows(band_hazard_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_band_crossing_hazards_placeholders",
                        title="Remaining placeholders",
                        description="Dataset-backed diagnostics still to replace the toy-backed first implementation.",
                        headers=("placeholder", "status"),
                        rows=(
                            TableRow(("real band minimum-gap k-map", "pending")),
                            TableRow(("same-label S-overlap continuation map", "pending")),
                            TableRow(("velocity anomaly overlay", "pending")),
                            TableRow(("active near-degenerate subspace comparison", "pending")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_velocity_validation",
                title="Velocity validation",
                description="Checks the band velocity used by conductivity before conductivity is tested.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_velocity_validation_prose",
                        title="Validation claim",
                        markdown="""**Claim.** Band velocities agree across analytic derivatives, finite differences, generalized Hellmann-Feynman derivatives, and fixed/generic symbol conventions.

This first implementation uses the separable cosine production-symbol toy. It validates the derivative machinery that finite-field conductivity will reuse. Modal Gamma reconstruction, Vincent velocity comparison, and physical unit scaling remain explicit pending rows rather than hidden assumptions.
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
                        title="Remaining placeholders",
                        description="Dataset-backed and Vincent-backed checks still to replace the toy-backed first implementation.",
                        headers=("placeholder", "status"),
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
                description="Reconstructs Vincent's reference calculation and isolates convention differences.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_vincent_reconstruction_prose",
                        title="Validation claim",
                        markdown="""**Claim.** The implementation reconstructs Vincent's reference calculation when the same dispersion, units, interpolation convention, k-grid measure, temperature, chemical potential, and relaxation time are used.

This first finite-field validation section reuses the existing Ashcroft/Vincent comparison domain. It exposes the current reconstruction status directly inside the finite-field ladder: velocity samples are resolved by adjacent-simplex Delaunay ambiguity, while the conductivity residual remains a formula/convention audit item.
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
                        title="Remaining placeholders",
                        description="Reconstruction checks still to wire directly into this finite-field validation domain.",
                        headers=("placeholder", "status"),
                        rows=(
                            TableRow(("component-level Vincent tensor table", "pending")),
                            TableRow(("2π/grid-measure audit rows", "pending")),
                            TableRow(("chemical-potential convention sweep", "pending")),
                            TableRow(("direct finite-field reproduction against Vincent inputs", "pending")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_strong_dc_validation",
                title="Strong DC validation",
                description="Checks the finite-field band-labelled strong DC tensor in regimes where independent formulae should meet.",
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
                        title="Remaining placeholders",
                        description="Checks still needed for the full finite-field validation.",
                        headers=("placeholder", "status"),
                        rows=(
                            TableRow(("component-level strong tensor table", "pending")),
                            TableRow(("temperature / smoothness regime table", "pending")),
                            TableRow(("dataset-backed band-labelled strong DC run", "pending")),
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_field_dc_validation_weak_dc_limit",
                title="Weak DC limit",
                description="Small-field limit of finite-field DC conductivity.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_weak_dc_limit_prose",
                        title="Validation claim",
                        markdown="""**Claim.** The finite-field band-labelled DC conductivity approaches the weak-field DC result as E -> 0 when both calculations use the same spectral derivative basis.

This first implementation reuses the analytic sinusoidal Ashcroft probe. It separates the clean matched-basis weak limit from the Vincent-grid derivative-definition residual exposed in the strong DC section.
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
                        title="Remaining placeholders",
                        description="Checks still needed for the full finite-field validation.",
                        headers=("placeholder", "status"),
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
                description="Closure checks for Gamma, F, and tilde(rho).",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_mode_decomposition_prose",
                        title="Validation claim",
                        markdown="""**Claim.** The lattice-mode decomposition into Gamma, F, and tilde(rho) reconstructs the total finite-field band-labelled DC conductivity tensor.

This first implementation checks the actual `BandIndexedStrongDcResult` mode objects: Gamma reconstructs the sampled velocity field, tilde(rho) reconstructs the sampled occupation, and summing the conductivity mode tensor reconstructs the total strong DC tensor.
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
                        title="Remaining placeholders",
                        description="Mode visualisation and dataset-backed closure checks still required.",
                        headers=("placeholder", "status"),
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
                        title="Remaining placeholders",
                        description="Analytic toy coverage still needed for the full finite-field validation.",
                        headers=("placeholder", "status"),
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
                description="Physical scaling and SI conversion checks.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_unit_scaling_prose",
                        title="Validation claim",
                        markdown="""**Claim.** The same physical calculation gives the same SI result after conversion from different internal unit systems.

This first implementation checks the core unit factors that all later finite-field comparisons depend on: Hartree to eV, Bohr to Å, hbar in each working context, velocity scaling, inverse-energy Fermi-window scaling, and the requirement that mu be converted with the Hamiltonian.
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
                        title="Remaining placeholders",
                        description="Full calculation-level unit checks still to replace this lightweight first implementation.",
                        headers=("placeholder", "status"),
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
                description="Refinement in N_u and N_v only.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_k_convergence_prose",
                        title="Validation claim",
                        markdown="""**Claim.** The reported conductivity is stable under refinement of N_u and N_v.

This first implementation checks the grid-measure part of that claim on a periodic analytic velocity-square average. It is not yet a dataset-backed conductivity convergence table, but it does exercise the normalisation convention before the stronger conductivity comparison is wired in.
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
                        title="Remaining placeholders",
                        description="Dataset-backed convergence still required for the full finite-field validation.",
                        headers=("placeholder", "status"),
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
                description="Tensor, k-space, and direction-sweep symmetry checks.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_symmetry_prose",
                        title="Validation claim",
                        markdown="""**Claim.** Tensor components and direction sweeps obey expected lattice and time-reversal symmetries up to finite-size and sampling defects.

This first implementation checks the symmetry algebra on a controlled separable cosine toy: even energy under k inversion, odd derivative under k inversion, and a symmetric diagonal velocity-square tensor with vanishing cross component.
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
                        title="Remaining placeholders",
                        description="Dataset-backed symmetry checks still required for the full finite-field validation.",
                        headers=("placeholder", "status"),
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
