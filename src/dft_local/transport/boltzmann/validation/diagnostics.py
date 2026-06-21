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
    TypstMathBlock,
)
from dft_local.diagnostics.user_strings import TypstMath
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
    finite_field_dataset_band_crossing_hazard_probe,
    FiniteFieldDatasetBandCrossingHazardProbe,
    FiniteFieldDatasetBandHazardPoint,
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
    finite_field_strong_eq830_limit_probe,
    FiniteFieldStrongEq830LimitProbe,
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


def _finite_field_dataset_band_hazard_rows(probe: FiniteFieldDatasetBandCrossingHazardProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "selected dataset/model source")),
        TableRow(("sample count", probe.sample_count, "N_u × N_v")),
        TableRow(("band count", probe.band_count, "number of sorted energy labels")),
        TableRow(("selected band", probe.selected_band, "band whose neighbouring crossings are relevant")),
        TableRow(("gap threshold", q("gap_threshold"), "hazard if selected-band adjacent gap is below this")),
        TableRow(("minimum adjacent gap", q("min_gap"), "larger is safer for energy-ordered labels")),
        TableRow(("selected gap q05", q("selected_gap_q05"), "5th percentile selected adjacent gap")),
        TableRow(("selected gap median", q("selected_gap_median"), "typical selected adjacent gap")),
        TableRow(("selected gap q95", q("selected_gap_q95"), "95th percentile selected adjacent gap")),
        TableRow(("selected gap max", q("selected_gap_max"), "largest selected adjacent gap")),
        TableRow(("min gap / threshold", q("min_gap_over_threshold"), "below 1 is hazardous")),
        TableRow(("median gap / threshold", q("median_gap_over_threshold"), "below 1 means typical selected gap is hazardous")),
        TableRow(("minimum-gap k1", q("min_gap_k1"), "location")),
        TableRow(("minimum-gap k2", q("min_gap_k2"), "location")),
        TableRow(("minimum-gap lower band", probe.min_gap_lower_band, "lower sorted label")),
        TableRow(("minimum-gap upper band", probe.min_gap_upper_band, "upper sorted label")),
        TableRow(("hazard count", probe.hazard_count, "number of adjacent gaps below threshold")),
        TableRow(("hazard fraction", q("hazard_fraction"), "hazard count / sample count")),
        TableRow(("has hazard", probe.has_hazard, "diagnostic flag")),
        TableRow(("max band neighbour jump", q("max_band_neighbour_jump"), "energy-label smoothness proxy")),
        TableRow(("max gap neighbour jump", q("max_gap_neighbour_jump"), "gap smoothness proxy")),
    )


def _finite_field_dataset_band_hazard_point_rows(
    points: tuple[FiniteFieldDatasetBandHazardPoint, ...],
) -> tuple[TableRow, ...]:
    rows: list[TableRow] = []
    for point in points:
        q = lambda field, point=point: diagnostic_scalar_quantity(point, field)
        rows.append(
            TableRow(
                (
                    q("k1"),
                    q("k2"),
                    point.lower_band,
                    point.upper_band,
                    q("lower_energy"),
                    q("upper_energy"),
                    q("gap"),
                    q("threshold"),
                )
            )
        )
    if not rows:
        return (TableRow(("none", "none", "none", "none", "none", "none", "none", "none")),)
    return tuple(rows)


def _finite_field_analytic_velocity_rows(probe: FiniteFieldVelocityValidationProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "controlled production-symbol toy")),
        TableRow(("k1", q("k1"), "sample point")),
        TableRow(("k2", q("k2"), "sample point")),
        TableRow(("finite-difference epsilon", q("finite_difference_eps"), "step")),
        TableRow(("analytic dE/dk1", q("analytic_dk1"), "reference")),
        TableRow(("finite-difference dE/dk1", q("finite_difference_dk1"), "central finite difference")),
        TableRow(("finite-difference dk1 error", q("finite_difference_dk1_abs_error"), "near finite-difference precision")),
        TableRow(("analytic dE/dk2", q("analytic_dk2"), "reference")),
        TableRow(("finite-difference dE/dk2", q("finite_difference_dk2"), "central finite difference")),
        TableRow(("finite-difference dk2 error", q("finite_difference_dk2_abs_error"), "near finite-difference precision")),
        TableRow(("production derivative dk1 error", q("production_dk1_abs_error"), "near 0")),
        TableRow(("production derivative dk2 error", q("production_dk2_abs_error"), "near 0")),
        TableRow(("Hellmann-Feynman dk1 error", q("hellmann_feynman_dk1_abs_error"), "near 0")),
        TableRow(("Hellmann-Feynman dk2 error", q("hellmann_feynman_dk2_abs_error"), "near 0")),
        TableRow(("generic/fixed symbol error", q("generic_fixed_symbol_abs_error"), "near 0")),
        TableRow(("generic/fixed dk1 error", q("generic_fixed_dk1_abs_error"), "near 0")),
        TableRow(("generic/fixed dk2 error", q("generic_fixed_dk2_abs_error"), "near 0")),
        TableRow(("unit scaling status", probe.unit_scaling_status, "separate section")),
    )


def _finite_field_vincent_velocity_rows(probe: FiniteFieldVelocityValidationProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("quoted point count", probe.vincent_sample_count, "Vincent quoted samples")),
        TableRow(("find-simplex max velocity error", q("vincent_find_simplex_max_velocity_error"), "bad vertex-simplex choice")),
        TableRow(("best-adjacent max velocity error", q("vincent_best_adjacent_max_velocity_error"), "roundoff")),
        TableRow(("velocity error reduction", q("vincent_velocity_error_reduction"), "large")),
        TableRow(("status", probe.vincent_velocity_status, "quoted samples")),
    )


def _finite_field_dataset_gamma_velocity_rows(probe: FiniteFieldVelocityValidationProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)
    rel_change = float(probe.dataset_velocity_mean_square_rel_change)
    rel_change_percent = "pending" if rel_change != rel_change else f"{100.0 * rel_change:.6g}%"

    return (
        TableRow(("grid", f"{probe.dataset_gamma_n_u} × {probe.dataset_gamma_n_v}", "selected validation grid")),
        TableRow(("band index", probe.dataset_gamma_band_index, "energy-ordered selected band")),
        TableRow(("same-grid Gamma abs error", q("dataset_gamma_same_grid_abs_error"), "near FFT roundoff when dataset is available")),
        TableRow(("same-grid Gamma rel L2 error", q("dataset_gamma_same_grid_rel_l2_error"), "near FFT roundoff when dataset is available")),
        TableRow(("coarse grid", f"{probe.dataset_gamma_coarse_n_u} × {probe.dataset_gamma_coarse_n_v}", "coarse comparison grid")),
        TableRow(("velocity mean-square rel change", rel_change_percent, "coarse-to-selected k-grid stability proxy")),
        TableRow(("hazard threshold", q("dataset_gamma_gap_threshold"), "same selected-band threshold as hazard section")),
        TableRow(("selected-band hazard count", probe.dataset_gamma_hazard_count, "same count as band-crossing hazard section")),
        TableRow(("selected-band hazard fraction", q("dataset_gamma_hazard_fraction"), "same fraction as band-crossing hazard section")),
        TableRow(("status", probe.dataset_gamma_status, "dataset-backed closure")),
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


def _finite_field_vincent_normalisation_rows(probe: FiniteFieldVincentReconstructionProbe) -> tuple[TableRow, ...]:
    two_pi_squared = (2.0 * 3.141592653589793) ** 2

    weak_no_denominator_trace = getattr(
        probe,
        "weak_trace",
        probe.continuum_weak_trace * two_pi_squared,
    )
    weak_target_trace = probe.continuum_weak_trace / (
        1.0 + probe.continuum_weak_trace_percent_error / 100.0
    )
    weak_no_denominator_percent_error = getattr(
        probe,
        "weak_trace_percent_error",
        100.0 * (weak_no_denominator_trace - weak_target_trace) / weak_target_trace,
    )

    eq830_no_denominator_trace = getattr(
        probe,
        "eq830_shifted_trace",
        probe.continuum_eq830_shifted_trace * two_pi_squared,
    )
    eq830_target_trace = probe.continuum_eq830_shifted_trace / (
        1.0 + probe.continuum_eq830_shifted_trace_percent_error / 100.0
    )
    eq830_no_denominator_percent_error = getattr(
        probe,
        "eq830_shifted_trace_percent_error",
        100.0 * (eq830_no_denominator_trace - eq830_target_trace) / eq830_target_trace,
    )

    return (
        TableRow((
            "reciprocal convention",
            "a_i dot b_j = 2π δ_ij",
            f"diag error {probe.reciprocal_dot_diag_max_abs_error:.3e}; offdiag {probe.reciprocal_dot_offdiag_max_abs:.3e}",
        )),
        TableRow((
            "reciprocal area relation",
            "det(B) = (2π)^2 / det(A)",
            f"ratio {probe.reciprocal_det_ratio:.15g}; error {probe.reciprocal_det_ratio_abs_error:.3e}",
        )),
        TableRow((
            "weak raw continuum trace",
            probe.continuum_weak_trace,
            f"{probe.continuum_weak_trace_percent_error:.12g}% vs Vincent",
        )),
        TableRow((
            "weak no-2π-denominator trace",
            weak_no_denominator_trace,
            f"{weak_no_denominator_percent_error:.12g}% vs Vincent",
        )),
        TableRow((
            "Eq. 8.30 raw continuum trace",
            probe.continuum_eq830_shifted_trace,
            f"{probe.continuum_eq830_shifted_trace_percent_error:.12g}% vs Vincent",
        )),
        TableRow((
            "Eq. 8.30 no-2π-denominator trace",
            eq830_no_denominator_trace,
            f"{eq830_no_denominator_percent_error:.12g}% vs Vincent",
        )),
        TableRow((
            "normalisation conclusion",
            "Vincent target is reproduced only by A_BZ/Nk, not by the continuum A_BZ/(Nk (2π)^2) measure",
            "physical SI normalisation remains an explicit audit item",
        )),
    )


def _finite_field_vincent_rows(probe: FiniteFieldVincentReconstructionProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("comparison source", probe.source, "external audit anchor")),
        TableRow(("Vincent target", q("target_trace"), "reference")),
        TableRow(("Eq. 8.30 finite-difference trace", q("shifted_830_trace"), q("shifted_830_trace_percent_error"))),
        TableRow(("weak chain-rule limit", q("weak_chain_trace"), q("weak_chain_trace_percent_error"))),
        TableRow(("Eq. 8.30 Gamma-Q-rho trace", q("eq830_modal_trace"), q("eq830_modal_trace_percent_error"))),
        TableRow(("Eq. 8.30 modal closure residual", q("eq830_modal_direct_trace_percent_error"), "modal trace minus direct trace; near 0")),
        TableRow(("strong spectral zero-field trace", q("strong_grid_trace"), q("strong_grid_trace_percent_error"))),
        TableRow(("conductivity residual", probe.residual_status, "open")),
    )


def _finite_field_strong_dc_rows(probe: FiniteFieldStrongDcValidationProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "reused strong DC implementation")),
        TableRow(("mode count", probe.mode_count, "FFT modes")),
        TableRow(("nonzero mode count", probe.nonzero_mode_count, "active modes")),
        TableRow(("strong continuum trace", q("continuum_strong_trace"), "default physical-continuum path")),
        TableRow(("weak continuum trace", q("continuum_weak_trace"), "default physical-continuum path")),
        TableRow(("strong no-2π-denominator trace", q("no_2pi_denominator_strong_trace"), "Vincent-comparison scale only")),
        TableRow(("weak no-2π-denominator trace", q("no_2pi_denominator_weak_trace"), "Vincent-comparison scale only")),
        TableRow(("Vincent target trace", q("vincent_target_trace"), "external target; not default normalisation")),
        TableRow(("strong/weak trace gap", q("strong_vs_weak_rel_trace_gap"), "scale-invariant derivative residual")),
        TableRow(("strong/Vincent trace error %", q("strong_vs_vincent_percent_error"), "computed on no-2π comparison scale")),
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


def _finite_field_strong_eq830_limit_rows(probe: FiniteFieldStrongEq830LimitProbe) -> tuple[TableRow, ...]:
    q = lambda field: diagnostic_scalar_quantity(probe, field)

    return (
        TableRow(("source", probe.source, "direct comparison target")),
        TableRow(("field row count", probe.field_row_count, "field sweep rows")),
        TableRow(("zero field", q("zero_field"), "V/m")),
        TableRow(("smallest nonzero field", q("smallest_nonzero_field"), "V/m")),
        TableRow(("largest field", q("largest_field"), "V/m")),
        TableRow(("strong continuum trace", q("strong_continuum_trace"), "differential response")),
        TableRow(("zero-field Eq. 8.30 continuum trace", q("zero_eq830_continuum_trace"), "finite-difference response")),
        TableRow(("small-field Eq. 8.30 continuum trace", q("smallest_eq830_continuum_trace"), "finite-difference response")),
        TableRow(("zero strong/Eq. 8.30 tensor discrepancy", q("zero_relative_tensor_discrepancy"), "exposed residual")),
        TableRow(("zero strong/Eq. 8.30 trace discrepancy", q("zero_relative_trace_discrepancy"), "exposed residual")),
        TableRow(("small strong/Eq. 8.30 tensor discrepancy", q("smallest_relative_tensor_discrepancy"), "small-field residual")),
        TableRow(("small strong/Eq. 8.30 trace discrepancy", q("smallest_relative_trace_discrepancy"), "small-field residual")),
        TableRow(("largest strong/Eq. 8.30 tensor discrepancy", q("largest_relative_tensor_discrepancy"), "finite-field departure")),
        TableRow(("largest strong/Eq. 8.30 trace discrepancy", q("largest_relative_trace_discrepancy"), "finite-field departure")),
        TableRow(("minimum tensor discrepancy over sweep", q("min_relative_tensor_discrepancy"), "best sampled agreement")),
        TableRow(("minimum abs trace discrepancy over sweep", q("min_abs_relative_trace_discrepancy"), "best sampled agreement")),
        TableRow(("Eq. 8.30 limit status", probe.eq830_limit_status, "audit note")),
        TableRow(("normalisation status", probe.continuum_normalisation_status, "guardrail")),
        TableRow(("limit validation pass", probe.limit_validation_pass, "finite residuals exposed")),
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
        InputSpec(
            "gap_threshold",
            "Band-gap hazard threshold",
            "float",
            0.05,
            min_value=0.0,
            help="Adjacent sorted-energy gap below which a sampled k-point is marked as a band-label hazard.",
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
        TableRow(("band-gap hazard threshold", _display_quantity(inputs.get("gap_threshold", 0.05), name="band-gap hazard threshold"))),
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
        TableRow(("band-crossing hazards", "partial", "dataset-backed adjacent-gap hazards are mapped; velocity overlays and label-overlap maps remain pending")),
        TableRow(("velocity validation", "partial", "analytic finite-difference, Vincent quoted-velocity, and dataset Gamma closure checks are live; velocity maps remain pending")),
        TableRow(("Vincent reconstruction", "open-audit", "velocity samples are resolved; 2π normalization and residual few-percent conductivity gap remain visible audit items")),
        TableRow(("strong DC contact", "partial", "strong band-labelled result is checked on Vincent-grid inputs; shared-regime comparison still being tightened")),
        TableRow(("weak DC limit", "toy-only", "finite-field result approaches weak-field result as E -> 0 in matched spectral-basis toy")),
        TableRow(("mode closure", "partial", "Gamma, F, and tilde(rho) reconstruct the strong-grid conductivity object; dataset-backed closure remains pending")),
        TableRow(("analytic toys", "partial", "periodic known-input tests are live; finite-field Gamma/Q/rho toy remains pending")),
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
    if KH_input is None or KS_input is None:
        dataset_band_hazard_probe = None
    else:
        dataset_band_hazard_probe = finite_field_dataset_band_crossing_hazard_probe(
            KH_input,
            KS_input,
            n_u=int(inputs.get("n_u", 11)),
            n_v=int(inputs.get("n_v", 11)),
            symmetrization=symmetrization,
            gap_threshold=float(inputs.get("gap_threshold", 0.05)),
            band_index=int(inputs.get("band_index", 0)),
            source=input_health_source,
        )

    band_hazard_probe = finite_field_band_crossing_hazard_probe(
        n_u=int(inputs.get("n_u", 11)),
        n_v=int(inputs.get("n_v", 11)),
    )
    velocity_probe = finite_field_velocity_validation_probe(
        KH_input,
        KS_input,
        n_u=int(inputs.get("n_u", 11)),
        n_v=int(inputs.get("n_v", 11)),
        band_index=int(inputs.get("band_index", 0)),
        symmetrization=symmetrization,
        gap_threshold=float(inputs.get("gap_threshold", 0.05)),
        dataset_hazard_probe=dataset_band_hazard_probe,
    )
    unit_scaling_probe = finite_field_unit_scaling_probe()
    analytic_toy_probe = finite_field_analytic_toy_coverage_probe()
    k_convergence_probe = finite_field_k_convergence_probe()
    symmetry_probe = finite_field_symmetry_sanity_probe()
    vincent_probe = finite_field_vincent_reconstruction_probe()
    strong_dc_probe = finite_field_strong_dc_validation_probe()
    strong_eq830_limit_probe = finite_field_strong_eq830_limit_probe()
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
                        markdown="""*Claim.* For the selected symmetrization scheme, the Hamiltonian and overlap symbols define stable Hermitian generalized eigenproblems over the sampled k-domain.

This section uses the selected dataset-backed `GdKernelArrays` kernel family when diagnostic context is available, then applies the selected symmetrization scheme. `raw` reports the extracted kernels as-is, `star` applies kernel-level star symmetrization, and `direct` applies Hermitian projection after forming each local symbol. The overlap symbol is checked as Hermitian positive definite, not unitary.

The input-health table reports Hermiticity at two different stages. *H kernel star defect max* checks the local kernel before forming any k-space symbol, by comparing each inverse-paired block $K_H (g^(-1))$ with the adjoint of $K_H (g)$. A nonzero value means the extracted local kernel is not itself star/Hermitian compatible. *$H(k)$ Hermiticity defect rel max* checks the dense symbol after evaluating the kernel over the sampled k-grid. In `star` mode both should fall to roundoff; in `direct` mode only the formed symbol is repaired, so the kernel defect may remain.
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
                        markdown="""*Claim.* The diagnostic can identify k-space regions where near crossings make energy-ordered band labels fragile.

This section identifies k-points where the selected energy-ordered band label becomes fragile. At each sampled k-point, the selected dataset-backed H/S symbol is solved and only adjacent gaps touching the selected band are inspected. For band `n`, this means the gaps to `n-1` and `n+1` when those neighbours exist. A k-point is marked hazardous when one of those selected-band adjacent gaps falls below the configured threshold. Crossings between unrelated bands are not reported here because they do not directly threaten the selected band-labelled conductivity quantity. The controlled two-level toy is retained only as a sanity check for the hazard logic; the production result is the dataset-backed hazard-point table.

The selected-gap quantiles make full-grid hazard counts interpretable. If the median selected gap is below the threshold, a full-grid hazard count can simply mean that the threshold is too large for this band scale, or that the selected band is globally close to an adjacent band. If the selected-gap quantiles are comfortably above the threshold but the hazard count remains high, that would point toward a detector logic or unit bug.

Interpretation rule: if `median gap / threshold` is below one, the chosen threshold or selected band makes the whole grid fragile. If `median gap / threshold` is comfortably above one but `min gap / threshold` is below one, the detector is finding isolated near-crossings. If the quantiles are above threshold but the hazard count is still large, that would indicate a detector logic or units bug.

For the current anchored dataset run with `band_index = 0` and `gap_threshold = 0.01`, the detector reports isolated hazards rather than global band fragility. The minimum selected adjacent gap is below threshold, so hazards are correctly flagged, but the median selected gap is hundreds of times larger than the threshold and only two points out of 10000 are hazardous. The two reported k-points are symmetry-related and have matching gaps, which is consistent with a genuine pair of local near-crossings rather than a detector-wide threshold or units failure.

Therefore this section should be read as a crossing-localisation diagnostic. The selected-band finite-difference velocity is generally safe away from the flagged points, but velocity anomalies near those k-points should be interpreted with care or handled by a crossing-aware band-continuity or degenerate-subspace method.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_dataset_band_crossing_hazards_table",
                        title="Dataset-backed band-crossing hazard summary",
                        description="Adjacent-gap hazard metrics for the selected dataset-backed H/S symbol model.",
                        headers=("metric", "value", "target"),
                        rows=(
                            _finite_field_dataset_band_hazard_rows(dataset_band_hazard_probe)
                            if dataset_band_hazard_probe is not None
                            else (TableRow(("source", "controlled fallback", "dataset context unavailable")),)
                        ),
                    ),
                    Table(
                        id="finite_field_dc_validation_dataset_band_crossing_hazard_points",
                        title="Dataset-backed hazardous k-points",
                        description="Smallest adjacent-gap hazards, sorted by gap. Empty means no sampled adjacent gap is below the threshold.",
                        headers=("k1", "k2", "lower band", "upper band", "E lower", "E upper", "gap", "threshold"),
                        rows=(
                            _finite_field_dataset_band_hazard_point_rows(dataset_band_hazard_probe.hazard_points)
                            if dataset_band_hazard_probe is not None
                            else (TableRow(("none", "none", "none", "none", "none", "none", "none", "none")),)
                        ),
                    ),
                    Table(
                        id="finite_field_dc_validation_band_crossing_hazards_toy_table",
                        title="Controlled toy hazard sanity check",
                        description="Toy-backed check that the adjacent-gap hazard machinery can flag fragile energy-ordered labels.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_band_hazard_rows(band_hazard_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_band_crossing_hazards_placeholders",
                        title="Remaining selected-band visual checks",
                        description="Dataset-backed hazard scalars and point tables are live. Remaining checks are visual overlays and label-continuity maps.",
                        headers=("check", "status"),
                        rows=(
                            TableRow(("selected-band adjacent-gap k-map", "pending")),
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
                        markdown="""*Claim.* The velocity layer is validated through three independent checks: analytic finite differences on a periodic toy, Vincent's quoted velocity samples, and Gamma reconstruction of a selected-band dataset-backed finite-difference velocity field.

The analytic toy checks that central finite differences reproduce a known periodic cosine velocity. The Vincent check reproduces quoted velocity samples through the best-adjacent Delaunay interpretation at the quoted k-points. The dataset-backed Gamma check reconstructs the selected-band finite-difference velocity sampled on the real H/S symbol grid. Physical `hbar`/unit-context scaling is handled in the unit-scaling section.
""",
                    ),
                    MarkdownBlock(
                        id="finite_field_dc_validation_analytic_velocity_prose",
                        title="Analytic finite-difference velocity",
                        markdown="""This table isolates the lowest-level velocity derivative machinery before using real dataset bands or external reference data.

The input is a controlled separable cosine production-symbol toy. Because the band energy is periodic and analytic, the exact derivatives `dE/dk1` and `dE/dk2` are known at the sampled point. The central finite-difference rows check that the finite-difference velocity calculation reproduces those analytic derivatives to finite-difference precision.

The production derivative rows check the direct symbol-derivative path, where the k-dependence enters through the representation factors in the group symbol. The Hellmann-Feynman rows check that the derivative of the local eigenvalue agrees with the generalized eigenproblem expression using `D_i H - E D_i S`. The generic/fixed rows check that the fixed-representation and generic-symbol code paths agree to roundoff.

Therefore this table proves that the local derivative engine used by the velocity calculation has the correct sign, axis, coefficient, eigenproblem plumbing, and representation-path consistency on a known periodic input. It does not by itself validate real dataset band continuity, physical m/s scaling, Vincent conductivity agreement, or Gamma reconstruction; those are checked in the neighbouring validation sections.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_analytic_velocity_table",
                        title="Analytic finite-difference velocity",
                        description="Known periodic cosine input checked against central finite differences and production derivative machinery.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_analytic_velocity_rows(velocity_probe),
                    ),
                    MarkdownBlock(
                        id="finite_field_dc_validation_vincent_simplex_ambiguity_prose",
                        title="Delaunay simplex ambiguity",
                        markdown="""Vincent's quoted velocity samples are reconstructed from a sampled energy surface by taking local gradients on a Delaunay triangulation.

At some quoted k-points the sample lies on, or numerically very close to, a grid vertex or simplex boundary. In that situation there is not a unique neighbouring simplex whose affine energy plane should be used. The default Delaunay `find_simplex` choice is deterministic, but it can select a neighbouring triangle whose local gradient is not the one used by Vincent's quoted value.

The correction procedure is therefore to inspect the adjacent candidate simplices around the quoted point, compute the velocity from each local affine gradient, and compare those candidates with the quoted Vincent velocity. The `find-simplex` row reports the error from the default simplex choice. The `best-adjacent` row reports the best error among the neighbouring simplex candidates. A collapse from a large `find-simplex` error to a roundoff-sized `best-adjacent` error means the quoted Vincent velocity is present in the reconstructed velocity field, and the discrepancy was a boundary/simplex convention rather than a units, sign, axis, or derivative-magnitude error.

This check validates Vincent's quoted velocity samples only. It does not by itself prove full conductivity agreement, because conductivity also depends on the reciprocal-space measure, weights, chemical potential, relaxation time, and tensor summation convention.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_vincent_velocity_table",
                        title="Vincent quoted velocity reconstruction",
                        description="Quoted velocity samples reproduced by resolving the adjacent-simplex ambiguity at grid vertices.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_vincent_velocity_rows(velocity_probe),
                    ),
                    MarkdownBlock(
                        id="finite_field_dc_validation_dataset_gamma_velocity_prose",
                        title="Dataset Gamma closure and k-grid stability",
                        markdown="""This table checks the modal closure of a dataset-backed finite-difference velocity field.

The selected energy-ordered band is first sampled on the dataset-backed H/S symbol grid. A periodic central finite difference then gives a sampled velocity field `v_fd(k)`. Gamma coefficients are computed from that sampled field by the discrete Fourier transform,

`Gamma_q = (1 / N_k) sum_k exp(-i q dot k) v_fd(k)`

and the field is reconstructed on the same grid by the inverse sum,

`v_reconstructed(k) = sum_q exp(i q dot k) Gamma_q`

The same-grid absolute and relative L2 errors measure `v_reconstructed - v_fd`. Values at machine precision mean that the Gamma coefficients faithfully encode the sampled finite-difference velocity field on that grid. This is an algebraic closure check of the mode representation, not yet a proof that the selected band is physically continuous.

The coarse-to-selected k-grid stability row compares a scalar mean-square velocity statistic between the coarse grid and the selected validation grid. It is displayed as a percentage. A small percentage means the bulk velocity-magnitude statistic is not changing much under this one refinement step, but it is only a proxy; a full convergence check would scan several grid sizes and inspect componentwise or spatially resolved residuals.

The selected-band hazard rows are copied from the band-crossing hazard probe used earlier in the report. They are not independently recomputed here. This keeps the Gamma velocity table consistent with the detailed selected-gap quantiles and hazardous-point table. A nonzero hazard count means that the sampled velocity field should be interpreted carefully near those k-points; it does not invalidate the same-grid Gamma closure itself.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_dataset_gamma_velocity_table",
                        title="Dataset Gamma reconstruction of finite-difference velocity",
                        description="Selected-band dataset-backed finite-difference velocity reconstructed from Gamma coefficients.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_dataset_gamma_velocity_rows(velocity_probe),
                    ),
                    MarkdownBlock(
                        id="finite_field_dc_validation_remaining_velocity_prose",
                        title="Remaining velocity work",
                        markdown="""The remaining velocity work is now mostly visual and continuity-oriented.

The scalar derivative checks, Vincent quoted-point maximum-error check, dataset Gamma closure, coarse-to-selected k-grid stability proxy, and selected-band hazard reuse are live. What remains is to display the velocity field over k-space, display the Gamma reconstruction residual field, expose Vincent's quoted samples component-by-component, and add a crossing-aware continuity visualisation near the flagged selected-band hazards.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_velocity_validation_placeholders",
                        title="Remaining production velocity checks",
                        description="Visualisations and crossing-aware continuity checks still needed before this section fully validates production band velocities.",
                        headers=("check", "status"),
                        rows=(
                            TableRow(("velocity k-map", "pending visualisation")),
                            TableRow(("velocity k-map with Gamma reconstruction residuals", "pending visualisation")),
                            TableRow(("component-level Vincent quoted-point table", "pending table; scalar max-error check is already live")),
                            TableRow(("physical hbar/unit-context scaling", "covered in unit-scaling section; production velocity-map overlay pending")),
                            TableRow(("crossing-aware velocity interpretation", "hazard metadata live; eigenvector-overlap / subspace-continuity map pending")),
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
                        markdown="""*Claim.* For a selected band on Vincent's sampled grid, the implementation reconstructs Vincent's Eq. 8.30 conductivity up to the known grid-measure convention and a remaining few-percent finite-difference/chain-rule residual.

This section is not a band-free theory check. It is a selected-band reconstruction check. The velocity construction has already been validated earlier, including the Gamma reconstruction of the sampled velocity field. Here the question is narrower: starting from the selected-band energy grid, can the diagnostic reproduce the conductivity implied by Vincent's Eq. 8.30 and compare it with the weak chain-rule limit?

The same Eq. 8.30 calculation is also reconstructed through a Gamma, Q, and tilde-rho modal decomposition. The existing strong spectral zero-field trace is kept as a separate comparison row because it is a different derivative construction, not the Eq. 8.30 modal reconstruction.

Vincent's finite-field expression shifts the Fermi factor and one velocity factor along the electric-field direction before integrating over the relaxation-time parameter.
""",
                    ),
                    TypstMathBlock(
                        id="finite_field_dc_validation_vincent_eq830_equation",
                        math=TypstMath(
                            "$ sigma_(alpha beta)^\"8.30\" (E) = (e^2) / (k_B T tau) sum_k w_k integral_0^infinity dd(s) space s exp(-s / tau) space g_0 (k + (s e) / hbar E) space  (1 - g_0 (k + (s e) / hbar E)) v_n^beta (k + (s e) / hbar E) v_n^alpha (k) $",
                            display=True,
                            name="finite_field_vincent_eq830_equation",
                        ),
                    ),
                    MarkdownBlock(
                        id="finite_field_dc_validation_vincent_eq830_discrete_prose",
                        title="Finite-difference Eq. 8.30 trace",
                        markdown="""The `Eq. 8.30 finite-difference trace` row is the direct numerical implementation of this selected-band formula. The sampled band energy is differentiated numerically to obtain the velocities, the shifted argument is evaluated on the grid convention used by the comparison domain, and the resulting tensor is displayed in Vincent's grid-measure convention. This is the row that should be compared directly with Vincent's reported conductivity after the known $(2 pi)^2$ measure issue is accounted for.

The weak chain-rule row is the zero-field limit of the same calculation.
""",
                    ),
                    TypstMathBlock(
                        id="finite_field_dc_validation_vincent_weak_limit_equation",
                        math=TypstMath(
                            "$ sigma_(alpha beta)^\"weak\" = e^2 tau sum_k w_k (-partial_E g_0 (E_k)) space v_n^alpha (k) v_n^beta (k) $",
                            display=True,
                            name="finite_field_vincent_weak_limit_equation",
                        ),
                    ),
                    MarkdownBlock(
                        id="finite_field_dc_validation_vincent_weak_limit_prose",
                        title="Weak chain-rule limit",
                        markdown="""The weak row should sit close to the finite-difference Eq. 8.30 row when the applied field shift is small and the finite-difference/quadrature choices are consistent. In the current comparison, these two rows differ only by the small finite-field and grid quadrature effect, while both remain a few percent from Vincent. That is the evidence that the remaining Vincent residual is not a missing velocity signal; it is already present in the finite-difference/chain-rule conductivity reconstruction.

The following modal check decomposes the same selected-band Eq. 8.30 calculation into lattice Fourier modes.
""",
                    ),
                    TypstMathBlock(
                        id="finite_field_dc_validation_vincent_modal_equation",
                        math=TypstMath(
                            "$ sigma_(alpha beta)^\"8.30 modal\" (E) = C sum_R Gamma_R^alpha Q_R (E) tilde(rho)_R^beta $",
                            display=True,
                            name="finite_field_vincent_modal_equation",
                        ),
                    ),
                    TypstMathBlock(
                        id="finite_field_dc_validation_vincent_modal_definitions_equation",
                        math=TypstMath(
                            "$ v_n^alpha (k) = sum_R e^(i R dot k) Gamma_R^alpha, quad rho_E^beta (k) = g_0 (k) (1 - g_0 (k)) v_n^beta (k) = sum_R e^(i R dot k) tilde(rho)_R^beta $",
                            display=True,
                            name="finite_field_vincent_modal_definitions_equation",
                        ),
                    ),
                    TypstMathBlock(
                        id="finite_field_dc_validation_vincent_modal_response_equation",
                        math=TypstMath(
                            "$ Q_R (E) = sum_l w_l t_l B_R (delta k_l), quad delta k_l = (tau t_l e) / hbar E $",
                            display=True,
                            name="finite_field_vincent_modal_response_equation",
                        ),
                    ),
                    MarkdownBlock(
                        id="finite_field_dc_validation_vincent_modal_prose",
                        title="Eq. 8.30 modal reconstruction",
                        markdown="""The `Eq. 8.30 Gamma-Q-rho trace` row is the selected-band modal reconstruction of the same finite-field Eq. 8.30 quadrature. It uses `Gamma` for the velocity coefficients and $tilde(rho)$ for the occupation-weighted velocity coefficients. The scalar $Q_R (E)$ is the quadrature-and-shift factor for mode $R$ in the direct Eq. 8.30 implementation.

Here $C$ collects the factors that are common to the direct Eq. 8.30 trace and the modal reconstruction: the physical conductivity prefactor, the k-cell/grid measure, and the normalisation used for the selected comparison row. The raw implementation uses the continuum Brillouin-zone measure $d^2 k / (2 pi)^2$, because the reciprocal grid cell is $A_("BZ") / N_k$ and the prefactor still contains $(2 pi)^(-2)$. The Vincent target is instead reproduced only after multiplying the raw tensor by $(2 pi)^2$, i.e. by using $A_("BZ") / N_k$ without the continuum denominator.

The table below therefore treats the Vincent comparison as a normalisation audit. It does not assert that the physical conductivity is defined up to convention. The physical SI normalisation should be unique; the point is that the copied Vincent target appears to use a different k-measure normalisation from the standard continuum form.

The quadrature index $l$ runs over the Gauss-Laguerre nodes used for the relaxation-time integral, with node $t_l$ and weight $w_l$. The field-dependent shift is $delta k_l = (tau t_l e) / hbar E$. The factor $B_R(delta k_l)$ is the periodic bilinear shift response used by the direct finite-difference implementation. If the shifted grid were evaluated by an exact Fourier shift rather than bilinear interpolation, this factor would reduce to the corresponding phase response for mode $R$.

This row is therefore a representation check, not a new physical formula. It asks whether the Gamma/Q/$tilde(rho)$ modal summation reconstructs the direct shifted-chain-rule Eq. 8.30 calculation. The `modal closure residual` row is the closure error for that check.

This $Q_R(E)$ is not the same object as the strong-field response factor $F_(n,R)^beta(E)$ used in the strong spectral formula below. The letter `F` is reserved here for the differential response factor from the thesis notes.
""",
                    ),
                    TypstMathBlock(
                        id="finite_field_dc_validation_vincent_strong_spectral_equation",
                        math=TypstMath(
                            "$ sigma_(alpha beta)^\"strong\" (E) = e A_\"BZ\" sum_n sum_R tilde(f)^0_(n,R) Gamma_(n,R)^alpha F_(n,R)^beta (E) $",
                            display=True,
                            name="finite_field_vincent_strong_spectral_equation",
                        ),
                    ),
                    TypstMathBlock(
                        id="finite_field_dc_validation_vincent_strong_response_equation",
                        math=TypstMath(
                            "$ F_(n,R)^beta (E) = (- i (e tau_n / hbar) R^beta) / (1 - i (e tau_n / hbar) E dot R)^2, quad F_(n,R)^beta (0) = - i (e tau_n / hbar) R^beta $",
                            display=True,
                            name="finite_field_vincent_strong_response_equation",
                        ),
                    ),
                    MarkdownBlock(
                        id="finite_field_dc_validation_vincent_strong_spectral_prose",
                        title="Strong spectral zero-field trace",
                        markdown="""The `strong spectral zero-field trace` row is a different modal object. It follows the strong-field spectral construction from the thesis notes. The sampled equilibrium occupation $f_0(k) = f_0(E_n(k))$ is stored as Fourier coefficients $tilde(f)^0_(n,R)$, and the field dependence is carried by the differential response factor $F_(n,R)^beta(E)$.

This $F$ is the notes-style field derivative factor. It comes from differentiating the strong-field denominator $(1 - i (e tau_n / hbar) E dot R)^(-1)$ with respect to the electric-field component $E^beta$. At zero field it reduces to multiplication by $- i (e tau_n / hbar) R^beta$, which is the spectral derivative of the sampled periodic occupation response.

This is not the same operation as the weak chain-rule row. The weak chain-rule row differentiates the selected-band energy first, forms the velocity, and then applies the pointwise derivative $f_0'(E(k)) partial_k E(k)$. The strong spectral row instead differentiates the already-sampled periodic occupation field $f_0(k)$ as a Fourier series. On a finite grid, especially near a sharp or under-resolved Fermi window, these two procedures need not agree.

Therefore the strong spectral row is included as a comparison and warning row: it shows the behaviour of the notes-style spectral derivative construction, while the Eq. 8.30 Gamma-Q-rho row is the modal reconstruction of the direct shifted-chain-rule Eq. 8.30 calculation.
""",
                    ),
                    TypstMathBlock(
                        id="finite_field_dc_validation_vincent_residual_equation",
                        math=TypstMath(
                            "$ Delta = 100 times abs(Tr(sigma^\"calc\") - Tr(sigma^\"vin\")) / abs(Tr(sigma^\"vin\")) $",
                            display=True,
                            name="finite_field_vincent_residual_equation",
                        ),
                    ),
                    MarkdownBlock(
                        id="finite_field_dc_validation_vincent_residual_prose",
                        title="Residual interpretation",
                        markdown="""The intended reading is: Eq. 8.30 finite differences reproduce Vincent's conductivity to the known grid-measure convention and a few-percent residual; the weak chain-rule limit explains why this residual is already present in the finite-difference chain-rule calculation; and the Eq. 8.30 Gamma-Q-rho row checks that the selected-band modal decomposition reconstructs the direct Eq. 8.30 implementation.

The strong spectral zero-field trace should not be read as a failed Eq. 8.30 reconstruction. It is a different comparison row, included to expose the difference between the pointwise chain-rule derivative and the spectral derivative of the sampled occupation.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_vincent_reconstruction_table",
                        title="Selected-band Eq. 8.30 reconstruction",
                        description="Trace reconstructions shown as value/residual pairs against Vincent's target trace.",
                        headers=("path", "value", "residual/status"),
                        rows=_finite_field_vincent_rows(vincent_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_vincent_normalisation_table",
                        title="Vincent normalisation audit",
                        description="Checks whether the comparison uses the continuum d²k/(2π)² measure or the A_BZ/Nk grid measure without the continuum denominator.",
                        headers=("check", "value", "status"),
                        rows=_finite_field_vincent_normalisation_rows(vincent_probe),
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
                        markdown="""*Claim.* The band-labelled strong DC conductivity is internally closed as a lattice-mode spectral tensor and its residual against the weak-chain calculation is exposed rather than hidden.

This first implementation reuses the existing `BandIndexedStrongDcResult` on Vincent's epsilon grid. It checks that the mode tensor re-sums to the reported strong tensor, that Fourier coefficients and response factors are finite, and that imaginary leakage is negligible. The displayed default trace rows use the continuum `A_BZ / (N_k (2π)^2)` normalisation; no-`(2π)^2` rows are exposed only as Vincent-comparison scale diagnostics.
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
                id="finite_field_dc_validation_strong_eq830_limit",
                title="Strong DC / Eq. 8.30 limit",
                description="Compares the strong differential-response DC tensor against the continuum-normalised Eq. 8.30 shifted finite-difference tensor.",
                collapsed=False,
                body=(
                    MarkdownBlock(
                        id="finite_field_dc_validation_strong_eq830_limit_prose",
                        title="Validation claim",
                        markdown="""*Claim.* The band-labelled strong DC tensor should agree with the finite-difference Eq. 8.30 response only in regimes where the finite-field shift, k-grid sampling, velocity basis, and occupation derivative define the same continuum object.

This section compares the strong differential-response tensor directly against the Eq. 8.30 shifted finite-difference tensor on Vincent-grid inputs with the continuum `A_BZ / (N_k (2π)^2)` normalisation. The residual is exposed as the object to study before applying Eq. 8.30 to the full physical dataset.
""",
                    ),
                    Table(
                        id="finite_field_dc_validation_strong_eq830_limit_table",
                        title="Strong DC / Eq. 8.30 limit metrics",
                        description="Continuum-normalised strong differential response versus Eq. 8.30 shifted finite-difference response.",
                        headers=("metric", "value", "target"),
                        rows=_finite_field_strong_eq830_limit_rows(strong_eq830_limit_probe),
                    ),
                    Table(
                        id="finite_field_dc_validation_strong_eq830_limit_placeholders",
                        title="Remaining strong/Eq. 8.30 limit checks",
                        description="Checks still needed before using Eq. 8.30 on the full physical dataset.",
                        headers=("check", "status"),
                        rows=(
                            TableRow(("k-grid density sweep", "pending")),
                            TableRow(("temperature / smoothness sweep", "pending")),
                            TableRow(("component-level tensor comparison", "pending")),
                            TableRow(("dataset-backed Eq. 8.30 run", "pending")),
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
                        markdown="""*Claim.* On a matched spectral-basis analytic toy, the finite-field DC conductivity approaches the weak-field DC result as E -> 0.

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
                        markdown="""*Claim.* On the current strong-grid mode object, the lattice-mode decomposition into Gamma, F, and tilde(rho) reconstructs the sampled fields and total strong spectral DC tensor.

This first implementation checks the actual `BandIndexedStrongDcResult` mode objects: Gamma reconstructs the sampled velocity field, tilde(rho) reconstructs the sampled occupation, and summing the conductivity mode tensor reconstructs the total strong DC tensor. The reported conductivity trace is converted to the continuum `A_BZ / (N_k (2π)^2)` normalisation. This validates the mode algebra used by the finite-field target, but dataset-backed finite-field mode closure remains pending.
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
                        markdown="""*Claim.* Controlled analytic toys isolate algebra, derivatives, crossings, overlap health, and unit scaling before any BigDFT dataset-specific effects are introduced.

This section now summarises the real toy-backed probes used by the validation ladder. The missing analytic target is the finite-field lattice-mode closure toy for Gamma/Q/rho decomposition.
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
                            TableRow(("finite-field Gamma/Q/rho closure toy", "pending")),
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
                        markdown="""*Claim.* The core unit factors needed by finite-field velocity and conductivity comparisons are explicit and numerically checked.

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
                        markdown="""*Claim.* On a periodic analytic velocity-square toy, the sampled k-grid measure matches the exact full-period average under N_u/N_v refinement.

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
                        markdown="""*Claim.* On a controlled separable-cosine toy, the expected k-inversion and velocity-square tensor symmetries are satisfied up to numerical roundoff.

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
