"""Diagnostics for validating the Boltzmann operator approach."""

from __future__ import annotations

import numpy as np

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


def _fmt_probe_value(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.8e}"
    if value is None:
        return "None"
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
        TableRow(("temperature T", str(inputs.get("temperature", 300.0)))),
        TableRow(("chemical potential mu", str(inputs.get("mu", 0.0)))),
        TableRow(("relaxation time tau", str(inputs.get("tau", 1.0)))),
        TableRow(("units", str(inputs.get("units", "eVAng")))),
        TableRow(("N_u, N_v", f"{inputs.get('n_u', 11)}, {inputs.get('n_v', 11)}")),
        TableRow(("electric field E", str(inputs.get("electric_field", 1.0)))),
        TableRow(("field angle theta", str(inputs.get("theta", 0.0)))),
        TableRow(("band index n", f"{inputs.get('band_index', 0)}; energy ordering")),
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
            _finite_field_dc_section(
                section_id="finite_field_dc_validation_band_crossing_hazards",
                title="Band-crossing hazards",
                description="Maps k-space regions where energy-ordered band labels become fragile.",
                claim="For the selected energy-ordered band, the diagnostic identifies k-space regions where near crossings or eigenvector jumps can contaminate velocity and conductivity.",
                evidence=(
                    ("minimum gap to adjacent bands", "dummy", "locate physical crossing hazards"),
                    ("same-label neighbour overlap", "dummy", "detect eigenvector or label jumps"),
                    ("velocity anomaly overlay", "dummy", "explain spikes using crossing maps"),
                    ("active subspace comparison", "dummy", "check whether summed near-degenerate bands are stable"),
                ),
                placeholders=("band energy surface", "minimum-gap k-map", "label-overlap k-map", "velocity anomaly overlay"),
            ),
            _finite_field_dc_section(
                section_id="finite_field_dc_validation_velocity_validation",
                title="Velocity validation",
                description="Checks velocity before it is used inside conductivity.",
                claim="The velocity implementation agrees with periodic analytic inputs, finite-difference checks, unit scaling, Gamma reconstruction, and Vincent's velocity field under matched inputs.",
                evidence=(
                    ("constant and cosine analytic bands", "dummy", "known periodic velocity outputs"),
                    ("finite-difference velocity comparison", "dummy", "local numerical derivative check"),
                    ("velocity unit scaling", "dummy", "energy-length-over-hbar scaling"),
                    ("Gamma modal reconstruction", "dummy", "local mode sum reconstructs velocity/current ingredient"),
                    ("Vincent velocity field comparison", "dummy", "external velocity contact point"),
                ),
                placeholders=("analytic velocity table", "finite-difference scatter", "Gamma closure table", "Vincent velocity comparison"),
            ),
            _finite_field_dc_section(
                section_id="finite_field_dc_validation_vincent_reconstruction",
                title="Vincent reconstruction",
                description="External reconstruction and residual audit.",
                claim="With Vincent's inputs, the implementation reconstructs his DC calculation once reciprocal-space normalization is audited; the remaining few-percent discrepancy is separated into derivative, interpolation, and k-sampling hypotheses.",
                evidence=(
                    ("2π normalization audit", "dummy", "show right and wrong placements against analytic inputs"),
                    ("Vincent tensor reconstruction", "dummy", "compare target and reproduced tensor"),
                    ("residual 3–4% hypothesis scan", "dummy", "separate remaining numerical/model choices"),
                    ("Fermi window / mu check", "dummy", "rule out simple input mismatch"),
                ),
                placeholders=("2π audit table", "Vincent tensor comparison", "residual hypothesis table", "mu scan"),
            ),
            _finite_field_dc_section(
                section_id="finite_field_dc_validation_strong_dc_validation",
                title="Strong DC validation",
                description="Checks the finite-field band-labelled strong DC tensor in regimes where independent formulae should meet.",
                claim="The band-labelled strong DC conductivity agrees with the Vincent/Ashcroft-style expression in their shared assumptions and parameter regime.",
                evidence=(
                    ("matched-input strong vs Vincent/Ashcroft", "dummy", "same dispersion, tau, T, mu, k-grid, and normalization"),
                    ("strong spectral decomposition", "dummy", "band-labelled tensor assembly is internally visible"),
                    ("temperature / smoothness regime check", "dummy", "avoid overclaiming under-resolved low-temperature derivatives"),
                ),
                placeholders=("strong DC tensor table", "strong/contact error plot", "spectral decomposition summary"),
            ),
            _finite_field_dc_section(
                section_id="finite_field_dc_validation_weak_dc_limit",
                title="Weak DC limit",
                description="Small-field limit of finite-field DC conductivity.",
                claim="The finite-field band-labelled DC conductivity approaches the weak-field DC result as E -> 0.",
                evidence=(
                    ("E sweep", "dummy", "show finite-field tensor tends to weak-field tensor"),
                    ("relative tensor error", "dummy", "measure convergence window"),
                    ("roundoff floor marker", "dummy", "avoid trusting too-small field shifts"),
                ),
                placeholders=("E sweep plot", "finite-minus-weak error plot", "asymptotic-window table"),
            ),
            _finite_field_dc_section(
                section_id="finite_field_dc_validation_mode_decomposition",
                title="Mode decomposition",
                description="Closure checks for Gamma, F, and tilde(rho).",
                claim="The lattice-mode decomposition into Gamma, F, and tilde(rho) reconstructs the total finite-field band-labelled DC conductivity tensor.",
                evidence=(
                    ("Gamma closure", "dummy", "velocity/current mode ingredient reconstructs direct value"),
                    ("F and tilde(rho) sanity", "dummy", "response and density factors have expected finite values"),
                    ("conductivity recomposition", "dummy", "mode sum reconstructs total tensor"),
                    ("cumulative mode contribution", "dummy", "show how many modes carry the tensor"),
                ),
                placeholders=("Gamma/F/tilde(rho) tables", "mode recomposition tensor", "difference tensor", "spatial mode maps"),
            ),
            _finite_field_dc_section(
                section_id="finite_field_dc_validation_analytic_toys",
                title="Analytic toys",
                description="Periodic known-input checks.",
                claim="On controlled periodic analytic inputs, the diagnostic gives expected zero, symmetry, scaling, and two-band behaviour.",
                evidence=(
                    ("constant band", "dummy", "zero velocity and zero conductivity"),
                    ("1D cosine band", "dummy", "periodic known derivative and even v^2 contribution"),
                    ("2D separable cosine band", "dummy", "controlled anisotropy and zero off-diagonal by symmetry"),
                    ("periodic two-level Dirac-like model", "dummy", "gapped and near-crossing two-band behaviour"),
                ),
                placeholders=("analytic toy summary table", "cosine derivative plot", "two-level band hazard map"),
            ),
            _finite_field_dc_section(
                section_id="finite_field_dc_validation_unit_scaling",
                title="Unit consistency",
                description="Physical scaling and SI conversion checks.",
                claim="The same physical calculation gives the same SI result after conversion from different internal unit systems.",
                evidence=(
                    ("velocity unit conversion", "dummy", "m/s agreement from different internal units"),
                    ("conductivity unit conversion", "dummy", "S/m agreement from different internal units"),
                    ("tau linearity", "dummy", "DC conductivity scales linearly with tau where expected"),
                    ("energy/length scale audit", "dummy", "catch missing hbar or length factors"),
                ),
                placeholders=("unit conversion table", "tau scaling plot", "scale-law table"),
            ),
            _finite_field_dc_section(
                section_id="finite_field_dc_validation_k_convergence",
                title="k-point convergence",
                description="Refinement in N_u and N_v only.",
                claim="The reported conductivity is stable under refinement of N_u and N_v.",
                evidence=(
                    ("sigma component convergence", "dummy", "tensor components stabilize"),
                    ("trace / norm convergence", "dummy", "robust scalar convergence metric"),
                    ("weak-limit error convergence", "dummy", "small-field comparison is not grid artefact"),
                    ("Vincent residual convergence", "dummy", "separate 3–4% residual from k-grid error"),
                ),
                placeholders=("N_u,N_v convergence plot", "relative-to-finest table", "component convergence table"),
            ),
            _finite_field_dc_section(
                section_id="finite_field_dc_validation_symmetry",
                title="Symmetry sanity",
                description="Tensor, k-space, and direction-sweep symmetry checks.",
                claim="Tensor components and direction sweeps obey expected lattice and time-reversal symmetries up to finite-size and sampling defects.",
                evidence=(
                    ("sigma_xy vs sigma_yx", "dummy", "Onsager/time-reversal sanity where applicable"),
                    ("direction sweep periodicity", "dummy", "lattice symmetry in field angle"),
                    ("k inversion checks", "dummy", "E(k)=E(-k), velocity oddness, conductivity evenness"),
                    ("finite-sample symmetry defect", "dummy", "separate physics from sampling/truncation defects"),
                ),
                placeholders=("tensor symmetry table", "direction sweep plot", "k-inversion defect map"),
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
