"""Diagnostics for validating the Boltzmann operator approach."""

from __future__ import annotations

import numpy as np

from dft_local.diagnostics.models import (
    DiagnosticResult,
    DiagnosticSection,
    DiagnosticSpec,
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
    )
