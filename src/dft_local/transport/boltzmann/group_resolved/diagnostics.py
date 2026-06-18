"""Diagnostics for group-resolved Boltzmann conductivity."""

from __future__ import annotations

import numpy as np

from dft_local.core.units import CONDUCTIVITY, DIMENSIONLESS, ENERGY, DisplayQuantity, VELOCITY
from dft_local.diagnostics.models import (
    Card,
    DiagnosticResult,
    DiagnosticSpec,
    HtmlBlock,
    InputSpec,
    WebGLView,
    DiagnosticSection,
    Table,
    TableRow,
)
from dft_local.transport.boltzmann.calculation.diagnostics import conductivity_grid
from dft_local.transport.boltzmann.calculation.core import BoltzmannConductivity
from dft_local.transport.boltzmann.strong_dc.core import (
    band_indexed_strong_dc_from_velocity_grid,
)
from dft_local.transport.boltzmann.group_resolved.core import (
    band_resolved_compact_conductivity,
)


def compute_overview(ctx, inputs: dict[str, object]) -> DiagnosticResult:
    kernel_choice = str(inputs.get("kernel", "average_star"))
    nu = int(inputs.get("nu", 5))
    nv = int(inputs.get("nv", 5))
    mu = float(inputs.get("mu", 0.0))
    temperature = float(inputs.get("temperature", 300.0))
    tau = float(inputs.get("tau", 1.0))
    band = int(inputs.get("band", 0))

    KH, KS = ctx.kernels(kernel_choice)
    k1, k2, weights = conductivity_grid(nu=nu, nv=nv, central_bz=False)

    calc = BoltzmannConductivity.from_arrays(
        KH,
        KS,
        k1,
        k2,
        irrep_weights=weights,
        irrep_to_physical_k=(1.0 / ctx.state.data.length_conversion_disk_to_working) * np.eye(2),
        unit_context=ctx.state.data.working_unit_context,
        mu=mu,
        temperature=temperature,
        omega=0.0,
        tau=tau,
        name=f"group-resolved compact check {kernel_choice}",
    ).run()

    assert calc.velocities is not None
    assert calc.ac_weights is not None
    assert calc.sigma is not None

    resolved = band_resolved_compact_conductivity(
        velocities=calc.velocities,
        weights=calc.ac_weights,
        physical_k_weights=calc.physical_k_weights,
    )

    band = max(0, min(band, resolved.sigma_band.shape[0] - 1))
    residual = resolved.sigma - calc.sigma
    assert calc.energies is not None

    nbands = int(resolved.sigma_band.shape[0])
    k1_grid = np.asarray(k1, dtype=float).reshape(nu, nv)
    k2_grid = np.asarray(k2, dtype=float).reshape(nu, nv)
    energy_grid = np.asarray(calc.energies, dtype=float).reshape(nu, nv, nbands)
    velocity_grid = np.asarray(calc.velocities, dtype=float).reshape(nu, nv, nbands, 2)
    vx_grid = velocity_grid[:, :, :, 0]
    vy_grid = velocity_grid[:, :, :, 1]
    speed_grid = np.sqrt(vx_grid * vx_grid + vy_grid * vy_grid)
    velocity_unit = calc.unit_context.unit_for_dimension(VELOCITY)
    velocity_unit_symbol = velocity_unit.symbol

    common_surface_payload = {
        "kind": "band-surface-preview",
        "nu": nu,
        "nv": nv,
        "k1": k1_grid.tolist(),
        "k2": k2_grid.tolist(),
        "mask": (
            (np.abs(k1_grid) <= np.pi + 1e-12)
            & (np.abs(k2_grid) <= np.pi + 1e-12)
            & (np.abs(k1_grid - k2_grid) <= np.pi + 1e-12)
        ).tolist(),
        "bands": list(range(nbands)),
        "nbands": nbands,
        "selected_band": band,
        "energy_unit": calc.unit_context.energy.symbol,
    }

    surface_payload = {
        **common_surface_payload,
        "energies": energy_grid.tolist(),
        "fields": [
            {"id": "energy", "label": "Energy", "unit": calc.unit_context.energy.symbol, "signed": True},
        ],
        "field_values": {
            "energy": energy_grid.tolist(),
        },
        "selected_field": "energy",
    }

    velocity_surface_payload = {
        **common_surface_payload,
        "energies": energy_grid.tolist(),
        "fields": [
            {"id": "vx", "label": "Velocity x", "unit": velocity_unit_symbol, "signed": True},
            {"id": "vy", "label": "Velocity y", "unit": velocity_unit_symbol, "signed": True},
            {"id": "speed", "label": "Speed", "unit": velocity_unit_symbol, "signed": False},
        ],
        "field_values": {
            "vx": vx_grid.tolist(),
            "vy": vy_grid.tolist(),
            "speed": speed_grid.tolist(),
        },
        "selected_field": "speed",
    }

    def conductivity_quantity(value: complex | float, name: str) -> DisplayQuantity:
        return DisplayQuantity(
            value=float(np.real(value)),
            dimension=CONDUCTIVITY,
            unit=calc.unit_context.unit_for_dimension(CONDUCTIVITY),
            name=name,
        )

    def velocity_quantity(value: complex | float, name: str) -> DisplayQuantity:
        return DisplayQuantity(
            value=float(np.real(value)),
            dimension=VELOCITY,
            unit=calc.unit_context.unit_for_dimension(VELOCITY),
            name=name,
        )

    def energy_quantity(value: complex | float, name: str) -> DisplayQuantity:
        return DisplayQuantity(
            value=float(np.real(value)),
            dimension=ENERGY,
            unit=calc.unit_context.unit_for_dimension(ENERGY),
            name=name,
        )

    def unitless_quantity(value: complex | float, name: str) -> DisplayQuantity:
        return DisplayQuantity(
            value=float(np.real(value)),
            dimension=DIMENSIONLESS,
            unit=calc.unit_context.unit_for_dimension(DIMENSIONLESS),
            name=name,
        )

    velocity_rows = tuple(
        TableRow((
            n,
            velocity_quantity(np.min(vx_grid[:, :, n]), "min vx"),
            velocity_quantity(np.mean(vx_grid[:, :, n]), "mean vx"),
            velocity_quantity(np.max(vx_grid[:, :, n]), "max vx"),
            velocity_quantity(np.min(vy_grid[:, :, n]), "min vy"),
            velocity_quantity(np.mean(vy_grid[:, :, n]), "mean vy"),
            velocity_quantity(np.max(vy_grid[:, :, n]), "max vy"),
            velocity_quantity(np.min(speed_grid[:, :, n]), "min speed"),
            velocity_quantity(np.mean(speed_grid[:, :, n]), "mean speed"),
            velocity_quantity(np.max(speed_grid[:, :, n]), "max speed"),
        ))
        for n in range(nbands)
    )

    h_hermitian_defects = []
    s_hermitian_defects = []
    s_min_eigs = []
    s_max_eigs = []
    s_condition_estimates = []

    for problem in calc.problems:
        Hk = np.asarray(problem.Hk)
        Sk = np.asarray(problem.Sk)

        h_hermitian_defects.append(float(np.linalg.norm(Hk - Hk.conj().T)))
        s_hermitian_defects.append(float(np.linalg.norm(Sk - Sk.conj().T)))

        s_eigs = np.linalg.eigvalsh(Sk)
        s_min = float(np.min(s_eigs).real)
        s_max = float(np.max(s_eigs).real)
        s_min_eigs.append(s_min)
        s_max_eigs.append(s_max)
        s_condition_estimates.append(s_max / s_min if s_min != 0.0 else np.inf)

    def card_quantity(quantity: DisplayQuantity) -> str:
        return f"{quantity.value:.6g} {quantity.unit.symbol}"


    symbol_sanity_rows = (
        TableRow((
            "H Hermiticity defect",
            unitless_quantity(np.max(h_hermitian_defects), "max H Hermiticity defect"),
            unitless_quantity(np.mean(h_hermitian_defects), "mean H Hermiticity defect"),
            "||H(k) - H(k)^†|| over sampled irreps",
        )),
        TableRow((
            "S Hermiticity defect",
            unitless_quantity(np.max(s_hermitian_defects), "max S Hermiticity defect"),
            unitless_quantity(np.mean(s_hermitian_defects), "mean S Hermiticity defect"),
            "||S(k) - S(k)^†|| over sampled irreps",
        )),
        TableRow((
            "S minimum eigenvalue",
            unitless_quantity(np.min(s_min_eigs), "minimum S eigenvalue"),
            unitless_quantity(np.mean(s_min_eigs), "mean minimum S eigenvalue"),
            "lowest eigenvalue of S(k)",
        )),
        TableRow((
            "S maximum eigenvalue",
            unitless_quantity(np.max(s_max_eigs), "maximum S eigenvalue"),
            unitless_quantity(np.mean(s_max_eigs), "mean maximum S eigenvalue"),
            "highest eigenvalue of S(k)",
        )),
        TableRow((
            "S condition estimate",
            unitless_quantity(np.max(s_condition_estimates), "maximum S condition estimate"),
            unitless_quantity(np.mean(s_condition_estimates), "mean S condition estimate"),
            "max eig(S(k)) / min eig(S(k))",
        )),
    )

    eig_residuals = []
    s_norm_defects = []
    s_offdiag_defects = []

    for ik, problem in enumerate(calc.problems):
        Hk = np.asarray(problem.Hk)
        Sk = np.asarray(problem.Sk)
        energies_k = np.asarray(calc.energies[ik])
        vectors_k = np.asarray(calc.vectors[ik])

        gram = vectors_k.conj().T @ Sk @ vectors_k
        s_offdiag_defects.append(float(np.max(np.abs(gram - np.diag(np.diag(gram))))))

        for n in range(vectors_k.shape[1]):
            phi = vectors_k[:, n]
            residual_n = Hk @ phi - energies_k[n] * (Sk @ phi)
            eig_residuals.append(float(np.linalg.norm(residual_n)))
            s_norm_defects.append(float(abs(complex(phi.conj() @ Sk @ phi) - 1.0)))

    eigensystem_sanity_rows = (
        TableRow((
            "generalized eigen residual",
            unitless_quantity(np.max(eig_residuals), "max generalized eigen residual"),
            unitless_quantity(np.mean(eig_residuals), "mean generalized eigen residual"),
            "||H(k) phi_n(k) - E_n(k) S(k) phi_n(k)||",
        )),
        TableRow((
            "S-normalisation defect",
            unitless_quantity(np.max(s_norm_defects), "max S-normalisation defect"),
            unitless_quantity(np.mean(s_norm_defects), "mean S-normalisation defect"),
            "|phi_n(k)^† S(k) phi_n(k) - 1|",
        )),
        TableRow((
            "S-orthogonality off-diagonal defect",
            unitless_quantity(np.max(s_offdiag_defects), "max S-orthogonality off-diagonal defect"),
            unitless_quantity(np.mean(s_offdiag_defects), "mean S-orthogonality off-diagonal defect"),
            "max offdiag |phi_i(k)^† S(k) phi_j(k)|",
        )),
    )

    eigensystem_energy_rows = tuple(
        TableRow((
            n,
            energy_quantity(np.min(energy_grid[:, :, n]), f"band {n} minimum energy"),
            energy_quantity(np.mean(energy_grid[:, :, n]), f"band {n} mean energy"),
            energy_quantity(np.max(energy_grid[:, :, n]), f"band {n} maximum energy"),
        ))
        for n in range(nbands)
    )

    recomputed_velocities = np.empty_like(calc.velocities)
    velocity_imag_leakage = []

    for ik, problem in enumerate(calc.problems):
        dH, dS = calc.physical_derivative_symbols(problem)
        energies_k = np.asarray(calc.energies[ik])
        vectors_k = np.asarray(calc.vectors[ik])

        for i in range(calc.dimension):
            for n in range(nbands):
                phi = vectors_k[:, n]
                numerator = np.vdot(phi, (dH[i] - energies_k[n] * dS[i]) @ phi)
                recomputed_velocities[ik, i, n] = float(np.real(numerator)) / float(calc.hbar_working)
                velocity_imag_leakage.append(float(abs(np.imag(numerator)) / float(calc.hbar_working)))

    velocity_recompute_delta = recomputed_velocities - np.asarray(calc.velocities)

    velocity_recompute_rows = (
        TableRow((
            "HF velocity recomputation delta",
            velocity_quantity(np.max(np.abs(velocity_recompute_delta)), "max HF velocity recomputation delta"),
            velocity_quantity(np.mean(np.abs(velocity_recompute_delta)), "mean HF velocity recomputation delta"),
            "recomputed diagonal HF velocity minus stored velocity",
        )),
        TableRow((
            "HF numerator imaginary leakage",
            velocity_quantity(np.max(velocity_imag_leakage), "max HF numerator imaginary leakage"),
            velocity_quantity(np.mean(velocity_imag_leakage), "mean HF numerator imaginary leakage"),
            "imaginary part discarded before taking the real velocity",
        )),
        TableRow((
            "stored velocity magnitude",
            velocity_quantity(np.max(np.abs(calc.velocities)), "max stored velocity magnitude"),
            velocity_quantity(np.mean(np.abs(calc.velocities)), "mean stored velocity magnitude"),
            "absolute value of stored band velocities",
        )),
    )

    sigma_from_bands = np.sum(resolved.sigma_band, axis=0)
    sigma_from_k = np.sum(calc.sigma_k, axis=0)
    sigma_from_bands_delta = sigma_from_bands - calc.sigma
    sigma_from_k_delta = sigma_from_k - calc.sigma
    compact_real_norm = float(np.linalg.norm(calc.sigma.real))
    compact_imag_norm = float(np.linalg.norm(calc.sigma.imag))
    compact_imag_over_real = compact_imag_norm / compact_real_norm if compact_real_norm != 0.0 else np.nan

    conductivity_assembly_rows = (
        TableRow((
            "band sum - compact",
            conductivity_quantity(np.linalg.norm(sigma_from_bands_delta), "band sum minus compact norm"),
            conductivity_quantity(np.trace(sigma_from_bands_delta), "band sum minus compact trace"),
            "sum_n sigma_n - sigma_compact",
        )),
        TableRow((
            "k sum - compact",
            conductivity_quantity(np.linalg.norm(sigma_from_k_delta), "k sum minus compact norm"),
            conductivity_quantity(np.trace(sigma_from_k_delta), "k sum minus compact trace"),
            "sum_k sigma_k - sigma_compact",
        )),
        TableRow((
            "compact tensor imaginary leakage",
            conductivity_quantity(compact_imag_norm, "compact tensor imaginary norm"),
            unitless_quantity(compact_imag_over_real, "compact tensor imaginary over real norm"),
            "||Im sigma_compact|| and ||Im sigma_compact|| / ||Re sigma_compact||",
        )),
        TableRow((
            "compact tensor magnitude",
            conductivity_quantity(np.linalg.norm(calc.sigma), "compact tensor norm"),
            conductivity_quantity(np.trace(calc.sigma), "compact tensor trace"),
            "overall compact conductivity scale",
        )),
    )

    strong_reference_rows = []
    strong_reference_band_sigma = np.empty_like(resolved.sigma_band)
    strong_reference_band_raw_sigma = np.empty_like(resolved.sigma_band)
    strong_reference_imaginary_leakage = np.empty((nbands,), dtype=float)

    energy_grid_Ha = energy_grid / ctx.state.data.energy_conversion_disk_to_working
    chemical_potential_J = mu * calc.unit_context.energy.scale_to_si
    phase_ai_bohr = np.eye(2, dtype=float)

    for n in range(nbands):
        epsilon_Ha = energy_grid_Ha[:, :, n]
        fh_sigma = resolved.sigma_band[n]
        fh_trace = float(np.trace(fh_sigma).real)

        strong_reference = band_indexed_strong_dc_from_velocity_grid(
            epsilon_Ha,
            velocity_grid[:, :, n, :],
            phase_ai_bohr,
            chemical_potential_J=chemical_potential_J,
            temperature_K=temperature,
            relaxation_time_s=tau,
            electric_field_V_per_m=np.zeros(2, dtype=float),
        )
        strong_raw_sigma = strong_reference.conductivity_tensor_S
        strong_sigma = strong_raw_sigma.real / ((2.0 * np.pi) ** 2)
        strong_reference_band_raw_sigma[n] = strong_raw_sigma
        strong_reference_band_sigma[n] = strong_sigma
        strong_reference_imaginary_leakage[n] = strong_reference.imaginary_leakage_S

        strong_trace = float(np.trace(strong_sigma).real)
        weak_trace = fh_trace
        strong_minus_weak = strong_trace - weak_trace
        strong_over_weak = strong_trace / weak_trace if weak_trace != 0.0 else np.nan
        tensor_delta = strong_sigma - fh_sigma.real
        tensor_relative_error = (
            float(np.linalg.norm(tensor_delta) / np.linalg.norm(fh_sigma.real))
            if np.linalg.norm(fh_sigma.real) != 0.0
            else np.nan
        )

        strong_reference_rows.append(
            TableRow((
                n,
                conductivity_quantity(strong_sigma[0, 0], "strong spectral sigma_xx"),
                conductivity_quantity(strong_sigma[0, 1], "strong spectral sigma_xy"),
                conductivity_quantity(strong_sigma[1, 0], "strong spectral sigma_yx"),
                conductivity_quantity(strong_sigma[1, 1], "strong spectral sigma_yy"),
                conductivity_quantity(strong_trace, "strong spectral trace"),
                conductivity_quantity(weak_trace, "weak compact trace"),
                conductivity_quantity(strong_minus_weak, "strong minus weak trace"),
                unitless_quantity(strong_over_weak, "strong / weak trace"),
                unitless_quantity(tensor_relative_error, "relative tensor discrepancy"),
                conductivity_quantity(strong_reference.imaginary_leakage_S / ((2.0 * np.pi) ** 2), "strong spectral imaginary leakage"),
            ))
        )


    return DiagnosticResult(
        title="Symbol method internal consistency",
        summary=(
            "Internal diagnostic for the group-symbol construction: local symbols, "
            "generalized Hellmann-Feynman velocities, weak conductivity, and strong spectral conductivity."
        ),
        cards=(
            Card("domain", "transport.boltzmann.group_resolved", "ok"),
            Card("scope", "internal symbol method", "ok"),
            Card("kernel", kernel_choice, "ok"),
            Card("bands", resolved.sigma_band.shape[0], "ok"),
            Card("||sum bands - compact||", card_quantity(conductivity_quantity(np.linalg.norm(residual), "sum bands minus compact norm")), "ok"),
        ),
        sections=(
            DiagnosticSection(
                id="symbol_sanity_checks",
                title="Symbol sanity checks",
                description=(
                    "Checks the local symbols before solving any eigenproblem. "
                    "The Hamiltonian and overlap symbols should be Hermitian on the sampled irreps, "
                    "and the overlap symbol should remain positive."
                ),
                tables=(
                    Table(
                        id="symbol_sanity_checks_table",
                        title="H(k) and S(k) sanity checks",
                        description="Global maxima and means over the sampled irrep grid.",
                        headers=("quantity", "worst", "mean", "meaning"),
                        rows=symbol_sanity_rows,
                        numeric=frozenset((1, 2)),
                    ),
                ),
            ),
            DiagnosticSection(
                id="band_energy_surfaces",
                title="Band energy surfaces",
                description="Band energy scalar fields on the conductivity grid.",
                body=(
                    HtmlBlock(
                        id="group_resolved_controls",
                        title="Band readout",
                        html=f"""<dft-kpoint-readout></dft-kpoint-readout>
""",
                    ),
                    WebGLView(
                        id="group_resolved_band_surface",
                        title="Band energy surface",
                        description=(
                            "Real band energies on the conductivity grid. "
                            "This viewer is model-island backed, so Datastar reruns patch data without replacing the component."
                        ),
                        renderer="region_surface",
                        payload=surface_payload,
                        interaction_channel="group_resolved_band_surface",
                    ),
                ),
            ),
            DiagnosticSection(
                id="hellmann_feynman_velocity_checks",
                title="Hellmann-Feynman velocity checks",
                description=(
                    "Recomputes diagonal generalized Hellmann-Feynman velocities from "
                    "symbol derivatives, energies, and eigenvectors, then compares them "
                    "with the stored velocity array."
                ),
                tables=(
                    Table(
                        id="hellmann_feynman_velocity_checks_table",
                        title="Velocity recomputation checks",
                        description="Checks over every sampled irrep, direction, and band.",
                        headers=("quantity", "worst", "mean", "meaning"),
                        rows=velocity_recompute_rows,
                        numeric=frozenset((1, 2)),
                    ),
                ),
            ),
            DiagnosticSection(
                id="band_velocity_surfaces",
                title="Band velocity surfaces",
                description="Physical velocity scalar fields from the symbol/Hellmann-Feynman Boltzmann calculation.",
                body=(
                    WebGLView(
                        id="group_resolved_band_velocity_surface",
                        title="Band velocity surface",
                        description=(
                            "Velocity x, velocity y, and speed on the same k-space grid. "
                            "The default view is speed; use the quantity dropdown to inspect components."
                        ),
                        renderer="region_surface",
                        payload=velocity_surface_payload,
                        interaction_channel="group_resolved_band_velocity_surface",
                    ),
                ),
            ),
            DiagnosticSection(
                id="band_velocity_summary",
                title="Band velocity summary",
                description="Physical velocity statistics from the symbol/Hellmann-Feynman Boltzmann calculation.",
                tables=(
                    Table(
                        id="band_velocity_summary_table",
                        title="Velocity min / mean / max by band",
                        description="Velocity components and speed on the displayed conductivity grid.",
                        headers=(
                            "band",
                            "min vx",
                            "mean vx",
                            "max vx",
                            "min vy",
                            "mean vy",
                            "max vy",
                            "min |v|",
                            "mean |v|",
                            "max |v|",
                        ),
                        rows=velocity_rows,
                        numeric=frozenset(range(10)),
                    ),
                ),
            ),
            DiagnosticSection(
                id="generalized_eigensystem_checks",
                title="Generalized eigensystem checks",
                description=(
                    "Checks the solved generalized eigenproblem before using eigenvectors in "
                    "Hellmann-Feynman velocities or conductivity weights."
                ),
                tables=(
                    Table(
                        id="generalized_eigensystem_residual_table",
                        title="Generalized eigensystem residuals",
                        description="Residuals over all sampled irreps and energy-ordered bands.",
                        headers=("quantity", "worst", "mean", "meaning"),
                        rows=eigensystem_sanity_rows,
                        numeric=frozenset((1, 2)),
                    ),
                    Table(
                        id="generalized_eigensystem_energy_table",
                        title="Energy range by band",
                        description="Band energy ranges over the sampled irrep grid.",
                        headers=("band", "min E", "mean E", "max E"),
                        rows=eigensystem_energy_rows,
                        numeric=frozenset((0, 1, 2, 3)),
                    ),
                ),
            ),
            DiagnosticSection(
                id="conductivity_assembly_checks",
                title="Conductivity assembly checks",
                description=(
                    "Checks that the compact weak-chain conductivity is reproduced by "
                    "the explicit band-resolved and k-resolved accumulated pieces."
                ),
                tables=(
                    Table(
                        id="conductivity_assembly_checks_table",
                        title="Weak conductivity assembly checks",
                        description="Residuals for assembling the compact weak conductivity tensor.",
                        headers=("quantity", "norm", "trace / ratio", "meaning"),
                        rows=conductivity_assembly_rows,
                        numeric=frozenset((1, 2)),
                    ),
                ),
            ),
            DiagnosticSection(
                id="strong_spectral_dc_reference",
                title="Strong spectral DC reference",
                description=(
                    "Per-band zero-field strong spectral conductivity using the shared spectral modal formula. "
                    "The raw modal tensor is divided by (2π)^2 to put it in the same continuum conductivity convention "
                    "as the compact weak-chain tensor.  This is the reference tensor that future band-free formulas "
                    "should reproduce."
                ),
                tables=(
                    Table(
                        id="strong_spectral_dc_reference_table",
                        title="Strong spectral zero-field reference by band",
                        description=(
                            "Strong/modal spectral tensor compared against the current compact Hellmann-Feynman weak-chain tensor."
                        ),
                        headers=(
                            "band",
                            "strong sigma xx",
                            "strong sigma xy",
                            "strong sigma yx",
                            "strong sigma yy",
                            "strong trace",
                            "weak trace",
                            "strong trace - weak trace",
                            "strong trace / weak trace",
                            "relative tensor discrepancy",
                            "imaginary leakage",
                        ),
                        rows=tuple(strong_reference_rows),
                        numeric=frozenset(range(11)),
                    ),
                ),
            ),
            DiagnosticSection(
                id="selected_band_tensor",
                title=f"Selected energy-ordered band {band}",
                description="Compact per-band tensor for the selected energy-ordered band.",
                tables=(
                    Table(
                        id="selected_band_tensor_table",
                        title="Selected band tensor",
                        description="Compact per-band conductivity tensor entries.",
                        headers=("entry", "real", "imag", "abs"),
                        rows=tuple(
                            TableRow((
                                f"sigma_{a}{b}",
                                conductivity_quantity(resolved.sigma_band[band, a, b].real, f"sigma_{a}{b} real"),
                                conductivity_quantity(resolved.sigma_band[band, a, b].imag, f"sigma_{a}{b} imag"),
                                conductivity_quantity(abs(resolved.sigma_band[band, a, b]), f"sigma_{a}{b} abs"),
                            ))
                            for a in range(resolved.sigma_band.shape[1])
                            for b in range(resolved.sigma_band.shape[2])
                        ),
                    ),
                ),
            ),
            DiagnosticSection(
                id="band_traces",
                title="Band trace contributions",
                description="Trace of each per-band compact tensor.",
                tables=(
                    Table(
                        id="band_trace_table",
                        title="Band trace contributions",
                        description="Trace of each per-band compact tensor.",
                        headers=("band", "trace real", "trace imag", "|trace|"),
                        rows=tuple(
                            TableRow((
                                n,
                                conductivity_quantity(np.trace(resolved.sigma_band[n]).real, f"band {n} trace real"),
                                conductivity_quantity(np.trace(resolved.sigma_band[n]).imag, f"band {n} trace imag"),
                                conductivity_quantity(abs(np.trace(resolved.sigma_band[n])), f"band {n} trace abs"),
                            ))
                            for n in range(resolved.sigma_band.shape[0])
                        ),
                    ),
                ),
            ),
        ),
    )


def diagnostics() -> tuple[DiagnosticSpec, ...]:
    return (
        DiagnosticSpec(
            id="transport.boltzmann.group_resolved.overview",
            title="Symbol method internal consistency",
            group="transport.boltzmann.group_resolved",
            description="Internal consistency checks for the group-symbol conductivity construction.",
            inputs=(
                InputSpec("kernel", "kernel", "select", "average_star", options=(("average_star", "average star"), ("average", "average"), ("anchored", "anchored"))),
                InputSpec("nu", "nu", "int", 5),
                InputSpec("nv", "nv", "int", 5),
                InputSpec("band", "band", "int", 0),
                InputSpec("temperature", "temperature", "float", 300.0),
                InputSpec("mu", "mu", "float", 0.0),
                InputSpec("tau", "tau", "float", 1.0),
            ),
            compute=compute_overview,
        ),
    )
