"""Diagnostics for group-resolved Boltzmann conductivity."""

from __future__ import annotations

import numpy as np

from dft_local.core.units import CONDUCTIVITY, DisplayQuantity, VELOCITY
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
from dft_local.transport.boltzmann.ashcroft_comparison.core import (
    conductivity_from_velocity_grid,
    velocity_from_epsilon_grid,
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

    # Finite-difference velocity of the displayed energy-ordered sheets.
    # This intentionally differs from calc.velocities, which uses the
    # generalized Hellmann-Feynman symbol derivative.  The finite-difference
    # version is useful for reproducing diagnostics that differentiate the
    # displayed band surface itself.
    dk1 = float(k1_grid[1, 0] - k1_grid[0, 0]) if nu > 1 else 1.0
    dk2 = float(k2_grid[0, 1] - k2_grid[0, 0]) if nv > 1 else 1.0
    fd_dE_dk1, fd_dE_dk2 = np.gradient(
        energy_grid,
        dk1,
        dk2,
        axis=(0, 1),
        edge_order=2 if min(nu, nv) >= 3 else 1,
    )
    fd_vx_grid = fd_dE_dk1 / calc.unit_context.hbar()
    fd_vy_grid = fd_dE_dk2 / calc.unit_context.hbar()
    fd_speed_grid = np.sqrt(fd_vx_grid * fd_vx_grid + fd_vy_grid * fd_vy_grid)

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

    finite_difference_velocity_surface_payload = {
        **common_surface_payload,
        "energies": energy_grid.tolist(),
        "fields": [
            {"id": "vx", "label": "FD velocity x", "unit": velocity_unit_symbol, "signed": True},
            {"id": "vy", "label": "FD velocity y", "unit": velocity_unit_symbol, "signed": True},
            {"id": "speed", "label": "FD speed", "unit": velocity_unit_symbol, "signed": False},
        ],
        "field_values": {
            "vx": fd_vx_grid.tolist(),
            "vy": fd_vy_grid.tolist(),
            "speed": fd_speed_grid.tolist(),
        },
        "selected_field": "speed",
    }

    def velocity_quantity(value: float, name: str) -> DisplayQuantity:
        return DisplayQuantity(
            value=float(value),
            dimension=VELOCITY,
            unit=velocity_unit,
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

    finite_difference_velocity_rows = tuple(
        TableRow((
            n,
            velocity_quantity(np.min(fd_vx_grid[:, :, n]), "min fd vx"),
            velocity_quantity(np.mean(fd_vx_grid[:, :, n]), "mean fd vx"),
            velocity_quantity(np.max(fd_vx_grid[:, :, n]), "max fd vx"),
            velocity_quantity(np.min(fd_vy_grid[:, :, n]), "min fd vy"),
            velocity_quantity(np.mean(fd_vy_grid[:, :, n]), "mean fd vy"),
            velocity_quantity(np.max(fd_vy_grid[:, :, n]), "max fd vy"),
            velocity_quantity(np.min(fd_speed_grid[:, :, n]), "min fd speed"),
            velocity_quantity(np.mean(fd_speed_grid[:, :, n]), "mean fd speed"),
            velocity_quantity(np.max(fd_speed_grid[:, :, n]), "max fd speed"),
        ))
        for n in range(nbands)
    )

    conductivity_unit = calc.unit_context.unit_for_dimension(CONDUCTIVITY)

    def conductivity_quantity(value: float, name: str) -> DisplayQuantity:
        return DisplayQuantity(
            value=float(value),
            dimension=CONDUCTIVITY,
            unit=conductivity_unit,
            name=name,
        )

    # Ashcroft validation convention:
    #   * epsilon grid is in Hartree
    #   * velocity helper differentiates a fractional reciprocal grid
    #   * conductivity helper assembles the SI Boltzmann tensor
    #
    # The group diagnostic grid is raw phase coordinates in [-pi, pi] with both
    # endpoints included.  Therefore use endpoint=True and an identity real-space
    # primitive in bohr: fractional u,v correspond to one 2*pi phase period.
    ashcroft_phase_ai_bohr = np.eye(2, dtype=float)
    energy_grid_Ha = energy_grid / ctx.state.data.energy_conversion_disk_to_working
    chemical_potential_J = mu * calc.unit_context.energy.scale_to_si

    ashcroft_validation_rows = []
    ashcroft_fd_vx = np.empty((nu, nv, nbands), dtype=float)
    ashcroft_fd_vy = np.empty((nu, nv, nbands), dtype=float)

    for n in range(nbands):
        epsilon_Ha = energy_grid_Ha[:, :, n]
        vx_m_s, vy_m_s = velocity_from_epsilon_grid(
            epsilon_Ha,
            ashcroft_phase_ai_bohr,
            endpoint=True,
        )
        ashcroft_fd_vx[:, :, n] = vx_m_s
        ashcroft_fd_vy[:, :, n] = vy_m_s

        ashcroft_sigma = conductivity_from_velocity_grid(
            epsilon_Ha,
            np.stack([vx_m_s, vy_m_s], axis=-1),
            ashcroft_phase_ai_bohr,
            chemical_potential_J=chemical_potential_J,
            temperature_K=temperature,
            relaxation_time_s=tau,
        ).conductivity_tensor_S

        fh_sigma = resolved.sigma_band[n]
        ashcroft_trace = float(np.trace(ashcroft_sigma).real)
        fh_trace = float(np.trace(fh_sigma).real)
        trace_delta = ashcroft_trace - fh_trace
        trace_ratio = ashcroft_trace / fh_trace if fh_trace != 0.0 else np.nan
        speed_m_s = np.sqrt(vx_m_s * vx_m_s + vy_m_s * vy_m_s)

        ashcroft_validation_rows.append(
            TableRow((
                n,
                velocity_quantity(np.min(vx_m_s), "Ashcroft FD min vx"),
                velocity_quantity(np.mean(vx_m_s), "Ashcroft FD mean vx"),
                velocity_quantity(np.max(vx_m_s), "Ashcroft FD max vx"),
                velocity_quantity(np.min(vy_m_s), "Ashcroft FD min vy"),
                velocity_quantity(np.mean(vy_m_s), "Ashcroft FD mean vy"),
                velocity_quantity(np.max(vy_m_s), "Ashcroft FD max vy"),
                velocity_quantity(np.min(speed_m_s), "Ashcroft FD min speed"),
                velocity_quantity(np.mean(speed_m_s), "Ashcroft FD mean speed"),
                velocity_quantity(np.max(speed_m_s), "Ashcroft FD max speed"),
                conductivity_quantity(ashcroft_sigma[0, 0], "Ashcroft sigma_xx"),
                conductivity_quantity(ashcroft_sigma[0, 1], "Ashcroft sigma_xy"),
                conductivity_quantity(ashcroft_sigma[1, 0], "Ashcroft sigma_yx"),
                conductivity_quantity(ashcroft_sigma[1, 1], "Ashcroft sigma_yy"),
                conductivity_quantity(ashcroft_trace, "Ashcroft trace"),
                conductivity_quantity(fh_sigma[0, 0].real, "FH sigma_xx"),
                conductivity_quantity(fh_sigma[0, 1].real, "FH sigma_xy"),
                conductivity_quantity(fh_sigma[1, 0].real, "FH sigma_yx"),
                conductivity_quantity(fh_sigma[1, 1].real, "FH sigma_yy"),
                conductivity_quantity(fh_trace, "FH trace"),
                conductivity_quantity(trace_delta, "Ashcroft minus FH trace"),
                f"{trace_ratio:.8e}",
            ))
        )

    return DiagnosticResult(
        title="Group-resolved Boltzmann conductivity",
        summary=(
            "Band-labelled compact DC conductivity decomposition. "
            "This page is the staging point for the lattice-index inverse-Fourier reconstruction."
        ),
        cards=(
            Card("domain", "transport.boltzmann.group_resolved", "ok"),
            Card("kernel", kernel_choice, "ok"),
            Card("bands", resolved.sigma_band.shape[0], "ok"),
            Card("||sum bands - compact||", f"{np.linalg.norm(residual):.3e}", "ok"),
        ),
        sections=(
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
                id="finite_difference_band_velocity_surfaces",
                title="Finite-difference band velocity surfaces",
                description=(
                    "Velocity fields obtained by finite-differencing the displayed energy-ordered band sheets. "
                    "This is useful for reproducing band-surface derivative diagnostics, but can differ from "
                    "projector/Hellmann-Feynman velocities near crossings or degeneracies."
                ),
                body=(
                    WebGLView(
                        id="group_resolved_fd_band_velocity_surface",
                        title="Finite-difference band velocity surface",
                        description=(
                            "Finite-difference velocity x, velocity y, and speed from E_n(k1,k2). "
                            "The default view is finite-difference speed."
                        ),
                        renderer="region_surface",
                        payload=finite_difference_velocity_surface_payload,
                        interaction_channel="group_resolved_fd_band_velocity_surface",
                    ),
                ),
            ),
            DiagnosticSection(
                id="finite_difference_band_velocity_summary",
                title="Finite-difference band velocity summary",
                description="Finite-difference velocity statistics by displayed energy-ordered band.",
                tables=(
                    Table(
                        id="finite_difference_band_velocity_summary_table",
                        title="Finite-difference velocity min / mean / max by band",
                        description="Finite-difference velocity components and speed on the displayed conductivity grid.",
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
                        rows=finite_difference_velocity_rows,
                        numeric=frozenset(range(10)),
                    ),
                ),
            ),
            DiagnosticSection(
                id="ashcroft_finite_difference_band_validation",
                title="Ashcroft finite-difference band validation",
                description=(
                    "Per-band validation using the same finite-difference velocity and conductivity assembly "
                    "as the Ashcroft comparison domain.  The displayed energy-ordered band sheet is converted "
                    "to Hartree, differentiated with endpoint=True over one phase period, then assembled with "
                    "Ashcroft conductivity_from_velocity_grid.  This table is meant to compare the band-sheet "
                    "finite-difference result against the Hellmann-Feynman band tensor."
                ),
                tables=(
                    Table(
                        id="ashcroft_finite_difference_band_validation_table",
                        title="Ashcroft finite-difference validation by band",
                        description=(
                            "Velocity statistics and conductivity tensor comparison for each displayed band."
                        ),
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
                            "A sigma xx",
                            "A sigma xy",
                            "A sigma yx",
                            "A sigma yy",
                            "A trace",
                            "FH sigma xx",
                            "FH sigma xy",
                            "FH sigma yx",
                            "FH sigma yy",
                            "FH trace",
                            "A trace - FH trace",
                            "A trace / FH trace",
                        ),
                        rows=tuple(ashcroft_validation_rows),
                        numeric=frozenset(range(22)),
                    ),
                ),
            ),
            DiagnosticSection(
                id="selected_band_tensor",
                title=f"Selected energy-ordered band {band}",
                description="Compact per-band tensor used as the Ashcroft-style comparison target.",
                tables=(
                    Table(
                        id="selected_band_tensor_table",
                        title="Selected band tensor",
                        description="Compact per-band conductivity tensor entries.",
                        headers=("entry", "real", "imag", "abs"),
                        rows=tuple(
                            TableRow((
                                f"sigma_{a}{b}",
                                f"{resolved.sigma_band[band, a, b].real:.8e}",
                                f"{resolved.sigma_band[band, a, b].imag:.8e}",
                                f"{abs(resolved.sigma_band[band, a, b]):.8e}",
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
                                f"{np.trace(resolved.sigma_band[n]).real:.8e}",
                                f"{np.trace(resolved.sigma_band[n]).imag:.8e}",
                                f"{abs(np.trace(resolved.sigma_band[n])):.8e}",
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
            title="Group-resolved Boltzmann conductivity",
            group="transport.boltzmann.group_resolved",
            description="Band-labelled compact target before lattice-index inverse-Fourier reconstruction.",
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
