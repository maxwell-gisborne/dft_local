"""Diagnostics for group-resolved Boltzmann conductivity."""

from __future__ import annotations

import numpy as np

from dft_local.core.units import DisplayQuantity, VELOCITY
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
