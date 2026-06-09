"""Diagnostics for group-resolved Boltzmann conductivity."""

from __future__ import annotations

import numpy as np

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

    surface_payload = {
        "kind": "band-surface-preview",
        "nu": nu,
        "nv": nv,
        "k1": np.asarray(k1, dtype=float).reshape(nu, nv).tolist(),
        "k2": np.asarray(k2, dtype=float).reshape(nu, nv).tolist(),
        "energies": np.asarray(calc.energies, dtype=float).reshape(
            nu,
            nv,
            resolved.sigma_band.shape[0],
        ).tolist(),
        "mask": (
            (np.abs(np.asarray(k1, dtype=float).reshape(nu, nv)) <= np.pi + 1e-12)
            & (np.abs(np.asarray(k2, dtype=float).reshape(nu, nv)) <= np.pi + 1e-12)
            & (
                np.abs(
                    np.asarray(k1, dtype=float).reshape(nu, nv)
                    - np.asarray(k2, dtype=float).reshape(nu, nv)
                )
                <= np.pi + 1e-12
            )
        ).tolist(),
        "bands": list(range(int(resolved.sigma_band.shape[0]))),
        "nbands": int(resolved.sigma_band.shape[0]),
        "selected_band": band,
        "energy_unit": calc.unit_context.energy.symbol,
    }

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
                id="interactive_band_controls",
                title="Interactive band controls",
                description="Reusable signal components used by later band surface and conductivity views.",
                body=(
                    HtmlBlock(
                        id="group_resolved_controls",
                        title="Band controls",
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
