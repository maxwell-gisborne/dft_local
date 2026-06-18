"""Regional scalar-band versus symbol-method conductivity diagnostics."""

from __future__ import annotations

import numpy as np

from dft_local.core.units import CONDUCTIVITY, DIMENSIONLESS, ENERGY, DisplayQuantity
from dft_local.diagnostics.models import Card, DiagnosticResult, DiagnosticSection, DiagnosticSpec, InputSpec, Table, TableRow
from dft_local.transport.boltzmann.calculation.core import BoltzmannConductivity
from dft_local.transport.boltzmann.calculation.diagnostics import conductivity_grid
from dft_local.transport.boltzmann.group_resolved.core import band_resolved_compact_conductivity


def _unitless(calc: BoltzmannConductivity, value: complex | float, name: str) -> DisplayQuantity:
    return DisplayQuantity(
        value=float(np.real(value)),
        dimension=DIMENSIONLESS,
        unit=calc.unit_context.unit_for_dimension(DIMENSIONLESS),
        name=name,
    )


def _energy(calc: BoltzmannConductivity, value: complex | float, name: str) -> DisplayQuantity:
    return DisplayQuantity(
        value=float(np.real(value)),
        dimension=ENERGY,
        unit=calc.unit_context.unit_for_dimension(ENERGY),
        name=name,
    )


def _conductivity(calc: BoltzmannConductivity, value: complex | float, name: str) -> DisplayQuantity:
    return DisplayQuantity(
        value=float(np.real(value)),
        dimension=CONDUCTIVITY,
        unit=calc.unit_context.unit_for_dimension(CONDUCTIVITY),
        name=name,
    )


def _region_mask_from_center(k1_grid: np.ndarray, k2_grid: np.ndarray, center: tuple[int, int], radius: int) -> np.ndarray:
    rows, cols = np.indices(k1_grid.shape)
    i0, j0 = center
    return (np.abs(rows - i0) <= radius) & (np.abs(cols - j0) <= radius)


def _normalised_region_weights(mask: np.ndarray) -> np.ndarray:
    weights = mask.astype(float)
    total = float(np.sum(weights))
    if total == 0.0:
        return weights
    return weights / total


def compute_regions(ctx, inputs: dict[str, object]) -> DiagnosticResult:
    kernel_choice = str(inputs.get("kernel", "average_star"))
    nu = int(inputs.get("nu", 9))
    nv = int(inputs.get("nv", 9))
    band = int(inputs.get("band", 0))
    mu = float(inputs.get("mu", 0.0))
    temperature = float(inputs.get("temperature", 300.0))
    tau = float(inputs.get("tau", 1.0))
    radius = int(inputs.get("radius", 1))

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
        name=f"regional scalar-band comparison {kernel_choice}",
    ).run()

    assert calc.energies is not None
    assert calc.velocities is not None
    assert calc.ac_weights is not None
    assert calc.sigma is not None
    assert calc.sigma_k is not None

    resolved = band_resolved_compact_conductivity(
        velocities=calc.velocities,
        weights=calc.ac_weights,
        physical_k_weights=calc.physical_k_weights,
    )

    nbands = int(calc.energies.shape[1])
    band = max(0, min(band, nbands - 1))

    k1_grid = np.asarray(k1, dtype=float).reshape(nu, nv)
    k2_grid = np.asarray(k2, dtype=float).reshape(nu, nv)
    energy_grid = np.asarray(calc.energies, dtype=float).reshape(nu, nv, nbands)
    sigma_k_grid = np.asarray(calc.sigma_k).reshape(nu, nv, 2, 2)

    gaps = []
    for ik in range(calc.energies.shape[0]):
        e = np.sort(calc.energies[ik])
        if len(e) > 1:
            gaps.append(float(np.min(np.diff(e))))
        else:
            gaps.append(np.inf)
    gap_grid = np.asarray(gaps, dtype=float).reshape(nu, nv)

    near_index = tuple(int(x) for x in np.unravel_index(np.argmin(gap_grid), gap_grid.shape))
    centre_index = (nu // 2, nv // 2)

    regions = (
        ("centre region", centre_index),
        ("minimum-gap region", near_index),
    )

    rows = []
    for name, center in regions:
        mask = _region_mask_from_center(k1_grid, k2_grid, center, radius)
        region_weights = _normalised_region_weights(mask)
        region_sigma = np.einsum("ij,ijab->ab", region_weights, sigma_k_grid)
        region_energy = energy_grid[:, :, band][mask]
        region_gap = gap_grid[mask]

        rows.append(TableRow((
            name,
            f"{center[0]}, {center[1]}",
            int(np.sum(mask)),
            _energy(calc, np.min(region_energy), f"{name} minimum selected-band energy"),
            _energy(calc, np.max(region_energy), f"{name} maximum selected-band energy"),
            _energy(calc, np.min(region_gap), f"{name} minimum local band gap"),
            _conductivity(calc, np.trace(region_sigma), f"{name} symbol trace"),
        )))

    return DiagnosticResult(
        title="Ashcroft versus symbol regions",
        summary=(
            "Regional diagnostic for comparing scalar-band/Ashcroft-style formulas "
            "against symbol-method conductivity on selected k-space patches."
        ),
        cards=(
            Card("domain", "transport.boltzmann.ashcroft_vs_band_free", "ok"),
            Card("kernel", kernel_choice, "ok"),
            Card("band", band, "ok"),
            Card("regions", len(regions), "ok"),
            Card("min gap", _energy(calc, np.min(gap_grid), "minimum sampled band gap"), "ok"),
            Card("||sum bands - compact||", _conductivity(calc, np.linalg.norm(resolved.sigma - calc.sigma), "band sum residual"), "ok"),
        ),
        sections=(
            DiagnosticSection(
                id="ashcroft_vs_symbol_region_summary",
                title="Regional summary",
                description=(
                    "First pass: locate a central patch and a minimum-gap patch. "
                    "The scalar-band comparison will be added on top of these selected regions."
                ),
                tables=(
                    Table(
                        id="ashcroft_vs_symbol_region_summary_table",
                        title="Selected k-space regions",
                        description="Region bookkeeping and symbol-method conductivity trace.",
                        headers=("region", "center index", "samples", "min E", "max E", "min local gap", "symbol trace"),
                        rows=tuple(rows),
                        numeric=frozenset((2, 3, 4, 5, 6)),
                    ),
                ),
            ),
        ),
    )


def diagnostics() -> tuple[DiagnosticSpec, ...]:
    return (
        DiagnosticSpec(
            id="transport.boltzmann.ashcroft_vs_band_free.regions",
            title="Ashcroft versus symbol regions",
            group="transport.boltzmann.ashcroft_comparison",
            description="Regional scalar-band/Ashcroft comparison against symbol-method conductivity.",
            inputs=(
                InputSpec("kernel", "kernel", "select", "average_star", options=(("average_star", "average star"), ("average", "average"), ("anchored", "anchored"))),
                InputSpec("nu", "nu", "number", 9),
                InputSpec("nv", "nv", "number", 9),
                InputSpec("band", "band", "number", 0),
                InputSpec("radius", "region radius", "number", 1),
                InputSpec("mu", "mu", "number", 0.0),
                InputSpec("temperature", "temperature", "number", 300.0),
                InputSpec("tau", "tau", "number", 1.0),
            ),
            compute=compute_regions,
        ),
    )
