"""Diagnostics for the Boltzmann conductivity domain.

Diagnostics live next to the domain code.  The generic diagnostics loader
discovers this module by calling `diagnostics()`.  There is no import-time
registration side effect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dft_local.diagnostics.models import (
    Card,
    DiagnosticResult,
    DiagnosticSpec,
    Graph2D,
    GraphPoint,
    GraphSeries,
    InputSpec,
    Matrix,
    MatrixCell,
    Table,
    TableRow,
)
from dft_local.transport.boltzmann.calculation.core import BoltzmannConductivity


def conductivity_grid(
    *,
    nu: int,
    nv: int,
    central_bz: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flat raw irrep samples and raw integration weights."""

    k1_axis = np.linspace(-np.pi, np.pi, nu)
    k2_axis = np.linspace(-np.pi, np.pi, nv)

    dk1 = float(k1_axis[1] - k1_axis[0]) if nu > 1 else 2.0 * np.pi
    dk2 = float(k2_axis[1] - k2_axis[0]) if nv > 1 else 2.0 * np.pi

    K1, K2 = np.meshgrid(k1_axis, k2_axis, indexing="ij")

    if central_bz:
        mask = (
            (np.abs(K1) <= np.pi + 1e-12)
            & (np.abs(K2) <= np.pi + 1e-12)
            & (np.abs(K1 - K2) <= np.pi + 1e-12)
        )
    else:
        mask = np.ones_like(K1, dtype=bool)

    k1 = K1[mask].reshape(-1)
    k2 = K2[mask].reshape(-1)
    weights = np.full(k1.shape, dk1 * dk2, dtype=np.float64)

    return k1, k2, weights


def _domain_root() -> Path:
    return Path(__file__).resolve().parent


def _rows_table(
    *,
    id: str,
    title: str,
    description: str,
    headers: tuple[str, ...],
    rows: list[list[object]],
    numeric: set[int] | None = None,
) -> Table:
    return Table(
        id=id,
        title=title,
        description=description,
        headers=headers,
        rows=tuple(TableRow(tuple(row)) for row in rows),
        numeric=frozenset(numeric or set()),
    )


def sigma_matrix(calc: BoltzmannConductivity) -> Matrix:
    calc.require_solved()
    assert calc.sigma is not None

    d = calc.sigma.shape[0]
    cells: list[MatrixCell] = []

    for i in range(d):
        for j in range(d):
            z = calc.sigma[i, j]
            cells.append(
                MatrixCell(
                    i=i,
                    j=j,
                    value={
                        "real": float(z.real),
                        "imag": float(z.imag),
                        "abs": float(abs(z)),
                    },
                    entity_id=f"sigma:{i}:{j}",
                )
            )

    return Matrix(
        id="sigma",
        title="Integrated conductivity matrix",
        description="Final integrated AC Boltzmann conductivity tensor",
        row_labels=tuple(f"i={i}" for i in range(d)),
        col_labels=tuple(f"j={j}" for j in range(d)),
        cells=tuple(cells),
    )


def sigma_rows(calc: BoltzmannConductivity) -> list[list[object]]:
    calc.require_solved()
    assert calc.sigma is not None

    rows: list[list[object]] = []

    for i in range(calc.sigma.shape[0]):
        for j in range(calc.sigma.shape[1]):
            z = calc.sigma[i, j]
            rows.append([i, j, float(z.real), float(z.imag), float(abs(z))])

    return rows


def unit_rows(calc: BoltzmannConductivity) -> list[list[object]]:
    kBT = 3.166811563e-6 * calc.units.E * calc.temperature

    return [
        ["units", repr(calc.units)],
        ["units.E", calc.units.E],
        ["units.L", calc.units.L],
        ["units.e", calc.units.e],
        ["units.hbar", calc.units.hbar],
        ["mu", calc.mu],
        ["temperature / K", calc.temperature],
        ["k_B T in energy units", kBT],
        ["omega", calc.omega],
        ["sum raw irrep weights", float(np.sum(calc.irrep_weights))],
        ["sum physical k weights", float(np.sum(calc.physical_k_weights))],
        ["irrep to physical k", calc.irrep_to_physical_k.tolist()],
    ]


def worst_sample_rows(calc: BoltzmannConductivity, *, n: int) -> list[list[object]]:
    calc.require_solved()
    assert calc.energies is not None
    assert calc.velocities is not None
    assert calc.ac_weights is not None
    assert calc.sigma_k is not None

    raw_k = calc.irrep_points
    physical_k = calc.physical_k_points

    sigma_abs = np.linalg.norm(calc.sigma_k.reshape(calc.nk, -1), axis=1)
    vmax = np.max(np.abs(calc.velocities), axis=(1, 2))
    wmax = np.max(np.abs(calc.ac_weights), axis=1)

    order = np.argsort(-sigma_abs)[:n]

    return [
        [
            int(s),
            float(raw_k[s, 0]),
            float(raw_k[s, 1]),
            float(physical_k[s, 0]),
            float(physical_k[s, 1]),
            float(np.min(calc.energies[s])),
            float(np.max(calc.energies[s])),
            float(vmax[s]),
            float(wmax[s]),
            float(sigma_abs[s]),
            float(calc.physical_k_weights[s]),
        ]
        for s in order
    ]


def near_fermi_rows(calc: BoltzmannConductivity, *, n: int) -> list[list[object]]:
    calc.require_solved()
    assert calc.energies is not None
    assert calc.velocities is not None
    assert calc.ac_weights is not None

    raw_k = calc.irrep_points
    physical_k = calc.physical_k_points

    rows: list[list[object]] = []

    for sample in range(calc.nk):
        for band in range(calc.energies.shape[1]):
            E = float(calc.energies[sample, band])
            rows.append(
                [
                    int(sample),
                    int(band),
                    float(raw_k[sample, 0]),
                    float(raw_k[sample, 1]),
                    float(physical_k[sample, 0]),
                    float(physical_k[sample, 1]),
                    E,
                    abs(E - float(calc.mu)),
                    float(calc.velocities[sample, 0, band]),
                    float(calc.velocities[sample, 1, band]),
                    float(abs(calc.ac_weights[sample, band])),
                ]
            )

    rows.sort(key=lambda row: row[7])
    return rows[:n]


def quantile_rows_by_band(values: np.ndarray) -> list[list[object]]:
    rows: list[list[object]] = []

    for band in range(values.shape[1]):
        x = values[:, band]
        rows.append(
            [
                band,
                float(np.min(x)),
                float(np.quantile(x, 0.25)),
                float(np.median(x)),
                float(np.quantile(x, 0.75)),
                float(np.max(x)),
            ]
        )

    return rows


def velocity_quantile_rows(calc: BoltzmannConductivity) -> list[list[object]]:
    calc.require_solved()
    assert calc.velocities is not None

    rows: list[list[object]] = []

    for direction in range(calc.dimension):
        x = np.abs(calc.velocities[:, direction, :]).reshape(-1)
        rows.append(
            [
                direction,
                float(np.min(x)),
                float(np.quantile(x, 0.25)),
                float(np.median(x)),
                float(np.quantile(x, 0.75)),
                float(np.max(x)),
            ]
        )

    return rows


def sigma_graph(calc: BoltzmannConductivity) -> Graph2D:
    calc.require_solved()
    assert calc.sigma_k is not None

    series: list[GraphSeries] = []
    x = np.arange(calc.nk, dtype=float)

    for i in range(calc.dimension):
        for j in range(calc.dimension):
            y = np.abs(calc.sigma_k[:, i, j])
            series.append(
                GraphSeries(
                    name=f"|sigma_{i}{j}(k)|",
                    points=tuple(
                        GraphPoint(
                            x=float(x[s]),
                            y=float(y[s]),
                            entity_id=f"sample:{s}",
                            label=str(s),
                            meta={"i": i, "j": j},
                        )
                        for s in range(calc.nk)
                    ),
                    kind="line",
                )
            )

    return Graph2D(
        id="sigma_k",
        title="k-resolved conductivity contribution",
        description="Magnitude of sigma_ij(k) over sample index",
        x_label="sample index",
        y_label="|sigma_ij(k)|",
        series=tuple(series),
    )


def compute_conductivity(ctx: Any, inputs: dict[str, object]) -> DiagnosticResult:
    state = ctx.state

    nu = int(inputs["nu"])
    nv = int(inputs["nv"])
    mu = float(inputs["mu"])
    temperature = float(inputs["temperature"])
    tau = float(inputs["tau"])
    omega = float(inputs["omega"])
    central_bz = bool(inputs["central_bz"])
    rows_shown = int(inputs["rows"])
    kernel_choice = str(inputs["kernel"])

    k_scale_input = float(inputs["k_scale"])
    k_scale = (1.0 / state.units.L) if k_scale_input == 0.0 else k_scale_input

    KH, KS = ctx.kernels(kernel_choice)

    k1, k2, weights = conductivity_grid(
        nu=nu,
        nv=nv,
        central_bz=central_bz,
    )

    calc = BoltzmannConductivity.from_arrays(
        KH,
        KS,
        k1,
        k2,
        irrep_weights=weights,
        irrep_to_physical_k=k_scale * np.eye(2),
        units=state.units,
        mu=mu,
        temperature=temperature,
        omega=omega,
        tau=tau,
        name=f"diagnostic conductivity {kernel_choice}",
    ).run()

    diag = calc.diagnostics()

    finite = bool(diag["finite"])
    vmax = float(diag["velocity_abs_max"])

    cards = (
        Card("samples", diag["nk"], "ok", "Number of k samples"),
        Card("bands", diag["nbands"], "ok", "Generalized eigenvalues per sample"),
        Card("finite", finite, "ok" if finite else "bad"),
        Card("|sigma|", diag["sigma_norm"], "ok" if finite else "bad"),
        Card("max |v|", vmax, "warn" if vmax > 1e20 else "ok"),
        Card("max |w|", diag["ac_weight_abs_max"], "neutral"),
        Card("kernel", kernel_choice, "neutral"),
        Card("k scale", k_scale, "neutral"),
    )

    assert calc.energies is not None
    assert calc.ac_weights is not None

    tables = (
        _rows_table(
            id="sigma_entries",
            title="Sigma entries",
            description="Integrated conductivity tensor entries",
            headers=("i", "j", "real", "imag", "abs"),
            rows=sigma_rows(calc),
            numeric={0, 1, 2, 3, 4},
        ),
        _rows_table(
            id="units",
            title="Units and integration measure",
            description="Unit and k-space measure values used by the calculation",
            headers=("quantity", "value"),
            rows=unit_rows(calc),
            numeric={1},
        ),
        _rows_table(
            id="worst_samples",
            title="Largest k-resolved conductivity samples",
            description="Samples sorted by norm of sigma_ij(k)",
            headers=(
                "sample",
                "k1 raw",
                "k2 raw",
                "k1 physical",
                "k2 physical",
                "E min",
                "E max",
                "max |v|",
                "max |w|",
                "||sigma_k||",
                "physical weight",
            ),
            rows=worst_sample_rows(calc, n=rows_shown),
            numeric=set(range(11)),
        ),
        _rows_table(
            id="near_fermi",
            title="States closest to chemical potential",
            description="States sorted by |E - mu|",
            headers=(
                "sample",
                "band",
                "k1 raw",
                "k2 raw",
                "k1 physical",
                "k2 physical",
                "E",
                "|E - mu|",
                "v0",
                "v1",
                "|w|",
            ),
            rows=near_fermi_rows(calc, n=rows_shown),
            numeric=set(range(11)),
        ),
        _rows_table(
            id="energy_quantiles",
            title="Energy quantiles by sorted band",
            description="Energy distribution over sampled k points",
            headers=("band", "min", "q25", "median", "q75", "max"),
            rows=quantile_rows_by_band(calc.energies),
            numeric={0, 1, 2, 3, 4, 5},
        ),
        _rows_table(
            id="velocity_quantiles",
            title="Velocity magnitude quantiles",
            description="Absolute velocity distribution by direction",
            headers=("direction", "min", "q25", "median", "q75", "max"),
            rows=velocity_quantile_rows(calc),
            numeric={0, 1, 2, 3, 4, 5},
        ),
        _rows_table(
            id="weight_quantiles",
            title="AC weight magnitude quantiles",
            description="AC Boltzmann weight distribution by sorted band",
            headers=("band", "min", "q25", "median", "q75", "max"),
            rows=quantile_rows_by_band(np.abs(calc.ac_weights)),
            numeric={0, 1, 2, 3, 4, 5},
        ),
    )

    return DiagnosticResult(
        title="Boltzmann AC conductivity",
        summary=(
            "Band-diagonal AC Boltzmann conductivity from generalized symbols. "
            "Each k sample is solved independently; no band continuation is used."
        ),
        body=(
            *cards,
            sigma_matrix(calc),
            sigma_graph(calc),
            *tables,
        ),
        notes=(
            "This is the semiclassical band-diagonal Boltzmann expression.",
            "The calculation differentiates symbols with respect to physical k, using k_scale to convert raw irrep coordinates.",
            "Use k_scale=0 for the automatic default 1 / units.L.",
        ),
    )


def compute_overview(ctx: Any, inputs: dict[str, object]) -> DiagnosticResult:
    del ctx, inputs

    root = _domain_root()
    docs_path = root / "docs.md"
    tests_path = root / "tests.py"
    docs_text = docs_path.read_text(encoding="utf-8") if docs_path.exists() else ""

    cards = (
        Card("domain", "transport.boltzmann.calculation", "ok"),
        Card("docs", "present" if docs_path.exists() else "missing", "ok" if docs_path.exists() else "bad", str(docs_path)),
        Card("test metadata", "present" if tests_path.exists() else "missing", "ok" if tests_path.exists() else "warn", str(tests_path)),
    )

    tables = (
        Table(
            id="boltzmann_files",
            title="Domain files",
            description="Files owned by the Boltzmann conductivity domain module",
            headers=("role", "path", "exists"),
            rows=(
                TableRow(("package", str(root / "__init__.py"), (root / "__init__.py").exists())),
                TableRow(("core", str(root / "core.py"), (root / "core.py").exists())),
                TableRow(("documentation", str(docs_path), docs_path.exists())),
                TableRow(("diagnostics", str(root / "diagnostics.py"), (root / "diagnostics.py").exists())),
                TableRow(("test metadata", str(tests_path), tests_path.exists())),
            ),
        ),
        Table(
            id="boltzmann_docs_preview",
            title="Documentation preview",
            description="First lines of the domain documentation",
            headers=("line", "text"),
            rows=tuple(TableRow((i + 1, line)) for i, line in enumerate(docs_text.splitlines()[:40])),
            numeric=frozenset({0}),
        ),
    )

    return DiagnosticResult(
        title="Boltzmann conductivity domain",
        summary="Overview of the domain-local Boltzmann conductivity module.",
        body=(
            *cards,
            *tables,
        ),
        notes=(
            "This diagnostic is discovered explicitly through diagnostics(), not by import-time registration.",
            "The domain owns its documentation, diagnostics, and test metadata locally.",
        ),
    )


def diagnostics() -> list[DiagnosticSpec]:
    return [
        DiagnosticSpec(
            id="transport.boltzmann.calculation.overview",
            group="transport",
            title="Boltzmann domain overview",
            description="Show documentation and local files for the Boltzmann conductivity domain.",
            inputs=(),
            compute=compute_overview,
            tier="instant",
        ),
        DiagnosticSpec(
            id="transport.boltzmann.calculation.conductivity",
            group="transport",
            title="Boltzmann conductivity",
            description="Compute band-diagonal AC Boltzmann conductivity.",
            inputs=(
                InputSpec("kernel", "kernel", "select", "average_star", options=(
                    ("anchored", "anchored"),
                    ("anchored_star", "anchored star"),
                    ("average", "average"),
                    ("average_star", "average star"),
                )),
                InputSpec("nu", "nu", "int", 31, min_value=3, max_value=251),
                InputSpec("nv", "nv", "int", 31, min_value=3, max_value=251),
                InputSpec("mu", "mu", "float", 0.0),
                InputSpec("temperature", "temperature / K", "float", 300.0, min_value=1e-12),
                InputSpec("tau", "tau", "float", 1.0, min_value=0.0),
                InputSpec("omega", "omega", "float", 0.0),
                InputSpec("k_scale", "k scale", "float", 0.0, help="0 means automatic 1 / units.L"),
                InputSpec("central_bz", "central BZ", "bool", True),
                InputSpec("rows", "rows shown", "int", 40, min_value=1, max_value=500),
            ),
            compute=compute_conductivity,
            tier="expensive",
        ),
    ]
