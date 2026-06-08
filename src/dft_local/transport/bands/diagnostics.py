"""Diagnostics for the band/path continuation domain."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from dft_local.diagnostics.models import (
    Card,
    DiagnosticResult,
    DiagnosticSpec,
    Graph2D,
    GraphPoint,
    GraphSeries,
    InputSpec,
    Table,
    TableRow,
    WebGLView,
)
from dft_local.transport.bands.core import LocalPath, LocalRegion, bz_hexagon_vertices


KERNEL_OPTIONS = (
    ("average_star", "Average star"),
    ("average", "Average"),
    ("anchored_star", "Anchored star"),
    ("anchored", "Anchored"),
)

MATCHING_OPTIONS = (
    ("energy_predict", "Energy prediction"),
    ("state_overlap", "State overlap"),
    ("energy_order", "Energy ordering"),
)

PATH_OPTIONS = (
    ("gamma_k_m_gamma", "Γ-K-M-Γ"),
    ("circle_k", "Circle around K"),
    ("circle_m", "Circle around M"),
    ("primitive_cell", "Primitive cell boundary"),
    ("hexagon", "Full Brillouin-zone hexagon"),
)


def _domain_root() -> Path:
    return Path(__file__).resolve().parent


def _circle_path(
    *,
    centre_label: str,
    centre: tuple[float, float],
    radius: float,
    n: int = 48,
) -> list[tuple[str, float, float]]:
    angles = np.linspace(0.0, 2.0 * np.pi, n + 1)
    c1, c2 = centre

    points = [
        (
            f"{centre_label}+r" if i == 0 else "",
            float(c1 + radius * np.cos(theta)),
            float(c2 + radius * np.sin(theta)),
        )
        for i, theta in enumerate(angles)
    ]

    points[0] = (f"{centre_label} loop", points[0][1], points[0][2])
    points[-1] = (f"{centre_label} loop", points[-1][1], points[-1][2])
    return points


def _k_to_cart(k1: float, k2: float) -> tuple[float, float]:
    """Map reciprocal-basis coordinates to Cartesian plotting coordinates.

    The diagnostic path is parameterised by the two phase coordinates k1 and k2.
    For visualisation, draw those coordinates in an oblique reciprocal basis
    rather than as orthogonal screen axes.
    """

    return (
        float(k1 + 0.5 * k2),
        float((np.sqrt(3.0) / 2.0) * k2),
    )


def _path_preset(name: str) -> list[tuple[str, float, float]]:
    p = np.pi
    gamma = ("Γ", 0.0, 0.0)

    hex_vertices = bz_hexagon_vertices()
    k_vertex = hex_vertices[0]
    next_vertex = hex_vertices[1]

    # Use one BZ convention everywhere:
    # K is a hexagon vertex, M is the midpoint of an adjacent hexagon edge.
    k = ("K", float(k_vertex[0]), float(k_vertex[1]))
    m = (
        "M",
        0.5 * (float(k_vertex[0]) + float(next_vertex[0])),
        0.5 * (float(k_vertex[1]) + float(next_vertex[1])),
    )

    if name == "gamma_k_m_gamma":
        return [gamma, k, m, gamma]

    if name == "circle_k":
        return _circle_path(
            centre_label="K",
            centre=(k[1], k[2]),
            radius=0.18 * p,
        )

    if name == "circle_m":
        return _circle_path(
            centre_label="M",
            centre=(m[1], m[2]),
            radius=0.18 * p,
        )

    if name == "primitive_cell":
        return [
            ("0", 0.0, 0.0),
            ("b1", 2.0 * p, 0.0),
            ("b1+b2", 2.0 * p, 2.0 * p),
            ("b2", 0.0, 2.0 * p),
            ("0", 0.0, 0.0),
        ]

    if name == "hexagon":
        vertices = bz_hexagon_vertices()
        return [
            (f"H{i}", float(k1), float(k2))
            for i, (k1, k2) in enumerate(vertices)
        ] + [("H0", float(vertices[0][0]), float(vertices[0][1]))]

    raise ValueError(f"Unknown path preset: {name}")


def compute_overview(ctx, inputs: dict[str, object]) -> DiagnosticResult:
    del ctx, inputs

    root = _domain_root()

    files = (
        ("package", root / "__init__.py"),
        ("core", root / "core.py"),
        ("diagnostics", root / "diagnostics.py"),
        ("test metadata", root / "tests.py"),
        ("business tests", root / "test_continuation_business_logic.py"),
    )

    return DiagnosticResult(
        title="Band/path continuation domain",
        summary="Overview of the band continuation module.",
        body=(
            Card("domain", "transport.bands", "ok"),
            Card("business tests", "present", "ok"),
            Card("implementation", "local", "ok", "Core owns band/path/region continuation implementation"),
            Table(
                id="bands_files",
                title="Domain files",
                description="Files owned by the band/path continuation domain",
                headers=("role", "path", "exists"),
                rows=tuple(TableRow((role, str(path), path.exists())) for role, path in files),
            ),
            Table(
                id="bands_api",
                title="Public API currently exposed",
                description="Band/path continuation symbols exposed from transport.bands.core",
                headers=("name", "role"),
                rows=(
                    TableRow(("LocalPath", "path solving and band continuation")),
                    TableRow(("LocalRegion", "grid of local paths")),
                    TableRow(("match_via_energies", "energy-prediction band matching")),
                    TableRow(("match_via_overlap", "state-overlap band matching")),
                    TableRow(("align_groups_and_fix_gauge", "degenerate-subspace alignment and phase fixing")),
                    TableRow(("detect_energy_order_crossings_between_steps", "band event detection")),
                ),
            ),
        ),
        notes=("This diagnostic is discovered explicitly through diagnostics(), not by import-time registration.",),
    )


def compute_band_path(ctx, inputs: dict[str, object]) -> DiagnosticResult:
    kernel_choice = str(inputs["kernel"])
    matching_strategy = str(inputs["matching"])
    path_preset = str(inputs["path"])
    points_per_segment = int(inputs["points_per_segment"])

    KH, KS = ctx.kernels(kernel_choice)
    points = _path_preset(path_preset)
    path_label = dict(PATH_OPTIONS)[path_preset]

    path = LocalPath.from_points(
        KH,
        KS,
        points,
        unit_context=ctx.state.data.working_unit_context,
        points_per_segment=points_per_segment,
        matching_strategy=matching_strategy,
        name=f"{path_label} ({kernel_choice})",
    ).solve_continuation()

    x = np.asarray(path.x, dtype=float)
    energies = np.asarray(path.energies, dtype=float) * ctx.state.data.energy_conversion_disk_to_working

    series = tuple(
        GraphSeries(
            name=f"band {band}",
            points=tuple(
                GraphPoint(
                    x=float(x[i]),
                    y=float(energies[i, band]),
                    entity_id=f"band:{band}:point:{i}",
                    label=f"band {band}",
                    meta={
                        "band": band,
                        "step": i,
                        "k1": float(path.k1[i]),
                        "k2": float(path.k2[i]),
                    },
                )
                for i in range(len(x))
            ),
            kind="line",
        )
        for band in range(energies.shape[1])
    )

    label_rows = tuple(
        TableRow((label, int(i), float(x[int(i)]), float(path.k1[int(i)]), float(path.k2[int(i)])))
        for i, label in path.labels
    )

    event_rows = tuple(
        TableRow(
            (
                event.kind,
                event.step,
                event.band_a,
                event.band_b,
                event.x,
                event.energy * ctx.state.data.energy_conversion_disk_to_working,
                event.gap * ctx.state.data.energy_conversion_disk_to_working,
                event.comment,
            )
        )
        for event in path.band_events
    )

    degenerate_rows = tuple(
        TableRow(
            (
                event.step,
                ", ".join(str(b) for b in event.bands),
                event.energy_min * ctx.state.data.energy_conversion_disk_to_working,
                event.energy_max * ctx.state.data.energy_conversion_disk_to_working,
                event.gap * ctx.state.data.energy_conversion_disk_to_working,
                event.subspace_score,
                event.min_singular_value,
            )
        )
        for event in path.degenerate_group_events
    )

    primitive_basis = [
        ("0", 0.0, 0.0),
        ("b1", 2.0 * np.pi, 0.0),
        ("b1+b2", 2.0 * np.pi, 2.0 * np.pi),
        ("b2", 0.0, 2.0 * np.pi),
        ("0", 0.0, 0.0),
    ]

    hex_vertices = bz_hexagon_vertices()
    hex_points = [
        GraphPoint(float(k1), float(k2), label=f"H{i}")
        for i, (k1, k2) in enumerate(hex_vertices)
    ]
    hex_points.append(GraphPoint(float(hex_vertices[0][0]), float(hex_vertices[0][1]), label="H0"))

    # Keep reference markers tied to the actual selected path samples.
    # This avoids a second, hard-coded Γ/K/M convention drifting away from the
    # k-path used to compute the band diagram.
    if path_label == "Γ-K-M-Γ":
        segment = max(1, (len(path.k1) - 1) // 3)
        reference_indices = (
            ("Γ", 0),
            ("K", segment),
            ("M", 2 * segment),
        )
    else:
        reference_indices = (("Γ", 0),)

    reference_points = tuple(
        GraphPoint(
            float(path.k1[index]),
            float(path.k2[index]),
            label=label,
        )
        for label, index in reference_indices
        if 0 <= index < len(path.k1)
    )

    kspace_series = (
        GraphSeries(
            name="selected path",
            points=tuple(
                GraphPoint(
                    x=float(path.k1[i]),
                    y=float(path.k2[i]),
                    entity_id=f"kpoint:{i}",
                    label=f"{i}",
                    meta={"step": i, "x": float(x[i])},
                )
                for i in range(len(path.k1))
            ),
            kind="line_points",
        ),
        GraphSeries(
            name="primitive cell",
            points=tuple(GraphPoint(float(k1), float(k2), label=label) for label, k1, k2 in primitive_basis),
            kind="line",
        ),
        GraphSeries(
            name="hexagon",
            points=tuple(hex_points),
            kind="line",
        ),
        GraphSeries(
            name="Γ K M",
            points=reference_points,
            kind="points",
        ),
    )

    return DiagnosticResult(
        title=f"Band path {path_label}",
        summary="Solved band continuation along the selected path. Energies are plotted against cumulative path distance.",
        body=(
            Card("path", path_label, "ok"),
            Card("kernel", kernel_choice, "ok"),
            Card("matching", matching_strategy, "ok"),
            Card("k-points", len(x), "ok"),
            Card("bands", energies.shape[1], "ok"),
            Card("energy min", float(np.min(energies)), "neutral", f"units: {path.unit_context.energy.symbol}"),
            Card("energy max", float(np.max(energies)), "neutral", f"units: {path.unit_context.energy.symbol}"),
            Card("band events", len(path.band_events), "warn" if path.band_events else "ok"),
            Card("degenerate groups", len(path.degenerate_group_events), "warn" if path.degenerate_group_events else "ok"),
            Graph2D(
                id="kspace_path",
                title=f"K-space path: {path_label}",
                description="Selected path shown with Γ/K/M reference points, primitive-cell boundary, and hexagon boundary.",
                x_label="k cartesian x",
                y_label="k cartesian y",
                series=kspace_series,
                interaction_channel="kspace_path",
            ),
            Graph2D(
                id="band_path",
                title=f"Band energies along {path_label}",
                description="Each line is one continued band label.",
                x_label="path coordinate",
                y_label=f"energy / {path.unit_context.energy.symbol}",
                series=series,
                interaction_channel="band_path",
            ),
            Table(
                id="path_labels",
                title="High-symmetry path labels",
                description="Labelled points along the path.",
                headers=("label", "step", "x", "k1", "k2"),
                rows=label_rows,
                numeric=frozenset({1, 2, 3, 4}),
            ),
            Table(
                id="band_events",
                title="Band continuation events",
                description="Detected crossings or ambiguous continuation matches.",
                headers=("kind", "step", "band a", "band b", "x", "energy", "gap", "comment"),
                rows=event_rows,
                numeric=frozenset({1, 2, 3, 4, 5, 6}),
            ),
            Table(
                id="degenerate_group_events",
                title="Degenerate group events",
                description="Degenerate-subspace alignment events.",
                headers=("step", "bands", "energy min", "energy max", "gap", "subspace score", "min singular value"),
                rows=degenerate_rows,
                numeric=frozenset({0, 2, 3, 4, 5, 6}),
            ),
        ),
    )


def compute_region_surface(ctx, inputs: dict[str, object]) -> DiagnosticResult:
    kernel_choice = str(inputs["kernel"])
    matching_strategy = str(inputs["matching"])
    nu = int(inputs["nu"])
    nv = int(inputs["nv"])

    KH, KS = ctx.kernels(kernel_choice)

    region = LocalRegion.from_parallelogram(
        KH,
        KS,
        origin=(-np.pi, -np.pi),
        edge_u=(2.0 * np.pi, 0.0),
        edge_v=(0.0, 2.0 * np.pi),
        nu=nu,
        nv=nv,
        name=f"central square ({kernel_choice})",
        unit_context=ctx.state.data.working_unit_context,
        matching_strategy=matching_strategy,
    ).solve(fix_gauge=False)

    payload = region.payload()
    energies = np.asarray(region.energies, dtype=float) * ctx.state.data.energy_conversion_disk_to_working
    payload["energies"] = energies.tolist()
    payload["energy_min"] = float(np.min(energies))
    payload["energy_max"] = float(np.max(energies))
    payload["energy_unit"] = ctx.state.data.working_unit_context.energy.symbol

    return DiagnosticResult(
        title="Band region surface",
        summary="Solved LocalRegion payload for reusable band surface viewer.",
        cards=(
            Card("kernel", kernel_choice, "ok"),
            Card("matching", matching_strategy, "ok"),
            Card("grid", f"{nu}×{nv}", "ok"),
            Card("bands", energies.shape[2], "ok"),
        ),
        webgl=(
            WebGLView(
                id="band_region_surface",
                title="Band surface viewer",
                description="Real LocalRegion.payload(region) data for E_n(k1,k2).",
                renderer="region_surface",
                payload=payload,
            ),
        ),
    )


def diagnostics() -> list[DiagnosticSpec]:
    return [
        DiagnosticSpec(
            id="transport.bands.overview",
            group="transport",
            title="Bands domain overview",
            description="Show local files and public API for band/path continuation.",
            inputs=(),
            compute=compute_overview,
            tier="instant",
        ),
        DiagnosticSpec(
            id="transport.bands.path",
            group="transport",
            title="Band path Γ-K-M-Γ",
            description="Plot continued bands along a high-symmetry path.",
            inputs=(
                InputSpec(
                    "kernel",
                    "Kernel",
                    "select",
                    "average_star",
                    options=KERNEL_OPTIONS,
                    help="Kernel variant used to form H(k) and S(k).",
                ),
                InputSpec(
                    "matching",
                    "Matching strategy",
                    "select",
                    "energy_predict",
                    options=MATCHING_OPTIONS,
                    help="Band continuation matching method.",
                ),
                InputSpec(
                    "path",
                    "Path",
                    "select",
                    "gamma_k_m_gamma",
                    options=PATH_OPTIONS,
                    help="Path through irrep/k-space.",
                ),
                InputSpec(
                    "points_per_segment",
                    "Points per segment",
                    "int",
                    24,
                    min_value=4,
                    max_value=120,
                    help="Number of sampled points on each path segment.",
                ),
            ),
            compute=compute_band_path,
            tier="real_data",
        ),
        DiagnosticSpec(
            id="transport.bands.region_surface",
            group="transport",
            title="Band region surface",
            description="Render a solved LocalRegion as a reusable band surface payload.",
            inputs=(
                InputSpec(
                    "kernel",
                    "Kernel",
                    "select",
                    "average_star",
                    options=KERNEL_OPTIONS,
                    help="Kernel variant used to form H(k) and S(k).",
                ),
                InputSpec(
                    "matching",
                    "Matching strategy",
                    "select",
                    "energy_order",
                    options=MATCHING_OPTIONS,
                    help="Band labelling/matching method.",
                ),
                InputSpec(
                    "nu",
                    "nu",
                    "int",
                    9,
                    min_value=3,
                    max_value=40,
                    help="Grid points along u.",
                ),
                InputSpec(
                    "nv",
                    "nv",
                    "int",
                    9,
                    min_value=3,
                    max_value=40,
                    help="Grid points along v.",
                ),
            ),
            compute=compute_region_surface,
            tier="real_data",
        ),
    ]
