"""Dataset overview diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from dft_local.diagnostics.models import (
    Card,
    DiagnosticResult,
    DiagnosticSection,
    DiagnosticSpec,
    Table,
    TableRow,
)


def _table(
    *,
    id: str,
    title: str,
    description: str,
    rows: list[tuple[object, ...]],
    headers: tuple[str, ...] = ("quantity", "value"),
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


def compute_overview(ctx: Any, inputs: dict[str, object]) -> DiagnosticResult:
    """Summarise the loaded sparse dataset."""

    if ctx is None:
        return DiagnosticResult(
            title="Dataset overview",
            summary="No diagnostic context was provided.",
            cards=(
                Card("context", "missing", "warn", "Run with a loaded data root to inspect a dataset."),
            ),
            sections=(
                DiagnosticSection(
                    id="dataset_missing_context",
                    title="Missing context",
                    tables=(
                        _table(
                            id="dataset_missing_context_table",
                            title="Context status",
                            description="Dataset overview needs a loaded DiagnosticContext.",
                            rows=[
                                ("context", "missing"),
                                ("expected", "DiagnosticContext from load_default_context(root)"),
                            ],
                        ),
                    ),
                ),
            ),
        )

    data = ctx.state.data
    meta = data.metadata

    symbol_counts = {
        str(symbol): int(count)
        for symbol, count in zip(*np.unique(meta.symbols, return_counts=True), strict=False)
    }

    matrix_rows = [
        ("H", str(data.H.shape), int(data.H.nnz), str(data.H.blocksize), int(data.H.indptr.size)),
        ("S", str(data.S.shape), int(data.S.nnz), str(data.S.blocksize), int(data.S.indptr.size)),
    ]

    return DiagnosticResult(
        title="Dataset overview",
        summary=(
            f"Loaded {meta.natoms} atoms, {meta.nbasis} basis functions, "
            f"{meta.nchannels} channels per atom."
        ),
        cards=(
            Card("atoms", meta.natoms, "ok", "Number of atoms in sparsematrix_metadata.dat."),
            Card("basis", meta.nbasis, "ok", "Number of basis functions."),
            Card("channels / atom", meta.nchannels, "ok", "Uniform local channel count."),
            Card("BigDFT log", "present" if data.bigdft_log is not None else "missing", "ok" if data.bigdft_log is not None else "warn"),
        ),
        sections=(
            DiagnosticSection(
                id="dataset_identity",
                title="Dataset identity",
                tables=(
                    _table(
                        id="dataset_identity_table",
                        title="Loaded dataset",
                        description="Basic identity and shape information for the loaded sparse dataset.",
                        rows=[
                            ("root", str(data.root)),
                            ("atoms", meta.natoms),
                            ("basis functions", meta.nbasis),
                            ("channels per atom", meta.nchannels),
                            ("symbols", ", ".join(f"{k}: {v}" for k, v in sorted(symbol_counts.items()))),
                            ("symbol dictionary", ", ".join(meta.symbols_dictionary)),
                            ("BigDFT log", "present" if data.bigdft_log is not None else "missing"),
                        ],
                        numeric={1},
                    ),
                ),
            ),
            DiagnosticSection(
                id="dataset_matrices",
                title="Sparse matrices",
                tables=(
                    _table(
                        id="dataset_sparse_matrices",
                        title="Sparse matrix block structure",
                        description="Atom-ordered BSR matrix structure for H and S.",
                        headers=("matrix", "shape", "nnz", "blocksize", "indptr size"),
                        rows=matrix_rows,
                        numeric={2, 4},
                    ),
                ),
            ),
            DiagnosticSection(
                id="dataset_units",
                title="Units",
                tables=(
                    _table(
                        id="dataset_units_table",
                        title="Unit provenance",
                        description="Disk-to-working unit conversion used when loading the dataset.",
                        rows=[tuple(row) for row in ctx.state.unit_provenance_rows()],
                        numeric={1},
                    ),
                ),
            ),
        ),
    )


def compute_geometry_overview(ctx: Any, inputs: dict[str, object]) -> DiagnosticResult:
    """Summarise nearest-neighbour geometry and G_d labels."""

    if ctx is None:
        return DiagnosticResult(
            title="Geometry overview",
            summary="No diagnostic context was provided.",
            cards=(
                Card("context", "missing", "warn", "Run with a loaded data root to inspect geometry."),
            ),
            sections=(
                DiagnosticSection(
                    id="geometry_missing_context",
                    title="Missing context",
                    tables=(
                        _table(
                            id="geometry_missing_context_table",
                            title="Context status",
                            description="Geometry overview needs a loaded DiagnosticContext.",
                            rows=[
                                ("context", "missing"),
                                ("expected", "DiagnosticContext from load_default_context(root)"),
                            ],
                        ),
                    ),
                ),
            ),
        )

    geom_diag = ctx.state.geom.diagnostics()
    edge_diag = ctx.state.edges.diagnostics(ctx.state.geom)
    label_diag = ctx.state.labels.diagnostics()
    err = label_diag["position_reconstruction_errors"]

    return DiagnosticResult(
        title="Geometry overview",
        summary=(
            f"Nearest-neighbour graph has {geom_diag['natoms']} atoms, "
            f"anchor atom {label_diag['anchor_atom']}, "
            f"{label_diag['visited_count']} labelled atoms."
        ),
        cards=(
            Card("atoms", geom_diag["natoms"], "ok", "Number of atoms in the neighbour graph."),
            Card("anchor", label_diag["anchor_atom"], "ok", "Bulk atom chosen as group identity."),
            Card("visited", label_diag["visited_count"], "ok", "Atoms assigned a G_d label."),
            Card("max reconstruction error", err["max"], "ok", "Maximum |R_G(g_a) - R_a|."),
        ),
        sections=(
            DiagnosticSection(
                id="geometry_graph",
                title="Nearest-neighbour graph",
                tables=(
                    _table(
                        id="geometry_graph_summary",
                        title="Graph summary",
                        description="Nearest-neighbour graph inferred from atom positions.",
                        rows=[
                            ("natoms", geom_diag["natoms"]),
                            ("a0", geom_diag["a0"]),
                            ("cutoff", geom_diag["cutoff"]),
                            ("cutoff / a0", geom_diag["cutoff_over_a0"]),
                            ("degree counts", geom_diag["degree_counts"]),
                            ("bulk atoms", geom_diag["num_bulk_atoms"]),
                            ("core bulk atoms", geom_diag["num_core_bulk_atoms"]),
                            ("anchor atom", geom_diag["anchor_atom"]),
                            ("nearest distance min", geom_diag["nearest_distance_min"]),
                            ("nearest distance median", geom_diag["nearest_distance_median"]),
                            ("nearest distance max", geom_diag["nearest_distance_max"]),
                            ("nearest distance std", geom_diag["nearest_distance_std"]),
                        ],
                        numeric={1},
                    ),
                ),
            ),
            DiagnosticSection(
                id="geometry_edge_directions",
                title="Edge directions",
                tables=(
                    _table(
                        id="geometry_edge_directions_summary",
                        title="Edge direction classification",
                        description="Generator directions d1, d2, d3 inferred at the anchor and used to classify every edge.",
                        rows=[
                            ("anchor atom", edge_diag["anchor_atom"]),
                            ("anchor neighbours", edge_diag["anchor_neighbours"]),
                            ("generator counts", edge_diag["generator_counts"]),
                            ("alignment min", edge_diag["alignment_min"]),
                            ("alignment median", edge_diag["alignment_median"]),
                            ("alignment max", edge_diag["alignment_max"]),
                        ],
                        numeric={1},
                    ),
                    _table(
                        id="geometry_edge_direction_vectors",
                        title="Anchor edge vectors",
                        description="The three generator vectors at the chosen anchor atom.",
                        headers=("generator", "dx", "dy", "dz"),
                        rows=[
                            (f"d{i + 1}", *vector)
                            for i, vector in enumerate(edge_diag["d_vectors"])
                        ],
                        numeric={1, 2, 3},
                    ),
                ),
            ),
            DiagnosticSection(
                id="geometry_group_labels",
                title="G_d labels",
                tables=(
                    _table(
                        id="geometry_group_label_summary",
                        title="Group-label summary",
                        description="BFS labelling of atoms by edge-generator group elements.",
                        rows=[
                            ("natoms", label_diag["natoms"]),
                            ("anchor atom", label_diag["anchor_atom"]),
                            ("visited count", label_diag["visited_count"]),
                            ("unvisited count", label_diag["unvisited_count"]),
                            ("eps counts", label_diag["eps_counts"]),
                            ("m min", label_diag["m_min"]),
                            ("m max", label_diag["m_max"]),
                            ("n min", label_diag["n_min"]),
                            ("n max", label_diag["n_max"]),
                            ("position error max", err["max"]),
                            ("position error mean", err["mean"]),
                            ("position error median", err["median"]),
                        ],
                        numeric={1},
                    ),
                ),
            ),
        ),
    )


def diagnostics() -> tuple[DiagnosticSpec, ...]:
    return (
        DiagnosticSpec(
            id="dataset.overview",
            group="dataset",
            title="Dataset overview",
            description="Loaded sparse dataset shape, units, and matrix sanity summary.",
            inputs=(),
            compute=compute_overview,
        ),
        DiagnosticSpec(
            id="geometry.overview",
            group="geometry",
            title="Geometry overview",
            description="Nearest-neighbour graph, edge directions, and G_d group labelling summary.",
            inputs=(),
            compute=compute_geometry_overview,
        ),
    )
