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
    )
