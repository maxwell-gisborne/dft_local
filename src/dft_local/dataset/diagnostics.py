"""Dataset overview diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from dft_local.core.local_problem import SymbolPair
from dft_local.core.numerics import DenseMatrixDiagnostics
from dft_local.diagnostics.models import (
    Card,
    DiagnosticResult,
    DiagnosticSection,
    DiagnosticSpec,
    InputSpec,
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



def _safe_relative(abs_value: float, ref_value: float) -> float:
    if ref_value == 0.0:
        return 0.0 if abs_value == 0.0 else float("inf")
    return abs_value / ref_value


def _sparse_frobenius_norm(M: Any) -> float:
    return float(np.sqrt(np.sum(np.abs(M.data) ** 2)))


def _sparse_hermitian_defect(M: Any) -> tuple[float, float]:
    defect = M - M.getH()
    defect_abs = _sparse_frobenius_norm(defect)
    ref = _sparse_frobenius_norm(M)
    return defect_abs, _safe_relative(defect_abs, ref)


def _block_coordinates(M: Any) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for row in range(M.indptr.size - 1):
        for col in M.indices[M.indptr[row] : M.indptr[row + 1]]:
            coords.add((int(row), int(col)))
    return coords


def _matrix_overview_rows(name: str, M: Any, nbasis: int, natoms: int) -> tuple[tuple[object, ...], dict[str, float]]:
    row_block_counts = np.diff(M.indptr)
    herm_abs, herm_rel = _sparse_hermitian_defect(M)
    expected_shape = (nbasis, nbasis)

    metrics = {
        "hermitian_abs": herm_abs,
        "hermitian_rel": herm_rel,
        "block_count": float(M.indices.size),
        "row_block_min": float(np.min(row_block_counts)) if len(row_block_counts) else 0.0,
        "row_block_median": float(np.median(row_block_counts)) if len(row_block_counts) else 0.0,
        "row_block_max": float(np.max(row_block_counts)) if len(row_block_counts) else 0.0,
    }

    row = (
        name,
        str(M.shape),
        M.shape == expected_shape,
        str(M.blocksize),
        M.indptr.size - 1,
        (M.indptr.size - 1) == natoms,
        int(M.nnz),
        int(M.indices.size),
        int(np.min(row_block_counts)) if len(row_block_counts) else 0,
        float(np.median(row_block_counts)) if len(row_block_counts) else 0.0,
        int(np.max(row_block_counts)) if len(row_block_counts) else 0,
        herm_abs,
        herm_rel,
    )
    return row, metrics


def compute_matrix_overview(ctx: Any, inputs: dict[str, object]) -> DiagnosticResult:
    """Summarise sparse H/S matrix structure and Hermiticity."""

    if ctx is None:
        return DiagnosticResult(
            title="Matrix overview",
            summary="No diagnostic context was provided.",
            cards=(
                Card("context", "missing", "warn", "Run with a loaded data root to inspect H and S."),
            ),
            sections=(
                DiagnosticSection(
                    id="matrix_missing_context",
                    title="Missing context",
                    tables=(
                        _table(
                            id="matrix_missing_context_table",
                            title="Context status",
                            description="Matrix overview needs a loaded DiagnosticContext.",
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

    h_row, h_metrics = _matrix_overview_rows("H", data.H, meta.nbasis, meta.natoms)
    s_row, s_metrics = _matrix_overview_rows("S", data.S, meta.nbasis, meta.natoms)

    h_blocks = _block_coordinates(data.H)
    s_blocks = _block_coordinates(data.S)
    shared_blocks = h_blocks & s_blocks
    union_blocks = h_blocks | s_blocks
    h_only = h_blocks - s_blocks
    s_only = s_blocks - h_blocks
    overlap_fraction = (
        len(shared_blocks) / len(union_blocks)
        if len(union_blocks)
        else 1.0
    )

    h_status = "ok" if h_metrics["hermitian_rel"] < 1e-5 else "warn"
    s_status = "ok" if s_metrics["hermitian_rel"] < 1e-8 else "warn"

    return DiagnosticResult(
        title="Matrix overview",
        summary=(
            f"H and S share {len(shared_blocks)} sparse atom-block positions "
            f"out of {len(union_blocks)} in the union."
        ),
        cards=(
            Card("H blocks", int(h_metrics["block_count"]), "ok", "Number of nonzero BSR atom blocks in H."),
            Card("S blocks", int(s_metrics["block_count"]), "ok", "Number of nonzero BSR atom blocks in S."),
            Card("H Hermiticity rel", h_metrics["hermitian_rel"], h_status, "||H - H†|| / ||H||."),
            Card("S Hermiticity rel", s_metrics["hermitian_rel"], s_status, "||S - S†|| / ||S||."),
            Card("H/S block overlap", overlap_fraction, "ok" if overlap_fraction == 1.0 else "warn", "Shared block positions divided by union block positions."),
        ),
        sections=(
            DiagnosticSection(
                id="matrix_structure",
                title="Sparse matrix structure",
                tables=(
                    _table(
                        id="matrix_structure_table",
                        title="BSR structure",
                        description="Shape, BSR block structure, row-block distribution, and global Hermiticity defects.",
                        headers=(
                            "matrix",
                            "shape",
                            "shape ok",
                            "blocksize",
                            "block rows",
                            "block rows ok",
                            "scalar nnz",
                            "block nnz",
                            "row blocks min",
                            "row blocks median",
                            "row blocks max",
                            "Hermiticity abs",
                            "Hermiticity rel",
                        ),
                        rows=[h_row, s_row],
                        numeric={4, 6, 7, 8, 9, 10, 11, 12},
                    ),
                ),
            ),
            DiagnosticSection(
                id="matrix_block_overlap",
                title="H/S support overlap",
                tables=(
                    _table(
                        id="matrix_block_overlap_table",
                        title="Atom-block support overlap",
                        description="Comparison of nonzero atom-block coordinates in H and S.",
                        rows=[
                            ("H block positions", len(h_blocks)),
                            ("S block positions", len(s_blocks)),
                            ("shared positions", len(shared_blocks)),
                            ("union positions", len(union_blocks)),
                            ("H-only positions", len(h_only)),
                            ("S-only positions", len(s_only)),
                            ("overlap fraction", overlap_fraction),
                        ],
                        numeric={1},
                    ),
                ),
            ),
        ),
    )



def _kernel_label_range(K: Any) -> tuple[int | None, int | None, int | None, int | None]:
    if K.support_size == 0:
        return None, None, None, None
    return int(np.min(K.h_m)), int(np.max(K.h_m)), int(np.min(K.h_n)), int(np.max(K.h_n))


def _kernel_summary_row(label: str, K: Any) -> tuple[object, ...]:
    diag = K.diagnostics()
    star = K.star_defect()
    m_min, m_max, n_min, n_max = _kernel_label_range(K)

    return (
        label,
        diag["matrix_name"],
        diag["support_size"],
        diag["blocksize"],
        diag["num_even"],
        diag["num_odd"],
        m_min,
        m_max,
        n_min,
        n_max,
        diag["norm_min"],
        diag["norm_median"],
        diag["norm_max"],
        star["num_missing_inverse"],
        star["star_defect_max"],
        star["star_defect_mean"],
        star["star_defect_median"],
    )


def _kernel_choice(inputs: dict[str, object]) -> str:
    choice = str(inputs.get("kernel_choice", "average_star"))
    allowed = {"anchored", "anchored_star", "average", "average_star"}
    if choice not in allowed:
        raise ValueError(f"kernel_choice must be one of {sorted(allowed)}, got {choice!r}")
    return choice


def compute_kernel_overview(ctx: Any, inputs: dict[str, object]) -> DiagnosticResult:
    """Summarise selected H/S local G_d kernels."""

    choice = _kernel_choice(inputs)

    if ctx is None:
        return DiagnosticResult(
            title="Kernel overview",
            summary="No diagnostic context was provided.",
            cards=(
                Card("context", "missing", "warn", "Run with a loaded data root to inspect kernels."),
                Card("kernel choice", choice, "ok", "Requested kernel variant."),
            ),
            sections=(
                DiagnosticSection(
                    id="kernel_missing_context",
                    title="Missing context",
                    tables=(
                        _table(
                            id="kernel_missing_context_table",
                            title="Context status",
                            description="Kernel overview needs a loaded DiagnosticContext.",
                            rows=[
                                ("context", "missing"),
                                ("expected", "DiagnosticContext from load_default_context(root)"),
                                ("kernel choice", choice),
                            ],
                        ),
                    ),
                ),
            ),
        )

    KH, KS = ctx.kernels(choice)
    h_star = KH.star_defect()
    s_star = KS.star_defect()

    h_status = "ok" if (h_star["num_missing_inverse"] == 0 and (h_star["star_defect_max"] or 0.0) < 1e-8) else "warn"
    s_status = "ok" if (s_star["num_missing_inverse"] == 0 and (s_star["star_defect_max"] or 0.0) < 1e-8) else "warn"

    return DiagnosticResult(
        title="Kernel overview",
        summary=(
            f"Kernel choice {choice!r}: H support {KH.support_size}, "
            f"S support {KS.support_size}, blocksize {KH.blocksize}."
        ),
        cards=(
            Card("kernel choice", choice, "ok", "Selected kernel variant."),
            Card("H support", KH.support_size, "ok", "Number of H kernel support labels."),
            Card("S support", KS.support_size, "ok", "Number of S kernel support labels."),
            Card("H star defect max", h_star["star_defect_max"], h_status, "Maximum relative K(h) - K(h^-1)† defect."),
            Card("S star defect max", s_star["star_defect_max"], s_status, "Maximum relative K(h) - K(h^-1)† defect."),
        ),
        sections=(
            DiagnosticSection(
                id="kernel_summary",
                title="Kernel summary",
                tables=(
                    _table(
                        id="kernel_summary_table",
                        title="Selected H/S kernels",
                        description="Support, label ranges, block norms, and star-defect diagnostics for the selected kernel variant.",
                        headers=(
                            "kernel",
                            "matrix name",
                            "support",
                            "blocksize",
                            "even",
                            "odd",
                            "m min",
                            "m max",
                            "n min",
                            "n max",
                            "norm min",
                            "norm median",
                            "norm max",
                            "missing inverse",
                            "star defect max",
                            "star defect mean",
                            "star defect median",
                        ),
                        rows=[
                            _kernel_summary_row("H", KH),
                            _kernel_summary_row("S", KS),
                        ],
                        numeric={2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                    ),
                ),
            ),
            DiagnosticSection(
                id="kernel_support_balance",
                title="Support balance",
                tables=(
                    _table(
                        id="kernel_support_balance_table",
                        title="Even/odd support split",
                        description="Support count split by the G_d eps label.",
                        headers=("kernel", "even", "odd", "total", "odd fraction"),
                        rows=[
                            (
                                "H",
                                int(np.sum(KH.h_eps == 0)),
                                int(np.sum(KH.h_eps == 1)),
                                KH.support_size,
                                float(np.sum(KH.h_eps == 1) / KH.support_size) if KH.support_size else 0.0,
                            ),
                            (
                                "S",
                                int(np.sum(KS.h_eps == 0)),
                                int(np.sum(KS.h_eps == 1)),
                                KS.support_size,
                                float(np.sum(KS.h_eps == 1) / KS.support_size) if KS.support_size else 0.0,
                            ),
                        ],
                        numeric={1, 2, 3, 4},
                    ),
                ),
            ),
        ),
    )



def _dense_diag_row(label: str, diag: dict[str, object]) -> tuple[object, ...]:
    return (
        label,
        diag["shape"],
        diag["dtype"],
        diag["finite"],
        diag["norm"],
        diag["hermitian_defect_abs"],
        diag["hermitian_defect_rel"],
        diag["eig_min"],
        diag["eig_max"],
        diag["condition_number_abs"],
        diag["positive_definite"],
    )


def _symbol_degree(inputs: dict[str, object]) -> int:
    degree = int(inputs.get("irrep_degree", 2))
    if degree not in (1, 2):
        raise ValueError(f"irrep_degree must be 1 or 2, got {degree}")
    return degree


def _symbol_sigma(inputs: dict[str, object], degree: int) -> int | None:
    if degree == 2:
        return None

    sigma = int(inputs.get("sigma", 1))
    if sigma not in (-1, 1):
        raise ValueError(f"sigma must be ±1 for degree-1 irreps, got {sigma}")
    return sigma


def compute_symbol_point(ctx: Any, inputs: dict[str, object]) -> DiagnosticResult:
    """Inspect H(k), S(k), overlap spectrum, and generalized energies at one irrep point."""

    choice = _kernel_choice(inputs)
    k1 = float(inputs.get("k1", 0.0))
    k2 = float(inputs.get("k2", 0.0))
    degree = _symbol_degree(inputs)
    sigma = _symbol_sigma(inputs, degree)

    if ctx is None:
        return DiagnosticResult(
            title="Symbol point",
            summary="No diagnostic context was provided.",
            cards=(
                Card("context", "missing", "warn", "Run with a loaded data root to inspect symbols."),
                Card("kernel choice", choice, "ok", "Requested kernel variant."),
            ),
            sections=(
                DiagnosticSection(
                    id="symbol_missing_context",
                    title="Missing context",
                    tables=(
                        _table(
                            id="symbol_missing_context_table",
                            title="Context status",
                            description="Symbol point needs a loaded DiagnosticContext.",
                            rows=[
                                ("context", "missing"),
                                ("expected", "DiagnosticContext from load_default_context(root)"),
                                ("kernel choice", choice),
                                ("k1", k1),
                                ("k2", k2),
                                ("irrep degree", degree),
                                ("sigma", sigma),
                            ],
                            numeric={1},
                        ),
                    ),
                ),
            ),
        )

    KH, KS = ctx.kernels(choice)
    pair = SymbolPair(KH, KS, k1, k2, degree=degree, sigma=sigma, name=choice)
    problem = pair.form()
    sym_problem = problem.symmetrised()

    H_diag = DenseMatrixDiagnostics.from_dense_matrix(problem.Hk, name="H(k)").as_dict()
    S_diag = DenseMatrixDiagnostics.from_dense_matrix(
        problem.Sk,
        name="S(k)",
        check_eigenvalues=True,
    ).as_dict()
    H_sym_diag = DenseMatrixDiagnostics.from_dense_matrix(
        sym_problem.Hk,
        name="Hermitian part H(k)",
    ).as_dict()
    S_sym_diag = DenseMatrixDiagnostics.from_dense_matrix(
        sym_problem.Sk,
        name="Hermitian part S(k)",
        check_eigenvalues=True,
    ).as_dict()

    overlap_eigs = problem.overlap_eigenvalues()

    energy_rows: list[tuple[object, ...]]
    energy_status = "ok"
    energy_detail = "Generalized eigenvalues from Hermitian parts of H(k), S(k)."

    try:
        energies = problem.energies()
        energy_rows = [(i, float(E)) for i, E in enumerate(energies)]
    except Exception as exc:
        energy_status = "error"
        energy_detail = str(exc)
        energy_rows = [("error", energy_detail)]

    return DiagnosticResult(
        title="Symbol point",
        summary=(
            f"{pair.label()} using kernel choice {choice!r}; "
            f"dense problem shape {problem.Hk.shape}."
        ),
        cards=(
            Card("kernel choice", choice, "ok", "Selected kernel variant."),
            Card("k1", k1, "ok", "First logical irrep coordinate."),
            Card("k2", k2, "ok", "Second logical irrep coordinate."),
            Card("degree", degree, "ok", "Irrep degree used to form the symbol."),
            Card("S positive", S_diag["positive_definite"], "ok" if S_diag["positive_definite"] else "warn", "Positive definiteness of Hermitian part of S(k)."),
            Card("energies", len(energy_rows), energy_status, energy_detail),
        ),
        sections=(
            DiagnosticSection(
                id="symbol_dense_diagnostics",
                title="Dense symbol diagnostics",
                tables=(
                    _table(
                        id="symbol_dense_diagnostics_table",
                        title="H(k), S(k) diagnostics",
                        description="Dense matrix diagnostics before and after taking Hermitian parts.",
                        headers=(
                            "matrix",
                            "shape",
                            "dtype",
                            "finite",
                            "norm",
                            "Hermiticity abs",
                            "Hermiticity rel",
                            "eig min",
                            "eig max",
                            "condition number",
                            "positive definite",
                        ),
                        rows=[
                            _dense_diag_row("H(k)", H_diag),
                            _dense_diag_row("S(k)", S_diag),
                            _dense_diag_row("Hermitian H(k)", H_sym_diag),
                            _dense_diag_row("Hermitian S(k)", S_sym_diag),
                        ],
                        numeric={4, 5, 6, 7, 8, 9},
                    ),
                ),
            ),
            DiagnosticSection(
                id="symbol_overlap",
                title="Overlap eigenvalues",
                tables=(
                    _table(
                        id="symbol_overlap_eigenvalues",
                        title="Eigenvalues of Hermitian part of S(k)",
                        description="Overlap spectrum used to check generalized eigenproblem conditioning.",
                        headers=("index", "eigenvalue"),
                        rows=[(i, float(v)) for i, v in enumerate(overlap_eigs)],
                        numeric={0, 1},
                    ),
                ),
            ),
            DiagnosticSection(
                id="symbol_energies",
                title="Generalized energies",
                tables=(
                    _table(
                        id="symbol_energy_table",
                        title="Generalized eigenvalues",
                        description=energy_detail,
                        headers=("band", "energy"),
                        rows=energy_rows,
                        numeric={0, 1} if energy_status == "ok" else set(),
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
        DiagnosticSpec(
            id="matrix.overview",
            group="matrix",
            title="Matrix overview",
            description="Sparse H/S BSR structure, support overlap, and global Hermiticity summary.",
            inputs=(),
            compute=compute_matrix_overview,
        ),
        DiagnosticSpec(
            id="kernel.overview",
            group="kernel",
            title="Kernel overview",
            description="Local G_d H/S kernel support, block norms, label ranges, and star-defect summary.",
            inputs=(
                InputSpec(
                    "kernel_choice",
                    "Kernel choice",
                    "select",
                    "average_star",
                    options=(
                        ("anchored", "anchored"),
                        ("anchored_star", "anchored star"),
                        ("average", "average"),
                        ("average_star", "average star"),
                    ),
                ),
            ),
            compute=compute_kernel_overview,
        ),
        DiagnosticSpec(
            id="symbol.point",
            group="symbol",
            title="Symbol point",
            description="Inspect dense H(k), S(k), overlap eigenvalues, and generalized energies at one irrep point.",
            inputs=(
                InputSpec(
                    "kernel_choice",
                    "Kernel choice",
                    "select",
                    "average_star",
                    options=(
                        ("anchored", "anchored"),
                        ("anchored_star", "anchored star"),
                        ("average", "average"),
                        ("average_star", "average star"),
                    ),
                ),
                InputSpec("k1", "k1", "float", 0.0),
                InputSpec("k2", "k2", "float", 0.0),
                InputSpec("irrep_degree", "irrep degree", "int", 2, min_value=1, max_value=2),
                InputSpec("sigma", "sigma", "int", 1, min_value=-1, max_value=1),
            ),
            compute=compute_symbol_point,
        ),
    )
