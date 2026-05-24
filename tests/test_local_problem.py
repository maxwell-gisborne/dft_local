import numpy as np
import pytest
from scipy.linalg import eigvalsh

from dft_local import (
    SparseDataset,
    NearestNeighbourGraph,
    EdgeDirections,
    EdgeGroupLabels,
    GdElement,
    GdKernelArrays,
    SymbolPair,
    LocalPath,
    DenseMatrixDiagnostics,
    block_row_raw,
    gd_inverse_label,
    hermitian_part,
)


@pytest.fixture(scope="session")
def data():
    return SparseDataset.load("test_run/run_dir/data")


@pytest.fixture(scope="session")
def geom(data):
    return NearestNeighbourGraph.from_positions(data.metadata.positions)


@pytest.fixture(scope="session")
def edge_dirs(geom):
    return EdgeDirections.from_geometry(geom)


@pytest.fixture(scope="session")
def labels(geom, edge_dirs):
    return EdgeGroupLabels.from_geometry(geom, edge_dirs)


@pytest.fixture(scope="session")
def KH(data, labels):
    return GdKernelArrays.from_anchored(data.H, labels, matrix_name="H")


@pytest.fixture(scope="session")
def KS(data, labels):
    return GdKernelArrays.from_anchored(data.S, labels, matrix_name="S")


@pytest.fixture(scope="session")
def KH_avg(data, labels):
    return GdKernelArrays.from_average(data.H, labels, matrix_name="H average")


@pytest.fixture(scope="session")
def KS_avg(data, labels):
    return GdKernelArrays.from_average(data.S, labels, matrix_name="S average")


@pytest.fixture(scope="session")
def KH_avg_star(KH_avg):
    return KH_avg.star_symmetrised(matrix_name="H average star")


@pytest.fixture(scope="session")
def KS_avg_star(KS_avg):
    return KS_avg.star_symmetrised(matrix_name="S average star")


def test_labels_anchor_is_identity(labels):
    assert labels.element(labels.anchor_atom) == GdElement.identity()


def test_labels_all_atoms_visited(labels):
    assert np.all(labels.visited)


def test_labels_balanced_sublattices(labels):
    vals, counts = np.unique(labels.eps[labels.visited], return_counts=True)
    got = dict(zip(map(int, vals), map(int, counts)))

    assert set(got) == {0, 1}
    assert got[0] == got[1]


def test_labels_reconstruct_positions(labels):
    err = labels.gd_position_errors()

    assert np.max(err) < 1e-6


def test_relative_labels_match_group_elements(labels, data):
    # sample a BSR row because those are the pairs used later
    a = labels.anchor_atom
    atoms_b, _blocks = block_row_raw(data.H, a)

    for b in atoms_b[:50]:
        b = int(b)
        h = labels.relative(a, b)

        assert h.m == labels.m[b]
        assert h.n == labels.n[b]
        assert h.eps == labels.eps[b]

def test_anchored_kernel_support_matches_bsr_row(data, labels, KH):
    atoms_b, blocks = block_row_raw(data.H, labels.anchor_atom)

    assert KH.support_size == len(atoms_b)
    assert KH.blocks.shape == blocks.shape


def test_identity_anchor_kernel_labels_are_absolute(data, labels, KH):
    atoms_b, _blocks = block_row_raw(data.H, labels.anchor_atom)

    assert np.all(KH.h_m == labels.m[atoms_b])
    assert np.all(KH.h_n == labels.n[atoms_b])
    assert np.all(KH.h_eps == labels.eps[atoms_b])


def test_kernel_inverse_label_formula():
    examples = [
        (0, 0, 0),
        (1, 0, 0),
        (-2, 3, 0),
        (0, 0, 1),
        (-1, 0, 1),
        (2, -3, 1),
    ]

    for m, n, eps in examples:
        inv = gd_inverse_label(m, n, eps)
        inv2 = gd_inverse_label(*inv)

        assert inv2 == (m, n, eps)


def test_star_symmetrised_kernel_has_zero_star_defect(KH_avg_star):
    d = KH_avg_star.star_defect()

    assert d["num_missing_inverse"] == 0
    assert d["star_defect_max"] < 1e-12


def test_average_kernel_more_star_symmetric_than_anchored(KH, KH_avg):
    d_anchor = KH.star_defect()
    d_avg = KH_avg.star_defect()

    assert d_avg["star_defect_mean"] < d_anchor["star_defect_mean"]
    assert d_avg["star_defect_max"] < d_anchor["star_defect_max"]


@pytest.mark.parametrize(
    "k1,k2",
    [
        (0.0, 0.0),
        (0.1, 0.2),
        (2 * np.pi / 3, -2 * np.pi / 3),
        (np.pi, 0.0),
    ],
)
def test_generic_symbol_shapes(KH_avg_star, KS_avg_star, k1, k2):
    Hk = KH_avg_star.symbol_generic(k1, k2)
    Sk = KS_avg_star.symbol_generic(k1, k2)

    q = KH_avg_star.blocksize

    assert Hk.shape == (2 * q, 2 * q)
    assert Sk.shape == (2 * q, 2 * q)
    assert np.all(np.isfinite(Hk))
    assert np.all(np.isfinite(Sk))


@pytest.mark.parametrize(
    "k1,k2",
    [
        (0.0, 0.0),
        (0.1, 0.2),
        (2 * np.pi / 3, -2 * np.pi / 3),
        (np.pi, 0.0),
    ],
)
def test_star_symbols_are_hermitian(KH_avg_star, KS_avg_star, k1, k2):
    Hk = KH_avg_star.symbol_generic(k1, k2)
    Sk = KS_avg_star.symbol_generic(k1, k2)

    assert np.allclose(Hk, Hk.conj().T, atol=1e-10)
    assert np.allclose(Sk, Sk.conj().T, atol=1e-10)


@pytest.mark.parametrize(
    "k1,k2",
    [
        (0.0, 0.0),
        (0.1, 0.2),
        (2 * np.pi / 3, -2 * np.pi / 3),
        (np.pi, 0.0),
    ],
)
def test_overlap_symbol_positive_definite(KS_avg_star, k1, k2):
    Sk = KS_avg_star.symbol_generic(k1, k2)
    eigs = eigvalsh(hermitian_part(Sk))

    assert np.min(eigs) > 1e-10


@pytest.mark.parametrize(
    "k1,k2",
    [
        (0.0, 0.0),
        (np.pi, 0.0),
        (0.0, np.pi),
        (np.pi, np.pi),
    ],
)
def test_fixed_point_spectrum_matches_generic(KH_avg_star, k1, k2):
    A2 = KH_avg_star.symbol_generic(k1, k2)

    A_plus = KH_avg_star.symbol_fixed(k1, k2, sigma=1)
    A_minus = KH_avg_star.symbol_fixed(k1, k2, sigma=-1)

    evals_2 = np.sort(np.linalg.eigvalsh(hermitian_part(A2)))
    evals_1 = np.sort(
        np.concatenate([
            np.linalg.eigvalsh(hermitian_part(A_plus)),
            np.linalg.eigvalsh(hermitian_part(A_minus)),
        ])
    )

    assert np.allclose(evals_2, evals_1, atol=1e-8)

def test_local_problem_diagnostics_usable(KH_avg_star, KS_avg_star):
    pair = SymbolPair(KH_avg_star, KS_avg_star, 0.1, 0.2)
    local = pair.form()

    Hdiag = DenseMatrixDiagnostics.from_dense_matrix(local.Hk, name="H")
    Sdiag = DenseMatrixDiagnostics.from_dense_matrix(
        local.Sk,
        name="S",
        check_eigenvalues=True,
    )

    assert Hdiag.finite
    assert Sdiag.finite
    assert Hdiag.hermitian_defect_rel < 1e-10
    assert Sdiag.hermitian_defect_rel < 1e-10
    assert Sdiag.positive_definite


def test_local_problem_solves(KH_avg_star, KS_avg_star):
    pair = SymbolPair(KH_avg_star, KS_avg_star, 0.1, 0.2)
    local = pair.form()

    E = local.energies()

    assert E.shape == (8,)
    assert np.all(np.isfinite(E))
    assert np.all(np.diff(E) >= 0)

@pytest.fixture(scope="session")
def sample_path(KH_avg_star, KS_avg_star):
    points = [
        ("Γ", 0.0, 0.0),
        ("K", 2 * np.pi / 3, -2 * np.pi / 3),
        ("M", np.pi, 0.0),
        ("Γ", 0.0, 0.0),
    ]

    return LocalPath.from_points(
        KH_avg_star,
        KS_avg_star,
        points,
        points_per_segment=8,
        name="test path",
    )


def test_local_path_shapes(sample_path):
    assert sample_path.k1.shape == sample_path.k2.shape
    assert sample_path.x is not None
    assert sample_path.x.shape == sample_path.k1.shape
    assert len(sample_path.labels) == 4


def test_local_path_energies_shape(sample_path):
    E = sample_path.energies()

    assert E.ndim == 2
    assert E.shape[0] == len(sample_path.k1)
    assert E.shape[1] == 8
    assert np.all(np.isfinite(E))


def test_local_path_x_monotone(sample_path):
    assert sample_path.x is not None
    assert np.all(np.diff(sample_path.x) >= 0)
