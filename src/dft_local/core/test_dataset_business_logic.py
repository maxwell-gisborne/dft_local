# Copied from repository-level tests during package migration.
# These tests should target dft_local only.

import numpy as np
import pytest
from scipy.sparse import bsr_matrix

from dft_local.core.dataset import SparseDataset


@pytest.fixture(scope="session")
def data():
    return SparseDataset.load("./test_run/run_dir/data")


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def sample_atoms(data, rng):
    n = data.metadata.natoms
    size = min(100, n)
    return rng.choice(n, size=size, replace=False)


def bsr_row_raw(M: bsr_matrix, atom: int):
    start = M.indptr[atom]
    stop = M.indptr[atom + 1]
    return M.indices[start:stop], M.data[start:stop]


def atom_block_bsr(M: bsr_matrix, a: int, b: int, *, copy: bool = False):
    atoms_b, blocks = bsr_row_raw(M, a)
    hits = np.flatnonzero(atoms_b == b)

    if len(hits) == 0:
        out = np.zeros(M.blocksize, dtype=M.dtype)
        return out.copy() if copy else out

    if len(hits) > 1:
        raise AssertionError(f"duplicate BSR block for a={a}, b={b}")

    block = blocks[int(hits[0])]
    return block.copy() if copy else block


def assert_blocks_close(A, B, *, atol=1e-8, rtol=1e-6, context=""):
    if not np.allclose(A, B, atol=atol, rtol=rtol):
        err = np.linalg.norm(A - B)
        scale = max(np.linalg.norm(A), np.linalg.norm(B), 1.0)
        rel = err / scale
        raise AssertionError(
            f"block mismatch {context}: norm error={err}, relative={rel}"
        )


def check_block_hermitian(M: bsr_matrix, atoms, *, atol=1e-7, rtol=1e-5):
    for a in atoms:
        a = int(a)
        atoms_b, blocks = bsr_row_raw(M, a)

        for b, Mab in zip(atoms_b, blocks):
            b = int(b)
            Mba = atom_block_bsr(M, b, a)

            assert_blocks_close(
                Mab,
                Mba.conj().T,
                atol=atol,
                rtol=rtol,
                context=f"a={a}, b={b}",
            )


def coupled_atom_blocks(M: bsr_matrix, positions: np.ndarray, a: int):
    atoms_b, blocks = bsr_row_raw(M, a)

    Ra = positions[a]
    dR = positions[atoms_b] - Ra
    distances = np.linalg.norm(dR, axis=1)
    norms = np.linalg.norm(blocks, axis=(1, 2))

    order = np.argsort(-norms)

    return atoms_b[order], blocks[order], distances[order], norms[order], dR[order]


def test_dataset_shapes(data):
    assert data.H.shape[0] == data.H.shape[1]
    assert data.S.shape == data.H.shape
    assert data.H.shape[0] == data.metadata.nbasis

    assert data.basis.nchannels == 4
    assert data.basis.atom_basis.shape == (
        data.metadata.natoms,
        data.basis.nchannels,
    )


def test_H_and_S_are_bsr(data):
    assert isinstance(data.H, bsr_matrix)
    assert isinstance(data.S, bsr_matrix)


def test_H_bsr_has_expected_blocksize(data):
    assert data.H.blocksize == (data.basis.nchannels, data.basis.nchannels)
    assert data.H.indptr.shape == (data.metadata.natoms + 1,)
    assert data.H.data.shape[1:] == data.H.blocksize


def test_S_bsr_has_expected_blocksize(data):
    assert data.S.blocksize == (data.basis.nchannels, data.basis.nchannels)
    assert data.S.indptr.shape == (data.metadata.natoms + 1,)
    assert data.S.data.shape[1:] == data.S.blocksize


def test_bsr_indices_are_atom_indices(data):
    assert np.all(data.H.indices >= 0)
    assert np.all(data.H.indices < data.metadata.natoms)

    assert np.all(data.S.indices >= 0)
    assert np.all(data.S.indices < data.metadata.natoms)


def test_bsr_rows_have_matching_lengths(data, sample_atoms):
    for M in [data.H, data.S]:
        for a in sample_atoms:
            atoms_b, blocks = bsr_row_raw(M, int(a))

            assert len(atoms_b) == len(blocks)
            assert blocks.ndim == 3
            assert blocks.shape[1:] == M.blocksize


def test_bsr_rows_have_no_duplicate_atom_columns(data, sample_atoms):
    for M in [data.H, data.S]:
        for a in sample_atoms:
            atoms_b, _blocks = bsr_row_raw(M, int(a))
            assert len(set(map(int, atoms_b))) == len(atoms_b)


def test_atom_block_bsr_matches_raw_row(data, sample_atoms):
    for M in [data.H, data.S]:
        for a in sample_atoms:
            a = int(a)
            atoms_b, blocks = bsr_row_raw(M, a)

            for b, block in zip(atoms_b, blocks):
                direct = atom_block_bsr(M, a, int(b))

                assert_blocks_close(
                    block,
                    direct,
                    atol=1e-12,
                    rtol=0.0,
                    context=f"a={a}, b={int(b)}",
                )


def test_missing_atom_block_returns_zero(data, sample_atoms):
    """
    Find at least one absent block in each sampled row and check zero return.
    """
    natoms = data.metadata.natoms

    for M in [data.H, data.S]:
        checked = False

        for a in sample_atoms:
            a = int(a)
            atoms_b, _blocks = bsr_row_raw(M, a)
            present = set(map(int, atoms_b))

            for b in range(natoms):
                if b not in present:
                    block = atom_block_bsr(M, a, b)
                    assert block.shape == M.blocksize
                    assert np.all(block == 0)
                    checked = True
                    break

            if checked:
                break

        assert checked


def test_sparse_values_are_finite(data):
    assert np.all(np.isfinite(data.H.data))
    assert np.all(np.isfinite(data.S.data))


def test_sparse_structure_is_frozen(data):
    for M in [data.H, data.S]:
        assert not M.data.flags.writeable
        assert not M.indices.flags.writeable
        assert not M.indptr.flags.writeable


def test_matrix_dimension_matches_atom_channels(data):
    expected = data.metadata.natoms * data.basis.nchannels

    assert data.H.shape == (expected, expected)
    assert data.S.shape == (expected, expected)


def test_basis_map_is_permutation(data):
    perm = data.basis.atom_basis.ravel()

    assert len(perm) == data.metadata.nbasis
    assert set(map(int, perm)) == set(range(data.metadata.nbasis))


def test_atom_basis_roundtrip(data):
    for atom in range(data.metadata.natoms):
        for channel in range(data.basis.nchannels):
            alpha = data.basis.atom_basis[atom, channel]

            assert data.metadata.atom_of_basis[alpha] == atom
            assert data.metadata.channel_of_basis[alpha] == channel


def test_coupled_atom_blocks_sorted_by_norm(data, sample_atoms):
    for a in sample_atoms:
        _atoms_b, _blocks, _distances, norms, _dR = coupled_atom_blocks(
            data.H,
            data.metadata.positions,
            int(a),
        )

        assert np.all(norms[:-1] >= norms[1:])


def test_coupled_atom_blocks_distances(data, sample_atoms):
    for a in sample_atoms:
        a = int(a)
        atoms_b, _blocks, distances, _norms, dR = coupled_atom_blocks(
            data.H,
            data.metadata.positions,
            a,
        )

        Ra = data.metadata.positions[a]

        for i, b in enumerate(atoms_b):
            expected_dR = data.metadata.positions[int(b)] - Ra
            expected_distance = np.linalg.norm(expected_dR)

            assert np.allclose(dR[i], expected_dR)
            assert np.isclose(distances[i], expected_distance)


def test_coupled_atom_blocks_norms(data, sample_atoms):
    for a in sample_atoms:
        _atoms_b, blocks, _distances, norms, _dR = coupled_atom_blocks(
            data.H,
            data.metadata.positions,
            int(a),
        )

        assert np.allclose(norms, np.linalg.norm(blocks, axis=(1, 2)))


def test_overlap_diagonal_blocks_are_finite(data, sample_atoms):
    for a in sample_atoms:
        a = int(a)
        Saa = atom_block_bsr(data.S, a, a)

        assert np.all(np.isfinite(Saa))
        assert np.linalg.norm(Saa) > 0


def test_overlap_diagonal_entries_positive(data, sample_atoms):
    for a in sample_atoms:
        Saa = atom_block_bsr(data.S, int(a), int(a))
        diag = np.diag(Saa)

        assert np.all(diag > 0)


def test_H_global_hermitian_defect(data):
    defect = data.H - data.H.getH()
    rel = np.linalg.norm(defect.data) / np.linalg.norm(data.H.data)

    # Raw BigDFT H is not exactly Hermitian, but should be close.
    assert rel < 1e-5


def test_S_global_hermitian_defect(data):
    defect = data.S - data.S.getH()

    if defect.nnz == 0:
        return

    rel = np.linalg.norm(defect.data) / np.linalg.norm(data.S.data)
    assert rel < 1e-10


@pytest.mark.xfail(reason="Raw BigDFT H is not exactly block-Hermitian before symmetrisation")
def test_H_block_hermitian_raw(data, sample_atoms):
    check_block_hermitian(data.H, sample_atoms)


def test_S_block_hermitian(data, sample_atoms):
    check_block_hermitian(data.S, sample_atoms, atol=1e-10, rtol=1e-8)


def test_symmetrised_H_global_hermitian(data):
    H_herm = 0.5 * (data.H + data.H.getH())
    defect = H_herm - H_herm.getH()

    if defect.nnz:
        assert np.max(np.abs(defect.data)) < 1e-12


def test_symmetrised_H_block_hermitian(data, sample_atoms):
    H_herm = 0.5 * (data.H + data.H.getH())

    check_block_hermitian(
        H_herm,
        sample_atoms,
        atol=1e-12,
        rtol=1e-10,
    )


def test_sparse_dataset_loads_bigdft_log_yaml() -> None:
    dataset = SparseDataset.load("test_run/run_dir/data")

    assert isinstance(dataset.bigdft_log, dict)
    assert "Last Iteration" in dataset.bigdft_log or "Ground State Optimization" in dataset.bigdft_log
