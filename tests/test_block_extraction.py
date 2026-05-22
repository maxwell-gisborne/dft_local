import numpy as np
import pytest

from dft_local import SparseDataset, AtomBlockMatrix


@pytest.fixture(scope="session")
def data():
    return SparseDataset.load("./test_run/run_dir/data")


@pytest.fixture(scope="session")
def H_blocks(data):
    return AtomBlockMatrix.from_sparse(data.H, data)


@pytest.fixture(scope="session")
def S_blocks(data):
    return AtomBlockMatrix.from_sparse(data.S, data)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def sample_atoms(data, rng):
    n = data.metadata.natoms
    size = min(100, n)
    return rng.choice(n, size=size, replace=False)



def assert_blocks_close(A, B, *, atol=1e-8, rtol=1e-6, context=""):
    if not np.allclose(A, B, atol=atol, rtol=rtol):
        err = np.linalg.norm(A - B)
        scale = max(np.linalg.norm(A), np.linalg.norm(B), 1.0)
        rel = err / scale
        raise AssertionError(
            f"block mismatch {context}: norm error={err}, relative={rel}"
        )

def check_raw_row_against_atom_block(data, block_matrix, M, a, atol=1e-12):
    atoms_b, blocks = block_matrix.atom_block_row_raw(a)

    for b, block in zip(atoms_b, blocks):
        direct = data.atom_block(M, a, int(b))
        assert_blocks_close(
            block,
            direct,
            atol=atol,
            context=f"a={a}, b={int(b)}",
        )



def check_pretty_against_raw(block_matrix, a, atol=1e-12):
    atoms_b, blocks = block_matrix.atom_block_row_raw(a)
    raw = {int(b): block for b, block in zip(atoms_b, blocks)}

    pretty = block_matrix.coupled_atom_blocks(a)

    # Works if pretty returns AtomBlock dataclasses.
    pretty_blocks = {int(x.atom_b): x.block for x in pretty}

    assert set(raw) == set(pretty_blocks)

    for b in raw:
        assert_blocks_close(
            raw[b],
            pretty_blocks[b],
            atol=atol,
            context=f"a={a}, b={b}",
        )


def check_bsr_roundtrip(data, block_matrix, M, atol=1e-12):
    perm = data.basis.atom_basis.ravel()

    M_atom_ordered = M[perm, :][:, perm].tocsr()
    M_from_bsr = block_matrix.M.tocsr()

    diff = M_atom_ordered - M_from_bsr

    if diff.nnz == 0:
        return

    max_err = np.max(np.abs(diff.data))
    assert max_err <= atol


def check_block_hermitian(data, block_matrix, M, atoms, atol=1e-7, rtol=1e-5):
    for a in atoms:
        a = int(a)
        atoms_b, blocks = block_matrix.atom_block_row_raw(a)

        for b, Mab in zip(atoms_b, blocks):
            b = int(b)
            Mba = data.atom_block(M, b, a)

            assert_blocks_close(
                Mab,
                Mba.conj().T,
                atol=atol,
                context=f"a={a}, b={b}",
            )


def test_dataset_shapes(data):
    assert data.H.shape[0] == data.H.shape[1]
    assert data.S.shape == data.H.shape
    assert data.H.shape[0] == data.metadata.nbasis
    assert data.basis.nchannels == 4
    assert data.basis.atom_basis.shape == (
        data.metadata.natoms,
        data.basis.nchannels,
    )


def test_H_bsr_roundtrip(data, H_blocks):
    check_bsr_roundtrip(data, H_blocks, data.H)


def test_S_bsr_roundtrip(data, S_blocks):
    check_bsr_roundtrip(data, S_blocks, data.S)


def test_H_raw_rows_match_direct_atom_blocks(data, H_blocks, sample_atoms):
    for a in sample_atoms:
        check_raw_row_against_atom_block(data, H_blocks, data.H, int(a))


def test_S_raw_rows_match_direct_atom_blocks(data, S_blocks, sample_atoms):
    for a in sample_atoms:
        check_raw_row_against_atom_block(data, S_blocks, data.S, int(a))


def test_H_pretty_rows_match_raw_rows(H_blocks, sample_atoms):
    for a in sample_atoms:
        check_pretty_against_raw(H_blocks, int(a))


def test_S_pretty_rows_match_raw_rows(S_blocks, sample_atoms):
    for a in sample_atoms:
        check_pretty_against_raw(S_blocks, int(a))



@pytest.mark.xfail(reason="Raw BigDFT H is not exactly block-Hermitian before symmetrisation")
def test_H_block_hermitian(data, H_blocks, sample_atoms):
    check_block_hermitian(data, H_blocks, data.H, sample_atoms)

def test_H_global_hermitian_defect(data):
    defect = data.H - data.H.getH()
    rel = np.linalg.norm(defect.data) / np.linalg.norm(data.H.data)
    assert rel < 1e-5

def test_S_global_hermitian_defect(data):
    defect = data.S - data.S.getH()
    rel = np.linalg.norm(defect.data) / np.linalg.norm(data.S.data)

    assert rel < 1e-10

def test_symmetrised_H_block_hermitian(data, sample_atoms):
    H_herm = 0.5 * (data.H + data.H.getH())
    H_herm_blocks = AtomBlockMatrix.from_sparse(H_herm, data)

    check_block_hermitian(
        data,
        H_herm_blocks,
        H_herm,
        sample_atoms,
        atol=1e-12,
        rtol=1e-10,
    )

def test_S_block_hermitian(data, S_blocks, sample_atoms):
    check_block_hermitian(data, S_blocks, data.S, sample_atoms)


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

def test_H_bsr_has_expected_blocksize(H_blocks, data):
    assert H_blocks.M.blocksize == (data.basis.nchannels, data.basis.nchannels)
    assert H_blocks.M.shape == data.H.shape


def test_S_bsr_has_expected_blocksize(S_blocks, data):
    assert S_blocks.M.blocksize == (data.basis.nchannels, data.basis.nchannels)
    assert S_blocks.M.shape == data.S.shape

def test_raw_rows_have_matching_lengths(H_blocks, sample_atoms):
    for a in sample_atoms:
        atoms_b, blocks = H_blocks.atom_block_row_raw(int(a))

        assert len(atoms_b) == len(blocks)
        assert blocks.ndim == 3
        assert blocks.shape[1:] == H_blocks.M.blocksize


def test_coupled_atom_blocks_sorted_by_norm(H_blocks, sample_atoms):
    for a in sample_atoms:
        rows = H_blocks.coupled_atom_blocks(int(a))
        norms = [x.norm for x in rows]

        assert norms == sorted(norms, reverse=True)


def test_coupled_atom_blocks_distances(data, H_blocks, sample_atoms):
    for a in sample_atoms:
        a = int(a)
        Ra = data.metadata.positions[a]

        for x in H_blocks.coupled_atom_blocks(a):
            expected_dR = data.metadata.positions[x.atom_b] - Ra
            expected_distance = np.linalg.norm(expected_dR)

            assert np.allclose(x.dR, expected_dR)
            assert np.isclose(x.dist, expected_distance)


def test_coupled_atom_blocks_norms(H_blocks, sample_atoms):
    for a in sample_atoms:
        for x in H_blocks.coupled_atom_blocks(int(a)):
            assert np.isclose(x.norm, np.linalg.norm(x.block))

def test_overlap_diagonal_blocks_are_finite(data, S_blocks, sample_atoms):
    for a in sample_atoms:
        a = int(a)
        direct = data.atom_block(data.S, a, a)

        assert np.all(np.isfinite(direct))
        assert np.linalg.norm(direct) > 0

def test_overlap_diagonal_entries_positive(data, sample_atoms):
    for a in sample_atoms:
        Saa = data.atom_block(data.S, int(a), int(a))
        diag = np.diag(Saa)

        assert np.all(diag > 0)


def test_sparse_values_are_finite(data):
    assert np.all(np.isfinite(data.H.data))
    assert np.all(np.isfinite(data.S.data))

def test_matrix_dimension_matches_atom_channels(data):
    expected = data.metadata.natoms * data.basis.nchannels

    assert data.H.shape == (expected, expected)
    assert data.S.shape == (expected, expected)



