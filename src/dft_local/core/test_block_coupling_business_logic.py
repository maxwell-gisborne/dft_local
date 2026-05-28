# Copied from repository-level tests during package migration.
# These tests should target dft_local only.

import numpy as np
import pytest

from dft_local.core.sparse import block_row_raw, block_view_bsr, coupled_atoms_table, coupled_atoms_table_by_distance



def test_atom_block_matches_raw_bsr_row(data, sample_atoms, assert_block_close):
    """
    Public block_view_bsr() must agree with direct BSR row lookup.
    This catches stale CSR-style atom_block implementations.
    """
    for a in sample_atoms:
        a = int(a)
        atoms_b, blocks = block_row_raw(data.H, a)

        for b, block in zip(atoms_b, blocks):
            got = block_view_bsr(data.H, a, int(b))
            assert_block_close(got, block, context=f"a={a}, b={int(b)}")


def test_atom_block_returns_zero_for_missing_block(data, sample_atoms):
    """
    Missing atom-atom blocks should return a zero 4x4 block.
    """
    n = data.metadata.natoms

    for a in sample_atoms[:10]:
        a = int(a)
        atoms_b, _ = block_row_raw(data.H, a)
        present = set(map(int, atoms_b))

        missing = next(b for b in range(n) if b not in present)

        block = block_view_bsr(data.H, a, missing)

        assert block.shape == data.H.blocksize
        assert np.allclose(block, 0.0)


def test_coupled_atom_blocks_runs_and_returns_atomblocks(data, sample_atoms):
    """
    This explicitly calls the public method. It catches the atom/a NameError.
    """
    for a in sample_atoms:
        rows = data.coupled_atoms(data.H, int(a))

        assert isinstance(rows, list)

        for row in rows:
            assert isinstance(row.atom_b, int)
            assert isinstance(row.distance, float)
            assert isinstance(row.norm, float)
            assert row.dR.shape == (3,)
            assert row.block.shape == data.H.blocksize


def test_coupled_atom_blocks_matches_raw_bsr_row(data, sample_atoms, assert_block_close):
    """
    Public coupled_atom_blocks() must return exactly the same blocks as raw BSR,
    only sorted by norm.
    """
    for a in sample_atoms:
        a = int(a)

        atoms_b, blocks = block_row_raw(data.H, a)
        raw = {int(b): block for b, block in zip(atoms_b, blocks)}

        pretty = data.coupled_atoms(data.H, a)
        pretty_by_atom = {row.atom_b: row for row in pretty}

        assert set(pretty_by_atom) == set(raw)

        for b, block in raw.items():
            row = pretty_by_atom[b]

            assert_block_close(row.block, block, context=f"a={a}, b={b}")

            expected_dR = data.metadata.positions[b] - data.metadata.positions[a]
            expected_dist = np.linalg.norm(expected_dR)
            expected_norm = np.linalg.norm(block)

            assert np.allclose(row.dR, expected_dR)
            assert np.isclose(row.distance, expected_dist)
            assert np.isclose(row.norm, expected_norm)


def test_coupled_atom_blocks_sorted_by_norm(data, sample_atoms):
    for a in sample_atoms:
        rows = data.coupled_atoms(data.H, int(a))
        norms = [row.norm for row in rows]

        assert norms == sorted(norms, reverse=True)


def test_coupled_atoms_alias_if_present(data, sample_atoms, assert_block_close):
    """
    If coupled_atoms still exists, it must behave like coupled_atom_blocks.
    Delete this test if coupled_atoms is removed.
    """
    if not hasattr(data, "coupled_atoms"):
        pytest.skip("coupled_atoms removed")

    for a in sample_atoms[:10]:
        a = int(a)

        old = data.coupled_atoms(data.H, a)
        new = data.coupled_atoms(data.H, a)

        assert len(old) == len(new)

        old_by_atom = {row.atom_b: row for row in old}
        new_by_atom = {row.atom_b: row for row in new}

        assert set(old_by_atom) == set(new_by_atom)

        for b in old_by_atom:
            assert_block_close(old_by_atom[b].block, new_by_atom[b].block)


def test_coupled_atoms_table_runs(data, sample_atoms):
    """
    Catches table functions accidentally calling old CSR-only coupled_atoms().
    """
    a = int(sample_atoms[0])

    table = coupled_atoms_table(data.H, data, a)

    expected_columns = {
        "atom_b",
        "symbol",
        "distance",
        "block_norm",
        "block",
        "dRx",
        "dRy",
        "dRz",
    }

    assert expected_columns <= set(table.columns)
    assert len(table) == len(data.coupled_atoms(data.H, a))


def test_coupled_atoms_table_by_distance_is_sorted(data, sample_atoms):
    a = int(sample_atoms[0])

    table = coupled_atoms_table_by_distance(data.H, data, a)
    distances = table["distance"].to_numpy()

    assert np.all(distances[:-1] <= distances[1:])
