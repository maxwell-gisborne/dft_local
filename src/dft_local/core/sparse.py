"""Sparse block-matrix helpers for the dft_local package."""

from __future__ import annotations

from scipy.sparse import bsr_matrix


def block_row_raw(M: bsr_matrix, atom: int):
    """Return raw BSR column indices and block data for one atom row."""

    start = M.indptr[atom]
    stop = M.indptr[atom + 1]
    return M.indices[start:stop], M.data[start:stop]


import numpy as np

from dft_local.core.numerics import freeze_array


def block_view_bsr(M: bsr_matrix, a: int, b: int) -> np.ndarray:
    """Return read-only block view if present, otherwise a new zero block."""

    start = M.indptr[a]
    stop = M.indptr[a + 1]

    row_cols = M.indices[start:stop]
    row_blocks = M.data[start:stop]

    matches = np.flatnonzero(row_cols == b)

    if len(matches) == 0:
        return freeze_array(np.zeros(M.blocksize, dtype=M.dtype))

    if len(matches) > 1:
        raise ValueError(f"Duplicate block entry for a={a}, b={b}")

    return row_blocks[int(matches[0])]


def coupled_atoms_table(M, data, a: int):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "atom_b": block.atom_b,
                "symbol": data.metadata.symbols[block.atom_b],
                "distance": block.distance,
                "block_norm": block.norm,
                "block": block.block,
                "dRx": block.dR[0],
                "dRy": block.dR[1],
                "dRz": block.dR[2],
            }
            for block in data.coupled_atoms(M, a)
        ]
    )


def coupled_atoms_table_by_distance(M, data, a: int):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "atom_b": block.atom_b,
                "symbol": data.metadata.symbols[block.atom_b],
                "distance": block.distance,
                "block_norm": block.norm,
                "block": block.block,
                "dRx": block.dR[0],
                "dRy": block.dR[1],
                "dRz": block.dR[2],
            }
            for block in sorted(data.coupled_atoms(M, a), key=lambda block: block.distance)
        ]
    )
