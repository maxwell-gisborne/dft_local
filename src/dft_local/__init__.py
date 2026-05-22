from dataclasses import dataclass, replace
from pathlib import Path
from scipy.io import mmread
from scipy.sparse import spmatrix
import numpy as np
from numpy import array, arange, ndarray
from collections import Counter


@dataclass(frozen=True)
class Units:
    E: float
    L: float
    e: float
    hbar: float
    name: str
    comment: str = ""

    def __reper__(self):
        return f'Units({self.name})'


AU = Units(E = 1,
           L = 1,
           e = 1,
           hbar = 1,
           name = 'bhor',
           comment = "this is the unit on disk")

eVag = Units(
        E = 27.21138386,  # Hatrees in eV
        L = 0.52917721092,  # bhor radius in to Angstroms
        e = 1.602e-19,      # charge on electron in Colombs
        hbar = 6.582e-16,  # hbar in eV•s
        name = 'angstroem',
        )


def file_exists(path:Path):
    assert path.exists(), path
    return path

@dataclass(frozen=True)
class Metafile:
    path:Path
    units:Units
    positions:ndarray | None = None           # atom-index => ionic cartisian position
    symbols:ndarray | None = None             # atom-index => symbol-index
    atom:ndarray | None = None                # alpha => atom-index
    channel:ndarray | None = None               # alpha => channel-index

    basis_range: ndarray | None = None   # alpha-index
    atom_range: ndarray | None = None    # atom-index
    symbol_range: ndarray | None = None  # type/symbol-index, maybe optional
    
    __call__ = replace

    def load(self):
        file_exists(self.path)
        positions = []
        symbols = []
        symbols_dictionary = []
        atom = []
        chanl = []
        with open(self.path, "r") as ifile:
            # Read the first line
            matinfo = next(ifile).split()
            matdim, natoms, ntypes = [int(x) for x in matinfo[:3]]

            symbol_index = range(ntypes)
            atom_index = range(natoms)
            alpha = range(matdim)

            next(ifile)          # Units
            line = next(ifile)   # skip geocode
            line = next(ifile)   # skip shift

            # generate symbols_dictionary
            for i, line in zip(symbol_index, ifile):
                nz, nelpsp, name = line.split()[:3]
                symbols_dictionary.append(name)

            # generate positions and symbols
            for i, line in zip(atom_index, ifile):
                sym_indx, x, y, z, *_ = line.split()
                symbols.append(symbols_dictionary[int(sym_indx) - 1])
                positions.append([float(x), float(y), float(z)])

            # generate SFs
            channel_count = {}
            for i, line in zip(alpha, ifile):
                atom_index = int(line.split()[0]) - 1
                atom.append(atom_index)
                cc = channel_count[atom_index] = (channel_count.get(atom_index) or 0) + 1
                chanl.append(cc-1)


            return self(
                positions = array(positions) * self.units.L,
                symbols = array(symbols),
                atom = array(atom),
                channel = array(chanl),
                basis_range = arange(matdim),
                atom_range = arange(natoms),
                symbol_range = arange(ntypes),
            )


def make_atom_basis_array(meta):
    natoms = len(meta.positions)
    counts = np.bincount(meta.atom, minlength=natoms)

    if not np.all(counts == counts[0]):
        raise ValueError(f"Unequal channel counts: {np.unique(counts, return_counts=True)}")

    nchan = counts[0]
    atom_basis = np.empty((natoms, nchan), dtype=int)

    for alpha, a in enumerate(meta.atom):
        c = meta.channel[alpha]
        atom_basis[a, c] = alpha

    return atom_basis


def atom_block(M, atom_basis, a: int, b: int):
    ia = atom_basis[a]
    ib = atom_basis[b]
    return M[ia[:, None], ib].toarray()


def coupled_atoms(M, meta, atom_basis, a: int):
    ia = atom_basis[a]
    row_block = M[ia, :].tocsr()

    basis_cols = row_block.nonzero()[1]
    atoms_b = np.unique(meta.atom[basis_cols])

    out = []
    Ra = meta.positions[a]

    for b in atoms_b:
        block = atom_block(M, atom_basis, a, b)
        norm = np.linalg.norm(block)

        dR = meta.positions[b] - Ra
        dist = np.linalg.norm(dR)

        out.append((b, dist, norm, dR))

    out.sort(key=lambda x: x[2], reverse=True)
    return out

def old_atom_basis_indices(meta, a):
    idx = np.where(meta.atom == a)[0]
    return idx[np.argsort(meta.channel[idx])]


def old_atom_block(M, meta, a: int, b: int):
    ia = old_atom_basis_indices(meta, a)
    ib = old_atom_basis_indices(meta, b)
    return M[ia[:, None], ib].toarray()

def old_coupled_atoms(M, meta, a: int):
    ia = old_atom_basis_indices(meta, a)
    row_block = M[ia, :].tocsr()

    basis_cols = row_block.nonzero()[1]
    atoms_b = np.unique(meta.atom[basis_cols])

    out = []
    Ra = meta.positions[a]

    for b in atoms_b:
        block = old_atom_block(M, meta, a, b)
        norm = np.linalg.norm(block)
        dist = np.linalg.norm(meta.positions[b] - Ra)
        out.append((b, dist, norm, meta.positions[b] - Ra))

    out.sort(key=lambda x: x[2], reverse=True)
    return out

def coupled_atoms_table(M, meta, a: int):
    import pandas as pd
    rows = []
    for b, dist, norm, dR in coupled_atoms(M, meta, a):
        rows.append({
            "atom_b": b,
            "symbol": meta.symbols[b],
            "distance": dist,
            "block_norm": norm,
            "dRx": dR[0],
            "dRy": dR[1],
            "dRz": dR[2],
        })
    return pd.DataFrame(rows)

def coupled_atoms_table_by_distance(M, meta, a: int):
    import pandas as pd
    rows = []
    data = coupled_atoms(M, meta, a)
    data.sort(key=lambda x: x[1])

    for b, dist, norm, dR in data:
        rows.append({
            "atom_b": b,
            "symbol": meta.symbols[b],
            "distance": dist,
            "block_norm": norm,
            "dRx": dR[0],
            "dRy": dR[1],
            "dRz": dR[2],
        })

    return pd.DataFrame(rows)


@dataclass(frozen=True)
class BasisIndex:
    alpha: int
    atom: int
    channel: int
    symbol: str
    position: ndarray
           

@dataclass(frozen=True)
class Data:
    dataroot:Path
    units = eVag
    H_sparse:spmatrix = None
    S_sparse:spmatrix = None
    meta:Metafile = None

    __call__ = replace
    def load_sparse_data(self):
        metadata = Metafile(path = file_exists(self.dataroot) / 'sparsematrix_metadata.dat',
                            units = self.units,
                            ).load()
        return self(
            H_sparse = mmread( self.dataroot / "hamiltonian_sparse.mtx").tocsr() * self.units.E,
            S_sparse = mmread(self.dataroot / "overlap_sparse.mtx").tocsr(),
            meta = metadata,
        ).load_sparse_data_check()

    def load_sparse_data_check(self):
        assert self.H_sparse.shape[0] == self.H_sparse.shape[1]
        assert self.S_sparse.shape == self.S_sparse.shape
        assert self.H_sparse.shape[0] == len(self.meta.atom)
        assert len(self.meta.channel) == len(self.meta.atom)
        assert self.meta.positions.shape[0] == len(self.meta.symbols)
        assert np.all(self.meta.atom >= 0)
        assert np.all(self.meta.atom < len(self.meta.positions))

        counts = np.bincount(self.meta.atom)
        assert np.all(counts == counts[0])
        nchan = counts[0]
        assert nchan == 4

        return self


data = Data(dataroot = Path('./test_run/run_dir/data')).load_sparse_data()
meta = data.meta
H = data.H_sparse
S = data.S_sparse

