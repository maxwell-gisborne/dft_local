from typing import Self
from dataclasses import dataclass, replace
from pathlib import Path
from scipy.io import mmread
from scipy.sparse import spmatrix, csr_matrix
import numpy as np
from numpy import array, arange, ndarray
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Units:
    E: float
    L: float
    e: float
    hbar: float
    name: str
    comment: str = ""

    def __repr__(self):
        return f'Units({self.name})'


AU = Units(E = 1,
           L = 1,
           e = 1,
           hbar = 1,
           name = 'bohr',
           comment = "this is the unit on disk")

eVag = Units(
        E = 27.21138386,  # Hatrees in eV
        L = 0.52917721092,  # Bohr radius in to Angstrom
        e = 1.602e-19,      # charge on electron in Colombs
        hbar = 6.582e-16,  # hbar in eV•s
        name = 'angstroem',
        )


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Expected file, got: {path}")
    return path


def require_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise ValueError(f"Expected directory, got: {path}")
    return path

@dataclass(frozen=True)
class SparseMetadata:
    positions: FloatArray              # atom -> cartesian position
    symbols: NDArray[np.str_]          # atom -> symbol
    atom_of_basis: IntArray            # alpha -> atom
    channel_of_basis: IntArray         # alpha -> local channel
    symbols_dictionary: tuple[str, ...]

    @property
    def nbasis(self) -> int:
        return len(self.atom_of_basis)

    @property
    def natoms(self) -> int:
        return len(self.positions)

    @property
    def nchannels(self) -> int:
        counts = np.bincount(self.atom_of_basis, minlength=self.natoms)
        if not np.all(counts == counts[0]):
            raise ValueError(f"Unequal channel counts: {np.unique(counts, return_counts=True)}")
        return int(counts[0])

    @classmethod
    def load(cls, path: Path, units: Units) -> Self:
        require_file(path)

        positions: list[list[float]] = []
        symbols: list[str] = []
        symbols_dictionary: list[str] = []
        atom_of_basis: list[int] = []
        channel_of_basis: list[int] = []

        with path.open("r") as f:
            matinfo = next(f).split()
            matdim, natoms, ntypes = [int(x) for x in matinfo[:3]]

            next(f)  # units line
            next(f)  # geocode
            next(f)  # shift

            for _ in range(ntypes):
                _nz, _nelpsp, name, *_ = next(f).split()
                symbols_dictionary.append(name)

            for _ in range(natoms):
                sym_index, x, y, z, *_ = next(f).split()
                symbols.append(symbols_dictionary[int(sym_index) - 1])
                positions.append([float(x), float(y), float(z)])

            channel_count: dict[int, int] = {}
            for _ in range(matdim):
                atom_index = int(next(f).split()[0]) - 1
                channel = channel_count.get(atom_index, 0)
                channel_count[atom_index] = channel + 1

                atom_of_basis.append(atom_index)
                channel_of_basis.append(channel)

        return cls(
            positions=np.asarray(positions, dtype=np.float64) * units.L,
            symbols=np.asarray(symbols),
            atom_of_basis=np.asarray(atom_of_basis, dtype=np.int64),
            channel_of_basis=np.asarray(channel_of_basis, dtype=np.int64),
            symbols_dictionary=tuple(symbols_dictionary),
        ).validate(expected_nbasis=matdim, expected_natoms=natoms)


    def validate(self, expected_nbasis: int | None = None, expected_natoms: int | None = None) -> Self:
        if expected_nbasis is not None and self.nbasis != expected_nbasis:
            raise ValueError(f"Basis count mismatch: {self.nbasis} != {expected_nbasis}")

        if expected_natoms is not None and self.natoms != expected_natoms:
            raise ValueError(f"Atom count mismatch: {self.natoms} != {expected_natoms}")

        if self.positions.shape != (self.natoms, 3):
            raise ValueError(f"Bad positions shape: {self.positions.shape}")

        if len(self.symbols) != self.natoms:
            raise ValueError("symbols and positions have different lengths")

        if np.any(self.atom_of_basis < 0) or np.any(self.atom_of_basis >= self.natoms):
            raise ValueError("atom_of_basis contains invalid atom indices")

        if len(self.channel_of_basis) != self.nbasis:
            raise ValueError("channel_of_basis and atom_of_basis have different lengths")

        return self



@dataclass(frozen=True)
class BasisMap:
    atom_basis: IntArray        # shape: (natoms, nchannels)
    atom_of_basis: IntArray
    channel_of_basis: IntArray

    @classmethod
    def from_metadata(cls, metadata: SparseMetadata) -> Self:
        natoms = metadata.natoms
        counts = np.bincount(metadata.atom_of_basis, minlength=natoms)

        if not np.all(counts == counts[0]):
            raise ValueError(f"Unequal channel counts: {np.unique(counts, return_counts=True)}")

        nchannels = int(counts[0])
        atom_basis = np.empty((natoms, nchannels), dtype=np.int64)

        for alpha, atom in enumerate(metadata.atom_of_basis):
            channel = metadata.channel_of_basis[alpha]
            atom_basis[atom, channel] = alpha

        return cls(
            atom_basis=atom_basis,
            atom_of_basis=metadata.atom_of_basis,
            channel_of_basis=metadata.channel_of_basis,
        )

    @property
    def natoms(self) -> int:
        return self.atom_basis.shape[0]

    @property
    def nchannels(self) -> int:
        return self.atom_basis.shape[1]

    def basis_indices(self, atom: int) -> IntArray:
        return self.atom_basis[atom]

@dataclass(frozen=True)
class AtomBlock:
    atom_b: int
    dist: float
    norm: float
    dR: np.ndarray
    block: np.ndarray

@dataclass(frozen=True)
class SparseDataset:
    root: Path
    units: Units
    metadata: SparseMetadata
    basis: BasisMap
    H: csr_matrix
    S: csr_matrix

    @classmethod
    def load(cls, root: Path, units: Units = eVag) -> Self:
        root = require_dir(root)

        metadata = SparseMetadata.load(root / "sparsematrix_metadata.dat", units=units)
        basis = BasisMap.from_metadata(metadata)

        H = mmread(require_file(root / "hamiltonian_sparse.mtx")).tocsr() * units.E
        S = mmread(require_file(root / "overlap_sparse.mtx")).tocsr()

        return cls(root=root, units=units, metadata=metadata, basis=basis, H=H, S=S).validate()

    def validate(self) -> Self:
        if self.H.shape[0] != self.H.shape[1]:
            raise ValueError(f"H is not square: {self.H.shape}")

        if self.S.shape != self.H.shape:
            raise ValueError(f"S shape {self.S.shape} != H shape {self.H.shape}")

        if self.H.shape[0] != self.metadata.nbasis:
            raise ValueError(f"Matrix dimension {self.H.shape[0]} != nbasis {self.metadata.nbasis}")

        if self.basis.nchannels != 4:
            raise ValueError(f"Expected 4 channels per atom, got {self.basis.nchannels}")

        return self


    def atom_block(self, M: spmatrix, a: int, b: int) -> np.ndarray:
        ia = self.basis.basis_indices(a)
        ib = self.basis.basis_indices(b)
        return M[ia[:, None], ib].toarray()

    # def coupled_atoms(self, M: spmatrix, a: int) -> list[tuple[int, float, float, np.ndarray]]:
    #     ia = self.basis.basis_indices(a)
    #     row_block = M[ia, :].tocsr()

    #     basis_cols = row_block.nonzero()[1]
    #     atoms_b = np.unique(self.metadata.atom_of_basis[basis_cols])

    #     out: list[tuple[int, float, float, np.ndarray]] = []
    #     Ra = self.metadata.positions[a]

    #     for b in atoms_b:
    #         block = self.atom_block(M, a, int(b))
    #         norm = float(np.linalg.norm(block))
    #         dR = self.metadata.positions[b] - Ra
    #         dist = float(np.linalg.norm(dR))
    #         out.append((int(b), dist, norm, dR))

    #     out.sort(key=lambda x: x[2], reverse=True)
    #     return out

    def coupled_atoms( self, M: spmatrix, a: int, ) -> list[AtomBlock]:
        """
        Return coupled atom blocks from atom a.

        Each entry is:
            AtomBlock(atom_b, distance, block_norm, dR, block)

        This avoids repeated sparse slicing by extracting the 4-row sparse block once.
        """
        ia = self.basis.basis_indices(a)
        row_block = M[ia, :].tocoo()

        atom_of_col = self.metadata.atom_of_basis[row_block.col]
        channel_of_col = self.metadata.channel_of_basis[row_block.col]

        blocks: dict[int, np.ndarray] = {}

        for local_row, atom_b, channel_b, value in zip(
            row_block.row,
            atom_of_col,
            channel_of_col,
            row_block.data,
        ):
            atom_b = int(atom_b)

            block = blocks.get(atom_b)
            if block is None:
                block = np.zeros((self.basis.nchannels, self.basis.nchannels), dtype=M.dtype)
                blocks[atom_b] = block

            block[int(local_row), int(channel_b)] += value

        Ra = self.metadata.positions[a]
        out: list[AtomBlock] = []

        for atom_b, block in blocks.items():
            dR = self.metadata.positions[atom_b] - Ra
            dist = float(np.linalg.norm(dR))
            norm = float(np.linalg.norm(block))
            out.append(AtomBlock(atom_b = atom_b,
                                 dist = dist,
                                 norm = norm,
                                 dR = dR,
                                 block = block))

        out.sort(key=lambda block: block.norm, reverse=True)
        return out



def coupled_atoms_table(M, data: SparseDataset, a: int):
    import pandas as pd
    return pd.DataFrame(
        [
            {
                "atom_b": block.atom_b,
                "symbol": data.metadata.symbols[block.atom_b],
                "distance": block.dist,
                "block_norm": block.norm,
                "block": block.block,
                "dRx": block.dR[0],
                "dRy": block.dR[1],
                "dRz": block.dR[2],
            }
            for block in data.coupled_atoms(M, a)
        ]
    )


def coupled_atoms_table_by_distance(M, data: SparseDataset, a: int):
    import pandas as pd
    return pd.DataFrame(
        [
            {
                "atom_b": block.atom_b,
                "symbol": data.metadata.symbols[block.atom_b],
                "distance": block.dist,
                "block_norm": block.norm,
                "block": block.block,
                "dRx": block.dR[0],
                "dRy": block.dR[1],
                "dRz": block.dR[2],
            }
            for block in sorted(data.coupled_atoms(M, a), key = lambda block:block.dist)
        ]
    )


@dataclass(frozen=True)
class BasisIndex:
    alpha: int
    atom: int
    channel: int
    symbol: str
    position: ndarray
           


data = SparseDataset.load(Path("./test_run/run_dir/data"))
meta = data.metadata
H = data.H
S = data.S



print(coupled_atoms_table(H, data, 0))
