"""Sparse dataset loading and atom-block inspection for the dft_local package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
from numpy.typing import NDArray
from scipy.io import mmread
from scipy.sparse import bsr_matrix

from dft_local.core.numerics import FloatArray, IntArray, Units, eVag, freeze_array
from dft_local.core.sparse import block_row_raw
from dft_local.core.units import ATOMIC_UNITS, DIMENSIONLESS, ENERGY, EV_ANGSTROM_FS, LENGTH, SI_UNITS, UnitContext, qarray


def freeze_bsr(M):
    M.data.flags.writeable = False
    M.indices.flags.writeable = False
    M.indptr.flags.writeable = False
    return M

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BlockArray = NDArray[np.float64] | NDArray[np.complex128]
MatchingStrategy = Literal["state_overlap", "energy_predict"]

AtomPositions = Annotated[FloatArray, qarray(LENGTH, ("atom", "cartesian"), role="atom positions")]
EnergySparseMatrix = Annotated[bsr_matrix, qarray(ENERGY, ("basis", "basis"), role="Hamiltonian")]
DimensionlessSparseMatrix = Annotated[bsr_matrix, qarray(DIMENSIONLESS, ("basis", "basis"), role="overlap matrix")]


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


def unit_context_from_legacy_units(units: Units) -> UnitContext:
    """Best-effort bridge from the legacy Units object to core UnitContext."""

    if units == eVag or getattr(units, "name", "") == "angstroem":
        return EV_ANGSTROM_FS

    if units == AU or getattr(units, "name", "") == "bohr":
        return ATOMIC_UNITS

    return SI_UNITS


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
    working_unit_context: UnitContext
    positions: AtomPositions          # atom -> cartesian position
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
        path = Path(path)
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
            working_unit_context=unit_context_from_legacy_units(units),
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
    distance: float
    norm: float
    dR: np.ndarray
    block: np.ndarray

def atom_ordered_bsr(M, basis):
    perm = basis.atom_basis.ravel()
    out = M[perm, :][:, perm].tobsr(blocksize=(basis.nchannels, basis.nchannels))
    if out.indptr.size != basis.natoms + 1:
        raise ValueError("BSR does not have one block row per atom")
    return out




@dataclass(frozen=True)
class SparseDataset:
    root: Path
    units: Units
    disk_unit_context: UnitContext
    working_unit_context: UnitContext
    metadata: SparseMetadata
    basis: BasisMap
    H: EnergySparseMatrix
    S: DimensionlessSparseMatrix

    @property
    def energy_conversion_disk_to_working(self) -> float:
        """Scale converting disk Hartree energies to loaded working energies."""

        return self.disk_unit_context.energy.scale_to_si / self.working_unit_context.energy.scale_to_si

    @property
    def length_conversion_disk_to_working(self) -> float:
        """Scale converting disk bohr lengths to loaded working lengths."""

        return self.disk_unit_context.length.scale_to_si / self.working_unit_context.length.scale_to_si

    @classmethod
    def load(cls, root: Path, units: Units = eVag) -> Self:
        root = Path(root)
        root = require_dir(root)

        metadata = SparseMetadata.load(root / "sparsematrix_metadata.dat", units=units)
        basis = BasisMap.from_metadata(metadata)

        H = atom_ordered_bsr(mmread(require_file(root / "hamiltonian_sparse.mtx")).tocsr() * units.E, basis)
        S = atom_ordered_bsr(mmread(require_file(root / "overlap_sparse.mtx")).tocsr(), basis)


        freeze_bsr(H)
        freeze_bsr(S)
        
        freeze_array(metadata.positions)
        freeze_array(metadata.atom_of_basis)
        freeze_array(metadata.channel_of_basis)
        freeze_array(metadata.symbols)

        return cls(
            root=root,
            units=units,
            disk_unit_context=ATOMIC_UNITS,
            working_unit_context=unit_context_from_legacy_units(units),
            metadata=metadata,
            basis=basis,
            H=H,
            S=S,
        ).validate()


    def validate(self) -> Self:
        expected_shape = (
            self.metadata.nbasis,
            self.metadata.nbasis,
        )

        for M, M_name in [ (self.H, 'H'), (self.S, 'S')]:
            if M.shape != expected_shape:
                raise ValueError(f"{M_name} shape {M.shape} != {expected_shape}")

            if M.blocksize != (self.basis.nchannels, self.basis.nchannels):
                raise ValueError(f"Bad {M_name} blocksize: {M.blocksize}")

            if M.indptr.size != self.metadata.natoms + 1:
                raise ValueError(f"{M_name} does not have one block row per atom")

        return self

    def coupled_atoms(self, M:bsr_matrix, atom: int):
        atoms_b, blocks = block_row_raw(M, atom)

        Ra = self.metadata.positions[atom]
        dR = self.metadata.positions[atoms_b] - Ra
        distances = np.linalg.norm(dR, axis=1)
        norms = np.linalg.norm(blocks, axis=(1, 2))

        order = np.argsort(-norms)

        return [
            AtomBlock(
                atom_b=int(atoms_b[i]),
                distance=float(distances[i]),
                norm=float(norms[i]),
                dR=dR[i],
                block=blocks[i],
            )
            for i in order
        ]


def coupled_atoms_table(M, data: SparseDataset, a: int):
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


def coupled_atoms_table_by_distance(M, data: SparseDataset, a: int):
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
            for block in sorted(data.coupled_atoms(M, a), key = lambda block:block.distance)
        ]
    )

