from typing import Self
from dataclasses import dataclass
from collections import Counter, deque
from pathlib import Path
from scipy.io import mmread
from scipy.sparse import spmatrix, csr_matrix, bsr_matrix
from scipy.spatial import cKDTree
import numpy as np
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
        root = Path(root)
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


    # note: that H blocsk are not hermitian!! symmetrize if needed
    def atom_block(self, M: spmatrix, a: int, b: int) -> np.ndarray:
        ia = self.basis.basis_indices(a)
        ib = self.basis.basis_indices(b)
        return M[ia[:, None], ib].toarray()

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


@dataclass(frozen=True)
class AtomBlockMatrix:
    M: bsr_matrix
    positions: np.ndarray
    symbols: np.ndarray

    @classmethod
    def from_sparse(cls, M: spmatrix, data: SparseDataset) -> Self:
        nchan = data.basis.nchannels
        perm = data.basis.atom_basis.ravel()

        # Put basis into atom-major order:
        # atom 0 channels, atom 1 channels, ...
        M_atom_ordered = M[perm, :][:, perm].tobsr(blocksize=(nchan, nchan))

        return cls(
            M=M_atom_ordered,
            positions=data.metadata.positions,
            symbols=data.metadata.symbols,
        )

    def atom_block_row_raw(self, a: int):
        start = self.M.indptr[a]
        stop = self.M.indptr[a + 1]
        return self.M.indices[start:stop], self.M.data[start:stop]

    def coupled_atom_blocks(self, a: int):
        atoms_b, blocks = self.atom_block_row_raw(a)

        Ra = self.positions[a]
        dR = self.positions[atoms_b] - Ra
        distances = np.linalg.norm(dR, axis=1)
        norms = np.linalg.norm(blocks, axis=(1, 2))

        order = np.argsort(-norms)

        return [
            AtomBlock(
                atom_b=int(atoms_b[i]),
                dist=float(distances[i]),
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

@dataclass(frozen=True, slots=True)
class GdElement:
    """
    Element of the edge-generator group in normal form

        g = x^m y^n t^eps

    where

        t = d1
        x = d1 d2
        y = d1 d3
        eps in {0, 1}

    Group law:

        (r, eps)(r', eps') = (r + (-1)^eps r', eps + eps' mod 2)

    with r = (m, n).
    """

    m: int
    n: int
    eps: int

    def __post_init__(self) -> None:
        if self.eps not in (0, 1):
            raise ValueError(f"eps must be 0 or 1, got {self.eps}")

    @staticmethod
    def identity() -> Self:
        return GdElement(0, 0, 0)

    @staticmethod
    def x() -> Self:
        return GdElement(1, 0, 0)

    @staticmethod
    def y() -> Self:
        return GdElement(0, 1, 0)

    @staticmethod
    def t() -> Self:
        return GdElement(0, 0, 1)

    @staticmethod
    def d1() -> Self:
        return GdElement.t()

    @staticmethod
    def d2() -> Self:
        # d2 = d1 x = t x = x^-1 t
        return GdElement(-1, 0, 1)

    @staticmethod
    def d3() -> Self:
        # d3 = d1 y = t y = y^-1 t
        return GdElement(0, -1, 1)

    def __mul__(self, other: Self) -> Self:
        if not isinstance(other, GdElement):
            return NotImplemented

        sign = -1 if self.eps else 1

        return GdElement(
            self.m + sign * other.m,
            self.n + sign * other.n,
            (self.eps + other.eps) & 1,
        )

    def inverse(self) -> Self:
        sign = -1 if self.eps else 1

        return GdElement(
            -sign * self.m,
            -sign * self.n,
            self.eps,
        )

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.m, self.n, self.eps)

    def translation(self) -> tuple[int, int]:
        return (self.m, self.n)

    def __repr__(self) -> str:
        return f"GdElement(m={self.m}, n={self.n}, eps={self.eps})"


@dataclass(frozen=True, slots=True)
class NearestNeighbourGraph:
    positions: FloatArray
    a0: float
    cutoff: float
    indptr: IntArray
    indices: IntArray
    distances: FloatArray
    vectors: FloatArray

    @classmethod
    def from_positions(
        cls,
        positions: FloatArray,
        *,
        cutoff_factor: float = 1.25,
        query_k: int = 5,
    ) -> Self:
        positions = np.asarray(positions, dtype=np.float64)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (natoms, 3), got {positions.shape}")

        tree = cKDTree(positions)

        # Query self + nearest neighbours. distances[:, 0] should be self-distance.
        distances_k, _indices_k = tree.query(positions, k=query_k)

        nearest_nonself = distances_k[:, 1]
        a0 = float(np.median(nearest_nonself))

        if not np.isfinite(a0) or a0 <= 0:
            raise ValueError(f"Bad estimated bond length: {a0}")

        cutoff = cutoff_factor * a0

        neighbour_lists = tree.query_ball_point(positions, r=cutoff)

        indptr = [0]
        indices = []
        distances = []
        vectors = []

        for a, raw_neighbours in enumerate(neighbour_lists):
            row = []

            Ra = positions[a]

            for b in raw_neighbours:
                if b == a:
                    continue

                dR = positions[b] - Ra
                dist = float(np.linalg.norm(dR))

                # query_ball_point should already ensure this, but keep explicit.
                if dist <= cutoff:
                    row.append((int(b), dist, dR))

            # Deterministic order: nearest first, then atom index.
            row.sort(key=lambda x: (x[1], x[0]))

            for b, dist, dR in row:
                indices.append(b)
                distances.append(dist)
                vectors.append(dR)

            indptr.append(len(indices))

        return cls(
            positions=positions,
            a0=a0,
            cutoff=float(cutoff),
            indptr=np.asarray(indptr, dtype=np.int64),
            indices=np.asarray(indices, dtype=np.int64),
            distances=np.asarray(distances, dtype=np.float64),
            vectors=np.asarray(vectors, dtype=np.float64),
        )

    @property
    def natoms(self) -> int:
        return self.positions.shape[0]

    @property
    def degree(self) -> IntArray:
        return np.diff(self.indptr)

    def row_slice(self, atom: int) -> slice:
        return slice(self.indptr[atom], self.indptr[atom + 1])

    def neighbours(self, atom: int) -> IntArray:
        return self.indices[self.row_slice(atom)]

    def neighbour_distances(self, atom: int) -> FloatArray:
        return self.distances[self.row_slice(atom)]

    def neighbour_vectors(self, atom: int) -> FloatArray:
        return self.vectors[self.row_slice(atom)]

    def bulk_atoms(self) -> IntArray:
        return np.flatnonzero(self.degree == 3)

    def core_bulk_atoms(self) -> IntArray:
        degree = self.degree
        out = []

        for a in np.flatnonzero(degree == 3):
            ns = self.neighbours(int(a))
            if np.all(degree[ns] == 3):
                out.append(int(a))

        return np.asarray(out, dtype=np.int64)

    def choose_anchor(self) -> int:
        candidates = self.core_bulk_atoms()

        if len(candidates) == 0:
            candidates = self.bulk_atoms()

        if len(candidates) == 0:
            raise ValueError("No degree-3 atoms found; cannot choose graphene bulk anchor")

        centre = np.mean(self.positions, axis=0)
        d = np.linalg.norm(self.positions[candidates] - centre, axis=1)

        return int(candidates[int(np.argmin(d))])

    def diagnostics(self) -> dict:
        degree_counts = dict(Counter(map(int, self.degree)))

        nearest = []
        for a in range(self.natoms):
            ds = self.neighbour_distances(a)
            if len(ds):
                nearest.append(float(np.min(ds)))

        nearest = np.asarray(nearest, dtype=np.float64)

        return {
            "natoms": self.natoms,
            "a0": self.a0,
            "cutoff": self.cutoff,
            "cutoff_over_a0": self.cutoff / self.a0,
            "degree_counts": degree_counts,
            "num_bulk_atoms": int(len(self.bulk_atoms())),
            "num_core_bulk_atoms": int(len(self.core_bulk_atoms())),
            "anchor_atom": self.choose_anchor(),
            "nearest_distance_min": float(np.min(nearest)) if len(nearest) else None,
            "nearest_distance_median": float(np.median(nearest)) if len(nearest) else None,
            "nearest_distance_max": float(np.max(nearest)) if len(nearest) else None,
            "nearest_distance_std": float(np.std(nearest)) if len(nearest) else None,
        }



@dataclass(frozen=True, slots=True)
class EdgeDirections:
    anchor_atom: int
    anchor_neighbours: IntArray      # shape (3,)
    d_vectors: FloatArray            # shape (3, 3)
    d_unit: FloatArray               # shape (3, 3)
    plane_e1: FloatArray             # shape (3,)
    plane_e2: FloatArray             # shape (3,)
    plane_normal: FloatArray         # shape (3,)

    @classmethod
    def from_geometry(
        cls,
        geom: "NearestNeighbourGraph",
        *,
        anchor_atom: int | None = None,
    ) -> "EdgeDirections":
        if anchor_atom is None:
            anchor_atom = geom.choose_anchor()

        neighbours = geom.neighbours(anchor_atom)
        vectors = geom.neighbour_vectors(anchor_atom)

        if len(neighbours) != 3:
            raise ValueError(
                f"Anchor atom must have exactly 3 neighbours, got {len(neighbours)}"
            )

        plane_e1, plane_e2, plane_normal = estimate_plane_basis(geom.positions)

        # Sort the three anchor bonds by angle in the graphene plane.
        x = vectors @ plane_e1
        y = vectors @ plane_e2
        angles = np.arctan2(y, x)

        order = np.argsort(angles)

        anchor_neighbours = np.asarray(neighbours[order], dtype=np.int64)
        d_vectors = np.asarray(vectors[order], dtype=np.float64)

        norms = np.linalg.norm(d_vectors, axis=1)
        if np.any(norms <= 0):
            raise ValueError("Zero-length anchor bond vector")

        d_unit = d_vectors / norms[:, None]

        return cls(
            anchor_atom=int(anchor_atom),
            anchor_neighbours=anchor_neighbours,
            d_vectors=d_vectors,
            d_unit=d_unit,
            plane_e1=plane_e1,
            plane_e2=plane_e2,
            plane_normal=plane_normal,
        )

    def classify_vector(
        self,
        dR: FloatArray,
        *,
        min_alignment: float = 0.95,
    ) -> int:
        """
        Return generator index 0, 1, or 2 for d_1, d_2, d_3.

        Uses abs(dot), because traversing an edge in either direction is the
        same involutive edge generator.
        """
        dR = np.asarray(dR, dtype=np.float64)
        norm = np.linalg.norm(dR)

        if norm <= 0:
            raise ValueError("Cannot classify zero-length vector")

        u = dR / norm
        scores = np.abs(self.d_unit @ u)
        i = int(np.argmax(scores))

        if scores[i] < min_alignment:
            raise ValueError(
                f"Could not classify edge vector; best alignment={scores[i]}"
            )

        return i

    def classify_vectors(
        self,
        dR: FloatArray,
        *,
        min_alignment: float = 0.95,
    ) -> IntArray:
        dR = np.asarray(dR, dtype=np.float64)
        norms = np.linalg.norm(dR, axis=1)

        if np.any(norms <= 0):
            raise ValueError("Cannot classify zero-length vector")

        u = dR / norms[:, None]
        scores = np.abs(u @ self.d_unit.T)
        labels = np.argmax(scores, axis=1)
        best = scores[np.arange(len(labels)), labels]

        if np.any(best < min_alignment):
            worst = float(np.min(best))
            raise ValueError(f"Could not classify all vectors; worst alignment={worst}")

        return labels.astype(np.int64)

    def diagnostics(self, geom: "NearestNeighbourGraph") -> dict:
        labels = self.classify_vectors(geom.vectors)
        counts = {int(i): int(np.sum(labels == i)) for i in range(3)}

        scores = np.abs(
            (geom.vectors / np.linalg.norm(geom.vectors, axis=1)[:, None])
            @ self.d_unit.T
        )
        best = np.max(scores, axis=1)

        return {
            "anchor_atom": self.anchor_atom,
            "anchor_neighbours": self.anchor_neighbours.tolist(),
            "d_vectors": self.d_vectors.tolist(),
            "generator_counts": counts,
            "alignment_min": float(np.min(best)),
            "alignment_median": float(np.median(best)),
            "alignment_max": float(np.max(best)),
        }


def estimate_plane_basis(positions: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Estimate graphene plane using PCA.

    Returns:
        e1, e2, normal
    """
    X = np.asarray(positions, dtype=np.float64)
    X = X - np.mean(X, axis=0)

    # Vt rows are principal axes.
    _U, _S, Vt = np.linalg.svd(X, full_matrices=False)

    e1 = Vt[0]
    e2 = Vt[1]
    normal = Vt[2]

    # Enforce right-handed orientation.
    if np.dot(np.cross(e1, e2), normal) < 0:
        normal = -normal

    return e1, e2, normal

@dataclass(frozen=True, slots=True)
class EdgeGroupLabels:
    """
    Atom -> G_d label in normal form

        g_a = x^m[a] y^n[a] d_1^eps[a]

    The arrays are SOA storage for later fast use.
    """

    m: IntArray
    n: IntArray
    eps: IntArray
    anchor_atom: int
    visited: NDArray[np.bool_]

    @classmethod
    def from_geometry(
        cls,
        geom: "NearestNeighbourGraph",
        edge_dirs: "EdgeDirections",
        *,
        anchor_atom: int | None = None,
        strict: bool = True,
    ) -> "EdgeGroupLabels":
        if anchor_atom is None:
            anchor_atom = edge_dirs.anchor_atom

        natoms = geom.natoms

        m = np.zeros(natoms, dtype=np.int64)
        n = np.zeros(natoms, dtype=np.int64)
        eps = np.zeros(natoms, dtype=np.int64)
        visited = np.zeros(natoms, dtype=bool)

        labels: list[GdElement | None] = [None] * natoms
        labels[anchor_atom] = GdElement.identity()
        visited[anchor_atom] = True

        edge_generators = [
            GdElement.d1(),
            GdElement.d2(),
            GdElement.d3(),
        ]

        queue: deque[int] = deque([anchor_atom])
        conflicts: list[tuple[int, int, GdElement, GdElement]] = []

        while queue:
            a = queue.popleft()
            g_a = labels[a]
            assert g_a is not None

            neighbours = geom.neighbours(a)
            vectors = geom.neighbour_vectors(a)
            edge_types = edge_dirs.classify_vectors(vectors)

            for b, edge_type in zip(neighbours, edge_types):
                b = int(b)
                edge_type = int(edge_type)

                candidate = g_a * edge_generators[edge_type]
                current = labels[b]

                if current is None:
                    labels[b] = candidate
                    visited[b] = True
                    queue.append(b)
                elif current != candidate:
                    conflicts.append((a, b, current, candidate))

        if strict and conflicts:
            a, b, current, candidate = conflicts[0]
            raise ValueError(
                "Inconsistent G_d labelling. "
                f"First conflict on edge a={a}, b={b}: "
                f"existing={current}, candidate={candidate}. "
                f"Total conflicts={len(conflicts)}"
            )

        for a, g in enumerate(labels):
            if g is None:
                continue

            m[a] = g.m
            n[a] = g.n
            eps[a] = g.eps

        return cls(
            m=m,
            n=n,
            eps=eps,
            anchor_atom=int(anchor_atom),
            visited=visited,
        )

    @property
    def natoms(self) -> int:
        return len(self.m)

    def element(self, atom: int) -> GdElement:
        if not self.visited[atom]:
            raise ValueError(f"Atom {atom} was not labelled")
        return GdElement(int(self.m[atom]), int(self.n[atom]), int(self.eps[atom]))

    def elements(self) -> list[GdElement | None]:
        out: list[GdElement | None] = []

        for a in range(self.natoms):
            if self.visited[a]:
                out.append(self.element(a))
            else:
                out.append(None)

        return out

    def relative(self, a: int, b: int) -> GdElement:
        """
        Return h = g_a^-1 g_b.
        """
        return self.element(a).inverse() * self.element(b)

    def diagnostics(self) -> dict:
        visited_count = int(np.sum(self.visited))
        eps_counts = dict(Counter(map(int, self.eps[self.visited])))

        return {
            "natoms": self.natoms,
            "anchor_atom": self.anchor_atom,
            "visited_count": visited_count,
            "unvisited_count": self.natoms - visited_count,
            "eps_counts": eps_counts,
            "m_min": int(np.min(self.m[self.visited])) if visited_count else None,
            "m_max": int(np.max(self.m[self.visited])) if visited_count else None,
            "n_min": int(np.min(self.n[self.visited])) if visited_count else None,
            "n_max": int(np.max(self.n[self.visited])) if visited_count else None,
        }


data = SparseDataset.load(Path("./test_run/run_dir/data"))
meta = data.metadata
H = data.H
S = data.S
H_blocks = AtomBlockMatrix.from_sparse(data.H, data)
S_blocks = AtomBlockMatrix.from_sparse(data.S, data)


geom = NearestNeighbourGraph.from_positions(data.metadata.positions)
anchor = geom.choose_anchor()
edges = EdgeDirections.from_geometry(geom)
labels = EdgeGroupLabels.from_geometry(geom, edges)


# print(coupled_atoms_table(H, data, 0))


from pprint import pprint
pprint(geom.diagnostics())
pprint(geom.neighbour_vectors(anchor))
pprint(edges.diagnostics(geom))
pprint(labels.diagnostics())
