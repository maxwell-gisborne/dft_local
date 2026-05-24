from typing import Self, Tuple
from dataclasses import dataclass, replace
from collections import Counter, deque
from pathlib import Path
from scipy.io import mmread
from scipy.sparse import bsr_matrix
from scipy.spatial import cKDTree
from scipy.linalg import eigvalsh, eigh
import numpy as np
from numpy.typing import NDArray

def freeze_array(a):
    a.flags.writeable = False
    return a

def freeze_bsr(M):
    M.data.flags.writeable = False
    M.indices.flags.writeable = False
    M.indptr.flags.writeable = False
    return M

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BlockArray = NDArray[np.float64] | NDArray[np.complex128]


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

def block_row_raw(M:bsr_matrix, atom: int):
    start = M.indptr[atom]
    stop = M.indptr[atom + 1]
    return M.indices[start:stop], M.data[start:stop]


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

@dataclass(frozen=True)
class SparseDataset:
    root: Path
    units: Units
    metadata: SparseMetadata
    basis: BasisMap
    H: bsr_matrix
    S: bsr_matrix

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

        return cls(root=root, units=units, metadata=metadata, basis=basis, H=H, S=S).validate()


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
            positions=freeze_array(positions),
            a0=a0,
            cutoff=float(cutoff),
            indptr=freeze_array(np.asarray(indptr, dtype=np.int64)),
            indices=freeze_array(np.asarray(indices, dtype=np.int64)),
            distances=freeze_array(np.asarray(distances, dtype=np.float64)),
            vectors=freeze_array(np.asarray(vectors, dtype=np.float64)),
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
        geom: NearestNeighbourGraph,
        *,
        anchor_atom: int | None = None,
    ) -> Self:
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
            d_vectors=freeze_array(d_vectors),
            d_unit=freeze_array(d_unit),
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
    element_to_atom: dict[GdElement, int]
    geometry: NearestNeighbourGraph
    edges: EdgeDirections

    @classmethod
    def from_geometry(
        cls,
        geom: NearestNeighbourGraph,
        edge_dirs: EdgeDirections,
        *,
        anchor_atom: int | None = None,
        strict: bool = True,
    ) -> Self:
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

        element_to_atom = {}
        for a, g in enumerate(labels):
            if g is None:
                continue

            m[a] = g.m
            n[a] = g.n
            eps[a] = g.eps

            if g in element_to_atom:
                raise ValueError(
                    f"Duplicate G_d label {g}: atoms {element_to_atom[g]} and {a}"
                )
            element_to_atom[g] = a

        return cls(
            m=freeze_array(m),
            n=freeze_array(n),
            eps=freeze_array(eps),
            anchor_atom=int(anchor_atom),
            visited=freeze_array(visited),
            element_to_atom=element_to_atom,
            geometry = geom,
            edges = edge_dirs,
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

    def gd_reconstructed_positions(
        labels:Self,
    ) -> FloatArray:
        R0 = geom.positions[labels.anchor_atom]

        d1, d2, d3 = labels.edges.d_vectors

        ax = d1 - d2
        ay = d1 - d3

        return (
            R0
            + labels.m[:, None] * ax[None, :]
            + labels.n[:, None] * ay[None, :]
            + labels.eps[:, None] * d1[None, :]
        )


    def gd_position_errors(
        labels: Self,
    ) -> FloatArray:
        R_pred = labels.gd_reconstructed_positions()
        return np.linalg.norm(R_pred - labels.geometry.positions, axis=1)


    def diagnostics(self) -> dict:
        visited_count = int(np.sum(self.visited))
        eps_counts = dict(Counter(map(int, self.eps[self.visited])))
        err = self.gd_position_errors()

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
            "position_reconstruction_errors": {
                "max": float(err.max()),
                "mean": float(err.mean()),
                "median": float(np.median(err)),
            },
        }


def relative_labels_for_row(labels: EdgeGroupLabels, atom: int, atoms_b: IntArray):
    """
    Vectorised relative labels for a BSR block row.

    Returns arrays h_m, h_n, h_eps such that

        h = g_atom^-1 g_b

    for each b in atoms_b.
    """

    '''
    For atoms a and b:

    g_a = (r_a, eps_a)
    g_b = (r_b, eps_b)

        with group law

    (r,e)(r',e') = (r + (-1)^e r', e xor e')

        Then

    h = g_a^-1 g_b

        is:

    h.r = (-1)^eps_a * (r_b - r_a)
    h.eps = eps_a xor eps_b
    ''' 

    atom = int(atom)
    atoms_b = np.asarray(atoms_b, dtype=np.int64)

    sign = 1 if labels.eps[atom] == 0 else -1

    h_m = sign * (labels.m[atoms_b] - labels.m[atom])
    h_n = sign * (labels.n[atoms_b] - labels.n[atom])
    h_eps = labels.eps[atom] ^ labels.eps[atoms_b]

    return h_m, h_n, h_eps

@dataclass(frozen=True, slots=True)
class GdKernelArrays:
    h_m: IntArray              # shape (N,)
    h_n: IntArray              # shape (N,)
    h_eps: IntArray            # shape (N,)
    blocks: BlockArray         # shape (N, q, q)
    matrix_name: str = ""

    @classmethod
    def from_anchored(
        cls,
        M: bsr_matrix,
        labels: EdgeGroupLabels,
        anchor_atom: int | None = None,
        matrix_name: str = "",
        copy_blocks: bool = False,
    ):
        if anchor_atom is None:
            anchor_atom = labels.anchor_atom

        atoms_b, blocks = block_row_raw(M, anchor_atom)

        h_m, h_n, h_eps = relative_labels_for_row(labels, anchor_atom, atoms_b)

        if copy_blocks:
            blocks = freeze_array(np.array(blocks, copy=True))

        return cls(
            h_m=freeze_array(np.asarray(h_m, dtype=np.int64)),
            h_n=freeze_array(np.asarray(h_n, dtype=np.int64)),
            h_eps=freeze_array(np.asarray(h_eps, dtype=np.int64)),
            blocks=np.asarray(blocks),
            matrix_name=matrix_name,
        )

    @classmethod
    def from_average(
        cls,
        M: bsr_matrix,
        labels: EdgeGroupLabels,
        anchors: IntArray | None = None,
        matrix_name: str = "",
    ) -> Self:
        """
        Average block rows over anchors to form an effective homogeneous kernel.

            K(h) = average_a M[a, a h]

        where h = g_a^-1 g_b.
        """
        if anchors is None:
            anchors = labels.geometry.core_bulk_atoms()

        anchors = np.asarray(anchors, dtype=np.int64)

        if len(anchors) == 0:
            raise ValueError("No anchors supplied for averaged kernel")

        sums: dict[tuple[int, int, int], np.ndarray] = {}
        counts: dict[tuple[int, int, int], int] = {}

        for a in anchors:
            a = int(a)
            atoms_b, blocks = block_row_raw(M, a)

            h_m, h_n, h_eps = relative_labels_for_row(labels, a, atoms_b)

            for hm, hn, he, block in zip(h_m, h_n, h_eps, blocks):
                key = (int(hm), int(hn), int(he))

                if key not in sums:
                    sums[key] = np.zeros_like(block, dtype=np.result_type(block, np.float64))
                    counts[key] = 0

                sums[key] += block
                counts[key] += 1

        # Deterministic ordering: eps, then m, then n, or choose m,n,eps.
        keys = sorted(sums.keys(), key=lambda x: (x[2], x[0], x[1]))

        h_m = np.asarray([k[0] for k in keys], dtype=np.int64)
        h_n = np.asarray([k[1] for k in keys], dtype=np.int64)
        h_eps = np.asarray([k[2] for k in keys], dtype=np.int64)

        blocks = np.asarray(
            [sums[k] / counts[k] for k in keys],
            dtype=np.result_type(M.data, np.float64),
        )

        freeze_array(h_m)
        freeze_array(h_n)
        freeze_array(h_eps)
        freeze_array(blocks)

        return cls(
            h_m=h_m,
            h_n=h_n,
            h_eps=h_eps,
            blocks=blocks,
            matrix_name=matrix_name,
        )

    def __post_init__(self) -> None:
        N = len(self.h_m)

        if self.h_n.shape != (N,):
            raise ValueError("h_n shape mismatch")
        if self.h_eps.shape != (N,):
            raise ValueError("h_eps shape mismatch")
        if self.blocks.ndim != 3:
            raise ValueError(f"blocks must have shape (N,q,q), got {self.blocks.shape}")
        if self.blocks.shape[0] != N:
            raise ValueError("blocks/support length mismatch")
        if self.blocks.shape[1] != self.blocks.shape[2]:
            raise ValueError("blocks must be square")
        if not np.all((self.h_eps == 0) | (self.h_eps == 1)):
            raise ValueError("h_eps must contain only 0 and 1")
        if not np.all(np.isfinite(self.blocks)):
            raise ValueError("blocks contain NaN or Inf")

    def star_symmetrised(
        self,
        missing: str = "zero",
        matrix_name: str | None = None,
    ) -> Self:
        """
        Return kernel satisfying

            K(h^-1) = K(h)^†

        by replacing

            K(h) -> 1/2 [ K(h) + K(h^-1)^† ]

        If missing='zero', absent inverse blocks are treated as zero.
        If missing='keep', absent inverse blocks are kept as 1/2 K(h), and the
        inverse support is added as 1/2 K(h)^†.
        """
        if missing not in ("zero", "keep"):
            raise ValueError("missing must be 'zero' or 'keep'")

        q = self.blocksize

        by_key: dict[tuple[int, int, int], np.ndarray] = {
            (int(m), int(n), int(e)): block
            for m, n, e, block in zip(self.h_m, self.h_n, self.h_eps, self.blocks)
        }

        all_keys = set(by_key)

        if missing == "keep":
            for key in list(by_key):
                all_keys.add(gd_inverse_label(*key))

        out: dict[tuple[int, int, int], np.ndarray] = {}

        zero = np.zeros((q, q), dtype=np.result_type(self.blocks, np.complex128))

        for key in all_keys:
            inv = gd_inverse_label(*key)

            K_h = by_key.get(key, zero)
            K_inv = by_key.get(inv, zero)

            out[key] = 0.5 * (K_h + K_inv.conj().T)

        keys = sorted(out.keys(), key=lambda x: (x[2], x[0], x[1]))

        h_m = np.asarray([k[0] for k in keys], dtype=np.int64)
        h_n = np.asarray([k[1] for k in keys], dtype=np.int64)
        h_eps = np.asarray([k[2] for k in keys], dtype=np.int64)
        blocks = np.asarray([out[k] for k in keys], dtype=np.complex128)

        freeze_array(h_m)
        freeze_array(h_n)
        freeze_array(h_eps)
        freeze_array(blocks)

        return type(self)(
            h_m=h_m,
            h_n=h_n,
            h_eps=h_eps,
            blocks=blocks,
            matrix_name=self.matrix_name + " star-sym" if matrix_name is None else matrix_name,
        )

    @property
    def support_size(self) -> int:
        return len(self.h_m)


    @property
    def blocksize(self) -> int:
        return self.blocks.shape[1]


    def symbol_generic(kernel:Self,  k1: float, k2:float) -> np.ndarray:
        """
        Generic 2D irrep symbol for G_d.

        Returns dense matrix with shape (2*q, 2*q).
        """
        K = kernel.blocks
        q = K.shape[1]

        theta = k1 * kernel.h_m + k2 * kernel.h_n
        phase = np.exp(1j * theta)

        out = np.zeros((2 * q, 2 * q), dtype=np.complex128)

        even = kernel.h_eps == 0
        odd = kernel.h_eps == 1

        # eps = 0:
        # Omega = [[phase, 0],
        #          [0, conj(phase)]]
        if np.any(even):
            K0 = K[even]
            p0 = phase[even]

            out[0:q, 0:q] += np.einsum("h,hij->ij", p0, K0)
            out[q:2*q, q:2*q] += np.einsum("h,hij->ij", np.conj(p0), K0)

        # eps = 1:
        # Omega = [[0, conj(phase)],
        #          [phase, 0]]
        if np.any(odd):
            K1 = K[odd]
            p1 = phase[odd]

            out[0:q, q:2*q] += np.einsum("h,hij->ij", np.conj(p1), K1)
            out[q:2*q, 0:q] += np.einsum("h,hij->ij", p1, K1)

        return out

    def symbol_fixed(self, k1: float, k2: float, sigma: int) -> np.ndarray:
        """
        One-dimensional fixed-point irrep symbol.

        Valid at k in {0, pi}^2.
        """
        if sigma not in (-1, 1):
            raise ValueError(f"sigma must be ±1, got {sigma}")

        theta = k1 * self.h_m + k2 * self.h_n
        coeff = np.exp(1j * theta) * (sigma ** self.h_eps)

        return np.einsum("h,hij->ij", coeff, self.blocks).astype(np.complex128)

    def diagnostics(self) -> dict:
        norms = np.linalg.norm(self.blocks, axis=(1, 2))

        return {
            "matrix_name": self.matrix_name,
            "support_size": self.support_size,
            "blocksize": self.blocksize,
            "num_even": int(np.sum(self.h_eps == 0)),
            "num_odd": int(np.sum(self.h_eps == 1)),
            "norm_min": float(np.min(norms)) if len(norms) else None,
            "norm_max": float(np.max(norms)) if len(norms) else None,
            "norm_median": float(np.median(norms)) if len(norms) else None,
        }

    def star_defect(self) -> dict:
        """
        Measure failure of K(h^-1) = K(h)^†.
        """
        by_key: dict[tuple[int, int, int], np.ndarray] = {
            (int(m), int(n), int(e)): block
            for m, n, e, block in zip(self.h_m, self.h_n, self.h_eps, self.blocks)
        }

        defects = []

        seen = set()

        for key, K_h in by_key.items():
            if key in seen:
                continue

            inv = gd_inverse_label(*key)
            seen.add(key)
            seen.add(inv)

            K_inv = by_key.get(inv)

            if K_inv is None:
                defects.append((key, None, np.inf, np.linalg.norm(K_h)))
                continue

            err = np.linalg.norm(K_h - K_inv.conj().T)
            scale = max(np.linalg.norm(K_h), np.linalg.norm(K_inv), 1.0)
            defects.append((key, inv, err, err / scale))

        finite = [d[3] for d in defects if np.isfinite(d[3])]
        missing = sum(1 for d in defects if d[1] is None)

        return {
            "matrix_name": self.matrix_name,
            "support_size": self.support_size,
            "num_missing_inverse": missing,
            "star_defect_max": float(np.max(finite)) if finite else None,
            "star_defect_mean": float(np.mean(finite)) if finite else None,
            "star_defect_median": float(np.median(finite)) if finite else None,
        }

    def star_defect_table(self):
        import pandas as pd

        by_key = {
            (int(m), int(n), int(e)): block
            for m, n, e, block in zip(self.h_m, self.h_n, self.h_eps, self.blocks)
        }

        rows = []
        seen = set()

        for key, K_h in by_key.items():
            if key in seen:
                continue

            inv = gd_inverse_label(*key)
            seen.add(key)
            seen.add(inv)

            K_inv = by_key.get(inv)

            if K_inv is None:
                err = np.inf
                rel = np.inf
                norm_h = float(np.linalg.norm(K_h))
                norm_inv = None
            else:
                err = float(np.linalg.norm(K_h - K_inv.conj().T))
                norm_h = float(np.linalg.norm(K_h))
                norm_inv = float(np.linalg.norm(K_inv))
                rel = err / max(norm_h, norm_inv, 1.0)

            rows.append({
                "m": key[0],
                "n": key[1],
                "eps": key[2],
                "inv_m": inv[0],
                "inv_n": inv[1],
                "inv_eps": inv[2],
                "norm": norm_h,
                "inv_norm": norm_inv,
                "star_error": err,
                "star_relative_error": rel,
            })

        return pd.DataFrame(rows).sort_values("star_relative_error", ascending=False)

    def star_defect_table_filtered(K, *, min_norm=1e-2, max_radius=None):
        table = K.star_defect_table()

        table = table[table["norm"] >= min_norm]

        if max_radius is not None:
            radius = np.maximum(np.abs(table["m"]), np.abs(table["n"]))
            table = table[radius <= max_radius]

        return table.sort_values("star_relative_error", ascending=False)


def hermitian_part(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.conj().T)



def gd_inverse_label(m: int, n: int, eps: int) -> tuple[int, int, int]:
    sign = -1 if eps else 1
    return (-sign * m, -sign * n, eps)


@dataclass(frozen=True, slots=True)
class DenseMatrixDiagnostics:
    name: str
    shape: tuple[int, int]
    dtype: str
    finite: bool
    norm: float
    hermitian_defect_abs: float
    hermitian_defect_rel: float
    eig_min: float | None = None
    eig_max: float | None = None
    condition_number_abs: float | None = None
    positive_definite: bool | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "finite": self.finite,
            "norm": self.norm,
            "hermitian_defect_abs": self.hermitian_defect_abs,
            "hermitian_defect_rel": self.hermitian_defect_rel,
            "eig_min": self.eig_min,
            "eig_max": self.eig_max,
            "condition_number_abs": self.condition_number_abs,
            "positive_definite": self.positive_definite,
        }


    @classmethod
    def from_dense_matrix(
        cls,
        A: np.ndarray,
        *,
        name: str = "",
        check_eigenvalues: bool = False,
        positive_tol: float = 1e-10,
        ) -> Self:

        A = np.asarray(A)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"{name}: expected square matrix, got shape {A.shape}")

        finite = bool(np.all(np.isfinite(A)))
        norm = float(np.linalg.norm(A))

        defect = A - A.conj().T
        defect_abs = float(np.linalg.norm(defect))
        defect_rel = defect_abs / max(norm, 1.0)

        eig_min = None
        eig_max = None
        condition_number_abs = None
        positive_definite = None

        if check_eigenvalues:
            # Use Hermitian part for diagnostic eigenvalues.
            Ah = 0.5 * (A + A.conj().T)
            eigs = eigvalsh(Ah)

            eig_min = float(np.min(eigs))
            eig_max = float(np.max(eigs))

            abs_eigs = np.abs(eigs)
            nonzero = abs_eigs[abs_eigs > 0]

            if len(nonzero):
                condition_number_abs = float(np.max(abs_eigs) / np.min(nonzero))
            else:
                condition_number_abs = np.inf

            positive_definite = bool(eig_min > positive_tol)

        return cls(
            name=name,
            shape=A.shape,
            dtype=str(A.dtype),
            finite=finite,
            norm=norm,
            hermitian_defect_abs=defect_abs,
            hermitian_defect_rel=defect_rel,
            eig_min=eig_min,
            eig_max=eig_max,
            condition_number_abs=condition_number_abs,
            positive_definite=positive_definite,
        )


@dataclass(frozen=True, slots=True)
class SymbolPair:
    KH: GdKernelArrays
    KS: GdKernelArrays
    k1: float
    k2: float
    degree: int = 2
    sigma: int | None = None
    name:str = ''

    def star_symmetrised(self) -> Self:
        return replace(
            self,
            KH=self.KH.star_symmetrised(matrix_name = self.name + " star"),
            KS=self.KS.star_symmetrised(matrix_name = self.name + " star"),
        )

    def form(self) -> "LocalProblem":
        match self.degree:
            case 2:
                if self.sigma is not None:
                    raise ValueError("sigma should be None for generic 2D irrep")

                Hk = self.KH.symbol_generic(self.k1, self.k2)
                Sk = self.KS.symbol_generic(self.k1, self.k2)

            case 1:
                if self.sigma is None:
                    raise ValueError("sigma is required for fixed-point 1D irrep")
                if self.sigma not in (-1, 1):
                    raise ValueError(f"sigma must be ±1, got {self.sigma}")

                Hk = self.KH.symbol_fixed(self.k1, self.k2, sigma=self.sigma)
                Sk = self.KS.symbol_fixed(self.k1, self.k2, sigma=self.sigma)

            case _:
                raise ValueError(f"Unsupported irrep degree: {self.degree}")

        return LocalProblem(Hk=Hk, Sk=Sk, pair=self)

       
    def label(self) -> str:
        if self.degree == 2:
            return f"k=({self.k1:.6g},{self.k2:.6g}), degree=2"
        return f"k=({self.k1:.6g},{self.k2:.6g}), degree=1, sigma={self.sigma}"


@dataclass(frozen=True, slots=True)
class LocalProblem:
    Hk: np.ndarray
    Sk: np.ndarray
    pair: SymbolPair

    def symmetrised(self) -> Self:
        return replace(
            self,
            Hk=hermitian_part(self.Hk),
            Sk=hermitian_part(self.Sk),
        )

    def overlap_eigenvalues(self) -> np.ndarray:
        return eigvalsh(hermitian_part(self.Sk))

    def check_overlap_positive(self, tol: float = 1e-10) -> None:
        vals = self.overlap_eigenvalues()
        vmin = float(np.min(vals))

        if vmin <= tol:
            raise ValueError(
                f"Overlap symbol not positive definite: min eigenvalue={vmin}"
            )

    def solve(
        self,
        *,
        symmetrise: bool = True,
        check_overlap: bool = True,
        overlap_tol: float = 1e-10,
        eigvals_only: bool = False,
    ):
        problem = self.symmetrised() if symmetrise else self

        if check_overlap:
            problem.check_overlap_positive(tol=overlap_tol)

        return eigh(problem.Hk, problem.Sk, eigvals_only=eigvals_only)

    def energies(
        self,
        *,
        symmetrise: bool = True,
        check_overlap: bool = True,
        overlap_tol: float = 1e-10,
    ) -> np.ndarray:
        return self.solve(
            symmetrise=symmetrise,
            check_overlap=check_overlap,
            overlap_tol=overlap_tol,
            eigvals_only=True,
        )

    def eigensystem(
        self,
        *,
        symmetrise: bool = True,
        check_overlap: bool = True,
        overlap_tol: float = 1e-10,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.solve(
            symmetrise=symmetrise,
            check_overlap=check_overlap,
            overlap_tol=overlap_tol,
            eigvals_only=False,
        )

    def diagnostics(problem: Self) -> dict:
        Hdiag = DenseMatrixDiagnostics.from_dense_matrix(
            problem.Hk,
            name="H(k)",
            check_eigenvalues=False,
        )

        Sdiag = DenseMatrixDiagnostics.from_dense_matrix(
            problem.Sk,
            name="S(k)",
            check_eigenvalues=True,
        )

        return {
            "pair": problem.pair.label(),
            "H": Hdiag.as_dict(),
            "S": Sdiag.as_dict(),
            "energies": problem.energies().tolist(),
        }


@dataclass(frozen=True, slots=True)
class LocalPath:
    KH: GdKernelArrays
    KS: GdKernelArrays
    k1: np.ndarray
    k2: np.ndarray
    x: np.ndarray | None = None
    labels: tuple[tuple[int, str], ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        if self.k1.shape != self.k2.shape:
            raise ValueError("k1 and k2 shape mismatch")
        if self.x is not None and self.x.shape != self.k1.shape:
            raise ValueError("x shape mismatch")

    @classmethod
    def from_points(
        cls,
        KH: GdKernelArrays,
        KS: GdKernelArrays,
        points: list[tuple[str, float, float]],
        *,
        points_per_segment: int = 80,
        name: str = "",
    ) -> Self:
        k1_parts = []
        k2_parts = []
        x_parts = []
        labels = []

        x_current = 0.0

        for seg_index, ((label_a, a1, a2), (label_b, b1, b2)) in enumerate(zip(points[:-1], points[1:])):
            # Avoid duplicating joint points except for first segment.
            t = np.linspace(0.0, 1.0, points_per_segment, endpoint=False)
            if seg_index == len(points) - 2:
                t = np.linspace(0.0, 1.0, points_per_segment + 1, endpoint=True)

            seg_k1 = (1 - t) * a1 + t * b1
            seg_k2 = (1 - t) * a2 + t * b2

            dk = float(np.sqrt((b1 - a1)**2 + (b2 - a2)**2))
            seg_x = x_current + t * dk

            if seg_index == 0:
                labels.append((0, label_a))

            k1_parts.append(seg_k1)
            k2_parts.append(seg_k2)
            x_parts.append(seg_x)

            x_current += dk
            labels.append((sum(len(p) for p in k1_parts) - 1, label_b))

        return cls(
            KH=KH,
            KS=KS,
            k1=np.concatenate(k1_parts),
            k2=np.concatenate(k2_parts),
            x=np.concatenate(x_parts),
            labels=tuple(labels),
            name=name,
        )

    def pair(self, i: int) -> SymbolPair:
        return SymbolPair(
            KH=self.KH,
            KS=self.KS,
            k1=float(self.k1[i]),
            k2=float(self.k2[i]),
            name=self.name,
        )

    def form(self, i: int) -> LocalProblem:
        return self.pair(i).form()

    def energies(
        self,
        *,
        symmetrise: bool = True,
        check_overlap: bool = True,
        overlap_tol: float = 1e-10,
    ) -> np.ndarray:
        rows = []

        for i in range(len(self.k1)):
            rows.append(
                self.form(i).energies(
                    symmetrise=symmetrise,
                    check_overlap=check_overlap,
                    overlap_tol=overlap_tol,
                )
            )

        return np.asarray(rows)

    def path_diagnostics(self: Self) -> dict:
        E = path.energies()

        return {
            "name": path.name,
            "num_kpoints": int(len(path.k1)),
            "num_bands": int(E.shape[1]),
            "energy_min": float(np.min(E)),
            "energy_max": float(np.max(E)),
            "x_min": float(np.min(path.x)) if path.x is not None else None,
            "x_max": float(np.max(path.x)) if path.x is not None else None,
            "labels": [
                {"index": int(i), "label": label}
                for i, label in path.labels
            ],
        }

@dataclass(frozen = True)
class LocalRegion:
    def form(self):
        pass
    def solve(self):
        pass


data = SparseDataset.load(Path("./test_run/run_dir/data"))
meta = data.metadata
H = data.H
S = data.S

geom = NearestNeighbourGraph.from_positions(data.metadata.positions)
anchor = geom.choose_anchor()
edges = EdgeDirections.from_geometry(geom)
labels = EdgeGroupLabels.from_geometry(geom, edges)


# print(coupled_atoms_table(H, data, 0))

from pprint import pprint

KH = GdKernelArrays.from_anchored(data.H, labels, matrix_name="H")
KS = GdKernelArrays.from_anchored(data.S, labels, matrix_name="S")

k1, k2 = 0.1, 0.2

pair = SymbolPair(KH, KS, k1, k2)
local = pair.form()
local_sym = local.symmetrised()

KH_avg = GdKernelArrays.from_average(data.H, labels, matrix_name="H average")
KS_avg = GdKernelArrays.from_average(data.S, labels, matrix_name="S average")

pair_avg = SymbolPair(KH_avg, KS_avg, k1, k2)
local_avg = pair_avg.form()
local_avg_sym = pair_avg.form().symmetrised()
local_avg_star = pair_avg.star_symmetrised().form()

KH_star = KH.star_symmetrised(matrix_name="H anchored star")
KS_star = KS.star_symmetrised(matrix_name="S anchored star")

KH_avg_star = KH_avg.star_symmetrised(matrix_name="H average star")
KS_avg_star = KS_avg.star_symmetrised(matrix_name="S average star")


# for K in [KH, KH_star, KH_avg, KH_avg_star]:
#     Hk = K.symbol_generic(0.1, 0.2)
#     print()
#     print(K.matrix_name)
#     pprint(DenseMatrixDiagnostics.from_dense_matrix(Hk, name=K.matrix_name).as_dict())

# print('\n KH.star_defect:')
# pprint(KH.star_defect())
# print('\n KH_avg.star_defect:')
# pprint(KH_avg.star_defect())

# pprint(KH.star_defect_table().head(20))
# pprint(KH_avg.star_defect_table().head(20))

# pprint(KH_avg.star_defect_table_filtered(min_norm=1e-2, max_radius=3).head(20))

KH_eff = KH_avg_star
KS_eff = KS_avg_star

points = [
    ("Γ", 0.0, 0.0),
    ("K", 2*np.pi/3, -2*np.pi/3),
    ("M", np.pi, 0.0),
    ("Γ", 0.0, 0.0),
]

path = LocalPath.from_points(
    KH_eff,
    KS_eff,
    points,
    points_per_segment=80,
    name="average star",
)

E = path.energies()
print(E.shape)  # (n_kpoints, 8)
