"""Small numeric helpers and unit definitions for the dft_local package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.linalg import eigvalsh
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BlockArray = NDArray[np.float64] | NDArray[np.complex128]


def freeze_array(a):
    """Mark an array read-only and return it."""

    a.flags.writeable = False
    return a


def hermitian_part(A: np.ndarray) -> np.ndarray:
    """Return the Hermitian part of a dense matrix."""

    return 0.5 * (A + A.conj().T)



@dataclass(frozen=True, slots=True)
class DenseMatrixDiagnostics:
    """Basic diagnostics for a dense square matrix."""

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
            Ah = hermitian_part(A)
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
