"""Local generalized eigenproblem objects.

This module owns the small `SymbolPair -> LocalProblem` layer in dft_local
package.  It uses local kernel and numerical utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Self

import numpy as np
from scipy.linalg import eigh, eigvalsh

from dft_local.core.numerics import DenseMatrixDiagnostics, hermitian_part

from dft_local.core.kernels import GdKernelArrays


@dataclass(frozen=True, slots=True)
class SymbolPair:
    """Pair of Hamiltonian and overlap symbols at one irrep coordinate."""

    KH: GdKernelArrays
    KS: GdKernelArrays
    k1: float
    k2: float
    degree: int = 2
    sigma: int | None = None
    name: str = ""

    def star_symmetrised(self) -> Self:
        """Return a symbol pair with star-symmetrised kernels."""

        return replace(
            self,
            KH=self.KH.star_symmetrised(matrix_name=self.name + " star"),
            KS=self.KS.star_symmetrised(matrix_name=self.name + " star"),
        )

    def form(self) -> "LocalProblem":
        """Construct the dense generalized local eigenproblem."""

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
        """Human-readable irrep label."""

        if self.degree == 2:
            return f"k=({self.k1:.6g},{self.k2:.6g}), degree=2"

        return f"k=({self.k1:.6g},{self.k2:.6g}), degree=1, sigma={self.sigma}"


@dataclass(frozen=True, slots=True)
class LocalProblem:
    """Dense generalized eigenproblem `H(k) phi = E S(k) phi`."""

    Hk: np.ndarray
    Sk: np.ndarray
    pair: SymbolPair

    def symmetrised(self) -> Self:
        """Return problem with Hermitian parts of H and S."""

        return replace(
            self,
            Hk=hermitian_part(self.Hk),
            Sk=hermitian_part(self.Sk),
        )

    def overlap_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the Hermitian part of the overlap matrix."""

        return eigvalsh(hermitian_part(self.Sk))

    def check_overlap_positive(self, tol: float = 1e-10) -> None:
        """Raise if the overlap symbol is not positive definite."""

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
        """Solve the generalized eigenproblem."""

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
        """Return sorted generalized eigenvalues."""

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
        """Return generalized eigenvalues and eigenvectors."""

        return self.solve(
            symmetrise=symmetrise,
            check_overlap=check_overlap,
            overlap_tol=overlap_tol,
            eigvals_only=False,
        )

    def diagnostics(self) -> dict:
        """Return dense matrix diagnostics for H(k), S(k), and energies."""

        Hdiag = DenseMatrixDiagnostics.from_dense_matrix(
            self.Hk,
            name="H(k)",
            check_eigenvalues=False,
        )

        Sdiag = DenseMatrixDiagnostics.from_dense_matrix(
            self.Sk,
            name="S(k)",
            check_eigenvalues=True,
        )

        return {
            "pair": self.pair.label(),
            "H": Hdiag.as_dict(),
            "S": Sdiag.as_dict(),
            "energies": self.energies().tolist(),
        }
