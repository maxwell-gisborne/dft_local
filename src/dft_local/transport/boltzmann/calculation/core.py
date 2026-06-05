"""Boltzmann conductivity implementation for the dft_local package.

This is the local Boltzmann conductivity implementation for dft_local
package. It uses shared core modules for units, kernels, and local
generalized eigenproblems.
"""

from dataclasses import dataclass, field
from typing import Annotated, Self

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

from dft_local.core.local_problem import LocalProblem, SymbolPair
from dft_local.core.numerics import (
    FloatArray,
    Units,
    eVag,
    hermitian_part,
)

from dft_local.core.kernels import GdKernelArrays
from dft_local.core.units import (
    CONDUCTIVITY,
    DIMENSIONLESS,
    ENERGY,
    KSPACE_AREA,
    VELOCITY,
    WAVEVECTOR,
    EV_ANGSTROM_FS,
    SI_UNITS,
    UnitContext,
    qarray,
)


ComplexArray = NDArray[np.complex128]

EnergyBands = Annotated[FloatArray, qarray(ENERGY, ("band",), role="band energies")]
Eigenvectors = Annotated[ComplexArray, qarray(DIMENSIONLESS, ("basis", "band"), role="generalized eigenvectors", dtype=np.complexfloating)]
BandVelocities = Annotated[FloatArray, qarray(VELOCITY, ("cartesian", "band"), role="band velocities")]
ConductivityTensor = Annotated[ComplexArray, qarray(CONDUCTIVITY, ("cartesian", "cartesian"), role="conductivity tensor", dtype=np.complexfloating)]

IrrepPoints = Annotated[FloatArray, qarray(DIMENSIONLESS, ("sample", "irrep_coordinate"), role="raw irrep sample coordinates")]
IrrepWeights = Annotated[FloatArray, qarray(DIMENSIONLESS, ("sample",), role="raw irrep integration weights")]
IrrepToPhysicalK = Annotated[FloatArray, qarray(WAVEVECTOR, ("cartesian", "irrep_coordinate"), role="map from raw irrep coordinates to physical k")]
KResolvedConductivity = Annotated[ComplexArray, qarray(CONDUCTIVITY * KSPACE_AREA.inverse(), ("sample", "cartesian", "cartesian"), role="k-resolved conductivity contribution", dtype=np.complexfloating)]

K_B_HARTREE_PER_K = 3.166811563e-6


def _as_float_array(x: object, *, name: str) -> FloatArray:
    out = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains NaN or Inf")
    return out


def _as_square_float_matrix(x: object, *, name: str) -> FloatArray:
    out = _as_float_array(x, name=name)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError(f"{name} must be square, got shape {out.shape}")
    return out


def fermi_window(
    E: FloatArray,
    *,
    mu: float,
    temperature: float,
    units: Units,
) -> FloatArray:
    """
    Return the positive Fermi-window factor

        - partial_E f_FD(E)

    in the same energy units as `E`.

    The Hamiltonian is assumed to already have been converted from disk units
    using `units.E`. Since `units.E` is the conversion from Hartree to the
    current energy unit, k_B T is

        K_B_HARTREE_PER_K * units.E * temperature

    For `eVag`, this gives k_B in eV / K.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    E = np.asarray(E, dtype=np.float64)
    kBT = K_B_HARTREE_PER_K * units.E * temperature

    x = (E - float(mu)) / kBT
    out = np.empty_like(x, dtype=np.float64)

    positive = x >= 0.0

    # Stable form of exp(x) / (exp(x) + 1)^2 / kBT.
    emx = np.exp(-x[positive])
    out[positive] = emx / (1.0 + emx) ** 2 / kBT

    ex = np.exp(x[~positive])
    out[~positive] = ex / (1.0 + ex) ** 2 / kBT

    return out


def gd_symbol_derivative_generic(
    kernel: GdKernelArrays,
    k1: float,
    k2: float,
    axis: int,
) -> ComplexArray:
    """
    Derivative of `GdKernelArrays.symbol_generic` with respect to an irrep
    coordinate.

    The existing generic symbol uses

        theta = k1 h_m + k2 h_n

    and then uses `exp(i theta)` and its conjugate in the 2D generic irrep.
    This function differentiates that exact convention.

    Parameters
    ----------
    kernel:
        Kernel whose symbol is being differentiated.
    k1, k2:
        Irrep coordinates.
    axis:
        0 for partial / partial k1, 1 for partial / partial k2.
    """

    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    K = kernel.blocks
    q = K.shape[1]

    h = kernel.h_m if axis == 0 else kernel.h_n

    theta = k1 * kernel.h_m + k2 * kernel.h_n
    phase = np.exp(1j * theta)

    dphase = 1j * h * phase
    dphase_conj = -1j * h * np.conj(phase)

    out = np.zeros((2 * q, 2 * q), dtype=np.complex128)

    even = kernel.h_eps == 0
    odd = kernel.h_eps == 1

    if np.any(even):
        K0 = K[even]
        dp0 = dphase[even]
        dpc0 = dphase_conj[even]

        out[0:q, 0:q] += np.einsum("h,hij->ij", dp0, K0)
        out[q:2 * q, q:2 * q] += np.einsum("h,hij->ij", dpc0, K0)

    if np.any(odd):
        K1 = K[odd]
        dp1 = dphase[odd]
        dpc1 = dphase_conj[odd]

        out[0:q, q:2 * q] += np.einsum("h,hij->ij", dpc1, K1)
        out[q:2 * q, 0:q] += np.einsum("h,hij->ij", dp1, K1)

    return out


def gd_symbol_derivative_fixed(
    kernel: GdKernelArrays,
    k1: float,
    k2: float,
    sigma: int,
    axis: int,
) -> ComplexArray:
    """
    Derivative of `GdKernelArrays.symbol_fixed`.

    This is only meaningful where the fixed-point representation itself is
    being used. It follows the existing convention

        coeff = exp(i theta) sigma^eps
    """

    if sigma not in (-1, 1):
        raise ValueError(f"sigma must be ±1, got {sigma}")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    h = kernel.h_m if axis == 0 else kernel.h_n

    theta = k1 * kernel.h_m + k2 * kernel.h_n
    coeff = np.exp(1j * theta) * (sigma ** kernel.h_eps)
    dcoeff = 1j * h * coeff

    return np.einsum("h,hij->ij", dcoeff, kernel.blocks).astype(np.complex128)


def gd_symbol_derivatives(
    pair: SymbolPair,
    kernel: GdKernelArrays,
) -> tuple[ComplexArray, ComplexArray]:
    """
    Return derivatives of a symbol with respect to raw irrep coordinates.

    The returned tuple is

        (partial_k1 symbol, partial_k2 symbol)

    These are not yet physical derivatives unless the irrep coordinates already
    are physical wave-vector coordinates.
    """

    match pair.degree:
        case 2:
            if pair.sigma is not None:
                raise ValueError("sigma should be None for generic 2D irrep")

            return (
                gd_symbol_derivative_generic(kernel, pair.k1, pair.k2, axis=0),
                gd_symbol_derivative_generic(kernel, pair.k1, pair.k2, axis=1),
            )

        case 1:
            if pair.sigma is None:
                raise ValueError("sigma is required for fixed-point 1D irrep")

            return (
                gd_symbol_derivative_fixed(kernel, pair.k1, pair.k2, pair.sigma, axis=0),
                gd_symbol_derivative_fixed(kernel, pair.k1, pair.k2, pair.sigma, axis=1),
            )

        case _:
            raise ValueError(f"Unsupported irrep degree: {pair.degree}")


@dataclass(frozen=True, slots=True)
class BoltzmannSampleResult:
    """
    Output for one sampled irrep point.

    `velocities` has shape `(dimension, nbands)`.
    `sigma` has shape `(dimension, dimension)`.
    """

    energies: EnergyBands
    vectors: Eigenvectors
    velocities: BandVelocities
    ac_weights: ComplexArray
    sigma: ConductivityTensor


@dataclass(frozen=True, slots=True)
class BoltzmannConductivity:
    """
    AC Boltzmann conductivity over an irregular set of irrep samples.

    This class does not do band continuation. It solves each local generalized
    eigenproblem independently, computes the diagonal band velocities from the
    generalized Hellmann-Feynman expression, and stores the k-resolved
    conductivity contribution.

    The calculation is

        H(k) U(k) = S(k) U(k) E(k)

    with

        U(k)^dagger S(k) U(k) = I

    The diagonal velocity is

        v_(n,i) =
            1 / hbar
            u_n^dagger (partial_i H - E_n partial_i S) u_n

    Here `partial_i` means derivative with respect to physical k. If the raw
    irrep coordinates are alpha and

        k_physical = irrep_to_physical_k @ alpha

    then derivatives are converted using

        partial / partial k_i =
            sum_a (irrep_to_physical_k^-1)_(a i)
            partial / partial alpha_a

    Integration weights are likewise converted using

        d^d k = abs(det(irrep_to_physical_k)) d^d alpha

    Parameters
    ----------
    problems:
        Flat array of `LocalProblem` objects, one per sample.
    irrep_points:
        Raw irrep coordinates, shape `(nk, dimension)`.
    irrep_weights:
        Integration weights in raw irrep-coordinate measure.
    irrep_to_physical_k:
        Matrix mapping raw irrep coordinates to physical k coordinates.
        If `units=eVag`, this should usually map into inverse Angstrom.
    units:
        Unit system used by the symbols.
    mu:
        Chemical potential in the same energy unit as the eigenvalues.
    temperature:
        Temperature in Kelvin.
    omega:
        Angular frequency. Must be inverse time compatible with `tau`.
    tau:
        Relaxation time. May be scalar, shape `(nbands,)`, or shape
        `(nk, nbands)`.
    charge:
        Carrier charge. The sign does not affect the present q^2 expression,
        but keeping it explicit makes later Hall terms less confusing.
    """

    problems: NDArray[np.object_]
    irrep_points: IrrepPoints
    irrep_weights: IrrepWeights
    irrep_to_physical_k: IrrepToPhysicalK

    units: Units = eVag
    mu: float = 0.0
    temperature: float = 300.0
    omega: float = 0.0
    tau: float | FloatArray = 1.0
    charge: float | None = None

    symmetrise: bool = True
    check_overlap: bool = True
    overlap_tol: float = 1e-10

    name: str = ""

    @property
    def unit_context(self) -> UnitContext:
        """Best-effort bridge from legacy numerics.Units to core UnitContext."""

        if self.units == eVag or getattr(self.units, "name", "") == "angstroem":
            return EV_ANGSTROM_FS

        return SI_UNITS

    energies: EnergyBands | None = field(default=None, init=False)
    vectors: ComplexArray | None = field(default=None, init=False)
    velocities: FloatArray | None = field(default=None, init=False)
    ac_weights: ComplexArray | None = field(default=None, init=False)
    sigma_k: KResolvedConductivity | None = field(default=None, init=False)
    sigma: ConductivityTensor | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        problems = np.asarray(self.problems, dtype=object).reshape(-1)
        irrep_points = _as_float_array(self.irrep_points, name="irrep_points")
        irrep_weights = _as_float_array(self.irrep_weights, name="irrep_weights")
        irrep_to_physical_k = _as_square_float_matrix(
            self.irrep_to_physical_k,
            name="irrep_to_physical_k",
        )

        if irrep_points.ndim != 2:
            raise ValueError(
                f"irrep_points must have shape (nk, dimension), got {irrep_points.shape}"
            )

        nk, dimension = irrep_points.shape

        if len(problems) != nk:
            raise ValueError(
                f"number of problems and irrep points differ: {len(problems)} != {nk}"
            )

        if irrep_weights.shape != (nk,):
            raise ValueError(
                f"irrep_weights must have shape {(nk,)}, got {irrep_weights.shape}"
            )

        if irrep_to_physical_k.shape != (dimension, dimension):
            raise ValueError(
                "irrep_to_physical_k shape must match irrep dimension: "
                f"{irrep_to_physical_k.shape} != {(dimension, dimension)}"
            )

        if abs(np.linalg.det(irrep_to_physical_k)) <= 0.0:
            raise ValueError("irrep_to_physical_k must be invertible")

        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

        object.__setattr__(self, "problems", problems)
        object.__setattr__(self, "irrep_points", irrep_points)
        object.__setattr__(self, "irrep_weights", irrep_weights)
        object.__setattr__(self, "irrep_to_physical_k", irrep_to_physical_k)

    @classmethod
    def from_arrays(
        cls,
        KH: GdKernelArrays,
        KS: GdKernelArrays,
        k1: FloatArray,
        k2: FloatArray,
        *,
        irrep_weights: FloatArray | None = None,
        irrep_to_physical_k: FloatArray | None = None,
        units: Units = eVag,
        mu: float = 0.0,
        temperature: float = 300.0,
        omega: float = 0.0,
        tau: float | FloatArray = 1.0,
        charge: float | None = None,
        name: str = "",
    ) -> Self:
        """
        Build from arrays of irrep coordinates.

        This mirrors `LocalPath.from_arrays`, but it does not create paths and
        does not do continuation. The arrays are flattened, so regular grids,
        irregular grids, and random samples all become the same internal sample
        list.
        """

        k1 = _as_float_array(k1, name="k1")
        k2 = _as_float_array(k2, name="k2")

        if k1.shape != k2.shape:
            raise ValueError(f"k1/k2 shape mismatch: {k1.shape} != {k2.shape}")

        flat_k1 = k1.reshape(-1)
        flat_k2 = k2.reshape(-1)
        nk = flat_k1.size

        if irrep_weights is None:
            irrep_weights = np.full(nk, 1.0 / nk, dtype=np.float64)
        else:
            irrep_weights = _as_float_array(irrep_weights, name="irrep_weights").reshape(-1)

        if irrep_weights.shape != (nk,):
            raise ValueError(
                f"irrep_weights must flatten to {(nk,)}, got {irrep_weights.shape}"
            )

        if irrep_to_physical_k is None:
            irrep_to_physical_k = np.eye(2, dtype=np.float64)

        problems = np.empty(nk, dtype=object)

        for s, (a, b) in enumerate(zip(flat_k1, flat_k2)):
            problems[s] = SymbolPair(
                KH=KH,
                KS=KS,
                k1=float(a),
                k2=float(b),
                name=name,
            ).form()

        irrep_points = np.column_stack([flat_k1, flat_k2])

        return cls(
            problems=problems,
            irrep_points=irrep_points,
            irrep_weights=irrep_weights,
            irrep_to_physical_k=irrep_to_physical_k,
            units=units,
            mu=mu,
            temperature=temperature,
            omega=omega,
            tau=tau,
            charge=charge,
            name=name,
        )

    @property
    def nk(self) -> int:
        return self.irrep_points.shape[0]

    @property
    def dimension(self) -> int:
        return self.irrep_points.shape[1]

    @property
    def physical_k_points(self) -> FloatArray:
        return self.irrep_points @ self.irrep_to_physical_k.T

    @property
    def physical_k_weights(self) -> FloatArray:
        jacobian = abs(float(np.linalg.det(self.irrep_to_physical_k)))
        return self.irrep_weights * jacobian / (2.0 * np.pi) ** self.dimension

    @property
    def charge_value(self) -> float:
        if self.charge is None:
            return -float(self.units.e)
        return float(self.charge)

    def tau_for_sample(self, sample_index: int, nbands: int) -> FloatArray:
        tau = np.asarray(self.tau, dtype=np.float64)

        if tau.ndim == 0:
            return np.full(nbands, float(tau), dtype=np.float64)

        if tau.shape == (nbands,):
            return tau

        if tau.shape == (self.nk, nbands):
            return tau[sample_index]

        raise ValueError(
            "tau must be scalar, shape (nbands,), or shape (nk, nbands); "
            f"got {tau.shape}"
        )

    def prepared_problem(self, sample_index: int) -> LocalProblem:
        problem = self.problems[sample_index]

        if not isinstance(problem, LocalProblem):
            raise TypeError(
                f"problems[{sample_index}] is not a LocalProblem: {type(problem)}"
            )

        if self.symmetrise:
            problem = problem.symmetrised()

        if self.check_overlap:
            problem.check_overlap_positive(tol=self.overlap_tol)

        return problem

    def solve_problem(self, problem: LocalProblem) -> tuple[FloatArray, ComplexArray]:
        E, U = eigh(problem.Hk, problem.Sk, eigvals_only=False)
        return np.asarray(E, dtype=np.float64), np.asarray(U, dtype=np.complex128)

    def physical_derivative_symbols(
        self,
        problem: LocalProblem,
    ) -> tuple[list[ComplexArray], list[ComplexArray]]:
        """
        Return derivative symbols with respect to physical k coordinates.

        First builds raw irrep-coordinate derivatives, then applies the inverse
        Jacobian from raw irrep coordinates to physical k.
        """

        dH_raw = gd_symbol_derivatives(problem.pair, problem.pair.KH)
        dS_raw = gd_symbol_derivatives(problem.pair, problem.pair.KS)

        if self.dimension != len(dH_raw):
            raise ValueError(
                f"sample dimension {self.dimension} but symbol has {len(dH_raw)} derivatives"
            )

        inv_j = np.linalg.inv(self.irrep_to_physical_k)

        dH_physical = []
        dS_physical = []

        for i in range(self.dimension):
            dH_i = sum(inv_j[a, i] * dH_raw[a] for a in range(self.dimension))
            dS_i = sum(inv_j[a, i] * dS_raw[a] for a in range(self.dimension))

            if self.symmetrise:
                dH_i = hermitian_part(dH_i)
                dS_i = hermitian_part(dS_i)

            dH_physical.append(np.asarray(dH_i, dtype=np.complex128))
            dS_physical.append(np.asarray(dS_i, dtype=np.complex128))

        return dH_physical, dS_physical

    def band_velocities(
        self,
        problem: LocalProblem,
        E: FloatArray,
        U: ComplexArray,
    ) -> FloatArray:
        """
        Compute diagonal Boltzmann velocities.

        Output shape is `(dimension, nbands)`.
        """

        dH, dS = self.physical_derivative_symbols(problem)

        nbands = E.shape[0]
        velocities = np.empty((self.dimension, nbands), dtype=np.float64)

        for i in range(self.dimension):
            for n in range(nbands):
                u = U[:, n]
                numerator = np.vdot(u, (dH[i] - E[n] * dS[i]) @ u)
                velocities[i, n] = float(np.real(numerator)) / float(self.units.hbar)

        return velocities

    def velocity_matrix_eig(
        self,
        problem: LocalProblem,
        E: FloatArray,
        U: ComplexArray,
        direction: int,
    ) -> ComplexArray:
        """
        Full velocity matrix in the eigenbasis.

        This includes off-diagonal interband matrix elements. The Boltzmann
        calculation uses only the diagonal, but this is useful for diagnostics
        and later Kubo-like comparisons.
        """

        dH, dS = self.physical_derivative_symbols(problem)

        M = U.conj().T @ dH[direction] @ U
        N = U.conj().T @ dS[direction] @ U

        E_left = E[:, None]
        E_right = E[None, :]

        return (M - 0.5 * (E_left + E_right) * N) / float(self.units.hbar)

    def ac_weight(
        self,
        E: FloatArray,
        sample_index: int,
    ) -> ComplexArray:
        """
        Compute the AC Boltzmann weight for each band at one sample.

        The returned array is

            q^2 tau / (1 - i omega tau) [-partial_E f_FD(E)]
        """

        tau = self.tau_for_sample(sample_index, len(E))

        if np.any(tau < 0.0):
            raise ValueError("tau must be non-negative")

        window = fermi_window(
            E,
            mu=self.mu,
            temperature=self.temperature,
            units=self.units,
        )

        q = self.charge_value

        return (
            q * q
            * tau
            / (1.0 - 1j * float(self.omega) * tau)
            * window
        ).astype(np.complex128)

    def compute_sample(self, sample_index: int) -> BoltzmannSampleResult:
        """
        Compute the k-resolved conductivity contribution at one sample.
        """

        problem = self.prepared_problem(sample_index)
        E, U = self.solve_problem(problem)

        velocities = self.band_velocities(problem, E, U)
        weights = self.ac_weight(E, sample_index)

        sigma = np.einsum(
            "n,in,jn->ij",
            weights,
            velocities,
            velocities,
            optimize=True,
        )

        sigma *= self.physical_k_weights[sample_index]

        return BoltzmannSampleResult(
            energies=E,
            vectors=U,
            velocities=velocities,
            ac_weights=weights,
            sigma=np.asarray(sigma, dtype=np.complex128),
        )

    def run(self) -> Self:
        """
        Run the full calculation.

        Arrays are allocated after the first sample determines the number of
        bands and state-space dimension, then reused for the remaining samples.
        The dataclass is frozen, but the stored numpy arrays are mutable.
        """

        if self.nk == 0:
            raise ValueError("Cannot run conductivity on zero samples")

        first = self.compute_sample(0)

        nbands = first.energies.shape[0]
        state_dim = first.vectors.shape[0]

        energies = np.empty((self.nk, nbands), dtype=np.float64)
        vectors = np.empty((self.nk, state_dim, nbands), dtype=np.complex128)
        velocities = np.empty((self.nk, self.dimension, nbands), dtype=np.float64)
        ac_weights = np.empty((self.nk, nbands), dtype=np.complex128)
        sigma_k = np.empty((self.nk, self.dimension, self.dimension), dtype=np.complex128)

        self._store_sample(0, first, energies, vectors, velocities, ac_weights, sigma_k)

        for sample_index in range(1, self.nk):
            result = self.compute_sample(sample_index)

            if result.energies.shape != (nbands,):
                raise ValueError(
                    f"band count changed at sample {sample_index}: "
                    f"{result.energies.shape} != {(nbands,)}"
                )

            if result.vectors.shape != (state_dim, nbands):
                raise ValueError(
                    f"eigenvector shape changed at sample {sample_index}: "
                    f"{result.vectors.shape} != {(state_dim, nbands)}"
                )

            self._store_sample(
                sample_index,
                result,
                energies,
                vectors,
                velocities,
                ac_weights,
                sigma_k,
            )

        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "ac_weights", ac_weights)
        object.__setattr__(self, "sigma_k", sigma_k)
        object.__setattr__(self, "sigma", np.sum(sigma_k, axis=0))

        return self

    @staticmethod
    def _store_sample(
        sample_index: int,
        result: BoltzmannSampleResult,
        energies: FloatArray,
        vectors: ComplexArray,
        velocities: FloatArray,
        ac_weights: ComplexArray,
        sigma_k: ComplexArray,
    ) -> None:
        energies[sample_index, :] = result.energies
        vectors[sample_index, :, :] = result.vectors
        velocities[sample_index, :, :] = result.velocities
        ac_weights[sample_index, :] = result.ac_weights
        sigma_k[sample_index, :, :] = result.sigma

    def require_solved(self) -> None:
        if (
            self.energies is None
            or self.vectors is None
            or self.velocities is None
            or self.ac_weights is None
            or self.sigma_k is None
            or self.sigma is None
        ):
            raise ValueError("BoltzmannConductivity has not been run")

    def diagnostics(self) -> dict[str, object]:
        """
        Return scalar diagnostics for checking a completed calculation.
        """

        out: dict[str, object] = {
            "name": self.name,
            "nk": self.nk,
            "dimension": self.dimension,
            "units": repr(self.units),
            "mu": float(self.mu),
            "temperature": float(self.temperature),
            "omega": float(self.omega),
            "irrep_weight_sum": float(np.sum(self.irrep_weights)),
            "physical_k_weight_sum": float(np.sum(self.physical_k_weights)),
            "physical_k_jacobian": float(abs(np.linalg.det(self.irrep_to_physical_k))),
            "solved": self.sigma is not None,
        }

        if self.sigma is None:
            return out

        assert self.energies is not None
        assert self.velocities is not None
        assert self.ac_weights is not None
        assert self.sigma_k is not None

        out.update(
            {
                "nbands": int(self.energies.shape[1]),
                "energy_min": float(np.min(self.energies)),
                "energy_max": float(np.max(self.energies)),
                "velocity_abs_max": float(np.max(np.abs(self.velocities))),
                "ac_weight_abs_max": float(np.max(np.abs(self.ac_weights))),
                "sigma_norm": float(np.linalg.norm(self.sigma)),
                "sigma_real_norm": float(np.linalg.norm(self.sigma.real)),
                "sigma_imag_norm": float(np.linalg.norm(self.sigma.imag)),
                "sigma_k_abs_max": float(np.max(np.abs(self.sigma_k))),
                "finite": bool(
                    np.all(np.isfinite(self.energies))
                    and np.all(np.isfinite(self.velocities))
                    and np.all(np.isfinite(self.ac_weights))
                    and np.all(np.isfinite(self.sigma_k))
                    and np.all(np.isfinite(self.sigma))
                ),
            }
        )

        return out

    def diagnostics_rows(self) -> list[list[object]]:
        """
        Diagnostics as rows for your existing HTML table helper.
        """

        return [[key, value] for key, value in self.diagnostics().items()]
