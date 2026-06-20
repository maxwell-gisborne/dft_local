"""Core validation helpers for the Boltzmann operator approach.

This module is deliberately independent of Vincent/Ashcroft comparison data.
The goal is to collect small analytic and algebraic checks that validate the
operator formulation before it is compared with any external implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OperatorValidationSummary:
    """Compact status summary for the operator-validation domain."""

    purpose: str
    current_scope: tuple[str, ...]
    planned_checks: tuple[str, ...]


def validation_summary() -> OperatorValidationSummary:
    """Return the initial validation plan for the Boltzmann operator approach."""

    return OperatorValidationSummary(
        purpose=(
            "Validate the Boltzmann operator approach independently of any "
            "single reference calculation."
        ),
        current_scope=(
            "define analytic test problems",
            "check operator algebra and tensor assembly",
            "separate local correctness from external convention matching",
        ),
        planned_checks=(
            "identity/operator-shape checks",
            "linearity checks",
            "positivity and symmetry checks",
            "known-function end-to-end conductivity checks",
            "basis-change covariance checks",
            "grid-measure and normalisation checks",
            "relaxation-time and velocity-scale laws",
        ),
    )


def symmetric_part(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""

    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"expected square matrix, got shape {array.shape}")

    return 0.5 * (array + array.T)


def antisymmetric_relative_norm(matrix: np.ndarray) -> float:
    """Return ||A - A.T|| / ||A||, with a safe zero convention."""

    array = np.asarray(matrix, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return 0.0

    return float(np.linalg.norm(array - array.T) / norm)


def is_positive_semidefinite(matrix: np.ndarray, *, tolerance: float = 1.0e-12) -> bool:
    """Check positive semidefiniteness of the symmetric part."""

    sym = symmetric_part(matrix)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(sym)))

    return min_eigenvalue >= -tolerance


def weighted_outer_product_tensor(velocity: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Assemble sum_k w_k v_a(k) v_b(k).

    This is the core tensor structure behind relaxation-time conductivity.
    """

    velocity = np.asarray(velocity, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)

    if velocity.ndim < 2 or velocity.shape[-1] != 2:
        raise ValueError("velocity must have shape (..., 2)")

    if weight.shape != velocity.shape[:-1]:
        raise ValueError(
            f"weight shape {weight.shape} does not match velocity grid shape {velocity.shape[:-1]}"
        )

    return np.einsum("ija,ijb,ij->ab", velocity, velocity, weight)


def tensor_invariant_report(tensor: np.ndarray) -> dict[str, float]:
    """Return simple invariants for a conductivity-like tensor."""

    tensor = np.asarray(tensor, dtype=np.float64)
    sym = symmetric_part(tensor)
    eigenvalues = np.linalg.eigvalsh(sym)
    trace = float(np.trace(tensor))
    scale = abs(trace) if trace != 0.0 else 1.0

    return {
        "trace": trace,
        "minimum_symmetric_eigenvalue": float(np.min(eigenvalues)),
        "antisymmetric_relative_norm": antisymmetric_relative_norm(tensor),
        "diagonal_anisotropy_over_trace": float(abs(tensor[0, 0] - tensor[1, 1]) / scale),
        "offdiagonal_over_trace": float(max(abs(tensor[0, 1]), abs(tensor[1, 0])) / scale),
    }



def symbol_from_kernel(kernel: np.ndarray) -> np.ndarray:
    """Return the discrete Fourier symbol of a translation-invariant operator.

    The kernel convention is circular convolution on a finite group:

        (K f)[g] = sum_h K[h] f[g - h]

    With NumPy FFT conventions, this means

        fft(K f) = fft(K) fft(f)

    so the symbol is simply fft2(kernel).
    """

    kernel = np.asarray(kernel, dtype=np.complex128)

    if kernel.ndim != 2:
        raise ValueError(f"expected 2D kernel, got shape {kernel.shape}")

    return np.fft.fft2(kernel)


def apply_operator_from_kernel(kernel: np.ndarray, function: np.ndarray) -> np.ndarray:
    """Apply a translation-invariant operator by circular convolution."""

    kernel = np.asarray(kernel, dtype=np.complex128)
    function = np.asarray(function, dtype=np.complex128)

    if kernel.shape != function.shape:
        raise ValueError(f"kernel shape {kernel.shape} != function shape {function.shape}")

    return np.fft.ifft2(np.fft.fft2(kernel) * np.fft.fft2(function))


def apply_operator_from_symbol(symbol: np.ndarray, function: np.ndarray) -> np.ndarray:
    """Apply a translation-invariant operator from its Fourier symbol."""

    symbol = np.asarray(symbol, dtype=np.complex128)
    function = np.asarray(function, dtype=np.complex128)

    if symbol.shape != function.shape:
        raise ValueError(f"symbol shape {symbol.shape} != function shape {function.shape}")

    return np.fft.ifft2(symbol * np.fft.fft2(function))


def reconstruct_kernel_from_symbol(symbol: np.ndarray) -> np.ndarray:
    """Recover the circular-convolution kernel from its symbol."""

    symbol = np.asarray(symbol, dtype=np.complex128)

    if symbol.ndim != 2:
        raise ValueError(f"expected 2D symbol, got shape {symbol.shape}")

    return np.fft.ifft2(symbol)


def central_difference_kernel(
    shape: tuple[int, int],
    *,
    axis: int,
    spacing: float,
) -> np.ndarray:
    """Return a periodic central-difference kernel for a finite 2D group."""

    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    if spacing <= 0.0:
        raise ValueError(f"spacing must be positive, got {spacing}")

    kernel = np.zeros(shape, dtype=np.complex128)

    plus = [0, 0]
    minus = [0, 0]

    # Kernel convention: (K f)[g] = sum_h K[h] f[g - h].
    # To get f[g + e] - f[g - e], coefficients sit at h=-e and h=+e.
    plus[axis] = shape[axis] - 1
    minus[axis] = 1

    kernel[tuple(plus)] = 1.0 / (2.0 * spacing)
    kernel[tuple(minus)] = -1.0 / (2.0 * spacing)

    return kernel


def central_difference_symbol(
    shape: tuple[int, int],
    *,
    axis: int,
    spacing: float,
) -> np.ndarray:
    """Return the analytic symbol of the periodic central-difference operator."""

    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    if spacing <= 0.0:
        raise ValueError(f"spacing must be positive, got {spacing}")

    frequencies = np.fft.fftfreq(shape[axis]) * shape[axis]
    theta = 2.0 * np.pi * frequencies / shape[axis]
    one_dimensional = 1j * np.sin(theta) / spacing

    if axis == 0:
        return one_dimensional[:, None] * np.ones((1, shape[1]), dtype=np.complex128)

    return np.ones((shape[0], 1), dtype=np.complex128) * one_dimensional[None, :]


def finite_group_mode(
    shape: tuple[int, int],
    mode: tuple[int, int],
) -> np.ndarray:
    """Return exp(2π i (m i / N0 + n j / N1)) on Z_N0 x Z_N1."""

    i = np.arange(shape[0], dtype=np.float64)[:, None]
    j = np.arange(shape[1], dtype=np.float64)[None, :]

    phase = (
        2.0 * np.pi * mode[0] * i / shape[0]
        + 2.0 * np.pi * mode[1] * j / shape[1]
    )

    return np.exp(1j * phase)


def periodic_cosine_energy_surface(
    shape: tuple[int, int],
    *,
    mu: float,
    amplitude_x: float,
    amplitude_y: float,
) -> np.ndarray:
    """Return a separable periodic test energy surface."""

    i = np.arange(shape[0], dtype=np.float64)[:, None]
    j = np.arange(shape[1], dtype=np.float64)[None, :]

    theta_x = 2.0 * np.pi * i / shape[0]
    theta_y = 2.0 * np.pi * j / shape[1]

    return (
        mu
        + amplitude_x * np.cos(theta_x)
        + amplitude_y * np.cos(theta_y)
    )


def analytic_central_difference_of_cosine_energy(
    shape: tuple[int, int],
    *,
    axis: int,
    spacing: float,
    amplitude_x: float,
    amplitude_y: float,
) -> np.ndarray:
    """Exact central-periodic finite-difference derivative of the cosine surface."""

    if axis == 0:
        i = np.arange(shape[0], dtype=np.float64)[:, None]
        theta = 2.0 * np.pi * i / shape[0]
        derivative = (
            -amplitude_x
            * np.sin(theta)
            * np.sin(2.0 * np.pi / shape[0])
            / spacing
        )
        return np.broadcast_to(derivative, shape)

    if axis == 1:
        j = np.arange(shape[1], dtype=np.float64)[None, :]
        theta = 2.0 * np.pi * j / shape[1]
        derivative = (
            -amplitude_y
            * np.sin(theta)
            * np.sin(2.0 * np.pi / shape[1])
            / spacing
        )
        return np.broadcast_to(derivative, shape)

    raise ValueError(f"axis must be 0 or 1, got {axis}")


def operator_symbol_validation_probe() -> dict[str, float]:
    """Run compact symbol/operator validation checks for diagnostics."""

    shape = (17, 19)
    dx = 0.25
    dy = 0.40

    identity_kernel = np.zeros(shape, dtype=np.complex128)
    identity_kernel[0, 0] = 1.0

    mode = finite_group_mode(shape, (3, 5))
    identity_symbol = symbol_from_kernel(identity_kernel)
    identity_applied = apply_operator_from_symbol(identity_symbol, mode)

    dx_kernel = central_difference_kernel(shape, axis=0, spacing=dx)
    dy_kernel = central_difference_kernel(shape, axis=1, spacing=dy)

    dx_symbol_from_kernel = symbol_from_kernel(dx_kernel)
    dy_symbol_from_kernel = symbol_from_kernel(dy_kernel)

    dx_symbol_expected = central_difference_symbol(shape, axis=0, spacing=dx)
    dy_symbol_expected = central_difference_symbol(shape, axis=1, spacing=dy)

    mu = -0.2
    ax = 0.03
    ay = 0.02
    energy = periodic_cosine_energy_surface(
        shape,
        mu=mu,
        amplitude_x=ax,
        amplitude_y=ay,
    )

    dx_energy = apply_operator_from_symbol(dx_symbol_from_kernel, energy).real
    dy_energy = apply_operator_from_symbol(dy_symbol_from_kernel, energy).real

    dx_energy_expected = analytic_central_difference_of_cosine_energy(
        shape,
        axis=0,
        spacing=dx,
        amplitude_x=ax,
        amplitude_y=ay,
    )
    dy_energy_expected = analytic_central_difference_of_cosine_energy(
        shape,
        axis=1,
        spacing=dy,
        amplitude_x=ax,
        amplitude_y=ay,
    )

    kernel_roundtrip = reconstruct_kernel_from_symbol(dx_symbol_from_kernel)

    return {
        "identity_mode_relative_error": float(
            np.linalg.norm(identity_applied - mode) / np.linalg.norm(mode)
        ),
        "kernel_symbol_roundtrip_error": float(
            np.linalg.norm(kernel_roundtrip - dx_kernel) / np.linalg.norm(dx_kernel)
        ),
        "dx_symbol_relative_error": float(
            np.linalg.norm(dx_symbol_from_kernel - dx_symbol_expected)
            / np.linalg.norm(dx_symbol_expected)
        ),
        "dy_symbol_relative_error": float(
            np.linalg.norm(dy_symbol_from_kernel - dy_symbol_expected)
            / np.linalg.norm(dy_symbol_expected)
        ),
        "dx_energy_surface_relative_error": float(
            np.linalg.norm(dx_energy - dx_energy_expected)
            / np.linalg.norm(dx_energy_expected)
        ),
        "dy_energy_surface_relative_error": float(
            np.linalg.norm(dy_energy - dy_energy_expected)
            / np.linalg.norm(dy_energy_expected)
        ),
    }



from scipy.linalg import eigh

from dft_local.core.dataset import LEGACY_EV_ANGSTROM_CONTEXT
from dft_local.core.kernels import GdKernelArrays
from dft_local.core.local_problem import SymbolPair
from dft_local.core.numerics import DenseMatrixDiagnostics, hermitian_part
from dft_local.core.units import ATOMIC_UNITS
from dft_local.transport.boltzmann.calculation.core import (
    gd_symbol_derivative_fixed,
    gd_symbol_derivative_generic,
    gd_symbol_derivatives,
)


def gd_kernel(
    h_m,
    h_n,
    h_eps,
    blocks,
    *,
    name: str = "validation kernel",
) -> GdKernelArrays:
    """Construct a production GdKernelArrays object for validation tests."""

    return GdKernelArrays(
        h_m=np.asarray(h_m, dtype=np.int64),
        h_n=np.asarray(h_n, dtype=np.int64),
        h_eps=np.asarray(h_eps, dtype=np.int64),
        blocks=np.asarray(blocks, dtype=np.complex128),
        matrix_name=name,
    )


def gd_identity_overlap_kernel(q: int = 1) -> GdKernelArrays:
    """Identity overlap kernel using the production symbol mechanism."""

    return gd_kernel(
        [0],
        [0],
        [0],
        [np.eye(q, dtype=np.complex128)],
        name="identity overlap",
    )


def gd_cosine_k1_kernel(scale: float = 1.0, q: int = 1) -> GdKernelArrays:
    """Kernel whose fixed/generic production symbol contains scale*cos(k1)."""

    block = 0.5 * scale * np.eye(q, dtype=np.complex128)

    return gd_kernel(
        [1, -1],
        [0, 0],
        [0, 0],
        [block, block],
        name="cos(k1) kernel",
    )


def gd_cosine_k2_kernel(scale: float = 1.0, q: int = 1) -> GdKernelArrays:
    """Kernel whose fixed/generic production symbol contains scale*cos(k2)."""

    block = 0.5 * scale * np.eye(q, dtype=np.complex128)

    return gd_kernel(
        [0, 0],
        [1, -1],
        [0, 0],
        [block, block],
        name="cos(k2) kernel",
    )


def gd_separable_cosine_kernel(
    *,
    c0: float,
    c1: float,
    c2: float,
) -> GdKernelArrays:
    """Scalar kernel with symbol c0 + c1 cos(k1) + c2 cos(k2)."""

    return gd_kernel(
        [0, 1, -1, 0, 0],
        [0, 0, 0, 1, -1],
        [0, 0, 0, 0, 0],
        [
            [[c0]],
            [[0.5 * c1]],
            [[0.5 * c1]],
            [[0.5 * c2]],
            [[0.5 * c2]],
        ],
        name="separable cosine kernel",
    )


def expected_separable_cosine_symbol(k1: float, k2: float, *, c0: float, c1: float, c2: float) -> float:
    """Expected scalar symbol for gd_separable_cosine_kernel."""

    return float(c0 + c1 * np.cos(k1) + c2 * np.cos(k2))


def expected_separable_cosine_derivative(
    k1: float,
    k2: float,
    *,
    axis: int,
    c1: float,
    c2: float,
) -> float:
    """Expected derivative of c0 + c1 cos(k1) + c2 cos(k2)."""

    if axis == 0:
        return float(-c1 * np.sin(k1))

    if axis == 1:
        return float(-c2 * np.sin(k2))

    raise ValueError(f"axis must be 0 or 1, got {axis}")


def generic_symbol_scalar_channels(symbol: np.ndarray) -> np.ndarray:
    """Return eigenvalues of a Hermitian scalar generic symbol.

    For even scalar kernels, generic degree-2 symbols are two equivalent
    channels. Eigenvalues are a convention-independent way to check the symbol
    without relying on block ordering.
    """

    return np.linalg.eigvalsh(hermitian_part(symbol))



def finite_field_input_health_probe(
    *,
    n_u: int = 11,
    n_v: int = 11,
    symmetrization: str = "star",
) -> dict[str, float | int | bool | str | None]:
    """Probe H/S symbol health for the finite-field DC validation scaffold.

    This first version uses controlled production kernels rather than the full
    dataset. It validates the same `GdKernelArrays -> SymbolPair -> LocalProblem`
    path used by the real calculation while keeping the diagnostic lightweight.
    """

    if n_u < 1:
        raise ValueError(f"n_u must be positive, got {n_u}")
    if n_v < 1:
        raise ValueError(f"n_v must be positive, got {n_v}")
    if symmetrization not in {"star", "direct", "raw"}:
        raise ValueError(f"unknown symmetrization scheme: {symmetrization!r}")

    KH = gd_separable_cosine_kernel(c0=1.25, c1=0.70, c2=-0.30)
    KS = gd_identity_overlap_kernel()

    if symmetrization == "star":
        KH = KH.star_symmetrised(matrix_name="finite-field validation H star")
        KS = KS.star_symmetrised(matrix_name="finite-field validation S star")

    k1_grid = np.linspace(-np.pi, np.pi, int(n_u), endpoint=False)
    k2_grid = np.linspace(-np.pi, np.pi, int(n_v), endpoint=False)

    max_h_hermitian_defect = 0.0
    max_s_hermitian_defect = 0.0
    min_s_eig = np.inf
    max_s_cond = 0.0
    max_energy_jump = 0.0
    previous_energy: float | None = None

    for k1 in k1_grid:
        for k2 in k2_grid:
            pair = SymbolPair(KH=KH, KS=KS, k1=float(k1), k2=float(k2), degree=1, sigma=1)
            problem = pair.form()

            if symmetrization == "direct":
                problem = problem.symmetrised()

            h_diag = DenseMatrixDiagnostics.from_dense_matrix(problem.Hk, name="H(k)")
            s_diag = DenseMatrixDiagnostics.from_dense_matrix(
                problem.Sk,
                name="S(k)",
                check_eigenvalues=True,
            )

            max_h_hermitian_defect = max(max_h_hermitian_defect, h_diag.hermitian_defect_rel)
            max_s_hermitian_defect = max(max_s_hermitian_defect, s_diag.hermitian_defect_rel)

            if s_diag.eig_min is not None:
                min_s_eig = min(min_s_eig, s_diag.eig_min)
            if s_diag.condition_number_abs is not None:
                max_s_cond = max(max_s_cond, s_diag.condition_number_abs)

            energy = float(problem.energies(symmetrise=(symmetrization != "raw"))[0])
            if previous_energy is not None:
                max_energy_jump = max(max_energy_jump, abs(energy - previous_energy))
            previous_energy = energy

    kh_star = KH.star_defect()
    ks_star = KS.star_defect()

    return {
        "n_u": int(n_u),
        "n_v": int(n_v),
        "sample_count": int(n_u) * int(n_v),
        "symmetrization": symmetrization,
        "h_star_defect_max": kh_star["star_defect_max"],
        "s_star_defect_max": ks_star["star_defect_max"],
        "h_hermitian_defect_rel_max": float(max_h_hermitian_defect),
        "s_hermitian_defect_rel_max": float(max_s_hermitian_defect),
        "s_eig_min": float(min_s_eig),
        "s_condition_number_abs_max": float(max_s_cond),
        "energy_neighbour_jump_max": float(max_energy_jump),
        "s_positive": bool(min_s_eig > 1.0e-10),
        "source": "controlled production GdKernelArrays toy",
    }


def periodic_two_level_dirac_hamiltonian(
    k1: float,
    k2: float,
    *,
    mass: float = 0.20,
) -> np.ndarray:
    """Periodic two-level Dirac-like toy Hamiltonian.

    H(k) = sin(k1) sigma_x + sin(k2) sigma_y
         + (mass + 2 - cos(k1) - cos(k2)) sigma_z

    This is periodic over the sampled k-domain and can be made close to a band
    crossing by taking a small positive mass.
    """

    dx = np.sin(k1)
    dy = np.sin(k2)
    dz = mass + 2.0 - np.cos(k1) - np.cos(k2)

    return np.asarray(
        [
            [dz, dx - 1j * dy],
            [dx + 1j * dy, -dz],
        ],
        dtype=np.complex128,
    )


def finite_field_band_crossing_hazard_probe(
    *,
    n_u: int = 11,
    n_v: int = 11,
    gap_threshold: float = 0.50,
    mass: float = 0.20,
) -> dict[str, float | int | bool | str]:
    """Map band-label hazards for a periodic two-level Dirac-like toy model."""

    if n_u < 1:
        raise ValueError(f"n_u must be positive, got {n_u}")
    if n_v < 1:
        raise ValueError(f"n_v must be positive, got {n_v}")
    if gap_threshold < 0.0:
        raise ValueError(f"gap_threshold must be non-negative, got {gap_threshold}")

    k1_grid = np.linspace(-np.pi, np.pi, int(n_u), endpoint=False)
    k2_grid = np.linspace(-np.pi, np.pi, int(n_v), endpoint=False)

    energies = np.empty((int(n_u), int(n_v), 2), dtype=np.float64)
    min_gap = np.inf
    min_gap_k1 = 0.0
    min_gap_k2 = 0.0
    hazard_count = 0

    for i, k1 in enumerate(k1_grid):
        for j, k2 in enumerate(k2_grid):
            vals = np.linalg.eigvalsh(periodic_two_level_dirac_hamiltonian(float(k1), float(k2), mass=mass))
            energies[i, j, :] = vals

            gap = float(vals[1] - vals[0])
            if gap < min_gap:
                min_gap = gap
                min_gap_k1 = float(k1)
                min_gap_k2 = float(k2)
            if gap < gap_threshold:
                hazard_count += 1

    max_band0_jump = 0.0
    max_band1_jump = 0.0
    max_gap_jump = 0.0

    for i in range(int(n_u)):
        for j in range(int(n_v)):
            neighbours = (
                ((i + 1) % int(n_u), j),
                (i, (j + 1) % int(n_v)),
            )
            for a, b in neighbours:
                max_band0_jump = max(max_band0_jump, abs(float(energies[a, b, 0] - energies[i, j, 0])))
                max_band1_jump = max(max_band1_jump, abs(float(energies[a, b, 1] - energies[i, j, 1])))

                gap_here = float(energies[i, j, 1] - energies[i, j, 0])
                gap_there = float(energies[a, b, 1] - energies[a, b, 0])
                max_gap_jump = max(max_gap_jump, abs(gap_there - gap_here))

    sample_count = int(n_u) * int(n_v)

    return {
        "source": "periodic two-level Dirac-like toy",
        "n_u": int(n_u),
        "n_v": int(n_v),
        "sample_count": sample_count,
        "mass": float(mass),
        "gap_threshold": float(gap_threshold),
        "min_gap": float(min_gap),
        "min_gap_k1": float(min_gap_k1),
        "min_gap_k2": float(min_gap_k2),
        "hazard_count": int(hazard_count),
        "hazard_fraction": float(hazard_count / sample_count),
        "has_hazard": bool(hazard_count > 0),
        "max_band0_neighbour_jump": float(max_band0_jump),
        "max_band1_neighbour_jump": float(max_band1_jump),
        "max_gap_neighbour_jump": float(max_gap_jump),
    }


def finite_difference_fixed_symbol_derivative(
    kernel: GdKernelArrays,
    k1: float,
    k2: float,
    *,
    sigma: int,
    axis: int,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Central finite-difference derivative of the fixed representation symbol."""

    if axis == 0:
        plus = kernel.symbol_fixed(k1 + eps, k2, sigma=sigma)
        minus = kernel.symbol_fixed(k1 - eps, k2, sigma=sigma)
    elif axis == 1:
        plus = kernel.symbol_fixed(k1, k2 + eps, sigma=sigma)
        minus = kernel.symbol_fixed(k1, k2 - eps, sigma=sigma)
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    return (plus - minus) / (2.0 * eps)


def finite_field_velocity_validation_probe() -> dict[str, float | str]:
    """Validate velocity ingredients on a controlled analytic production toy."""

    c0 = 1.25
    c1 = 0.70
    c2 = -0.30
    k1 = 0.37
    k2 = -0.44
    eps = 1.0e-6

    KH = gd_separable_cosine_kernel(c0=c0, c1=c1, c2=c2)
    KS = gd_identity_overlap_kernel()

    expected_dk1 = expected_separable_cosine_derivative(k1, k2, axis=0, c1=c1, c2=c2)
    expected_dk2 = expected_separable_cosine_derivative(k1, k2, axis=1, c1=c1, c2=c2)

    fixed_dk1 = gd_symbol_derivative_fixed(KH, k1, k2, sigma=1, axis=0)
    fixed_dk2 = gd_symbol_derivative_fixed(KH, k1, k2, sigma=1, axis=1)

    fd_dk1 = finite_difference_fixed_symbol_derivative(KH, k1, k2, sigma=1, axis=0, eps=eps)
    fd_dk2 = finite_difference_fixed_symbol_derivative(KH, k1, k2, sigma=1, axis=1, eps=eps)

    pair = SymbolPair(KH=KH, KS=KS, k1=k1, k2=k2, degree=1, sigma=1)
    problem = pair.form()
    energies, vectors = problem.eigensystem()
    dH = gd_symbol_derivatives(pair, KH)
    dS = gd_symbol_derivatives(pair, KS)

    u = vectors[:, 0]
    E = float(energies[0])
    hf_dk1 = float(np.real(np.vdot(u, (dH[0] - E * dS[0]) @ u)))
    hf_dk2 = float(np.real(np.vdot(u, (dH[1] - E * dS[1]) @ u)))

    generic_symbol_probe = gd_symbol_production_validation_probe()

    return {
        "source": "separable cosine production symbol toy",
        "k1": float(k1),
        "k2": float(k2),
        "finite_difference_eps": float(eps),
        "analytic_dk1": float(expected_dk1),
        "analytic_dk2": float(expected_dk2),
        "production_dk1_abs_error": float(abs(fixed_dk1[0, 0].real - expected_dk1)),
        "production_dk2_abs_error": float(abs(fixed_dk2[0, 0].real - expected_dk2)),
        "finite_difference_dk1_abs_error": float(abs(fd_dk1[0, 0].real - expected_dk1)),
        "finite_difference_dk2_abs_error": float(abs(fd_dk2[0, 0].real - expected_dk2)),
        "hellmann_feynman_dk1_abs_error": float(abs(hf_dk1 - expected_dk1)),
        "hellmann_feynman_dk2_abs_error": float(abs(hf_dk2 - expected_dk2)),
        "generic_fixed_symbol_abs_error": float(generic_symbol_probe["generic_symbol_channel_abs_error"]),
        "generic_fixed_dk1_abs_error": float(generic_symbol_probe["generic_dk1_channel_abs_error"]),
        "generic_fixed_dk2_abs_error": float(generic_symbol_probe["generic_dk2_channel_abs_error"]),
        "unit_scaling_status": "pending physical hbar/unit-context check",
        "vincent_velocity_status": "pending Vincent field comparison",
    }


def finite_field_unit_scaling_probe() -> dict[str, float | bool | str]:
    """Check core unit conversions used by finite-field validation.

    This is deliberately calculation-light. It validates the conversion factors
    that later velocity/conductivity comparisons depend on.
    """

    evag = LEGACY_EV_ANGSTROM_CONTEXT

    energy_disk_to_ev = ATOMIC_UNITS.energy.scale_to_si / evag.energy.scale_to_si
    length_disk_to_angstrom = ATOMIC_UNITS.length.scale_to_si / evag.length.scale_to_si

    hbar_au = ATOMIC_UNITS.hbar()
    hbar_evag = evag.hbar()

    # Existing Boltzmann tests scale the Hamiltonian energy and k-map together.
    # With that convention, the same physical velocity converts from AU-like
    # output to legacy eV/angstrom output by the length factor alone.
    velocity_au_to_evag = length_disk_to_angstrom
    expected_velocity_factor = 0.52917721092

    fermi_window_ev_from_au_factor = 1.0 / energy_disk_to_ev

    return {
        "source": "core UnitContext conversion factors",
        "atomic_energy_to_ev": float(energy_disk_to_ev),
        "atomic_length_to_angstrom": float(length_disk_to_angstrom),
        "hbar_atomic": float(hbar_au),
        "hbar_ev_angstrom": float(hbar_evag),
        "velocity_au_to_evag_factor": float(velocity_au_to_evag),
        "expected_velocity_au_to_evag_factor": float(expected_velocity_factor),
        "velocity_factor_abs_error": float(abs(velocity_au_to_evag - expected_velocity_factor)),
        "fermi_window_ev_from_au_factor": float(fermi_window_ev_from_au_factor),
        "mu_conversion_required": True,
        "conductivity_si_status": "pending full conductivity unit-conversion run",
    }


def finite_field_analytic_toy_coverage_probe() -> dict[str, float | int | bool | str]:
    """Summarise analytic toy coverage for the finite-field validation ladder."""

    symbol_probe = gd_symbol_production_validation_probe()
    input_health = finite_field_input_health_probe(n_u=5, n_v=7, symmetrization="star")
    band_hazards = finite_field_band_crossing_hazard_probe(
        n_u=10,
        n_v=10,
        gap_threshold=0.50,
        mass=0.20,
    )
    velocity = finite_field_velocity_validation_probe()
    units = finite_field_unit_scaling_probe()

    max_symbol_error = max(
        float(symbol_probe["fixed_symbol_abs_error"]),
        float(symbol_probe["generic_symbol_channel_abs_error"]),
        float(symbol_probe["energy_surface_max_abs_error"]),
    )
    max_derivative_error = max(
        float(symbol_probe["fixed_dk1_abs_error"]),
        float(symbol_probe["fixed_dk2_abs_error"]),
        float(symbol_probe["generic_dk1_channel_abs_error"]),
        float(symbol_probe["generic_dk2_channel_abs_error"]),
        float(symbol_probe["energy_surface_dk1_max_abs_error"]),
        float(symbol_probe["energy_surface_dk2_max_abs_error"]),
        float(velocity["finite_difference_dk1_abs_error"]),
        float(velocity["finite_difference_dk2_abs_error"]),
    )

    return {
        "source": "summary of controlled analytic probes",
        "toy_count": 4,
        "separable_cosine_symbol_max_error": float(max_symbol_error),
        "separable_cosine_derivative_max_error": float(max_derivative_error),
        "identity_overlap_min_eig": float(input_health["s_eig_min"]),
        "identity_overlap_condition": float(input_health["s_condition_number_abs_max"]),
        "periodic_dirac_min_gap": float(band_hazards["min_gap"]),
        "periodic_dirac_hazard_count": int(band_hazards["hazard_count"]),
        "velocity_hf_max_error": float(max(
            velocity["hellmann_feynman_dk1_abs_error"],
            velocity["hellmann_feynman_dk2_abs_error"],
        )),
        "unit_velocity_factor_error": float(units["velocity_factor_abs_error"]),
        "all_current_toys_pass": bool(
            max_symbol_error < 1.0e-12
            and max_derivative_error < 1.0e-9
            and float(input_health["s_eig_min"]) > 1.0e-10
            and int(band_hazards["hazard_count"]) >= 1
            and float(units["velocity_factor_abs_error"]) < 1.0e-12
        ),
        "missing_toy": "finite-field lattice-mode Gamma/F/rho closure toy",
    }


def finite_field_k_convergence_probe(
    grid_sizes: tuple[int, ...] = (5, 7, 11, 17, 23),
) -> dict[str, float | int | bool | str]:
    """Check k-grid convergence/measure on a periodic analytic velocity toy.

    For E(k) = c0 + c1 cos(k1) + c2 cos(k2), derivatives are
    dE/dk1 = -c1 sin(k1), dE/dk2 = -c2 sin(k2).

    The full-period average of sin(k)^2 is 1/2, so the exact reference for
    <|grad E|^2> is (c1^2 + c2^2) / 2.
    """

    if not grid_sizes:
        raise ValueError("grid_sizes must be non-empty")
    if any(n < 2 for n in grid_sizes):
        raise ValueError(f"all grid sizes must be >= 2, got {grid_sizes!r}")

    c1 = 0.70
    c2 = -0.30
    reference = 0.5 * (c1 * c1 + c2 * c2)

    previous_error: float | None = None
    finest_value = 0.0
    finest_error = 0.0
    max_error = 0.0
    rows_checked = 0
    improved_or_equal_steps = 0

    for n in grid_sizes:
        axis = np.linspace(-np.pi, np.pi, int(n), endpoint=False)
        total = 0.0

        for k1 in axis:
            for k2 in axis:
                dk1 = expected_separable_cosine_derivative(float(k1), float(k2), axis=0, c1=c1, c2=c2)
                dk2 = expected_separable_cosine_derivative(float(k1), float(k2), axis=1, c1=c1, c2=c2)
                total += dk1 * dk1 + dk2 * dk2

        value = total / float(n * n)
        error = abs(value - reference)

        if previous_error is not None and error <= previous_error + 1.0e-14:
            improved_or_equal_steps += 1

        previous_error = error
        finest_value = float(value)
        finest_error = float(error)
        max_error = max(max_error, float(error))
        rows_checked += 1

    return {
        "source": "periodic separable-cosine velocity-square average",
        "grid_count": int(rows_checked),
        "coarsest_n": int(grid_sizes[0]),
        "finest_n": int(grid_sizes[-1]),
        "reference_average_grad_e_sq": float(reference),
        "finest_average_grad_e_sq": float(finest_value),
        "finest_abs_error": float(finest_error),
        "max_abs_error": float(max_error),
        "improved_or_equal_steps": int(improved_or_equal_steps),
        "all_grid_errors_small": bool(max_error < 1.0e-12),
        "measure_status": "uniform full-period average matches analytic reference",
        "conductivity_convergence_status": "pending dataset-backed conductivity refinement",
    }


def finite_field_symmetry_sanity_probe(
    *,
    n: int = 17,
) -> dict[str, float | int | bool | str]:
    """Check symmetry identities on the separable periodic cosine toy."""

    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")

    c0 = 1.25
    c1 = 0.70
    c2 = -0.30

    axis = np.linspace(-np.pi, np.pi, int(n), endpoint=False)

    max_energy_inversion_error = 0.0
    max_dk1_odd_error = 0.0
    max_dk2_odd_error = 0.0

    tensor = np.zeros((2, 2), dtype=np.float64)

    for k1 in axis:
        for k2 in axis:
            k1 = float(k1)
            k2 = float(k2)

            e = expected_separable_cosine_symbol(k1, k2, c0=c0, c1=c1, c2=c2)
            e_inv = expected_separable_cosine_symbol(-k1, -k2, c0=c0, c1=c1, c2=c2)
            max_energy_inversion_error = max(max_energy_inversion_error, abs(e - e_inv))

            dk1 = expected_separable_cosine_derivative(k1, k2, axis=0, c1=c1, c2=c2)
            dk2 = expected_separable_cosine_derivative(k1, k2, axis=1, c1=c1, c2=c2)
            dk1_inv = expected_separable_cosine_derivative(-k1, -k2, axis=0, c1=c1, c2=c2)
            dk2_inv = expected_separable_cosine_derivative(-k1, -k2, axis=1, c1=c1, c2=c2)

            max_dk1_odd_error = max(max_dk1_odd_error, abs(dk1 + dk1_inv))
            max_dk2_odd_error = max(max_dk2_odd_error, abs(dk2 + dk2_inv))

            v = np.asarray([dk1, dk2], dtype=np.float64)
            tensor += np.outer(v, v)

    tensor /= float(n * n)

    xy_abs = abs(float(tensor[0, 1]))
    yx_abs = abs(float(tensor[1, 0]))
    antisym_abs = abs(float(tensor[0, 1] - tensor[1, 0]))

    return {
        "source": "separable cosine inversion and tensor symmetry toy",
        "n": int(n),
        "sample_count": int(n * n),
        "energy_inversion_max_error": float(max_energy_inversion_error),
        "dk1_odd_max_error": float(max_dk1_odd_error),
        "dk2_odd_max_error": float(max_dk2_odd_error),
        "tensor_xx": float(tensor[0, 0]),
        "tensor_yy": float(tensor[1, 1]),
        "tensor_xy": float(tensor[0, 1]),
        "tensor_yx": float(tensor[1, 0]),
        "tensor_xy_abs": float(xy_abs),
        "tensor_yx_abs": float(yx_abs),
        "tensor_antisym_abs": float(antisym_abs),
        "all_symmetry_checks_pass": bool(
            max_energy_inversion_error < 1.0e-12
            and max_dk1_odd_error < 1.0e-12
            and max_dk2_odd_error < 1.0e-12
            and xy_abs < 1.0e-12
            and yx_abs < 1.0e-12
            and antisym_abs < 1.0e-12
        ),
        "dataset_automorphism_status": "pending H/S/H_star/S_star automorphism checks",
    }


def finite_field_vincent_reconstruction_probe() -> dict[str, float | bool | str]:
    """Summarise existing Vincent/Ashcroft reconstruction checks."""

    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        band_indexed_strong_dc_from_velocity_grid,
        conductivity_830_shifted_chain_rule_from_velocity_grid,
        conductivity_from_epsilon_grid,
        load_vincent_input_data,
        vincent_delaunay_adjacent_simplex_velocity_probe,
        vincent_reference,
    )

    reference = vincent_reference()
    inputs = load_vincent_input_data()
    ai = inputs.primitive_lattice_vectors_bohr
    epsilon = inputs.epsilon_of_k
    sigma_target = reference.expected_conductivity_S_per_m

    local = conductivity_from_epsilon_grid(
        epsilon,
        ai,
        chemical_potential_J=float(np.mean(epsilon) * ATOMIC_UNITS.energy.scale_to_si),
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
    )
    weak_sigma = local.conductivity_tensor_S * ((2.0 * np.pi) ** 2)
    weak_trace = float(np.trace(weak_sigma))
    target_trace = float(np.trace(sigma_target))
    weak_trace_percent_error = 100.0 * (weak_trace - target_trace) / target_trace

    strong = band_indexed_strong_dc_from_velocity_grid(
        epsilon,
        local.velocity_m_per_s,
        ai,
        chemical_potential_J=local.chemical_potential_J,
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
        electric_field_V_per_m=np.zeros(2),
    )
    strong_grid_sigma = strong.conductivity_tensor_S.real
    strong_grid_trace = float(np.trace(strong_grid_sigma))
    strong_grid_trace_percent_error = 100.0 * (strong_grid_trace - target_trace) / target_trace

    shifted = conductivity_830_shifted_chain_rule_from_velocity_grid(
        epsilon,
        local.velocity_m_per_s,
        ai,
        chemical_potential_J=local.chemical_potential_J,
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
        electric_field_V_per_m=reference.electric_field_V_per_m,
    )
    shifted_sigma = shifted.conductivity_tensor_S * ((2.0 * np.pi) ** 2)
    shifted_trace = float(np.trace(shifted_sigma))
    shifted_trace_percent_error = 100.0 * (shifted_trace - target_trace) / target_trace

    adjacent = vincent_delaunay_adjacent_simplex_velocity_probe(epsilon, ai)
    find_simplex_max_error = max(float(row["find_simplex_error"]) for row in adjacent)
    best_adjacent_max_error = max(float(row["best_adjacent_error"]) for row in adjacent)
    velocity_error_reduction = find_simplex_max_error / best_adjacent_max_error

    return {
        "source": "existing Ashcroft/Vincent comparison domain",
        "target_trace_S_per_m": float(target_trace),
        "weak_chain_trace_S_per_m": float(weak_trace),
        "weak_chain_trace_percent_error": float(weak_trace_percent_error),
        "strong_grid_trace_S_per_m": float(strong_grid_trace),
        "strong_grid_trace_percent_error": float(strong_grid_trace_percent_error),
        "shifted_830_trace_S_per_m": float(shifted_trace),
        "shifted_830_trace_percent_error": float(shifted_trace_percent_error),
        "find_simplex_max_velocity_error_m_per_s": float(find_simplex_max_error),
        "best_adjacent_max_velocity_error_m_per_s": float(best_adjacent_max_error),
        "velocity_error_reduction": float(velocity_error_reduction),
        "best_adjacent_matches_vincent": bool(best_adjacent_max_error < 1.0e-3),
        "residual_status": "velocity samples resolved; conductivity residual remains formula/convention audit",
    }


def finite_field_strong_dc_validation_probe() -> dict[str, float | int | bool | str]:
    """Validate the band-indexed strong spectral DC tensor on Vincent inputs."""

    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        band_indexed_strong_dc_from_velocity_grid,
        conductivity_from_epsilon_grid,
        load_vincent_input_data,
        vincent_reference,
    )

    reference = vincent_reference()
    inputs = load_vincent_input_data()
    ai = inputs.primitive_lattice_vectors_bohr
    epsilon = inputs.epsilon_of_k

    local = conductivity_from_epsilon_grid(
        epsilon,
        ai,
        chemical_potential_J=float(np.mean(epsilon) * ATOMIC_UNITS.energy.scale_to_si),
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
    )

    weak_grid_sigma = local.conductivity_tensor_S * ((2.0 * np.pi) ** 2)

    strong = band_indexed_strong_dc_from_velocity_grid(
        epsilon,
        local.velocity_m_per_s,
        ai,
        chemical_potential_J=local.chemical_potential_J,
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
        electric_field_V_per_m=np.zeros(2),
    )

    strong_grid_sigma = strong.conductivity_tensor_S.real
    strong_from_modes = np.sum(strong.conductivity_mode_tensor_S, axis=(0, 1))
    mode_reconstruction_abs_error = float(
        np.max(np.abs(strong_from_modes - strong.conductivity_tensor_S))
    )

    strong_trace = float(np.trace(strong_grid_sigma))
    weak_trace = float(np.trace(weak_grid_sigma))
    target_trace = float(np.trace(reference.expected_conductivity_S_per_m))

    strong_vs_weak_rel_trace_gap = (strong_trace - weak_trace) / weak_trace
    strong_vs_vincent_percent_error = 100.0 * (strong_trace - target_trace) / target_trace

    conductivity_norm = float(np.linalg.norm(strong.conductivity_tensor_S))
    imaginary_leakage_ratio = float(
        strong.imaginary_leakage_S / conductivity_norm
        if conductivity_norm > 0.0
        else np.nan
    )

    mode_abs = np.linalg.norm(strong.conductivity_mode_tensor_S.reshape((-1, 2, 2)), axis=(1, 2))
    total_mode_abs = float(np.sum(mode_abs))
    strongest_mode_fraction = float(np.max(mode_abs) / total_mode_abs) if total_mode_abs > 0.0 else np.nan

    nonzero_mode_count = int(np.count_nonzero(mode_abs > 1.0e-18))
    mode_count = int(mode_abs.size)

    return {
        "source": "BandIndexedStrongDcResult on Vincent epsilon grid",
        "mode_count": mode_count,
        "nonzero_mode_count": nonzero_mode_count,
        "strong_grid_trace_S_per_m": float(strong_trace),
        "weak_chain_grid_trace_S_per_m": float(weak_trace),
        "vincent_target_trace_S_per_m": float(target_trace),
        "strong_vs_weak_rel_trace_gap": float(strong_vs_weak_rel_trace_gap),
        "strong_vs_vincent_percent_error": float(strong_vs_vincent_percent_error),
        "mode_reconstruction_abs_error": float(mode_reconstruction_abs_error),
        "imaginary_leakage_S": float(strong.imaginary_leakage_S),
        "imaginary_leakage_ratio": float(imaginary_leakage_ratio),
        "strongest_mode_fraction": float(strongest_mode_fraction),
        "occupation_coeff_shape_0": int(strong.occupation_coefficients.shape[0]),
        "occupation_coeff_shape_1": int(strong.occupation_coefficients.shape[1]),
        "response_factor_finite": bool(np.isfinite(strong.response_factor).all()),
        "velocity_coefficients_finite": bool(np.isfinite(strong.velocity_coefficients_m_per_s_per_m2).all()),
        "strong_dc_internal_pass": bool(
            mode_reconstruction_abs_error < 1.0e-18
            and imaginary_leakage_ratio < 1.0e-12
            and np.isfinite(strongest_mode_fraction)
            and np.isfinite(strong_vs_weak_rel_trace_gap)
        ),
        "residual_status": "strong spectral tensor is internally closed; weak-chain gap is derivative-definition residual",
    }


def finite_field_weak_dc_limit_probe() -> dict[str, float | int | bool | str]:
    """Check strong finite-field DC approaches weak DC in a matched spectral basis."""

    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        analytic_sinusoidal_conductivity_probe,
    )

    probe = analytic_sinusoidal_conductivity_probe()
    field_rows = list(probe["strong_weak_field_rows"])

    zero_row = field_rows[0]
    small_nonzero_row = field_rows[1]
    largest_row = field_rows[-1]

    min_nonzero_eta = min(float(row["eta"]) for row in field_rows if float(row["eta"]) > 0.0)
    max_eta = max(float(row["eta"]) for row in field_rows)

    max_field_tensor_discrepancy = max(float(row["relative_tensor_discrepancy"]) for row in field_rows)
    max_field_trace_discrepancy = max(abs(float(row["relative_trace_discrepancy"])) for row in field_rows)
    max_imaginary_leakage = max(float(row["imaginary_leakage"]) for row in field_rows)

    return {
        "source": "analytic sinusoidal Ashcroft strong/weak sweep",
        "field_row_count": int(len(field_rows)),
        "zero_eta": float(zero_row["eta"]),
        "zero_field_V_per_m": float(zero_row["field_V_per_m"]),
        "zero_relative_tensor_discrepancy": float(zero_row["relative_tensor_discrepancy"]),
        "zero_relative_trace_discrepancy": float(zero_row["relative_trace_discrepancy"]),
        "small_eta": float(small_nonzero_row["eta"]),
        "small_field_V_per_m": float(small_nonzero_row["field_V_per_m"]),
        "small_relative_tensor_discrepancy": float(small_nonzero_row["relative_tensor_discrepancy"]),
        "small_relative_trace_discrepancy": float(small_nonzero_row["relative_trace_discrepancy"]),
        "largest_eta": float(largest_row["eta"]),
        "largest_relative_tensor_discrepancy": float(largest_row["relative_tensor_discrepancy"]),
        "largest_relative_trace_discrepancy": float(largest_row["relative_trace_discrepancy"]),
        "min_nonzero_eta": float(min_nonzero_eta),
        "max_eta": float(max_eta),
        "max_field_tensor_discrepancy": float(max_field_tensor_discrepancy),
        "max_abs_field_trace_discrepancy": float(max_field_trace_discrepancy),
        "max_imaginary_leakage": float(max_imaginary_leakage),
        "relative_weak_limit_error": float(probe["relative_weak_limit_error"]),
        "strong_zero_field_imaginary_leakage": float(probe["strong_zero_field_imaginary_leakage"]),
        "weak_limit_pass": bool(
            float(probe["relative_weak_limit_error"]) < 1.0e-12
            and float(zero_row["relative_tensor_discrepancy"]) < 1.0e-12
            and abs(float(zero_row["relative_trace_discrepancy"])) < 1.0e-12
            and np.isfinite(max_field_tensor_discrepancy)
        ),
        "roundoff_floor_status": "zero-field agreement checked; finite eta sweep exposes nonlinear departure",
    }

def gd_symbol_production_validation_probe() -> dict[str, float]:
    """Validate the production GdKernelArrays symbol and derivative mechanisms."""

    c0 = 1.25
    c1 = 0.70
    c2 = -0.30
    k1 = 0.37
    k2 = -0.44

    KH = gd_separable_cosine_kernel(c0=c0, c1=c1, c2=c2)
    KS = gd_identity_overlap_kernel()

    expected = expected_separable_cosine_symbol(k1, k2, c0=c0, c1=c1, c2=c2)
    expected_dk1 = expected_separable_cosine_derivative(k1, k2, axis=0, c1=c1, c2=c2)
    expected_dk2 = expected_separable_cosine_derivative(k1, k2, axis=1, c1=c1, c2=c2)

    fixed_symbol = KH.symbol_fixed(k1, k2, sigma=1)
    generic_symbol = KH.symbol_generic(k1, k2)

    fixed_dk1 = gd_symbol_derivative_fixed(KH, k1, k2, sigma=1, axis=0)
    fixed_dk2 = gd_symbol_derivative_fixed(KH, k1, k2, sigma=1, axis=1)

    generic_dk1 = gd_symbol_derivative_generic(KH, k1, k2, axis=0)
    generic_dk2 = gd_symbol_derivative_generic(KH, k1, k2, axis=1)

    pair = SymbolPair(KH=KH, KS=KS, k1=k1, k2=k2, degree=1, sigma=1)
    problem = pair.form()
    dH_pair = gd_symbol_derivatives(pair, KH)

    fixed_energy = float(eigh(problem.Hk, problem.Sk, eigvals_only=True)[0])

    dE_dk1 = float(np.real(fixed_dk1[0, 0]))
    dE_dk2 = float(np.real(fixed_dk2[0, 0]))

    # A grid of energy values produced by the production symbol should match
    # the analytic energy surface c0 + c1 cos(k1) + c2 cos(k2).
    n1 = 23
    n2 = 29
    k1_grid = np.linspace(-np.pi, np.pi, n1, endpoint=False)
    k2_grid = np.linspace(-np.pi, np.pi, n2, endpoint=False)

    max_surface_error = 0.0
    max_surface_dk1_error = 0.0
    max_surface_dk2_error = 0.0

    for a in k1_grid:
        for b in k2_grid:
            surface_value = float(KH.symbol_fixed(a, b, sigma=1)[0, 0].real)
            surface_expected = expected_separable_cosine_symbol(a, b, c0=c0, c1=c1, c2=c2)

            dk1_value = float(gd_symbol_derivative_fixed(KH, a, b, sigma=1, axis=0)[0, 0].real)
            dk2_value = float(gd_symbol_derivative_fixed(KH, a, b, sigma=1, axis=1)[0, 0].real)

            dk1_expected = expected_separable_cosine_derivative(a, b, axis=0, c1=c1, c2=c2)
            dk2_expected = expected_separable_cosine_derivative(a, b, axis=1, c1=c1, c2=c2)

            max_surface_error = max(max_surface_error, abs(surface_value - surface_expected))
            max_surface_dk1_error = max(max_surface_dk1_error, abs(dk1_value - dk1_expected))
            max_surface_dk2_error = max(max_surface_dk2_error, abs(dk2_value - dk2_expected))

    generic_eigs = generic_symbol_scalar_channels(generic_symbol)
    generic_dk1_eigs = generic_symbol_scalar_channels(generic_dk1)
    generic_dk2_eigs = generic_symbol_scalar_channels(generic_dk2)

    return {
        "fixed_symbol_abs_error": float(abs(fixed_symbol[0, 0].real - expected)),
        "generic_symbol_channel_abs_error": float(np.max(np.abs(generic_eigs - expected))),
        "fixed_dk1_abs_error": float(abs(fixed_dk1[0, 0].real - expected_dk1)),
        "fixed_dk2_abs_error": float(abs(fixed_dk2[0, 0].real - expected_dk2)),
        "generic_dk1_channel_abs_error": float(np.max(np.abs(generic_dk1_eigs - expected_dk1))),
        "generic_dk2_channel_abs_error": float(np.max(np.abs(generic_dk2_eigs - expected_dk2))),
        "symbol_pair_energy_abs_error": float(abs(fixed_energy - expected)),
        "symbol_pair_dk1_abs_error": float(abs(dH_pair[0][0, 0].real - expected_dk1)),
        "symbol_pair_dk2_abs_error": float(abs(dH_pair[1][0, 0].real - expected_dk2)),
        "energy_surface_max_abs_error": float(max_surface_error),
        "energy_surface_dk1_max_abs_error": float(max_surface_dk1_error),
        "energy_surface_dk2_max_abs_error": float(max_surface_dk2_error),
        "hellmann_feynman_dk1_abs_error": float(abs(dE_dk1 - expected_dk1)),
        "hellmann_feynman_dk2_abs_error": float(abs(dE_dk2 - expected_dk2)),
    }
