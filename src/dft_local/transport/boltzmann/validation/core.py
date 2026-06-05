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

from dft_local.core.kernels import GdKernelArrays
from dft_local.core.local_problem import SymbolPair
from dft_local.core.numerics import hermitian_part
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
