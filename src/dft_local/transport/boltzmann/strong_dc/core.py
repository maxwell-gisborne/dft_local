"""Shared spectral strong-DC Boltzmann conductivity formulas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np

from numpy.typing import NDArray

from dft_local.core.units import ATOMIC_UNITS, CONDUCTIVITY, KSPACE_AREA, LENGTH, VELOCITY, qarray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

ConductivityTensor = Annotated[np.ndarray, qarray(CONDUCTIVITY, ("cartesian", "cartesian"), role="conductivity tensor")]
VelocityGrid = Annotated[np.ndarray, qarray(VELOCITY, ("k1", "k2", "cartesian"), role="velocity grid")]
KResolvedConductivityTensor = Annotated[
    np.ndarray,
    qarray(
        CONDUCTIVITY * KSPACE_AREA.inverse(),
        ("k1", "k2", "cartesian", "cartesian"),
        role="k-resolved conductivity tensor",
    ),
]

ELECTRON_CHARGE_C = 1.602176634e-19
HBAR_J_S = 1.054571817e-34
KB_J_K = 1.380649e-23
HARTREE_TO_J = ATOMIC_UNITS.energy.scale_to_si
BOHR_TO_M = ATOMIC_UNITS.length.scale_to_si


def reciprocal_lattice_vectors_from_primitives(ai_bohr: np.ndarray) -> np.ndarray:
    """Return reciprocal basis rows b_i satisfying a_i . b_j = 2 pi delta_ij."""

    ai_bohr = np.asarray(ai_bohr, dtype=float)

    if ai_bohr.shape != (2, 2):
        raise ValueError(f"Expected primitive lattice shape (2, 2), got {ai_bohr.shape}")

    return 2.0 * np.pi * np.linalg.inv(ai_bohr).T


def fermi_factor(epsilon_J: np.ndarray, chemical_potential_J: float, temperature_K: float) -> np.ndarray:
    epsilon_J = np.asarray(epsilon_J, dtype=np.float64)
    beta_arg = (epsilon_J - chemical_potential_J) / (KB_J_K * temperature_K)

    # Stable logistic. Large positive => f ~ 0, large negative => f ~ 1.
    out = np.empty_like(beta_arg, dtype=np.float64)
    positive = beta_arg >= 0.0

    exp_neg = np.exp(-beta_arg[positive])
    out[positive] = exp_neg / (1.0 + exp_neg)

    exp_pos = np.exp(beta_arg[~positive])
    out[~positive] = 1.0 / (1.0 + exp_pos)

    return out


@dataclass(frozen=True, slots=True)
class BandIndexedStrongDcResult:
    chemical_potential_J: float
    temperature_K: float
    relaxation_time_s: float
    area_bz_per_m2: float
    electric_field_V_per_m: np.ndarray
    mode_indices: np.ndarray
    lattice_vectors_m: np.ndarray
    occupation: np.ndarray
    velocity_m_per_s: VelocityGrid
    occupation_coefficients: np.ndarray
    velocity_coefficients_m_per_s_per_m2: np.ndarray
    response_factor: np.ndarray
    conductivity_mode_tensor_S: KResolvedConductivityTensor
    conductivity_tensor_S: ConductivityTensor
    imaginary_leakage_S: float


def lattice_mode_indices(shape: tuple[int, int]) -> np.ndarray:
    """Return integer FFT mode indices ``(a,b)`` in FFT mode order."""

    n1, n2 = shape
    a_modes = (np.fft.fftfreq(n1) * float(n1)).astype(int)
    b_modes = (np.fft.fftfreq(n2) * float(n2)).astype(int)
    aa, bb = np.meshgrid(a_modes, b_modes, indexing="ij")
    return np.stack((aa, bb), axis=-1)


def lattice_mode_vectors_m(primitive_lattice_vectors_bohr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return real-space lattice vectors R_(a,b) in FFT mode order."""

    ai_m = np.asarray(primitive_lattice_vectors_bohr, dtype=float) * BOHR_TO_M
    if ai_m.shape != (2, 2):
        raise ValueError(f"Expected primitive lattice shape (2, 2), got {ai_m.shape}")

    mode_indices = lattice_mode_indices(shape)
    return mode_indices[..., 0, None] * ai_m[0] + mode_indices[..., 1, None] * ai_m[1]


def band_indexed_strong_dc_from_velocity_grid(
    epsilon_Ha: np.ndarray,
    velocity_m_per_s: np.ndarray,
    primitive_lattice_vectors_bohr: np.ndarray,
    *,
    chemical_potential_J: float,
    temperature_K: float,
    relaxation_time_s: float,
    electric_field_V_per_m: np.ndarray | None = None,
    spin_degeneracy: float = 1.0,
) -> BandIndexedStrongDcResult:
    """Compute the band-indexed steady strong-DC lattice-mode formula.

    This implements the thesis Section 7.5 formula in the single-band grid
    convention used by the Ashcroft comparison domain.  The lattice index
    ``(a,b)`` is the FFT/Fourier mode index, not an atom-pair index.

    At zero applied field this should reconstruct the weak DC formula, up to
    the same reciprocal-space normalisation convention used by the comparison.
    """

    epsilon_J = np.asarray(epsilon_Ha, dtype=np.float64) * HARTREE_TO_J
    velocity = np.asarray(velocity_m_per_s, dtype=np.float64)

    if epsilon_J.ndim != 2:
        raise ValueError(f"Expected a 2D epsilon grid, got shape {epsilon_J.shape}")

    if velocity.shape != epsilon_J.shape + (2,):
        raise ValueError(
            f"velocity shape {velocity.shape} does not match epsilon shape {epsilon_J.shape} + (2,)"
        )

    field = (
        np.zeros(2, dtype=np.float64)
        if electric_field_V_per_m is None
        else np.asarray(electric_field_V_per_m, dtype=np.float64)
    )
    if field.shape != (2,):
        raise ValueError(f"Expected electric field shape (2,), got {field.shape}")

    reciprocal_bohr = reciprocal_lattice_vectors_from_primitives(primitive_lattice_vectors_bohr)
    reciprocal_per_m = reciprocal_bohr / BOHR_TO_M
    area_bz = abs(float(np.linalg.det(reciprocal_per_m)))

    occupation = fermi_factor(epsilon_J, chemical_potential_J, temperature_K)

    # Thesis convention:
    #   f0(k) = sum_R f0_tilde[R] exp(+i R.k)
    # Therefore f0_tilde is the positive-phase Fourier coefficient.
    occupation_coeff = np.fft.fft2(occupation) / float(occupation.size)

    # Thesis convention:
    #   Gamma_alpha[R] = (1 / A_BZ) int exp(+i R.k) v_alpha(k) dk
    # and
    #   v_alpha(k) = A_BZ sum_R Gamma_alpha[R] exp(-i R.k)
    #
    # The discrete average is the numerical analogue of (1 / A_BZ) int dk.
    # Do not divide by A_BZ again.
    velocity_coeff = np.empty(velocity.shape, dtype=np.complex128)
    for alpha in range(2):
        velocity_coeff[..., alpha] = np.fft.ifft2(velocity[..., alpha])

    r_vectors = lattice_mode_vectors_m(primitive_lattice_vectors_bohr, epsilon_J.shape)
    field_dot_r = np.einsum("...a,a->...", r_vectors, field)

    scale = ELECTRON_CHARGE_C * relaxation_time_s / HBAR_J_S
    denominator = 1.0 - 1j * scale * field_dot_r

    response = np.empty(epsilon_J.shape + (2,), dtype=np.complex128)
    for beta in range(2):
        response[..., beta] = -1j * scale * r_vectors[..., beta] / (denominator * denominator)

    sigma_modes = np.empty(epsilon_J.shape + (2, 2), dtype=np.complex128)
    mode_prefactor = spin_degeneracy * ELECTRON_CHARGE_C * area_bz
    for alpha in range(2):
        for beta in range(2):
            sigma_modes[..., alpha, beta] = (
                mode_prefactor
                * occupation_coeff
                * velocity_coeff[..., alpha]
                * response[..., beta]
            )

    sigma = np.sum(sigma_modes, axis=(0, 1))

    return BandIndexedStrongDcResult(
        chemical_potential_J=float(chemical_potential_J),
        temperature_K=float(temperature_K),
        relaxation_time_s=float(relaxation_time_s),
        area_bz_per_m2=float(area_bz),
        electric_field_V_per_m=field,
        mode_indices=lattice_mode_indices(epsilon_J.shape),
        lattice_vectors_m=r_vectors,
        occupation=occupation,
        velocity_m_per_s=velocity,
        occupation_coefficients=occupation_coeff,
        velocity_coefficients_m_per_s_per_m2=velocity_coeff,
        response_factor=response,
        conductivity_mode_tensor_S=sigma_modes,
        conductivity_tensor_S=sigma,
        imaginary_leakage_S=float(np.linalg.norm(sigma.imag)),
    )
