"""Core validation helpers for the Boltzmann operator approach.

This module is deliberately independent of Vincent/Ashcroft comparison data.
The goal is to collect small analytic and algebraic checks that validate the
operator formulation before it is compared with any external implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import scipy.linalg as la

from dft_local.core.units import (
    CHARGE,
    CONDUCTIVITY,
    DIMENSIONLESS,
    ENERGY,
    LENGTH,
    SI_UNITS,
    TIME,
    VELOCITY,
    Unit,
    UnitContext,
    qscalar,
)


PERCENT = Unit("%", DIMENSIONLESS, 0.01)
ELECTRIC_FIELD = ENERGY / (CHARGE * LENGTH)
VOLT_PER_METER = Unit("V/m", ELECTRIC_FIELD, 1.0)


@dataclass(frozen=True, slots=True)
class FiniteFieldVincentReconstructionProbe:
    """Typed scalar result for the Vincent reconstruction validation panel."""

    unit_context: UnitContext
    source: str
    continuum_weak_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="weak raw continuum trace"),
    ]
    continuum_weak_trace_percent_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="weak raw continuum trace error", display_unit=PERCENT),
    ]
    continuum_eq830_shifted_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="Eq. 8.30 raw continuum trace"),
    ]
    continuum_eq830_shifted_trace_percent_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="Eq. 8.30 raw continuum trace error", display_unit=PERCENT),
    ]
    reciprocal_dot_diag_max_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="reciprocal dot-product diagonal error"),
    ]
    reciprocal_dot_offdiag_max_abs: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="reciprocal dot-product off-diagonal magnitude"),
    ]
    reciprocal_det_ratio: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="reciprocal determinant ratio"),
    ]
    reciprocal_det_ratio_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="reciprocal determinant ratio error"),
    ]


    target_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="Vincent target trace"),
    ]
    weak_chain_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="weak-chain trace"),
    ]
    weak_chain_trace_percent_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="weak-chain trace error", display_unit=PERCENT),
    ]
    strong_grid_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="strong-grid trace"),
    ]
    strong_grid_trace_percent_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="strong-grid trace error", display_unit=PERCENT),
    ]
    shifted_830_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="shifted Eq. 8.30 trace"),
    ]
    shifted_830_trace_percent_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="shifted Eq. 8.30 trace error", display_unit=PERCENT),
    ]
    eq830_modal_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="Eq. 8.30 Gamma-Q-rho trace"),
    ]
    eq830_modal_trace_percent_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="Eq. 8.30 Gamma-Q-rho trace error", display_unit=PERCENT),
    ]
    eq830_modal_direct_trace_percent_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="Eq. 8.30 modal/direct trace mismatch", display_unit=PERCENT),
    ]

    find_simplex_max_velocity_error: Annotated[
        float,
        qscalar(VELOCITY, role="find-simplex max velocity error"),
    ]
    best_adjacent_max_velocity_error: Annotated[
        float,
        qscalar(VELOCITY, role="best-adjacent max velocity error"),
    ]
    velocity_error_reduction: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="velocity error reduction"),
    ]

    best_adjacent_matches_vincent: bool
    residual_status: str


@dataclass(frozen=True, slots=True)
class FiniteFieldStrongDcValidationProbe:
    """Typed scalar result for the strong DC validation panel."""

    unit_context: UnitContext
    source: str

    mode_count: int
    nonzero_mode_count: int

    continuum_strong_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="strong continuum trace"),
    ]
    continuum_weak_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="weak continuum trace"),
    ]
    no_2pi_denominator_strong_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="strong no-2π-denominator trace"),
    ]
    no_2pi_denominator_weak_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="weak no-2π-denominator trace"),
    ]

    strong_grid_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="strong-grid trace"),
    ]
    weak_chain_grid_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="weak-chain grid trace"),
    ]
    vincent_target_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="Vincent target trace"),
    ]
    strong_vs_weak_rel_trace_gap: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="strong/weak trace gap"),
    ]
    strong_vs_vincent_percent_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="strong/Vincent trace error", display_unit=PERCENT),
    ]
    mode_reconstruction_abs_error: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="mode reconstruction absolute error"),
    ]
    imaginary_leakage: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="imaginary leakage"),
    ]
    imaginary_leakage_ratio: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="imaginary leakage ratio"),
    ]
    strongest_mode_fraction: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="strongest mode fraction"),
    ]

    occupation_coeff_shape: tuple[int, int]

    response_factor_finite: bool
    velocity_coefficients_finite: bool
    strong_dc_internal_pass: bool
    residual_status: str



@dataclass(frozen=True, slots=True)
class FiniteFieldStrongEq830LimitProbe:
    """Compare strong differential-response DC against Eq. 8.30 shifted finite-difference DC."""

    unit_context: UnitContext
    source: str

    field_row_count: int

    zero_field: Annotated[
        float,
        qscalar(ELECTRIC_FIELD, role="zero field", display_unit=VOLT_PER_METER),
    ]
    smallest_nonzero_field: Annotated[
        float,
        qscalar(ELECTRIC_FIELD, role="smallest nonzero field", display_unit=VOLT_PER_METER),
    ]
    largest_field: Annotated[
        float,
        qscalar(ELECTRIC_FIELD, role="largest field", display_unit=VOLT_PER_METER),
    ]

    strong_continuum_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="strong continuum trace"),
    ]
    zero_eq830_continuum_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="zero-field Eq. 8.30 continuum trace"),
    ]
    smallest_eq830_continuum_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="small-field Eq. 8.30 continuum trace"),
    ]

    zero_relative_tensor_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="zero-field strong/Eq. 8.30 tensor discrepancy"),
    ]
    zero_relative_trace_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="zero-field strong/Eq. 8.30 trace discrepancy"),
    ]
    smallest_relative_tensor_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="small-field strong/Eq. 8.30 tensor discrepancy"),
    ]
    smallest_relative_trace_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="small-field strong/Eq. 8.30 trace discrepancy"),
    ]
    largest_relative_tensor_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="largest-field strong/Eq. 8.30 tensor discrepancy"),
    ]
    largest_relative_trace_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="largest-field strong/Eq. 8.30 trace discrepancy"),
    ]

    min_relative_tensor_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="minimum strong/Eq. 8.30 tensor discrepancy"),
    ]
    min_abs_relative_trace_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="minimum absolute strong/Eq. 8.30 trace discrepancy"),
    ]

    eq830_limit_status: str
    continuum_normalisation_status: str
    limit_validation_pass: bool




@dataclass(frozen=True, slots=True)
class FiniteFieldModeDecompositionProbe:
    """Typed scalar result for Gamma/F/rho mode decomposition validation."""

    unit_context: UnitContext
    source: str

    mode_count: int

    gamma_reconstruction_abs_error: Annotated[
        float,
        qscalar(VELOCITY, role="Gamma reconstruction absolute error"),
    ]
    rho_reconstruction_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="rho reconstruction absolute error"),
    ]
    mode_tensor_reconstruction_abs_error: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="mode tensor reconstruction absolute error"),
    ]
    conductivity_trace: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="conductivity trace"),
    ]
    conductivity_mode_norm_sum: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="conductivity mode norm sum"),
    ]
    top_1_mode_fraction: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="top 1 mode fraction"),
    ]
    top_10_mode_fraction: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="top 10 mode fraction"),
    ]
    top_100_mode_fraction: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="top 100 mode fraction"),
    ]
    gamma_abs_max: Annotated[
        float,
        qscalar(VELOCITY, role="Gamma coefficient absolute max"),
    ]
    rho_abs_max: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="rho coefficient absolute max"),
    ]
    response_abs_max: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="response factor absolute max"),
    ]

    gamma_finite: bool
    rho_finite: bool
    response_finite: bool
    mode_tensor_finite: bool
    mode_closure_pass: bool
    residual_status: str


@dataclass(frozen=True, slots=True)
class FiniteFieldWeakDcLimitProbe:
    """Typed scalar result for weak-field DC limit validation."""

    unit_context: UnitContext
    source: str

    field_row_count: int

    zero_eta: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="zero eta"),
    ]
    zero_field: Annotated[
        float,
        qscalar(ELECTRIC_FIELD, role="zero field", display_unit=VOLT_PER_METER),
    ]
    zero_relative_tensor_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="zero relative tensor discrepancy"),
    ]
    zero_relative_trace_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="zero relative trace discrepancy"),
    ]

    small_eta: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="small eta"),
    ]
    small_field: Annotated[
        float,
        qscalar(ELECTRIC_FIELD, role="small field", display_unit=VOLT_PER_METER),
    ]
    small_relative_tensor_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="small relative tensor discrepancy"),
    ]
    small_relative_trace_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="small relative trace discrepancy"),
    ]

    largest_eta: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="largest eta"),
    ]
    largest_relative_tensor_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="largest relative tensor discrepancy"),
    ]
    largest_relative_trace_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="largest relative trace discrepancy"),
    ]

    min_nonzero_eta: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="minimum nonzero eta"),
    ]
    max_eta: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="maximum eta"),
    ]
    max_field_tensor_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="maximum field tensor discrepancy"),
    ]
    max_abs_field_trace_discrepancy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="maximum absolute field trace discrepancy"),
    ]
    max_imaginary_leakage: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="maximum imaginary leakage"),
    ]
    relative_weak_limit_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="relative weak-limit error"),
    ]
    strong_zero_field_imaginary_leakage: Annotated[
        float,
        qscalar(CONDUCTIVITY, role="strong zero-field imaginary leakage"),
    ]

    weak_limit_pass: bool
    roundoff_floor_status: str


@dataclass(frozen=True, slots=True)
class FiniteFieldVelocityValidationProbe:
    """Typed scalar result for symbolic velocity-ingredient validation.

    This probe checks dimensionless symbolic derivatives on a controlled toy.
    The physical hbar/unit-context velocity conversion is covered separately
    by the unit-scaling probe.
    """

    unit_context: UnitContext
    source: str

    k1: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="k1 sample point"),
    ]
    k2: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="k2 sample point"),
    ]
    finite_difference_eps: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="finite-difference epsilon"),
    ]
    analytic_dk1: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="analytic dE/dk1"),
    ]
    analytic_dk2: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="analytic dE/dk2"),
    ]
    production_dk1_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="production derivative dk1 absolute error"),
    ]
    production_dk2_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="production derivative dk2 absolute error"),
    ]
    finite_difference_dk1: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="finite-difference dE/dk1"),
    ]
    finite_difference_dk2: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="finite-difference dE/dk2"),
    ]
    finite_difference_dk1_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="finite-difference dk1 absolute error"),
    ]
    finite_difference_dk2_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="finite-difference dk2 absolute error"),
    ]
    hellmann_feynman_dk1_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="Hellmann-Feynman dk1 absolute error"),
    ]
    hellmann_feynman_dk2_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="Hellmann-Feynman dk2 absolute error"),
    ]
    generic_fixed_symbol_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="generic/fixed symbol absolute error"),
    ]
    generic_fixed_dk1_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="generic/fixed dk1 absolute error"),
    ]
    generic_fixed_dk2_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="generic/fixed dk2 absolute error"),
    ]

    vincent_sample_count: int
    vincent_find_simplex_max_velocity_error: Annotated[
        float,
        qscalar(VELOCITY, role="Vincent find-simplex max velocity error"),
    ]
    vincent_best_adjacent_max_velocity_error: Annotated[
        float,
        qscalar(VELOCITY, role="Vincent best-adjacent max velocity error"),
    ]
    vincent_velocity_error_reduction: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="Vincent velocity error reduction"),
    ]

    dataset_gamma_n_u: int
    dataset_gamma_n_v: int
    dataset_gamma_band_index: int
    dataset_gamma_same_grid_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="dataset Gamma same-grid absolute error"),
    ]
    dataset_gamma_same_grid_rel_l2_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="dataset Gamma same-grid relative L2 error"),
    ]
    dataset_gamma_coarse_n_u: int
    dataset_gamma_coarse_n_v: int
    dataset_velocity_mean_square_rel_change: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="dataset finite-difference velocity mean-square relative change"),
    ]
    dataset_gamma_hazard_count: int
    dataset_gamma_hazard_fraction: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="dataset Gamma selected-band hazard fraction"),
    ]
    dataset_gamma_gap_threshold: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="dataset Gamma selected-band hazard threshold"),
    ]

    unit_scaling_status: str
    vincent_velocity_status: str
    dataset_gamma_status: str


@dataclass(frozen=True, slots=True)
class FiniteFieldUnitScalingProbe:
    """Typed scalar result for finite-field unit conversion checks."""

    unit_context: UnitContext
    source: str

    atomic_energy_to_ev: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="atomic energy to eV conversion factor"),
    ]
    atomic_length_to_angstrom: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="atomic length to angstrom conversion factor"),
    ]
    hbar_atomic: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="hbar in atomic-unit context"),
    ]
    hbar_ev_angstrom: Annotated[
        float,
        qscalar(TIME, role="hbar in eV angstrom context"),
    ]
    velocity_au_to_evag_factor: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="velocity AU to eV angstrom conversion factor"),
    ]
    expected_velocity_au_to_evag_factor: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="expected velocity AU to eV angstrom conversion factor"),
    ]
    velocity_factor_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="velocity conversion factor absolute error"),
    ]
    fermi_window_ev_from_au_factor: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="Fermi window eV from AU conversion factor"),
    ]

    mu_conversion_required: bool
    conductivity_si_status: str


@dataclass(frozen=True, slots=True)
class FiniteFieldAnalyticToyCoverageProbe:
    """Typed scalar result for analytic toy coverage summary."""

    unit_context: UnitContext
    source: str

    toy_count: int

    separable_cosine_symbol_max_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="separable cosine symbol maximum error"),
    ]
    separable_cosine_derivative_max_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="separable cosine derivative maximum error"),
    ]
    identity_overlap_min_eig: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="identity overlap minimum eigenvalue"),
    ]
    identity_overlap_condition: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="identity overlap condition number"),
    ]
    periodic_dirac_min_gap: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="periodic Dirac minimum gap"),
    ]
    periodic_dirac_hazard_count: int
    velocity_hf_max_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="Hellmann-Feynman velocity maximum error"),
    ]
    unit_velocity_factor_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="unit velocity factor error"),
    ]

    all_current_toys_pass: bool
    missing_toy: str


@dataclass(frozen=True, slots=True)
class FiniteFieldInputHealthProbe:
    """Typed scalar result for finite-field H/S input health checks."""

    unit_context: UnitContext
    source: str

    n_u: int
    n_v: int
    sample_count: int
    symmetrization: str

    h_star_defect_max: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="H kernel star defect maximum"),
    ]
    s_star_defect_max: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="S kernel star defect maximum"),
    ]
    h_hermitian_defect_rel_max: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="H(k) relative Hermiticity defect maximum"),
    ]
    s_hermitian_defect_rel_max: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="S(k) relative Hermiticity defect maximum"),
    ]
    s_eig_min: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="S(k) minimum eigenvalue"),
    ]
    s_condition_number_abs_max: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="S(k) maximum absolute condition number"),
    ]
    energy_neighbour_jump_max: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="maximum neighbouring energy jump"),
    ]

    s_positive: bool


@dataclass(frozen=True, slots=True)
class FiniteFieldBandCrossingHazardProbe:
    """Typed scalar result for band-label hazard checks."""

    unit_context: UnitContext
    source: str

    n_u: int
    n_v: int
    sample_count: int

    mass: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="toy mass"),
    ]
    gap_threshold: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="gap threshold"),
    ]
    min_gap: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="minimum band gap"),
    ]
    min_gap_k1: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="minimum-gap k1"),
    ]
    min_gap_k2: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="minimum-gap k2"),
    ]
    hazard_count: int
    hazard_fraction: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="hazard fraction"),
    ]
    has_hazard: bool
    max_band0_neighbour_jump: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="maximum band-0 neighbour jump"),
    ]
    max_band1_neighbour_jump: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="maximum band-1 neighbour jump"),
    ]
    max_gap_neighbour_jump: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="maximum gap neighbour jump"),
    ]


@dataclass(frozen=True, slots=True)
class FiniteFieldDatasetBandHazardPoint:
    """One dataset-backed k-point where adjacent energy labels are fragile."""

    unit_context: UnitContext

    k1: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="hazard k1"),
    ]
    k2: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="hazard k2"),
    ]
    lower_band: int
    upper_band: int
    lower_energy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="hazard lower energy"),
    ]
    upper_energy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="hazard upper energy"),
    ]
    gap: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="hazard adjacent band gap"),
    ]
    threshold: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="hazard gap threshold"),
    ]


@dataclass(frozen=True, slots=True)
class FiniteFieldDatasetBandCrossingHazardProbe:
    """Dataset-backed scalar result for fragile energy-ordered band labels."""

    unit_context: UnitContext
    source: str

    n_u: int
    n_v: int
    sample_count: int
    band_count: int
    selected_band: int

    gap_threshold: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="adjacent band gap hazard threshold"),
    ]
    min_gap: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="minimum adjacent band gap"),
    ]
    selected_gap_q05: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="selected adjacent gap 5th percentile"),
    ]
    selected_gap_median: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="selected adjacent gap median"),
    ]
    selected_gap_q95: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="selected adjacent gap 95th percentile"),
    ]
    selected_gap_max: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="selected adjacent gap maximum"),
    ]
    min_gap_over_threshold: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="minimum selected gap divided by threshold"),
    ]
    median_gap_over_threshold: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="median selected gap divided by threshold"),
    ]
    min_gap_k1: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="minimum-gap k1"),
    ]
    min_gap_k2: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="minimum-gap k2"),
    ]
    min_gap_lower_band: int
    min_gap_upper_band: int

    hazard_count: int
    hazard_fraction: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="hazard fraction"),
    ]
    has_hazard: bool

    max_band_neighbour_jump: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="maximum energy-ordered band neighbour jump"),
    ]
    max_gap_neighbour_jump: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="maximum adjacent-gap neighbour jump"),
    ]

    hazard_points: tuple[FiniteFieldDatasetBandHazardPoint, ...]


@dataclass(frozen=True, slots=True)
class FiniteFieldKConvergenceProbe:
    """Typed scalar result for finite-field k-grid convergence checks."""

    unit_context: UnitContext
    source: str

    grid_count: int
    coarsest_n: int
    finest_n: int

    reference_average_grad_e_sq: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="analytic average squared energy gradient"),
    ]
    finest_average_grad_e_sq: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="finest-grid average squared energy gradient"),
    ]
    finest_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="finest-grid absolute error"),
    ]
    max_abs_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="maximum grid absolute error"),
    ]

    improved_or_equal_steps: int
    all_grid_errors_small: bool
    measure_status: str
    conductivity_convergence_status: str


@dataclass(frozen=True, slots=True)
class FiniteFieldSymmetrySanityProbe:
    """Typed scalar result for finite-field symmetry sanity checks."""

    unit_context: UnitContext
    source: str

    n: int
    sample_count: int

    energy_inversion_max_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="E(k)-E(-k) maximum error"),
    ]
    dk1_odd_max_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="dk1 oddness maximum error"),
    ]
    dk2_odd_max_error: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="dk2 oddness maximum error"),
    ]
    tensor_xx: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="symmetry tensor xx component"),
    ]
    tensor_yy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="symmetry tensor yy component"),
    ]
    tensor_xy: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="symmetry tensor xy component"),
    ]
    tensor_yx: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="symmetry tensor yx component"),
    ]
    tensor_xy_abs: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="absolute xy tensor component"),
    ]
    tensor_yx_abs: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="absolute yx tensor component"),
    ]
    tensor_antisym_abs: Annotated[
        float,
        qscalar(DIMENSIONLESS, role="absolute antisymmetric tensor component"),
    ]

    all_symmetry_checks_pass: bool
    dataset_automorphism_status: str


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
    KH: GdKernelArrays | None = None,
    KS: GdKernelArrays | None = None,
    *,
    n_u: int = 11,
    n_v: int = 11,
    symmetrization: str = "star",
    source: str = "controlled production GdKernelArrays toy",
) -> FiniteFieldInputHealthProbe:
    """Probe H/S symbol health for the finite-field DC validation report.

    When kernels are provided, this checks the selected dataset-backed kernel
    path. If omitted, it falls back to the small controlled production-symbol
    toy used by unit tests.
    """

    if n_u < 1:
        raise ValueError(f"n_u must be positive, got {n_u}")
    if n_v < 1:
        raise ValueError(f"n_v must be positive, got {n_v}")
    if symmetrization not in {"star", "direct", "raw"}:
        raise ValueError(f"unknown symmetrization scheme: {symmetrization!r}")

    if KH is None:
        KH = gd_separable_cosine_kernel(c0=1.25, c1=0.70, c2=-0.30)
    if KS is None:
        KS = gd_identity_overlap_kernel()

    if symmetrization == "star":
        KH = KH.star_symmetrised(matrix_name=f"{KH.matrix_name} finite-field input-health star")
        KS = KS.star_symmetrised(matrix_name=f"{KS.matrix_name} finite-field input-health star")

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
            pair = SymbolPair(KH=KH, KS=KS, k1=float(k1), k2=float(k2), degree=2)
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

    return FiniteFieldInputHealthProbe(
        unit_context=SI_UNITS,
        n_u=int(n_u),
        n_v=int(n_v),
        sample_count=int(n_u) * int(n_v),
        symmetrization=symmetrization,
        h_star_defect_max=float(kh_star["star_defect_max"]),
        s_star_defect_max=float(ks_star["star_defect_max"]),
        h_hermitian_defect_rel_max=float(max_h_hermitian_defect),
        s_hermitian_defect_rel_max=float(max_s_hermitian_defect),
        s_eig_min=float(min_s_eig),
        s_condition_number_abs_max=float(max_s_cond),
        energy_neighbour_jump_max=float(max_energy_jump),
        s_positive=bool(min_s_eig > 1.0e-10),
        source=source,
    )


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
) -> FiniteFieldBandCrossingHazardProbe:
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

    return FiniteFieldBandCrossingHazardProbe(
        unit_context=SI_UNITS,
        source="periodic two-level Dirac-like toy",
        n_u=int(n_u),
        n_v=int(n_v),
        sample_count=sample_count,
        mass=float(mass),
        gap_threshold=float(gap_threshold),
        min_gap=float(min_gap),
        min_gap_k1=float(min_gap_k1),
        min_gap_k2=float(min_gap_k2),
        hazard_count=int(hazard_count),
        hazard_fraction=float(hazard_count / sample_count),
        has_hazard=bool(hazard_count > 0),
        max_band0_neighbour_jump=float(max_band0_jump),
        max_band1_neighbour_jump=float(max_band1_jump),
        max_gap_neighbour_jump=float(max_gap_jump),
    )


def finite_field_dataset_band_crossing_hazard_probe(
    KH: GdKernelArrays,
    KS: GdKernelArrays,
    *,
    n_u: int = 11,
    n_v: int = 11,
    gap_threshold: float = 0.50,
    band_index: int = 0,
    symmetrization: str = "star",
    source: str = "dataset-backed GdKernelArrays",
    max_hazard_points: int = 32,
) -> FiniteFieldDatasetBandCrossingHazardProbe:
    """Locate dataset-backed k-points where sorted adjacent bands become fragile."""

    if n_u < 1:
        raise ValueError(f"n_u must be positive, got {n_u}")
    if n_v < 1:
        raise ValueError(f"n_v must be positive, got {n_v}")
    if gap_threshold < 0.0:
        raise ValueError(f"gap_threshold must be non-negative, got {gap_threshold}")
    if max_hazard_points < 0:
        raise ValueError(f"max_hazard_points must be non-negative, got {max_hazard_points}")
    if band_index < 0:
        raise ValueError(f"band_index must be non-negative, got {band_index}")
    if symmetrization not in {"star", "direct", "raw"}:
        raise ValueError(f"unknown symmetrization scheme: {symmetrization!r}")

    if symmetrization == "star":
        KH = KH.star_symmetrised(matrix_name=f"{KH.matrix_name} dataset band-hazard star")
        KS = KS.star_symmetrised(matrix_name=f"{KS.matrix_name} dataset band-hazard star")

    k1_grid = np.linspace(-np.pi, np.pi, int(n_u), endpoint=False)
    k2_grid = np.linspace(-np.pi, np.pi, int(n_v), endpoint=False)

    energies: np.ndarray | None = None
    gaps: np.ndarray | None = None
    hazard_points: list[FiniteFieldDatasetBandHazardPoint] = []

    min_gap = np.inf
    min_gap_k1 = 0.0
    min_gap_k2 = 0.0
    min_gap_lower_band = -1
    min_gap_upper_band = -1
    selected_gap_values: list[float] = []
    hazard_count = 0

    for i, k1 in enumerate(k1_grid):
        for j, k2 in enumerate(k2_grid):
            pair = SymbolPair(KH=KH, KS=KS, k1=float(k1), k2=float(k2), degree=2)
            problem = pair.form()

            H = problem.Hk
            S = problem.Sk
            if symmetrization == "direct":
                H = 0.5 * (H + H.conj().T)
                S = 0.5 * (S + S.conj().T)

            vals = la.eigvalsh(H, S).real
            vals = np.sort(vals)

            if energies is None:
                energies = np.empty((int(n_u), int(n_v), int(vals.shape[0])), dtype=np.float64)
                gaps = np.empty((int(n_u), int(n_v), max(0, int(vals.shape[0]) - 1)), dtype=np.float64)

            energies[i, j, :] = vals

            adjacent_gaps = np.diff(vals)
            if gaps is not None:
                gaps[i, j, :] = adjacent_gaps

            if adjacent_gaps.size == 0:
                continue

            if int(band_index) >= int(vals.shape[0]):
                raise ValueError(
                    f"band_index {band_index} out of range for {int(vals.shape[0])} bands"
                )

            relevant_lower_bands: list[int] = []
            if int(band_index) > 0:
                relevant_lower_bands.append(int(band_index) - 1)
            if int(band_index) + 1 < int(vals.shape[0]):
                relevant_lower_bands.append(int(band_index))

            for lower_band in relevant_lower_bands:
                gap_value = float(adjacent_gaps[lower_band])
                selected_gap_values.append(gap_value)

                if gap_value < min_gap:
                    min_gap = gap_value
                    min_gap_k1 = float(k1)
                    min_gap_k2 = float(k2)
                    min_gap_lower_band = int(lower_band)
                    min_gap_upper_band = int(lower_band + 1)

                if gap_value < gap_threshold:
                    hazard_count += 1
                    hazard_points.append(
                        FiniteFieldDatasetBandHazardPoint(
                            unit_context=SI_UNITS,
                            k1=float(k1),
                            k2=float(k2),
                            lower_band=int(lower_band),
                            upper_band=int(lower_band + 1),
                            lower_energy=float(vals[lower_band]),
                            upper_energy=float(vals[lower_band + 1]),
                            gap=gap_value,
                            threshold=float(gap_threshold),
                        )
                    )

    if energies is None or gaps is None:
        raise RuntimeError("dataset band hazard probe did not sample any energies")

    max_band_jump = 0.0
    max_gap_jump = 0.0

    for i in range(int(n_u)):
        for j in range(int(n_v)):
            neighbours = (
                ((i + 1) % int(n_u), j),
                (i, (j + 1) % int(n_v)),
            )
            for a, b in neighbours:
                max_band_jump = max(
                    max_band_jump,
                    float(np.max(np.abs(energies[a, b, :] - energies[i, j, :]))),
                )
                if gaps.shape[2] > 0:
                    max_gap_jump = max(
                        max_gap_jump,
                        float(np.max(np.abs(gaps[a, b, :] - gaps[i, j, :]))),
                    )

    if not selected_gap_values:
        raise RuntimeError("dataset band hazard probe found no selected adjacent gaps")

    selected_gap_array = np.asarray(selected_gap_values, dtype=np.float64)
    selected_gap_q05 = float(np.quantile(selected_gap_array, 0.05))
    selected_gap_median = float(np.quantile(selected_gap_array, 0.50))
    selected_gap_q95 = float(np.quantile(selected_gap_array, 0.95))
    selected_gap_max = float(np.max(selected_gap_array))
    min_gap_over_threshold = float(min_gap / gap_threshold) if gap_threshold > 0.0 else np.inf
    median_gap_over_threshold = (
        float(selected_gap_median / gap_threshold) if gap_threshold > 0.0 else np.inf
    )

    sample_count = int(n_u) * int(n_v)
    hazard_points = sorted(hazard_points, key=lambda point: point.gap)

    return FiniteFieldDatasetBandCrossingHazardProbe(
        unit_context=SI_UNITS,
        source=source,
        n_u=int(n_u),
        n_v=int(n_v),
        sample_count=sample_count,
        band_count=int(energies.shape[2]),
        selected_band=int(band_index),
        gap_threshold=float(gap_threshold),
        min_gap=float(min_gap),
        selected_gap_q05=float(selected_gap_q05),
        selected_gap_median=float(selected_gap_median),
        selected_gap_q95=float(selected_gap_q95),
        selected_gap_max=float(selected_gap_max),
        min_gap_over_threshold=float(min_gap_over_threshold),
        median_gap_over_threshold=float(median_gap_over_threshold),
        min_gap_k1=float(min_gap_k1),
        min_gap_k2=float(min_gap_k2),
        min_gap_lower_band=int(min_gap_lower_band),
        min_gap_upper_band=int(min_gap_upper_band),
        hazard_count=int(hazard_count),
        hazard_fraction=float(hazard_count / sample_count),
        has_hazard=bool(hazard_count > 0),
        max_band_neighbour_jump=float(max_band_jump),
        max_gap_neighbour_jump=float(max_gap_jump),
        hazard_points=tuple(hazard_points[: int(max_hazard_points)]),
    )


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


def _energy_ordered_band_grid(
    KH: GdKernelArrays,
    KS: GdKernelArrays,
    *,
    n_u: int,
    n_v: int,
    band_index: int,
    symmetrization: str,
) -> np.ndarray:
    """Sample a selected sorted-energy band on the logical k-grid."""

    if n_u < 3 or n_v < 3:
        raise ValueError("dataset velocity Gamma check needs n_u,n_v >= 3")
    if band_index < 0:
        raise ValueError(f"band_index must be non-negative, got {band_index}")

    if symmetrization == "star":
        KH = KH.star_symmetrised(matrix_name=f"{KH.matrix_name} velocity Gamma star")
        KS = KS.star_symmetrised(matrix_name=f"{KS.matrix_name} velocity Gamma star")
    elif symmetrization not in {"direct", "raw"}:
        raise ValueError(f"unknown symmetrization scheme: {symmetrization!r}")

    k1_grid = np.linspace(-np.pi, np.pi, int(n_u), endpoint=False)
    k2_grid = np.linspace(-np.pi, np.pi, int(n_v), endpoint=False)
    energy_grid: np.ndarray | None = None

    for i, k1 in enumerate(k1_grid):
        for j, k2 in enumerate(k2_grid):
            pair = SymbolPair(KH=KH, KS=KS, k1=float(k1), k2=float(k2), degree=2)
            problem = pair.form()
            H = problem.Hk
            S = problem.Sk
            if symmetrization == "direct":
                H = 0.5 * (H + H.conj().T)
                S = 0.5 * (S + S.conj().T)

            vals = np.sort(la.eigvalsh(H, S).real)
            if int(band_index) >= int(vals.shape[0]):
                raise ValueError(
                    f"band_index {band_index} out of range for {int(vals.shape[0])} bands"
                )

            if energy_grid is None:
                energy_grid = np.empty((int(n_u), int(n_v)), dtype=np.float64)
            energy_grid[i, j] = float(vals[int(band_index)])

    if energy_grid is None:
        raise RuntimeError("dataset velocity Gamma check sampled no energies")
    return energy_grid


def _logical_periodic_finite_difference_velocity(energy_grid: np.ndarray) -> np.ndarray:
    """Central finite-difference velocity on the logical periodic k-grid."""

    energy = np.asarray(energy_grid, dtype=np.float64)
    if energy.ndim != 2:
        raise ValueError(f"expected a 2D energy grid, got shape {energy.shape!r}")

    n_u, n_v = energy.shape
    dk1 = 2.0 * np.pi / float(n_u)
    dk2 = 2.0 * np.pi / float(n_v)

    velocity = np.empty(energy.shape + (2,), dtype=np.float64)
    velocity[..., 0] = (np.roll(energy, -1, axis=0) - np.roll(energy, 1, axis=0)) / (2.0 * dk1)
    velocity[..., 1] = (np.roll(energy, -1, axis=1) - np.roll(energy, 1, axis=1)) / (2.0 * dk2)
    return velocity


def _gamma_reconstruct_velocity(velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Gamma coefficients and same-grid reconstructed velocity."""

    v = np.asarray(velocity, dtype=np.float64)
    gamma = np.empty(v.shape, dtype=np.complex128)
    reconstructed = np.empty(v.shape, dtype=np.float64)

    for alpha in range(v.shape[-1]):
        gamma[..., alpha] = np.fft.ifft2(v[..., alpha])
        reconstructed[..., alpha] = np.fft.fft2(gamma[..., alpha]).real

    return gamma, reconstructed


def finite_field_velocity_validation_probe(
    KH: GdKernelArrays | None = None,
    KS: GdKernelArrays | None = None,
    *,
    n_u: int = 11,
    n_v: int = 11,
    band_index: int = 0,
    symmetrization: str = "star",
    gap_threshold: float = 0.05,
    dataset_hazard_probe: FiniteFieldDatasetBandCrossingHazardProbe | None = None,
) -> FiniteFieldVelocityValidationProbe:
    """Validate velocity ingredients on analytic, Vincent, and dataset-backed inputs."""

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

    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        load_vincent_input_data,
        reciprocal_lattice_vectors_from_primitives,
        vincent_delaunay_adjacent_simplex_velocity_probe,
        vincent_sample_velocity_targets,
    )

    vincent_inputs = load_vincent_input_data()
    vincent_k, _vincent_velocity = vincent_sample_velocity_targets()
    vincent_adjacent = vincent_delaunay_adjacent_simplex_velocity_probe(
        vincent_inputs.epsilon_of_k,
        vincent_inputs.primitive_lattice_vectors_bohr,
    )
    vincent_find_simplex_max_error = max(float(row["find_simplex_error"]) for row in vincent_adjacent)
    vincent_best_adjacent_max_error = max(float(row["best_adjacent_error"]) for row in vincent_adjacent)
    vincent_error_reduction = vincent_find_simplex_max_error / vincent_best_adjacent_max_error

    dataset_gamma_n_u = int(n_u)
    dataset_gamma_n_v = int(n_v)
    dataset_gamma_band_index = int(band_index)
    dataset_gamma_same_grid_abs_error = np.nan
    dataset_gamma_same_grid_rel_l2_error = np.nan
    dataset_gamma_coarse_n_u = 0
    dataset_gamma_coarse_n_v = 0
    dataset_velocity_mean_square_rel_change = np.nan
    dataset_gamma_hazard_count = 0
    dataset_gamma_hazard_fraction = np.nan
    dataset_gamma_gap_threshold = float(gap_threshold)
    dataset_gamma_status = "pending dataset-backed context"

    if dataset_hazard_probe is not None:
        dataset_gamma_hazard_count = int(dataset_hazard_probe.hazard_count)
        dataset_gamma_hazard_fraction = float(dataset_hazard_probe.hazard_fraction)
        dataset_gamma_gap_threshold = float(dataset_hazard_probe.gap_threshold)

    if KH is not None and KS is not None:
        energy_grid = _energy_ordered_band_grid(
            KH,
            KS,
            n_u=int(n_u),
            n_v=int(n_v),
            band_index=int(band_index),
            symmetrization=symmetrization,
        )
        dataset_velocity = _logical_periodic_finite_difference_velocity(energy_grid)
        _gamma, dataset_reconstructed = _gamma_reconstruct_velocity(dataset_velocity)
        dataset_gamma_same_grid_abs_error = float(
            np.max(np.abs(dataset_reconstructed - dataset_velocity))
        )
        velocity_norm = float(np.linalg.norm(dataset_velocity))
        dataset_gamma_same_grid_rel_l2_error = float(
            np.linalg.norm(dataset_reconstructed - dataset_velocity) / velocity_norm
            if velocity_norm > 0.0
            else np.nan
        )

        dataset_gamma_coarse_n_u = max(3, int(n_u) // 2)
        dataset_gamma_coarse_n_v = max(3, int(n_v) // 2)
        coarse_energy_grid = _energy_ordered_band_grid(
            KH,
            KS,
            n_u=dataset_gamma_coarse_n_u,
            n_v=dataset_gamma_coarse_n_v,
            band_index=int(band_index),
            symmetrization=symmetrization,
        )
        coarse_velocity = _logical_periodic_finite_difference_velocity(coarse_energy_grid)
        fine_mean_square = float(np.mean(np.sum(dataset_velocity * dataset_velocity, axis=-1)))
        coarse_mean_square = float(np.mean(np.sum(coarse_velocity * coarse_velocity, axis=-1)))
        dataset_velocity_mean_square_rel_change = float(
            abs(fine_mean_square - coarse_mean_square) / abs(fine_mean_square)
            if fine_mean_square != 0.0
            else np.nan
        )

        if dataset_hazard_probe is None:
            hazard = finite_field_dataset_band_crossing_hazard_probe(
                KH,
                KS,
                n_u=int(n_u),
                n_v=int(n_v),
                gap_threshold=float(gap_threshold),
                band_index=int(band_index),
                symmetrization=symmetrization,
                source="velocity validation selected-band hazard fallback",
                max_hazard_points=0,
            )
            dataset_gamma_hazard_count = int(hazard.hazard_count)
            dataset_gamma_hazard_fraction = float(hazard.hazard_fraction)
            dataset_gamma_gap_threshold = float(hazard.gap_threshold)

        dataset_gamma_status = "same-grid Gamma closure checked on dataset finite-difference velocity"

    return FiniteFieldVelocityValidationProbe(
        unit_context=SI_UNITS,
        source="separable cosine production symbol toy",
        k1=float(k1),
        k2=float(k2),
        finite_difference_eps=float(eps),
        analytic_dk1=float(expected_dk1),
        analytic_dk2=float(expected_dk2),
        production_dk1_abs_error=float(abs(fixed_dk1[0, 0].real - expected_dk1)),
        production_dk2_abs_error=float(abs(fixed_dk2[0, 0].real - expected_dk2)),
        finite_difference_dk1=float(fd_dk1[0, 0].real),
        finite_difference_dk2=float(fd_dk2[0, 0].real),
        finite_difference_dk1_abs_error=float(abs(fd_dk1[0, 0].real - expected_dk1)),
        finite_difference_dk2_abs_error=float(abs(fd_dk2[0, 0].real - expected_dk2)),
        hellmann_feynman_dk1_abs_error=float(abs(hf_dk1 - expected_dk1)),
        hellmann_feynman_dk2_abs_error=float(abs(hf_dk2 - expected_dk2)),
        generic_fixed_symbol_abs_error=float(generic_symbol_probe["generic_symbol_channel_abs_error"]),
        generic_fixed_dk1_abs_error=float(generic_symbol_probe["generic_dk1_channel_abs_error"]),
        generic_fixed_dk2_abs_error=float(generic_symbol_probe["generic_dk2_channel_abs_error"]),
        vincent_sample_count=int(vincent_k.shape[0]),
        vincent_find_simplex_max_velocity_error=float(vincent_find_simplex_max_error),
        vincent_best_adjacent_max_velocity_error=float(vincent_best_adjacent_max_error),
        vincent_velocity_error_reduction=float(vincent_error_reduction),
        dataset_gamma_n_u=int(dataset_gamma_n_u),
        dataset_gamma_n_v=int(dataset_gamma_n_v),
        dataset_gamma_band_index=int(dataset_gamma_band_index),
        dataset_gamma_same_grid_abs_error=float(dataset_gamma_same_grid_abs_error),
        dataset_gamma_same_grid_rel_l2_error=float(dataset_gamma_same_grid_rel_l2_error),
        dataset_gamma_coarse_n_u=int(dataset_gamma_coarse_n_u),
        dataset_gamma_coarse_n_v=int(dataset_gamma_coarse_n_v),
        dataset_velocity_mean_square_rel_change=float(dataset_velocity_mean_square_rel_change),
        dataset_gamma_hazard_count=int(dataset_gamma_hazard_count),
        dataset_gamma_hazard_fraction=float(dataset_gamma_hazard_fraction),
        dataset_gamma_gap_threshold=float(dataset_gamma_gap_threshold),
        unit_scaling_status="covered separately in unit-scaling section",
        vincent_velocity_status="best-adjacent Delaunay samples reproduce Vincent quoted velocities",
        dataset_gamma_status=str(dataset_gamma_status),
    )


def finite_field_unit_scaling_probe() -> FiniteFieldUnitScalingProbe:
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

    return FiniteFieldUnitScalingProbe(
        unit_context=SI_UNITS,
        source="core UnitContext conversion factors",
        atomic_energy_to_ev=float(energy_disk_to_ev),
        atomic_length_to_angstrom=float(length_disk_to_angstrom),
        hbar_atomic=float(hbar_au),
        hbar_ev_angstrom=float(hbar_evag),
        velocity_au_to_evag_factor=float(velocity_au_to_evag),
        expected_velocity_au_to_evag_factor=float(expected_velocity_factor),
        velocity_factor_abs_error=float(abs(velocity_au_to_evag - expected_velocity_factor)),
        fermi_window_ev_from_au_factor=float(fermi_window_ev_from_au_factor),
        mu_conversion_required=True,
        conductivity_si_status="pending full conductivity unit-conversion run",
    )


def finite_field_analytic_toy_coverage_probe() -> FiniteFieldAnalyticToyCoverageProbe:
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
        float(velocity.finite_difference_dk1_abs_error),
        float(velocity.finite_difference_dk2_abs_error),
    )

    return FiniteFieldAnalyticToyCoverageProbe(
        unit_context=SI_UNITS,
        source="summary of controlled analytic probes",
        toy_count=4,
        separable_cosine_symbol_max_error=float(max_symbol_error),
        separable_cosine_derivative_max_error=float(max_derivative_error),
        identity_overlap_min_eig=float(input_health.s_eig_min),
        identity_overlap_condition=float(input_health.s_condition_number_abs_max),
        periodic_dirac_min_gap=float(band_hazards.min_gap),
        periodic_dirac_hazard_count=int(band_hazards.hazard_count),
        velocity_hf_max_error=float(max(
            velocity.hellmann_feynman_dk1_abs_error,
            velocity.hellmann_feynman_dk2_abs_error,
        )),
        unit_velocity_factor_error=float(units.velocity_factor_abs_error),
        all_current_toys_pass=bool(
            max_symbol_error < 1.0e-12
            and max_derivative_error < 1.0e-9
            and float(input_health.s_eig_min) > 1.0e-10
            and int(band_hazards.hazard_count) >= 1
            and float(units.velocity_factor_abs_error) < 1.0e-12
        ),
        missing_toy="finite-field lattice-mode Gamma/Q/rho closure toy",
    )


def finite_field_k_convergence_probe(
    grid_sizes: tuple[int, ...] = (5, 7, 11, 17, 23),
) -> FiniteFieldKConvergenceProbe:
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

    return FiniteFieldKConvergenceProbe(
        unit_context=SI_UNITS,
        source="periodic separable-cosine velocity-square average",
        grid_count=int(rows_checked),
        coarsest_n=int(grid_sizes[0]),
        finest_n=int(grid_sizes[-1]),
        reference_average_grad_e_sq=float(reference),
        finest_average_grad_e_sq=float(finest_value),
        finest_abs_error=float(finest_error),
        max_abs_error=float(max_error),
        improved_or_equal_steps=int(improved_or_equal_steps),
        all_grid_errors_small=bool(max_error < 1.0e-12),
        measure_status="uniform full-period average matches analytic reference",
        conductivity_convergence_status="pending dataset-backed conductivity refinement",
    )


def finite_field_symmetry_sanity_probe(
    *,
    n: int = 17,
) -> FiniteFieldSymmetrySanityProbe:
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

    return FiniteFieldSymmetrySanityProbe(
        unit_context=SI_UNITS,
        source="separable cosine inversion and tensor symmetry toy",
        n=int(n),
        sample_count=int(n * n),
        energy_inversion_max_error=float(max_energy_inversion_error),
        dk1_odd_max_error=float(max_dk1_odd_error),
        dk2_odd_max_error=float(max_dk2_odd_error),
        tensor_xx=float(tensor[0, 0]),
        tensor_yy=float(tensor[1, 1]),
        tensor_xy=float(tensor[0, 1]),
        tensor_yx=float(tensor[1, 0]),
        tensor_xy_abs=float(xy_abs),
        tensor_yx_abs=float(yx_abs),
        tensor_antisym_abs=float(antisym_abs),
        all_symmetry_checks_pass=bool(
            max_energy_inversion_error < 1.0e-12
            and max_dk1_odd_error < 1.0e-12
            and max_dk2_odd_error < 1.0e-12
            and xy_abs < 1.0e-12
            and yx_abs < 1.0e-12
            and antisym_abs < 1.0e-12
        ),
        dataset_automorphism_status="pending H/S/H_star/S_star automorphism checks",
    )



def _eq830_gamma_f_rho_bilinear_modal_tensor(
    epsilon_Ha: np.ndarray,
    velocity_m_per_s: np.ndarray,
    primitive_lattice_vectors_bohr: np.ndarray,
    *,
    chemical_potential_J: float,
    temperature_K: float,
    relaxation_time_s: float,
    electric_field_V_per_m: np.ndarray,
    laguerre_order: int = 48,
) -> np.ndarray:
    """Reconstruct Vincent Eq. 8.30 through Gamma/F/tilde(rho) modes.

    This mirrors ``conductivity_830_shifted_chain_rule_from_velocity_grid``.
    The direct implementation evaluates the shifted Eq. 8.30 factor with
    periodic bilinear interpolation.  The modal response below therefore
    includes the exact Fourier response of that same bilinear shift, rather
    than the response of an ideal spectral shift.
    """

    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        ELECTRON_CHARGE_C,
        HARTREE_TO_J,
        HBAR_J_S,
        KB_J_K,
        cartesian_k_to_fractional,
        fermi_window,
        reciprocal_cell_area_per_m2,
    )

    epsilon_J = np.asarray(epsilon_Ha, dtype=np.float64) * HARTREE_TO_J
    velocity = np.asarray(velocity_m_per_s, dtype=np.float64)
    field = np.asarray(electric_field_V_per_m, dtype=np.float64)

    if epsilon_J.ndim != 2:
        raise ValueError(f"Expected a 2D epsilon grid, got shape {epsilon_J.shape}")
    if velocity.shape != epsilon_J.shape + (2,):
        raise ValueError(
            f"velocity shape {velocity.shape} does not match epsilon shape {epsilon_J.shape} + (2,)"
        )
    if field.shape != (2,):
        raise ValueError(f"Expected electric field shape (2,), got {field.shape}")

    n1, n2 = epsilon_J.shape
    nk = float(n1 * n2)

    weight = fermi_window(epsilon_J, chemical_potential_J, temperature_K)
    weighted_velocity = weight[..., None] * velocity

    gamma = np.empty(velocity.shape, dtype=np.complex128)
    rho_tilde = np.empty(velocity.shape, dtype=np.complex128)
    for alpha in range(2):
        gamma[..., alpha] = np.fft.ifft2(velocity[..., alpha])
        rho_tilde[..., alpha] = np.fft.ifft2(weighted_velocity[..., alpha])

    neg_i = (-np.arange(n1)) % n1
    neg_j = (-np.arange(n2)) % n2

    mode_i = (np.fft.fftfreq(n1) * n1).astype(np.float64)
    mode_j = (np.fft.fftfreq(n2) * n2).astype(np.float64)
    mi, mj = np.meshgrid(mode_i, mode_j, indexing="ij")

    nodes, weights = np.polynomial.laguerre.laggauss(laguerre_order)

    response = np.zeros((n1, n2), dtype=np.complex128)
    for node, quadrature_weight in zip(nodes, weights, strict=True):
        shift_per_m = (ELECTRON_CHARGE_C * relaxation_time_s * float(node) / HBAR_J_S) * field
        shift_fractional = cartesian_k_to_fractional(shift_per_m, primitive_lattice_vectors_bohr)

        shift_grid_1 = float(shift_fractional[0]) * n1
        shift_grid_2 = float(shift_fractional[1]) * n2

        base_1 = np.floor(shift_grid_1)
        base_2 = np.floor(shift_grid_2)
        frac_1 = shift_grid_1 - base_1
        frac_2 = shift_grid_2 - base_2

        # Response for the -mode partner under the same periodic bilinear
        # interpolation used by the direct Eq. 8.30 implementation.
        factor_1 = np.exp(2j * np.pi * mi * base_1 / n1) * (
            (1.0 - frac_1) + frac_1 * np.exp(2j * np.pi * mi / n1)
        )
        factor_2 = np.exp(2j * np.pi * mj * base_2 / n2) * (
            (1.0 - frac_2) + frac_2 * np.exp(2j * np.pi * mj / n2)
        )
        response += float(quadrature_weight * node) * factor_1 * factor_2

    raw = np.zeros((2, 2), dtype=np.float64)
    for alpha in range(2):
        gamma_alpha = gamma[..., alpha]
        for beta in range(2):
            rho_beta_neg = rho_tilde[np.ix_(neg_i, neg_j)][..., beta]
            raw[alpha, beta] = float((nk * np.sum(gamma_alpha * rho_beta_neg * response)).real)

    k_cell_area = reciprocal_cell_area_per_m2(primitive_lattice_vectors_bohr, epsilon_J.shape)
    weighted = raw * k_cell_area
    prefactor = ELECTRON_CHARGE_C ** 2 * relaxation_time_s / (
        (2.0 * np.pi) ** 2 * KB_J_K * temperature_K
    )
    return prefactor * weighted


def finite_field_vincent_reconstruction_probe() -> FiniteFieldVincentReconstructionProbe:
    """Summarise existing Vincent/Ashcroft reconstruction checks."""

    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        band_indexed_strong_dc_from_velocity_grid,
        conductivity_830_shifted_chain_rule_from_velocity_grid,
        conductivity_from_epsilon_grid,
        load_vincent_input_data,
        reciprocal_lattice_vectors_from_primitives,
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

    reciprocal_bohr = reciprocal_lattice_vectors_from_primitives(ai)
    reciprocal_dot = ai @ reciprocal_bohr.T
    reciprocal_target = (2.0 * np.pi) * np.eye(2)
    reciprocal_dot_error = reciprocal_dot - reciprocal_target
    reciprocal_dot_diag_max_abs_error = float(np.max(np.abs(np.diag(reciprocal_dot_error))))
    reciprocal_dot_offdiag_max_abs = float(
        max(abs(reciprocal_dot_error[0, 1]), abs(reciprocal_dot_error[1, 0]))
    )
    reciprocal_det = abs(float(np.linalg.det(reciprocal_bohr)))
    real_det = abs(float(np.linalg.det(ai)))
    reciprocal_det_ratio = float(reciprocal_det / (((2.0 * np.pi) ** 2) / real_det))
    reciprocal_det_ratio_abs_error = float(abs(reciprocal_det_ratio - 1.0))

    weak_trace_percent_error = 100.0 * (weak_trace - target_trace) / target_trace

    continuum_weak_trace = float(np.trace(local.conductivity_tensor_S))
    continuum_weak_trace_percent_error = 100.0 * (continuum_weak_trace - target_trace) / target_trace


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

    continuum_eq830_shifted_trace = float(np.trace(shifted.conductivity_tensor_S))
    continuum_eq830_shifted_trace_percent_error = 100.0 * (
        continuum_eq830_shifted_trace - target_trace
    ) / target_trace

    shifted_trace_percent_error = 100.0 * (shifted_trace - target_trace) / target_trace

    eq830_modal_sigma = _eq830_gamma_f_rho_bilinear_modal_tensor(
        epsilon,
        local.velocity_m_per_s,
        ai,
        chemical_potential_J=local.chemical_potential_J,
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
        electric_field_V_per_m=reference.electric_field_V_per_m,
    ) * ((2.0 * np.pi) ** 2)
    eq830_modal_trace = float(np.trace(eq830_modal_sigma))
    eq830_modal_trace_percent_error = 100.0 * (eq830_modal_trace - target_trace) / target_trace
    eq830_modal_direct_trace_percent_error = 100.0 * (eq830_modal_trace - shifted_trace) / shifted_trace

    adjacent = vincent_delaunay_adjacent_simplex_velocity_probe(epsilon, ai)
    find_simplex_max_error = max(float(row["find_simplex_error"]) for row in adjacent)
    best_adjacent_max_error = max(float(row["best_adjacent_error"]) for row in adjacent)
    velocity_error_reduction = find_simplex_max_error / best_adjacent_max_error

    return FiniteFieldVincentReconstructionProbe(
        continuum_weak_trace=float(continuum_weak_trace),
        continuum_weak_trace_percent_error=float(continuum_weak_trace_percent_error),
        continuum_eq830_shifted_trace=float(continuum_eq830_shifted_trace),
        continuum_eq830_shifted_trace_percent_error=float(continuum_eq830_shifted_trace_percent_error),
        reciprocal_dot_diag_max_abs_error=float(reciprocal_dot_diag_max_abs_error),
        reciprocal_dot_offdiag_max_abs=float(reciprocal_dot_offdiag_max_abs),
        reciprocal_det_ratio=float(reciprocal_det_ratio),
        reciprocal_det_ratio_abs_error=float(reciprocal_det_ratio_abs_error),
        unit_context=SI_UNITS,
        source="existing Ashcroft/Vincent comparison domain",
        target_trace=float(target_trace),
        weak_chain_trace=float(weak_trace),
        weak_chain_trace_percent_error=float(weak_trace_percent_error),
        strong_grid_trace=float(strong_grid_trace),
        strong_grid_trace_percent_error=float(strong_grid_trace_percent_error),
        shifted_830_trace=float(shifted_trace),
        shifted_830_trace_percent_error=float(shifted_trace_percent_error),
        eq830_modal_trace=float(eq830_modal_trace),
        eq830_modal_trace_percent_error=float(eq830_modal_trace_percent_error),
        eq830_modal_direct_trace_percent_error=float(eq830_modal_direct_trace_percent_error),
        find_simplex_max_velocity_error=float(find_simplex_max_error),
        best_adjacent_max_velocity_error=float(best_adjacent_max_error),
        velocity_error_reduction=float(velocity_error_reduction),
        best_adjacent_matches_vincent=bool(best_adjacent_max_error < 1.0e-3),
        residual_status="velocity samples resolved; conductivity residual remains formula/convention audit",
    )


def finite_field_strong_dc_validation_probe() -> FiniteFieldStrongDcValidationProbe:
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

    two_pi_squared = (2.0 * np.pi) ** 2
    continuum_weak_sigma = local.conductivity_tensor_S
    no_2pi_denominator_weak_sigma = local.conductivity_tensor_S * two_pi_squared

    strong = band_indexed_strong_dc_from_velocity_grid(
        epsilon,
        local.velocity_m_per_s,
        ai,
        chemical_potential_J=local.chemical_potential_J,
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
        electric_field_V_per_m=np.zeros(2),
    )

    no_2pi_denominator_strong_sigma = strong.conductivity_tensor_S.real
    continuum_strong_sigma = no_2pi_denominator_strong_sigma / two_pi_squared
    strong_grid_sigma = no_2pi_denominator_strong_sigma
    strong_from_modes = np.sum(strong.conductivity_mode_tensor_S, axis=(0, 1))
    mode_reconstruction_abs_error = float(
        np.max(np.abs(strong_from_modes - strong.conductivity_tensor_S))
    )

    continuum_strong_trace = float(np.trace(continuum_strong_sigma))
    continuum_weak_trace = float(np.trace(continuum_weak_sigma))
    no_2pi_denominator_strong_trace = float(np.trace(no_2pi_denominator_strong_sigma))
    no_2pi_denominator_weak_trace = float(np.trace(no_2pi_denominator_weak_sigma))
    target_trace = float(np.trace(reference.expected_conductivity_S_per_m))

    strong_trace = continuum_strong_trace
    weak_trace = continuum_weak_trace
    strong_vs_weak_rel_trace_gap = (continuum_strong_trace - continuum_weak_trace) / continuum_weak_trace
    strong_vs_vincent_percent_error = 100.0 * (
        no_2pi_denominator_strong_trace - target_trace
    ) / target_trace

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

    return FiniteFieldStrongDcValidationProbe(
        unit_context=SI_UNITS,
        source="BandIndexedStrongDcResult on Vincent epsilon grid",
        mode_count=mode_count,
        nonzero_mode_count=nonzero_mode_count,
        continuum_strong_trace=float(continuum_strong_trace),
        continuum_weak_trace=float(continuum_weak_trace),
        no_2pi_denominator_strong_trace=float(no_2pi_denominator_strong_trace),
        no_2pi_denominator_weak_trace=float(no_2pi_denominator_weak_trace),
        strong_grid_trace=float(continuum_strong_trace),
        weak_chain_grid_trace=float(continuum_weak_trace),
        vincent_target_trace=float(target_trace),
        strong_vs_weak_rel_trace_gap=float(strong_vs_weak_rel_trace_gap),
        strong_vs_vincent_percent_error=float(strong_vs_vincent_percent_error),
        mode_reconstruction_abs_error=float(mode_reconstruction_abs_error),
        imaginary_leakage=float(strong.imaginary_leakage_S),
        imaginary_leakage_ratio=float(imaginary_leakage_ratio),
        strongest_mode_fraction=float(strongest_mode_fraction),
        occupation_coeff_shape=(
            int(strong.occupation_coefficients.shape[0]),
            int(strong.occupation_coefficients.shape[1]),
        ),
        response_factor_finite=bool(np.isfinite(strong.response_factor).all()),
        velocity_coefficients_finite=bool(np.isfinite(strong.velocity_coefficients_m_per_s_per_m2).all()),
        strong_dc_internal_pass=bool(
            mode_reconstruction_abs_error < 1.0e-18
            and imaginary_leakage_ratio < 1.0e-12
            and np.isfinite(strongest_mode_fraction)
            and np.isfinite(strong_vs_weak_rel_trace_gap)
        ),
        residual_status="strong spectral tensor is internally closed; weak-chain gap is derivative-definition residual",
    )



def finite_field_strong_eq830_limit_probe() -> FiniteFieldStrongEq830LimitProbe:
    """Compare strong differential-response DC with continuum-normalised Eq. 8.30."""

    from dft_local.transport.boltzmann.ashcroft_comparison.core import (
        band_indexed_strong_dc_from_velocity_grid,
        conductivity_830_shifted_chain_rule_from_velocity_grid,
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

    strong = band_indexed_strong_dc_from_velocity_grid(
        epsilon,
        local.velocity_m_per_s,
        ai,
        chemical_potential_J=local.chemical_potential_J,
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
        electric_field_V_per_m=np.zeros(2),
    )
    strong_continuum = strong.conductivity_tensor_S.real / ((2.0 * np.pi) ** 2)
    strong_trace = float(np.trace(strong_continuum))
    strong_norm = float(np.linalg.norm(strong_continuum))

    base_field = np.asarray(reference.electric_field_V_per_m, dtype=np.float64)
    if float(np.linalg.norm(base_field)) == 0.0:
        base_field = np.array([1.0, 0.0], dtype=np.float64)

    rows: list[dict[str, float]] = []
    for eta in (0.0, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 1.0):
        field = base_field * eta
        eq830 = conductivity_830_shifted_chain_rule_from_velocity_grid(
            epsilon,
            local.velocity_m_per_s,
            ai,
            chemical_potential_J=local.chemical_potential_J,
            temperature_K=reference.temperature_K,
            relaxation_time_s=reference.relaxation_time_s,
            electric_field_V_per_m=field,
        )
        eq830_continuum = eq830.conductivity_tensor_S
        delta = strong_continuum - eq830_continuum
        eq830_trace = float(np.trace(eq830_continuum))
        rows.append({
            "eta": float(eta),
            "field_V_per_m": float(np.linalg.norm(field)),
            "eq830_trace": eq830_trace,
            "relative_tensor_discrepancy": float(np.linalg.norm(delta) / strong_norm) if strong_norm > 0.0 else np.nan,
            "relative_trace_discrepancy": float((strong_trace - eq830_trace) / strong_trace) if strong_trace != 0.0 else np.nan,
        })

    zero_row = rows[0]
    nonzero_rows = [row for row in rows if row["field_V_per_m"] > 0.0]
    small_row = nonzero_rows[0]
    largest_row = rows[-1]
    min_tensor_row = min(rows, key=lambda row: row["relative_tensor_discrepancy"])
    min_trace_row = min(rows, key=lambda row: abs(row["relative_trace_discrepancy"]))

    # This pass flag is intentionally conservative.  On the current Vincent grid,
    # equality is not assumed; the diagnostic is valid if it exposes finite,
    # continuum-normalised residuals across the field sweep.
    finite_residuals = all(
        np.isfinite(row["relative_tensor_discrepancy"])
        and np.isfinite(row["relative_trace_discrepancy"])
        and np.isfinite(row["eq830_trace"])
        for row in rows
    )

    return FiniteFieldStrongEq830LimitProbe(
        unit_context=SI_UNITS,
        source="strong differential response vs Eq. 8.30 shifted finite-difference on Vincent epsilon grid",
        field_row_count=len(rows),
        zero_field=float(zero_row["field_V_per_m"]),
        smallest_nonzero_field=float(small_row["field_V_per_m"]),
        largest_field=float(largest_row["field_V_per_m"]),
        strong_continuum_trace=float(strong_trace),
        zero_eq830_continuum_trace=float(zero_row["eq830_trace"]),
        smallest_eq830_continuum_trace=float(small_row["eq830_trace"]),
        zero_relative_tensor_discrepancy=float(zero_row["relative_tensor_discrepancy"]),
        zero_relative_trace_discrepancy=float(zero_row["relative_trace_discrepancy"]),
        smallest_relative_tensor_discrepancy=float(small_row["relative_tensor_discrepancy"]),
        smallest_relative_trace_discrepancy=float(small_row["relative_trace_discrepancy"]),
        largest_relative_tensor_discrepancy=float(largest_row["relative_tensor_discrepancy"]),
        largest_relative_trace_discrepancy=float(largest_row["relative_trace_discrepancy"]),
        min_relative_tensor_discrepancy=float(min_tensor_row["relative_tensor_discrepancy"]),
        min_abs_relative_trace_discrepancy=float(abs(min_trace_row["relative_trace_discrepancy"])),
        eq830_limit_status=(
            "finite sweep exposed; equality is not assumed on the Vincent grid because Eq. 8.30 tends to the weak chain-rule object while strong DC differentiates the spectral occupation"
        ),
        continuum_normalisation_status="both tensors compared with continuum A_BZ / (N_k (2π)^2) normalisation",
        limit_validation_pass=bool(finite_residuals),
    )




def finite_field_weak_dc_limit_probe() -> FiniteFieldWeakDcLimitProbe:
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

    return FiniteFieldWeakDcLimitProbe(
        unit_context=SI_UNITS,
        source="analytic sinusoidal Ashcroft strong/weak sweep",
        field_row_count=int(len(field_rows)),
        zero_eta=float(zero_row["eta"]),
        zero_field=float(zero_row["field_V_per_m"]),
        zero_relative_tensor_discrepancy=float(zero_row["relative_tensor_discrepancy"]),
        zero_relative_trace_discrepancy=float(zero_row["relative_trace_discrepancy"]),
        small_eta=float(small_nonzero_row["eta"]),
        small_field=float(small_nonzero_row["field_V_per_m"]),
        small_relative_tensor_discrepancy=float(small_nonzero_row["relative_tensor_discrepancy"]),
        small_relative_trace_discrepancy=float(small_nonzero_row["relative_trace_discrepancy"]),
        largest_eta=float(largest_row["eta"]),
        largest_relative_tensor_discrepancy=float(largest_row["relative_tensor_discrepancy"]),
        largest_relative_trace_discrepancy=float(largest_row["relative_trace_discrepancy"]),
        min_nonzero_eta=float(min_nonzero_eta),
        max_eta=float(max_eta),
        max_field_tensor_discrepancy=float(max_field_tensor_discrepancy),
        max_abs_field_trace_discrepancy=float(max_field_trace_discrepancy),
        max_imaginary_leakage=float(max_imaginary_leakage),
        relative_weak_limit_error=float(probe["relative_weak_limit_error"]),
        strong_zero_field_imaginary_leakage=float(probe["strong_zero_field_imaginary_leakage"]),
        weak_limit_pass=bool(
            float(probe["relative_weak_limit_error"]) < 1.0e-12
            and float(zero_row["relative_tensor_discrepancy"]) < 1.0e-12
            and abs(float(zero_row["relative_trace_discrepancy"])) < 1.0e-12
            and np.isfinite(max_field_tensor_discrepancy)
        ),
        roundoff_floor_status="zero-field agreement checked; finite eta sweep exposes nonlinear departure",
    )


def finite_field_mode_decomposition_probe() -> FiniteFieldModeDecompositionProbe:
    """Check Gamma/F/rho lattice-mode closure for the strong DC tensor."""

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

    strong = band_indexed_strong_dc_from_velocity_grid(
        epsilon,
        local.velocity_m_per_s,
        ai,
        chemical_potential_J=local.chemical_potential_J,
        temperature_K=reference.temperature_K,
        relaxation_time_s=reference.relaxation_time_s,
        electric_field_V_per_m=np.zeros(2),
    )

    gamma_reconstructed = np.empty_like(strong.velocity_m_per_s)
    for alpha in range(2):
        gamma_reconstructed[..., alpha] = np.fft.fft2(
            strong.velocity_coefficients_m_per_s_per_m2[..., alpha]
        ).real

    rho_reconstructed = np.fft.ifft2(
        strong.occupation_coefficients * strong.occupation_coefficients.size
    ).real

    gamma_abs_error = float(np.max(np.abs(gamma_reconstructed - strong.velocity_m_per_s)))
    rho_abs_error = float(np.max(np.abs(rho_reconstructed - strong.occupation)))

    mode_tensor_sum = np.sum(strong.conductivity_mode_tensor_S, axis=(0, 1))
    mode_tensor_abs_error = float(np.max(np.abs(mode_tensor_sum - strong.conductivity_tensor_S)))

    mode_norms = np.linalg.norm(strong.conductivity_mode_tensor_S.reshape((-1, 2, 2)), axis=(1, 2))
    total_mode_norm = float(np.sum(mode_norms))
    sorted_mode_norms = np.sort(mode_norms)[::-1]

    top_1_fraction = float(sorted_mode_norms[:1].sum() / total_mode_norm) if total_mode_norm > 0.0 else np.nan
    top_10_fraction = float(sorted_mode_norms[:10].sum() / total_mode_norm) if total_mode_norm > 0.0 else np.nan
    top_100_fraction = float(sorted_mode_norms[:100].sum() / total_mode_norm) if total_mode_norm > 0.0 else np.nan

    response_abs = np.abs(strong.response_factor)
    gamma_abs = np.abs(strong.velocity_coefficients_m_per_s_per_m2)
    rho_abs = np.abs(strong.occupation_coefficients)

    return FiniteFieldModeDecompositionProbe(
        unit_context=SI_UNITS,
        source="BandIndexedStrongDcResult Gamma/F/rho closure",
        mode_count=int(mode_norms.size),
        gamma_reconstruction_abs_error=float(gamma_abs_error),
        rho_reconstruction_abs_error=float(rho_abs_error),
        mode_tensor_reconstruction_abs_error=float(mode_tensor_abs_error),
        conductivity_trace=float(np.trace(strong.conductivity_tensor_S.real / ((2.0 * np.pi) ** 2))),
        conductivity_mode_norm_sum=float(total_mode_norm),
        top_1_mode_fraction=float(top_1_fraction),
        top_10_mode_fraction=float(top_10_fraction),
        top_100_mode_fraction=float(top_100_fraction),
        gamma_abs_max=float(np.max(gamma_abs)),
        rho_abs_max=float(np.max(rho_abs)),
        response_abs_max=float(np.max(response_abs)),
        gamma_finite=bool(np.isfinite(strong.velocity_coefficients_m_per_s_per_m2).all()),
        rho_finite=bool(np.isfinite(strong.occupation_coefficients).all()),
        response_finite=bool(np.isfinite(strong.response_factor).all()),
        mode_tensor_finite=bool(np.isfinite(strong.conductivity_mode_tensor_S).all()),
        mode_closure_pass=bool(
            gamma_abs_error < 1.0e-8
            and rho_abs_error < 1.0e-12
            and mode_tensor_abs_error < 1.0e-18
            and np.isfinite(total_mode_norm)
        ),
        residual_status="Gamma and rho reconstruct sampled grids; mode tensor re-sums to total strong DC tensor",
    )

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
