"""Band-labelled and group-resolved Boltzmann conductivity routines.

First target: band-labelled weak-field/DC conductivity.

The lattice-index formulas in the thesis are Fourier-resolved versions of the
standard compact band-indexed Boltzmann formula.  Therefore the first invariant
implemented here is the compact per-band tensor.  The lattice-index inverse
reconstruction will be validated against this object band by band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class BandResolvedConductivity:
    """Per-band conductivity tensors.

    Attributes:
        sigma_band_k:
            Shape ``(nk, nbands, dim, dim)``.  Contribution of each sample and
            each labelled band.
        sigma_band:
            Shape ``(nbands, dim, dim)``.  Integrated tensor for each labelled
            band.
        sigma:
            Shape ``(dim, dim)``.  Sum over bands.
    """

    sigma_band_k: ComplexArray
    sigma_band: ComplexArray
    sigma: ComplexArray


def band_resolved_compact_conductivity(
    *,
    velocities: FloatArray,
    weights: ComplexArray,
    physical_k_weights: FloatArray,
) -> BandResolvedConductivity:
    """Build compact per-band conductivity from existing solved arrays.

    This mirrors the current compact Boltzmann assembly, but preserves the band
    axis instead of summing it immediately.

    Args:
        velocities:
            Shape ``(nk, dim, nbands)``.
        weights:
            Shape ``(nk, nbands)``.  This is the scalar band response weight,
            for example ``q^2 tau [-df/dE]`` for DC, or the AC Drude version.
        physical_k_weights:
            Shape ``(nk,)``.  The k-space integration weight for each sample.
    """

    v = np.asarray(velocities, dtype=np.float64)
    w = np.asarray(weights, dtype=np.complex128)
    k_w = np.asarray(physical_k_weights, dtype=np.float64)

    if v.ndim != 3:
        raise ValueError(f"velocities must have shape (nk, dim, nbands), got {v.shape}")

    nk, dim, nbands = v.shape

    if w.shape != (nk, nbands):
        raise ValueError(f"weights shape mismatch: {w.shape} != {(nk, nbands)}")

    if k_w.shape != (nk,):
        raise ValueError(f"physical_k_weights shape mismatch: {k_w.shape} != {(nk,)}")

    sigma_band_k = np.einsum("san,sbn,sn,s->snab", v, v, w, k_w).astype(np.complex128)
    sigma_band = np.sum(sigma_band_k, axis=0)
    sigma = np.sum(sigma_band, axis=0)

    return BandResolvedConductivity(
        sigma_band_k=sigma_band_k,
        sigma_band=sigma_band,
        sigma=sigma,
    )
