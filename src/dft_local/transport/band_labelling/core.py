"""Band labelling and indexing business logic.

This domain assigns local band indices to solved eigenvalues/eigenvectors.
The simplest labelling rule is energy ordering: at each sample, band index 0
means the lowest eigenvalue, band index 1 the next eigenvalue, and so on.

This is intentionally not continuation. It does not try to follow a band through
crossings. It only defines a local, per-sample ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ComplexArray = NDArray[np.complex128]

BandOrderKind = Literal["energy"]


@dataclass(frozen=True, slots=True)
class BandOrder:
    """Per-sample map from labelled-band index to original eigenpair index."""

    indices: IntArray
    kind: BandOrderKind = "energy"

    @property
    def nk(self) -> int:
        return int(self.indices.shape[0])

    @property
    def nbands(self) -> int:
        return int(self.indices.shape[1])


def _as_energy_matrix(energies: FloatArray) -> FloatArray:
    E = np.asarray(energies, dtype=np.float64)

    if E.ndim != 2:
        raise ValueError(f"energies must have shape (nk, nbands), got {E.shape}")

    return E


def energy_order(energies: FloatArray) -> BandOrder:
    """Return stable per-sample energy ordering.

    The result has shape ``(nk, nbands)``. Entry ``indices[s, b]`` is the
    original eigenpair index assigned to labelled band ``b`` at sample ``s``.

    Stable sorting is deliberate: exact degeneracies keep the solver's original
    order instead of adding extra arbitrary permutation noise.
    """

    E = _as_energy_matrix(energies)
    return BandOrder(indices=np.argsort(E, axis=1, kind="stable").astype(np.int64))


def apply_band_order(values: np.ndarray, order: BandOrder, *, band_axis: int = -1) -> np.ndarray:
    """Reorder an array with one band axis using a ``BandOrder``.

    ``values`` must have sample axis 0 and one band axis. The order is applied
    independently for each sample.

    Examples:
        energies:   values shape (nk, nbands), band_axis=1 or -1
        velocities: values shape (nk, dim, nbands), band_axis=2 or -1
        vectors:    values shape (nk, state_dim, nbands), band_axis=2 or -1
    """

    arr = np.asarray(values)

    if arr.shape[0] != order.nk:
        raise ValueError(f"sample count mismatch: {arr.shape[0]} != {order.nk}")

    axis = band_axis if band_axis >= 0 else arr.ndim + band_axis

    if axis <= 0 or axis >= arr.ndim:
        raise ValueError(f"band_axis must refer to a non-sample axis, got {band_axis}")

    if arr.shape[axis] != order.nbands:
        raise ValueError(f"band count mismatch: {arr.shape[axis]} != {order.nbands}")

    moved = np.moveaxis(arr, axis, -1)
    gathered = np.take_along_axis(
        moved,
        order.indices.reshape((order.nk,) + (1,) * (moved.ndim - 2) + (order.nbands,)),
        axis=-1,
    )
    return np.moveaxis(gathered, -1, axis)
