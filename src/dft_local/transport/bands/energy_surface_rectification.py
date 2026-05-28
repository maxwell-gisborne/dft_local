"""
Energy surface rectification.

Second-pass labelling for a solved LocalRegion.  The pass assumes each
individual v-path is already reasonably continued, then walks across the
parallel paths and applies whole-path band permutations to reduce transverse
energy roughness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class EnergyRectificationReport:
    """Diagnostics from the energy-surface rectification pass."""

    orders: IntArray
    costs: FloatArray
    accepted: NDArray[np.bool_]
    before: dict[str, float]
    after: dict[str, float]


def hungarian_order_from_costs(costs: FloatArray) -> IntArray:
    """Return order where order[row] is the chosen column for a cost matrix."""

    row_ind, col_ind = linear_sum_assignment(costs)
    order = np.empty(costs.shape[0], dtype=np.int64)
    order[row_ind] = col_ind
    return order


def transverse_energy_prediction(
    E_rect: FloatArray,
    i: int,
    *,
    prediction_order: int = 1,
) -> FloatArray:
    """Predict path i from already-rectified previous paths."""

    if i <= 0:
        raise ValueError("Cannot predict the first path")

    if prediction_order <= 0 or i == 1:
        return np.array(E_rect[i - 1], copy=True)

    return 2.0 * E_rect[i - 1] - E_rect[i - 2]


def transverse_path_cost(E_pred: FloatArray, E_curr: FloatArray) -> FloatArray:
    """Mean-square cost of assigning current path bands to predicted sheets."""

    if E_pred.ndim != 2 or E_curr.ndim != 2:
        raise ValueError("E_pred and E_curr must both have shape (nv, nbands)")

    if E_pred.shape != E_curr.shape:
        raise ValueError(
            f"shape mismatch: E_pred={E_pred.shape}, E_curr={E_curr.shape}"
        )

    _nv, nbands = E_pred.shape
    costs = np.empty((nbands, nbands), dtype=np.float64)

    for a in range(nbands):
        for b in range(nbands):
            d = E_curr[:, b] - E_pred[:, a]
            costs[a, b] = float(np.mean(d * d))

    return costs


def energy_surface_roughness(E: FloatArray) -> dict[str, float]:
    """Return simple curvature roughness diagnostics for labelled surfaces."""

    if E.ndim != 3:
        raise ValueError("E must have shape (nu, nv, nbands)")

    out: dict[str, float] = {}

    if E.shape[0] >= 3:
        du2 = E[2:, :, :] - 2.0 * E[1:-1, :, :] + E[:-2, :, :]
        out["max_abs_du2"] = float(np.max(np.abs(du2)))
        out["mean_abs_du2"] = float(np.mean(np.abs(du2)))
        out["rms_du2"] = float(np.sqrt(np.mean(du2 * du2)))
    else:
        out["max_abs_du2"] = out["mean_abs_du2"] = out["rms_du2"] = 0.0

    if E.shape[1] >= 3:
        dv2 = E[:, 2:, :] - 2.0 * E[:, 1:-1, :] + E[:, :-2, :]
        out["max_abs_dv2"] = float(np.max(np.abs(dv2)))
        out["mean_abs_dv2"] = float(np.mean(np.abs(dv2)))
        out["rms_dv2"] = float(np.sqrt(np.mean(dv2 * dv2)))
    else:
        out["max_abs_dv2"] = out["mean_abs_dv2"] = out["rms_dv2"] = 0.0

    if E.shape[0] >= 2 and E.shape[1] >= 2:
        duv = E[1:, 1:, :] - E[1:, :-1, :] - E[:-1, 1:, :] + E[:-1, :-1, :]
        out["max_abs_duv"] = float(np.max(np.abs(duv)))
        out["mean_abs_duv"] = float(np.mean(np.abs(duv)))
        out["rms_duv"] = float(np.sqrt(np.mean(duv * duv)))
    else:
        out["max_abs_duv"] = out["mean_abs_duv"] = out["rms_duv"] = 0.0

    return out


def _apply_band_order_to_path_vectors(U_curr: np.ndarray, order: IntArray) -> np.ndarray:
    """Apply one path band order to vectors shaped (nv, dim, nbands)."""

    if U_curr.ndim != 3:
        raise ValueError("path vectors must have shape (nv, dim, nbands)")

    return U_curr[:, :, order]


def rectify_energy_arrays_across_u(
    energies: FloatArray,
    vectors: np.ndarray | None = None,
    *,
    prediction_order: int = 1,
    accept_ratio: float = 0.98,
) -> tuple[FloatArray, np.ndarray | None, EnergyRectificationReport]:
    """Rectify energy surfaces by whole-v-path band permutations."""

    E0 = np.asarray(energies, dtype=np.float64)

    if E0.ndim != 3:
        raise ValueError("energies must have shape (nu, nv, nbands)")

    nu, nv, nbands = E0.shape

    if vectors is not None:
        U0 = np.asarray(vectors)
        if U0.ndim != 4:
            raise ValueError("vectors must have shape (nu, nv, dim, nbands)")
        if U0.shape[0] != nu or U0.shape[1] != nv or U0.shape[3] != nbands:
            raise ValueError(
                f"vectors shape {U0.shape} incompatible with energies {E0.shape}"
            )
    else:
        U0 = None

    before = energy_surface_roughness(E0)

    E = np.array(E0, copy=True)
    U = None if U0 is None else np.array(U0, copy=True)

    orders = np.empty((max(nu - 1, 0), nbands), dtype=np.int64)
    costs_out = np.empty((max(nu - 1, 0), nbands), dtype=np.float64)
    accepted = np.zeros((max(nu - 1, 0),), dtype=np.bool_)

    identity = np.arange(nbands, dtype=np.int64)

    for i in range(1, nu):
        E_curr = E0[i]
        E_pred = transverse_energy_prediction(
            E,
            i,
            prediction_order=prediction_order,
        )

        costs = transverse_path_cost(E_pred, E_curr)
        order = hungarian_order_from_costs(costs)

        identity_cost = float(np.sum(costs[np.arange(nbands), identity]))
        proposed_cost = float(np.sum(costs[np.arange(nbands), order]))

        if (not np.array_equal(order, identity)) and (
            proposed_cost < accept_ratio * identity_cost
        ):
            use_order = order
            accepted[i - 1] = True
        else:
            use_order = identity
            accepted[i - 1] = False

        E[i] = E_curr[:, use_order]
        if U is not None:
            U[i] = _apply_band_order_to_path_vectors(U0[i], use_order)

        orders[i - 1] = use_order
        costs_out[i - 1] = costs[np.arange(nbands), use_order]

    after = energy_surface_roughness(E)

    report = EnergyRectificationReport(
        orders=orders,
        costs=costs_out,
        accepted=accepted,
        before=before,
        after=after,
    )

    return E, U, report


def rectify_local_region_energy_surfaces(
    region: Any,
    *,
    prediction_order: int = 1,
    accept_ratio: float = 0.98,
    freeze_array=None,
):
    """Return a new LocalRegion-like dataclass with rectified energies."""

    from dataclasses import fields, replace

    if getattr(region, "energies", None) is None:
        raise ValueError("Cannot rectify an unsolved LocalRegion: energies is None")

    E, U, report = rectify_energy_arrays_across_u(
        np.asarray(region.energies),
        None if getattr(region, "vectors", None) is None else np.asarray(region.vectors),
        prediction_order=prediction_order,
        accept_ratio=accept_ratio,
    )

    freezer = freeze_array if freeze_array is not None else (lambda x: x)

    updates: dict[str, Any] = {"energies": freezer(E)}

    if U is not None:
        updates["vectors"] = freezer(U)

    field_names = {f.name for f in fields(region)}

    optional_updates = {
        "energy_rectification_orders": freezer(report.orders),
        "energy_rectification_costs": freezer(report.costs),
        "energy_rectification_accepted": freezer(report.accepted),
    }

    for name, value in optional_updates.items():
        if name in field_names:
            updates[name] = value

    return replace(region, **updates)
