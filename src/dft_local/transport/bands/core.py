"""Band continuation and path API for the dft_local package.

This module now owns the pure band-matching business logic locally.

Band/path/region continuation logic is local to this module. Low-level shared
objects live in dft_local.core.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Literal, Self
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, eigvalsh
from scipy.optimize import linear_sum_assignment

MatchingStrategy = Literal["state_overlap", "energy_predict"]

from dft_local.core.local_problem import LocalProblem, SymbolPair
from dft_local.core.numerics import (
    FloatArray,
    IntArray,
    Units,
    eVag,
    freeze_array,
)

from dft_local.core.kernels import GdKernelArrays


@dataclass(frozen=True, slots=True)
class DegenerateGroupEvent:
    step: int
    bands: tuple[int, ...]
    energy_min: float
    energy_max: float
    gap: float
    subspace_score: float
    min_singular_value: float



@dataclass(frozen=True, slots=True)
class BandEvent:
    """Diagnostic event for a band crossing or ambiguous continuation match."""

    kind: str
    step: int
    band_a: int
    band_b: int
    x: float | None
    k1: float
    k2: float
    energy: float
    gap: float
    overlap_a: float | None
    overlap_b: float | None
    comment: str = ""


def metric_between(Sa: np.ndarray, Sb: np.ndarray, kind: str = "midpoint") -> np.ndarray:
    """Return a metric matrix used to compare neighbouring eigenvectors."""

    if kind == "previous":
        return Sa
    if kind == "current":
        return Sb
    if kind == "midpoint":
        return 0.5 * (Sa + Sb)
    if kind == "identity":
        return np.eye(Sa.shape[0], dtype=np.complex128)

    raise ValueError(f"Unknown metric: {kind}")


def fix_gauge_against_previous_arrays(
    U_prev: np.ndarray,
    S_metric: np.ndarray,
    U_curr: np.ndarray,
    *,
    eps: float = 1e-14,
) -> np.ndarray:
    """Phase-fix current eigenvectors by parallel continuation."""

    U = np.array(U_curr, copy=True)

    for j in range(U.shape[1]):
        z = U_prev[:, j].conj().T @ S_metric @ U[:, j]

        if abs(z) > eps:
            U[:, j] *= np.exp(-1j * np.angle(z))

    return U


def hungarian_order_from_costs(costs: np.ndarray) -> np.ndarray:
    """Return order such that column order[i] is assigned to row i."""

    row_ind, col_ind = linear_sum_assignment(costs)

    order = np.empty(costs.shape[0], dtype=np.int64)
    order[row_ind] = col_ind

    return order


def hungarian_order_from_scores(scores: np.ndarray) -> np.ndarray:
    """Return order that maximises score[row, column]."""

    return hungarian_order_from_costs(-np.asarray(scores, dtype=np.float64))


def predicted_energies_from_history(
    energies: np.ndarray,
    i: int,
    *,
    order: int = 2,
) -> np.ndarray:
    """Predict labelled energies at step i from stored labelled history."""

    if i <= 0:
        raise ValueError("Cannot predict first point")

    if order <= 0 or i == 1:
        return np.array(energies[i - 1], copy=True)

    if order == 1 or i == 2:
        return 2.0 * energies[i - 1] - energies[i - 2]

    return (
        3.0 * energies[i - 1]
        - 3.0 * energies[i - 2]
        + energies[i - 3]
    )


def match_via_energies(
    E_raw: np.ndarray,
    energies: np.ndarray,
    i: int,
    *,
    prediction_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Match raw eigenvalues to smooth predicted energy sheets."""

    E_raw = np.asarray(E_raw, dtype=np.float64)

    E_pred = predicted_energies_from_history(
        energies,
        i,
        order=prediction_order,
    )

    if E_pred.shape != E_raw.shape:
        raise ValueError(
            f"Energy prediction shape mismatch: E_pred={E_pred.shape}, "
            f"E_raw={E_raw.shape}"
        )

    costs = np.abs(E_pred[:, None] - E_raw[None, :]) ** 2
    order = hungarian_order_from_costs(costs)
    matched_costs = costs[np.arange(len(E_raw)), order]

    return order, matched_costs


def eigenvector_overlap_scores(
    U_prev: np.ndarray,
    S_metric: np.ndarray,
    U_curr: np.ndarray,
) -> np.ndarray:
    """Return squared overlap scores |U_prev^dagger S U_curr|^2."""

    return np.abs(U_prev.conj().T @ S_metric @ U_curr) ** 2


def align_degenerate_group_with_reorder(
    U_prev: np.ndarray,
    S_metric: np.ndarray,
    U_curr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Procrustes-align a degenerate current subspace to previous basis."""

    A = U_prev.conj().T @ S_metric @ U_curr
    L, singular_values, Mh = np.linalg.svd(A)

    rotation = Mh.conj().T @ L.conj().T
    U_aligned = U_curr @ rotation

    score_after = np.abs(U_prev.conj().T @ S_metric @ U_aligned) ** 2
    order_after = hungarian_order_from_scores(score_after)
    U_aligned = U_aligned[:, order_after]

    return U_aligned, singular_values, order_after


def energy_degenerate_groups(
    E: np.ndarray,
    gap_tol: float,
) -> list[np.ndarray]:
    """Find connected groups of bands degenerate within gap_tol."""

    E = np.asarray(E, dtype=np.float64)
    nbands = len(E)

    parent = np.arange(nbands, dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(nbands):
        for j in range(i + 1, nbands):
            if abs(float(E[i] - E[j])) <= gap_tol:
                union(i, j)

    groups_by_root: dict[int, list[int]] = {}

    for i in range(nbands):
        root = find(i)
        groups_by_root.setdefault(root, []).append(i)

    groups = [
        np.asarray(group, dtype=np.int64)
        for group in groups_by_root.values()
        if len(group) > 1
    ]
    groups.sort(key=lambda g: int(np.min(g)))

    return groups


def align_groups_and_fix_gauge(
    prev_U: np.ndarray,
    S_metric: np.ndarray,
    curr_U: np.ndarray,
    curr_E: np.ndarray,
    *,
    step: int,
    gap_tol: float,
    phase_eps: float = 1e-14,
) -> tuple[np.ndarray, tuple[DegenerateGroupEvent, ...]]:
    """Align degenerate subspaces and phase-fix isolated bands."""

    U = np.array(curr_U, copy=True)
    nbands = U.shape[1]

    groups = energy_degenerate_groups(curr_E, gap_tol=gap_tol)

    in_group = np.zeros(nbands, dtype=bool)
    events: list[DegenerateGroupEvent] = []

    for group in groups:
        U_aligned, singular_values, _order_after = align_degenerate_group_with_reorder(
            prev_U[:, group],
            S_metric,
            U[:, group],
        )

        U[:, group] = U_aligned
        in_group[group] = True

        group_E = curr_E[group]
        subspace_score = float(np.sum(singular_values**2))

        events.append(
            DegenerateGroupEvent(
                step=int(step),
                bands=tuple(int(i) for i in group),
                energy_min=float(np.min(group_E)),
                energy_max=float(np.max(group_E)),
                gap=float(np.max(group_E) - np.min(group_E)),
                subspace_score=subspace_score,
                min_singular_value=float(np.min(singular_values)),
            )
        )

    for j in range(nbands):
        if in_group[j]:
            continue

        z = prev_U[:, j].conj().T @ S_metric @ U[:, j]

        if abs(z) > phase_eps:
            U[:, j] *= np.exp(-1j * np.angle(z))

    return U, tuple(events)


def match_via_overlap(
    prev_U: np.ndarray,
    S_metric: np.ndarray,
    E_raw: np.ndarray,
    U_raw: np.ndarray,
    *,
    fix_gauge: bool = True,
    align_degenerate: bool = True,
    degeneracy_tol: float = 1e-7,
    step: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[DegenerateGroupEvent, ...]]:
    """Match current raw eigenvectors to previous continued vectors."""

    E_raw = np.asarray(E_raw, dtype=np.float64)
    nbands = len(E_raw)

    score = eigenvector_overlap_scores(prev_U, S_metric, U_raw)
    order = hungarian_order_from_scores(score)

    E = E_raw[order]
    U = U_raw[:, order]

    group_events: tuple[DegenerateGroupEvent, ...] = ()

    if fix_gauge:
        if align_degenerate:
            U, group_events = align_groups_and_fix_gauge(
                prev_U,
                S_metric,
                U,
                E,
                step=step,
                gap_tol=degeneracy_tol,
            )
        else:
            U = fix_gauge_against_previous_arrays(prev_U, S_metric, U)

    score_final = eigenvector_overlap_scores(prev_U, S_metric, U)
    matched_scores = np.clip(
        np.diag(score_final).real,
        0.0,
        1.0 + 1e-10,
    )

    if matched_scores.shape != (nbands,):
        raise ValueError("matched_scores shape mismatch")

    return E, U, matched_scores, group_events


def detect_energy_order_crossings_between_steps(
    path: LocalPath,
    step: int,
    E_prev: np.ndarray,
    E_curr: np.ndarray,
    matched_scores: np.ndarray,
    crossing_tol: float = 1e-8,
    ambiguity_overlap_tol: float = 0.5,
) -> list[BandEvent]:
    """Detect continued-band energy order changes between adjacent steps."""

    nbands = len(E_prev)
    out: list[BandEvent] = []

    for a in range(nbands):
        for b in range(a + 1, nbands):
            d0 = float(E_prev[a] - E_prev[b])
            d1 = float(E_curr[a] - E_curr[b])

            crossed = d0 * d1 < 0.0
            touched = abs(d0) <= crossing_tol or abs(d1) <= crossing_tol

            if not crossed and not touched:
                continue

            denom = abs(d0) + abs(d1)
            t = 0.5 if denom == 0.0 else abs(d0) / denom

            x = None
            if path.x is not None:
                x = float((1.0 - t) * path.x[step] + t * path.x[step + 1])

            k1 = float((1.0 - t) * path.k1[step] + t * path.k1[step + 1])
            k2 = float((1.0 - t) * path.k2[step] + t * path.k2[step + 1])

            Ea = float((1.0 - t) * E_prev[a] + t * E_curr[a])
            Eb = float((1.0 - t) * E_prev[b] + t * E_curr[b])

            overlap_a = float(matched_scores[a])
            overlap_b = float(matched_scores[b])
            overlap_min = min(overlap_a, overlap_b)

            kind = "energy_order_crossing"
            comment = "continued bands exchange energy order"

            if overlap_min < ambiguity_overlap_tol:
                kind = "ambiguous_match"
                comment = (
                    "continued bands exchange energy order, but matching overlap is weak"
                )

            out.append(
                BandEvent(
                    kind=kind,
                    step=int(step),
                    band_a=int(a),
                    band_b=int(b),
                    x=x,
                    k1=k1,
                    k2=k2,
                    energy=0.5 * (Ea + Eb),
                    gap=abs(Ea - Eb),
                    overlap_a=overlap_a,
                    overlap_b=overlap_b,
                    comment=comment,
                )
            )

    return out


@dataclass(frozen=True, slots=True)
class LocalPath:
    """Solved band continuation along a one-dimensional path in irrep space."""

    KH: GdKernelArrays
    KS: GdKernelArrays
    k1: np.ndarray
    k2: np.ndarray
    units: Units
    x: np.ndarray | None = None
    labels: tuple[tuple[int, str], ...] = ()
    degenerate_group_events: tuple[DegenerateGroupEvent, ...] = ()
    band_events: tuple[BandEvent, ...] = ()
    name: str = ""

    energies: np.ndarray = field(default_factory=lambda: np.array([]))
    vectors: np.ndarray = field(default_factory=lambda: np.array([]))
    overlaps: np.ndarray = field(default_factory=lambda: np.array([]))

    matching_strategy: Literal[
        "state_overlap",
        "energy_predict",
    ] = "state_overlap"

    def __post_init__(self) -> None:
        if self.k1.shape != self.k2.shape:
            raise ValueError("k1 and k2 shape mismatch")
        if self.x is not None and self.x.shape != self.k1.shape:
            raise ValueError("x shape mismatch")

    @classmethod
    def from_points(
        cls,
        KH: GdKernelArrays,
        KS: GdKernelArrays,
        points: list[tuple[str, float, float]],
        units: Units,
        points_per_segment: int = 80,
        matching_strategy="energy_predict",
        name: str = "",
    ) -> Self:
        k1_parts = []
        k2_parts = []
        x_parts = []
        labels = []

        x_current = 0.0

        for seg_index, ((label_a, a1, a2), (label_b, b1, b2)) in enumerate(
            zip(points[:-1], points[1:])
        ):
            t = np.linspace(0.0, 1.0, points_per_segment, endpoint=False)
            if seg_index == len(points) - 2:
                t = np.linspace(0.0, 1.0, points_per_segment + 1, endpoint=True)

            seg_k1 = (1 - t) * a1 + t * b1
            seg_k2 = (1 - t) * a2 + t * b2

            dk = float(np.sqrt((b1 - a1) ** 2 + (b2 - a2) ** 2))
            seg_x = x_current + t * dk

            if seg_index == 0:
                labels.append((0, label_a))

            k1_parts.append(seg_k1)
            k2_parts.append(seg_k2)
            x_parts.append(seg_x)

            x_current += dk
            labels.append((sum(len(part) for part in k1_parts) - 1, label_b))

        return cls(
            KH=KH,
            KS=KS,
            k1=np.concatenate(k1_parts),
            k2=np.concatenate(k2_parts),
            x=np.concatenate(x_parts),
            labels=tuple(labels),
            name=name,
            units=units,
            matching_strategy=matching_strategy,
        )

    @classmethod
    def from_arrays(
        cls,
        KH: GdKernelArrays,
        KS: GdKernelArrays,
        k1: np.ndarray,
        k2: np.ndarray,
        units: Units,
        x: np.ndarray | None = None,
        labels: tuple[tuple[int, str], ...] = (),
        name: str = "",
        matching_strategy="energy_predict",
    ) -> Self:
        k1 = np.asarray(k1, dtype=np.float64)
        k2 = np.asarray(k2, dtype=np.float64)

        if k1.shape != k2.shape:
            raise ValueError("k1 and k2 shape mismatch")

        if x is None:
            dx = np.sqrt(np.diff(k1) ** 2 + np.diff(k2) ** 2)
            x = np.concatenate([[0.0], np.cumsum(dx)])
        else:
            x = np.asarray(x, dtype=np.float64)

        return cls(
            KH=KH,
            KS=KS,
            k1=freeze_array(k1),
            k2=freeze_array(k2),
            x=freeze_array(x),
            labels=labels,
            name=name,
            units=units,
            matching_strategy=matching_strategy,
        )

    def pair(self, i: int) -> SymbolPair:
        return SymbolPair(
            KH=self.KH,
            KS=self.KS,
            k1=float(self.k1[i]),
            k2=float(self.k2[i]),
            name=self.name,
        )

    def form(self, i: int) -> LocalProblem:
        return self.pair(i).form()

    def is_closed(self, tol: float = 1e-10) -> bool:
        return (
            abs(float(self.k1[0] - self.k1[-1])) < tol
            and abs(float(self.k2[0] - self.k2[-1])) < tol
        )

    def solve_continuation(
        self,
        symmetrise: bool = True,
        check_overlap: bool = True,
        overlap_tol: float = 1e-10,
        degeneracy_tol: float = 1e-7,
        align_degenerate: bool = True,
        metric: str = "midpoint",
        fix_gauge: bool = True,
        initial_EU: None | tuple[NDArray, NDArray] = None,
        energy_prediction_order: int = 2,
    ) -> Self:
        """Solve generalized eigenproblems along the path with continuation."""

        nk = len(self.k1)

        if nk == 0:
            raise ValueError("Cannot solve empty path")

        p0 = self.form(0)
        p0 = p0.symmetrised() if symmetrise else p0

        if check_overlap:
            p0.check_overlap_positive(tol=overlap_tol)

        if initial_EU is None:
            E0, U0 = eigh(p0.Hk, p0.Sk, eigvals_only=False)
        else:
            E0, U0 = initial_EU
            E0 = np.asarray(E0, dtype=np.float64)
            U0 = np.asarray(U0, dtype=np.complex128)

            if U0.ndim != 2:
                raise ValueError(f"initial U must be 2D, got {U0.shape}")

            if E0.shape != (U0.shape[1],):
                raise ValueError(
                    f"initial E/U mismatch: E shape={E0.shape}, U shape={U0.shape}"
                )

        nbands = len(E0)
        dim = U0.shape[0]

        energies = np.empty((nk, nbands), dtype=np.float64)
        vectors = np.empty((nk, dim, nbands), dtype=np.complex128)
        overlaps = np.empty((max(nk - 1, 0), nbands), dtype=np.float64)

        band_events: list[BandEvent] = []
        degenerate_group_events: list[DegenerateGroupEvent] = []

        energies[0] = E0
        vectors[0] = U0

        prev_U = U0
        prev_S = p0.Sk

        for i in range(1, nk):
            p_i = self.form(i)
            p_i = p_i.symmetrised() if symmetrise else p_i

            if check_overlap:
                p_i.check_overlap_positive(tol=overlap_tol)

            E_raw, U_raw = eigh(p_i.Hk, p_i.Sk, eigvals_only=False)

            if E_raw.shape != (nbands,):
                raise ValueError(
                    f"Band count changed at step {i}: "
                    f"got {E_raw.shape}, expected {(nbands,)}"
                )

            if U_raw.shape != (dim, nbands):
                raise ValueError(
                    f"Eigenvector shape changed at step {i}: "
                    f"got {U_raw.shape}, expected {(dim, nbands)}"
                )

            S_metric = metric_between(prev_S, p_i.Sk, kind=metric)

            match self.matching_strategy:
                case "energy_predict":
                    order, _matched_costs = match_via_energies(
                        E_raw,
                        energies,
                        i,
                        prediction_order=energy_prediction_order,
                    )

                    E = E_raw[order]
                    U = U_raw[:, order]

                    if fix_gauge:
                        U = fix_gauge_against_previous_arrays(prev_U, S_metric, U)

                    score_final = eigenvector_overlap_scores(prev_U, S_metric, U)
                    matched_scores = np.clip(
                        np.diag(score_final).real,
                        0.0,
                        1.0 + 1e-10,
                    )

                case "state_overlap":
                    E, U, matched_scores, group_events = match_via_overlap(
                        prev_U,
                        S_metric,
                        E_raw,
                        U_raw,
                        fix_gauge=fix_gauge,
                        align_degenerate=align_degenerate,
                        degeneracy_tol=degeneracy_tol,
                        step=i - 1,
                    )
                    degenerate_group_events.extend(group_events)

                case _:
                    raise ValueError(f"Unknown matching strategy: {self.matching_strategy}")

            overlaps[i - 1] = matched_scores

            band_events.extend(
                detect_energy_order_crossings_between_steps(
                    self,
                    step=i - 1,
                    E_prev=energies[i - 1],
                    E_curr=E,
                    matched_scores=matched_scores,
                )
            )

            energies[i] = E
            vectors[i] = U

            prev_U = U
            prev_S = p_i.Sk

        return replace(
            self,
            energies=freeze_array(energies),
            vectors=freeze_array(vectors),
            overlaps=freeze_array(overlaps),
            band_events=tuple(band_events),
            degenerate_group_events=tuple(degenerate_group_events),
        )

    local_path_continuity_headers = [
        "band",
        "min overlap",
        "mean overlap",
        "geom mean overlap",
        "log product",
        "product",
        "max |ΔE|",
        "max |Δ²E|",
    ]

    def local_path_continuity_rows(self: Self) -> list[list[object]]:
        if self.energies is None or self.vectors is None or self.overlaps is None:
            raise ValueError("Path must be solved with continuation")

        E = self.energies
        O = np.clip(self.overlaps, 1e-300, 1.0)

        nbands = E.shape[1]
        rows = []

        dE = np.diff(E, axis=0)
        ddE = np.diff(E, n=2, axis=0)

        for band in range(nbands):
            log_product = float(np.sum(np.log(O[:, band]))) if len(O) else 0.0
            product = float(np.exp(log_product)) if log_product > -700 else 0.0

            rows.append([
                band,
                float(np.min(O[:, band])) if len(O) else None,
                float(np.mean(O[:, band])) if len(O) else None,
                float(np.exp(np.mean(np.log(O[:, band])))) if len(O) else None,
                log_product,
                product,
                float(np.max(np.abs(dE[:, band]))) if len(dE) else None,
                float(np.max(np.abs(ddE[:, band]))) if len(ddE) else None,
            ])

        return rows

    def worst_continuation_steps(self, n: int = 10):
        if self.overlaps is None:
            raise ValueError("LocalPath has not been solved with continuation")

        rows = []

        for i in range(self.overlaps.shape[0]):
            for band in range(self.overlaps.shape[1]):
                rows.append({
                    "step": i,
                    "from_k": i,
                    "to_k": i + 1,
                    "band": band,
                    "overlap": float(self.overlaps[i, band]),
                    "x_from": float(self.x[i]) if self.x is not None else None,
                    "x_to": float(self.x[i + 1]) if self.x is not None else None,
                    "k1_from": float(self.k1[i]),
                    "k2_from": float(self.k2[i]),
                    "k1_to": float(self.k1[i + 1]),
                    "k2_to": float(self.k2[i + 1]),
                })

        rows.sort(key=lambda row: row["overlap"])
        return rows[:n]

    local_path_matrix_smoothness_headers = [
        "step",
        "x left",
        "x right",
        "k1 left",
        "k2 left",
        "k1 right",
        "k2 right",
        "||ΔH||",
        "rel ||ΔH||",
        "||ΔS||",
        "rel ||ΔS||",
        "min eig S",
        "max eig S",
        "cond S",
    ]

    def local_path_matrix_smoothness_rows(self: Self) -> list[list[object]]:
        nk = len(self.k1)

        if nk < 2:
            raise ValueError("Need at least two k-points")

        Hs = []
        Ss = []

        for i in range(nk):
            problem = self.form(i).symmetrised()
            Hs.append(problem.Hk)
            Ss.append(problem.Sk)

        rows = []

        for i in range(nk - 1):
            dH = Hs[i + 1] - Hs[i]
            dS = Ss[i + 1] - Ss[i]

            H_norm = np.linalg.norm(Hs[i])
            S_norm = np.linalg.norm(Ss[i])
            dH_norm = np.linalg.norm(dH)
            dS_norm = np.linalg.norm(dS)
            S_eigs = eigvalsh(Ss[i])

            rows.append([
                i,
                float(self.x[i]) if self.x is not None else None,
                float(self.x[i + 1]) if self.x is not None else None,
                float(self.k1[i]),
                float(self.k2[i]),
                float(self.k1[i + 1]),
                float(self.k2[i + 1]),
                dH_norm,
                dH_norm / H_norm if H_norm > 0 else None,
                dS_norm,
                dS_norm / S_norm if S_norm > 0 else None,
                float(np.min(S_eigs)),
                float(np.max(S_eigs)),
                float(np.max(S_eigs) / np.min(S_eigs)),
            ])

        return rows

    local_path_matrix_curvature_headers = [
        "step",
        "x",
        "k1",
        "k2",
        "||Δ²H||",
        "rel ||Δ²H||",
        "||Δ²S||",
        "rel ||Δ²S||",
    ]

    def local_path_matrix_curvature_rows(self: Self) -> list[list[object]]:
        nk = len(self.k1)

        if nk < 3:
            return []

        Hs = []
        Ss = []

        for i in range(nk):
            problem = self.form(i).symmetrised()
            Hs.append(problem.Hk)
            Ss.append(problem.Sk)

        rows = []

        for i in range(1, nk - 1):
            ddH = Hs[i + 1] - 2.0 * Hs[i] + Hs[i - 1]
            ddS = Ss[i + 1] - 2.0 * Ss[i] + Ss[i - 1]

            H_norm = np.linalg.norm(Hs[i])
            S_norm = np.linalg.norm(Ss[i])

            rows.append([
                i,
                float(self.x[i]) if self.x is not None else None,
                float(self.k1[i]),
                float(self.k2[i]),
                float(np.linalg.norm(ddH)),
                float(np.linalg.norm(ddH) / H_norm) if H_norm > 0 else None,
                float(np.linalg.norm(ddS)),
                float(np.linalg.norm(ddS) / S_norm) if S_norm > 0 else None,
            ])

        rows.sort(key=lambda row: row[4], reverse=True)
        return rows

    def diagnostics(self: Self) -> dict:
        E = self.energies

        if self.overlaps is None:
            continuation_diagnostics = None
        else:
            continuation_diagnostics = {
                "num_kpoints": int(len(self.k1)),
                "num_bands": int(self.energies.shape[1]),
                "min_overlap": float(np.min(self.overlaps)) if len(self.overlaps) else None,
                "mean_overlap": float(np.mean(self.overlaps)) if len(self.overlaps) else None,
                "median_overlap": float(np.median(self.overlaps)) if len(self.overlaps) else None,
                "energy_min": float(np.min(self.energies)),
                "energy_max": float(np.max(self.energies)),
            }

        return {
            "name": self.name,
            "num_kpoints": int(len(self.k1)),
            "num_bands": int(E.shape[1]),
            "energy_min": float(np.min(E)),
            "energy_max": float(np.max(E)),
            "x_min": float(np.min(self.x)) if self.x is not None else None,
            "x_max": float(np.max(self.x)) if self.x is not None else None,
            "labels": [{"index": int(i), "label": label} for i, label in self.labels],
            "matching_strategy": self.matching_strategy,
            "continuation": continuation_diagnostics,
            "num_degenerate_group_events": len(self.degenerate_group_events),
            "min_degenerate_subspace_singular_value": (
                None
                if not self.degenerate_group_events
                else float(
                    min(ev.min_singular_value for ev in self.degenerate_group_events)
                )
            ),
            "band_events": [asdict(event) for event in self.band_events],
            "degenerate_groups": [
                {
                    "step": ev.step,
                    "bands": ev.bands,
                    "energy_min": ev.energy_min,
                    "energy_max": ev.energy_max,
                    "gap": ev.gap,
                    "subspace_score": ev.subspace_score,
                    "min_singular_value": ev.min_singular_value,
                }
                for ev in self.degenerate_group_events
            ],
        }


def transverse_path_cost(
    E_pred: np.ndarray,
    E_curr: np.ndarray,
) -> np.ndarray:
    """Mean-square cost for matching whole v-path energy sheets."""

    nbands = E_pred.shape[1]
    cost = np.empty((nbands, nbands), dtype=np.float64)

    for a in range(nbands):
        for b in range(nbands):
            d = E_curr[:, b] - E_pred[:, a]
            cost[a, b] = float(np.mean(d * d))

    return cost


def candidate_energy_groups(
    E_pred: np.ndarray,
    E_curr: np.ndarray,
    group_gap_tol: float,
) -> list[list[int]]:
    """Find candidate band groups that may interact across paths."""

    nbands = E_pred.shape[1]
    parent = list(range(nbands))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a in range(nbands):
        for b in range(nbands):
            min_gap = float(np.min(np.abs(E_pred[:, a] - E_curr[:, b])))

            if min_gap < group_gap_tol:
                union(a, b)

    groups_by_root: dict[int, list[int]] = {}

    for a in range(nbands):
        groups_by_root.setdefault(find(a), []).append(a)

    return list(groups_by_root.values())


def grouped_hungarian_order_from_costs(
    cost: np.ndarray,
    E_pred: np.ndarray,
    E_curr: np.ndarray,
    group_gap_tol: float,
) -> np.ndarray:
    """Run Hungarian matching only inside energy-near candidate groups."""

    nbands = cost.shape[0]
    order = np.arange(nbands)

    groups = candidate_energy_groups(
        E_pred,
        E_curr,
        group_gap_tol=group_gap_tol,
    )

    for group in groups:
        if len(group) == 1:
            continue

        g = np.asarray(group, dtype=np.int64)
        sub_cost = cost[np.ix_(g, g)]
        sub_order = hungarian_order_from_costs(sub_cost)

        order[g] = g[sub_order]

    return order


def bz_hexagon_vertices() -> list[list[float]]:
    """Return central Brillouin-zone hexagon vertices in raw irrep coordinates."""

    p = np.pi
    return [
        [p, 0.0],
        [p, p],
        [0.0, p],
        [-p, 0.0],
        [-p, -p],
        [0.0, -p],
    ]


@dataclass(frozen=True, slots=True)
class LocalRegion:
    KH: GdKernelArrays
    KS: GdKernelArrays

    k1: FloatArray          # shape (nu, nv)
    k2: FloatArray          # shape (nu, nv)

    u: FloatArray           # shape (nu,)
    v: FloatArray           # shape (nv,)

    name: str = ""
    units: Units = eVag
    matching_strategy: MatchingStrategy = "energy_predict"

    energies: FloatArray | None = None      # shape (nu, nv, nbands)
    vectors: np.ndarray | None = None       # shape (nu, nv, dim, nbands)

    overlaps_u: FloatArray | None = None    # shape (nu-1, nbands)
    overlaps_v: FloatArray | None = None    # shape (nu, nv-1, nbands)

    transverse_overlaps_u: FloatArray | None = None
    transverse_orders_u: IntArray | None = None

    energy_rectification_orders: IntArray | None = None
    energy_rectification_costs: FloatArray | None = None

    @property
    def solved(self) -> bool:
        return self.energies is not None and self.vectors is not None

    @property
    def shape(self) -> tuple[int, int]:
        return self.k1.shape

    @property
    def nu(self) -> int:
        return self.k1.shape[0]

    @property
    def nv(self) -> int:
        return self.k1.shape[1]

    @classmethod
    def from_parallelogram(
        cls,
        KH: GdKernelArrays,
        KS: GdKernelArrays,
        *,
        origin: tuple[float, float],
        edge_u: tuple[float, float],
        edge_v: tuple[float, float],
        nu: int,
        nv: int,
        name: str = "",
        units: Units = eVag,
        matching_strategy: MatchingStrategy = "energy_predict",
    ) -> Self:
        if nu < 2 or nv < 2:
            raise ValueError("nu and nv must both be at least 2")

        u = np.linspace(0.0, 1.0, nu)
        v = np.linspace(0.0, 1.0, nv)

        origin = np.asarray(origin, dtype=np.float64)
        edge_u = np.asarray(edge_u, dtype=np.float64)
        edge_v = np.asarray(edge_v, dtype=np.float64)

        k1, k2 = (
            origin[:, None, None]
            + edge_u[:, None, None] * u[None, :, None]
            + edge_v[:, None, None] * v[None, None, :]
        )


        return cls(
            KH=KH,
            KS=KS,
            k1=freeze_array(k1),
            k2=freeze_array(k2),
            u=freeze_array(u),
            v=freeze_array(v),
            name=name,
            units=units,
            matching_strategy=matching_strategy,
        )


    def seed_edge_path(self, matching_strategy: MatchingStrategy | None = None) -> LocalPath:
        return LocalPath.from_arrays(
            self.KH,
            self.KS,
            self.k1[:, 0],
            self.k2[:, 0],
            x=self.u,
            labels=((0, "u=0"), (self.nu - 1, "u=1")),
            name=self.name + " seed edge",
            units=self.units,
            matching_strategy=self.matching_strategy if matching_strategy is None else matching_strategy,
        )


    def solve_seed_edge(
        self,
        symmetrise: bool = True,
        check_overlap: bool = True,
        overlap_tol: float = 1e-10,
        metric: str = "midpoint",
        fix_gauge: bool = True,
        degeneracy_tol: float = 1e-7,
        align_degenerate: bool = True,
        matching_strategy: MatchingStrategy | None = None,
        energy_prediction_order: int = 2,
    ) -> LocalPath:
        return self.seed_edge_path(matching_strategy=matching_strategy).solve_continuation(
            symmetrise=symmetrise,
            check_overlap=check_overlap,
            overlap_tol=overlap_tol,
            metric=metric,
            fix_gauge=fix_gauge,
            degeneracy_tol=degeneracy_tol,
            align_degenerate=align_degenerate,
            energy_prediction_order=energy_prediction_order,
        )

    def v_path(self, i: int, matching_strategy: MatchingStrategy | None = None) -> LocalPath:
        if i < 0 or i >= self.nu:
            raise IndexError(i)

        return LocalPath.from_arrays(
            self.KH,
            self.KS,
            self.k1[i, :],
            self.k2[i, :],
            x=self.v,
            labels=((0, f"u={i}, v=0"), (self.nv - 1, f"u={i}, v=1")),
            name=f"{self.name} v-path {i}",
            units=self.units,
            matching_strategy=self.matching_strategy if matching_strategy is None else matching_strategy,
        )


    def solve(
        self,
        symmetrise: bool = True,
        check_overlap: bool = True,
        overlap_tol: float = 1e-10,
        metric: str = "midpoint",
        fix_gauge: bool = True,
        degeneracy_tol: float = 1e-5,
        align_degenerate: bool = True,
        matching_strategy: MatchingStrategy | None = None,
        energy_prediction_order: int = 2,
    ) -> Self:
        strategy = self.matching_strategy if matching_strategy is None else matching_strategy

        seed = self.solve_seed_edge(
            symmetrise=symmetrise,
            check_overlap=check_overlap,
            overlap_tol=overlap_tol,
            metric=metric,
            fix_gauge=fix_gauge,
            degeneracy_tol=degeneracy_tol,
            align_degenerate=align_degenerate,
            matching_strategy=strategy,
            energy_prediction_order=energy_prediction_order,
        )

        if seed.energies is None or seed.vectors is None:
            raise ValueError("Seed edge did not solve")

        nbands = seed.energies.shape[1]
        dim = seed.vectors.shape[1]

        energies = np.empty((self.nu, self.nv, nbands), dtype=np.float64)
        vectors = np.empty((self.nu, self.nv, dim, nbands), dtype=np.complex128)

        overlaps_u = None
        if seed.overlaps is not None:
            overlaps_u = np.array(seed.overlaps, copy=True)

        overlaps_v = np.empty((self.nu, self.nv - 1, nbands), dtype=np.float64)

        # Fill v=0 from seed edge.
        energies[:, 0, :] = seed.energies
        vectors[:, 0, :, :] = seed.vectors

        for i in range(self.nu):
            path = self.v_path(i, matching_strategy=strategy)

            initial_EU = (
                energies[i, 0, :],
                vectors[i, 0, :, :],
            )

            solved = path.solve_continuation(
                symmetrise=symmetrise,
                check_overlap=check_overlap,
                overlap_tol=overlap_tol,
                degeneracy_tol=degeneracy_tol,
                align_degenerate=align_degenerate,
                metric=metric,
                fix_gauge=fix_gauge,
                initial_EU=initial_EU,
                energy_prediction_order=energy_prediction_order,
            )

            if solved.energies is None or solved.vectors is None:
                raise ValueError(f"v path {i} did not solve")

            energies[i, :, :] = solved.energies
            vectors[i, :, :, :] = solved.vectors

            if solved.overlaps is not None:
                overlaps_v[i, :, :] = solved.overlaps

        solved = replace(
            self,
            matching_strategy=strategy,
            energies=freeze_array(energies),
            vectors=freeze_array(vectors),
            overlaps_u=None if overlaps_u is None else freeze_array(overlaps_u),
            overlaps_v=freeze_array(overlaps_v),
        )

        transverse_overlaps_u = solved.region_transverse_overlap_scores()
        transverse_orders_u = solved.region_transverse_orders()

        return replace(
            solved,
            transverse_overlaps_u=transverse_overlaps_u,
            transverse_orders_u=transverse_orders_u,
        )

    def rectify_energy_surfaces(
        self,
        *,
        prediction_order: int = 1,
        group_gap_tol: float | None = None,
    ) -> Self:
        if not self.solved:
            raise ValueError("Cannot rectify unsolved LocalRegion")

        E0 = np.asarray(self.energies)
        U0 = np.asarray(self.vectors)

        E = np.array(E0, copy=True)
        U = np.array(U0, copy=True)

        nu, nv, nbands = E.shape

        path_orders = np.empty((max(nu - 1, 0), nbands), dtype=np.int64)
        path_costs = np.empty((max(nu - 1, 0), nbands), dtype=np.float64)

        for i in range(1, nu):
            E_curr = E0[i]      # raw path as solved by original method
            U_curr = U0[i]

            if prediction_order <= 0 or i == 1:
                E_pred = E[i - 1]
            else:
                E_pred = 2.0 * E[i - 1] - E[i - 2]

            cost = transverse_path_cost(E_pred, E_curr)

            if group_gap_tol is None:
                order = hungarian_order_from_costs(cost)
            else:
                order = grouped_hungarian_order_from_costs(
                    cost,
                    E_pred=E_pred,
                    E_curr=E_curr,
                    group_gap_tol=group_gap_tol,
                )

            E[i] = E_curr[:, order]
            U[i] = U_curr[:, :, order]

            path_orders[i - 1] = order
            path_costs[i - 1] = cost[np.arange(nbands), order]

        return replace(
            self,
            energies=freeze_array(E),
            vectors=freeze_array(U),
            energy_rectification_orders=freeze_array(path_orders),
            energy_rectification_costs=freeze_array(path_costs),
        )


    def region_transverse_overlap_scores(
        self: Self,
        metric: str = "midpoint",
    ) -> FloatArray:
        """
        Compare neighbouring v-paths at fixed v.

        Returns:
            scores_u[i, j, band] for transition
                (i,j) -> (i+1,j)

        Shape:
            (nu - 1, nv, nbands)

        This is not the same as overlaps_u.
        overlaps_u is only the seed edge v=0.
        This compares all neighbouring paths across u.
        """
        if self.energies is None or self.vectors is None:
            raise ValueError("self must be solved")

        nu, nv, nbands = self.energies.shape

        out = np.empty((nu - 1, nv, nbands), dtype=np.float64)

        for i in range(nu - 1):
            for j in range(nv):
                p_a = SymbolPair(
                    self.KH,
                    self.KS,
                    float(self.k1[i, j]),
                    float(self.k2[i, j]),
                    name=self.name,
                ).form().symmetrised()

                p_b = SymbolPair(
                    self.KH,
                    self.KS,
                    float(self.k1[i + 1, j]),
                    float(self.k2[i + 1, j]),
                    name=self.name,
                ).form().symmetrised()

                S_metric = metric_between(p_a.Sk, p_b.Sk, kind=metric)

                U_a = self.vectors[i, j]
                U_b = self.vectors[i + 1, j]

                score = eigenvector_overlap_scores(U_a, S_metric, U_b)

                # Diagonal means: same band label on neighbouring paths.
                out[i, j, :] = np.diag(score)

        return freeze_array(out)

    def region_transverse_orders(
        self: Self,
        metric: str = "midpoint",
    ) -> IntArray:
        """
        For each neighbouring path pair at fixed v, compute the Hungarian order.

        order_u[i,j,a] = b means:
            band a on path i best matches band b on path i+1

        Shape:
            (nu - 1, nv, nbands)
        """
        if self.energies is None or self.vectors is None:
            raise ValueError("self must be solved")

        nu, nv, nbands = self.energies.shape

        out = np.empty((nu - 1, nv, nbands), dtype=np.int64)

        for i in range(nu - 1):
            for j in range(nv):
                p_a = SymbolPair(
                    self.KH,
                    self.KS,
                    float(self.k1[i, j]),
                    float(self.k2[i, j]),
                    name=self.name,
                ).form().symmetrised()

                p_b = SymbolPair(
                    self.KH,
                    self.KS,
                    float(self.k1[i + 1, j]),
                    float(self.k2[i + 1, j]),
                    name=self.name,
                ).form().symmetrised()

                S_metric = metric_between(p_a.Sk, p_b.Sk, kind=metric)

                score = eigenvector_overlap_scores(
                    self.vectors[i, j],
                    S_metric,
                    self.vectors[i + 1, j],
                )

                out[i, j, :] = hungarian_order_from_scores(score)

        return freeze_array(out)

    def region_transverse_switch_mask(self: Self) -> NDArray[np.bool_]:
        orders = self.region_transverse_orders()
        nbands = orders.shape[2]
        identity = np.arange(nbands, dtype=np.int64)
        return freeze_array(np.any(orders != identity[None, None, :], axis=2))

    local_region_path_continuity_headers = [
        "u path",
        "band",
        "min overlap",
        "geom mean",
        "log product",
        "product",
        "n < .99",
        "n < .95",
        "n < .90",
        "n < .75",
        "n < .60",
        "worst damage",
        "worst fraction",
    ]

    def local_region_path_continuity_rows(self: Self) -> list[list[object]]:
        if self.energies is None or self.overlaps_v is None:
            raise ValueError("Region must be solved")

        rows = []

        for i in range(self.nu):
            E = self.energies[i]  # shape (nv, nbands)
            O = np.clip(self.overlaps_v[i], 1e-300, 1.0)


            dE = np.diff(E, axis=0)
            ddE = np.diff(E, n=2, axis=0)

            for b in range(E.shape[1]):
                damage = -np.log(O[:, b])
                total_damage = float(np.sum(damage))
                worst_damage = float(np.max(damage))
                worst_fraction = worst_damage / total_damage if total_damage > 0 else 0.0

                rows.append([
                    i,
                    b,
                    float(np.min(O[:, b])),
                    float(np.exp(np.mean(np.log(O[:, b])))),
                    -total_damage,
                    float(np.exp(-total_damage)),
                    int(np.sum(O[:, b] < 0.99)),
                    int(np.sum(O[:, b] < 0.95)),
                    int(np.sum(O[:, b] < 0.90)),
                    int(np.sum(O[:, b] < 0.75)),
                    int(np.sum(O[:, b] < 0.60)),
                    worst_damage,
                    worst_fraction,
                ])

        rows.sort(key=lambda r: r[2])  # worst min overlap first
        return rows

    def worst_transverse_region_matches(
        self: Self,
        n: int = 30,
    ) -> list[dict]:
        if self.transverse_overlaps_u is None:
            raise ValueError("self has no transverse diagnostics")

        rows = []
        scores = self.transverse_overlaps_u

        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                for band in range(scores.shape[2]):
                    rows.append({
                        "u_step": i,
                        "v_index": j,
                        "band": band,
                        "same_label_overlap": float(scores[i, j, band]),
                        "u_left": float(self.u[i]),
                        "u_right": float(self.u[i + 1]),
                        "v": float(self.v[j]),
                        "k1_left": float(self.k1[i, j]),
                        "k2_left": float(self.k2[i, j]),
                        "k1_right": float(self.k1[i + 1, j]),
                        "k2_right": float(self.k2[i + 1, j]),
                    })

        rows.sort(key=lambda r: r["same_label_overlap"])
        return rows[:n]

    def transverse_permutation_rows(self: Self, *, n: int = 50) -> list[list[object]]:
        if self.transverse_orders_u is None or self.transverse_overlaps_u is None:
            raise ValueError("self has no transverse diagnostics")

        orders = self.transverse_orders_u
        scores = self.transverse_overlaps_u
        nbands = orders.shape[2]
        identity = np.arange(nbands)

        rows = []

        for i in range(orders.shape[0]):
            for j in range(orders.shape[1]):
                order = orders[i, j]

                if np.all(order == identity):
                    continue

                rows.append([
                    i,
                    j,
                    float(self.u[i]),
                    float(self.v[j]),
                    " ".join(map(str, order.tolist())),
                    float(np.min(scores[i, j])),
                    float(np.mean(scores[i, j])),
                ])

        rows.sort(key=lambda r: r[5])
        return rows[:n]


    def diagnostics(self) -> dict:
        out = {
            "name": self.name,
            "shape": tuple(map(int, self.shape)),
            "nu": int(self.nu),
            "nv": int(self.nv),
            "solved": self.solved,
            "matching_strategy": self.matching_strategy,
        }

        if self.energies is not None:
            out.update({
                "num_bands": int(self.energies.shape[2]),
                "energy_min": float(np.min(self.energies)),
                "energy_max": float(np.max(self.energies)),
            })

        if self.overlaps_u is not None:
            out["overlaps_u"] = {
                "min": float(np.min(self.overlaps_u)),
                "mean": float(np.mean(self.overlaps_u)),
                "median": float(np.median(self.overlaps_u)),
            }

        if self.overlaps_v is not None:
            out["overlaps_v"] = {
                "min": float(np.min(self.overlaps_v)),
                "mean": float(np.mean(self.overlaps_v)),
                "median": float(np.median(self.overlaps_v)),
            }

        return out

    def payload(region: Self) -> dict:
        def central_bz_mask(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
            return (
                (np.abs(k1) <= np.pi + 1e-12)
                & (np.abs(k2) <= np.pi + 1e-12)
                & (np.abs(k1 - k2) <= np.pi + 1e-12)
            )

        transverse_badness = None
        if region.transverse_overlaps_u is not None:
            # shape (nu-1, nv)
            transverse_badness = (1.0 - np.min(region.transverse_overlaps_u, axis=2)).tolist()

        if region.energies is None:
            raise ValueError("Region must be solved")

        mask = central_bz_mask(region.k1, region.k2)

        return {
            "name": region.name,
            "nu": int(region.nu),
            "nv": int(region.nv),
            "k1": region.k1.tolist(),
            "k2": region.k2.tolist(),
            "mask": mask.tolist(),
            "energies": region.energies.tolist(),
            "bands": list(range(region.energies.shape[2])),
            "energy_min": float(np.min(region.energies)),
            "energy_max": float(np.max(region.energies)),
            "matching_strategy": region.matching_strategy,
            "bz_hexagon": bz_hexagon_vertices(),
            "transverse_badness_u": transverse_badness,
        }

    def local_problem_at(self: Self, i: int, j: int) -> LocalProblem:
        return SymbolPair(
            self.KH,
            self.KS,
            float(self.k1[i, j]),
            float(self.k2[i, j]),
            name=self.name,
        ).form().symmetrised()

    def region_transverse_score_matrix(
        self: Self,
        i: int,
        j: int,
        metric: str = "midpoint",
    ) -> np.ndarray:
        """
        Compare region point (i,j) with neighbouring path point (i+1,j).

        Returns:
            score[a,b] = |u_(i,j,a)^dagger S_mid u_(i+1,j,b)|^2
        """
        if self.vectors is None or self.energies is None:
            raise ValueError("Region must be solved")

        if i < 0 or i >= self.nu - 1:
            raise IndexError(i)
        if j < 0 or j >= self.nv:
            raise IndexError(j)

        p0 = self.local_problem_at(i, j)
        p1 = self.local_problem_at(i + 1, j)

        S_metric = metric_between(p0.Sk, p1.Sk, kind=metric)

        U0 = self.vectors[i, j]
        U1 = self.vectors[i + 1, j]

        return eigenvector_overlap_scores(U0, S_metric, U1)

    def region_transverse_nonalignment_matrix(
        self:Self,
        metric: str = "midpoint",
    ) -> np.ndarray:
        """
        Matrix over transverse neighbour pairs.

        badness[i,j] = 1 - min_b score[b,b]

        Shape:
            (nu - 1, nv)
        """
        if self.vectors is None or self.energies is None:
            raise ValueError("Region must be solved")

        badness = np.empty((self.nu - 1, self.nv), dtype=np.float64)

        for i in range(self.nu - 1):
            for j in range(self.nv):
                score = self.region_transverse_score_matrix(i, j, metric=metric)
                same_label = np.diag(score)
                badness[i, j] = 1.0 - float(np.min(same_label))

        return badness

    def region_transverse_order_matrix(
        self: Self,
        metric: str = "midpoint",
    ) -> np.ndarray:
        """
        order[i,j,a] = b means:
            band a at (i,j) best matches band b at (i+1,j)

        Shape:
            (nu - 1, nv, nbands)
        """
        if self.vectors is None or self.energies is None:
            raise ValueError("Region must be solved")

        nbands = self.energies.shape[2]
        orders = np.empty((self.nu - 1, self.nv, nbands), dtype=np.int64)

        for i in range(self.nu - 1):
            for j in range(self.nv):
                score = self.region_transverse_score_matrix(i, j, metric=metric)
                orders[i, j] = hungarian_order_from_scores(score)

        return orders

    def region_transverse_worst_rows(
        self: Self,
        *,
        n: int = 100,
        metric: str = "midpoint",
    ) -> list[list[object]]:
        badness = self.region_transverse_nonalignment_matrix(metric=metric)
        orders = self.region_transverse_order_matrix(metric=metric)

        nbands = self.energies.shape[2]
        identity = np.arange(nbands, dtype=np.int64)

        rows = []
        def permutation_cycles(order: np.ndarray) -> str:
            order = np.asarray(order, dtype=np.int64)
            seen = np.zeros(len(order), dtype=bool)
            cycles = []

            for i in range(len(order)):
                if seen[i] or order[i] == i:
                    seen[i] = True
                    continue

                cyc = []
                j = i
                while not seen[j]:
                    seen[j] = True
                    cyc.append(j)
                    j = int(order[j])

                if len(cyc) > 1:
                    cycles.append("(" + " ".join(map(str, cyc)) + ")")

            return " ".join(cycles) if cycles else "identity"

        for i in range(self.nu - 1):
            for j in range(self.nv):
                score = self.region_transverse_score_matrix(i, j, metric=metric)
                diag = np.diag(score)
                order = orders[i, j]
                is_permutation = not np.all(order == identity)

                rows.append([
                    i,
                    j,
                    float(self.u[i]),
                    float(self.u[i + 1]),
                    float(self.v[j]),
                    float(badness[i, j]),
                    float(np.min(diag)),
                    float(np.mean(diag)),
                    permutation_cycles(order),
                    bool(is_permutation),
                ])

        rows.sort(key=lambda r: r[5], reverse=True)
        return rows[:n]



__all__ = [
    "BandEvent",
    "bz_hexagon_vertices",
    "grouped_hungarian_order_from_costs",
    "candidate_energy_groups",
    "transverse_path_cost",
    "DegenerateGroupEvent",
    "LocalPath",
    "LocalProblem",
    "LocalRegion",
    "SymbolPair",
    "align_degenerate_group_with_reorder",
    "align_groups_and_fix_gauge",
    "detect_energy_order_crossings_between_steps",
    "eigenvector_overlap_scores",
    "energy_degenerate_groups",
    "fix_gauge_against_previous_arrays",
    "hungarian_order_from_costs",
    "hungarian_order_from_scores",
    "match_via_energies",
    "match_via_overlap",
    "metric_between",
    "predicted_energies_from_history",
]
