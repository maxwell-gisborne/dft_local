"""Kernel arrays for the dft_local package.

This module owns `GdKernelArrays`, the homogeneous group-labelled block-kernel
object used by local symbols, band continuation, and Boltzmann conductivity.

Kernel construction uses local sparse and geometry helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.sparse import bsr_matrix

from dft_local.core.numerics import BlockArray, IntArray, freeze_array

from dft_local.core.geometry import EdgeGroupLabels
from dft_local.core.sparse import block_row_raw


def gd_inverse_label(m: int, n: int, eps: int) -> tuple[int, int, int]:
    sign = -1 if eps else 1
    return (-sign * m, -sign * n, eps)


def relative_labels_for_row(labels: EdgeGroupLabels, atom: int, atoms_b: IntArray):
    """
    Vectorised relative labels for a BSR block row.

    Returns arrays h_m, h_n, h_eps such that

        h = g_atom^-1 g_b

    for each b in atoms_b.
    """

    '''
    For atoms a and b:

    g_a = (r_a, eps_a)
    g_b = (r_b, eps_b)

        with group law

    (r,e)(r',e') = (r + (-1)^e r', e xor e')

        Then

    h = g_a^-1 g_b

        is:

    h.r = (-1)^eps_a * (r_b - r_a)
    h.eps = eps_a xor eps_b
    ''' 

    atom = int(atom)
    atoms_b = np.asarray(atoms_b, dtype=np.int64)

    sign = 1 if labels.eps[atom] == 0 else -1

    h_m = sign * (labels.m[atoms_b] - labels.m[atom])
    h_n = sign * (labels.n[atoms_b] - labels.n[atom])
    h_eps = labels.eps[atom] ^ labels.eps[atoms_b]

    return h_m, h_n, h_eps


@dataclass(frozen=True, slots=True)
class GdKernelArrays:
    h_m: IntArray              # shape (N,)
    h_n: IntArray              # shape (N,)
    h_eps: IntArray            # shape (N,)
    blocks: BlockArray         # shape (N, q, q)
    matrix_name: str = ""

    @classmethod
    def from_anchored(
        cls,
        M: bsr_matrix,
        labels: EdgeGroupLabels,
        anchor_atom: int | None = None,
        matrix_name: str = "",
        copy_blocks: bool = False,
    ):
        if anchor_atom is None:
            anchor_atom = labels.anchor_atom

        atoms_b, blocks = block_row_raw(M, anchor_atom)

        h_m, h_n, h_eps = relative_labels_for_row(labels, anchor_atom, atoms_b)

        if copy_blocks:
            blocks = freeze_array(np.array(blocks, copy=True))

        return cls(
            h_m=freeze_array(np.asarray(h_m, dtype=np.int64)),
            h_n=freeze_array(np.asarray(h_n, dtype=np.int64)),
            h_eps=freeze_array(np.asarray(h_eps, dtype=np.int64)),
            blocks=np.asarray(blocks),
            matrix_name=matrix_name,
        )

    @classmethod
    def from_average(
        cls,
        M: bsr_matrix,
        labels: EdgeGroupLabels,
        anchors: IntArray | None = None,
        matrix_name: str = "",
    ) -> Self:
        """
        Average block rows over anchors to form an effective homogeneous kernel.

            K(h) = average_a M[a, a h]

        where h = g_a^-1 g_b.
        """
        if anchors is None:
            anchors = labels.geometry.core_bulk_atoms()

        anchors = np.asarray(anchors, dtype=np.int64)

        if len(anchors) == 0:
            raise ValueError("No anchors supplied for averaged kernel")

        sums: dict[tuple[int, int, int], np.ndarray] = {}
        counts: dict[tuple[int, int, int], int] = {}

        for a in anchors:
            a = int(a)
            atoms_b, blocks = block_row_raw(M, a)

            h_m, h_n, h_eps = relative_labels_for_row(labels, a, atoms_b)

            for hm, hn, he, block in zip(h_m, h_n, h_eps, blocks):
                key = (int(hm), int(hn), int(he))

                if key not in sums:
                    sums[key] = np.zeros_like(block, dtype=np.result_type(block, np.float64))
                    counts[key] = 0

                sums[key] += block
                counts[key] += 1

        # Deterministic ordering: eps, then m, then n, or choose m,n,eps.
        keys = sorted(sums.keys(), key=lambda x: (x[2], x[0], x[1]))

        h_m = np.asarray([k[0] for k in keys], dtype=np.int64)
        h_n = np.asarray([k[1] for k in keys], dtype=np.int64)
        h_eps = np.asarray([k[2] for k in keys], dtype=np.int64)

        blocks = np.asarray(
            [sums[k] / counts[k] for k in keys],
            dtype=np.result_type(M.data, np.float64),
        )

        freeze_array(h_m)
        freeze_array(h_n)
        freeze_array(h_eps)
        freeze_array(blocks)

        return cls(
            h_m=h_m,
            h_n=h_n,
            h_eps=h_eps,
            blocks=blocks,
            matrix_name=matrix_name,
        )

    def __post_init__(self) -> None:
        N = len(self.h_m)

        if self.h_n.shape != (N,):
            raise ValueError("h_n shape mismatch")
        if self.h_eps.shape != (N,):
            raise ValueError("h_eps shape mismatch")
        if self.blocks.ndim != 3:
            raise ValueError(f"blocks must have shape (N,q,q), got {self.blocks.shape}")
        if self.blocks.shape[0] != N:
            raise ValueError("blocks/support length mismatch")
        if self.blocks.shape[1] != self.blocks.shape[2]:
            raise ValueError("blocks must be square")
        if not np.all((self.h_eps == 0) | (self.h_eps == 1)):
            raise ValueError("h_eps must contain only 0 and 1")
        if not np.all(np.isfinite(self.blocks)):
            raise ValueError("blocks contain NaN or Inf")

    def star_symmetrised(
        self,
        missing: str = "zero",
        matrix_name: str | None = None,
    ) -> Self:
        """
        Return kernel satisfying

            K(h^-1) = K(h)^†

        by replacing

            K(h) -> 1/2 [ K(h) + K(h^-1)^† ]

        If missing='zero', absent inverse blocks are treated as zero.
        If missing='keep', absent inverse blocks are kept as 1/2 K(h), and the
        inverse support is added as 1/2 K(h)^†.
        """
        if missing not in ("zero", "keep"):
            raise ValueError("missing must be 'zero' or 'keep'")

        q = self.blocksize

        by_key: dict[tuple[int, int, int], np.ndarray] = {
            (int(m), int(n), int(e)): block
            for m, n, e, block in zip(self.h_m, self.h_n, self.h_eps, self.blocks)
        }

        all_keys = set(by_key)

        if missing == "keep":
            for key in list(by_key):
                all_keys.add(gd_inverse_label(*key))

        out: dict[tuple[int, int, int], np.ndarray] = {}

        zero = np.zeros((q, q), dtype=np.result_type(self.blocks, np.complex128))

        for key in all_keys:
            inv = gd_inverse_label(*key)

            K_h = by_key.get(key, zero)
            K_inv = by_key.get(inv, zero)

            out[key] = 0.5 * (K_h + K_inv.conj().T)

        keys = sorted(out.keys(), key=lambda x: (x[2], x[0], x[1]))

        h_m = np.asarray([k[0] for k in keys], dtype=np.int64)
        h_n = np.asarray([k[1] for k in keys], dtype=np.int64)
        h_eps = np.asarray([k[2] for k in keys], dtype=np.int64)
        blocks = np.asarray([out[k] for k in keys], dtype=np.complex128)

        freeze_array(h_m)
        freeze_array(h_n)
        freeze_array(h_eps)
        freeze_array(blocks)

        return type(self)(
            h_m=h_m,
            h_n=h_n,
            h_eps=h_eps,
            blocks=blocks,
            matrix_name=self.matrix_name + " star-sym" if matrix_name is None else matrix_name,
        )

    @property
    def support_size(self) -> int:
        return len(self.h_m)


    @property
    def blocksize(self) -> int:
        return self.blocks.shape[1]


    def symbol_generic(kernel:Self,  k1: float, k2:float) -> np.ndarray:
        """
        Generic 2D irrep symbol for G_d.

        Returns dense matrix with shape (2*q, 2*q).
        """
        K = kernel.blocks
        q = K.shape[1]

        theta = k1 * kernel.h_m + k2 * kernel.h_n
        phase = np.exp(1j * theta)

        out = np.zeros((2 * q, 2 * q), dtype=np.complex128)

        even = kernel.h_eps == 0
        odd = kernel.h_eps == 1

        # eps = 0:
        # Omega = [[phase, 0],
        #          [0, conj(phase)]]
        if np.any(even):
            K0 = K[even]
            p0 = phase[even]

            out[0:q, 0:q] += np.einsum("h,hij->ij", p0, K0)
            out[q:2*q, q:2*q] += np.einsum("h,hij->ij", np.conj(p0), K0)

        # eps = 1:
        # Omega = [[0, conj(phase)],
        #          [phase, 0]]
        if np.any(odd):
            K1 = K[odd]
            p1 = phase[odd]

            out[0:q, q:2*q] += np.einsum("h,hij->ij", np.conj(p1), K1)
            out[q:2*q, 0:q] += np.einsum("h,hij->ij", p1, K1)

        return out

    def symbol_fixed(self, k1: float, k2: float, sigma: int) -> np.ndarray:
        """
        One-dimensional fixed-point irrep symbol.

        Valid at k in {0, pi}^2.
        """
        if sigma not in (-1, 1):
            raise ValueError(f"sigma must be ±1, got {sigma}")

        theta = k1 * self.h_m + k2 * self.h_n
        coeff = np.exp(1j * theta) * (sigma ** self.h_eps)

        return np.einsum("h,hij->ij", coeff, self.blocks).astype(np.complex128)

    def diagnostics(self) -> dict:
        norms = np.linalg.norm(self.blocks, axis=(1, 2))

        return {
            "matrix_name": self.matrix_name,
            "support_size": self.support_size,
            "blocksize": self.blocksize,
            "num_even": int(np.sum(self.h_eps == 0)),
            "num_odd": int(np.sum(self.h_eps == 1)),
            "norm_min": float(np.min(norms)) if len(norms) else None,
            "norm_max": float(np.max(norms)) if len(norms) else None,
            "norm_median": float(np.median(norms)) if len(norms) else None,
        }

    def star_defect(self) -> dict:
        """
        Measure failure of K(h^-1) = K(h)^†.
        """
        by_key: dict[tuple[int, int, int], np.ndarray] = {
            (int(m), int(n), int(e)): block
            for m, n, e, block in zip(self.h_m, self.h_n, self.h_eps, self.blocks)
        }

        defects = []

        seen = set()

        for key, K_h in by_key.items():
            if key in seen:
                continue

            inv = gd_inverse_label(*key)
            seen.add(key)
            seen.add(inv)

            K_inv = by_key.get(inv)

            if K_inv is None:
                defects.append((key, None, np.inf, np.linalg.norm(K_h)))
                continue

            err = np.linalg.norm(K_h - K_inv.conj().T)
            scale = max(np.linalg.norm(K_h), np.linalg.norm(K_inv), 1.0)
            defects.append((key, inv, err, err / scale))

        finite = [d[3] for d in defects if np.isfinite(d[3])]
        missing = sum(1 for d in defects if d[1] is None)

        return {
            "matrix_name": self.matrix_name,
            "support_size": self.support_size,
            "num_missing_inverse": missing,
            "star_defect_max": float(np.max(finite)) if finite else None,
            "star_defect_mean": float(np.mean(finite)) if finite else None,
            "star_defect_median": float(np.median(finite)) if finite else None,
        }

    def star_defect_table(self):
        import pandas as pd

        by_key = {
            (int(m), int(n), int(e)): block
            for m, n, e, block in zip(self.h_m, self.h_n, self.h_eps, self.blocks)
        }

        rows = []
        seen = set()

        for key, K_h in by_key.items():
            if key in seen:
                continue

            inv = gd_inverse_label(*key)
            seen.add(key)
            seen.add(inv)

            K_inv = by_key.get(inv)

            if K_inv is None:
                err = np.inf
                rel = np.inf
                norm_h = float(np.linalg.norm(K_h))
                norm_inv = None
            else:
                err = float(np.linalg.norm(K_h - K_inv.conj().T))
                norm_h = float(np.linalg.norm(K_h))
                norm_inv = float(np.linalg.norm(K_inv))
                rel = err / max(norm_h, norm_inv, 1.0)

            rows.append({
                "m": key[0],
                "n": key[1],
                "eps": key[2],
                "inv_m": inv[0],
                "inv_n": inv[1],
                "inv_eps": inv[2],
                "norm": norm_h,
                "inv_norm": norm_inv,
                "star_error": err,
                "star_relative_error": rel,
            })

        return pd.DataFrame(rows).sort_values("star_relative_error", ascending=False)

    def star_defect_table_filtered(K, *, min_norm=1e-2, max_radius=None):
        table = K.star_defect_table()

        table = table[table["norm"] >= min_norm]

        if max_radius is not None:
            radius = np.maximum(np.abs(table["m"]), np.abs(table["n"]))
            table = table[radius <= max_radius]

        return table.sort_values("star_relative_error", ascending=False)

