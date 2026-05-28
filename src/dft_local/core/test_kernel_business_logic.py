# Copied from repository-level tests during package migration.
# These tests should target dft_local only.

import pytest
import numpy as np

from dft_local.core.geometry import GdElement
from dft_local.core.kernels import GdKernelArrays, relative_labels_for_row
from dft_local.core.sparse import block_row_raw, block_view_bsr



def test_anchored_kernel_identity_labels(data, labels):
    K = GdKernelArrays.from_anchored(data.H, labels)

    atoms_b, _ = block_row_raw(data.H, labels.anchor_atom)

    assert np.all(K.h_m == labels.m[atoms_b])
    assert np.all(K.h_n == labels.n[atoms_b])
    assert np.all(K.h_eps == labels.eps[atoms_b])


def test_gd_kernel_symbol_generic_shape(data, labels):
    K = GdKernelArrays.from_anchored(data.H, labels)
    Hk = K.symbol_generic(0.1, 0.2)

    q = data.basis.nchannels
    assert Hk.shape == (2 * q, 2 * q)

def test_gd_kernel_symbol_fixed_shape(data, labels):
    K = GdKernelArrays.from_anchored(data.H, labels)
    Hk = K.symbol_fixed(0.0, 0.0, sigma=1)

    q = data.basis.nchannels
    assert Hk.shape == (q, q)

def test_kernel_support_size_matches_block_row(data, labels):
    K = GdKernelArrays.from_anchored(data.H, labels)

    atoms_b, blocks = block_row_raw(data.H, labels.anchor_atom)

    assert K.support_size == len(atoms_b)
    assert K.blocks.shape == blocks.shape

def test_kernel_identity_anchor_relative_labels_are_absolute(data, labels):
    K = GdKernelArrays.from_anchored(data.H, labels)
    atoms_b, _blocks = block_row_raw(data.H, labels.anchor_atom)

    assert labels.element(labels.anchor_atom) == GdElement.identity()

    assert np.array_equal(K.h_m, labels.m[atoms_b])
    assert np.array_equal(K.h_n, labels.n[atoms_b])
    assert np.array_equal(K.h_eps, labels.eps[atoms_b])

def test_kernel_blocks_match_raw_block_row(data, labels):
    K = GdKernelArrays.from_anchored(data.H, labels)
    _atoms_b, blocks = block_row_raw(data.H, labels.anchor_atom)

    assert np.allclose(K.blocks, blocks, atol=0.0, rtol=0.0)

def test_relative_labels_for_row_match_gdelement(data, labels, sample_atoms):
    for a in sample_atoms[:20]:
        a = int(a)
        atoms_b, _blocks = block_row_raw(data.H, a)

        h_m, h_n, h_eps = relative_labels_for_row(labels, a, atoms_b)

        for i, b in enumerate(atoms_b):
            h = labels.relative(a, int(b))

            assert h_m[i] == h.m
            assert h_n[i] == h.n
            assert h_eps[i] == h.eps

def test_kernel_arrays_are_readonly(data, labels):
    K = GdKernelArrays.from_anchored(data.H, labels)

    assert not K.h_m.flags.writeable
    assert not K.h_n.flags.writeable
    assert not K.h_eps.flags.writeable
    assert not K.blocks.flags.writeable

def test_anchored_kernel_has_unique_relative_labels(data, labels):
    K = GdKernelArrays.from_anchored(data.H, labels)

    support = list(zip(
        map(int, K.h_m),
        map(int, K.h_n),
        map(int, K.h_eps),
    ))

    assert len(support) == len(set(support))


def omega_gd_generic(m: int, n: int, eps: int, k1: float, k2: float) -> np.ndarray:
    theta = k1 * m + k2 * n
    p = np.exp(1j * theta)

    if eps == 0:
        return np.array(
            [
                [p, 0.0],
                [0.0, np.conj(p)],
            ],
            dtype=np.complex128,
        )

    if eps == 1:
        return np.array(
            [
                [0.0, np.conj(p)],
                [p, 0.0],
            ],
            dtype=np.complex128,
        )

    raise ValueError(eps)


def slow_gd_symbol_generic(K: GdKernelArrays, k1: float, k2: float) -> np.ndarray:
    q = K.blocks.shape[1]
    out = np.zeros((2 * q, 2 * q), dtype=np.complex128)

    for m, n, eps, block in zip(K.h_m, K.h_n, K.h_eps, K.blocks):
        omega = omega_gd_generic(int(m), int(n), int(eps), k1, k2)
        out += np.kron(omega, block)

    return out

def test_symbol_generic_matches_slow_explicit_sum(data, labels):
    K = GdKernelArrays.from_anchored(data.H, labels)

    for k1, k2 in [
        (0.0, 0.0),
        (0.1, 0.2),
        (np.pi / 3, -np.pi / 5),
        (2 * np.pi / 3, -2 * np.pi / 3),
    ]:
        fast = K.symbol_generic(k1, k2)
        slow = slow_gd_symbol_generic(K, k1, k2)

        assert np.allclose(fast, slow, atol=1e-12, rtol=1e-12)


def slow_gd_symbol_fixed(K: GdKernelArrays, k1: float, k2: float, sigma: int) -> np.ndarray:
    q = K.blocks.shape[1]
    out = np.zeros((q, q), dtype=np.complex128)

    for m, n, eps, block in zip(K.h_m, K.h_n, K.h_eps, K.blocks):
        theta = k1 * int(m) + k2 * int(n)
        coeff = np.exp(1j * theta) * (sigma ** int(eps))
        out += coeff * block

    return out


def test_symbol_fixed_matches_slow_explicit_sum(data, labels):
    K = GdKernelArrays.from_anchored(data.H, labels)

    for k1, k2 in [
        (0.0, 0.0),
        (np.pi, 0.0),
        (0.0, np.pi),
        (np.pi, np.pi),
    ]:
        for sigma in [-1, 1]:
            fast = K.symbol_fixed(k1, k2, sigma)
            slow = slow_gd_symbol_fixed(K, k1, k2, sigma)

            assert np.allclose(fast, slow, atol=1e-12, rtol=1e-12)


def parity_transform(q: int) -> np.ndarray:
    I = np.eye(q, dtype=np.complex128)

    U = np.block(
        [
            [I, I],
            [I, -I],
        ]
    ) / np.sqrt(2.0)

    return U

def test_generic_fixed_point_decomposes_into_fixed_symbols(data, labels):
    K = GdKernelArrays.from_anchored(data.H, labels)
    q = K.blocks.shape[1]
    U = parity_transform(q)

    for k1, k2 in [
        (0.0, 0.0),
        (np.pi, 0.0),
        (0.0, np.pi),
        (np.pi, np.pi),
    ]:
        generic = K.symbol_generic(k1, k2)

        # U is real/unitary, and columns are parity basis.
        transformed = U.conj().T @ generic @ U

        plus = K.symbol_fixed(k1, k2, sigma=1)
        minus = K.symbol_fixed(k1, k2, sigma=-1)

        assert np.allclose(transformed[:q, :q], plus, atol=1e-12, rtol=1e-12)
        assert np.allclose(transformed[q:, q:], minus, atol=1e-12, rtol=1e-12)
        assert np.allclose(transformed[:q, q:], 0.0, atol=1e-12, rtol=1e-12)
        assert np.allclose(transformed[q:, :q], 0.0, atol=1e-12, rtol=1e-12)


def test_kernel_contains_onsite_identity_block(data, labels):
    K = GdKernelArrays.from_anchored(data.H, labels)

    identity_mask = (K.h_m == 0) & (K.h_n == 0) & (K.h_eps == 0)
    assert np.sum(identity_mask) == 1

    onsite_from_kernel = K.blocks[np.flatnonzero(identity_mask)[0]]
    onsite_direct = block_view_bsr(data.H, labels.anchor_atom, labels.anchor_atom)

    assert np.allclose(onsite_from_kernel, onsite_direct, atol=0.0, rtol=0.0)

def test_generic_fixed_symbols_match_spectra_at_gamma(data, labels):
    K = GdKernelArrays.from_anchored(data.H, labels, labels.anchor_atom)

    A_generic = K.symbol_generic(0.0, 0.0)
    A_plus = K.symbol_fixed(0.0, 0.0, sigma=1)
    A_minus = K.symbol_fixed(0.0, 0.0, sigma=-1)

    evals_generic = np.sort_complex(np.linalg.eigvals(A_generic))
    evals_fixed = np.sort_complex(
        np.concatenate([
            np.linalg.eigvals(A_plus),
            np.linalg.eigvals(A_minus),
        ])
    )

    assert np.allclose(evals_generic, evals_fixed, atol=1e-8)
