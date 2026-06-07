from __future__ import annotations

import numpy as np
import pytest

from dft_local.transport.band_labelling import (
    BandOrder,
    apply_band_order,
    energy_order,
)


def test_energy_order_sorts_each_sample_by_eigenvalue() -> None:
    E = np.array(
        [
            [3.0, 1.0, 2.0],
            [0.5, -1.0, 0.0],
        ],
        dtype=np.float64,
    )

    order = energy_order(E)

    assert isinstance(order, BandOrder)
    assert order.kind == "energy"
    assert np.array_equal(order.indices, [[1, 2, 0], [1, 2, 0]])


def test_energy_order_is_stable_at_exact_degeneracy() -> None:
    E = np.array([[2.0, 1.0, 1.0, 3.0]], dtype=np.float64)

    order = energy_order(E)

    assert np.array_equal(order.indices, [[1, 2, 0, 3]])


def test_apply_band_order_reorders_energy_matrix() -> None:
    E = np.array([[3.0, 1.0, 2.0]], dtype=np.float64)
    order = energy_order(E)

    got = apply_band_order(E, order, band_axis=1)

    assert np.array_equal(got, [[1.0, 2.0, 3.0]])


def test_apply_band_order_reorders_velocity_band_axis() -> None:
    velocities = np.array(
        [
            [
                [30.0, 10.0, 20.0],
                [300.0, 100.0, 200.0],
            ]
        ],
        dtype=np.float64,
    )
    order = BandOrder(np.array([[1, 2, 0]], dtype=np.int64))

    got = apply_band_order(velocities, order, band_axis=2)

    assert np.array_equal(
        got,
        [
            [
                [10.0, 20.0, 30.0],
                [100.0, 200.0, 300.0],
            ]
        ],
    )


def test_apply_band_order_rejects_shape_mismatch() -> None:
    order = BandOrder(np.array([[0, 1]], dtype=np.int64))

    with pytest.raises(ValueError, match="band count mismatch"):
        apply_band_order(np.zeros((1, 3)), order, band_axis=1)
