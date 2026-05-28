from __future__ import annotations

import numpy as np

from dft_local.core.geometry import (
    EdgeDirections,
    EdgeGroupLabels,
    GdElement,
    NearestNeighbourGraph,
)
from dft_local.core.kernels import GdKernelArrays, gd_inverse_label


def test_gd_element_inverse_and_generators() -> None:
    for g in [
        GdElement.identity(),
        GdElement.x(),
        GdElement.y(),
        GdElement.d1(),
        GdElement.d2(),
        GdElement.d3(),
        GdElement(2, -3, 1),
    ]:
        assert g * g.inverse() == GdElement.identity()
        assert g.inverse() * g == GdElement.identity()

    assert GdElement.d1() * GdElement.d1() == GdElement.identity()
    assert GdElement.d2() * GdElement.d2() == GdElement.identity()
    assert GdElement.d3() * GdElement.d3() == GdElement.identity()


def test_gd_inverse_label_is_involutive() -> None:
    examples = [
        (0, 0, 0),
        (1, 0, 0),
        (-2, 3, 0),
        (0, 0, 1),
        (-1, 0, 1),
        (2, -3, 1),
    ]

    for label in examples:
        assert gd_inverse_label(*gd_inverse_label(*label)) == label


def test_nearest_neighbour_graph_and_edge_labels_on_toy_star() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
            [-0.5, -np.sqrt(3.0) / 2.0, 0.0],
        ],
        dtype=np.float64,
    )

    geom = NearestNeighbourGraph.from_positions(
        positions,
        cutoff_factor=1.1,
        query_k=4,
    )
    edges = EdgeDirections.from_geometry(geom, anchor_atom=0)
    labels = EdgeGroupLabels.from_geometry(geom, edges, anchor_atom=0)

    assert geom.natoms == 4
    assert labels.element(0) == GdElement.identity()
    assert np.all(labels.visited)
    assert np.max(labels.gd_position_errors()) < 1e-12


def test_gd_kernel_symbols_and_star_symmetrisation() -> None:
    K = GdKernelArrays(
        h_m=np.asarray([1, -1], dtype=np.int64),
        h_n=np.asarray([0, 0], dtype=np.int64),
        h_eps=np.asarray([0, 0], dtype=np.int64),
        blocks=np.asarray([[[0.5]], [[0.5]]], dtype=np.complex128),
        matrix_name="cosine",
    )

    H = K.symbol_generic(0.3, 0.0)
    fixed = K.symbol_fixed(0.3, 0.0, sigma=1)

    assert H.shape == (2, 2)
    assert fixed.shape == (1, 1)
    assert np.allclose(np.linalg.eigvalsh(H), [np.cos(0.3), np.cos(0.3)])
    assert np.allclose(fixed[0, 0], np.cos(0.3))

    d = K.star_symmetrised().star_defect()
    assert d["num_missing_inverse"] == 0
    assert d["star_defect_max"] < 1e-12
