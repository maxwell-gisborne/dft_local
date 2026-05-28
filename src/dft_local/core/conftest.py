from __future__ import annotations

import numpy as np
import pytest

from dft_local.core.dataset import SparseDataset
from dft_local.core.geometry import (
    EdgeDirections,
    EdgeGroupLabels,
    NearestNeighbourGraph,
)


@pytest.fixture(scope="session")
def data():
    return SparseDataset.load("test_run/run_dir/data")


@pytest.fixture(scope="session")
def geom(data):
    return NearestNeighbourGraph.from_positions(data.metadata.positions)


@pytest.fixture(scope="session")
def edge_dirs(geom):
    return EdgeDirections.from_geometry(geom)


@pytest.fixture(scope="session")
def labels(geom, edge_dirs):
    return EdgeGroupLabels.from_geometry(geom, edge_dirs)


@pytest.fixture(scope="session")
def sample_atoms(labels):
    visited = np.flatnonzero(labels.visited)
    return visited[: min(50, len(visited))]


@pytest.fixture
def assert_block_close():
    def _assert_block_close(A, B, *, atol=1e-8, rtol=1e-6, context=""):
        if not np.allclose(A, B, atol=atol, rtol=rtol):
            err = np.linalg.norm(A - B)
            scale = max(np.linalg.norm(A), np.linalg.norm(B), 1.0)
            rel = err / scale
            raise AssertionError(
                f"block mismatch {context}: norm error={err}, relative={rel}"
            )

    return _assert_block_close
