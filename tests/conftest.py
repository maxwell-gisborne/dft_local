import numpy as np
import pytest

from dft_local import NearestNeighbourGraph,SparseDataset,EdgeDirections, EdgeGroupLabels


@pytest.fixture(scope="session")
def data():
    return SparseDataset.load("test_run/run_dir/data")


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def sample_atoms(data, rng):
    n = data.metadata.natoms
    size = min(100, n)
    return rng.choice(n, size=size, replace=False)

@pytest.fixture
def assert_block_close():
    def _assert_block_close(A, B, *, atol=1e-12, rtol=0.0, context=""):
        if not np.allclose(A, B, atol=atol, rtol=rtol):
            err = np.linalg.norm(A - B)
            scale = max(np.linalg.norm(A), np.linalg.norm(B), 1.0)
            rel = err / scale
            raise AssertionError(
                f"block mismatch {context}: norm error={err}, relative={rel}"
            )

    return _assert_block_close


@pytest.fixture(scope="session")
def geom(data):
    return NearestNeighbourGraph.from_positions(data.metadata.positions)


@pytest.fixture(scope="session")
def edge_dirs(geom):
    return EdgeDirections.from_geometry(geom)

@pytest.fixture(scope="session")
def edges(geom):
    return EdgeDirections.from_geometry(geom)


@pytest.fixture(scope="session")
def labels(geom, edges):
    return EdgeGroupLabels.from_geometry(geom, edges)


