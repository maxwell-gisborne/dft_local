import numpy as np
import pytest

from dft_local import NearestNeighbourGraph,SparseDataset,EdgeDirections, GdElement, EdgeGroupLabels

@pytest.fixture(scope="session")
def data():
    return SparseDataset.load("./test_run/run_dir/data")


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


def test_geometry_a0_positive(geom):
    assert np.isfinite(geom.a0)
    assert geom.a0 > 0


def test_geometry_cutoff_between_first_and_second_shell(geom):
    assert geom.cutoff > geom.a0
    assert geom.cutoff < np.sqrt(3.0) * geom.a0


def test_geometry_shapes(geom):
    assert geom.positions.shape == (geom.natoms, 3)
    assert geom.indptr.shape == (geom.natoms + 1,)
    assert geom.indices.ndim == 1
    assert geom.distances.shape == geom.indices.shape
    assert geom.vectors.shape == (len(geom.indices), 3)


def test_geometry_no_self_neighbours(geom):
    for a in range(geom.natoms):
        assert a not in set(map(int, geom.neighbours(a)))


def test_geometry_all_neighbour_distances_within_cutoff(geom):
    assert np.all(geom.distances <= geom.cutoff)


def test_geometry_all_neighbour_distances_near_a0(geom):
    # Loose enough for numerical/edge strain, strict enough to catch second shell.
    assert np.all(geom.distances > 0.75 * geom.a0)
    assert np.all(geom.distances < 1.25 * geom.a0)


def test_geometry_graph_is_symmetric(geom):
    neighbour_sets = [
        set(map(int, geom.neighbours(a)))
        for a in range(geom.natoms)
    ]

    for a in range(geom.natoms):
        for b in neighbour_sets[a]:
            assert a in neighbour_sets[b]


def test_geometry_has_bulk_atoms(geom):
    assert len(geom.bulk_atoms()) > 0


def test_geometry_choose_anchor_is_bulk(geom):
    anchor = geom.choose_anchor()
    assert geom.degree[anchor] == 3


def test_geometry_core_anchor_if_available(geom):
    core = set(map(int, geom.core_bulk_atoms()))
    anchor = geom.choose_anchor()

    if core:
        assert anchor in core
        assert np.all(geom.degree[geom.neighbours(anchor)] == 3)





def test_edge_directions_anchor_has_three_neighbours(geom, edge_dirs):
    assert geom.degree[edge_dirs.anchor_atom] == 3
    assert edge_dirs.anchor_neighbours.shape == (3,)
    assert edge_dirs.d_vectors.shape == (3, 3)
    assert edge_dirs.d_unit.shape == (3, 3)


def test_edge_directions_units_are_normalised(edge_dirs):
    norms = np.linalg.norm(edge_dirs.d_unit, axis=1)
    assert np.allclose(norms, 1.0)


def test_edge_directions_plane_basis_orthonormal(edge_dirs):
    e1 = edge_dirs.plane_e1
    e2 = edge_dirs.plane_e2
    n = edge_dirs.plane_normal

    assert np.isclose(np.linalg.norm(e1), 1.0)
    assert np.isclose(np.linalg.norm(e2), 1.0)
    assert np.isclose(np.linalg.norm(n), 1.0)

    assert np.isclose(np.dot(e1, e2), 0.0, atol=1e-12)
    assert np.isclose(np.dot(e1, n), 0.0, atol=1e-12)
    assert np.isclose(np.dot(e2, n), 0.0, atol=1e-12)


def test_anchor_vectors_classify_as_themselves(edge_dirs):
    labels = edge_dirs.classify_vectors(edge_dirs.d_vectors)
    assert np.array_equal(labels, np.array([0, 1, 2]))


def test_reversed_anchor_vectors_classify_as_themselves(edge_dirs):
    labels = edge_dirs.classify_vectors(-edge_dirs.d_vectors)
    assert np.array_equal(labels, np.array([0, 1, 2]))


def test_all_graph_edges_are_classifiable(geom, edge_dirs):
    labels = edge_dirs.classify_vectors(geom.vectors)

    assert labels.shape == geom.indices.shape
    assert set(map(int, labels)) <= {0, 1, 2}


def test_all_generators_are_used(geom, edge_dirs):
    labels = edge_dirs.classify_vectors(geom.vectors)

    counts = np.bincount(labels, minlength=3)
    assert np.all(counts > 0)


def test_edge_classification_alignment_is_good(geom, edge_dirs):
    unit_vectors = geom.vectors / np.linalg.norm(geom.vectors, axis=1)[:, None]
    scores = np.abs(unit_vectors @ edge_dirs.d_unit.T)
    best = np.max(scores, axis=1)

    assert np.min(best) > 0.95



def test_labels_anchor_is_identity(labels):
    assert labels.element(labels.anchor_atom) == GdElement.identity()


def test_labels_all_atoms_visited(labels):
    assert np.all(labels.visited)


def test_labels_eps_is_binary(labels):
    assert set(map(int, labels.eps[labels.visited])) <= {0, 1}


def test_labels_nearest_neighbour_relative_elements_are_edge_generators(
    geom,
    edge_dirs,
    labels,
):
    edge_generators = {
        GdElement.d1(),
        GdElement.d2(),
        GdElement.d3(),
    }

    for a in range(geom.natoms):
        for b in geom.neighbours(a):
            h = labels.relative(a, int(b))
            assert h in edge_generators


def test_labels_edge_type_matches_relative_element(geom, edge_dirs, labels):
    edge_generators = [
        GdElement.d1(),
        GdElement.d2(),
        GdElement.d3(),
    ]

    for a in range(geom.natoms):
        neighbours = geom.neighbours(a)
        vectors = geom.neighbour_vectors(a)
        edge_types = edge_dirs.classify_vectors(vectors)

        for b, edge_type in zip(neighbours, edge_types):
            h = labels.relative(a, int(b))
            assert h == edge_generators[int(edge_type)]


def test_labels_neighbour_eps_flips(geom, labels):
    for a in range(geom.natoms):
        for b in geom.neighbours(a):
            assert labels.eps[int(b)] == 1 - labels.eps[a]


def test_labels_translation_same_sublattice_after_two_steps(labels):
    x = GdElement.x()
    y = GdElement.y()

    assert x.eps == 0
    assert y.eps == 0

def reconstructed_positions(geom, edge_dirs, labels):
    R0 = geom.positions[labels.anchor_atom]

    d1, d2, d3 = edge_dirs.d_vectors

    ax = d1 - d2
    ay = d1 - d3

    return (
        R0
        + labels.m[:, None] * ax[None, :]
        + labels.n[:, None] * ay[None, :]
        + labels.eps[:, None] * d1[None, :]
    )


def label_position_errors(geom, edge_dirs, labels):
    R_pred = reconstructed_positions(geom, edge_dirs, labels)
    err = np.linalg.norm(R_pred - geom.positions, axis=1)
    return err

def test_labels_reconstruct_positions(geom, edge_dirs, labels):
    err = label_position_errors(geom, edge_dirs, labels)

    assert np.max(err) < 1e-6
