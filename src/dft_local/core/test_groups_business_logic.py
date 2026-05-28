# Copied from repository-level tests during package migration.
# These tests should target dft_local only.

import numpy as np
import pytest

from dft_local.core.geometry import GdElement

def test_gd_identity():
    e = GdElement.identity()
    x = GdElement.x()
    y = GdElement.y()
    t = GdElement.t()

    assert e * x == x
    assert x * e == x
    assert e * y == y
    assert y * e == y
    assert e * t == t
    assert t * e == t


def test_gd_inverses():
    elems = [
        GdElement.identity(),
        GdElement.x(),
        GdElement.y(),
        GdElement.t(),
        GdElement(3, -2, 0),
        GdElement(3, -2, 1),
    ]

    e = GdElement.identity()

    for g in elems:
        assert g * g.inverse() == e
        assert g.inverse() * g == e


def test_edge_generators_are_involutions():
    e = GdElement.identity()

    for d in [GdElement.d1(), GdElement.d2(), GdElement.d3()]:
        assert d * d == e


def test_translation_generators_from_edges():
    d1 = GdElement.d1()
    d2 = GdElement.d2()
    d3 = GdElement.d3()

    assert d1 * d2 == GdElement.x()
    assert d1 * d3 == GdElement.y()


def test_reflection_action():
    x = GdElement.x()
    y = GdElement.y()
    t = GdElement.t()

    assert t * x * t == x.inverse()
    assert t * y * t == y.inverse()


def test_hexagon_relation():
    d1 = GdElement.d1()
    d2 = GdElement.d2()
    d3 = GdElement.d3()
    e = GdElement.identity()

    assert d3 * d2 * d1 * d3 * d2 * d1 == e

def test_group_label_reverse_lookup(labels):
    for a in range(labels.natoms):
        g = labels.element(a)
        assert labels.element_to_atom.get(g) == a


def test_gd_labels_reconstruct_positions(labels):
    err = labels.gd_position_errors()
    assert np.max(err) < 1e-6
