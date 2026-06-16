from __future__ import annotations

import numpy as np

from dft_local.core.kernels import GdKernelArrays
from dft_local.diagnostics.discovery import load_diagnostics
from dft_local.transport.symmetry_audit.diagnostics import (
    audit_kernel,
    edge_generator_automorphisms,
)


def kernel(keys, values) -> GdKernelArrays:
    h_m = np.asarray([k[0] for k in keys], dtype=np.int64)
    h_n = np.asarray([k[1] for k in keys], dtype=np.int64)
    h_eps = np.asarray([k[2] for k in keys], dtype=np.int64)
    blocks = np.asarray([[[v]] for v in values], dtype=np.float64)

    return GdKernelArrays(
        h_m=h_m,
        h_n=h_n,
        h_eps=h_eps,
        blocks=blocks,
        matrix_name="test",
    )


def test_edge_generator_automorphisms_are_six_bijections_on_generators() -> None:
    automorphisms = edge_generator_automorphisms()

    assert len(automorphisms) == 6

    edge_generators = {
        (0, 0, 1),
        (-1, 0, 1),
        (0, -1, 1),
    }

    for automorphism in automorphisms:
        image = {automorphism.map_key(key) for key in edge_generators}
        assert image == edge_generators


def test_edge_generator_automorphisms_include_reflections_and_rotations() -> None:
    kinds = {a.kind for a in edge_generator_automorphisms()}

    assert "identity" in kinds
    assert "rotation" in kinds
    assert "reflection" in kinds


def test_identity_automorphism_has_zero_error() -> None:
    k = kernel(
        [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ],
        [1.0, 2.0, 3.0, 4.0],
    )

    identity = next(a for a in edge_generator_automorphisms() if a.permutation == (0, 1, 2))
    audit = audit_kernel(object_name="K", kernel=k, automorphism=identity)

    assert audit.kind == "identity"
    assert audit.missing_count == 0
    assert audit.extra_count == 0
    assert audit.max_abs == 0.0
    assert audit.max_rel == 0.0


def test_support_mismatch_is_reported() -> None:
    k = kernel(
        [
            (0, 0, 1),
            (-1, 0, 1),
        ],
        [1.0, 2.0],
    )

    swap = next(a for a in edge_generator_automorphisms() if a.permutation == (0, 2, 1))
    audit = audit_kernel(object_name="K", kernel=k, automorphism=swap)

    assert audit.kind == "reflection"
    assert audit.missing_count > 0
    assert audit.extra_count > 0


def test_value_mismatch_is_reported_when_support_matches() -> None:
    k = kernel(
        [
            (0, 0, 1),
            (-1, 0, 1),
            (0, -1, 1),
        ],
        [1.0, 2.0, 10.0],
    )

    swap = next(a for a in edge_generator_automorphisms() if a.permutation == (0, 2, 1))
    audit = audit_kernel(object_name="K", kernel=k, automorphism=swap)

    assert audit.kind == "reflection"
    assert audit.missing_count == 0
    assert audit.extra_count == 0
    assert audit.max_abs > 0.0
    assert audit.max_rel > 0.0


def test_symmetry_audit_is_discovered_with_translation_input() -> None:
    specs = load_diagnostics()
    spec = next(spec for spec in specs if spec.id == "symmetry_audit")

    assert spec.group == "Graphene / group diagnostics"
    assert any(inp.name == "max_anchors" for inp in spec.inputs)
