"""Automorphism symmetry audit diagnostics for local G_d kernels."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np

from dft_local.core.kernels import GdKernelArrays
from dft_local.diagnostics.models import (
    Card,
    DiagnosticResult,
    DiagnosticSection,
    DiagnosticSpec,
    InputSpec,
    Table,
    TableRow,
)


KernelKey = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class EdgeAutomorphism:
    """Affine automorphism preserving the three honeycomb edge generators.

    The edge-generator group has elements ``(r, eps)`` with
    ``r = (m, n)`` and ``eps in {0,1}``.

    These point-group automorphisms have the form

        (r, eps) -> (A r + eps c, eps)

    where ``A`` and ``c`` are determined by a permutation of
    ``d1=(0,0,1)``, ``d2=(-1,0,1)``, and ``d3=(0,-1,1)``.
    """

    name: str
    kind: str
    matrix: tuple[tuple[int, int], tuple[int, int]]
    odd_shift: tuple[int, int]
    permutation: tuple[int, int, int]

    def map_key(self, key: KernelKey) -> KernelKey:
        m, n, eps = key
        a00, a01 = self.matrix[0]
        a10, a11 = self.matrix[1]
        c0, c1 = self.odd_shift

        return (
            int(a00 * m + a01 * n + eps * c0),
            int(a10 * m + a11 * n + eps * c1),
            int(eps),
        )


@dataclass(frozen=True, slots=True)
class KernelAudit:
    object_name: str
    automorphism: str
    kind: str
    support_size: int
    image_support_size: int
    common_count: int
    missing_count: int
    extra_count: int
    max_abs: float
    mean_abs: float
    max_rel: float
    mean_rel: float
    worst_source: KernelKey | None
    worst_target: KernelKey | None


@dataclass(frozen=True, slots=True)
class TranslationAudit:
    object_name: str
    anchor_atom: int
    anchor_label: KernelKey
    compared_to_atom: int
    compared_to_label: KernelKey
    support_size: int
    other_support_size: int
    common_count: int
    missing_count: int
    extra_count: int
    max_abs: float
    mean_abs: float
    max_rel: float
    mean_rel: float
    worst_h: KernelKey | None


def edge_generator_automorphisms() -> tuple[EdgeAutomorphism, ...]:
    """Return all six point-group automorphisms from permutations of d1,d2,d3."""

    odd_generators = (
        (0, 0),    # d1 = t
        (-1, 0),   # d2 = x^-1 t
        (0, -1),   # d3 = y^-1 t
    )

    names = ("d1", "d2", "d3")
    out: list[EdgeAutomorphism] = []

    for perm in permutations(range(3)):
        c = odd_generators[perm[0]]

        image_d2_delta = (
            odd_generators[perm[1]][0] - c[0],
            odd_generators[perm[1]][1] - c[1],
        )
        image_d3_delta = (
            odd_generators[perm[2]][0] - c[0],
            odd_generators[perm[2]][1] - c[1],
        )

        # Source d2 has translation (-1, 0), so A[:,0] = -image_d2_delta.
        # Source d3 has translation (0, -1), so A[:,1] = -image_d3_delta.
        matrix = (
            (-image_d2_delta[0], -image_d3_delta[0]),
            (-image_d2_delta[1], -image_d3_delta[1]),
        )

        det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        if abs(det) != 1:
            raise ValueError(f"Bad edge automorphism determinant {det} for {perm}")

        if perm == (0, 1, 2):
            kind = "identity"
        elif det > 0:
            kind = "rotation"
        else:
            kind = "reflection"

        label = "_".join(
            f"{names[i]}->{names[perm[i]]}"
            for i in range(3)
        )

        out.append(
            EdgeAutomorphism(
                name=label,
                kind=kind,
                matrix=matrix,
                odd_shift=c,
                permutation=tuple(int(i) for i in perm),
            )
        )

    return tuple(out)


def kernel_block_map(kernel: GdKernelArrays) -> dict[KernelKey, np.ndarray]:
    return {
        (int(m), int(n), int(e)): np.asarray(block)
        for m, n, e, block in zip(kernel.h_m, kernel.h_n, kernel.h_eps, kernel.blocks)
    }


def _compare_block_maps(
    *,
    object_name: str,
    left: dict[KernelKey, np.ndarray],
    right: dict[KernelKey, np.ndarray],
    eps: float = 1.0e-300,
) -> tuple[int, int, int, int, int, float, float, float, float, KernelKey | None]:
    left_support = set(left)
    right_support = set(right)
    common = left_support & right_support
    missing = left_support - right_support
    extra = right_support - left_support

    abs_errors: list[float] = []
    rel_errors: list[float] = []
    worst_h: KernelKey | None = None
    worst_rel = -1.0

    for h in common:
        a = left[h]
        b = right[h]

        abs_err = float(np.linalg.norm(a - b))
        denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), eps)
        rel_err = abs_err / denom

        abs_errors.append(abs_err)
        rel_errors.append(rel_err)

        if rel_err > worst_rel:
            worst_rel = rel_err
            worst_h = h

    if abs_errors:
        max_abs = float(np.max(abs_errors))
        mean_abs = float(np.mean(abs_errors))
        max_rel = float(np.max(rel_errors))
        mean_rel = float(np.mean(rel_errors))
    else:
        max_abs = np.nan
        mean_abs = np.nan
        max_rel = np.nan
        mean_rel = np.nan

    return (
        len(left_support),
        len(right_support),
        len(common),
        len(missing),
        len(extra),
        max_abs,
        mean_abs,
        max_rel,
        mean_rel,
        worst_h,
    )


def audit_kernel(
    *,
    object_name: str,
    kernel: GdKernelArrays,
    automorphism: EdgeAutomorphism,
) -> KernelAudit:
    blocks = kernel_block_map(kernel)
    transformed = {
        automorphism.map_key(key): block
        for key, block in blocks.items()
    }

    (
        support_size,
        image_support_size,
        common_count,
        missing_count,
        extra_count,
        max_abs,
        mean_abs,
        max_rel,
        mean_rel,
        worst_h,
    ) = _compare_block_maps(
        object_name=object_name,
        left=blocks,
        right=transformed,
    )

    worst_target = automorphism.map_key(worst_h) if worst_h is not None else None

    return KernelAudit(
        object_name=object_name,
        automorphism=automorphism.name,
        kind=automorphism.kind,
        support_size=support_size,
        image_support_size=image_support_size,
        common_count=common_count,
        missing_count=missing_count,
        extra_count=extra_count,
        max_abs=max_abs,
        mean_abs=mean_abs,
        max_rel=max_rel,
        mean_rel=mean_rel,
        worst_source=worst_h,
        worst_target=worst_target,
    )


def audit_objects(objects: dict[str, GdKernelArrays]) -> tuple[KernelAudit, ...]:
    automorphisms = edge_generator_automorphisms()
    rows: list[KernelAudit] = []

    for object_name, kernel in objects.items():
        for automorphism in automorphisms:
            rows.append(
                audit_kernel(
                    object_name=object_name,
                    kernel=kernel,
                    automorphism=automorphism,
                )
            )

    return tuple(rows)


def _starred_anchor_kernel(matrix: Any, labels: Any, anchor_atom: int, matrix_name: str) -> GdKernelArrays:
    return GdKernelArrays.from_anchored(
        matrix,
        labels,
        anchor_atom=anchor_atom,
        matrix_name=matrix_name,
        copy_blocks=True,
    ).star_symmetrised(matrix_name=f"{matrix_name} star")


def audit_translation_rows(ctx: Any, *, max_anchors: int) -> tuple[TranslationAudit, ...]:
    """Compare kernel rows from translated bulk anchors against the reference anchor.

    Translation symmetry is not a nontrivial automorphism of the relative
    coordinate h = g_a^-1 g_b.  If the full operator is translation homogeneous,
    each anchor row should produce the same relative kernel K(h).
    """

    labels = ctx.state.labels
    anchor_atom = int(labels.anchor_atom)
    anchors = [int(a) for a in labels.geometry.core_bulk_atoms() if int(a) != anchor_atom]
    anchors = anchors[:max_anchors]

    object_builders = (
        (
            "H row",
            lambda a: GdKernelArrays.from_anchored(
                ctx.state.data.H,
                labels,
                anchor_atom=a,
                matrix_name=f"H row {a}",
                copy_blocks=True,
            ),
        ),
        (
            "S row",
            lambda a: GdKernelArrays.from_anchored(
                ctx.state.data.S,
                labels,
                anchor_atom=a,
                matrix_name=f"S row {a}",
                copy_blocks=True,
            ),
        ),
        (
            "H row star",
            lambda a: _starred_anchor_kernel(ctx.state.data.H, labels, a, f"H row {a}"),
        ),
        (
            "S row star",
            lambda a: _starred_anchor_kernel(ctx.state.data.S, labels, a, f"S row {a}"),
        ),
    )

    rows: list[TranslationAudit] = []
    anchor_label = labels.element(anchor_atom).as_tuple()

    for object_name, builder in object_builders:
        reference = kernel_block_map(builder(anchor_atom))

        for a in anchors:
            other = kernel_block_map(builder(a))
            (
                support_size,
                other_support_size,
                common_count,
                missing_count,
                extra_count,
                max_abs,
                mean_abs,
                max_rel,
                mean_rel,
                worst_h,
            ) = _compare_block_maps(
                object_name=object_name,
                left=reference,
                right=other,
            )

            rows.append(
                TranslationAudit(
                    object_name=object_name,
                    anchor_atom=anchor_atom,
                    anchor_label=anchor_label,
                    compared_to_atom=a,
                    compared_to_label=labels.element(a).as_tuple(),
                    support_size=support_size,
                    other_support_size=other_support_size,
                    common_count=common_count,
                    missing_count=missing_count,
                    extra_count=extra_count,
                    max_abs=max_abs,
                    mean_abs=mean_abs,
                    max_rel=max_rel,
                    mean_rel=mean_rel,
                    worst_h=worst_h,
                )
            )

    return tuple(rows)


def _fmt_key(key: KernelKey | None) -> str:
    if key is None:
        return ""
    return f"({key[0]}, {key[1]}, {key[2]})"


def _status_for_error(max_rel: float, missing_count: int, extra_count: int) -> str:
    if missing_count or extra_count:
        return "support mismatch"
    if not np.isfinite(max_rel):
        return "no common support"
    if max_rel < 1.0e-10:
        return "green"
    if max_rel < 1.0e-7:
        return "yellow"
    return "red"


def compute_overview(ctx, inputs: dict[str, object]) -> DiagnosticResult:
    max_anchors = int(inputs.get("max_anchors", 32))

    objects = {
        "H anchored": ctx.state.KH,
        "S anchored": ctx.state.KS,
        "H anchored star": ctx.state.KH_star,
        "S anchored star": ctx.state.KS_star,
        "H average": ctx.state.KH_avg,
        "S average": ctx.state.KS_avg,
        "H average star": ctx.state.KH_avg_star,
        "S average star": ctx.state.KS_avg_star,
    }

    audits = audit_objects(objects)
    translation_audits = audit_translation_rows(ctx, max_anchors=max_anchors)

    point_rows = tuple(
        TableRow((
            a.object_name,
            a.kind,
            a.automorphism,
            _status_for_error(a.max_rel, a.missing_count, a.extra_count),
            a.support_size,
            a.image_support_size,
            a.common_count,
            a.missing_count,
            a.extra_count,
            a.max_abs,
            a.mean_abs,
            a.max_rel,
            a.mean_rel,
            _fmt_key(a.worst_source),
            _fmt_key(a.worst_target),
        ))
        for a in audits
    )

    translation_rows = tuple(
        TableRow((
            a.object_name,
            a.anchor_atom,
            _fmt_key(a.anchor_label),
            a.compared_to_atom,
            _fmt_key(a.compared_to_label),
            _status_for_error(a.max_rel, a.missing_count, a.extra_count),
            a.support_size,
            a.other_support_size,
            a.common_count,
            a.missing_count,
            a.extra_count,
            a.max_abs,
            a.mean_abs,
            a.max_rel,
            a.mean_rel,
            _fmt_key(a.worst_h),
        ))
        for a in translation_audits
    )

    support_failures = sum(1 for a in audits if a.missing_count or a.extra_count)
    reflection_failures = sum(
        1
        for a in audits
        if a.kind == "reflection" and (a.missing_count or a.extra_count or (np.isfinite(a.max_rel) and a.max_rel >= 1.0e-7))
    )
    translation_support_failures = sum(1 for a in translation_audits if a.missing_count or a.extra_count)
    translation_value_failures = sum(
        1
        for a in translation_audits
        if not (a.missing_count or a.extra_count)
        and np.isfinite(a.max_rel)
        and a.max_rel >= 1.0e-7
    )

    return DiagnosticResult(
        title="Symmetry audit",
        summary=(
            "Local G_d symmetry audit for H, S, H star, and S star. "
            "Checks point-group automorphisms, explicit reflections, and translation row homogeneity."
        ),
        cards=(
            Card("point automorphisms", len(edge_generator_automorphisms()), "neutral", "All permutations of d1, d2, d3."),
            Card("reflection failures", reflection_failures, "neutral", "Reflection rows with support or large value mismatch."),
            Card("point support failures", support_failures, "neutral", "Rows where transformed support differs."),
            Card("translation anchors", min(max_anchors, len(ctx.state.labels.geometry.core_bulk_atoms())), "neutral", "Core-bulk anchors compared to reference row."),
            Card("translation support failures", translation_support_failures, "neutral", "Anchor rows with different relative support."),
            Card("translation value failures", translation_value_failures, "neutral", "Anchor rows with max relative block error >= 1e-7."),
        ),
        sections=(
            DiagnosticSection(
                id="point_group_automorphism_audit",
                title="Point-group automorphism audit",
                description=(
                    "For each relative kernel K(h), compare K(h) with K(alpha(h)). "
                    "The automorphisms are the six permutations of graphene edge generators d1, d2, d3. "
                    "Rows are classified as identity, rotation, or reflection."
                ),
                tables=(
                    Table(
                        id="point_group_automorphism_audit_table",
                        title="Point-group support and block errors",
                        description="One row per object and edge-generator automorphism.",
                        headers=(
                            "object",
                            "kind",
                            "automorphism",
                            "status",
                            "support",
                            "image support",
                            "common",
                            "missing",
                            "extra",
                            "max abs",
                            "mean abs",
                            "max rel",
                            "mean rel",
                            "worst h",
                            "worst alpha(h)",
                        ),
                        rows=point_rows,
                        numeric=frozenset({4, 5, 6, 7, 8, 9, 10, 11, 12}),
                    ),
                ),
            ),
            DiagnosticSection(
                id="translation_row_audit",
                title="Translation row audit",
                description=(
                    "Translation symmetry is checked by extracting anchored relative kernels from many core-bulk atoms. "
                    "For a homogeneous operator, each translated row should give the same K(h) as the reference anchor."
                ),
                tables=(
                    Table(
                        id="translation_row_audit_table",
                        title="Translated anchor row errors",
                        description="One row per object and compared core-bulk anchor.",
                        headers=(
                            "object",
                            "reference atom",
                            "reference label",
                            "other atom",
                            "other label",
                            "status",
                            "support",
                            "other support",
                            "common",
                            "missing",
                            "extra",
                            "max abs",
                            "mean abs",
                            "max rel",
                            "mean rel",
                            "worst h",
                        ),
                        rows=translation_rows,
                        numeric=frozenset({1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14}),
                    ),
                ),
            ),
        ),
        notes=(
            "Point-group audit currently uses identity channel transforms. If an automorphism acts nontrivially on local orbital/channel basis, add U_alpha around the block comparison.",
            "Translation audit is row homogeneity, not a map h -> alpha(h). Left translation cancels in h = g_a^-1 g_b.",
            "Symbol covariance and band/projector symmetry checks should be added after this local support/value audit is trusted.",
        ),
    )


def diagnostics() -> tuple[DiagnosticSpec, ...]:
    return (
        DiagnosticSpec(
            id="symmetry_audit",
            group="Graphene / group diagnostics",
            title="Symmetry audit",
            description="Audit local G_d kernel support, reflection automorphisms, and translation row homogeneity.",
            inputs=(
                InputSpec(
                    name="max_anchors",
                    label="Max translation anchors",
                    kind="int",
                    default=32,
                    min_value=1,
                    max_value=512,
                    help="Maximum number of core-bulk anchor rows to compare for translation symmetry.",
                ),
            ),
            compute=compute_overview,
            tier="cheap",
        ),
    )
