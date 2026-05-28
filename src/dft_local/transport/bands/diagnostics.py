"""Diagnostics for the band/path continuation domain."""

from __future__ import annotations

from pathlib import Path

from dft_local.diagnostics.models import (
    Card,
    DiagnosticResult,
    DiagnosticSpec,
    Table,
    TableRow,
)


def _domain_root() -> Path:
    return Path(__file__).resolve().parent


def compute_overview(ctx, inputs: dict[str, object]) -> DiagnosticResult:
    del ctx, inputs

    root = _domain_root()

    files = (
        ("package", root / "__init__.py"),
        ("core", root / "core.py"),
        ("diagnostics", root / "diagnostics.py"),
        ("test metadata", root / "tests.py"),
        ("business tests", root / "test_continuation_business_logic.py"),
    )

    return DiagnosticResult(
        title="Band/path continuation domain",
        summary=(
            "Overview of the domain-local band continuation module. "
            "The current implementation is a compatibility wrapper around "
            "the working dft_local API, with copied business tests."
        ),
        cards=(
            Card("domain", "transport.bands", "ok"),
            Card("business tests", "present", "ok"),
            Card("implementation", "local", "ok", "Core owns local band/path/region continuation implementation"),
        ),
        tables=(
            Table(
                id="bands_files",
                title="Domain files",
                description="Files owned by the band/path continuation domain",
                headers=("role", "path", "exists"),
                rows=tuple(
                    TableRow((role, str(path), path.exists()))
                    for role, path in files
                ),
            ),
            Table(
                id="bands_api",
                title="Public API currently exposed",
                description="Band/path continuation symbols exposed from transport.bands.core",
                headers=("name", "role"),
                rows=(
                    TableRow(("LocalPath", "path solving and band continuation")),
                    TableRow(("LocalRegion", "grid of local paths")),
                    TableRow(("match_via_energies", "energy-prediction band matching")),
                    TableRow(("match_via_overlap", "state-overlap band matching")),
                    TableRow(("align_groups_and_fix_gauge", "degenerate-subspace alignment and phase fixing")),
                    TableRow(("detect_energy_order_crossings_between_steps", "band event detection")),
                ),
            ),
        ),
        notes=(
            "This diagnostic is discovered explicitly through diagnostics(), not by import-time registration.",
            "Next migration step is to expand data/geometry tests around core.geometry and core.kernels.",
        ),
    )


def diagnostics() -> list[DiagnosticSpec]:
    return [
        DiagnosticSpec(
            id="transport.bands.overview",
            group="transport",
            title="Bands domain overview",
            description="Show local files and public API for band/path continuation.",
            inputs=(),
            compute=compute_overview,
            tier="instant",
        )
    ]
