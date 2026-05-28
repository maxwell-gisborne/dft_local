from __future__ import annotations

from dft_local.diagnostics.models import DiagnosticSpec
from dft_local.transport.bands.diagnostics import (
    compute_overview,
    diagnostics,
)


def test_bands_diagnostics_exports_overview_spec() -> None:
    specs = diagnostics()
    by_id = {spec.id: spec for spec in specs}

    assert len(specs) == 2
    assert isinstance(by_id["transport.bands.overview"], DiagnosticSpec)
    assert isinstance(by_id["transport.bands.path"], DiagnosticSpec)


def test_bands_overview_mentions_public_api() -> None:
    result = compute_overview(None, {})

    assert result.title == "Band/path continuation domain"

    table_ids = {table.id for table in result.tables}
    assert "bands_files" in table_ids
    assert "bands_api" in table_ids

    api_table = next(table for table in result.tables if table.id == "bands_api")
    names = {row.cells[0] for row in api_table.rows}

    assert "LocalPath" in names
    assert "match_via_energies" in names
    assert "match_via_overlap" in names


def test_bands_diagnostics_exports_path_spec() -> None:
    specs = diagnostics()
    by_id = {spec.id: spec for spec in specs}

    spec = by_id["transport.bands.path"]

    assert spec.title == "Band path Γ-K-M-Γ"
    assert spec.group == "transport"
    assert spec.inputs


def test_bands_path_spec_has_inputs() -> None:
    specs = diagnostics()
    by_id = {spec.id: spec for spec in specs}

    spec = by_id["transport.bands.path"]

    assert spec.title == "Band path Γ-K-M-Γ"
    assert spec.group == "transport"
    assert {inp.name for inp in spec.inputs} == {
        "kernel",
        "matching",
        "path",
        "points_per_segment",
    }
