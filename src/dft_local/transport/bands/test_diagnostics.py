from __future__ import annotations

from dft_local.diagnostics.models import DiagnosticSpec
from dft_local.transport.bands.diagnostics import (
    compute_overview,
    diagnostics,
)


def test_bands_diagnostics_exports_overview_spec() -> None:
    specs = diagnostics()

    assert len(specs) == 1
    assert isinstance(specs[0], DiagnosticSpec)
    assert specs[0].id == "transport.bands.overview"


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
