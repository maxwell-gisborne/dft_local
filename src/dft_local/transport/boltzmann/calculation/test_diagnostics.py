from __future__ import annotations

from dft_local.diagnostics.models import Card, DiagnosticSpec, Table
from dft_local.transport.boltzmann.calculation.diagnostics import (
    compute_overview,
    diagnostics,
)


def test_boltzmann_diagnostics_exports_specs() -> None:
    specs = diagnostics()
    ids = {spec.id for spec in specs}

    assert all(isinstance(spec, DiagnosticSpec) for spec in specs)
    assert ids == {
        "transport.boltzmann.calculation.overview",
        "transport.boltzmann.calculation.conductivity",
    }


def test_boltzmann_overview_mentions_domain_files() -> None:
    result = compute_overview(None, {})

    assert result.title == "Boltzmann conductivity domain"
    assert any(isinstance(block, Card) for block in result.body)

    tables = tuple(block for block in result.body if isinstance(block, Table))
    table_ids = {table.id for table in (*result.tables, *tables)}

    assert "boltzmann_files" in table_ids
    assert "boltzmann_docs_preview" in table_ids

    file_table = next(table for table in (*result.tables, *tables) if table.id == "boltzmann_files")
    roles = {row.cells[0] for row in file_table.rows}

    assert "core" in roles
    assert "documentation" in roles
    assert "diagnostics" in roles
    assert "test metadata" in roles
