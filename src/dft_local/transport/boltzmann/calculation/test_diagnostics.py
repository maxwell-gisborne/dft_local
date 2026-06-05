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


def test_boltzmann_conductivity_array_fields_have_quantity_schema() -> None:
    from dft_local.core.units import CONDUCTIVITY, ENERGY, VELOCITY, WAVEVECTOR, quantity_array_specs
    from dft_local.transport.boltzmann.calculation.core import (
        BoltzmannConductivity,
        BoltzmannSampleResult,
    )

    sample_specs = quantity_array_specs(BoltzmannSampleResult)
    assert sample_specs["energies"].dimension == ENERGY
    assert sample_specs["energies"].axes == ("band",)
    assert sample_specs["velocities"].dimension == VELOCITY
    assert sample_specs["velocities"].axes == ("cartesian", "band")
    assert sample_specs["sigma"].dimension == CONDUCTIVITY

    calc_specs = quantity_array_specs(BoltzmannConductivity)
    assert calc_specs["irrep_to_physical_k"].dimension == WAVEVECTOR
    assert calc_specs["irrep_to_physical_k"].axes == ("cartesian", "irrep_coordinate")
    assert calc_specs["sigma"].dimension == CONDUCTIVITY
