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



def test_boltzmann_conductivity_units_table_uses_display_quantity() -> None:
    from dft_local.core.units import DisplayQuantity, ENERGY
    from dft_local.transport.boltzmann.calculation.diagnostics import unit_rows
    from dft_local.transport.boltzmann.calculation.core import BoltzmannConductivity

    np = __import__("numpy")
    calc = BoltzmannConductivity(
        problems=[],
        irrep_points=np.zeros((0, 1)),
        irrep_weights=np.zeros((0,)),
        irrep_to_physical_k=np.eye(1),
        temperature=300.0,
    )

    rows = unit_rows(calc)
    row_map = {row[0]: row[1] for row in rows}

    assert isinstance(row_map["k_B T"], DisplayQuantity)
    assert row_map["k_B T"].dimension == ENERGY
    assert row_map["k_B T"].unit.symbol == "eV"



def test_boltzmann_conductivity_exposes_unit_context_bridge() -> None:
    from dft_local.core.units import EV_ANGSTROM_FS
    from dft_local.transport.boltzmann.calculation.core import BoltzmannConductivity

    assert BoltzmannConductivity.__dataclass_fields__["units"].default == BoltzmannConductivity(
        problems=[],
        irrep_points=__import__("numpy").zeros((0, 1)),
        irrep_weights=__import__("numpy").zeros((0,)),
        irrep_to_physical_k=__import__("numpy").eye(1),
    ).units

    calc = BoltzmannConductivity(
        problems=[],
        irrep_points=__import__("numpy").zeros((0, 1)),
        irrep_weights=__import__("numpy").zeros((0,)),
        irrep_to_physical_k=__import__("numpy").eye(1),
    )

    assert calc.unit_context == EV_ANGSTROM_FS



def test_boltzmann_units_table_marks_basic_inputs_as_display_quantities() -> None:
    import numpy as np

    from dft_local.core.units import DisplayQuantity, ENERGY, TEMPERATURE, TIME
    from dft_local.transport.boltzmann.calculation.core import BoltzmannConductivity
    from dft_local.transport.boltzmann.calculation.diagnostics import unit_rows

    calc = BoltzmannConductivity(
        problems=[],
        irrep_points=np.zeros((0, 1)),
        irrep_weights=np.zeros((0,)),
        irrep_to_physical_k=np.eye(1),
        mu=0.25,
        temperature=310.0,
        omega=2.0,
    )

    row_map = {row[0]: row[1] for row in unit_rows(calc)}

    assert isinstance(row_map["mu"], DisplayQuantity)
    assert row_map["mu"].dimension == ENERGY

    assert isinstance(row_map["temperature"], DisplayQuantity)
    assert row_map["temperature"].dimension == TEMPERATURE

    assert isinstance(row_map["omega"], DisplayQuantity)
    assert row_map["omega"].dimension == TIME.inverse()



def test_boltzmann_sigma_display_values_carry_conductivity_units() -> None:
    import numpy as np

    from dft_local.core.units import CONDUCTIVITY, DisplayQuantity
    from dft_local.transport.boltzmann.calculation.diagnostics import sigma_matrix, sigma_rows
    from dft_local.transport.boltzmann.calculation.core import BoltzmannConductivity

    calc = BoltzmannConductivity(
        problems=[],
        irrep_points=np.zeros((0, 1)),
        irrep_weights=np.zeros((0,)),
        irrep_to_physical_k=np.eye(1),
    )
    object.__setattr__(calc, "energies", np.zeros((0, 1)))
    object.__setattr__(calc, "vectors", np.zeros((0, 1, 1), dtype=np.complex128))
    object.__setattr__(calc, "velocities", np.zeros((0, 1, 1)))
    object.__setattr__(calc, "ac_weights", np.zeros((0, 1), dtype=np.complex128))
    object.__setattr__(calc, "sigma_k", np.zeros((0, 1, 1), dtype=np.complex128))
    object.__setattr__(calc, "sigma", np.array([[1.0 + 2.0j]], dtype=np.complex128))

    rows = sigma_rows(calc)
    assert isinstance(rows[0][2], DisplayQuantity)
    assert rows[0][2].dimension == CONDUCTIVITY
    assert rows[0][3].dimension == CONDUCTIVITY
    assert rows[0][4].dimension == CONDUCTIVITY

    matrix = sigma_matrix(calc)
    assert isinstance(matrix.cells[0].value, DisplayQuantity)
    assert matrix.cells[0].value.dimension == CONDUCTIVITY



def test_velocity_quantile_rows_carry_velocity_units() -> None:
    import numpy as np

    from dft_local.core.units import DisplayQuantity, VELOCITY
    from dft_local.transport.boltzmann.calculation.core import BoltzmannConductivity
    from dft_local.transport.boltzmann.calculation.diagnostics import velocity_quantile_rows

    calc = BoltzmannConductivity(
        problems=[],
        irrep_points=np.zeros((0, 2)),
        irrep_weights=np.zeros((0,)),
        irrep_to_physical_k=np.eye(2),
    )
    object.__setattr__(calc, "energies", np.zeros((2, 1)))
    object.__setattr__(calc, "vectors", np.zeros((2, 1, 1), dtype=np.complex128))
    object.__setattr__(calc, "velocities", np.array([[[1.0], [2.0]], [[3.0], [4.0]]]))
    object.__setattr__(calc, "ac_weights", np.zeros((2, 1), dtype=np.complex128))
    object.__setattr__(calc, "sigma_k", np.zeros((2, 2, 2), dtype=np.complex128))
    object.__setattr__(calc, "sigma", np.zeros((2, 2), dtype=np.complex128))

    rows = velocity_quantile_rows(calc)

    assert isinstance(rows[0][1], DisplayQuantity)
    assert rows[0][1].dimension == VELOCITY
    assert rows[0][5].dimension == VELOCITY



def test_energy_quantile_rows_carry_energy_units() -> None:
    import numpy as np

    from dft_local.core.units import DisplayQuantity, ENERGY
    from dft_local.transport.boltzmann.calculation.core import BoltzmannConductivity
    from dft_local.transport.boltzmann.calculation.diagnostics import energy_quantile_rows

    calc = BoltzmannConductivity(
        problems=[],
        irrep_points=np.zeros((0, 1)),
        irrep_weights=np.zeros((0,)),
        irrep_to_physical_k=np.eye(1),
    )
    object.__setattr__(calc, "energies", np.array([[1.0, 2.0], [3.0, 4.0]]))
    object.__setattr__(calc, "vectors", np.zeros((2, 1, 2), dtype=np.complex128))
    object.__setattr__(calc, "velocities", np.zeros((2, 1, 2)))
    object.__setattr__(calc, "ac_weights", np.zeros((2, 2), dtype=np.complex128))
    object.__setattr__(calc, "sigma_k", np.zeros((2, 1, 1), dtype=np.complex128))
    object.__setattr__(calc, "sigma", np.zeros((1, 1), dtype=np.complex128))

    rows = energy_quantile_rows(calc)

    assert isinstance(rows[0][1], DisplayQuantity)
    assert rows[0][1].dimension == ENERGY
    assert rows[0][5].dimension == ENERGY



def test_near_fermi_rows_carry_units_for_physical_columns() -> None:
    import numpy as np

    from dft_local.core.units import DisplayQuantity, ENERGY, VELOCITY, WAVEVECTOR
    from dft_local.transport.boltzmann.calculation.core import BoltzmannConductivity
    from dft_local.transport.boltzmann.calculation.diagnostics import near_fermi_rows

    calc = BoltzmannConductivity(
        problems=[object()],
        irrep_points=np.array([[0.0, 0.0]]),
        irrep_weights=np.array([1.0]),
        irrep_to_physical_k=np.eye(2),
        mu=1.5,
    )
    object.__setattr__(calc, "energies", np.array([[1.0]]))
    object.__setattr__(calc, "vectors", np.zeros((1, 1, 1), dtype=np.complex128))
    object.__setattr__(calc, "velocities", np.array([[[2.0], [3.0]]]))
    object.__setattr__(calc, "ac_weights", np.ones((1, 1), dtype=np.complex128))
    object.__setattr__(calc, "sigma_k", np.zeros((1, 2, 2), dtype=np.complex128))
    object.__setattr__(calc, "sigma", np.zeros((2, 2), dtype=np.complex128))

    row = near_fermi_rows(calc, n=1)[0]

    assert isinstance(row[4], DisplayQuantity)
    assert row[4].dimension == WAVEVECTOR
    assert row[5].dimension == WAVEVECTOR
    assert row[6].dimension == ENERGY
    assert row[7].dimension == ENERGY
    assert row[8].dimension == VELOCITY
    assert row[9].dimension == VELOCITY



def test_worst_sample_rows_carry_units_for_physical_columns() -> None:
    import numpy as np

    from dft_local.core.units import DisplayQuantity, ENERGY, KSPACE_AREA, VELOCITY, WAVEVECTOR
    from dft_local.transport.boltzmann.calculation.core import BoltzmannConductivity
    from dft_local.transport.boltzmann.calculation.diagnostics import worst_sample_rows

    calc = BoltzmannConductivity(
        problems=[object()],
        irrep_points=np.array([[0.0, 0.0]]),
        irrep_weights=np.array([1.0]),
        irrep_to_physical_k=np.eye(2),
    )
    object.__setattr__(calc, "energies", np.array([[1.0]]))
    object.__setattr__(calc, "vectors", np.zeros((1, 1, 1), dtype=np.complex128))
    object.__setattr__(calc, "velocities", np.array([[[2.0], [3.0]]]))
    object.__setattr__(calc, "ac_weights", np.ones((1, 1), dtype=np.complex128))
    object.__setattr__(calc, "sigma_k", np.ones((1, 2, 2), dtype=np.complex128))
    object.__setattr__(calc, "sigma", np.zeros((2, 2), dtype=np.complex128))

    row = worst_sample_rows(calc, n=1)[0]

    assert isinstance(row[3], DisplayQuantity)
    assert row[3].dimension == WAVEVECTOR
    assert row[4].dimension == WAVEVECTOR
    assert row[5].dimension == ENERGY
    assert row[6].dimension == ENERGY
    assert row[7].dimension == VELOCITY
    assert row[10].dimension == KSPACE_AREA



def test_unit_rows_physical_k_weight_sum_carries_kspace_area_units() -> None:
    import numpy as np

    from dft_local.core.units import DisplayQuantity, KSPACE_AREA
    from dft_local.transport.boltzmann.calculation.core import BoltzmannConductivity
    from dft_local.transport.boltzmann.calculation.diagnostics import unit_rows

    calc = BoltzmannConductivity(
        problems=[],
        irrep_points=np.zeros((0, 2)),
        irrep_weights=np.zeros((0,)),
        irrep_to_physical_k=np.eye(2),
    )

    row_map = {row[0]: row[1] for row in unit_rows(calc)}

    assert isinstance(row_map["sum physical k weights"], DisplayQuantity)
    assert row_map["sum physical k weights"].dimension == KSPACE_AREA



def test_boltzmann_diagnostic_physical_values_render_as_quantities() -> None:
    from dft_local.diagnostics.render import render_result
    from dft_local.diagnostics.server import load_default_context
    from dft_local.transport.boltzmann.calculation.diagnostics import compute_conductivity

    result = compute_conductivity(
        load_default_context("test_run/run_dir/data"),
        {
            "kernel": "average_star",
            "temperature": "300",
            "mu": "0",
            "tau": "1",
            "omega": "0",
            "nu": "3",
            "nv": "3",
            "central_bz": "0",
            "run": "1",
            "rows": "5",
            "k_scale": "0",
        },
    )

    html = render_result(result)

    assert "display-quantity" in html
    assert "data-unit='eV'" in html
    assert "data-unit='K'" in html
    assert "data-unit='angstrom fs^-1'" in html
    assert "data-unit='angstrom^-1'" in html
    assert "data-unit='angstrom^-2'" in html



def test_boltzmann_conductivity_can_use_dataset_unit_context_override() -> None:
    import numpy as np

    from dft_local.core.units import SI_UNITS
    from dft_local.transport.boltzmann.calculation.core import BoltzmannConductivity

    calc = BoltzmannConductivity(
        problems=[],
        irrep_points=np.zeros((0, 1)),
        irrep_weights=np.zeros((0,)),
        irrep_to_physical_k=np.eye(1),
        unit_context_override=SI_UNITS,
    )

    assert calc.unit_context == SI_UNITS
