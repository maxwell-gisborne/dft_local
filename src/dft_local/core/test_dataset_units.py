from __future__ import annotations

from dft_local.core.dataset import LEGACY_EV_ANGSTROM_CONTEXT, SparseDataset
from dft_local.core.units import ATOMIC_UNITS


def test_legacy_ev_angstrom_context_preserves_existing_disk_conversions() -> None:
    context = LEGACY_EV_ANGSTROM_CONTEXT

    assert context.length.symbol == "angstrom"
    assert context.energy.symbol == "eV"
    assert ATOMIC_UNITS.energy.scale_to_si / context.energy.scale_to_si == 27.21138386
    assert ATOMIC_UNITS.length.scale_to_si / context.length.scale_to_si == 0.52917721092


def test_sparse_dataset_exposes_disk_and_working_unit_contexts() -> None:
    dataset = SparseDataset.load("test_run/run_dir/data")

    assert dataset.disk_unit_context == ATOMIC_UNITS
    assert dataset.working_unit_context.energy.symbol == "eV"
    assert dataset.working_unit_context.length.symbol == "angstrom"



def test_sparse_metadata_exposes_working_unit_context_for_positions() -> None:
    dataset = SparseDataset.load("test_run/run_dir/data")

    assert dataset.metadata.working_unit_context == dataset.working_unit_context
    assert dataset.metadata.working_unit_context.energy.symbol == "eV"
    assert dataset.metadata.working_unit_context.length.symbol == "angstrom"



def test_sparse_metadata_positions_have_quantity_schema() -> None:
    from dft_local.core.units import LENGTH, display_quantity, quantity_array_specs

    dataset = SparseDataset.load("test_run/run_dir/data")

    specs = quantity_array_specs(type(dataset.metadata))
    assert specs["positions"].dimension == LENGTH
    assert specs["positions"].axes == ("atom", "cartesian")

    quantity = display_quantity(dataset.metadata, "positions", dataset.metadata.positions[0, 0])
    assert quantity.dimension == LENGTH
    assert quantity.unit == dataset.metadata.working_unit_context.length



def test_sparse_dataset_matrices_have_quantity_schema() -> None:
    from dft_local.core.units import DIMENSIONLESS, ENERGY, quantity_array_specs

    specs = quantity_array_specs(SparseDataset)

    assert specs["H"].dimension == ENERGY
    assert specs["H"].axes == ("basis", "basis")
    assert specs["S"].dimension == DIMENSIONLESS
    assert specs["S"].axes == ("basis", "basis")



def test_sparse_dataset_exposes_disk_to_working_conversion_factors() -> None:
    dataset = SparseDataset.load("test_run/run_dir/data")

    assert dataset.energy_conversion_disk_to_working == dataset.disk_unit_context.energy.scale_to_si / dataset.working_unit_context.energy.scale_to_si
    assert dataset.length_conversion_disk_to_working == dataset.disk_unit_context.length.scale_to_si / dataset.working_unit_context.length.scale_to_si
    assert dataset.energy_conversion_disk_to_working > 1.0
    assert dataset.length_conversion_disk_to_working < 1.0



def test_sparse_dataset_load_matches_legacy_conversion_factors() -> None:
    dataset = SparseDataset.load("test_run/run_dir/data")

    assert dataset.energy_conversion_disk_to_working == 27.21138386
    assert dataset.length_conversion_disk_to_working == 0.52917721092
