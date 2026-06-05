from __future__ import annotations

from dft_local.core.dataset import AU, SparseDataset, eVag, unit_context_from_legacy_units
from dft_local.core.units import ATOMIC_UNITS, EV_ANGSTROM_FS


def test_legacy_unit_context_bridge_maps_known_units() -> None:
    assert unit_context_from_legacy_units(eVag) == EV_ANGSTROM_FS
    assert unit_context_from_legacy_units(AU) == ATOMIC_UNITS


def test_sparse_dataset_exposes_disk_and_working_unit_contexts() -> None:
    dataset = SparseDataset.load("test_run/run_dir/data")

    assert dataset.disk_unit_context == ATOMIC_UNITS
    assert dataset.working_unit_context == EV_ANGSTROM_FS



def test_sparse_metadata_exposes_working_unit_context_for_positions() -> None:
    dataset = SparseDataset.load("test_run/run_dir/data")

    assert dataset.metadata.working_unit_context == dataset.working_unit_context
    assert dataset.metadata.working_unit_context == EV_ANGSTROM_FS



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
