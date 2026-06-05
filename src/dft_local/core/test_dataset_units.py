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
