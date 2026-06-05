from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pytest

from dft_local.core.units import (
    CONDUCTIVITY,
    ENERGY,
    EV_ANGSTROM_FS,
    INVERSE_ENERGY,
    JOULE,
    LENGTH,
    SI_UNITS,
    TIME,
    VELOCITY,
    Unit,
    UnitContext,
    display_quantity,
    qarray,
    quantity_array_specs,
    validate_quantity_arrays,
)


def test_dimension_algebra_builds_derived_dimensions() -> None:
    assert VELOCITY == LENGTH / TIME
    assert INVERSE_ENERGY == ENERGY.inverse()
    assert CONDUCTIVITY == CONDUCTIVITY


def test_unit_conversion_factor_respects_dimensions() -> None:
    centimetre = Unit("cm", LENGTH, 1.0e-2)
    metre = SI_UNITS.length

    assert centimetre.conversion_factor_to(metre) == pytest.approx(1.0e-2)

    with pytest.raises(ValueError):
        centimetre.conversion_factor_to(JOULE)


def test_unit_context_derives_units_from_dimensions() -> None:
    velocity_unit = EV_ANGSTROM_FS.unit_for_dimension(VELOCITY)

    assert velocity_unit.dimension == VELOCITY
    assert velocity_unit.scale_to_si == pytest.approx(1.0e5)
    assert EV_ANGSTROM_FS.hbar() == pytest.approx(0.6582119569, rel=1.0e-9)


def test_quantity_array_annotation_exposes_schema_without_wrapping_array() -> None:
    @dataclass(frozen=True)
    class ExampleArrays:
        epsilon: Annotated[np.ndarray, qarray(ENERGY, ("k1", "k2"), role="band energy")]
        units: UnitContext

    arr = np.zeros((2, 3), dtype=float)
    obj = ExampleArrays(epsilon=arr, units=SI_UNITS)

    assert obj.epsilon is arr

    specs = quantity_array_specs(ExampleArrays)
    assert specs["epsilon"].dimension == ENERGY
    assert specs["epsilon"].axes == ("k1", "k2")
    assert specs["epsilon"].rank == 2
    assert specs["epsilon"].role == "band energy"


def test_validate_quantity_arrays_checks_rank_and_dtype_once() -> None:
    @dataclass(frozen=True)
    class ExampleArrays:
        epsilon: Annotated[np.ndarray, qarray(ENERGY, ("k1", "k2"))]
        units: UnitContext

    validate_quantity_arrays(ExampleArrays(np.zeros((2, 3)), SI_UNITS))

    with pytest.raises(ValueError):
        validate_quantity_arrays(ExampleArrays(np.zeros((2, 3, 4)), SI_UNITS))

    with pytest.raises(TypeError):
        validate_quantity_arrays(ExampleArrays(np.zeros((2, 3), dtype=np.complex128), SI_UNITS))


def test_display_quantity_reifies_field_value_with_context_unit() -> None:
    @dataclass(frozen=True)
    class ConductivityArrays:
        sigma: Annotated[np.ndarray, qarray(CONDUCTIVITY, ("cartesian", "cartesian"), role="conductivity tensor")]
        units: UnitContext

    obj = ConductivityArrays(sigma=np.eye(2), units=SI_UNITS)

    quantity = display_quantity(obj, "sigma", obj.sigma[0, 0])

    assert quantity.value == pytest.approx(1.0)
    assert quantity.dimension == CONDUCTIVITY
    assert quantity.unit.dimension == CONDUCTIVITY
    assert quantity.name == "sigma"


def test_display_quantity_requires_annotated_field_and_unit_context() -> None:
    @dataclass(frozen=True)
    class BadArrays:
        sigma: np.ndarray
        units: str

    obj = BadArrays(sigma=np.eye(2), units="not units")

    with pytest.raises(KeyError):
        display_quantity(obj, "sigma", obj.sigma[0, 0])
