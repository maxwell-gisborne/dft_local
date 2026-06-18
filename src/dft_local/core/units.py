"""Units, dimensions, and quantity-array schemas.

This module deliberately does not wrap hot-loop arrays.  Array fields can be
annotated with physical dimensions using ``typing.Annotated`` metadata, while
the owning SOA object carries the concrete unit context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, get_args, get_origin, get_type_hints
from types import UnionType

import numpy as np
from scipy import constants as scipy_constants


@dataclass(frozen=True, slots=True)
class Dimension:
    """Physical dimension as powers of base dimensions.

    The base order is:

    - L: length
    - E: energy
    - T: time
    - Q: charge
    - Θ: temperature

    Energy is used as a base dimension rather than mass because the DFT code
    naturally works in unit systems such as eV/angstrom/fs as well as SI.
    """

    powers: tuple[int, int, int, int, int]

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a + b for a, b in zip(self.powers, other.powers)))

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a - b for a, b in zip(self.powers, other.powers)))

    def __pow__(self, n: int) -> Dimension:
        return Dimension(tuple(n * a for a in self.powers))

    def inverse(self) -> Dimension:
        return self ** -1


DIMENSIONLESS = Dimension((0, 0, 0, 0, 0))
LENGTH = Dimension((1, 0, 0, 0, 0))
ENERGY = Dimension((0, 1, 0, 0, 0))
TIME = Dimension((0, 0, 1, 0, 0))
CHARGE = Dimension((0, 0, 0, 1, 0))
TEMPERATURE = Dimension((0, 0, 0, 0, 1))

WAVEVECTOR = LENGTH.inverse()
VELOCITY = LENGTH / TIME
INVERSE_ENERGY = ENERGY.inverse()
KSPACE_AREA = WAVEVECTOR ** 2
ACTION = ENERGY * TIME
CONDUCTIVITY = (CHARGE ** 2) / (ENERGY * TIME * LENGTH)


@dataclass(frozen=True, slots=True)
class Unit:
    """Concrete unit realising a physical dimension."""

    symbol: str
    dimension: Dimension
    scale_to_si: float

    def __mul__(self, other: Unit) -> Unit:
        return Unit(
            symbol=_join_unit_symbols(self.symbol, other.symbol, " "),
            dimension=self.dimension * other.dimension,
            scale_to_si=self.scale_to_si * other.scale_to_si,
        )

    def __truediv__(self, other: Unit) -> Unit:
        return Unit(
            symbol=_join_unit_symbols(self.symbol, other.symbol, " / "),
            dimension=self.dimension / other.dimension,
            scale_to_si=self.scale_to_si / other.scale_to_si,
        )

    def __pow__(self, n: int) -> Unit:
        if n == 0:
            return Unit("1", DIMENSIONLESS, 1.0)
        if n == 1:
            return self
        return Unit(
            symbol=f"{self.symbol}^{n}",
            dimension=self.dimension ** n,
            scale_to_si=self.scale_to_si**n,
        )

    def conversion_factor_to(self, target: Unit) -> float:
        if self.dimension != target.dimension:
            raise ValueError(
                f"Cannot convert {self.symbol} to {target.symbol}: "
                f"{self.dimension} != {target.dimension}"
            )
        return self.scale_to_si / target.scale_to_si


def _join_unit_symbols(left: str, right: str, sep: str) -> str:
    if left == "1":
        return right if sep.strip() != "/" else f"1 / {right}"
    if right == "1":
        return left
    return f"{left}{sep}{right}"


METRE = Unit("m", LENGTH, 1.0)
JOULE = Unit("J", ENERGY, 1.0)
SECOND = Unit("s", TIME, 1.0)
COULOMB = Unit("C", CHARGE, 1.0)
KELVIN = Unit("K", TEMPERATURE, 1.0)

ELECTRON_VOLT = Unit("eV", ENERGY, scipy_constants.electron_volt)
ANGSTROM = Unit("angstrom", LENGTH, scipy_constants.angstrom)
FEMTOSECOND = Unit("fs", TIME, 1.0e-15)
BOHR = Unit("bohr", LENGTH, scipy_constants.physical_constants["Bohr radius"][0])
HARTREE = Unit("hartree", ENERGY, scipy_constants.physical_constants["Hartree energy"][0])


@dataclass(frozen=True, slots=True)
class UnitContext:
    """Concrete units used by an SOA calculation state."""

    length: Unit
    energy: Unit
    time: Unit
    charge: Unit
    temperature: Unit

    def unit_for_dimension(self, dimension: Dimension) -> Unit:
        length_power, energy_power, time_power, charge_power, temperature_power = dimension.powers
        return (
            (self.length ** length_power)
            * (self.energy ** energy_power)
            * (self.time ** time_power)
            * (self.charge ** charge_power)
            * (self.temperature ** temperature_power)
        )

    def hbar(self) -> float:
        action_unit = self.unit_for_dimension(ACTION)
        return scipy_constants.hbar / action_unit.scale_to_si

    def k_b(self) -> float:
        entropy_unit = self.unit_for_dimension(ENERGY / TEMPERATURE)
        return scipy_constants.k / entropy_unit.scale_to_si

    def electron_charge(self) -> float:
        return scipy_constants.e / self.charge.scale_to_si


SI_UNITS = UnitContext(
    length=METRE,
    energy=JOULE,
    time=SECOND,
    charge=COULOMB,
    temperature=KELVIN,
)

EV_ANGSTROM_FS = UnitContext(
    length=ANGSTROM,
    energy=ELECTRON_VOLT,
    time=FEMTOSECOND,
    charge=COULOMB,
    temperature=KELVIN,
)

ATOMIC_UNITS = UnitContext(
    length=BOHR,
    energy=HARTREE,
    time=SECOND,
    charge=COULOMB,
    temperature=KELVIN,
)


@dataclass(frozen=True, slots=True)
class QuantityArray:
    """Semantic schema for an ndarray field.

    This metadata belongs on a dataclass field annotation.  It describes the
    physical dimension and axis meaning, not the concrete unit.
    """

    dimension: Dimension
    axes: tuple[str, ...]
    role: str = ""
    dtype: Any = np.floating

    @property
    def rank(self) -> int:
        return len(self.axes)


def qarray(
    dimension: Dimension,
    axes: tuple[str, ...],
    *,
    role: str = "",
    dtype: Any = np.floating,
) -> QuantityArray:
    return QuantityArray(
        dimension=dimension,
        axes=axes,
        role=role,
        dtype=dtype,
    )


def _quantity_array_spec_from_hint(hint: Any) -> QuantityArray | None:
    """Extract QuantityArray metadata from an annotation.

    Handles both direct Annotated fields and optional fields such as
    ``Annotated[np.ndarray, qarray(...)] | None``.
    """

    if get_origin(hint) is Annotated:
        _base, *metadata = get_args(hint)
        for item in metadata:
            if isinstance(item, QuantityArray):
                return item
        return None

    origin = get_origin(hint)
    if origin is UnionType or origin is getattr(__import__("typing"), "Union"):
        for arg in get_args(hint):
            spec = _quantity_array_spec_from_hint(arg)
            if spec is not None:
                return spec

    return None


def quantity_array_specs(cls: type) -> dict[str, QuantityArray]:
    """Return QuantityArray metadata attached with typing.Annotated."""

    hints = get_type_hints(cls, include_extras=True)
    specs: dict[str, QuantityArray] = {}

    for name, hint in hints.items():
        spec = _quantity_array_spec_from_hint(hint)
        if spec is not None:
            specs[name] = spec

    return specs


def validate_quantity_arrays(obj: object) -> None:
    """Validate storage-level array facts once at object construction."""

    for name, spec in quantity_array_specs(type(obj)).items():
        value = getattr(obj, name)

        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be np.ndarray, got {type(value)!r}")

        if value.ndim != spec.rank:
            raise ValueError(f"{name} has rank {value.ndim}, expected {spec.rank}")

        if not np.issubdtype(value.dtype, spec.dtype):
            raise TypeError(f"{name} has dtype {value.dtype}, expected {spec.dtype}")


@dataclass(frozen=True, slots=True)
class DisplayQuantity:
    """Standalone display value with explicit dimension and unit."""

    value: Any
    dimension: Dimension
    unit: Unit
    name: str = ""


def display_unit_symbol(unit: Unit | str) -> str:
    """Return the user-facing unit symbol for diagnostic rendering.

    Dimensionless units are displayed without a visible unit rather than as
    ``1``.
    """

    symbol = unit.symbol if isinstance(unit, Unit) else str(unit)
    return "" if symbol == "1" else symbol


def diagnostic_quantity(
    value: complex | float,
    dimension: Dimension,
    unit: Unit | str,
    name: str,
) -> DisplayQuantity:
    """Build a scalar quantity for diagnostic tables and cards."""

    return DisplayQuantity(
        value=float(np.real(value)),
        dimension=dimension,
        unit=unit,
        name=name,
    )


def diagnostic_context_quantity(
    unit_context: UnitContext,
    value: complex | float,
    dimension: Dimension,
    name: str,
) -> DisplayQuantity:
    """Build a diagnostic quantity using a UnitContext.

    Dimensionless quantities use the canonical UNITLESS unit so renderers do
    not show a literal ``1`` unit.
    """

    unit = Unit("", DIMENSIONLESS, 1.0) if dimension == DIMENSIONLESS else unit_context.unit_for_dimension(dimension)
    return diagnostic_quantity(value, dimension, unit, name)


def diagnostic_card_value(quantity: DisplayQuantity) -> str:
    """Compact string for summary cards."""

    unit = display_unit_symbol(quantity.unit)
    value = f"{quantity.value:.6g}"
    return value if unit == "" else f"{value} {unit}"




def display_quantity(obj: object, field_name: str, value: Any) -> DisplayQuantity:
    """Reify one SOA field value into a standalone display quantity."""

    specs = quantity_array_specs(type(obj))
    if field_name not in specs:
        raise KeyError(f"{type(obj).__name__}.{field_name} has no QuantityArray spec")

    units = getattr(obj, "units", None)
    if units is None:
        units = getattr(obj, "working_unit_context", None)

    if not isinstance(units, UnitContext):
        raise TypeError(
            f"{type(obj).__name__} must expose a UnitContext as "
            ".units or .working_unit_context"
        )

    spec = specs[field_name]
    return DisplayQuantity(
        value=value,
        dimension=spec.dimension,
        unit=units.unit_for_dimension(spec.dimension),
        name=field_name,
    )
