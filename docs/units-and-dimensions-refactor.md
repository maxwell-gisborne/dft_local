# Units and dimensions refactor

This note records the intended architecture for units, dimensions, quantity arrays, and diagnostic display values.

## Goal

The code should support calculations in more than one unit system without attaching unit bookkeeping to every float or array operation.

The main design principle is:

```text
SOA calculation objects own unit context
SOA field annotations define physical dimensions
arrays remain raw numerical arrays
display values become standalone quantities
unit conversion commutation tests validate dimensional stability
````

This means numerical kernels can remain fast and simple, while diagnostics and tests can reason about dimensions and units explicitly.

## Layers

There are three layers.

### 1. Dimension schema

Fields are annotated with physical dimensions and semantic array structure.

Example:

```python
@dataclass(frozen=True)
class LocalConductivityArrays:
    epsilon: Annotated[np.ndarray, qarray(ENERGY, ("k1", "k2"), role="band energy")]
    velocity: Annotated[np.ndarray, qarray(VELOCITY, ("k1", "k2", "cartesian"), role="group velocity")]
    fermi_weight: Annotated[np.ndarray, qarray(INVERSE_ENERGY, ("k1", "k2"), role="Fermi window")]
    sigma: Annotated[np.ndarray, qarray(CONDUCTIVITY, ("cartesian", "cartesian"), role="conductivity tensor")]

    units: UnitContext
```

The annotation describes what the field means. It does not describe the concrete unit.

### 2. SOA unit context

The SOA object carries the concrete unit system used by all its arrays.

Example:

```python
@dataclass(frozen=True)
class UnitContext:
    length: Unit
    mass: Unit
    time: Unit
    charge: Unit
    temperature: Unit
```

The field annotation says `epsilon` has dimension `ENERGY`.

The unit context says how `ENERGY` is realised for this object, for example eV or J.

Thus:

```text
epsilon dimension = ENERGY
epsilon unit = arrays.units.unit_for_dimension(ENERGY)
```

Units are therefore attached to the SOA state, not to each array.

### 3. Display quantities

At the diagnostic boundary, selected values are reified into standalone display quantities.

Example:

```python
DisplayQuantity(
    value=arrays.sigma[0, 0],
    dimension=CONDUCTIVITY,
    unit=arrays.units.unit_for_dimension(CONDUCTIVITY),
    name="sigma_xx",
)
```

Inside computation:

```python
arrays.sigma[0, 0]
```

is a raw float.

At display:

```python
display_quantity(arrays, "sigma", arrays.sigma[0, 0])
```

becomes value plus dimension plus unit.

## Dimension algebra

Dimensions should be immutable exponent vectors.

Suggested base dimensions:

```text
L  length
M  mass
T  time
Q  charge
Θ  temperature
```

Example implementation sketch:

```python
@dataclass(frozen=True, slots=True)
class Dimension:
    # L, M, T, Q, Θ
    powers: tuple[int, int, int, int, int]

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a + b for a, b in zip(self.powers, other.powers)))

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a - b for a, b in zip(self.powers, other.powers)))

    def __pow__(self, n: int) -> "Dimension":
        return Dimension(tuple(n * a for a in self.powers))

    def inverse(self) -> "Dimension":
        return self ** -1
```

Useful derived dimensions:

```python
DIMENSIONLESS = Dimension((0, 0, 0, 0, 0))
LENGTH = Dimension((1, 0, 0, 0, 0))
MASS = Dimension((0, 1, 0, 0, 0))
TIME = Dimension((0, 0, 1, 0, 0))
CHARGE = Dimension((0, 0, 0, 1, 0))
TEMPERATURE = Dimension((0, 0, 0, 0, 1))

ENERGY = MASS * (LENGTH ** 2) / (TIME ** 2)
WAVEVECTOR = LENGTH.inverse()
VELOCITY = LENGTH / TIME
INVERSE_ENERGY = ENERGY.inverse()
KSPACE_AREA = WAVEVECTOR ** 2
CONDUCTIVITY = (CHARGE ** 2) * TIME / (MASS * (LENGTH ** 3))
```

For 2D sheet conductivity or conductance-like quantities, define separate dimensions explicitly rather than overloading 3D conductivity.

## Quantity array metadata

`qarray` should describe the semantic contract of an array field.

Recommended minimal form:

```python
@dataclass(frozen=True, slots=True)
class QuantityArray:
    dimension: Dimension
    axes: tuple[str, ...]
    role: str = ""
    dtype: object = np.floating

    @property
    def rank(self) -> int:
        return len(self.axes)


def qarray(
    dimension: Dimension,
    axes: tuple[str, ...],
    *,
    role: str = "",
    dtype: object = np.floating,
) -> QuantityArray:
    return QuantityArray(
        dimension=dimension,
        axes=axes,
        role=role,
        dtype=dtype,
    )
```

Use `axes`, not just `rank`, because axes describe meaning.

Examples:

```python
qarray(ENERGY, ("k1", "k2"), role="band energy")
qarray(VELOCITY, ("k1", "k2", "cartesian"), role="group velocity")
qarray(CONDUCTIVITY, ("cartesian", "cartesian"), role="conductivity tensor")
```

Do not put concrete units in `qarray`.

Good:

```python
qarray(ENERGY, ("k1", "k2"))
```

Bad:

```python
qarray(ENERGY, ("k1", "k2"), unit="eV")
```

The unit belongs to the nearest `UnitContext`.

## Unit algebra

A `Unit` is a concrete realisation of a dimension.

```python
@dataclass(frozen=True, slots=True)
class Unit:
    symbol: str
    dimension: Dimension
    scale_to_si: float

    def __mul__(self, other: "Unit") -> "Unit":
        ...

    def __truediv__(self, other: "Unit") -> "Unit":
        ...

    def __pow__(self, n: int) -> "Unit":
        ...
```

A unit context can derive a unit for any dimension:

```python
@dataclass(frozen=True)
class UnitContext:
    length: Unit
    mass: Unit
    time: Unit
    charge: Unit
    temperature: Unit

    def unit_for_dimension(self, dim: Dimension) -> Unit:
        l, m, t, q, theta = dim.powers
        return (
            (self.length ** l)
            * (self.mass ** m)
            * (self.time ** t)
            * (self.charge ** q)
            * (self.temperature ** theta)
        )
```

This allows one SOA to be in SI and another to be in eV/angstrom/fs, while the field dimensions remain the same.

## Constants

Use SciPy constants as source values for SI constants.

Example:

```python
from scipy import constants as C

HBAR_SI = C.hbar
K_B_SI = C.k
E_CHARGE_SI = C.e
EV_SI = C.electron_volt
ANGSTROM_SI = C.angstrom
```

Then derive constants in the current unit context:

```python
def hbar_in(units: UnitContext) -> float:
    return HBAR_SI / (units.energy.scale_to_si * units.time.scale_to_si)
```

If `UnitContext` only stores base units, `energy` can be derived from dimension:

```python
energy_unit = units.unit_for_dimension(ENERGY)
time_unit = units.unit_for_dimension(TIME)
hbar = HBAR_SI / (energy_unit.scale_to_si * time_unit.scale_to_si)
```

## Validation by unit commutation

The core test principle is:

```text
convert inputs then compute
=
compute then convert outputs
```

For calculation `F`:

```text
x_A -> F_A -> y_A
x_A -> convert A to B -> x_B -> F_B -> y_B -> convert B to A -> y_A'
```

Then assert:

```text
y_A ≈ y_A'
```

This validates dimensional stability without forcing a single canonical unit system.

This should be applied first to:

```text
velocity
Fermi window
k-space measure
conductivity tensor
```

The `2π` convention should be tested separately:

```text
raw_grid_area / continuum_dos = (2 pi)^2
```

## Runtime overhead

Using `Annotated[np.ndarray, qarray(...)]` has no hot-loop overhead.

The array remains a normal `np.ndarray`.

The metadata lives on the class annotation.

Validation cost is construction-time only and scales with number of fields, not number of elements.

Display conversion cost happens only for values being displayed.

## Design rule

```text
Field annotation = physical dimension and array meaning
SOA unit context = concrete unit system
SOA arrays = raw numerical storage
DisplayQuantity = standalone value with dimension and unit
```

This keeps numerical code fast while making diagnostics, display, and unit-conversion tests principled.
EOF


