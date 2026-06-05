# Ashcroft comparison

Diagnostic page: `/d/transport.boltzmann.ashcroft_comparison.overview`


This domain is for reproducing Vincent's Ashcroft-style Boltzmann conductivity result.

## Target

The first milestone is to reproduce the recorded reference calculation:

    T = 300 K
    tau = 1e-14 s
    E = [1e5, 0] V / m

The target conductivity tensor is:

    [[ 6.45179383e-02 -8.80479820e-05]
     [-8.73823365e-05  6.44024548e-02]]

in S/m.

The recorded Fermi-window statistics are:

    max f(1-f)  = 2.499e-01
    min f(1-f)  = 0.000e+00
    mean f(1-f) = 3.907e-03

## Purpose

The main `transport.boltzmann.calculation` domain owns the local implementation.

This domain owns comparison work:

- preserve Vincent's reference values
- reproduce the constants and units
- compare tensor components directly
- identify unit, normalisation, k-grid, or chemical-potential differences
- keep reference reproduction separate from production calculation code

## Diagnostic ids

    transport.boltzmann.ashcroft_comparison.overview


## Current reproduction status

The k-grid convention is now identified:

- Vincent's listed k-points advance along `b2 / 100`
- the primitive lattice and reciprocal lattice convention are matched

The velocity convention is not yet reproduced.

Tested and rejected as complete explanations:

- sign flip
- transposed epsilon grid
- forward, backward, central, and `np.gradient` stencils
- semiclassical electric-field shift from `E`, `tau`, and `hbar`
- Vincent's printed shifted k as a directional finite difference
- nearby grid offsets
- baseline velocity subtraction

The best simple local probe is central Cartesian finite difference on the matched k-grid, but it still has an RMS error of about `1e4 m/s` against Vincent's first five reported velocity samples.

## Tried conventions log

| convention | result |
| --- | --- |
| primitive / reciprocal basis | matched: Vincent k-points advance along `b2 / 100` |
| plain Cartesian gradient | rejected |
| epsilon grid transpose | rejected |
| sign flip | rejected |
| forward / backward / central stencils | rejected |
| `np.gradient`, `edge_order=1` and `edge_order=2` | rejected |
| semiclassical `E tau / hbar` shift | rejected: direction/magnitude differs from Vincent printout |
| Vincent printed shifted-k directional difference | partial: improves `v_y` scale but cannot produce `v_x` |
| nearby grid-index offsets | rejected: best offset is still row 0, column 0 |
| baseline velocity subtraction | rejected |
| central Cartesian finite difference | best simple probe so far, still mismatched |

## Derivative test status

The local derivative machinery is now tested against known linear functions.

A bug was found and fixed in the reciprocal-basis transform: because reciprocal vectors are stored as rows, the conversion is:

    grad_k = grad_q @ inv(B.T)

not:

    grad_k = grad_q @ inv(B)

After this fix, the local derivative reproduces analytic linear test functions, but Vincent's reported sample velocities still do not match. The remaining mismatch is therefore probably in Vincent's velocity convention, interpolation convention, or source data path, rather than a basic reciprocal-basis derivative bug.


## Systematic velocity error

The printed Vincent k-points identify the sample path:

    epsilon[0, j] -> k[j] = j * b2 / 100

So the comparison is not ambiguous at the k-point level.

The diagnostic now includes a `Velocity systematic error` table. It compares Vincent's reported sample velocities with the local derivative evaluated at the same confirmed k-points and reports:

- absolute delta in `v_x` and `v_y`
- percentage error
- step-to-step changes in the delta

This is intended to show that the mismatch is structured. The local derivative passes analytic linear-function tests, so the remaining issue is likely a velocity convention, interpolation convention, or missing detail from Vincent's original velocity routine.


## Offset search status

The offset hypothesis was tested directly.

Two searches were run:

- nearest local velocity anywhere on the `100 x 100` grid for each Vincent sample
- best straight grid path of the form `(row, col) = (row0, col0) + j * (drow, dcol)`

The best straight path was:

    start = (0, 0)
    step  = (0, 1)

This is exactly the confirmed sample path:

    epsilon[0, j] -> k[j] = j * b2 / 100

So the mismatch is not explained by a simple row, column, or path offset.


## Shifted k-point discrepancy

Vincent's notes include a printed shifted k-point:

    [0, -29498522.56891833]

Using the stated constants, the usual semiclassical shift would be:

    delta k = -e E tau / hbar

With `E = [1e5, 0] V/m` and `tau = 1e-14 s`, this shift points in the `x` direction, not the `y` direction.

So the printed shifted point is not explained by the stated electric field and relaxation time alone. The diagnostic includes a `Shifted k-point discrepancy` table to make this mismatch explicit.


## Shift unit-error hypotheses

The printed shifted k-point is important because it does not match the shift implied by the stated constants.

The usual semiclassical shift is:

    delta k = -e E tau / hbar

With `E = [1e5, 0] V/m` and `tau = 1e-14 s`, this points in the `x` direction.

Vincent's printed shifted point is:

    [0, -29498522.56891833]

which points in the `y` direction.

The diagnostic now checks common unit mistakes involving `e`, `hbar`, `h`, Bohr conversion, and repeated/missing unit conversions. These can change the magnitude of the shift, but they do not explain a rotation from the `x` direction to the `y` direction.


## Shift axis-swap hypothesis

The shifted-k discrepancy has two parts:

1. The expected nonzero component is in `k_x`, because the stated electric field is `E = [1e5, 0] V/m`.
2. Vincent's printed shifted point has its nonzero component in `k_y`.

This suggests a possible axis swap. However, a pure axis swap is not enough, because the magnitude also differs.

The diagnostic includes a `Shift axis-swap probe` table comparing:

    reported_y / expected_x

If this ratio were close to `1`, an axis swap alone would explain the shift. Instead it is much larger, so the evidence points to an axis swap plus an additional scale or unit convention mismatch.


## Velocity axis-swap check

Swapping the electric-field axes explains the direction of Vincent's printed shifted k-point:

    E = [1e5, 0] -> expected shift in k_x
    swapped E = [0, 1e5] -> expected shift in k_y

However, the magnitude still differs by a factor of about:

    19.416

A separate velocity-axis-swap check was also run. Swapping `k_x, k_y` and/or swapping velocity output components gives only a small improvement in the velocity RMS error. Therefore an axis swap alone does not explain the reported velocity samples.
