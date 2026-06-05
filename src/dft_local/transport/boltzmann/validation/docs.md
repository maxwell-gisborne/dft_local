
# Boltzmann operator validation

This domain collects diagnostics that validate the Boltzmann operator approach
independently of any one reference implementation.

The purpose is to separate two questions:

1. Does the local operator formulation behave correctly on analytic and algebraic tests?
2. Which conventions are needed to match an external calculation?

The first question belongs here. The second belongs in comparison domains such as
`transport.boltzmann.ashcroft_comparison`.

Planned checks include:

- identity and shape checks for operator objects
- linearity checks
- positive-semidefinite tensor checks
- basis-change covariance checks
- known-function conductivity tests
- grid-measure and normalisation checks
- relaxation-time and velocity-scale laws
