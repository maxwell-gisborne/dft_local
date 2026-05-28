# Boltzmann conductivity module

This module owns the band-diagonal AC Boltzmann conductivity calculation.

The calculation is local in `k` samples and does not require band continuation.

For each sample:

1. Build generalized symbols `H(k)` and `S(k)`
2. Solve the generalized eigenproblem `H U = S U E`
3. Build derivative symbols with respect to physical `k`
4. Compute diagonal generalized Hellmann-Feynman velocities
5. Compute the AC Boltzmann Fermi-window weights
6. Accumulate `sigma_ij(k)`
7. Integrate over the sample weights

The physical derivative must be with respect to physical wave-vector units, not
just raw irrep coordinates.  If `k_physical = J alpha`, then

    partial / partial k_i = sum_a (J^-1)_(a i) partial / partial alpha_a

and the integration measure uses

    d^d k = abs(det J) d^d alpha

The semiclassical Boltzmann expression uses the diagonal energy-basis
velocities.  Full velocity-matrix products include interband matrix elements
and represent a different, more Kubo-like object.
