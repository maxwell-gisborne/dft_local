"""Compatibility imports for the Boltzmann calculation diagnostics."""

from dft_local.transport.boltzmann.calculation.diagnostics import *  # noqa: F401,F403

from dft_local.transport.boltzmann.ashcroft_comparison.regions import (
    diagnostics as ashcroft_region_diagnostics,
)
from dft_local.transport.boltzmann.validation.diagnostics import (
    diagnostics as validation_diagnostics,
)


def diagnostics():
    return (
        *validation_diagnostics(),
        *ashcroft_region_diagnostics(),
    )
