"""Compatibility imports for the Boltzmann calculation diagnostics."""

from dft_local.transport.boltzmann.calculation.diagnostics import *  # noqa: F401,F403


from dft_local.transport.boltzmann.validation.diagnostics import (
    diagnostics as validation_diagnostics,
)


def diagnostics():
    return (
        *validation_diagnostics(),
    )
