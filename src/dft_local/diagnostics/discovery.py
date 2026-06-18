"""Diagnostic discovery for the dft_local package.

dft_local uses local structured diagnostic models and explicit diagnostic
discovery. Diagnostic modules expose a plain `diagnostics()` function instead
of mutating a global registry during import.
"""

from __future__ import annotations

from dft_local.diagnostics.models import DiagnosticSpec

from dft_local.core.discovery import collect_from_modules, require_unique


DEFAULT_DIAGNOSTIC_MODULES = (
    "dft_local.testsuite.diagnostics",
    "dft_local.transport.boltzmann.calculation.diagnostics",
    "dft_local.transport.boltzmann.ashcroft_comparison.diagnostics",
    "dft_local.transport.boltzmann.ashcroft_comparison.regions",
    "dft_local.transport.boltzmann.validation.diagnostics",
    "dft_local.transport.boltzmann.group_resolved.diagnostics",
    "dft_local.transport.symmetry_audit.diagnostics",
    "dft_local.transport.bands.diagnostics",
)


def load_diagnostics(
    module_names: tuple[str, ...] = DEFAULT_DIAGNOSTIC_MODULES,
) -> tuple[DiagnosticSpec, ...]:
    """Load all diagnostic specs from explicit module names."""

    specs = collect_from_modules(
        module_names,
        "diagnostics",
        item_type=DiagnosticSpec,
    )

    return require_unique(specs, key=lambda spec: spec.id, name="diagnostic id")
