"""Test discovery for the dft_local package.

Domain modules may expose a function

    pytest_files() -> list[str]

returning test file paths or pytest node ids.  This module gathers those paths
without importing any domain implementation into a central test registry.
"""

from __future__ import annotations

from dft_local.core.discovery import collect_from_modules


DEFAULT_TEST_MODULES = (
    "dft_local.transport.boltzmann.calculation.tests",
    "dft_local.transport.boltzmann.ashcroft_comparison.tests",
    "dft_local.transport.bands.tests",
)


def load_pytest_files(
    module_names: tuple[str, ...] = DEFAULT_TEST_MODULES,
) -> tuple[str, ...]:
    """Return pytest targets exposed by domain modules."""

    return collect_from_modules(
        module_names,
        "pytest_files",
        item_type=str,
    )
