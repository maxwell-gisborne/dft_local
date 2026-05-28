"""Test metadata for the band/path continuation domain."""

from __future__ import annotations


def pytest_files() -> list[str]:
    """Return pytest targets covering this domain."""

    return [
        "src/dft_local/transport/bands/test_continuation_business_logic.py",
        "src/dft_local/transport/bands/test_energy_surface_rectification_business_logic.py",
        "src/dft_local/transport/bands/test_diagnostics.py",
    ]
