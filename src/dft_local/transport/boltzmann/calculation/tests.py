"""Test metadata for the Boltzmann conductivity domain.

For now the actual tests still live in the repository-level `tests/`
directory.  This module makes that dependency explicit and discoverable by
dft_local test system.
"""

from __future__ import annotations


def pytest_files() -> list[str]:
    """Return pytest targets covering this domain.

    Do not include tests that call `run_discovered_pytest()` here.  That creates
    recursion: the discovered suite runs a test, which starts the discovered
    suite, which runs the same test again.
    """

    return [
        "src/dft_local/transport/boltzmann/calculation/test_conductivity_business_logic.py",
        "src/dft_local/core/test_discovery.py",
        "src/dft_local/core/test_geometry_and_kernels.py",
        "src/dft_local/core/test_local_problem_business_logic.py",
        "src/dft_local/core/test_kernel_business_logic.py",
        "src/dft_local/core/test_geometry_business_logic.py",
        "src/dft_local/core/test_groups_business_logic.py",
        "src/dft_local/core/test_dataset_business_logic.py",
        "src/dft_local/core/test_block_coupling_business_logic.py",
        "src/dft_local/diagnostics/test_discovery.py",
        "src/dft_local/diagnostics/test_models_business_logic.py",
        "src/dft_local/diagnostics/test_formatting_business_logic.py",
        "src/dft_local/diagnostics/test_asgi_server.py",
        "src/dft_local/diagnostics/test_server.py",
        "src/dft_local/diagnostics/test_conductivity_server_integration.py",
        "src/dft_local/testsuite/test_discovery.py",
        "src/dft_local/testsuite/test_runner.py",
        "src/dft_local/transport/boltzmann/calculation/test_core.py",
        "src/dft_local/transport/boltzmann/calculation/test_diagnostics.py",
    ]
