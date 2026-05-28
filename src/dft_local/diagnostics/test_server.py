from __future__ import annotations

from dft_local.diagnostics.server import DiagnosticApp


def test_dft_local_server_loads_discovered_specs() -> None:
    app = DiagnosticApp()

    assert "dft_local.testsuite" in app.specs
    assert "transport.boltzmann.overview" in app.specs


def test_dft_local_server_index_contains_diagnostics() -> None:
    app = DiagnosticApp()
    html = app.index()

    assert "dft_local diagnostics" in html
    assert "dft_local.testsuite" in html
    assert "transport.boltzmann.overview" in html


def test_dft_local_server_can_render_static_boltzmann_overview() -> None:
    app = DiagnosticApp()
    html = app.diagnostic_page("transport.boltzmann.overview", {})

    assert "Boltzmann conductivity domain" in html
    assert "Domain files" in html
    assert "Documentation preview" in html


def test_dft_local_server_can_render_testsuite_without_running_tests() -> None:
    app = DiagnosticApp()
    html = app.diagnostic_page(
        "dft_local.testsuite",
        {
            "run_tests": "",
            "timeout": "30",
        },
    )

    assert "Test suite" in html
    assert "Discovered pytest targets" in html
    assert "src/dft_local/transport/boltzmann/test_conductivity_business_logic.py" in html
