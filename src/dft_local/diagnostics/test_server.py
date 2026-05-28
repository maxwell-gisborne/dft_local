from __future__ import annotations

from dft_local.diagnostics.server import DiagnosticApp, load_default_context


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


def test_band_path_page_renders_svg_graph() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.path",
        {
            "kernel": "average_star",
            "matching": "energy_predict",
            "path": "gamma_k_m_gamma",
            "points_per_segment": "8",
        },
    )

    assert "Band path Γ-K-M-Γ" in html
    assert "<svg" in html
    assert "K-space path" in html
    assert "k cartesian x" in html
    assert "k cartesian y" in html
    assert "primitive cell" in html
    assert "hexagon" in html
    assert "Γ K M" in html
    assert "band 0" in html
    assert "Graph payload" not in html


def test_select_inputs_render_as_select_controls() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.path",
        {
            "kernel": "average_star",
            "matching": "energy_predict",
            "path": "gamma_k_m_gamma",
            "points_per_segment": "8",
        },
    )

    assert "<select name='kernel'>" in html
    assert "<select name='matching'>" in html
    assert "<select name='path'>" in html
    assert "<input name='kernel'" not in html
    assert "Average star" in html
    assert "Energy prediction" in html
    assert "Circle around K" in html
    assert "Full Brillouin-zone hexagon" in html
