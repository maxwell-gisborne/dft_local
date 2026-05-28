from __future__ import annotations

from dft_local.diagnostics.server import DiagnosticApp, load_default_context


def test_dft_local_server_loads_discovered_specs() -> None:
    app = DiagnosticApp()

    assert "dft_local.testsuite" in app.specs
    assert "transport.boltzmann.calculation.overview" in app.specs


def test_dft_local_server_index_contains_diagnostics() -> None:
    app = DiagnosticApp()
    html = app.index()

    assert "dft_local diagnostics" in html
    assert "dft_local.testsuite" in html
    assert "transport.boltzmann.calculation.overview" in html


def test_dft_local_server_can_render_static_boltzmann_overview() -> None:
    app = DiagnosticApp()
    html = app.diagnostic_page("transport.boltzmann.calculation.overview", {})

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
    assert "src/dft_local/transport/boltzmann/calculation/test_conductivity_business_logic.py" in html


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
    assert "k1" in html
    assert "k2" in html
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


def test_band_path_page_mounts_graph_components() -> None:
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

    assert "src='/static/dft-local-components.js'" in html
    assert "<script type='application/json' id='data-kspace_path'>" in html
    assert "<script type='application/json' id='data-band_path'>" in html
    assert "<dft-kspace-plot data-source='data-kspace_path'>" in html
    assert "<dft-line-graph data-source='data-band_path'>" in html
    assert "<svg" in html


def test_graph_json_payload_is_parseable_by_browser_component() -> None:
    import json
    import re

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

    match = re.search(
        r"<script type='application/json' id='data-band_path'>(.*?)</script>",
        html,
        re.S,
    )

    assert match is not None
    payload = json.loads(match.group(1))
    assert payload["id"] == "band_path"
    assert payload["series"]
    assert "&quot;" not in match.group(1)


def test_band_path_tables_render_selection_controls() -> None:
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

    assert "class='table-step-select'" in html
    assert "data-table-select='all'" in html
    assert "data-table-select='none'" in html
    assert "data-step='" in html
    assert "data-path-x='" in html


def test_index_reflects_diagnostic_domain_hierarchy() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.index()

    assert "<strong>transport</strong>" in html
    assert "<strong>boltzmann</strong>" in html
    assert "<strong>calculation</strong>" in html
    assert "<code>transport.boltzmann.calculation.overview</code>" in html
    assert "transport.boltzmann ·" not in html


def test_docs_index_reflects_source_domain_hierarchy() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.docs_page("")

    assert "<strong>transport</strong>" in html
    assert "<strong>boltzmann</strong>" in html
    assert "href='/docs/transport.boltzmann.calculation'" in html
    assert "<code>transport.boltzmann.calculation</code>" in html


def test_docs_page_renders_markdown_document() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.docs_page("transport.boltzmann.calculation")

    assert "<nav><a href='/'>diagnostics</a> · <a href='/docs/'>docs</a></nav>" in html
    assert "<code>transport.boltzmann.calculation</code>" in html
    assert "<h1>" in html or "<h2>" in html
