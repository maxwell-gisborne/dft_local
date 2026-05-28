from __future__ import annotations

from dft_local.diagnostics.server import (
    DiagnosticApp,
    load_default_context,
)


def test_dft_local_server_renders_real_boltzmann_conductivity_page() -> None:
    """dft_local server renders real data through its local context."""

    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.boltzmann.calculation.conductivity",
        {
            "kernel": "average_star",
            "temperature": "300",
            "mu": "0",
            "tau_fs": "10",
            "run": "1",
        },
    )

    assert "Boltzmann conductivity" in html
    assert "sigma_xx" in html or "conductivity" in html
    assert "Error" not in html
