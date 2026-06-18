from __future__ import annotations

from dft_local.diagnostics.discovery import load_diagnostics
from dft_local.diagnostics.render import render_result
from dft_local.diagnostics.server import DiagnosticApp, load_default_context
from dft_local.transport.boltzmann.group_resolved.diagnostics import diagnostics


def test_group_resolved_diagnostics_expose_overview() -> None:
    specs = {spec.id: spec for spec in diagnostics()}

    assert "transport.boltzmann.group_resolved.overview" in specs


def test_group_resolved_diagnostic_is_discovered() -> None:
    specs = {spec.id: spec for spec in load_diagnostics()}

    assert "transport.boltzmann.group_resolved.overview" in specs


def test_group_resolved_overview_renders() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.boltzmann.group_resolved.overview",
        {
            "kernel": "average_star",
            "nu": "3",
            "nv": "3",
            "band": "0",
        },
    )

    assert "Symbol method internal consistency" in html
    assert "Symbol sanity checks" in html
    assert "H Hermiticity defect" in html
    assert "S minimum eigenvalue" in html
    assert "Selected energy-ordered band" in html
    assert "Band trace contributions" in html
    assert "sum bands - compact" in html
    assert "<dft-kpoint-readout>" in html
    assert "id='dft-model-group_resolved_band_surface'" in html
    assert "data-dft-model='group_resolved_band_surface'" in html
    assert "<dft-band-surface-viewer data-source='dft-model-group_resolved_band_surface'" in html


def test_group_resolved_result_contains_compact_invariant() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    spec = {spec.id: spec for spec in diagnostics()}["transport.boltzmann.group_resolved.overview"]

    result = spec.compute(ctx, {"kernel": "average_star", "nu": "3", "nv": "3"})
    html = render_result(result)

    assert "||sum bands - compact||" in html
    assert "sigma_00" in html



def test_group_resolved_surface_payload_contains_real_band_data() -> None:
    import json
    import re

    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.boltzmann.group_resolved.overview",
        {
            "kernel": "average_star",
            "nu": "3",
            "nv": "3",
            "band": "0",
        },
    )

    match = re.search(
        r"<script type='application/json' id='dft-model-group_resolved_band_surface' data-dft-model='group_resolved_band_surface'>(.*?)</script>",
        html,
        re.S,
    )
    assert match is not None

    payload = json.loads(match.group(1))

    assert payload["nu"] == 3
    assert payload["nv"] == 3
    assert len(payload["k1"]) == 3
    assert len(payload["k1"][0]) == 3
    assert len(payload["k2"]) == 3
    assert len(payload["k2"][0]) == 3
    assert len(payload["energies"]) == 3
    assert len(payload["energies"][0]) == 3
    assert len(payload["energies"][0][0]) == payload["nbands"]
    assert any(not value for row in payload["mask"] for value in row)


def test_group_resolved_datastar_stream_patches_surface_model() -> None:
    from dft_local.diagnostics.server import DiagnosticApp, load_default_context

    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    stream = app.diagnostic_run_stream(
        "transport.boltzmann.group_resolved.overview",
        {
            "kernel": "average_star",
            "nu": "3",
            "nv": "3",
            "band": "0",
        },
    )

    assert "data: selector #dft-model-group_resolved_band_surface" in stream
    assert '"nu": 3' in stream
    assert '"nv": 3' in stream
    assert "dft-block-group_resolved_band_surface" not in stream


def test_group_resolved_overview_contains_strong_spectral_reference() -> None:
    from dft_local.diagnostics.server import load_default_context
    from dft_local.diagnostics.render import render_result

    ctx = load_default_context("test_run/run_dir/data")
    spec = {spec.id: spec for spec in diagnostics()}["transport.boltzmann.group_resolved.overview"]
    html = render_result(spec.compute(ctx, {}))

    assert "strong_spectral_dc_reference" in html
    assert "Strong spectral DC reference" in html
    assert "strong_spectral_dc_reference_table" in html
    assert "strong trace / weak trace" in html
    assert "relative tensor discrepancy" in html


def test_group_resolved_overview_is_ashcroft_free() -> None:
    from dft_local.diagnostics.server import load_default_context
    from dft_local.diagnostics.render import render_result

    ctx = load_default_context("test_run/run_dir/data")
    spec = {spec.id: spec for spec in diagnostics()}["transport.boltzmann.group_resolved.overview"]
    html = render_result(spec.compute(ctx, {"kernel": "average_star", "nu": "3", "nv": "3"}))

    assert "Ashcroft" not in html
    assert "finite-difference" not in html
    assert "legacy scalar-band" not in html
    assert "A sigma" not in html
    assert "A trace" not in html
