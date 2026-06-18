from dft_local.diagnostics.render import render_result
from dft_local.diagnostics.server import load_default_context
from dft_local.transport.boltzmann.ashcroft_comparison.regions import diagnostics


def test_ashcroft_vs_symbol_regions_renders() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    spec = {spec.id: spec for spec in diagnostics()}["transport.boltzmann.ashcroft_vs_band_free.regions"]

    html = render_result(spec.compute(ctx, {"nu": "5", "nv": "5", "radius": "1"}))

    assert "Ashcroft versus symbol regions" in html
    assert "Regional summary" in html
    assert "minimum-gap region" in html
    assert "symbol trace" in html
    assert "Scalar-band comparison" in html
    assert "relative tensor delta" in html
    assert "max velocity delta" in html


def test_ashcroft_vs_symbol_regions_registered() -> None:
    from dft_local.transport.boltzmann.diagnostics import diagnostics as boltzmann_diagnostics

    specs = {spec.id: spec for spec in boltzmann_diagnostics()}

    assert "transport.boltzmann.ashcroft_vs_band_free.regions" in specs
