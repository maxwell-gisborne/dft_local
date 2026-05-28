from __future__ import annotations

from dft_local.diagnostics.discovery import load_diagnostics


def test_ashcroft_comparison_diagnostic_is_discovered() -> None:
    specs = {spec.id: spec for spec in load_diagnostics()}

    assert "transport.boltzmann.ashcroft_comparison.overview" in specs


def test_ashcroft_comparison_overview_renders() -> None:
    specs = {spec.id: spec for spec in load_diagnostics()}
    spec = specs["transport.boltzmann.ashcroft_comparison.overview"]

    result = spec.compute(None, {})

    assert result.title == "Ashcroft comparison"
    assert result.cards
