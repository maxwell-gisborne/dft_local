from __future__ import annotations

from dft_local.dataset.diagnostics import compute_geometry_overview, compute_overview, diagnostics
from dft_local.diagnostics.models import DiagnosticSpec


def test_dataset_diagnostics_exports_overview_spec() -> None:
    specs = {spec.id: spec for spec in diagnostics()}

    assert isinstance(specs["dataset.overview"], DiagnosticSpec)


def test_dataset_overview_handles_missing_context() -> None:
    result = compute_overview(None, {})

    assert result.title == "Dataset overview"
    assert "No diagnostic context" in result.summary
    assert result.cards[0].status == "warn"


def test_dataset_overview_renders_loaded_context() -> None:
    from dft_local.diagnostics.server import load_default_context

    result = compute_overview(load_default_context("test_run/run_dir/data"), {})

    assert "Loaded" in result.summary
    assert result.cards[0].label == "atoms"
    assert result.cards[1].label == "basis"
    assert any(section.id == "dataset_matrices" for section in result.sections)



def test_geometry_overview_handles_missing_context() -> None:
    result = compute_geometry_overview(None, {})

    assert result.title == "Geometry overview"
    assert "No diagnostic context" in result.summary
    assert result.cards[0].status == "warn"


def test_geometry_overview_renders_loaded_context() -> None:
    from dft_local.diagnostics.server import load_default_context

    result = compute_geometry_overview(load_default_context("test_run/run_dir/data"), {})

    assert "Nearest-neighbour graph" in result.summary
    assert result.cards[0].label == "atoms"
    assert result.cards[1].label == "anchor"
    assert any(section.id == "geometry_group_labels" for section in result.sections)


def test_dataset_diagnostics_exports_geometry_spec() -> None:
    specs = {spec.id: spec for spec in diagnostics()}

    assert isinstance(specs["geometry.overview"], DiagnosticSpec)
