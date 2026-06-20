from __future__ import annotations

from dft_local.dataset.diagnostics import compute_geometry_overview, compute_kernel_overview, compute_matrix_overview, compute_overview, compute_symbol_point, diagnostics
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



def test_matrix_overview_handles_missing_context() -> None:
    result = compute_matrix_overview(None, {})

    assert result.title == "Matrix overview"
    assert "No diagnostic context" in result.summary
    assert result.cards[0].status == "warn"


def test_matrix_overview_renders_loaded_context() -> None:
    from dft_local.diagnostics.server import load_default_context

    result = compute_matrix_overview(load_default_context("test_run/run_dir/data"), {})

    assert "H and S share" in result.summary
    assert result.cards[0].label == "H blocks"
    assert result.cards[1].label == "S blocks"
    assert any(section.id == "matrix_block_overlap" for section in result.sections)


def test_dataset_diagnostics_exports_matrix_spec() -> None:
    specs = {spec.id: spec for spec in diagnostics()}

    assert isinstance(specs["matrix.overview"], DiagnosticSpec)



def test_kernel_overview_handles_missing_context() -> None:
    result = compute_kernel_overview(None, {"kernel_choice": "average_star"})

    assert result.title == "Kernel overview"
    assert "No diagnostic context" in result.summary
    assert result.cards[0].status == "warn"


def test_kernel_overview_renders_loaded_context() -> None:
    from dft_local.diagnostics.server import load_default_context

    result = compute_kernel_overview(
        load_default_context("test_run/run_dir/data"),
        {"kernel_choice": "average_star"},
    )

    assert "Kernel choice" in result.summary or "kernel choice" in result.summary
    assert result.cards[0].label == "kernel choice"
    assert result.cards[1].label == "H support"
    assert any(section.id == "kernel_support_balance" for section in result.sections)


def test_dataset_diagnostics_exports_kernel_spec() -> None:
    specs = {spec.id: spec for spec in diagnostics()}

    assert isinstance(specs["kernel.overview"], DiagnosticSpec)



def test_symbol_point_handles_missing_context() -> None:
    result = compute_symbol_point(None, {"kernel_choice": "average_star", "k1": 0.0, "k2": 0.0})

    assert result.title == "Symbol point"
    assert "No diagnostic context" in result.summary
    assert result.cards[0].status == "warn"


def test_symbol_point_renders_loaded_context() -> None:
    from dft_local.diagnostics.server import load_default_context

    result = compute_symbol_point(
        load_default_context("test_run/run_dir/data"),
        {"kernel_choice": "average_star", "k1": 0.1, "k2": 0.2, "irrep_degree": 2},
    )

    assert "degree=2" in result.summary
    assert result.cards[0].label == "kernel choice"
    assert any(section.id == "symbol_overlap" for section in result.sections)
    assert any(section.id == "symbol_energies" for section in result.sections)


def test_symbol_point_renders_fixed_irrep_context() -> None:
    from dft_local.diagnostics.server import load_default_context

    result = compute_symbol_point(
        load_default_context("test_run/run_dir/data"),
        {"kernel_choice": "average_star", "k1": 0.0, "k2": 0.0, "irrep_degree": 1, "sigma": 1},
    )

    assert "degree=1" in result.summary
    assert any(section.id == "symbol_dense_diagnostics" for section in result.sections)


def test_dataset_diagnostics_exports_symbol_point_spec() -> None:
    specs = {spec.id: spec for spec in diagnostics()}

    assert isinstance(specs["symbol.point"], DiagnosticSpec)
