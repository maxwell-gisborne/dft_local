from __future__ import annotations
import numpy as np

from dft_local.diagnostics.models import DiagnosticSpec
from dft_local.transport.bands.diagnostics import (
    compute_overview,
    compute_synthetic_surface,
    diagnostics,
)


def test_bands_diagnostics_exports_overview_spec() -> None:
    specs = diagnostics()
    by_id = {spec.id: spec for spec in specs}

    assert len(specs) >= 4
    assert isinstance(by_id["transport.bands.overview"], DiagnosticSpec)
    assert isinstance(by_id["transport.bands.path"], DiagnosticSpec)
    assert "transport.bands.synthetic_surface" in by_id


def test_bands_overview_mentions_public_api() -> None:
    result = compute_overview(None, {})

    assert result.title == "Band/path continuation domain"

    from dft_local.diagnostics.models import Table

    table_ids = {table.id for table in result.tables} | {
        block.id for block in result.body if isinstance(block, Table)
    }
    assert "bands_files" in table_ids
    assert "bands_api" in table_ids

    api_table = next(
        table
        for table in (*result.tables, *result.body)
        if isinstance(table, Table) and table.id == "bands_api"
    )
    names = {row.cells[0] for row in api_table.rows}

    assert "LocalPath" in names
    assert "match_via_energies" in names
    assert "match_via_overlap" in names


def test_bands_diagnostics_exports_path_spec() -> None:
    specs = diagnostics()
    by_id = {spec.id: spec for spec in specs}

    spec = by_id["transport.bands.path"]

    assert spec.title == "Band path Γ-K-M-Γ"
    assert spec.group == "transport"
    assert spec.inputs


def test_bands_path_spec_has_inputs() -> None:
    specs = diagnostics()
    by_id = {spec.id: spec for spec in specs}

    spec = by_id["transport.bands.path"]

    assert spec.title == "Band path Γ-K-M-Γ"
    assert spec.group == "transport"
    assert {inp.name for inp in spec.inputs} == {
        "kernel",
        "matching",
        "path",
        "points_per_segment",
    }



def test_band_path_matching_options_include_energy_order() -> None:
    from dft_local.transport.bands.diagnostics import MATCHING_OPTIONS

    assert ("energy_order", "Energy ordering") in MATCHING_OPTIONS



def test_band_region_surface_diagnostic_renders_webgl_payload() -> None:
    from dft_local.diagnostics.discovery import load_diagnostics
    from dft_local.diagnostics.render import render_result
    from dft_local.diagnostics.server import load_default_context

    ctx = load_default_context("test_run/run_dir/data")
    specs = {spec.id: spec for spec in load_diagnostics()}

    result = specs["transport.bands.region_surface"].compute(
        ctx,
        {
            "kernel": "average_star",
            "matching": "energy_order",
            "nu": "3",
            "nv": "3",
        },
    )
    html = render_result(result)

    assert "Band region surface" in html
    assert "id='data-band_region_surface'" in html
    assert "<dft-band-surface-viewer data-source='data-band_region_surface'></dft-band-surface-viewer>" in html
    assert '"energies"' in html
    assert '"nbands"' in html or '"bands"' in html



def test_synthetic_surface_gaussian_payload_is_rotational() -> None:
    result = compute_synthetic_surface(
        None,
        {
            "surface": "gaussian",
            "nu": "5",
            "nv": "5",
        },
    )

    view = result.body[0]
    payload = view.payload

    assert payload["nu"] == 5
    assert payload["nv"] == 5
    assert payload["nbands"] == 1
    assert len(payload["energies"]) == 5
    assert len(payload["energies"][0]) == 5
    assert len(payload["energies"][0][0]) == 1

    centre = payload["energies"][2][2][0]
    corner = payload["energies"][0][0][0]

    assert centre > corner
    assert any(not value for row in payload["mask"] for value in row)


def test_synthetic_surface_plane_x_and_plane_y_differ() -> None:
    plane_x = compute_synthetic_surface(
        None,
        {
            "surface": "plane_x",
            "nu": "3",
            "nv": "3",
        },
    ).body[0].payload

    plane_y = compute_synthetic_surface(
        None,
        {
            "surface": "plane_y",
            "nu": "3",
            "nv": "3",
        },
    ).body[0].payload

    assert plane_x["energies"] != plane_y["energies"]



def test_synthetic_plane_waves_are_oscillatory() -> None:
    payload = compute_synthetic_surface(
        None,
        {
            "surface": "plane_x",
            "nu": "21",
            "nv": "21",
        },
    ).body[0].payload

    values = np.asarray(payload["energies"], dtype=float)[:, :, 0]

    assert np.max(values) <= 1.0 + 1e-12
    assert np.min(values) >= -1.0 - 1e-12
    assert np.any(values > 0.5)
    assert np.any(values < -0.5)
