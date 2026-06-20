from pathlib import Path
import json

from dft_local.diagnostics.models import (
    Card,
    DiagnosticResult,
    DiagnosticSection,
    Graph2D,
    GraphPoint,
    GraphSeries,
    Table,
    TableRow,
    WebGLView,
)
from dft_local.diagnostics.typst_bundle import default_typst_lib_source, export_typst_bundle


def test_typst_bundle_exports_manifest_components_and_data(tmp_path: Path) -> None:
    result = DiagnosticResult(
        title="Validation report",
        summary="A frozen diagnostic report.",
        cards=(Card(label="status", value="ok"),),
        sections=(
            DiagnosticSection(
                id="main",
                title="Main section",
                description="Section prose.",
                body=(
                    Graph2D(
                        id="field_strength",
                        title="Field strength",
                        description="Weak and strong curves.",
                        x_label="E [V/m]",
                        y_label="sigma [S/m]",
                        series=(
                            GraphSeries(
                                name="sigma_xx",
                                points=(
                                    GraphPoint(0.0, 1.0),
                                    GraphPoint(1.0, 2.0),
                                ),
                            ),
                        ),
                    ),
                    Table(
                        id="params",
                        title="Parameters",
                        description="Frozen parameters.",
                        headers=("name", "value"),
                        rows=(TableRow(("mu", 0.0)),),
                        numeric=frozenset({1}),
                    ),
                ),
            ),
        ),
    )

    out = export_typst_bundle(
        result,
        tmp_path / "bundle",
        lib_mode="none",
        provenance={"created_at": "now", "code_commit": "abc"},
    )

    assert (out / "diagnostics.json").exists()
    assert (out / "manifest.json").exists()
    assert (out / "diagnostics.typ").exists()
    assert (out / "components.typ").exists()
    assert not (out / "lib").is_symlink()
    assert (out / "lib" / "mod.typ").exists()
    assert default_typst_lib_source().is_absolute()
    assert default_typst_lib_source().exists()
    assert (out / "data" / "field_strength.json").exists()
    assert (out / "data" / "params.json").exists()

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["report_id"] == "Validation_report"
    assert manifest["provenance"]["code_commit"] == "abc"
    assert any(item["kind"] == "graph2d" and item["component"] == "field_strength" for item in manifest["items"])

    graph = json.loads((out / "data" / "field_strength.json").read_text())
    assert graph["series"][0]["points"][1] == {
        "entity_id": None,
        "label": "",
        "meta": {},
        "x": 1.0,
        "y": 2.0,
    }

    components = (out / "components.typ").read_text()
    assert '#let field_strength-data = json("data/field_strength.json")' in components
    assert "#let field_strength-plot() = line-graph(field_strength-data)" in components
    assert "#let field_strength() = diagnostic-figure(" in components
    assert "  field_strength-plot()," in components

    report = (out / "diagnostics.typ").read_text()
    assert "#main()" in report


def test_typst_bundle_preserves_webgl_as_static_placeholder(tmp_path: Path) -> None:
    result = DiagnosticResult(
        title="Viewer report",
        summary="Has a viewer.",
        webgl=(
            WebGLView(
                id="band_surface",
                title="Band viewer",
                description="Interactive 3D viewer.",
                renderer="region_surface",
                payload={"points": [1, 2, 3]},
            ),
        ),
    )

    out = export_typst_bundle(result, tmp_path / "bundle", lib_mode="none", provenance={})

    manifest = json.loads((out / "manifest.json").read_text())
    item = next(item for item in manifest["items"] if item["id"] == "band_surface")
    assert item["kind"] == "webgl-placeholder"
    assert item["static_support"] == "placeholder"

    data = json.loads((out / "data" / "band_surface.json").read_text())
    assert data["static_support"] == "placeholder"
    assert data["payload"] == {"points": [1, 2, 3]}

    components = (out / "components.typ").read_text()
    assert "unsupported-view(band_surface-data)" in components


def test_diagnostic_app_exports_typst_bundle_route(tmp_path: Path, monkeypatch) -> None:
    from dft_local.diagnostics.server import DiagnosticApp

    monkeypatch.chdir(tmp_path)

    app = DiagnosticApp(ctx=None)
    html = app.diagnostic_typst_export_page("transport.boltzmann.calculation.overview", {})

    out = tmp_path / "diagnostic_bundles" / "transport_boltzmann_calculation_overview"
    assert "Typst diagnostic bundle exported" in html
    assert (out / "diagnostics.typ").exists()
    assert (out / "components.typ").exists()
    assert (out / "lib").is_symlink()
    assert (out / "manifest.json").exists()
    assert (out / "diagnostics.json").exists()

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["report_id"] == "transport_boltzmann_calculation_overview"
    assert manifest["items"]


def test_diagnostic_page_links_to_typst_bundle_export() -> None:
    from dft_local.diagnostics.server import DiagnosticApp

    app = DiagnosticApp(ctx=None)
    html = app.diagnostic_page("transport.boltzmann.calculation.overview", {})

    assert "export Typst bundle" in html
    assert "/d-export/transport.boltzmann.calculation.overview" in html



def test_typst_bundle_smoke_compiles_with_typst_when_available(tmp_path: Path) -> None:
    import shutil
    import subprocess

    if shutil.which("typst") is None:
        return

    result = DiagnosticResult(
        title="Compile smoke",
        summary="Small compileable diagnostic.",
        sections=(
            DiagnosticSection(
                id="plots",
                title="Plots",
                body=(
                    Graph2D(
                        id="line_plot",
                        title="Line plot",
                        description="A small line plot.",
                        x_label="x",
                        y_label="y",
                        series=(
                            GraphSeries(
                                name="series",
                                points=(GraphPoint(0.0, 0.0), GraphPoint(1.0, 1.0)),
                            ),
                        ),
                    ),
                    Table(
                        id="small_table",
                        title="Small table",
                        description="A small table.",
                        headers=("name", "value [arb]"),
                        rows=(TableRow(("a", 1.0)),),
                    ),
                ),
            ),
        ),
    )

    out = export_typst_bundle(result, tmp_path / "bundle", lib_mode="none", provenance={"created_at": "now", "code_commit": "abc"})
    pdf = tmp_path / "bundle.pdf"

    proc = subprocess.run(
        ["typst", "compile", str(out / "diagnostics.typ"), str(pdf)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert pdf.exists()



def test_diagnostic_app_dispatches_typst_bundle_export_route(tmp_path: Path, monkeypatch) -> None:
    from urllib.parse import urlencode

    from dft_local.diagnostics.server import DiagnosticApp

    monkeypatch.chdir(tmp_path)
    app = DiagnosticApp(ctx=None)

    environ = {
        "PATH_INFO": "/d-export/transport.boltzmann.calculation.overview",
        "QUERY_STRING": urlencode({}),
    }
    seen = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        seen["status"] = status
        seen["headers"] = headers

    body = b"".join(app(environ, start_response)).decode("utf-8")

    assert seen["status"] == "200 OK"
    assert "Typst diagnostic bundle exported" in body
    assert (tmp_path / "diagnostic_bundles" / "transport_boltzmann_calculation_overview" / "diagnostics.typ").exists()
