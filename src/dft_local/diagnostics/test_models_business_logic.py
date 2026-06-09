# Copied from old diagnostics panel tests during package migration.
# These tests should target dft_local only.

import json
import pytest

from dft_local.diagnostics.models import (
    DiagnosticSetup,
    DiagnosticSpec,
    DiagnosticResult,
    Graph2D,
    GraphPoint,
    GraphSeries,
    InputParseError,
    InputSpec,
    Table,
    TableRow,
    parse_inputs,
    serialize_inputs,
)


def dummy_compute(ctx, inputs):
    return DiagnosticResult(title="dummy", summary="dummy")


def test_input_spec_parses_basic_types():
    specs = (
        InputSpec("n", "N", "int", 5, min_value=1, max_value=10),
        InputSpec("x", "X", "float", 0.5, min_value=0.0, max_value=1.0),
        InputSpec("flag", "Flag", "bool", False),
        InputSpec("name", "Name", "str", "a"),
        InputSpec("choice", "Choice", "select", "a", options=(("a", "A"), ("b", "B"))),
    )
    spec = DiagnosticSpec("dummy", "Debug", "Dummy", "", specs, dummy_compute)
    out = parse_inputs(spec, {"n": "7", "x": "0.25", "flag": "true", "name": "bob", "choice": "b"})
    assert out == {"n": 7, "x": 0.25, "flag": True, "name": "bob", "choice": "b"}


def test_input_spec_defaults_and_bounds():
    inp = InputSpec("n", "N", "int", 3, min_value=1, max_value=5)
    assert inp.parse(None) == 3
    with pytest.raises(InputParseError):
        inp.parse("0")
    with pytest.raises(InputParseError):
        inp.parse("6")


def test_select_rejects_unknown_option():
    inp = InputSpec("kernel", "Kernel", "select", "average_star", options=(("average_star", "Average star"),))
    with pytest.raises(InputParseError):
        inp.parse("anchored")


def test_serialize_inputs_is_stable_json_compatible():
    spec = DiagnosticSpec(
        "dummy",
        "Debug",
        "Dummy",
        "",
        (InputSpec("n", "N", "int", 5), InputSpec("x", "X", "float", 0.5), InputSpec("flag", "Flag", "bool", False)),
        dummy_compute,
    )
    parsed = parse_inputs(spec, {"n": "8", "x": "0.25", "flag": "on"})
    serial = serialize_inputs(spec, parsed)
    assert serial == {"n": 8, "x": 0.25, "flag": True}
    json.dumps(serial)


def test_diagnostic_setup_roundtrip():
    setup = DiagnosticSetup(
        version=1,
        diagnostic_id="region.matching",
        inputs={"kernel": "average_star", "nu": 51},
        view={"selected": {"entityId": "x"}},
    )
    text = setup.to_json()
    got = DiagnosticSetup.from_json(text)
    assert got == setup


def test_graph_payload_contains_entity_ids_and_channel():
    graph = Graph2D(
        id="g",
        title="G",
        description="desc",
        x_label="x",
        y_label="y",
        interaction_channel="samples",
        series=(
            GraphSeries(
                "s",
                (GraphPoint(1.0, 2.0, entity_id="sample:1", label="one", meta={"badness": 0.5}),),
                "points",
            ),
        ),
    )
    payload = graph.payload()
    assert payload["interaction_channel"] == "samples"
    assert payload["series"][0]["points"][0]["entity_id"] == "sample:1"
    assert payload["series"][0]["points"][0]["meta"]["badness"] == 0.5


def test_table_rows_carry_entity_ids():
    table = Table(
        id="t",
        title="T",
        description="desc",
        headers=("a",),
        rows=(TableRow((1,), entity_id="row:1"),),
        interaction_channel="rows",
    )
    assert table.rows[0].entity_id == "row:1"
    assert table.interaction_channel == "rows"


def test_iter_typst_math_finds_nested_user_strings():
    from dft_local.diagnostics.models import Card, DiagnosticResult, DiagnosticSection, Table, TableRow
    from dft_local.diagnostics.user_strings import TypstMath, iter_typst_math

    alpha = TypstMath("$ alpha $", name="alpha")
    beta = TypstMath("$ beta $", name="beta")
    gamma = TypstMath("$ gamma $", name="gamma")

    result = DiagnosticResult(
        title=alpha,
        summary="plain summary",
        cards=(Card(label=beta, value=1.0),),
        sections=(
            DiagnosticSection(
                id="s",
                title="section",
                description=gamma,
                tables=(
                    Table(
                        id="t",
                        title="table",
                        description="plain",
                        headers=("x",),
                        rows=(TableRow((1.0,)),),
                    ),
                ),
            ),
        ),
    )

    assert [item.name for item in iter_typst_math(result)] == ["alpha", "beta", "gamma"]


def test_all_registered_diagnostic_typst_math_compiles() -> None:
    """Every authored TypstMath snippet in registered specs must compile.

    This protects documentation/discussion strings from silently rotting.
    Diagnostic results can get a similar test once more diagnostics use
    TypstMath in computed output.
    """

    from dft_local.diagnostics.discovery import load_diagnostics
    from dft_local.diagnostics.typst import render_typst_math_to_svg
    from dft_local.diagnostics.user_strings import iter_typst_math

    specs = load_diagnostics()
    failures: list[str] = []

    for spec in specs:
        for snippet in iter_typst_math(spec):
            label = snippet.name or f"{spec.id}: {snippet.source!r}"
            try:
                render_typst_math_to_svg(snippet.source, display=snippet.display)
            except Exception as exc:  # noqa: BLE001 - collect all compile failures
                failures.append(f"{label}: {exc}")

    assert not failures, "\n".join(failures)


def test_bad_typst_math_is_rejected_by_compile_test() -> None:
    """A broken Typst snippet must fail the compile path used by tests."""

    import pytest

    from dft_local.diagnostics.typst import TypstRenderError, render_typst_math_to_svg

    with pytest.raises(TypstRenderError):
        render_typst_math_to_svg("$ unknown_symbol_without_spaces $")



def test_rich_user_string_helpers_collect_typst_math() -> None:
    from dft_local.diagnostics.user_strings import iter_typst_math, math, rich

    text = rich("Energy ", math("$ epsilon_n (k) $", name="energy_label"), " [eV]")

    snippets = list(iter_typst_math(text))

    assert len(snippets) == 1
    assert snippets[0].name == "energy_label"


def test_diagnostic_section_body_renders_ordered_document_blocks() -> None:
    from dft_local.diagnostics.models import DiagnosticSection, MarkdownBlock, Table, TableRow, TypstMathBlock
    from dft_local.diagnostics.render import render_diagnostic_section
    from dft_local.diagnostics.user_strings import TypstMath

    section = DiagnosticSection(
        id="ordered_body",
        title="Ordered body",
        body=(
            MarkdownBlock(
                id="intro",
                title="Intro",
                markdown="Before equation.",
            ),
            TypstMathBlock(
                id="equation",
                math=TypstMath("$ x = 1 $", display=True, name="ordered_body_equation"),
            ),
            MarkdownBlock(
                id="after",
                title="After",
                markdown="After equation.",
            ),
            Table(
                id="table",
                title="Table",
                description="",
                headers=("name", "value"),
                rows=(TableRow(("a", "b")),),
            ),
        ),
    )

    html = render_diagnostic_section(section)

    points = [
        html.find("Before equation."),
        html.find("id='equation'"),
        html.find("After equation."),
        html.find("id='table'"),
    ]

    assert all(point >= 0 for point in points)
    assert points == sorted(points)
    assert "typst-error" not in html


def test_diagnostic_section_body_takes_prebuilt_block_tuple() -> None:
    from dft_local.diagnostics.models import DiagnosticSection, ProseBlock, EquationBlock
    from dft_local.diagnostics.render import render_diagnostic_section
    from dft_local.diagnostics.user_strings import TypstMath

    body = (
        ProseBlock(id="p", title="Prose", markdown="A prose block."),
        EquationBlock(id="e", math=TypstMath("$ y = 2 $", display=True, name="body_tuple_equation")),
    )

    section = DiagnosticSection(
        id="prebuilt_body",
        title="Prebuilt body",
        body=body,
    )

    html = render_diagnostic_section(section)

    assert html.find("A prose block.") < html.find("id='e'")
    assert "body_tuple_equation" in html
    assert "typst-error" not in html


def test_document_helpers_construct_ordered_blocks() -> None:
    from dft_local.diagnostics.document import equation, prose
    from dft_local.diagnostics.models import DiagnosticSection
    from dft_local.diagnostics.render import render_diagnostic_section

    section = DiagnosticSection(
        id="helper_body",
        title="Helper body",
        body=(
            prose("intro", "Intro", "Before."),
            equation("eq", "$ z = 3 $"),
            prose("after", "After", "After."),
        ),
    )

    html = render_diagnostic_section(section)

    assert html.find("Before.") < html.find("id='eq'") < html.find("After.")
    assert "typst-error" not in html



def test_diagnostic_result_body_renders_ordered_document_blocks() -> None:
    from dft_local.diagnostics.document import equation, prose
    from dft_local.diagnostics.models import DiagnosticResult, Table, TableRow
    from dft_local.diagnostics.render import render_result

    result = DiagnosticResult(
        title="Ordered result",
        summary="Summary",
        body=(
            prose("intro", "Intro", "Before table."),
            Table(
                id="result_table",
                title="Result table",
                description="A table in result body.",
                headers=("name", "value"),
                rows=(TableRow(("a", "b")),),
            ),
            equation("result_equation", "$ q = 4 $"),
            prose("after", "After", "After equation."),
        ),
    )

    html = render_result(result)

    points = [
        html.find("Before table."),
        html.find("id='result_table'"),
        html.find("id='result_equation'"),
        html.find("After equation."),
    ]

    assert all(point >= 0 for point in points)
    assert points == sorted(points)
    assert "typst-error" not in html



def test_render_display_quantity_in_table_cell() -> None:
    from dft_local.core.units import DisplayQuantity, ENERGY, JOULE
    from dft_local.diagnostics.models import DiagnosticResult, Table, TableRow
    from dft_local.diagnostics.render import render_result

    result = DiagnosticResult(
        title="Quantity display",
        summary="Summary",
        body=(
            Table(
                id="quantity_table",
                title="Quantity table",
                description="Shows units.",
                headers=("name", "value"),
                rows=(
                    TableRow((
                        "energy",
                        DisplayQuantity(
                            value=1.25,
                            dimension=ENERGY,
                            unit=JOULE,
                            name="energy",
                        ),
                    )),
                ),
            ),
        ),
    )

    html = render_result(result)

    assert "1.25" in html
    assert "display-quantity" in html
    assert "data-unit='J'" in html
    assert "display-unit" in html



def test_webgl_view_renders_band_surface_component() -> None:
    from dft_local.diagnostics.models import DiagnosticResult, WebGLView
    from dft_local.diagnostics.render import render_result

    result = DiagnosticResult(
        title="Surface",
        summary="Band surface payload.",
        webgl=(
            WebGLView(
                id="surface",
                title="Band surface",
                description="Solved region.",
                renderer="region_surface",
                payload={
                    "nu": 2,
                    "nv": 2,
                    "nbands": 1,
                    "energies": [[[0.0], [1.0]], [[2.0], [3.0]]],
                },
            ),
        ),
    )

    html = render_result(result)

    assert "id='dft-model-surface'" in html
    assert "data-dft-model='surface'" in html
    assert "<dft-band-surface-viewer" in html
    assert "data-source='dft-model-surface'" in html
    assert "data-dft-model='dft-model-surface'" in html
    assert "\"nbands\": 1" in html
