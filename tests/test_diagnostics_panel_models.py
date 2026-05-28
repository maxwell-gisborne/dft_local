import json
import pytest

from dft_local.diagnostics_pannel.models import (
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
