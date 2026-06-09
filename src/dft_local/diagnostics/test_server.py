from __future__ import annotations

from dft_local.diagnostics.server import DiagnosticApp, load_default_context


def test_dft_local_server_loads_discovered_specs() -> None:
    app = DiagnosticApp()

    assert "dft_local.testsuite" in app.specs
    assert "transport.boltzmann.calculation.overview" in app.specs


def test_dft_local_server_index_contains_diagnostics() -> None:
    app = DiagnosticApp()
    html = app.index()

    assert "dft_local diagnostics" in html
    assert "dft_local.testsuite" in html
    assert "transport.boltzmann.calculation.overview" in html


def test_dft_local_server_can_render_static_boltzmann_overview() -> None:
    app = DiagnosticApp()
    html = app.diagnostic_page("transport.boltzmann.calculation.overview", {})

    assert "Boltzmann conductivity domain" in html
    assert "Domain files" in html
    assert "Documentation preview" in html


def test_dft_local_server_can_render_testsuite_without_running_tests() -> None:
    app = DiagnosticApp()
    html = app.diagnostic_page(
        "dft_local.testsuite",
        {
            "run_tests": "",
            "timeout": "30",
        },
    )

    assert "Test suite" in html
    assert "Discovered pytest targets" in html
    assert "src/dft_local/transport/boltzmann/calculation/test_conductivity_business_logic.py" in html


def test_band_path_page_renders_svg_graph() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.path",
        {
            "kernel": "average_star",
            "matching": "energy_predict",
            "path": "gamma_k_m_gamma",
            "points_per_segment": "8",
        },
    )

    assert "Band path Γ-K-M-Γ" in html
    assert "<svg" in html
    assert "K-space path" in html
    assert "k1" in html
    assert "k2" in html
    assert "primitive cell" in html
    assert "hexagon" in html
    assert "Γ K M" in html
    assert "band 0" in html
    assert "Graph payload" not in html


def test_select_inputs_render_as_select_controls() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.path",
        {
            "kernel": "average_star",
            "matching": "energy_predict",
            "path": "gamma_k_m_gamma",
            "points_per_segment": "8",
        },
    )

    assert "<select name='kernel'>" in html
    assert "<select name='matching'>" in html
    assert "<select name='path'>" in html
    assert "<input name='kernel'" not in html
    assert "Average star" in html
    assert "Energy prediction" in html
    assert "Circle around K" in html
    assert "Full Brillouin-zone hexagon" in html


def test_band_path_page_mounts_graph_components() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.path",
        {
            "kernel": "average_star",
            "matching": "energy_predict",
            "path": "gamma_k_m_gamma",
            "points_per_segment": "8",
        },
    )

    assert "src='/static/dft-local-components.js'" in html
    assert "id='dft-model-kspace_path'" in html
    assert "data-dft-model='kspace_path'" in html
    assert "id='dft-model-band_path'" in html
    assert "data-dft-model='band_path'" in html
    assert "<dft-kspace-plot" in html
    assert "data-source='dft-model-kspace_path'" in html
    assert "data-dft-model='dft-model-kspace_path'" in html
    assert "<dft-line-graph" in html
    assert "data-source='dft-model-band_path'" in html
    assert "data-dft-model='dft-model-band_path'" in html
    assert "<svg" in html


def test_graph_json_payload_is_parseable_by_browser_component() -> None:
    import json
    import re

    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.path",
        {
            "kernel": "average_star",
            "matching": "energy_predict",
            "path": "gamma_k_m_gamma",
            "points_per_segment": "8",
        },
    )

    match = re.search(
        r"<script type='application/json' id='dft-model-band_path'[^>]*>(.*?)</script>",
        html,
        re.S,
    )

    assert match is not None
    payload = json.loads(match.group(1))
    assert payload["id"] == "band_path"
    assert payload["series"]
    assert "&quot;" not in match.group(1)


def test_band_path_tables_render_selection_controls() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.path",
        {
            "kernel": "average_star",
            "matching": "energy_predict",
            "path": "gamma_k_m_gamma",
            "points_per_segment": "8",
        },
    )

    assert "class='table-step-select'" in html
    assert "data-table-select='all'" in html
    assert "data-table-select='none'" in html
    assert "data-step='" in html
    assert "data-path-x='" in html


def test_index_reflects_diagnostic_domain_hierarchy() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.index()

    assert "<strong>transport</strong>" in html
    assert "<strong>boltzmann</strong>" in html
    assert "<strong>calculation</strong>" in html
    assert "<code>transport.boltzmann.calculation.overview</code>" in html
    assert "transport.boltzmann ·" not in html


def test_docs_index_reflects_source_domain_hierarchy() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.docs_page("")

    assert "<strong>transport</strong>" in html
    assert "<strong>boltzmann</strong>" in html
    assert "href='/docs/transport.boltzmann.calculation'" in html
    assert "<code>transport.boltzmann.calculation</code>" in html


def test_docs_page_renders_markdown_document() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.docs_page("transport.boltzmann.calculation")

    assert "<nav><a href='/'>diagnostics</a> · <a href='/docs/'>docs</a></nav>" in html
    assert "<code>transport.boltzmann.calculation</code>" in html
    assert "<h1>" in html or "<h2>" in html


def test_docs_markdown_renderer_uses_markdown_features() -> None:
    html = DiagnosticApp.render_markdown("# Title\n\n- one\n- two\n\n`inline`")

    assert "<h1" in html
    assert "Title" in html
    assert "<li>" in html
    assert "one" in html
    assert "inline" in html
    assert "- two" not in html
    assert "\\n\\n" not in html


def test_diagnostic_pages_link_to_matching_docs() -> None:
    app = DiagnosticApp(ctx=None)

    html = app.diagnostic_page("transport.boltzmann.ashcroft_comparison.overview", {})

    assert "/docs/transport.boltzmann.ashcroft_comparison" in html


def test_docs_pages_link_to_matching_diagnostics() -> None:
    app = DiagnosticApp(ctx=None)

    html = app.docs_page("transport.boltzmann.ashcroft_comparison")

    assert "/d/transport.boltzmann.ashcroft_comparison.overview" in html


def test_kspace_hexagon_payload_renders_regular_on_screen() -> None:
    import json
    import math
    import re

    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.path",
        {
            "kernel": "average_star",
            "matching": "energy_predict",
            "path": "gamma_k_m_gamma",
            "points_per_segment": "8",
        },
    )

    match = re.search(
        r"<script type='application/json' id='([^']+)'[^>]*>\s*(\{.*?\})\s*</script>",
        html,
        flags=re.S,
    )
    assert match is not None

    payloads = []
    for _source_id, raw_json in re.findall(
        r"<script type='application/json' id='([^']+)'[^>]*>\s*(\{.*?\})\s*</script>",
        html,
        flags=re.S,
    ):
        payload = json.loads(raw_json)
        if any("hex" in series["name"].lower() for series in payload.get("series", [])):
            payloads.append(payload)

    assert payloads, "expected a k-space payload containing a hexagon series"

    payload = payloads[0]
    hex_series = next(
        series for series in payload["series"]
        if "hex" in series["name"].lower()
    )

    raw_points = hex_series["points"]

    # Drop closing point if repeated.
    if raw_points[0]["x"] == raw_points[-1]["x"] and raw_points[0]["y"] == raw_points[-1]["y"]:
        raw_points = raw_points[:-1]

    assert len(raw_points) == 6

    def k_basis_to_cartesian(k1: float, k2: float) -> tuple[float, float]:
        return (k1 - 0.5 * k2, math.sqrt(3.0) * 0.5 * k2)
    def raw_cartesian(k1: float, k2: float) -> tuple[float, float]:
        return (k1, k2)

    def polygon_lengths(points, transform):
        coords = [transform(float(point["x"]), float(point["y"])) for point in points]
        return [
            math.hypot(
                coords[(i + 1) % len(coords)][0] - coords[i][0],
                coords[(i + 1) % len(coords)][1] - coords[i][1],
            )
            for i in range(len(coords))
        ]

    raw_lengths = polygon_lengths(raw_points, raw_cartesian)
    basis_converted_lengths = polygon_lengths(raw_points, k_basis_to_cartesian)

    # Match JS graph layout for k-space plots.
    width = 1000
    height = 520
    margin = {"left": 78, "right": 150, "top": 28, "bottom": 62}
    inner_w = width - margin["left"] - margin["right"]
    inner_h = height - margin["top"] - margin["bottom"]

    all_cartesian = []
    for series in payload["series"]:
        for point in series["points"]:
            all_cartesian.append(k_basis_to_cartesian(float(point["x"]), float(point["y"])))

    xmin = min(x for x, _y in all_cartesian)
    xmax = max(x for x, _y in all_cartesian)
    ymin = min(y for _x, y in all_cartesian)
    ymax = max(y for _x, y in all_cartesian)

    # Match JS graphBounds padding.
    xpad = 0.06 * (xmax - xmin)
    ypad = 0.10 * (ymax - ymin)
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    # Match JS equalPixelAspectView.
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    xspan = xmax - xmin
    yspan = ymax - ymin
    pixel_aspect = inner_w / inner_h
    data_aspect = xspan / yspan

    if data_aspect < pixel_aspect:
        xspan = yspan * pixel_aspect
        xmin = xmid - 0.5 * xspan
        xmax = xmid + 0.5 * xspan
    else:
        yspan = xspan / pixel_aspect
        ymin = ymid - 0.5 * yspan
        ymax = ymid + 0.5 * yspan

    def project(point: dict[str, float]) -> tuple[float, float]:
        x, y = k_basis_to_cartesian(float(point["x"]), float(point["y"]))
        sx = margin["left"] + ((x - xmin) / (xmax - xmin)) * inner_w
        sy = margin["top"] + ((ymax - y) / (ymax - ymin)) * inner_h
        return sx, sy

    screen_points = [project(point) for point in raw_points]
    lengths = []
    for i, point in enumerate(screen_points):
        next_point = screen_points[(i + 1) % len(screen_points)]
        lengths.append(math.hypot(point[0] - next_point[0], point[1] - next_point[1]))

    min_length = min(lengths)
    max_length = max(lengths)

    assert min_length > 0.0
    assert max_length / min_length < 1.000000001, lengths


def test_kspace_hexagon_path_points_lie_on_hexagon_perimeter() -> None:
    import json
    import math
    import re

    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.path",
        {
            "kernel": "average_star",
            "matching": "energy_predict",
            "path": "hexagon",
            "points_per_segment": "8",
        },
    )

    payload = None
    for _source_id, raw_json in re.findall(
        r"<script type='application/json' id='([^']+)'[^>]*>\s*(\{.*?\})\s*</script>",
        html,
        flags=re.S,
    ):
        candidate = json.loads(raw_json)
        names = {series["name"] for series in candidate.get("series", [])}
        if "hexagon" in names and "selected path" in names:
            payload = candidate
            break

    assert payload is not None, "expected k-space payload with hexagon and selected path"

    series_by_name = {series["name"]: series for series in payload["series"]}
    hex_points = series_by_name["hexagon"]["points"]
    selected_points = series_by_name["selected path"]["points"]

    if hex_points[0]["x"] == hex_points[-1]["x"] and hex_points[0]["y"] == hex_points[-1]["y"]:
        hex_points = hex_points[:-1]

    assert len(hex_points) == 6
    assert selected_points

    def k_basis_to_cartesian(point: dict[str, float]) -> tuple[float, float]:
        k1 = float(point["x"])
        k2 = float(point["y"])
        return (k1 - 0.5 * k2, math.sqrt(3.0) * 0.5 * k2)

    hex_cart = [k_basis_to_cartesian(point) for point in hex_points]
    selected_cart = [k_basis_to_cartesian(point) for point in selected_points]

    def distance_to_segment(
        point: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        px, py = point
        ax, ay = a
        bx, by = b

        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay

        vv = vx * vx + vy * vy
        if vv == 0.0:
            return math.hypot(px - ax, py - ay)

        t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
        qx = ax + t * vx
        qy = ay + t * vy
        return math.hypot(px - qx, py - qy)

    edge_lengths = [
        math.hypot(
            hex_cart[(i + 1) % len(hex_cart)][0] - hex_cart[i][0],
            hex_cart[(i + 1) % len(hex_cart)][1] - hex_cart[i][1],
        )
        for i in range(len(hex_cart))
    ]

    assert max(edge_lengths) / min(edge_lengths) < 1.000000001

    scale = max(edge_lengths)
    max_distance = max(
        min(
            distance_to_segment(point, hex_cart[i], hex_cart[(i + 1) % len(hex_cart)])
            for i in range(len(hex_cart))
        )
        for point in selected_cart
    )

    assert max_distance / scale < 1.0e-10


def obsolete_test_kspace_reference_m_lies_on_hexagon_boundary() -> None:
    import json
    import math
    import re

    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.path",
        {
            "kernel": "average_star",
            "matching": "energy_predict",
            "path": "gamma_k_m_gamma",
            "points_per_segment": "8",
        },
    )

    payload = None
    for _source_id, raw_json in re.findall(
        r"<script type='application/json' id='([^']+)'[^>]*>\s*(\{.*?\})\s*</script>",
        html,
        flags=re.S,
    ):
        candidate = json.loads(raw_json)
        names = {series["name"] for series in candidate.get("series", [])}
        if "hexagon" in names and "Γ K M" in names:
            payload = candidate
            break

    assert payload is not None

    series_by_name = {series["name"]: series for series in payload["series"]}
    hex_points = series_by_name["hexagon"]["points"]
    ref_points = {point["label"]: point for point in series_by_name["Γ K M"]["points"]}

    if hex_points[0]["x"] == hex_points[-1]["x"] and hex_points[0]["y"] == hex_points[-1]["y"]:
        hex_points = hex_points[:-1]

    def k_basis_to_cartesian(point: dict[str, float]) -> tuple[float, float]:
        k1 = float(point["x"])
        k2 = float(point["y"])
        return (k1 - 0.5 * k2, math.sqrt(3.0) * 0.5 * k2)

    def distance_to_segment(
        point: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        px, py = point
        ax, ay = a
        bx, by = b
        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay
        vv = vx * vx + vy * vy
        if vv == 0.0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
        qx = ax + t * vx
        qy = ay + t * vy
        return math.hypot(px - qx, py - qy)

    hex_cart = [k_basis_to_cartesian(point) for point in hex_points]
    m_cart = k_basis_to_cartesian(ref_points["M"])
    k_cart = k_basis_to_cartesian(ref_points["K"])

    edge_lengths = [
        math.hypot(
            hex_cart[(i + 1) % len(hex_cart)][0] - hex_cart[i][0],
            hex_cart[(i + 1) % len(hex_cart)][1] - hex_cart[i][1],
        )
        for i in range(len(hex_cart))
    ]
    scale = max(edge_lengths)

    m_distance = min(
        distance_to_segment(m_cart, hex_cart[i], hex_cart[(i + 1) % len(hex_cart)])
        for i in range(len(hex_cart))
    )
    k_distance = min(math.hypot(k_cart[0] - x, k_cart[1] - y) for x, y in hex_cart)

    assert m_distance / scale < 1.0e-12
    assert k_distance / scale < 1.0e-12


def test_kspace_reference_markers_lie_on_selected_path() -> None:
    import json
    import math
    import re

    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.path",
        {
            "kernel": "average_star",
            "matching": "energy_predict",
            "path": "gamma_k_m_gamma",
            "points_per_segment": "8",
        },
    )

    payload = None
    for _source_id, raw_json in re.findall(
        r"<script type='application/json' id='([^']+)'[^>]*>\s*(\{.*?\})\s*</script>",
        html,
        flags=re.S,
    ):
        candidate = json.loads(raw_json)
        names = {series["name"] for series in candidate.get("series", [])}
        if "selected path" in names and "Γ K M" in names:
            payload = candidate
            break

    assert payload is not None

    series_by_name = {series["name"]: series for series in payload["series"]}
    selected = series_by_name["selected path"]["points"]
    references = series_by_name["Γ K M"]["points"]

    assert {point["label"] for point in references} == {"Γ", "K", "M"}

    def distance(a: dict[str, float], b: dict[str, float]) -> float:
        return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))

    for ref in references:
        nearest = min(distance(ref, point) for point in selected)
        assert nearest < 1.0e-12, ref


def test_gamma_k_m_rendered_path_m_lies_on_hexagon_edge() -> None:
    import json
    import math
    import re

    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.path",
        {
            "kernel": "average_star",
            "matching": "energy_predict",
            "path": "gamma_k_m_gamma",
            "points_per_segment": "8",
        },
    )

    payload = None
    for _source_id, raw_json in re.findall(
        r"<script type='application/json' id='([^']+)'[^>]*>\s*(\{.*?\})\s*</script>",
        html,
        flags=re.S,
    ):
        candidate = json.loads(raw_json)
        names = {series["name"] for series in candidate.get("series", [])}
        if {"selected path", "hexagon", "Γ K M"} <= names:
            payload = candidate
            break

    assert payload is not None, "expected rendered k-space payload"

    series_by_name = {series["name"]: series for series in payload["series"]}
    selected = series_by_name["selected path"]["points"]
    hex_points = series_by_name["hexagon"]["points"]
    references = {point["label"]: point for point in series_by_name["Γ K M"]["points"]}

    assert {"Γ", "K", "M"} <= set(references)

    if hex_points[0]["x"] == hex_points[-1]["x"] and hex_points[0]["y"] == hex_points[-1]["y"]:
        hex_points = hex_points[:-1]

    def to_cart(point: dict[str, float]) -> tuple[float, float]:
        k1 = float(point["x"])
        k2 = float(point["y"])
        return (k1 - 0.5 * k2, math.sqrt(3.0) * 0.5 * k2)

    def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def distance_to_segment(
        point: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        px, py = point
        ax, ay = a
        bx, by = b
        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay
        vv = vx * vx + vy * vy
        if vv == 0.0:
            return distance(point, a)
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
        qx = ax + t * vx
        qy = ay + t * vy
        return distance(point, (qx, qy))

    hex_cart = [to_cart(point) for point in hex_points]
    selected_cart = [to_cart(point) for point in selected]
    k_cart = to_cart(references["K"])
    m_cart = to_cart(references["M"])

    edge_lengths = [
        distance(hex_cart[i], hex_cart[(i + 1) % len(hex_cart)])
        for i in range(len(hex_cart))
    ]
    scale = max(edge_lengths)

    # First prove the reference marker really is a selected path sample.
    nearest_selected_to_m = min(distance(m_cart, point) for point in selected_cart)
    assert nearest_selected_to_m / scale < 1.0e-12

    # Then prove K and M are on the BZ boundary.
    k_boundary_distance = min(distance(k_cart, point) for point in hex_cart)
    m_boundary_distance = min(
        distance_to_segment(m_cart, hex_cart[i], hex_cart[(i + 1) % len(hex_cart)])
        for i in range(len(hex_cart))
    )

    assert k_boundary_distance / scale < 1.0e-12
    assert m_boundary_distance / scale < 1.0e-12, {
        "M": references["M"],
        "M_cart": m_cart,
        "boundary_distance": m_boundary_distance,
        "edge_scale": scale,
        "relative_distance": m_boundary_distance / scale,
    }

    # Finally prove the selected K->M segment itself lies on one hexagon edge.
    k_index = min(range(len(selected_cart)), key=lambda i: distance(selected_cart[i], k_cart))
    m_index = min(range(len(selected_cart)), key=lambda i: distance(selected_cart[i], m_cart))
    lo, hi = sorted((k_index, m_index))

    max_segment_distance = max(
        min(
            distance_to_segment(point, hex_cart[i], hex_cart[(i + 1) % len(hex_cart)])
            for i in range(len(hex_cart))
        )
        for point in selected_cart[lo : hi + 1]
    )

    assert max_segment_distance / scale < 1.0e-12, {
        "k_index": k_index,
        "m_index": m_index,
        "max_segment_distance": max_segment_distance,
        "edge_scale": scale,
        "relative_distance": max_segment_distance / scale,
    }


def test_tables_render_in_breakout_scroll_container() -> None:
    from dft_local.diagnostics.models import DiagnosticResult, Table, TableRow
    from dft_local.diagnostics.render import render_result

    result = DiagnosticResult(
        title="Wide table",
        summary="Table overflow check",
        tables=(
            Table(
                id="wide_table",
                title="Wide table",
                description="A deliberately wide diagnostic table.",
                headers=("first", "second", "third", "fourth"),
                rows=(TableRow(("a", "b", "c", "d")),),
            ),
        ),
    )

    html = render_result(result)

    assert "class='diagnostic-table-section'" in html
    assert "class='table-breakout'" in html
    assert "tabindex='0'" in html
    assert "<table" in html
    assert "data-dft-table='wide_table'" in html


def test_table_breakout_has_paper_background_without_table_row_override() -> None:
    from pathlib import Path

    css_source = Path("src/dft_local/diagnostics/render.py").read_text()

    assert ".diagnostic-paper .table-breakout {" in css_source
    breakout_rule = css_source.split(".diagnostic-paper .table-breakout {", 1)[1].split("}", 1)[0]
    assert "background: var(--paper);" in breakout_rule

    table_rule = css_source.split(".diagnostic-paper .table-breakout table {", 1)[1].split("}", 1)[0]
    assert "background: transparent;" in table_rule

    # Alternating table rows should remain separate from the breakout background.
    assert "tbody tr:nth-child(even)" in css_source


def test_diagnostic_title_renders_before_sections() -> None:
    from dft_local.diagnostics.models import DiagnosticResult, DiagnosticSection
    from dft_local.diagnostics.render import render_result

    html = render_result(
        DiagnosticResult(
            title="Ordered title",
            summary="Summary first.",
            sections=(
                DiagnosticSection(
                    id="example_section",
                    title="Example section",
                    description="Section body.",
                ),
            ),
        )
    )

    assert html.index("<h1>Ordered title</h1>") < html.index("id='example_section'")


def test_typst_math_renders_to_inline_svg() -> None:
    from dft_local.diagnostics.models import DiagnosticResult
    from dft_local.diagnostics.render import render_result
    from dft_local.diagnostics.user_strings import TypstMath

    html = render_result(
        DiagnosticResult(
            title=TypstMath("$ H(k) u = E dot S(k) u $", name="generalized_eigenproblem"),
            summary="plain summary",
        )
    )

    assert "class='typst-math inline'" in html
    assert "<svg" in html
    assert "plain summary" in html


def test_plain_user_strings_are_still_escaped() -> None:
    from dft_local.diagnostics.models import DiagnosticResult
    from dft_local.diagnostics.render import render_result

    html = render_result(
        DiagnosticResult(
            title="<script>alert(1)</script>",
            summary="<b>not bold</b>",
        )
    )

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;b&gt;not bold&lt;/b&gt;" in html



def test_rich_user_string_renders_plain_text_as_text_and_math_as_svg() -> None:
    from dft_local.diagnostics.render import render_user_string
    from dft_local.diagnostics.user_strings import TypstMath, rich

    html = render_user_string(
        rich(
            "Conductivity ",
            TypstMath("$ sigma_(alpha beta) $", name="sigma_label"),
            " [S/m]",
        )
    )

    assert html.startswith("Conductivity ")
    assert "data-typst-name='sigma_label'" in html
    assert "<svg" in html
    assert html.endswith(" [S/m]")
    assert "typst-error" not in html



def test_diagnostics_context_exposes_unit_provenance_rows() -> None:
    from dft_local.diagnostics.server import load_default_context

    ctx = load_default_context("test_run/run_dir/data")
    rows = ctx.state.unit_provenance_rows()
    row_map = {row[0]: row[1] for row in rows}

    assert row_map["disk energy unit"] == "hartree"
    assert row_map["working energy unit"] == "eV"
    assert row_map["disk length unit"] == "bohr"
    assert row_map["working length unit"] == "angstrom"
    assert row_map["energy disk-to-working factor"] == ctx.state.data.energy_conversion_disk_to_working
    assert row_map["length disk-to-working factor"] == ctx.state.data.length_conversion_disk_to_working



def test_band_path_page_renders_with_energy_order_matching() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.path",
        {
            "kernel": "average_star",
            "matching": "energy_order",
            "path": "gamma_k_m_gamma",
            "points_per_segment": "8",
        },
    )

    assert "Band path Γ-K-M-Γ" in html
    assert "Energy ordering" in html or "energy_order" in html
    assert "<svg" in html



def test_diagnostic_page_includes_three_import_map() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.region_surface",
        {
            "kernel": "average_star",
            "matching": "energy_order",
            "nu": "3",
            "nv": "3",
        },
    )

    assert 'type="importmap"' in html
    assert '"three": "https://unpkg.com/three@0.160.0/build/three.module.js"' in html



def test_diagnostic_blocks_have_stable_block_shells() -> None:
    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    html = app.diagnostic_page(
        "transport.bands.synthetic_surface",
        {
            "surface": "gaussian",
            "nu": "5",
            "nv": "5",
        },
    )

    assert "data-dft-block=" in html
    assert "data-dft-block-kind=" in html
    assert "data-dft-block-kind='json-rendered'" in html
    assert "data-dft-model=" in html
    assert "id='dft-model-" in html


def test_table_blocks_are_stateful_html_blocks() -> None:
    app = DiagnosticApp()
    html = app.diagnostic_page("transport.boltzmann.calculation.overview", {})

    assert "data-dft-block-kind='stateful-html'" in html



def test_diagnostic_page_includes_datastar_dependency_and_result_outlet() -> None:
    app = DiagnosticApp()
    html = app.diagnostic_page("transport.boltzmann.calculation.overview", {})

    assert "datastar" in html.lower()
    assert "id='diagnostic-result'" in html
    assert "data-dft-diagnostic-result" in html
    assert "data-on-submit" in html
    assert "/d-run/transport.boltzmann.calculation.overview" in html


def test_diagnostic_run_stream_returns_datastar_sse_patch() -> None:
    app = DiagnosticApp()
    stream = app.diagnostic_run_stream("transport.boltzmann.calculation.overview", {})

    assert "event: datastar-patch-elements" in stream
    assert "data: selector #diagnostic-result" in stream
    assert "id='diagnostic-result'" in stream
    assert "event: datastar-execute-script" in stream
    assert "dftRefreshModels" in stream
