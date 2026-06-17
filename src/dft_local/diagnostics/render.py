"""Minimal HTML renderer for diagnostics."""

from __future__ import annotations

from html import escape
import re
from typing import Any
import math
import json

from dft_local.diagnostics.models import DiagnosticSection, HtmlBlock, MarkdownBlock, DiagnosticResult, Graph2D, TypstMathBlock, Card, Table
from dft_local.diagnostics.typst import TypstRenderError, render_typst_error, render_typst_math_to_svg
from dft_local.diagnostics.user_strings import RichText, TypstMath, rich
from dft_local.core.units import DisplayQuantity




def render_user_string(value: Any) -> str:
    """Render a user-facing string to safe HTML.

    Plain strings are escaped.  TypstMath snippets are compiled to inline SVG.
    RichText concatenates plain text and rich inline parts so only mathematical
    content is rendered by Typst.
    """

    if isinstance(value, RichText):
        return "".join(render_user_string(part) for part in value.parts)

    if isinstance(value, TypstMath):
        try:
            svg = render_typst_math_to_svg(value.source, display=value.display)
        except TypstRenderError as exc:
            return render_typst_error(value.source, exc)

        cls = "typst-math display" if value.display else "typst-math inline"
        source = escape(value.source, quote=True)
        name = escape(value.name, quote=True)
        return f"<span class='{cls}' data-typst-source='{source}' data-typst-name='{name}'>{svg}</span>"

    return render_user_string(value) if isinstance(value, (RichText, TypstMath)) else escape(str(value))


def render_display_value(value: Any) -> str:
    """Render a value that appears in a table cell or compact value slot."""

    if isinstance(value, DisplayQuantity):
        return (
            f"<span class='display-quantity' data-unit='{escape(value.unit.symbol)}'>"
            f"{escape(fmt(value.value))} "
            f"<span class='display-unit'>{escape(value.unit.symbol)}</span>"
            "</span>"
        )

    if isinstance(value, (RichText, TypstMath)):
        return render_user_string(value)

    return escape(fmt(value))


def fmt(value: Any) -> str:
    """Format values for compact diagnostic display."""

    if value is None:
        return "—"

    if isinstance(value, dict):
        if {"real", "imag", "abs"} <= set(value):
            return (
                f"real={fmt(value['real'])}; "
                f"imag={fmt(value['imag'])}; "
                f"abs={fmt(value['abs'])}"
            )
        return ", ".join(f"{key}={fmt(val)}" for key, val in value.items())

    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 1e-4 or abs(value) > 1e5:
            return f"{value:.6e}"
        return f"{value:.6g}"

    if isinstance(value, complex):
        return f"{value.real:.6e} + {value.imag:.6e} i"

    return str(value)


def _nice(value: float) -> str:
    if not math.isfinite(value):
        return ""
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.4g}"


def _render_graph_svg(graph: Graph2D) -> str:
    """Render a Graph2D as static inline SVG."""

    width = 1000.0
    height = 520.0
    left = 78.0
    right = 24.0
    top = 28.0
    bottom = 62.0
    inner_w = width - left - right
    inner_h = height - top - bottom

    all_points = [
        point
        for series in graph.series
        for point in series.points
        if math.isfinite(point.x) and math.isfinite(point.y)
    ]

    if not all_points:
        return "<p>No graph data.</p>"

    xmin = min(point.x for point in all_points)
    xmax = max(point.x for point in all_points)
    ymin = min(point.y for point in all_points)
    ymax = max(point.y for point in all_points)

    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0

    ypad = 0.06 * (ymax - ymin)
    ymin -= ypad
    ymax += ypad

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * inner_w

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * inner_h

    colours = (
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
        "#be123c",
        "#65a30d",
        "#7c3aed",
        "#0f766e",
        "#b45309",
    )

    parts: list[str] = [
        (
            "<svg class='graph-svg' "
            f"viewBox='0 0 {width:.0f} {height:.0f}' "
            "role='img' "
            f"aria-label='{escape(str(graph.title), quote=True)}'>"
        ),
        "<rect class='graph-bg' x='0' y='0' width='1000' height='520' />",
    ]

    for i in range(6):
        t = i / 5.0
        x = left + t * inner_w
        xv = xmin + t * (xmax - xmin)
        parts.append(f"<line class='grid' x1='{x:.3f}' y1='{top:.3f}' x2='{x:.3f}' y2='{top + inner_h:.3f}' />")
        parts.append(f"<text class='axis-label' x='{x:.3f}' y='{height - 24:.3f}' text-anchor='middle'>{escape(_nice(xv))}</text>")

    for i in range(6):
        t = i / 5.0
        y = top + t * inner_h
        yv = ymax - t * (ymax - ymin)
        parts.append(f"<line class='grid' x1='{left:.3f}' y1='{y:.3f}' x2='{left + inner_w:.3f}' y2='{y:.3f}' />")
        parts.append(f"<text class='axis-label' x='{left - 10:.3f}' y='{y + 4:.3f}' text-anchor='end'>{escape(_nice(yv))}</text>")

    parts.append(f"<line class='axis' x1='{left:.3f}' y1='{top + inner_h:.3f}' x2='{left + inner_w:.3f}' y2='{top + inner_h:.3f}' />")
    parts.append(f"<line class='axis' x1='{left:.3f}' y1='{top:.3f}' x2='{left:.3f}' y2='{top + inner_h:.3f}' />")
    parts.append(f"<text class='axis-title' x='{left + inner_w / 2:.3f}' y='{height - 6:.3f}' text-anchor='middle'>{render_user_string(graph.x_label)}</text>")
    parts.append(
        f"<text class='axis-title' x='18' y='{top + inner_h / 2:.3f}' text-anchor='middle' "
        f"transform='rotate(-90 18 {top + inner_h / 2:.3f})'>{render_user_string(graph.y_label)}</text>"
    )

    for series_index, series in enumerate(graph.series):
        colour = colours[series_index % len(colours)]
        points = [
            (sx(point.x), sy(point.y))
            for point in series.points
            if math.isfinite(point.x) and math.isfinite(point.y)
        ]

        if not points:
            continue

        if series.kind in {"line", "line_points"}:
            d = " ".join(
                f"{'M' if i == 0 else 'L'} {x:.3f} {y:.3f}"
                for i, (x, y) in enumerate(points)
            )
            parts.append(
                f"<path class='series-line' d='{d}' fill='none' "
                f"stroke='{colour}' stroke-width='1.8' vector-effect='non-scaling-stroke' />"
            )

        if series.kind in {"points", "line_points"}:
            for x, y in points:
                parts.append(f"<circle cx='{x:.3f}' cy='{y:.3f}' r='2.5' fill='{colour}' />")

        lx, ly = points[-1]
        parts.append(
            f"<text class='series-label' x='{lx + 5:.3f}' y='{ly + 4:.3f}' "
            f"fill='{colour}'>{render_user_string(series.name)}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render_markdown_text(markdown: Any) -> str:
    """Render lightweight prose content.

    Plain strings use simple paragraph/list rendering. RichText keeps prose as
    normal HTML text while rendering only TypstMath parts as inline SVG.
    """

    if isinstance(markdown, RichText):
        paragraphs: list[list[Any]] = [[]]

        for part in markdown.parts:
            if isinstance(part, str):
                pieces = part.split("\n\n")
                for index, piece in enumerate(pieces):
                    if index > 0:
                        paragraphs.append([])
                    if piece:
                        paragraphs[-1].append(piece)
            else:
                paragraphs[-1].append(part)

        html_parts: list[str] = []
        for paragraph in paragraphs:
            if not paragraph:
                continue

            # Do not replace newlines after render_user_string().  Typst SVG
            # output contains internal newlines; converting those to <br>
            # creates huge visual gaps in inline math.
            normalised_parts = tuple(
                part.replace("\n", " ") if isinstance(part, str) else part
                for part in paragraph
            )
            content = rich(*normalised_parts)
            html_parts.append("<p>" + render_user_string(content) + "</p>")
        return "".join(html_parts)

    if not isinstance(markdown, str):
        return "<p>" + render_user_string(markdown) + "</p>"

    lines = markdown.splitlines()
    parts: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            parts.append("<p>" + " ".join(paragraph) + "</p>")
            paragraph.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            parts.append("<ul><li>" + escape(stripped[2:]) + "</li></ul>")
            continue

        paragraph.append(escape(stripped))

    flush_paragraph()
    return "".join(parts)


def render_typst_math_block(block: TypstMathBlock) -> str:
    math = block.math
    if not math.display:
        math = TypstMath(math.source, display=True, name=math.name)

    return (
        f"<div id='{escape(block.id)}' class='typst-math-block'>"
        + render_user_string(math)
        + "</div>"
    )


def render_html_block(block: HtmlBlock) -> str:
    return (
        f"<section id='{escape(block.id)}' class='html-block'>"
        f"<h2>{render_user_string(block.title)}</h2>"
        + block.html
        + "</section>"
    )


def render_markdown_block(block: MarkdownBlock) -> str:
    return (
        f"<section id='{escape(block.id)}' class='markdown-block'>"
        f"<h2>{render_user_string(block.title)}</h2>"
        + render_markdown_text(block.markdown)
        + "</section>"
    )


def render_card(card) -> str:
    title = getattr(card, "title", getattr(card, "label", ""))
    value = getattr(card, "value", "")
    subtitle = getattr(card, "subtitle", getattr(card, "kind", ""))

    return (
        "<div class='card'>"
        f"<strong>{render_user_string(title)}</strong>"
        f"<div>{render_user_string(value) if isinstance(value, (RichText, TypstMath)) else escape(str(value))}</div>"
        f"<small>{render_user_string(subtitle)}</small>"
        "</div>"
    )


def dom_safe_id(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-")
    return text or "block"


def render_block_shell(block_id: object, kind: str, html: str) -> str:
    safe_id = dom_safe_id(block_id)
    return (
        f"<section id='dft-block-{escape(safe_id)}'"
        f" data-dft-block='{escape(str(block_id))}'"
        f" data-dft-block-kind='{escape(kind)}'>"
        f"{html}"
        "</section>"
    )


def render_json_model(model_id: object, payload: object) -> str:
    safe_id = dom_safe_id(model_id)
    payload_json = json.dumps(payload).replace("</", "<\\/")
    return (
        f"<script type='application/json'"
        f" id='dft-model-{escape(safe_id)}'"
        f" data-dft-model='{escape(str(model_id))}'>"
        f"{payload_json}"
        "</script>"
    )



def table_row_dom_id(table_id: object, row_index: int, row: object) -> str:
    cells = getattr(row, "cells", ())
    entity_id = getattr(row, "entity_id", None)

    if entity_id is not None:
        return f"{table_id}:{entity_id}"

    if cells:
        return f"{table_id}:row:{row_index}:{cells[0]}"

    return f"{table_id}:row:{row_index}"




def table_records_json(table: Table) -> str:
    records = [
        {
            str(header): fmt(cell)
            for header, cell in zip(table.headers, row.cells, strict=False)
        }
        for row in table.rows
    ]
    return json.dumps(records, ensure_ascii=False, indent=2)


def render_table_copy_button(table: Table) -> str:
    payload = escape(table_records_json(table), quote=True)
    title = escape(str(table.title), quote=True)
    return (
        "<button type='button' class='table-copy-json' "
        f"data-table-json='{payload}' "
        f"aria-label='Copy {title} table as JSON' "
        f"title='Copy {title} table as JSON'>"
        "⧉"
        "</button>"
    )

def render_table(table) -> str:
    has_step = "step" in table.headers
    step_index = table.headers.index("step") if has_step else -1
    x_index = table.headers.index("x") if "x" in table.headers else -1
    energy_index = table.headers.index("energy") if "energy" in table.headers else -1
    row_label_index = 0

    parts: list[str] = []
    parts.append(
        f"<section id='{escape(str(table.id))}' class='diagnostic-table-section'>"
        "<div class='diagnostic-table-header'>"
        f"<h2>{render_user_string(table.title)}</h2>"
        "</div>"
    )
    parts.append(f"<p>{render_user_string(table.description)}</p>")

    if has_step:
        parts.append(
            f"<div class='table-select-controls' data-table-id='{escape(table.id)}'>"
            "<button type='button' data-table-select='all'>Select all</button>"
            "<button type='button' data-table-select='none'>Clear all</button>"
            "</div>"
        )

    selectable_attr = " data-dft-selectable-table" if has_step else ""
    parts.append(
        f"<div class='table-breakout' tabindex='0'>"
        f"{render_table_copy_button(table)}"
        f"<table data-dft-table='{escape(str(table.id))}'{selectable_attr}><thead><tr>"
    )
    if has_step:
        parts.append("<th>select</th>")

    for header in table.headers:
        parts.append(f"<th>{render_user_string(header)}</th>")
    parts.append("</tr></thead><tbody>")

    for row_index, row in enumerate(table.rows):
        row_id = table_row_dom_id(table.id, row_index, row)
        step_value = row.cells[step_index] if has_step else None
        x_value = row.cells[x_index] if x_index >= 0 else step_value
        energy_value = row.cells[energy_index] if energy_index >= 0 else ""
        row_label_value = row.cells[row_label_index] if row.cells else step_value

        attrs = f" data-dft-row-id='{escape(row_id)}'"
        if has_step:
            attrs = (
                f" data-step='{escape(str(step_value))}'"
                f" data-path-x='{escape(str(x_value))}'"
                f" data-energy='{escape(str(energy_value))}'"
                f" data-label='{escape(str(row_label_value))}'"
                f" data-table-id='{escape(table.id)}'"
            )

        parts.append(f"<tr{attrs}>")

        if has_step:
            parts.append(
                "<td>"
                f"<input type='checkbox' class='table-step-select'"
                f" data-step='{escape(str(step_value))}'"
                f" data-path-x='{escape(str(x_value))}'"
                f" data-energy='{escape(str(energy_value))}'"
                f" data-label='{escape(str(row_label_value))}'"
                f" data-table-id='{escape(table.id)}'>"
                "</td>"
            )

        for cell in row.cells:
            parts.append(f"<td>{render_display_value(cell)}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table></div>")
    return render_block_shell(table.id, "stateful-html", "".join(parts))


def render_matrix(matrix) -> str:
    parts: list[str] = []

    parts.append(f"<section><h2>{render_user_string(matrix.title)}</h2>")
    parts.append(f"<p>{render_user_string(matrix.description)}</p>")
    parts.append("<table><thead><tr><th></th>")

    for label in matrix.col_labels:
        parts.append(f"<th>{render_user_string(label)}</th>")
    parts.append("</tr></thead><tbody>")

    cells = {(cell.i, cell.j): cell.value for cell in matrix.cells}
    for i, row_label in enumerate(matrix.row_labels):
        parts.append(f"<tr><th>{render_user_string(row_label)}</th>")
        for j, _col_label in enumerate(matrix.col_labels):
            parts.append(f"<td>{render_display_value(cells.get((i, j)))}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table>")
    return render_block_shell(matrix.id, "static-html", "".join(parts))


def render_webgl_view(view) -> str:
    data_id = f"dft-model-{dom_safe_id(view.id)}"

    model = render_json_model(view.id, view.payload)

    if view.renderer == "region_surface":
        component = "dft-band-surface-viewer"
    elif view.renderer == "graphene_viewer":
        component = "dft-graphene-viewer"
    else:
        raise ValueError(f"Unknown WebGL renderer: {view.renderer}")

    html = (
        f"<div class='webgl-panel'><h2>{render_user_string(view.title)}</h2>"
        f"<p>{render_user_string(view.description)}</p>"
        f"<{component} data-source='{escape(data_id)}' data-dft-model='{escape(data_id)}'></{component}>"
        "</div>"
    )

    return render_block_shell(view.id, "json-rendered", model + html)


def render_graph(graph: Graph2D) -> str:
    data_id = f"dft-model-{dom_safe_id(graph.id)}"
    model = render_json_model(graph.id, graph.payload())
    component = "dft-kspace-plot" if "kspace" in graph.id else "dft-line-graph"

    html = (
        f"<div class='graph-panel'><h2>{render_user_string(graph.title)}</h2>"
        f"<p>{render_user_string(graph.description)}</p>"
        f"<{component} data-source='{escape(data_id)}' data-dft-model='{escape(data_id)}'>"
        f"{_render_graph_svg(graph)}"
        f"</{component}>"
        "</div>"
    )

    return render_block_shell(graph.id, "json-rendered", model + html)


def render_document_block(block: Any) -> str:
    """Render one ordered diagnostic document block."""

    if isinstance(block, TypstMathBlock):
        return render_typst_math_block(block)

    if isinstance(block, HtmlBlock):
        return render_html_block(block)

    if isinstance(block, MarkdownBlock):
        return "<div class='markdown-math-group'>" + render_markdown_block(block) + "</div>"

    if isinstance(block, Card):
        return render_card(block)

    if isinstance(block, Table):
        return render_table(block)

    if isinstance(block, Graph2D):
        return render_graph(block)

    if isinstance(block, DiagnosticSection):
        return render_diagnostic_section(block)

    # Late imports avoided; isinstance on optional classes by name keeps this
    # renderer tolerant while the model is still migrating.
    if block.__class__.__name__ == "Matrix":
        return render_matrix(block)

    if block.__class__.__name__ == "WebGLView":
        return render_webgl_view(block)

    raise TypeError(f"Unsupported diagnostic document block: {type(block)!r}")



def _toc_title(value: object) -> str:
    if value is None:
        return ""
    return render_user_string(value)


def _iter_toc_entries(blocks: tuple[object, ...], depth: int = 0) -> list[tuple[int, str, str]]:
    entries: list[tuple[int, str, str]] = []

    for block in blocks:
        nested_depth = depth

        if isinstance(block, DiagnosticSection):
            block_id = getattr(block, "id", "")
            title = getattr(block, "title", "")
            if block_id and title:
                entries.append((depth, str(block_id), _toc_title(title)))
            nested_depth = depth + 1

        nested_body = tuple(getattr(block, "body", ()) or ())
        nested_sections = tuple(getattr(block, "sections", ()) or ())
        nested = nested_body or nested_sections
        if nested:
            entries.extend(_iter_toc_entries(nested, nested_depth))

    return entries


def render_result_toc(result: DiagnosticResult) -> str:
    blocks: list[object] = []

    if getattr(result, "body", ()):
        blocks.extend(result.body)
    else:
        blocks.extend(result.markdowns)
        blocks.extend(result.cards)
        blocks.extend(result.sections)
        blocks.extend(result.matrices)
        blocks.extend(result.webgl)
        blocks.extend(result.graphs)
        blocks.extend(result.tables)

    entries = _iter_toc_entries(tuple(blocks))
    if not entries:
        return ""

    links = []
    for depth, block_id, title in entries:
        depth_class = f"diagnostic-toc-depth-{min(depth, 4)}"
        links.append(
            "<a "
            f"class='diagnostic-toc-link {depth_class}' "
            f"href='#{escape(block_id)}'>"
            f"<span>{title}</span>"
            "</a>"
        )

    return (
        "<details class='diagnostic-toc'>"
        "<summary aria-label='Table of contents'><span class='diagnostic-toc-icon'>☰</span><span class='diagnostic-toc-label'>Contents</span></summary>"
        "<nav class='diagnostic-toc-panel' aria-label='Diagnostic table of contents'>"
        + "".join(links)
        + "</nav>"
        "</details>"
    )


def render_diagnostic_section(section: DiagnosticSection) -> str:
    open_attr = "" if section.collapsed else " open"
    body: list[str] = []

    if section.description:
        body.append(f"<p>{render_user_string(section.description)}</p>")

    if getattr(section, "body", ()):
        for block in section.body:
            body.append(render_document_block(block))
    else:
        blocks = tuple(section.markdowns)
        i = 0
        while i < len(blocks):
            block = blocks[i]

            if isinstance(block, MarkdownBlock):
                attached_math: list[TypstMathBlock] = []
                j = i + 1
                while j < len(blocks) and isinstance(blocks[j], TypstMathBlock):
                    attached_math.append(blocks[j])
                    j += 1

                group = ["<div class='markdown-math-group'>", render_markdown_block(block)]
                group.extend(render_typst_math_block(math_block) for math_block in attached_math)
                group.append("</div>")
                body.append("".join(group))
                i = j
                continue

            if isinstance(block, TypstMathBlock):
                body.append(render_typst_math_block(block))
                i += 1
                continue

            body.append(render_markdown_block(block))
            i += 1

        for block in getattr(section, "math_blocks", ()):
            body.append(render_typst_math_block(block))

        for card in section.cards:
            body.append(render_card(card))

        for table in section.tables:
            body.append(render_table(table))

        for subsection in section.sections:
            body.append(render_diagnostic_section(subsection))

    return (
        f"<details id='{escape(section.id)}' class='diagnostic-section'{open_attr}>"
        f"<summary>{render_user_string(section.title)}</summary>"
        + "".join(body)
        + "</details>"
    )


def render_result(result: DiagnosticResult) -> str:
    """Render a diagnostic result body as simple HTML."""

    parts: list[str] = []

    toc = render_result_toc(result)
    if toc:
        parts.append(toc)

    parts.append(f"<h1>{render_user_string(result.title)}</h1>")
    parts.append(f"<p>{render_user_string(result.summary)}</p>")

    if getattr(result, "body", ()):
        for block in result.body:
            parts.append(render_document_block(block))
        return "\n".join(parts)

    rendered_markdowns = ''.join(render_markdown_block(block) for block in result.markdowns)

    if result.cards:
        parts.append("<section class='cards'>")
        for card in result.cards:
            parts.append(render_card(card))
        parts.append("</section>")

    for section in result.sections:
        parts.append(render_diagnostic_section(section))

    for matrix in result.matrices:
        parts.append(f"<section><h2>{render_user_string(matrix.title)}</h2>")
        parts.append(f"<p>{render_user_string(matrix.description)}</p>")
        parts.append("<table><thead><tr><th></th>")
        for label in matrix.col_labels:
            parts.append(f"<th>{render_user_string(label)}</th>")
        parts.append("</tr></thead><tbody>")

        cells = {(cell.i, cell.j): cell.value for cell in matrix.cells}
        for i, row_label in enumerate(matrix.row_labels):
            parts.append(f"<tr><th>{render_user_string(row_label)}</th>")
            for j, _col_label in enumerate(matrix.col_labels):
                parts.append(f"<td>{render_display_value(cells.get((i, j)))}</td>")
            parts.append("</tr>")

        parts.append("</tbody></table></section>")

    for view in result.webgl:
        parts.append(render_webgl_view(view))

    for graph in result.graphs:
        parts.append(render_graph(graph))

    for table in result.tables:
        parts.append(render_table(table))

    if result.notes:
        parts.append("<section><h2>Notes</h2>")
        for note in result.notes:
            parts.append(f"<p>{render_user_string(note)}</p>")
        parts.append("</section>")

    return rendered_markdowns + "\n".join(parts)



ACADEMIC_STYLE = """
<style>
:root {
  --paper: #fbfaf7;
  --ink: #1f2933;
  --muted: #606f7b;
  --rule: #d7d0c4;
  --soft-rule: #ebe4d8;
  --accent: #284b63;
  --accent-soft: #eef4f7;
  --mono-bg: #f4f1eb;
  --shadow: 0 1px 2px rgba(31, 41, 51, 0.08);
}

html {
  background: #ece7dd;
  color: var(--ink);
  font-size: 16px;
}

body {
  max-width: 980px;
  margin: 0 auto;
  padding: 3rem 2.25rem 5rem;
  background: var(--paper);
  font-family: "Latin Modern Roman", "Computer Modern Serif", "CMU Serif", Georgia, "Times New Roman", serif;
  line-height: 1.62;
  box-shadow: 0 0 0 1px rgba(80, 70, 55, 0.10), 0 10px 30px rgba(80, 70, 55, 0.12);
  min-height: 100vh;
}

.diagnostic-paper h1, .diagnostic-paper h2, .diagnostic-paper h3, .diagnostic-paper h4 {
  color: #111827;
  font-weight: 600;
  line-height: 1.2;
  letter-spacing: -0.015em;
}

.diagnostic-paper h1 {
  margin: 0 0 1.2rem;
  font-size: 2.25rem;
  text-align: center;
  border-bottom: 1px solid var(--rule);
  padding-bottom: 0.8rem;
}

.diagnostic-paper h2 {
  margin-top: 2.4rem;
  font-size: 1.45rem;
  border-bottom: 1px solid var(--soft-rule);
  padding-bottom: 0.25rem;
}

.diagnostic-paper h3 {
  margin-top: 1.8rem;
  font-size: 1.15rem;
}

.diagnostic-paper p {
  margin: 0.75rem 0 1rem;
}

.diagnostic-paper a {
  color: var(--accent);
  text-decoration-thickness: 0.08em;
  text-underline-offset: 0.14em;
}

.diagnostic-paper nav {
  margin: -1rem 0 2rem;
  padding: 0.7rem 0;
  border-bottom: 1px solid var(--rule);
  color: var(--muted);
  font-size: 0.92rem;
}

.diagnostic-paper details.diagnostic-toc {
  position: fixed;
  top: 1rem;
  left: 1rem;
  z-index: 20;
  margin: 0;
  font-family: "Latin Modern Roman", "Computer Modern Serif", "CMU Serif", Georgia, "Times New Roman", serif;
}

.diagnostic-paper details.diagnostic-toc > summary {
  list-style: none;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.45rem 0.65rem;
  border: 1px solid var(--rule);
  border-radius: 4px;
  background: #fffdf8;
  color: var(--accent);
  box-shadow: var(--shadow);
  font-weight: 600;
}

.diagnostic-paper .diagnostic-toc-label {
  display: inline-block;
  max-width: 0;
  overflow: hidden;
  opacity: 0;
  white-space: nowrap;
  transition: max-width 140ms ease, opacity 140ms ease;
}

.diagnostic-paper details.diagnostic-toc:hover .diagnostic-toc-label,
.diagnostic-paper details.diagnostic-toc[open] .diagnostic-toc-label {
  max-width: 8rem;
  opacity: 1;
}

.diagnostic-paper details.diagnostic-toc > summary::-webkit-details-marker {
  display: none;
}

.diagnostic-paper .diagnostic-toc-icon {
  font-size: 1.1rem;
  line-height: 1;
}

.diagnostic-paper .diagnostic-toc-panel {
  width: min(24rem, calc(100vw - 2rem));
  max-height: calc(100vh - 5rem);
  overflow: auto;
  margin: 0.45rem 0 0;
  padding: 0.55rem;
  border: 1px solid var(--rule);
  border-radius: 4px;
  background: #fffdf8;
  box-shadow: 0 10px 30px rgba(80, 70, 55, 0.18);
}

.diagnostic-paper .diagnostic-toc-link {
  display: block;
  padding: 0.26rem 0.35rem;
  border-radius: 3px;
  color: var(--ink);
  text-decoration: none;
  line-height: 1.25;
}

.diagnostic-paper .diagnostic-toc-link:hover {
  background: var(--accent-soft);
}

.diagnostic-paper .diagnostic-toc-depth-1 {
  padding-left: 0.9rem;
}

.diagnostic-paper .diagnostic-toc-depth-2 {
  padding-left: 1.8rem;
}

.diagnostic-paper .diagnostic-toc-depth-3,
.diagnostic-paper .diagnostic-toc-depth-4 {
  padding-left: 2.7rem;
}


.diagnostic-paper code, .diagnostic-paper pre, .diagnostic-paper kbd {
  font-family: "Latin Modern Mono", "Computer Modern Typewriter", "CMU Typewriter Text", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

.diagnostic-paper code {
  background: var(--mono-bg);
  border: 1px solid var(--soft-rule);
  border-radius: 3px;
  padding: 0.08rem 0.25rem;
  font-size: 0.92em;
}

.diagnostic-paper pre {
  overflow-x: auto;
  background: var(--mono-bg);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 0.85rem 1rem;
}

pre .diagnostic-paper code {
  border: 0;
  background: transparent;
  padding: 0;
}

.diagnostic-paper table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0 1.6rem;
  font-size: 0.92rem;
  line-height: 1.35;
}

.diagnostic-paper thead {
  border-top: 1.5px solid #2f3640;
  border-bottom: 1px solid #2f3640;
}

.diagnostic-paper tbody {
  border-bottom: 1.5px solid #2f3640;
}

.diagnostic-paper th, .diagnostic-paper td {
  padding: 0.42rem 0.55rem;
  text-align: right;
  vertical-align: top;
  border: 0;
}

.diagnostic-paper th:first-child, .diagnostic-paper td:first-child {
  text-align: left;
}

.diagnostic-paper tbody tr:nth-child(even) {
  background: rgba(40, 75, 99, 0.035);
}

.diagnostic-paper small {
  color: var(--muted);
}

.diagnostic-paper ul, .diagnostic-paper ol {
  padding-left: 1.5rem;
}

.diagnostic-paper .card {
  display: inline-block;
  vertical-align: top;
  min-width: 11rem;
  max-width: 18rem;
  margin: 0.35rem 0.45rem 0.55rem 0;
  padding: 0.65rem 0.8rem;
  border: 1px solid var(--rule);
  border-radius: 3px;
  background: #fffdf8;
  box-shadow: var(--shadow);
}

.diagnostic-paper .card strong {
  display: block;
  font-variant: small-caps;
  letter-spacing: 0.035em;
}


/* Document prose block, optionally followed by one or more display equations. */
.diagnostic-paper .markdown-math-group {
  margin: 1rem 0;
  padding: 0.85rem 1rem;
  border-left: 3px solid var(--accent);
  background: color-mix(in srgb, var(--accent) 5%, transparent);
  overflow-x: hidden;
}

.diagnostic-paper .markdown-math-group .markdown-block {
  margin: 0;
  padding: 0;
  border-left: 0;
  background: transparent;
}

.diagnostic-paper .markdown-math-group .typst-math-block {
  margin: 0.7rem 0 0;
  background: transparent;
}

.diagnostic-paper .typst-math-block {
  display: block;
  box-sizing: border-box;
  width: 100%;
  max-width: 100%;
  margin: 1rem 0 1.2rem;
  padding: 0.15rem 0;
  overflow-x: auto;
  overflow-y: hidden;
  text-align: center;
  contain: inline-size;
}

.diagnostic-paper .typst-math-block .typst-math {
  display: inline-flex;
  max-width: 100%;
  justify-content: center;
  vertical-align: middle;
}

.diagnostic-paper .typst-math-block svg {
  display: block;
  max-width: 100%;
  height: auto;
  flex: 0 1 auto;
}

.diagnostic-paper .typst-math {
  display: inline-flex;
  align-items: center;
  max-width: 100%;
  vertical-align: -0.12em;
  line-height: 1;
}

.diagnostic-paper .typst-math.inline {
  height: 1.15em;
}

.diagnostic-paper .typst-math.inline svg {
  display: inline-block;
  height: 1.15em !important;
  width: auto !important;
  max-width: 100%;
  overflow: visible;
}

.diagnostic-paper .typst-math.display svg {
  display: block;
  max-width: 100%;
  height: auto;
}

.diagnostic-paper p:has(> .typst-math.display:only-child) {
  text-align: center;
}

.diagnostic-paper p > .typst-math.display:only-child {
  display: inline-flex;
  justify-content: center;
  margin: 0.85rem auto;
}

.diagnostic-paper .markdown-block > .typst-math.display:only-child,
.diagnostic-paper .diagnostic-section > .typst-math.display:only-child {
  display: flex;
  justify-content: center;
  margin: 0.85rem auto;
}

.diagnostic-paper .typst-error {
  color: #8a1f11;
  border-bottom: 1px dotted #8a1f11;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

.diagnostic-paper .card div {
  font-size: 1.05rem;
  margin-top: 0.15rem;
}

.card .diagnostic-paper small {
  display: block;
  margin-top: 0.2rem;
}

.diagnostic-paper .markdown-block {
  margin: 1.2rem 0 1.4rem;
  padding: 0.95rem 1.15rem;
  border-left: 3px solid var(--accent);
  background: var(--accent-soft);
}

.markdown-block .diagnostic-paper h2 {
  margin-top: 0;
  border-bottom: 0;
  font-size: 1.18rem;
}

.diagnostic-paper details.diagnostic-section {
  margin: 1.35rem 0;
  border: 1px solid var(--rule);
  border-radius: 4px;
  background: #fffdf8;
  box-shadow: var(--shadow);
}

.diagnostic-paper details.diagnostic-section > summary {
  cursor: pointer;
  padding: 0.75rem 0.95rem;
  font-weight: 600;
  color: #111827;
  background: linear-gradient(#fffdf8, #f7f1e8);
  border-bottom: 1px solid transparent;
}

.diagnostic-paper details.diagnostic-section[open] > summary {
  border-bottom-color: var(--rule);
}

.diagnostic-paper details.diagnostic-section > *:not(summary) {
  margin-left: 1rem;
  margin-right: 1rem;
}

details.diagnostic-section .diagnostic-paper details.diagnostic-section {
  margin-left: 0.5rem;
  margin-right: 0.5rem;
  background: #fbfaf7;
}

@media (max-width: 760px) {
  .diagnostic-paper details.diagnostic-toc {
    top: 0.5rem;
    left: 0.5rem;
  }

  .diagnostic-paper details.diagnostic-toc > summary span:last-child {
    display: none;
  }

  .diagnostic-paper .diagnostic-toc-panel {
    width: calc(100vw - 1rem);
    max-height: calc(100vh - 4rem);
  }

  body {
    padding: 1.4rem 1rem 3rem;
    box-shadow: none;
  }

  .diagnostic-paper h1 {
    font-size: 1.8rem;
  }

  .diagnostic-paper table {
    display: block;
    overflow-x: auto;
    white-space: nowrap;
  }

  .diagnostic-paper .card {
    display: block;
    max-width: none;
  }
}


/* Interactive graph components own their own geometry and colour rules.
   Keep the paper theme from leaking into their layout calculations. */
.diagnostic-paper dft-line-graph,
.diagnostic-paper dft-kspace-graph,
.diagnostic-paper dft-band-graph {
  display: block;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: normal;
  color: #111827;
  background: transparent;
  box-sizing: content-box;
  contain: layout style;
}


/* Graph SVG layout only: do not style internal geometry. */
.diagnostic-paper svg.graph-svg-component {
  max-width: 100%;
  height: auto;
}

.diagnostic-paper svg.kspace-svg {
  width: min(720px, 100%);
  height: auto;
  aspect-ratio: 1 / 1;
}


/* Reusable horizontal overflow treatment for wide diagnostic tables. */
.diagnostic-paper .diagnostic-table-header {
  display: block;
  border-bottom: 1px solid var(--rule);
}

.diagnostic-paper .diagnostic-table-header h2 {
  margin: 0;
  border-bottom: 0;
}

.diagnostic-paper .table-copy-json {
  appearance: none;
  position: absolute;
  top: 0.45rem;
  right: 0.45rem;
  z-index: 2;
  width: 2rem;
  height: 2rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--rule);
  border-radius: 999px;
  background: rgba(255, 253, 248, 0.92);
  color: #374151;
  cursor: pointer;
  font: inherit;
  font-size: 1rem;
  line-height: 1;
  padding: 0;
  box-shadow: var(--shadow);
  backdrop-filter: blur(2px);
}

.diagnostic-paper .table-copy-json:hover {
  background: var(--accent-soft);
  border-color: var(--accent);
}

.diagnostic-paper .table-copy-json.copied {
  color: #166534;
  border-color: #86efac;
  background: #f0fdf4;
}

.diagnostic-paper .diagnostic-table-section {
  min-width: 0;
}

.diagnostic-paper .table-breakout {
  position: relative;
  max-width: min(96vw, 1200px);
  width: max-content;
  min-width: 100%;
  margin-left: 50%;
  transform: translateX(-50%);
  overflow-x: auto;
  overflow-y: hidden;
  padding: 0.2rem 0.75rem 0.45rem;
  scrollbar-width: thin;
  background: var(--paper);
  border-radius: 3px;
}

.diagnostic-paper .table-breakout:focus {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

.diagnostic-paper .table-breakout table {
  width: max-content;
  min-width: 100%;
  margin: 1rem 0 1.2rem;
  background: transparent;
}

.diagnostic-paper details.diagnostic-section .table-breakout {
  max-width: min(94vw, 1200px);
}

@media (max-width: 760px) {
  .diagnostic-paper .table-breakout {
    max-width: 100vw;
    margin-left: 50%;
    transform: translateX(-50%);
    padding-left: 1rem;
    padding-right: 1rem;
  }
}

</style>
"""


DATASTAR_SCRIPT = (
    "  <script type='module' "
    "src='/static/datastar.js'>"
    "</script>\n"
)


THREE_IMPORT_MAP = """  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js"
    }
  }
  </script>
"""


TABLE_COPY_SCRIPT = """  <script>
document.addEventListener("click", async (event) => {
  const button = event.target.closest(".table-copy-json[data-table-json]");
  if (!button) {
    return;
  }

  const originalText = button.textContent;
  const payload = button.dataset.tableJson || "[]";

  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(payload);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = payload;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }

    button.textContent = "Copied";
    button.classList.add("copied");
    window.setTimeout(() => {
      button.textContent = originalText;
      button.classList.remove("copied");
    }, 1200);
  } catch (error) {
    button.textContent = "Copy failed";
    window.setTimeout(() => {
      button.textContent = originalText;
    }, 1600);
  }
});
  </script>
"""


def render_page(title: str, body: str) -> str:
    return (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        "  <meta charset='utf-8'>\n"
        f"  <title>{escape(title)}</title>\n"
        f"{ACADEMIC_STYLE}\n"
        f"{THREE_IMPORT_MAP}\n"
        f"{DATASTAR_SCRIPT}\n"
        "  <script type='module' src='/static/dft-local-components.js'></script>\n"
        f"{TABLE_COPY_SCRIPT}\n"
        "</head>\n"
        "<body>\n"
        "<main class='diagnostic-paper'>\n"
        f"{body}\n"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )

